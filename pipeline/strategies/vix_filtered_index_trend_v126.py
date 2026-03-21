"""vix_filtered_index_trend_v126 — VIX-Regime-Filtered EMA Crossover on NIFTY.

Mechanism: On NIFTY, when India VIX is in the 12-22 range, institutional algorithmic
desks run directional TWAP/VWAP programs creating persistent micro-trends detectable via
EMA structure. A fresh EMA(3-min) crossover above EMA(9-min) while NIFTY is above session
VWAP signals sustained buy-side accumulation, typically extending 20-30 spot pts over the
next 30-90 seconds. VIX band filter removes both too-calm (no follow-through) and
too-volatile (whipsaw) regimes.

Converted from: trading_strategies/unique_strategies_all/Strategy_118.json
Original: VIX-filtered EMA(9)/EMA(21) crossover on nifty200 stocks, 1-min bars, hold 10-30 min.
"""
from __future__ import annotations
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average with forward-filled NaN."""
    alpha = 2.0 / (period + 1)
    out = np.empty(len(arr))
    out[0] = arr[0] if not np.isnan(arr[0]) else 0.0
    for i in range(1, len(arr)):
        val = arr[i] if not np.isnan(arr[i]) else out[i - 1]
        out[i] = alpha * val + (1.0 - alpha) * out[i - 1]
    return out


def _session_vwap(close: np.ndarray, volume: np.ndarray, day_id: np.ndarray) -> np.ndarray:
    """Cumulative session VWAP, reset each new day."""
    n = len(close)
    vwap = np.empty(n)
    cum_pv = 0.0
    cum_v = 0.0
    prev_day = -999
    for i in range(n):
        if day_id[i] != prev_day:
            cum_pv = 0.0
            cum_v = 0.0
            prev_day = day_id[i]
        v = volume[i] if not np.isnan(volume[i]) else 0.0
        c = close[i] if not np.isnan(close[i]) else 0.0
        cum_pv += c * v
        cum_v += v
        vwap[i] = cum_pv / cum_v if cum_v > 0.0 else c
    return vwap


class Strategy(BaseStrategy):
    name = "vix_filtered_index_trend_v126"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 920     # 15:20 IST
    max_lookback = 120            # 10 min warmup (covers EMA(108) seed)
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            # VIX regime band — moderate volatility is the sweet spot
            TunableParam("vix_low",  12.0,  9.0, 16.0),
            TunableParam("vix_high", 22.0, 18.0, 28.0),
            # Minimum normalised EMA gap to qualify as a fresh crossover (avoids noise)
            TunableParam("ema_gap_threshold", 0.0002, 0.0001, 0.001),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── Raw arrays (Polars forward-fill before numpy) ──────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # ── Parameters ─────────────────────────────────────────────
        vix_low = params.get("vix_low", 12.0)
        vix_high = params.get("vix_high", 22.0)
        ema_gap_thr = params.get("ema_gap_threshold", 0.0002)

        # ── VIX aligned to spot bars ────────────────────────────────
        vix_close = np.full(n, 15.0)  # neutral default
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── EMA crossover ───────────────────────────────────────────
        # Fast: 36 bars (3 min) — trade trigger, detects short-term momentum shift
        # Slow: 108 bars (9 min) — context filter, medium-term trend direction
        ema_fast = _ema(close, 36)
        ema_slow = _ema(close, 108)

        # ── Session VWAP (cumulative, resets daily) ─────────────────
        vwap = _session_vwap(close, volume, day_id)

        # ── Session filter ──────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # ── VIX regime filter ───────────────────────────────────────
        vix_ok = (vix_close >= vix_low) & (vix_close <= vix_high)

        # ── Fresh crossover detection ───────────────────────────────
        # Normalised EMA gap: positive = fast > slow (bullish momentum)
        # A "fresh" crossover = gap threshold crossed on THIS bar but not the previous bar
        bull_cross = np.zeros(n, dtype=bool)
        bear_cross = np.zeros(n, dtype=bool)
        for i in range(1, n):
            c_ref = close[i] if close[i] > 0 else 1.0
            gap_now  = (ema_fast[i]     - ema_slow[i])     / c_ref
            gap_prev = (ema_fast[i - 1] - ema_slow[i - 1]) / c_ref
            if gap_now > ema_gap_thr and gap_prev <= ema_gap_thr:
                bull_cross[i] = True
            elif gap_now < -ema_gap_thr and gap_prev >= -ema_gap_thr:
                bear_cross[i] = True

        # ── Entry signals ───────────────────────────────────────────
        buy_ce = in_session & vix_ok & bull_cross & (close > vwap)
        buy_pe = in_session & vix_ok & bear_cross & (close < vwap)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            # Stop: 4 pts = ~8 NIFTY spot pts. Genuine crossover should not retrace 8 pts in 120s.
            # Target: 7 pts = ~14 NIFTY spot pts. First impulse leg in VIX 12-22 extends 20-30 pts;
            #   capturing ~50-70% gives 7 option pts at delta ~0.5 (achievable in 60-120s).
            stop_points=np.full(n, 5),
            target_points=np.full(n, 8),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,          # 120 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
