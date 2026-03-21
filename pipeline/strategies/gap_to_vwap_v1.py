"""gap_to_vwap_v1 — Gap-to-VWAP Mean Reversion on NIFTY (5-second bars)

After NIFTY gaps open and continues to deviate 20+ bps from developing session VWAP,
VWAP-benchmarked institutional algorithms (index fund rebalancers, delta-hedgers, program
traders) begin accumulating against the gap direction to maintain VWAP-benchmark fills.
We enter on the 25-second EMA micro-cross that signals institutional absorption has begun,
targeting the first 15-25 spot point convergence thrust (7 option pts) within 60-90 seconds.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_vwap(close: np.ndarray, volume: np.ndarray, day_id: np.ndarray) -> np.ndarray:
    """Session VWAP — cumulative from first bar of each day_id."""
    n = len(close)
    vwap = np.zeros(n)
    cum_pv = 0.0
    cum_v = 0.0
    current_day = -1
    for i in range(n):
        if day_id[i] != current_day:
            cum_pv = 0.0
            cum_v = 0.0
            current_day = day_id[i]
        vol_i = volume[i] if volume[i] > 0 else 1.0
        cum_pv += close[i] * vol_i
        cum_v += vol_i
        vwap[i] = cum_pv / cum_v
    return vwap


def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder RSI; returns 50.0 before sufficient warmup."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n <= period:
        return rsi

    gains = np.zeros(n)
    losses = np.zeros(n)
    for i in range(1, n):
        diff = close[i] - close[i - 1]
        if diff > 0:
            gains[i] = diff
        else:
            losses[i] = -diff

    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[period] = np.mean(gains[1 : period + 1])
    avg_loss[period] = np.mean(losses[1 : period + 1])
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period

    for i in range(period, n):
        if avg_loss[i] == 0.0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def _compute_ema(close: np.ndarray, period: int) -> np.ndarray:
    """Standard EMA with k = 2/(period+1)."""
    n = len(close)
    ema = np.zeros(n)
    k = 2.0 / (period + 1)
    ema[0] = close[0]
    for i in range(1, n):
        ema[i] = close[i] * k + ema[i - 1] * (1.0 - k)
    return ema


def _compute_gap_pct(
    close: np.ndarray, open_: np.ndarray, day_id: np.ndarray
) -> np.ndarray:
    """Gap pct = (day_open - prev_day_close) / prev_day_close, constant per day.

    Day 1 returns 0.0 (no prior day).
    """
    n = len(close)
    gap = np.zeros(n)
    prev_day_close = np.nan
    day_open = np.nan
    current_day: int = -999999

    for i in range(n):
        d = int(day_id[i])
        if d != current_day:
            if current_day != -999999:
                prev_day_close = close[i - 1]
            day_open = open_[i]
            current_day = d

        if not np.isnan(prev_day_close) and prev_day_close > 0:
            gap[i] = (day_open - prev_day_close) / prev_day_close
        else:
            gap[i] = 0.0
    return gap


class Strategy(BaseStrategy):
    name = "gap_to_vwap_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — skip first 15 min of unreliable VWAP
    session_end_minutes = 855     # 14:15 IST — gap institutional urgency dissipates after this
    max_trades_per_day = 6
    max_lookback = 240            # 20 min warmup for VWAP stabilization

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vwap_dev_threshold_bps", 20.0, 12.0, 40.0),
            TunableParam("gap_threshold_pct", 0.002, 0.001, 0.005),
            TunableParam("rsi_oversold", 40.0, 30.0, 50.0),
            TunableParam("rsi_overbought", 60.0, 50.0, 70.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict,
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract spot arrays (forward-fill NaN in Polars before numpy) ──
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        open_ = spot_df["open"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        day_id = spot_df["day_id"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        # ── Parameters ──
        vwap_dev_threshold = params.get("vwap_dev_threshold_bps", 20.0)
        gap_threshold = params.get("gap_threshold_pct", 0.002)
        rsi_oversold = params.get("rsi_oversold", 40.0)
        rsi_overbought = params.get("rsi_overbought", 60.0)

        # ── Indicators ──
        vwap = _compute_vwap(close, volume, day_id)
        # VWAP deviation in basis points
        vwap_dev = np.where(vwap > 0, (close - vwap) / vwap * 10_000.0, 0.0)

        # Day-open gap vs prior close (constant per day)
        gap_pct = _compute_gap_pct(close, open_, day_id)

        # RSI(12) = 1-minute RSI at 5s resolution — entry trigger for exhaustion
        rsi12 = _compute_rsi(close, 12)

        # EMA(5) = 25-second micro EMA — micro-cross confirmation
        ema5 = _compute_ema(close, 5)

        # EMA micro-cross detection
        ema5_cross_up = np.zeros(n, dtype=bool)
        ema5_cross_down = np.zeros(n, dtype=bool)
        for i in range(1, n):
            ema5_cross_up[i] = (close[i] > ema5[i]) and (close[i - 1] <= ema5[i - 1])
            ema5_cross_down[i] = (close[i] < ema5[i]) and (close[i - 1] >= ema5[i - 1])

        # ── VIX filter ──
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(
                    ["datetime", pl.col("close").alias("vix_close")]
                ).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Session and regime filters ──
        in_session = (time_min >= self.session_start_minutes) & (
            time_min < self.session_end_minutes
        )
        vix_ok = vix_close < 20.0  # High VIX → slow reversion, degrades 90s time stop

        # ── Entry signals ──
        # Gap-down + price below VWAP + RSI oversold + EMA micro-cross up → buy CE
        buy_ce = (
            in_session
            & vix_ok
            & (gap_pct < -gap_threshold)
            & (vwap_dev < -vwap_dev_threshold)
            & (rsi12 < rsi_oversold)
            & ema5_cross_up
        )

        # Gap-up + price above VWAP + RSI overbought + EMA micro-cross down → buy PE
        buy_pe = (
            in_session
            & vix_ok
            & (gap_pct > gap_threshold)
            & (vwap_dev > vwap_dev_threshold)
            & (rsi12 > rsi_overbought)
            & ema5_cross_down
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            # Stop: 4 pts = ~8 spot pts further from VWAP; if thesis not reversing, exit
            stop_points=np.full(n, 4),
            # Target: 7 pts = ~14 spot pts convergence; ~70% of median first-leg reversion
            target_points=np.full(n, 8),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
