"""RSI Extreme Reversion v1 — NIFTY 5-second index options strategy.

Mechanism: When NIFTY's 2-bar RSI drops below 5 (or rises above 95), two consecutive
5-second bars have closed aggressively in one direction — a micro-exhaustion signal from
retail stop-loss cascades or algo momentum overshoot near VWAP. Delta-neutral market makers
and VWAP-benchmarked institutional algos absorb this imbalance, creating a snap-back that
typically completes within 30-90 seconds.

Entry: RSI(2) extreme + RSI(36) neutral trend + near VWAP + VIX < threshold
Exit: 3 pt stop, 5 pt target, 18-bar (90s) time stop
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder's RSI. Returns 50.0 for bars before warmup completes."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n <= period:
        return rsi

    delta = np.zeros(n)
    delta[1:] = close[1:] - close[:-1]

    gain = np.where(delta > 0.0, delta, 0.0)
    loss = np.where(delta < 0.0, -delta, 0.0)

    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)

    # Seed with simple average over first `period` changes
    avg_gain[period] = np.mean(gain[1 : period + 1])
    avg_loss[period] = np.mean(loss[1 : period + 1])

    # Wilder's smoothing
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period

    for i in range(period, n):
        if avg_loss[i] == 0.0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


def _compute_session_vwap(close: np.ndarray, volume: np.ndarray, day_id: np.ndarray) -> np.ndarray:
    """Session VWAP reset each trading day. Falls back to close when volume=0."""
    n = len(close)
    vwap = close.copy()
    cum_pv = 0.0
    cum_vol = 0.0
    current_day = day_id[0] if n > 0 else -1

    for i in range(n):
        if day_id[i] != current_day:
            cum_pv = 0.0
            cum_vol = 0.0
            current_day = day_id[i]
        cum_pv += close[i] * volume[i]
        cum_vol += volume[i]
        vwap[i] = cum_pv / cum_vol if cum_vol > 0.0 else close[i]

    return vwap


class Strategy(BaseStrategy):
    name = "rsi_extreme_reversion_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — skip opening auction noise
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 8
    max_lookback = 72             # 6 min warmup — enough for RSI(36) to stabilise

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rsi2_low",   5.0,  2.0, 10.0),   # RSI(2) oversold threshold
            TunableParam("rsi2_high", 95.0, 90.0, 98.0),   # RSI(2) overbought threshold
            TunableParam("rsi36_min", 35.0, 25.0, 45.0),   # RSI(36) lower bound for neutral trend
            TunableParam("rsi36_max", 65.0, 55.0, 75.0),   # RSI(36) upper bound for neutral trend
            TunableParam("vwap_band",  0.01, 0.005, 0.02), # max deviation from VWAP (fraction)
            TunableParam("vix_max",   25.0, 18.0, 30.0),   # India VIX ceiling
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── Extract spot arrays ──
        close  = spot_df["close"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        day_id = spot_df["day_id"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        # ── Parameters ──
        rsi2_low  = params.get("rsi2_low",   5.0)
        rsi2_high = params.get("rsi2_high", 95.0)
        rsi36_min = params.get("rsi36_min", 35.0)
        rsi36_max = params.get("rsi36_max", 65.0)
        vwap_band = params.get("vwap_band",  0.01)
        vix_max   = params.get("vix_max",   25.0)

        # ── Indicators ──
        rsi_2  = _compute_rsi(close, 2)
        rsi_36 = _compute_rsi(close, 36)
        vwap   = _compute_session_vwap(close, volume, day_id)

        # ── VIX (join asof, default 15.0) ──
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Filters ──
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        low_vix    = vix_close < vix_max
        mid_trend  = (rsi_36 >= rsi36_min) & (rsi_36 <= rsi36_max)

        # VWAP proximity: close within `vwap_band` of VWAP on the correct side
        near_vwap_up = close > vwap * (1.0 - vwap_band)  # for longs: not too far below VWAP
        near_vwap_dn = close < vwap * (1.0 + vwap_band)  # for shorts: not too far above VWAP

        base_filter = in_session & low_vix & mid_trend

        # ── Signals ──
        # RSI(2) < rsi2_low  → micro-selling climax → buy CE (bullish reversion)
        buy_ce = base_filter & near_vwap_up & (rsi_2 < rsi2_low)
        # RSI(2) > rsi2_high → micro-buying climax → buy PE (bearish reversion)
        buy_pe = base_filter & near_vwap_dn & (rsi_2 > rsi2_high)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, 3.0),    # 3 pts = ~6 NIFTY spot pts; beyond this thesis is wrong
            target_points=np.full(n, 5.0),  # 5 pts = ~10 NIFTY spot pts; median micro-snap magnitude
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,              # 90 seconds; snap-backs complete within 30-90s
            max_trades_per_day=self.max_trades_per_day,
        )
