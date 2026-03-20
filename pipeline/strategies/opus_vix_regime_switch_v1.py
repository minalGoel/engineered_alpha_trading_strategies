# AUDIT FIX: Added session window time filter — entries were firing outside session_start/session_end
"""VIX Regime Switch — Opus_23

Thesis: When VIX spikes above its Bollinger upper band but starts declining,
a volatility crush (and equity rally) often follows.  Conversely, when
VIX drops below the lower band and starts rising, a sell-off may begin.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Wilder's ATR."""
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i - 1]),
                     abs(low[i] - close[i - 1]))
    if n < period:
        return atr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


class Strategy(BaseStrategy):
    name = "opus_vix_regime_switch_v1"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 870     # 14:30
    max_trades_per_day = 2

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("bb_mult", default=2.0, low=1.5, high=2.5),
            TunableParam("index_ret_floor", default=-0.005, low=-0.01, high=-0.002),
            TunableParam("stop_loss_pct", default=0.004, low=0.002, high=0.006),
            TunableParam("target_pct", default=0.006, low=0.004, high=0.01),
            TunableParam("trailing_stop_pct", default=0.003, low=0.002, high=0.005),
            TunableParam("trailing_activate_pct", default=0.004, low=0.002, high=0.006),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        bb_mult = params.get("bb_mult", 2.0)
        idx_ret_floor = params.get("index_ret_floor", -0.005)
        stop_pct = params.get("stop_loss_pct", 0.004)
        target_pct = params.get("target_pct", 0.006)
        trail_pct = params.get("trailing_stop_pct", 0.003)
        trail_act = params.get("trailing_activate_pct", 0.004)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=20.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── Bollinger Bands on VIX (20-bar) ──
        bb_period = 20
        vix_sma = np.zeros(n, dtype=np.float64)
        vix_std = np.zeros(n, dtype=np.float64)
        for i in range(bb_period - 1, n):
            window = vix[i - bb_period + 1: i + 1]
            vix_sma[i] = np.mean(window)
            vix_std[i] = np.std(window)

        bb_upper = vix_sma + bb_mult * vix_std
        bb_lower = vix_sma - bb_mult * vix_std

        # ── VIX change over 60 bars ──
        vix_change_60 = np.zeros(n, dtype=np.float64)
        for i in range(60, n):
            vix_change_60[i] = vix[i] - vix[i - 60]

        # ── Index 60-bar return ──
        index_ret_60 = np.zeros(n, dtype=np.float64)
        for i in range(60, n):
            if index_close[i - 60] > 0:
                index_ret_60[i] = (index_close[i] - index_close[i - 60]) / index_close[i - 60]

        atr14 = _compute_atr(high, low, close, 14)

        # ── Session window filter ──
        time_ok = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # ── Long: VIX > BB_upper AND declining AND index not crashing ──
        long_entry = ((vix > bb_upper) &
                      (vix_change_60 < 0) &
                      (index_ret_60 > idx_ret_floor) &
                      (bb_upper > 0) &
                      time_ok)

        # ── Short: VIX < BB_lower AND rising ──
        short_entry = ((vix < bb_lower) &
                       (vix_change_60 > 0) &
                       (bb_lower > 0) &
                       time_ok)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr14,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            trailing_stop_pct=trail_pct,
            trailing_activate_pct=trail_act,
            time_stop_bars=300,  # exit by 14:30
        )
