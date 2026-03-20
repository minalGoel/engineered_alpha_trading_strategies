# AUDIT FIX: Add session window filter to entry signals (entries were firing outside session_start/session_end)
"""Vol Breakout Squeeze v1 — cursor_opus46max_080

Thesis: When ATR(14) contracts to 120-bar min AND BB width percentile < 10
(dual squeeze), the subsequent expansion produces tradeable breakouts. Enter
in the direction of the first expansion bar with volume confirmation.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_atr(high, low, close, period):
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    if n < period:
        return atr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


class Strategy(BaseStrategy):
    name = "cursor_opus46max_080"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 910     # 15:10
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("bb_width_pctile_thresh", default=10.0, low=5.0, high=20.0),
            TunableParam("squeeze_lookback", default=120.0, low=60.0, high=180.0),
            TunableParam("vol_mult", default=1.3, low=1.0, high=2.0),
            TunableParam("stop_loss_pct", default=0.0025, low=0.0015, high=0.004),
            TunableParam("target_atr_mult", default=2.0, low=1.0, high=3.0),
            TunableParam("trailing_stop_pct", default=0.003, low=0.001, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        bb_pctile_thresh = params.get("bb_width_pctile_thresh", 10.0)
        squeeze_lb = int(params.get("squeeze_lookback", 120.0))
        vol_mult = params.get("vol_mult", 1.3)
        stop_pct = params.get("stop_loss_pct", 0.0025)
        target_atr_mult = params.get("target_atr_mult", 2.0)
        trailing_pct = params.get("trailing_stop_pct", 0.003)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        time_min = df["time_minutes"].to_numpy()
        in_session = (time_min >= self.session_start) & (time_min <= self.session_end)

        # ── ATR(14) ──
        atr = _compute_atr(high, low, close, 14)

        # ── ATR at 120-bar min ──
        atr_at_min = np.zeros(n, dtype=np.bool_)
        for i in range(squeeze_lb, n):
            atr_min_val = np.min(atr[i - squeeze_lb:i + 1])
            if atr[i] > 0 and atr[i] <= atr_min_val * 1.001:
                atr_at_min[i] = True

        # ── BB width and percentile ──
        sma20 = np.zeros(n, dtype=np.float64)
        std20 = np.zeros(n, dtype=np.float64)
        for i in range(20, n):
            sma20[i] = np.mean(close[i - 20:i])
            std20[i] = np.std(close[i - 20:i])
        bb_width = np.where(sma20 > 0, (2.0 * std20 * 2.0) / sma20, 1.0)

        bb_width_pctile = np.full(n, 50.0, dtype=np.float64)
        for i in range(squeeze_lb, n):
            window = bb_width[i - squeeze_lb:i + 1]
            bb_width_pctile[i] = (np.sum(window < bb_width[i]) / len(window)) * 100.0

        # ── Dual squeeze ──
        dual_squeeze = atr_at_min & (bb_width_pctile < bb_pctile_thresh)

        # ── Squeeze was true within last 5 bars ──
        recent_squeeze = np.zeros(n, dtype=np.bool_)
        for i in range(5, n):
            if np.any(dual_squeeze[max(0, i - 5):i]):
                recent_squeeze[i] = True

        # ── ATR expanding (inflection) ──
        atr_expanding = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if atr[i] > atr[i - 1] and i >= 2:
                atr_expanding[i] = True

        # ── Expansion bar direction ──
        bullish_bar = close > opn
        bearish_bar = close < opn

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > vol_mult * avg_vol

        # ── Entry ──
        long_entry = recent_squeeze & atr_expanding & bullish_bar & (close > sma20) & vol_ok & in_session
        short_entry = recent_squeeze & atr_expanding & bearish_bar & (close < sma20) & vol_ok & in_session

        # ── Signal exit: ATR starts contracting within 10 bars ──
        signal_exit_long = np.zeros(n, dtype=np.bool_)
        signal_exit_short = np.zeros(n, dtype=np.bool_)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_atr_mult=target_atr_mult,
            trailing_stop_pct=trailing_pct,
            trailing_activate_pct=trailing_pct,
            time_stop_bars=45,
        )
