"""Budget Day Sector Reversion — cursor_opus46max_129

Thesis: Budget day creates sector-specific overreactions during the speech
(11:00-12:30). After 12:30, extreme sector movers partially revert. Fade the
biggest movers. Proxied by: large cumulative intraday return + reversal signal.
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
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    if n < period:
        return atr
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr


class Strategy(BaseStrategy):
    name = "cursor_opus46max_129"
    is_long_only = False
    session_start = 750   # 12:30
    session_end = 915     # 15:15
    max_trades_per_day = 6
    assumptions = [
        "Budget day calendar not available; strategy fires when large intraday moves detected",
        "Sector-specific data not available; uses single-stock cumulative return",
    ]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("cum_return_thresh_bps", default=100.0, low=50.0, high=200.0),
            TunableParam("reversion_target_pct", default=0.4, low=0.2, high=0.6),
            TunableParam("stop_extension_pct", default=0.5, low=0.3, high=0.7),
            TunableParam("vol_mult", default=2.0, low=1.3, high=3.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        cum_thresh = params.get("cum_return_thresh_bps", 100.0)
        rev_target = params.get("reversion_target_pct", 0.4)
        stop_ext = params.get("stop_extension_pct", 0.5)
        vol_mult = params.get("vol_mult", 2.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        day_id = df["day_id"].to_numpy()

        atr = _compute_atr(high, low, close, 14)

        # Daily open
        day_open = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if i == 0 or day_id[i] != day_id[i-1]:
                day_open[i] = open_[i]
            else:
                day_open[i] = day_open[i-1]

        # Cumulative return from open (bps)
        safe_open = np.clip(day_open, 1e-10, None)
        cum_ret = (close - day_open) / safe_open * 10000.0

        # Rolling high/low of the day
        day_high = np.zeros(n, dtype=np.float64)
        day_low = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if i == 0 or day_id[i] != day_id[i-1]:
                day_high[i] = high[i]
                day_low[i] = low[i]
            else:
                day_high[i] = max(day_high[i-1], high[i])
                day_low[i] = min(day_low[i-1], low[i])

        # Volume filter
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > vol_mult * avg_vol

        # Time: post-speech reversion (12:30-13:30 entry window)
        time_entry = (time_mins >= 750) & (time_mins <= 810)

        # Long: stock dropped > cum_thresh bps, not making new lows
        long_entry = (
            (cum_ret < -cum_thresh) &
            (close > day_low * 1.001) &
            vol_ok &
            time_entry
        )

        # Short: stock rallied > cum_thresh bps, not making new highs
        short_entry = (
            (cum_ret > cum_thresh) &
            (close < day_high * 0.999) &
            vol_ok &
            time_entry
        )

        # Target: partial reversion
        target_pct = cum_thresh * rev_target / 10000.0
        stop_pct = cum_thresh * stop_ext / 10000.0

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            trailing_stop_pct=0.002,
            trailing_activate_pct=0.003,
            time_stop_bars=90,
        )
