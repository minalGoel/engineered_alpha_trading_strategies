"""Gap Fade Reversion — cursor_gemini31pro_strategy_103

Thesis: Overnight gaps caused by retail overreaction are often overextended.
Institutional participants fade the gap and revert to previous day's close.
Entry on gap-down with bullish bar (close > open) and volume surge, vice versa
for shorts. Target is previous day close (gap fill). 5min timeframe.
Gap threshold: 21 bps. VIX filter: 12-22. Time stop: 60 bars.
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
    name = "gemini31pro_gap_fade_reversion_v103"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 915     # 15:15
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_thresh_bps", default=21.0, low=10.0, high=50.0),
            TunableParam("vix_low", default=12.0, low=8.0, high=18.0),
            TunableParam("vix_high", default=22.0, low=18.0, high=30.0),
            TunableParam("stop_atr_mult", default=1.5, low=0.8, high=3.0),
            TunableParam("breakeven_pct", default=0.005, low=0.002, high=0.01),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        gap_thresh = params.get("gap_thresh_bps", 21.0)
        vix_lo = params.get("vix_low", 12.0)
        vix_hi = params.get("vix_high", 22.0)
        stop_atr = params.get("stop_atr_mult", 1.5)
        be_pct = params.get("breakeven_pct", 0.005)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(df["volume"].to_numpy().astype(np.float64), nan=1.0)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)
        day_id = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        atr14 = _compute_atr(high, low, close, 14)

        # Compute previous day close and gap percentage per bar
        day_last_close = {}
        for i in range(n):
            day_last_close[day_id[i]] = close[i]
        sorted_days = sorted(set(day_id))
        day_to_prev_close = {}
        for idx, d in enumerate(sorted_days):
            if idx == 0:
                day_to_prev_close[d] = np.nan
            else:
                day_to_prev_close[d] = day_last_close[sorted_days[idx - 1]]

        prev_close = np.full(n, np.nan, dtype=np.float64)
        day_open = np.zeros(n, dtype=np.float64)
        for i in range(n):
            prev_close[i] = day_to_prev_close.get(day_id[i], np.nan)
        prev_close = np.nan_to_num(prev_close, nan=close[0])

        # Day open: first bar's open for each day
        day_open_map = {}
        for i in range(n):
            d = day_id[i]
            if d not in day_open_map:
                day_open_map[d] = opn[i]
        for i in range(n):
            day_open[i] = day_open_map[day_id[i]]

        safe_prev = np.where(prev_close > 1e-10, prev_close, 1e-10)
        gap_pct_bps = (day_open - prev_close) / safe_prev * 10000.0  # in bps

        # Volume SMA(5)
        vol_sma5 = np.full(n, 1.0, dtype=np.float64)
        for i in range(4, n):
            vol_sma5[i] = np.mean(volume[i - 4:i + 1])
        vol_sma5 = np.clip(vol_sma5, 1.0, None)
        vol_ok = volume > vol_sma5

        # Filters
        vix_ok = (vix >= vix_lo) & (vix <= vix_hi)
        time_ok = (time_mins >= 570) & (time_mins <= 870)  # 09:30-14:30

        # Entry: fade the gap
        # Long: gap down (< -thresh), bullish bar (close > open), volume surge
        long_entry = (gap_pct_bps < -gap_thresh) & (close > opn) & vol_ok & vix_ok & time_ok
        # Short: gap up (> thresh), bearish bar (close < open), volume surge
        short_entry = (gap_pct_bps > gap_thresh) & (close < opn) & vol_ok & vix_ok & time_ok

        # Target indicator: previous day close (gap fill target)
        target_ind = prev_close.copy()

        # Signal exit: for longs, exit when close >= prev_close; for shorts, when close <= prev_close
        signal_exit_long = close >= prev_close
        signal_exit_short = close <= prev_close

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr14,
            target_indicator=target_ind,
            use_target_indicator=True,
            stop_loss_atr_mult=stop_atr,
            breakeven_pct=be_pct,
            time_stop_bars=60,
        )
