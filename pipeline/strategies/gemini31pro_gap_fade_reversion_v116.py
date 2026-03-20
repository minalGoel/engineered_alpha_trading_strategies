"""Gap Fade Reversion — cursor_gemini31pro_strategy_116

Thesis: Overnight gaps caused by retail overreaction are often overextended.
15min timeframe, FNO universe. Gap threshold: 50 bps (large gaps only).
VIX > 15. Volume filter: vol > SMA(vol,20). Time stop: 60 bars.
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
    name = "gemini31pro_gap_fade_reversion_v116"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 915     # 15:15
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_thresh_bps", default=50.0, low=20.0, high=100.0),
            TunableParam("vix_min", default=15.0, low=10.0, high=25.0),
            TunableParam("stop_atr_mult", default=1.5, low=0.8, high=3.0),
            TunableParam("breakeven_pct", default=0.005, low=0.002, high=0.01),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        gap_thresh = params.get("gap_thresh_bps", 50.0)
        vix_min = params.get("vix_min", 15.0)
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

        # Previous day close
        day_last_close = {}
        for i in range(n):
            day_last_close[day_id[i]] = close[i]
        sorted_days = sorted(set(day_id))
        day_to_prev_close = {}
        for idx, d in enumerate(sorted_days):
            day_to_prev_close[d] = day_last_close[sorted_days[idx - 1]] if idx > 0 else np.nan

        prev_close = np.full(n, np.nan, dtype=np.float64)
        for i in range(n):
            prev_close[i] = day_to_prev_close.get(day_id[i], np.nan)
        prev_close = np.nan_to_num(prev_close, nan=close[0])

        # Day open
        day_open_map = {}
        day_open = np.zeros(n, dtype=np.float64)
        for i in range(n):
            d = day_id[i]
            if d not in day_open_map:
                day_open_map[d] = opn[i]
        for i in range(n):
            day_open[i] = day_open_map[day_id[i]]

        safe_prev = np.where(prev_close > 1e-10, prev_close, 1e-10)
        gap_pct_bps = (day_open - prev_close) / safe_prev * 10000.0

        # Volume SMA(5) for bar-level confirmation
        vol_sma5 = np.full(n, 1.0, dtype=np.float64)
        for i in range(4, n):
            vol_sma5[i] = np.mean(volume[i - 4:i + 1])
        vol_sma5 = np.clip(vol_sma5, 1.0, None)
        vol_ok_bar = volume > vol_sma5

        # Volume SMA(20) for overall filter
        vol_sma20 = np.full(n, 1.0, dtype=np.float64)
        for i in range(19, n):
            vol_sma20[i] = np.mean(volume[i - 19:i + 1])
        vol_sma20 = np.clip(vol_sma20, 1.0, None)
        vol_ok_filter = volume > vol_sma20

        vix_ok = vix > vix_min
        time_ok = (time_mins >= 570) & (time_mins <= 870)

        long_entry = (gap_pct_bps < -gap_thresh) & (close > opn) & vol_ok_bar & vol_ok_filter & vix_ok & time_ok
        short_entry = (gap_pct_bps > gap_thresh) & (close < opn) & vol_ok_bar & vol_ok_filter & vix_ok & time_ok

        signal_exit_long = close >= prev_close
        signal_exit_short = close <= prev_close

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr14,
            target_indicator=prev_close.copy(),
            use_target_indicator=True,
            stop_loss_atr_mult=stop_atr,
            breakeven_pct=be_pct,
            time_stop_bars=60,
        )
