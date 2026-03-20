"""Gap-and-Go with VWAP Pullback — Opus_31

Thesis: Gaps >0.5% with extreme volume in the first 5 minutes, followed
by a pullback to VWAP that holds, signal continuation in the gap direction.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_atr(high, low, close, period):
    """Wilder's ATR."""
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    if n < period + 1:
        return atr
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr[period] = np.mean(tr[1:period + 1])
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


class Strategy(BaseStrategy):
    name = "opus_gap_and_go_v1"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 690     # 11:30
    max_trades_per_day = 2

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_thresh", default=0.005, low=0.003, high=0.01),
            TunableParam("vol_ratio_thresh", default=3.0, low=2.0, high=5.0),
            TunableParam("target_pct", default=0.005, low=0.003, high=0.008),
            TunableParam("stop_loss_pct", default=0.003, low=0.002, high=0.005),
            TunableParam("trailing_activate_pct", default=0.003, low=0.002, high=0.005),
            TunableParam("trailing_stop_pct", default=0.002, low=0.001, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        gap_thresh = params.get("gap_thresh", 0.005)
        vol_ratio_thresh = params.get("vol_ratio_thresh", 3.0)
        target_pct = params.get("target_pct", 0.005)
        stop_pct = params.get("stop_loss_pct", 0.003)
        trail_activate = params.get("trailing_activate_pct", 0.003)
        trail_stop = params.get("trailing_stop_pct", 0.002)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        day_id = df["day_id"].to_numpy().astype(np.int64)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── VWAP (daily reset) ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── Previous day close ──
        day_last_close = {}
        for i in range(n):
            day_last_close[day_id[i]] = close[i]

        prev_day_close = np.zeros(n, dtype=np.float64)
        for i in range(n):
            d = day_id[i]
            if d - 1 in day_last_close:
                prev_day_close[i] = day_last_close[d - 1]

        # ── Per-day gap ──
        day_first_open = {}
        for i in range(n):
            d = day_id[i]
            if d not in day_first_open:
                day_first_open[d] = open_[i]

        day_gap = np.zeros(n, dtype=np.float64)
        for i in range(n):
            d = day_id[i]
            if prev_day_close[i] > 1e-10:
                day_gap[i] = (day_first_open[d] - prev_day_close[i]) / prev_day_close[i]

        # ── First-5-min volume ratio per day ──
        # Sum volume in first 5 bars (555-560) vs rolling avg
        day_vol_5min = {}
        day_vol_5min_count = {}
        for i in range(n):
            d = day_id[i]
            if 555 <= time_mins[i] < 560:
                day_vol_5min[d] = day_vol_5min.get(d, 0.0) + volume[i]
                day_vol_5min_count[d] = day_vol_5min_count.get(d, 0) + 1

        avg_vol = df["volume"].rolling_mean(30).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)

        vol_5min_ratio = np.zeros(n, dtype=np.float64)
        for i in range(n):
            d = day_id[i]
            if d in day_vol_5min and day_vol_5min_count.get(d, 0) > 0:
                avg_bar_vol_5min = day_vol_5min[d] / day_vol_5min_count[d]
                if avg_vol[i] > 1e-10:
                    vol_5min_ratio[i] = avg_bar_vol_5min / avg_vol[i]

        # ── VWAP touch: low near VWAP (within 0.1%) ──
        safe_vwap = np.clip(np.abs(vwap), 1e-10, None)
        vwap_touch = np.abs(low - vwap) / safe_vwap < 0.001

        # ── ATR(20) ──
        atr = _compute_atr(high, low, close, 20)

        # ── Filters ──
        time_ok = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # ── Entries ──
        long_entry = (
            (day_gap > gap_thresh)
            & (vol_5min_ratio > vol_ratio_thresh)
            & vwap_touch
            & (close > vwap)
            & time_ok
        )
        short_entry = (
            (day_gap < -gap_thresh)
            & (vol_5min_ratio > vol_ratio_thresh)
            & (np.abs(high - vwap) / safe_vwap < 0.001)
            & (close < vwap)
            & time_ok
        )

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=target_pct,
            stop_loss_pct=stop_pct,
            trailing_stop_pct=trail_stop,
            trailing_activate_pct=trail_activate,
            time_stop_bars=130,  # ~by 690 from 560
        )
