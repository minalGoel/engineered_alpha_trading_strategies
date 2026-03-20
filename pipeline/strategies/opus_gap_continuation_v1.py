"""Large Gap Continuation with ORB Confirmation — Opus_29

Thesis: Gaps >1.5% that hold above the Opening Range Breakout high tend
to continue in the gap direction, driven by institutional momentum.
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
    name = "opus_gap_continuation_v1"
    is_long_only = False
    session_start = 555   # 09:15
    session_end = 660     # 11:00
    max_trades_per_day = 1

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_thresh", default=0.015, low=0.01, high=0.025),
            TunableParam("stop_loss_pct", default=0.005, low=0.003, high=0.008),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        gap_thresh = params.get("gap_thresh", 0.015)
        stop_pct = params.get("stop_loss_pct", 0.005)

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
        prev_day_close = np.zeros(n, dtype=np.float64)
        last_close_by_day = {}
        for i in range(n):
            d = day_id[i]
            if d - 1 in last_close_by_day:
                prev_day_close[i] = last_close_by_day[d - 1]
            elif d in last_close_by_day and time_mins[i] == time_mins[0]:
                prev_day_close[i] = last_close_by_day[d]
            last_close_by_day[d] = close[i]

        # Recompute: use first bar of each day to find prev close
        # Track the last close of the previous day
        day_last_close = {}
        for i in range(n):
            day_last_close[day_id[i]] = close[i]

        prev_day_close = np.zeros(n, dtype=np.float64)
        for i in range(n):
            d = day_id[i]
            if d - 1 in day_last_close:
                prev_day_close[i] = day_last_close[d - 1]

        # ── Gap pct ──
        safe_prev = np.clip(prev_day_close, 1e-10, None)
        gap_pct = np.where(prev_day_close > 0, (open_[0:n] - prev_day_close) / safe_prev, 0.0)
        # Per-day gap: use first bar open of each day
        day_gap = np.zeros(n, dtype=np.float64)
        day_first_open = {}
        for i in range(n):
            d = day_id[i]
            if d not in day_first_open:
                day_first_open[d] = open_[i]
        for i in range(n):
            d = day_id[i]
            if prev_day_close[i] > 1e-10:
                day_gap[i] = (day_first_open[d] - prev_day_close[i]) / prev_day_close[i]

        # ── 15-min Opening Range (555-570) ──
        orb_high = np.zeros(n, dtype=np.float64)
        orb_low = np.full(n, 1e10, dtype=np.float64)
        day_orb_high = {}
        day_orb_low = {}
        for i in range(n):
            d = day_id[i]
            if 555 <= time_mins[i] < 570:
                if d not in day_orb_high:
                    day_orb_high[d] = high[i]
                    day_orb_low[d] = low[i]
                else:
                    day_orb_high[d] = max(day_orb_high[d], high[i])
                    day_orb_low[d] = min(day_orb_low[d], low[i])
        for i in range(n):
            d = day_id[i]
            if d in day_orb_high:
                orb_high[i] = day_orb_high[d]
                orb_low[i] = day_orb_low[d]
            else:
                orb_high[i] = 0.0
                orb_low[i] = 1e10

        # ── Avg volume ──
        avg_vol = df["volume"].rolling_mean(30).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)

        # ── ATR(20) ──
        atr = _compute_atr(high, low, close, 20)

        # ── Filters ──
        time_ok = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # ── Entries ──
        long_entry = (
            (day_gap > gap_thresh)
            & (close > orb_high)
            & (close > vwap)
            & (volume > avg_vol)
            & (time_mins >= 570)
            & time_ok
        )
        short_entry = (
            (day_gap < -gap_thresh)
            & (close < orb_low)
            & (time_mins >= 570)
            & time_ok
        )

        # ── Signal exit: close crosses back inside ORB ──
        orb_mid = (orb_high + orb_low) / 2.0
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if close[i] < orb_high[i] and close[i - 1] >= orb_high[i - 1] and orb_high[i] > 0:
                sig_exit_long[i] = True
            if close[i] > orb_low[i] and close[i - 1] <= orb_low[i - 1] and orb_low[i] < 1e9:
                sig_exit_short[i] = True

        # ── Target: 0.5 * |gap_pct| ──
        # Use median gap as target_pct approximation
        abs_gap = np.abs(day_gap)
        median_gap = np.median(abs_gap[abs_gap > 0]) if np.any(abs_gap > 0) else 0.005
        target_pct = max(0.5 * median_gap, 0.002)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=target_pct,
            stop_loss_pct=stop_pct,
            time_stop_bars=90,
        )
