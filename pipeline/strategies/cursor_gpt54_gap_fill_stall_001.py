"""Gap Fill Stall — cursor_gpt54_gap_fill_stall_001

Thesis: Exploits intraday partial gap reversion after an overnight move stalls
against session flow and fails to extend beyond the early range.  Pre-open
price discovery is coarse and traders overreact to overnight headlines.

Entry long: gap down >= 1.2%, price pierced OR-low then reclaimed, close < VWAP,
    bullish bar, relative volume >= 1.2, index neutral.
Entry short: gap up >= 1.2%, price pierced OR-high then reclaimed, close > VWAP,
    bearish bar, relative volume >= 1.2, index neutral.
Target: VWAP (use_target_indicator).  Stop: 0.50%.  BE after +0.30%.
Time stop: 15 bars.
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
    name = "gap_fill_stall_001"
    is_long_only = False
    session_start = 565   # 09:25
    session_end = 645     # 10:45
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_thresh", default=0.012, low=0.008, high=0.020),
            TunableParam("rel_vol_min", default=1.2, low=1.0, high=2.0),
            TunableParam("index_ret_max", default=0.004, low=0.002, high=0.008),
            TunableParam("stop_loss_pct", default=0.005, low=0.003, high=0.008),
            TunableParam("breakeven_pct", default=0.003, low=0.002, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        gap_thresh = params.get("gap_thresh", 0.012)
        rel_vol_min = params.get("rel_vol_min", 1.2)
        index_ret_max = params.get("index_ret_max", 0.004)
        stop_pct = params.get("stop_loss_pct", 0.005)
        be_pct = params.get("breakeven_pct", 0.003)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        day_ids = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        index_close = df["index_close"].to_numpy().astype(np.float64)

        atr = _compute_atr(high, low, close, 14)

        # ── VWAP (daily reset) ──
        vwap_df = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume")).cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns(
            (pl.col("_ctv") / pl.col("_cv")).alias("_vwap"),
        )
        vwap = vwap_df["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── Relative volume (volume / SMA(volume,20)) ──
        vol_sma20 = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        vol_sma20 = np.nan_to_num(vol_sma20, nan=1.0)
        vol_sma20 = np.clip(vol_sma20, 1.0, None)
        rel_vol = volume / vol_sma20

        # ── Gap pct per day ──
        gap_pct = np.zeros(n, dtype=np.float64)
        unique_days = np.unique(day_ids)
        prev_close = np.nan
        for d in unique_days:
            mask = day_ids == d
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            if not np.isnan(prev_close) and prev_close > 0:
                gap_pct[idx] = (open_[idx[0]] - prev_close) / prev_close
            prev_close = close[idx[-1]]

        # ── Opening range (first 15 market minutes: 09:15-09:29 = 555-569) ──
        or_high = np.full(n, np.nan, dtype=np.float64)
        or_low = np.full(n, np.nan, dtype=np.float64)
        for d in unique_days:
            mask = day_ids == d
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            or_bars = idx[time_mins[idx] < 570]  # first 15 min (555-569)
            if len(or_bars) == 0:
                continue
            oh = np.max(high[or_bars])
            ol = np.min(low[or_bars])
            or_high[idx] = oh
            or_low[idx] = ol
        or_high = np.nan_to_num(or_high, nan=0.0)
        or_low = np.nan_to_num(or_low, nan=0.0)

        # ── Index return over 15 bars ──
        index_ret_15 = np.zeros(n, dtype=np.float64)
        for i in range(15, n):
            if index_close[i - 15] > 0:
                index_ret_15[i] = index_close[i] / index_close[i - 15] - 1.0

        # ── OR range filter: skip if first 15-min range > 1.8 * ATR(14) ──
        or_range_ok = np.ones(n, dtype=np.bool_)
        for d in unique_days:
            mask = day_ids == d
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            or_rng = or_high[idx[0]] - or_low[idx[0]]
            atr_val = atr[idx[0]] if atr[idx[0]] > 0 else atr[min(idx[-1], n - 1)]
            if atr_val > 0 and or_rng > 1.8 * atr_val:
                or_range_ok[idx] = False

        # ── Time filter: only 09:25-10:45 (565-645) ──
        time_ok = (time_mins >= 565) & (time_mins <= 645)

        # ── Entry conditions ──
        bullish = close > open_
        bearish = close < open_

        long_entry = (
            (gap_pct <= -gap_thresh) &
            (low < or_low) &
            (close > or_low) &
            (close < vwap) &
            (rel_vol >= rel_vol_min) &
            (np.abs(index_ret_15) <= index_ret_max) &
            bullish &
            time_ok &
            or_range_ok
        )

        short_entry = (
            (gap_pct >= gap_thresh) &
            (high > or_high) &
            (close < or_high) &
            (close > vwap) &
            (rel_vol >= rel_vol_min) &
            (np.abs(index_ret_15) <= index_ret_max) &
            bearish &
            time_ok &
            or_range_ok
        )

        # ── Signal exit: price fills to VWAP OR re-breaks OR extreme ──
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            # Long exit: close >= vwap or close < or_low (re-break)
            if close[i] >= vwap[i] and vwap[i] > 0:
                sig_exit_long[i] = True
            if close[i] < or_low[i] and or_low[i] > 0:
                sig_exit_long[i] = True
            # Short exit: close <= vwap or close > or_high (re-break)
            if close[i] <= vwap[i] and vwap[i] > 0:
                sig_exit_short[i] = True
            if close[i] > or_high[i] and or_high[i] > 0:
                sig_exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=vwap,
            stop_loss_pct=stop_pct,
            use_target_indicator=True,
            breakeven_pct=be_pct,
            time_stop_bars=15,
        )
