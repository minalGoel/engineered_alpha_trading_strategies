"""Block Deal Momentum v1 — cursor_opus46max_125

Thesis: Block deals signal large institutional position changes.  The stock
drifts in the block deal direction for 1-2 hours.  Proxy block deals using
extreme volume bars early in the session with directional price movement.
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
    name = "cursor_opus46max_125"
    is_long_only = False
    session_start = 555   # 09:15
    session_end = 720     # 12:00 (drift fades by midday)
    max_trades_per_day = 3
    assumptions = ["Block deals proxied via extreme volume bars (>3x average) with price direction"]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_extreme_mult", default=3.0, low=2.0, high=5.0),
            TunableParam("price_move_bps", default=30.0, low=15.0, high=60.0),
            TunableParam("vix_max", default=25.0, low=18.0, high=30.0),
            TunableParam("stop_loss_pct", default=0.002, low=0.001, high=0.004),
            TunableParam("target_pct", default=0.005, low=0.003, high=0.008),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vol_extreme = params.get("vol_extreme_mult", 3.0)
        price_move = params.get("price_move_bps", 30.0)
        vix_max = params.get("vix_max", 25.0)
        stop_pct = params.get("stop_loss_pct", 0.002)
        target_pct = params.get("target_pct", 0.005)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(df["volume"].to_numpy().astype(np.float64), nan=1.0)
        volume = np.clip(volume, 1.0, None)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)
        day_id = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = np.nan_to_num(df_v["_vwap"].to_numpy().astype(np.float64), nan=close[0] if n > 0 else 0.0)

        # Rolling average volume (20 bars)
        avg_vol = np.ones(n, dtype=np.float64)
        for i in range(19, n):
            avg_vol[i] = np.mean(volume[i-19:i+1])
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ratio = volume / avg_vol

        # Block deal proxy: extreme volume bar with significant price move
        block_detected = np.zeros(n, dtype=np.bool_)
        block_direction = np.zeros(n, dtype=np.float64)  # +1 buy, -1 sell
        for i in range(1, n):
            if vol_ratio[i] > vol_extreme:
                safe_prev = close[i-1] if close[i-1] > 1e-10 else 1e-10
                bar_move = (close[i] - close[i-1]) / safe_prev * 10000.0
                if abs(bar_move) > price_move * 0.5:
                    block_detected[i] = True
                    block_direction[i] = np.sign(bar_move)

        # Intraday return from open
        day_open = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if i == 0 or day_id[i] != day_id[i-1]:
                day_open[i] = opn[i]
            else:
                day_open[i] = day_open[i-1]
        safe_do = np.where(day_open > 1e-10, day_open, 1e-10)
        intraday_ret = (close - day_open) / safe_do * 10000.0

        vix_ok = vix < vix_max

        # Entry: after block detection, wait a few bars for settling, then follow
        # Look for block deal within last 15-20 bars
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)
        for i in range(15, n):
            # Check if block was detected in last 15 bars
            block_found = False
            blk_dir = 0.0
            for j in range(max(0, i-15), i):
                if block_detected[j]:
                    block_found = True
                    blk_dir = block_direction[j]
                    break
            if not block_found:
                continue
            if not vix_ok[i]:
                continue
            # Confirm drift continuing
            if blk_dir > 0 and intraday_ret[i] > 0 and close[i] > vwap[i] and vol_ratio[i] > 1.5:
                long_entry[i] = True
            elif blk_dir < 0 and intraday_ret[i] < 0 and close[i] < vwap[i] and vol_ratio[i] > 1.5:
                short_entry[i] = True

        # Limit to early session
        early = (time_mins >= 570) & (time_mins <= 660)
        long_entry = long_entry & early
        short_entry = short_entry & early

        # Signal exit: volume ratio drops below 1.0
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if vol_ratio[i] < 1.0 and vol_ratio[i-1] >= 1.0:
                sig_exit_long[i] = True
                sig_exit_short[i] = True

        atr = _compute_atr(high, low, close, 14)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            trailing_stop_pct=0.0015,
            trailing_activate_pct=0.003,
            time_stop_bars=90,
        )
