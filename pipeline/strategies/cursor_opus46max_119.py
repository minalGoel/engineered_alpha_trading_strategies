# AUDIT FIX: Add session window filtering to prevent entries outside session_start/session_end
"""Sector VWAP Divergence v1 — cursor_opus46max_119

Thesis: When a stock trades below its VWAP while the index (sector proxy)
trades above its VWAP, the divergence resolves in favor of the sector
direction ~60% of the time (stock reverts to sector trend).
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
    name = "cursor_opus46max_119"
    is_long_only = False
    session_start = 600   # 10:00
    session_end = 915     # 15:15
    max_trades_per_day = 8
    assumptions = ["Sector VWAP position proxied using index_close relative to index rolling VWAP"]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vwap_dev_thresh", default=5.0, low=3.0, high=10.0),
            TunableParam("divergence_bars", default=5, low=3, high=10),
            TunableParam("stop_loss_pct", default=0.0015, low=0.0008, high=0.003),
            TunableParam("target_pct", default=0.0025, low=0.0012, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vwap_dev_thresh = params.get("vwap_dev_thresh", 5.0)
        div_bars = int(params.get("divergence_bars", 5))
        stop_pct = params.get("stop_loss_pct", 0.0015)
        target_pct = params.get("target_pct", 0.0025)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(df["volume"].to_numpy().astype(np.float64), nan=1.0)
        volume = np.clip(volume, 1.0, None)
        index_close = np.nan_to_num(df["index_close"].to_numpy().astype(np.float64), nan=0.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        in_session = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # Stock VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        stock_vwap = np.nan_to_num(df_v["_vwap"].to_numpy().astype(np.float64), nan=close[0] if n > 0 else 0.0)

        # Index VWAP proxy: SMA of index_close (since we don't have index volume)
        index_vwap = np.zeros(n, dtype=np.float64)
        day_id = df["day_id"].to_numpy()
        idx_sum = 0.0
        idx_count = 0
        prev_day = day_id[0] if n > 0 else -1
        for i in range(n):
            if day_id[i] != prev_day:
                idx_sum = 0.0
                idx_count = 0
                prev_day = day_id[i]
            idx_sum += index_close[i]
            idx_count += 1
            index_vwap[i] = idx_sum / idx_count if idx_count > 0 else index_close[i]

        # VWAP positions in bps
        safe_svwap = np.where(stock_vwap > 1e-10, stock_vwap, 1e-10)
        safe_ivwap = np.where(index_vwap > 1e-10, index_vwap, 1e-10)
        stock_vwap_pos = (close - stock_vwap) / safe_svwap * 10000.0
        index_vwap_pos = (index_close - index_vwap) / safe_ivwap * 10000.0

        # Divergence: different signs of VWAP position
        divergence_duration = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            s_sign = 1 if stock_vwap_pos[i] > 0 else (-1 if stock_vwap_pos[i] < 0 else 0)
            i_sign = 1 if index_vwap_pos[i] > 0 else (-1 if index_vwap_pos[i] < 0 else 0)
            if s_sign != i_sign and s_sign != 0 and i_sign != 0:
                divergence_duration[i] = divergence_duration[i-1] + 1
            else:
                divergence_duration[i] = 0

        # Volume filter
        avg_vol = np.ones(n, dtype=np.float64)
        for i in range(19, n):
            avg_vol[i] = np.mean(volume[i-19:i+1])
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > 0.5 * avg_vol

        # Long: index above VWAP, stock below -> stock should catch up
        # Confirmation: stock crosses above its VWAP
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if (index_vwap_pos[i] > vwap_dev_thresh and stock_vwap_pos[i-1] < -vwap_dev_thresh and
                    divergence_duration[i-1] >= div_bars and
                    close[i] >= stock_vwap[i] and close[i-1] < stock_vwap[i-1] and vol_ok[i] and
                    in_session[i]):
                long_entry[i] = True
            if (index_vwap_pos[i] < -vwap_dev_thresh and stock_vwap_pos[i-1] > vwap_dev_thresh and
                    divergence_duration[i-1] >= div_bars and
                    close[i] <= stock_vwap[i] and close[i-1] > stock_vwap[i-1] and vol_ok[i] and
                    in_session[i]):
                short_entry[i] = True

        # Signal exit: sector VWAP position changes sign
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if index_vwap_pos[i] <= 0 and index_vwap_pos[i-1] > 0:
                sig_exit_long[i] = True
            if index_vwap_pos[i] >= 0 and index_vwap_pos[i-1] < 0:
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
            trailing_stop_pct=0.0007,
            trailing_activate_pct=0.0012,
            time_stop_bars=30,
        )
