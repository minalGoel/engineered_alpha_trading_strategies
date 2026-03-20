"""FII/DII Flow Signal v1 — cursor_opus46max_199

Thesis: Institutional (large order) flow diverging from retail (small order)
flow predicts future direction. When institutions are net buying while retail
is selling, follow institutions. Adapted: use volume-weighted price pressure
as proxy for institutional vs retail flow.
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
    name = "cursor_opus46max_199"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 910     # 15:10
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("flow_window", default=30.0, low=15.0, high=50.0),
            TunableParam("pressure_thresh", default=0.3, low=0.15, high=0.50),
            TunableParam("confirm_bars", default=5.0, low=3.0, high=8.0),
            TunableParam("vix_max", default=25.0, low=18.0, high=30.0),
            TunableParam("stop_loss_pct", default=0.0025, low=0.0015, high=0.004),
            TunableParam("target_pct", default=0.004, low=0.003, high=0.006),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        flow_win = int(params.get("flow_window", 30.0))
        press_th = params.get("pressure_thresh", 0.3)
        confirm = int(params.get("confirm_bars", 5.0))
        vix_max = params.get("vix_max", 25.0)
        stop_pct = params.get("stop_loss_pct", 0.0025)
        target_pct = params.get("target_pct", 0.004)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(index_close, nan=0.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # Volume-weighted price pressure (proxy for institutional flow)
        # Large volume bars with small range = institutional accumulation/distribution
        # Small volume bars with large range = retail noise
        bar_range = high - low
        safe_range = np.where(bar_range > 0, bar_range, 1e-10)

        # "Institutional" pressure: high volume, low range, directional
        vol_intensity = volume / np.median(volume[volume > 0]) if np.any(volume > 0) else volume
        range_norm = bar_range / np.median(bar_range[bar_range > 0]) if np.any(bar_range > 0) else bar_range

        # Direction: close vs open
        bar_dir = np.sign(close - opn)

        # Institutional proxy: high volume + low range = accumulation
        inst_flow = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if range_norm[i] > 0:
                inst_flow[i] = bar_dir[i] * vol_intensity[i] / max(range_norm[i], 0.1)

        # Rolling institutional pressure
        inst_pressure = np.zeros(n, dtype=np.float64)
        for i in range(flow_win, n):
            seg = inst_flow[i-flow_win+1:i+1]
            pos_sum = np.sum(seg[seg > 0])
            neg_sum = np.sum(np.abs(seg[seg < 0]))
            total = pos_sum + neg_sum
            if total > 1e-10:
                inst_pressure[i] = (pos_sum - neg_sum) / total

        # Retail pressure: small volume bars
        retail_flow = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if vol_intensity[i] < 0.7:  # below-average volume = likely retail
                retail_flow[i] = bar_dir[i]

        retail_pressure = np.zeros(n, dtype=np.float64)
        for i in range(flow_win, n):
            seg = retail_flow[i-flow_win+1:i+1]
            retail_pressure[i] = np.mean(seg)

        # Flow divergence: institutions and retail in opposite directions
        flow_div = np.sign(inst_pressure) - np.sign(retail_pressure)

        # Persistence
        press_pos_consec = np.zeros(n, dtype=np.int32)
        press_neg_consec = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            press_pos_consec[i] = (press_pos_consec[i-1] + 1) if inst_pressure[i] > press_th * 0.7 else 0
            press_neg_consec[i] = (press_neg_consec[i-1] + 1) if inst_pressure[i] < -press_th * 0.7 else 0

        time_ok = (time_mins >= 570) & (time_mins <= 910)
        vix_ok = vix < vix_max

        long_entry = (
            (inst_pressure > press_th)
            & (flow_div > 0)
            & (close >= vwap)
            & (press_pos_consec >= confirm)
            & vix_ok & time_ok
        )
        short_entry = (
            (inst_pressure < -press_th)
            & (flow_div < 0)
            & (close <= vwap)
            & (press_neg_consec >= confirm)
            & vix_ok & time_ok
        )

        # Signal exit: institutional pressure reverses
        sig_exit_long = inst_pressure < 0
        sig_exit_short = inst_pressure > 0

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
            trailing_activate_pct=0.0025,
            trailing_stop_pct=0.0015,
            time_stop_bars=75,
        )
