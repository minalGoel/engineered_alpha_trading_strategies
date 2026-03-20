"""Telecom Pairs (Airtel vs Jio/RIL proxy) — cursor_opus46max_139

Thesis: Airtel and Jio compete for the same subscribers. When telecom-specific
news drives Airtel away from its expected correlation with the index (Jio proxy),
it reverts. Uses VWAP confirmation.
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
    name = "cursor_opus46max_139"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 915     # 15:15
    max_trades_per_day = 3
    assumptions = [
        "Jio/RIL and Vi data not available; using stock vs index spread as proxy",
    ]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("spread_entry_bps", default=15.0, low=8.0, high=25.0),
            TunableParam("spread_exit_bps", default=5.0, low=2.0, high=10.0),
            TunableParam("spread_stop_bps", default=30.0, low=15.0, high=50.0),
            TunableParam("vix_max", default=22.0, low=15.0, high=28.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        spr_entry = params.get("spread_entry_bps", 15.0)
        spr_exit = params.get("spread_exit_bps", 5.0)
        spr_stop = params.get("spread_stop_bps", 30.0)
        vix_max = params.get("vix_max", 22.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        day_id = df["day_id"].to_numpy()
        idx_close = df["index_close"].to_numpy().astype(np.float64)
        idx_close = np.nan_to_num(idx_close, nan=0.0)

        atr = _compute_atr(high, low, close, 14)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # Daily opens
        day_open = np.zeros(n, dtype=np.float64)
        idx_day_open = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if i == 0 or day_id[i] != day_id[i-1]:
                day_open[i] = open_[i]
                idx_day_open[i] = idx_close[i] if idx_close[i] > 0 else 1.0
            else:
                day_open[i] = day_open[i-1]
                idx_day_open[i] = idx_day_open[i-1]

        safe_do = np.clip(day_open, 1e-10, None)
        safe_ido = np.clip(idx_day_open, 1e-10, None)
        # Weight index return by 0.3 (Jio ~30% of RIL)
        stock_ret = (close - day_open) / safe_do * 10000.0
        idx_ret = (idx_close - idx_day_open) / safe_ido * 10000.0
        spread = stock_ret - idx_ret * 0.3

        vix_ok = vix < vix_max
        time_ok = (time_mins >= 585) & (time_mins <= 900)

        # Volume normal
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > 0.8 * avg_vol

        # VWAP confirmation
        above_vwap = close > vwap * 0.999
        below_vwap = close < vwap * 1.001

        # Long: Airtel underperforming, turning up near VWAP
        long_entry = (
            (spread < -spr_entry) &
            above_vwap &
            vol_ok &
            vix_ok &
            time_ok
        )
        # Short: Airtel outperforming, turning down
        short_entry = (
            (spread > spr_entry) &
            below_vwap &
            vol_ok &
            vix_ok &
            time_ok
        )

        sig_exit_long = (spread > -spr_exit) | (spread < -spr_stop)
        sig_exit_short = (spread < spr_exit) | (spread > spr_stop)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=spr_stop / 10000.0,
            trailing_stop_pct=0.0005,
            trailing_activate_pct=0.001,
            time_stop_bars=45,
        )
