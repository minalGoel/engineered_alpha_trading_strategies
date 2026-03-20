"""Energy Upstream-Downstream Pairs — cursor_opus46max_140

Thesis: Upstream oil producers benefit from high crude while downstream OMCs
are hurt. When crude moves sharply, the upstream-downstream spread should widen.
Proxied by stock vs index divergence in the direction of the overall move.
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
    name = "cursor_opus46max_140"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 915     # 15:15
    max_trades_per_day = 3
    assumptions = [
        "Brent crude and upstream/downstream group data not available",
        "Using stock vs index deviation as proxy for energy spread",
    ]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("dev_entry_bps", default=15.0, low=8.0, high=30.0),
            TunableParam("idx_move_thresh", default=50.0, low=25.0, high=80.0),
            TunableParam("target_bps", default=25.0, low=12.0, high=40.0),
            TunableParam("stop_bps", default=15.0, low=8.0, high=25.0),
            TunableParam("vix_max", default=25.0, low=18.0, high=30.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        dev_entry = params.get("dev_entry_bps", 15.0)
        idx_thresh = params.get("idx_move_thresh", 50.0)
        target_bps = params.get("target_bps", 25.0)
        stop_bps = params.get("stop_bps", 15.0)
        vix_max = params.get("vix_max", 25.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        day_id = df["day_id"].to_numpy()
        idx_close = df["index_close"].to_numpy().astype(np.float64)
        idx_close = np.nan_to_num(idx_close, nan=0.0)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        atr = _compute_atr(high, low, close, 14)

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
        stock_ret = (close - day_open) / safe_do * 10000.0
        idx_ret = (idx_close - idx_day_open) / safe_ido * 10000.0
        spread = stock_ret - idx_ret

        # Oil-driven regime: large index move (proxy for crude-driven day)
        oil_driven = np.abs(idx_ret) > idx_thresh

        vix_ok = vix < vix_max
        time_ok = (time_mins >= 585) & (time_mins <= 900)

        # Spread widening confirmation
        spread_widening_up = np.zeros(n, dtype=np.bool_)
        spread_widening_dn = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if spread[i] > spread[i-1] and spread[i] > dev_entry:
                spread_widening_up[i] = True
            if spread[i] < spread[i-1] and spread[i] < -dev_entry:
                spread_widening_dn[i] = True

        # Long upstream: when oil is up (idx_ret > 0), stock should outperform
        long_entry = (
            oil_driven &
            (idx_ret > 0) &
            (spread < dev_entry) &
            (close > vwap) &
            vix_ok &
            time_ok
        )

        # Short: reverse
        short_entry = (
            oil_driven &
            (idx_ret < 0) &
            (spread > -dev_entry) &
            (close < vwap) &
            vix_ok &
            time_ok
        )

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_bps / 10000.0,
            target_pct=target_bps / 10000.0,
            trailing_stop_pct=0.0008,
            trailing_activate_pct=0.0015,
            time_stop_bars=60,
        )
