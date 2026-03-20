"""Global Cue Reaction v1 — cursor_opus46max_124

Thesis: The opening gap in NIFTY relative to overnight global cues provides
a tradeable signal.  When NIFTY opens at a premium/discount to its expected
level (based on index overnight move), the excess fades within 30 minutes.
Proxy SGX NIFTY using index overnight gap analysis.
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
    name = "cursor_opus46max_124"
    is_long_only = False
    session_start = 555   # 09:15
    session_end = 630     # 10:30 (morning only strategy)
    max_trades_per_day = 2
    assumptions = ["SGX NIFTY implied open proxied via index gap vs stock gap divergence"]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("excess_gap_bps", default=10.0, low=5.0, high=20.0),
            TunableParam("vix_max", default=25.0, low=18.0, high=30.0),
            TunableParam("stop_loss_pct", default=0.0015, low=0.0008, high=0.003),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        excess_thresh = params.get("excess_gap_bps", 10.0)
        vix_max = params.get("vix_max", 25.0)
        stop_pct = params.get("stop_loss_pct", 0.0015)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(df["index_close"].to_numpy().astype(np.float64), nan=0.0)
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

        # Gap computation: stock gap vs index gap
        prev_close_stock = np.zeros(n, dtype=np.float64)
        prev_close_index = np.zeros(n, dtype=np.float64)
        day_open_stock = np.zeros(n, dtype=np.float64)
        day_open_index = np.zeros(n, dtype=np.float64)
        pcs = close[0]
        pci = index_close[0]
        for i in range(n):
            if i == 0 or day_id[i] != day_id[i-1]:
                prev_close_stock[i] = pcs
                prev_close_index[i] = pci
                day_open_stock[i] = opn[i]
                day_open_index[i] = index_close[i]
                if i > 0:
                    pcs = close[i-1]
                    pci = index_close[i-1]
            else:
                prev_close_stock[i] = prev_close_stock[i-1]
                prev_close_index[i] = prev_close_index[i-1]
                day_open_stock[i] = day_open_stock[i-1]
                day_open_index[i] = day_open_index[i-1]

        # Stock gap in bps
        safe_pcs = np.where(prev_close_stock > 1e-10, prev_close_stock, 1e-10)
        stock_gap = (day_open_stock - prev_close_stock) / safe_pcs * 10000.0

        # Index gap in bps
        safe_pci = np.where(prev_close_index > 1e-10, prev_close_index, 1e-10)
        index_gap = (day_open_index - prev_close_index) / safe_pci * 10000.0

        # Excess gap: stock opened more than index suggests
        excess_gap = stock_gap - index_gap

        vix_ok = vix < vix_max
        early = (time_mins >= 555) & (time_mins <= 585)

        # Long: stock opened at discount vs index (excess_gap < -thresh)
        # Confirmation: first uptick
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if (early[i] and excess_gap[i] < -excess_thresh and
                    close[i] > opn[i] and vix_ok[i]):
                long_entry[i] = True
            if (early[i] and excess_gap[i] > excess_thresh and
                    close[i] < opn[i] and vix_ok[i]):
                short_entry[i] = True

        # Signal exit: excess gap converges (price aligns with index-implied level)
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        # Current deviation from VWAP as proxy for convergence
        safe_vwap = np.where(vwap > 1e-10, vwap, 1e-10)
        vwap_dev = (close - vwap) / safe_vwap * 10000.0
        for i in range(1, n):
            if abs(vwap_dev[i]) < 3:  # close to VWAP = converged
                sig_exit_long[i] = True
                sig_exit_short[i] = True

        atr = _compute_atr(high, low, close, 14)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=vwap.copy(),
            use_target_indicator=True,
            stop_loss_pct=stop_pct,
            trailing_stop_pct=0.0005,
            trailing_activate_pct=0.001,
            time_stop_bars=30,
        )
