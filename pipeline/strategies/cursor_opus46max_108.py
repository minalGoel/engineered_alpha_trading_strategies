"""Liquidity Provision v1 — cursor_opus46max_108

Thesis: By placing limit orders around running VWAP, provide liquidity to
institutional VWAP algorithms.  Enter when price touches VWAP bands, exit
on VWAP mean reversion.  Low VIX filter critical.
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
    name = "cursor_opus46max_108"
    is_long_only = False
    session_start = 575   # 09:35
    session_end = 915     # 15:15
    max_trades_per_day = 40
    assumptions = ["Limit order fills approximated by close touching VWAP band levels"]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("band_width", default=0.5, low=0.3, high=1.0),
            TunableParam("vwap_dev_thresh", default=5.0, low=3.0, high=10.0),
            TunableParam("vix_max", default=18.0, low=12.0, high=22.0),
            TunableParam("stop_loss_pct", default=0.0015, low=0.0008, high=0.003),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        band_w = params.get("band_width", 0.5)
        vwap_dev_thresh = params.get("vwap_dev_thresh", 5.0)
        vix_max = params.get("vix_max", 18.0)
        stop_pct = params.get("stop_loss_pct", 0.0015)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(df["volume"].to_numpy().astype(np.float64), nan=1.0)
        volume = np.clip(volume, 1.0, None)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = np.nan_to_num(df_v["_vwap"].to_numpy().astype(np.float64), nan=close[0] if n > 0 else 0.0)

        # VWAP bands: VWAP +/- band_w * stdev(close - vwap, 30)
        dev = close - vwap
        std_period = 30
        vwap_band_upper = np.full(n, 1e10, dtype=np.float64)
        vwap_band_lower = np.full(n, -1e10, dtype=np.float64)
        for i in range(std_period - 1, n):
            s = np.std(dev[i - std_period + 1: i + 1])
            vwap_band_upper[i] = vwap[i] + band_w * s
            vwap_band_lower[i] = vwap[i] - band_w * s

        # VWAP deviation in bps
        safe_vwap = np.where(vwap > 1e-10, vwap, 1e-10)
        vwap_dev = (close - vwap) / safe_vwap * 10000.0

        # Volume participation filter
        avg_vol = np.ones(n, dtype=np.float64)
        for i in range(19, n):
            avg_vol[i] = np.mean(volume[i-19:i+1])
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > 0.8 * avg_vol

        vix_ok = vix < vix_max

        long_entry = ((close <= vwap_band_lower) & (vwap_dev < -vwap_dev_thresh) &
                      vol_ok & vix_ok)
        short_entry = ((close >= vwap_band_upper) & (vwap_dev > vwap_dev_thresh) &
                       vol_ok & vix_ok)

        # Signal exit: price touches VWAP
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if close[i] >= vwap[i] and close[i-1] < vwap[i-1]:
                sig_exit_long[i] = True
            if close[i] <= vwap[i] and close[i-1] > vwap[i-1]:
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
            time_stop_bars=5,
        )
