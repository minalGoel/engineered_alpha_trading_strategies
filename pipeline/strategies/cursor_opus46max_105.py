"""Price Impact Reversion v1 — cursor_opus46max_105

Thesis: Large trades cause temporary price impact that partially reverts
within 2-5 minutes.  Detect large impact bars via volume and price change,
then fade the move expecting mean reversion.
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
    name = "cursor_opus46max_105"
    is_long_only = False
    session_start = 580   # 09:40
    session_end = 915     # 15:15
    max_trades_per_day = 10
    assumptions = ["Large trade detection proxied via volume spikes and bar price impact"]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_spike_mult", default=2.0, low=1.5, high=4.0),
            TunableParam("impact_bps_thresh", default=15.0, low=8.0, high=25.0),
            TunableParam("vwap_dev_thresh", default=20.0, low=10.0, high=35.0),
            TunableParam("vix_max", default=25.0, low=18.0, high=30.0),
            TunableParam("stop_loss_pct", default=0.0015, low=0.0008, high=0.003),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vol_spike = params.get("vol_spike_mult", 2.0)
        impact_thresh = params.get("impact_bps_thresh", 15.0)
        vwap_dev_thresh = params.get("vwap_dev_thresh", 20.0)
        vix_max = params.get("vix_max", 25.0)
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

        # Volume SMA(20)
        avg_vol = np.ones(n, dtype=np.float64)
        for i in range(19, n):
            avg_vol[i] = np.mean(volume[i-19:i+1])
        avg_vol = np.clip(avg_vol, 1.0, None)

        # Large trade flag: volume > vol_spike * avg_vol
        large_flag = volume > vol_spike * avg_vol

        # Impact in bps: (close - close[1]) / close[1] * 10000
        impact_bps = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if close[i-1] > 1e-10:
                impact_bps[i] = (close[i] - close[i-1]) / close[i-1] * 10000.0

        # VWAP deviation in bps
        safe_vwap = np.where(vwap > 1e-10, vwap, 1e-10)
        vwap_dev = (close - vwap) / safe_vwap * 10000.0

        vix_ok = vix < vix_max

        # Long: fade large sell impact (impact < -thresh, pushed below VWAP)
        # Confirmation: close > low of impact bar
        long_entry = (large_flag & (impact_bps < -impact_thresh) &
                      (vwap_dev < -vwap_dev_thresh) & (close > low) & vix_ok)

        # Short: fade large buy impact
        short_entry = (large_flag & (impact_bps > impact_thresh) &
                       (vwap_dev > vwap_dev_thresh) & (close < high) & vix_ok)

        # Signal exit: price returns to VWAP
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
            trailing_stop_pct=0.0005,
            trailing_activate_pct=0.001,
            time_stop_bars=5,
        )
