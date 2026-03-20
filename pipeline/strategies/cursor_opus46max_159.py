"""Mean-Variance Optimal v1 — cursor_opus46max_159

Thesis: Compute rolling instantaneous Sharpe (EMA of returns / sqrt(variance))
and enter when it exceeds threshold (favorable risk/reward). High Sharpe =
positive momentum with compressed volatility (coiled spring).
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_ema(arr, period):
    n = len(arr)
    ema = np.zeros(n, dtype=np.float64)
    if n == 0:
        return ema
    ema[0] = arr[0]
    k = 2.0 / (period + 1)
    for i in range(1, n):
        ema[i] = arr[i] * k + ema[i - 1] * (1.0 - k)
    return ema


class Strategy(BaseStrategy):
    name = "cursor_opus46max_159"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 910     # 15:10
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("sharpe_entry", default=2.0, low=1.0, high=3.5),
            TunableParam("sharpe_confirm", default=1.5, low=0.8, high=2.5),
            TunableParam("target_pct", default=0.0045, low=0.003, high=0.007),
            TunableParam("stop_loss_pct", default=0.0025, low=0.0015, high=0.004),
            TunableParam("trailing_stop_pct", default=0.0015, low=0.001, high=0.003),
            TunableParam("trailing_activate_pct", default=0.0025, low=0.0015, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        sharpe_entry = params.get("sharpe_entry", 2.0)
        sharpe_conf = params.get("sharpe_confirm", 1.5)
        target_pct = params.get("target_pct", 0.0045)
        stop_pct = params.get("stop_loss_pct", 0.0025)
        trail_pct = params.get("trailing_stop_pct", 0.0015)
        trail_act = params.get("trailing_activate_pct", 0.0025)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # 1-bar returns
        returns = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if close[i - 1] > 0:
                returns[i] = close[i] / close[i - 1] - 1.0

        # Rolling return (EMA smoothed, period=10)
        rolling_ret = _compute_ema(returns, 10)

        # Rolling variance (30-bar window)
        rolling_var = np.zeros(n, dtype=np.float64)
        for i in range(29, n):
            window = returns[i - 29:i + 1]
            rolling_var[i] = np.var(window)

        # Instantaneous Sharpe
        instant_sharpe = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if rolling_var[i] > 1e-20:
                instant_sharpe[i] = rolling_ret[i] / np.sqrt(rolling_var[i])

        # Volume filter
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > avg_vol

        time_ok = (time_mins >= 570) & (time_mins <= 910)
        vix_ok = vix < 25.0

        # Entry: Sharpe exceeds threshold, confirmed next bar
        long_raw = (instant_sharpe > sharpe_entry) & (rolling_ret > 0) & (close > vwap)
        short_raw = (instant_sharpe < -sharpe_entry) & (rolling_ret < 0) & (close < vwap)

        # Confirmation: Sharpe remains above confirm threshold on next bar
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if long_raw[i - 1] and instant_sharpe[i] > sharpe_conf:
                long_entry[i] = True
            if short_raw[i - 1] and instant_sharpe[i] < -sharpe_conf:
                short_entry[i] = True

        long_entry = long_entry & vol_ok & time_ok & vix_ok
        short_entry = short_entry & vol_ok & time_ok & vix_ok

        # Signal exit: Sharpe crosses zero against position
        sig_exit_long = instant_sharpe < 0
        sig_exit_short = instant_sharpe > 0

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            trailing_stop_pct=trail_pct,
            trailing_activate_pct=trail_act,
            time_stop_bars=60,
        )
