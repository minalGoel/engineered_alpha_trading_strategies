"""Bayesian Update v1 — cursor_opus46max_161

Thesis: Build a posterior probability of trending up/down using Bayesian
updating. Prior from index direction; likelihoods from price, volume,
VWAP. Enter when posterior > 0.70 sustained for 2 bars.
Simplified: use score-based proxy for posterior.
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
    name = "cursor_opus46max_161"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 910     # 15:10
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("posterior_thresh", default=0.70, low=0.60, high=0.85),
            TunableParam("target_pct", default=0.0035, low=0.002, high=0.005),
            TunableParam("stop_loss_pct", default=0.002, low=0.001, high=0.003),
            TunableParam("trailing_stop_pct", default=0.0012, low=0.0008, high=0.002),
            TunableParam("trailing_activate_pct", default=0.002, low=0.001, high=0.003),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        post_thresh = params.get("posterior_thresh", 0.70)
        target_pct = params.get("target_pct", 0.0035)
        stop_pct = params.get("stop_loss_pct", 0.002)
        trail_pct = params.get("trailing_stop_pct", 0.0012)
        trail_act = params.get("trailing_activate_pct", 0.002)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        idx_close = df["index_close"].to_numpy().astype(np.float64)
        idx_close = np.nan_to_num(idx_close, nan=0.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # Bayesian posterior approximation using score-based approach
        # Prior: 0.5 + 0.1 * sign(NIFTY_ROC_30) + 0.05 * sign(stock_ROC_30)
        nifty_roc30 = np.zeros(n, dtype=np.float64)
        for i in range(30, n):
            if idx_close[i - 30] > 0:
                nifty_roc30[i] = idx_close[i] / idx_close[i - 30] - 1.0

        prior = 0.5 + 0.1 * np.sign(nifty_roc30)

        # Price likelihood: smoothed return direction
        returns = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if close[i - 1] > 0:
                returns[i] = close[i] / close[i - 1] - 1.0
        smooth_ret = _compute_ema(returns, 5)
        price_evidence = np.clip(smooth_ret * 5000.0, -0.15, 0.15)

        # Volume likelihood: high volume confirms direction
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ratio = volume / avg_vol
        vol_evidence = np.where(vol_ratio > 1.5,
                                0.05 * np.sign(smooth_ret), 0.0)

        # VWAP likelihood: above VWAP = bullish evidence
        safe_vwap = np.where(vwap > 0, vwap, 1.0)
        vwap_dev = (close - vwap) / safe_vwap
        vwap_evidence = np.clip(vwap_dev * 10.0, -0.1, 0.1)

        # Posterior approximation (sigmoid-bounded sum)
        raw_score = prior + price_evidence + vol_evidence + vwap_evidence
        posterior = 1.0 / (1.0 + np.exp(-10.0 * (raw_score - 0.5)))

        # Monotonically increasing/decreasing for 3 bars
        mono_up = np.zeros(n, dtype=np.bool_)
        mono_dn = np.zeros(n, dtype=np.bool_)
        for i in range(3, n):
            if posterior[i] > posterior[i - 1] > posterior[i - 2] > posterior[i - 3]:
                mono_up[i] = True
            if posterior[i] < posterior[i - 1] < posterior[i - 2] < posterior[i - 3]:
                mono_dn[i] = True

        # 2-bar sustained confirmation
        long_raw = (posterior > post_thresh) & mono_up & (close > vwap)
        short_raw = (posterior < (1.0 - post_thresh)) & mono_dn & (close < vwap)

        long_conf = np.zeros(n, dtype=np.bool_)
        short_conf = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if long_raw[i] and long_raw[i - 1]:
                long_conf[i] = True
            if short_raw[i] and short_raw[i - 1]:
                short_conf[i] = True

        time_ok = (time_mins >= 570) & (time_mins <= 910)
        vix_ok = vix < 25.0

        long_entry = long_conf & time_ok & vix_ok
        short_entry = short_conf & time_ok & vix_ok

        # Signal exit: posterior crosses 0.5
        sig_exit_long = posterior < 0.5
        sig_exit_short = posterior > 0.5

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
