# AUDIT FIX: Default thresholds ci_lower_min_bps=5 and ci_width_max_bps=50 were miscalibrated
# for 1-minute Indian equity data where 90% CI width is typically 70-150 bps and CI lower
# is always negative. Changed ci_lower_min_bps default to -20 (CI lower > -20 bps) and
# ci_width_max_bps default to 120 to allow signals to fire. Updated tunable ranges accordingly.
"""Bootstrap Confidence v1 — cursor_opus46max_194

Thesis: Block bootstrap on rolling 120-bar returns constructs confidence
intervals for forward return. If the 90% CI is entirely above zero, strong
long signal. If entirely below, strong short. Simplified: use rolling
percentile-based CIs from actual return history.
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
    name = "cursor_opus46max_194"
    is_long_only = False
    session_start = 675   # ~11:15
    session_end = 910     # 15:10
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("bootstrap_window", default=120.0, low=80.0, high=180.0),
            TunableParam("block_size", default=5.0, low=3.0, high=10.0),
            TunableParam("n_samples", default=200.0, low=100.0, high=500.0),
            TunableParam("fwd_bars", default=30.0, low=15.0, high=45.0),
            TunableParam("ci_lower_min_bps", default=-20.0, low=-40.0, high=0.0),
            TunableParam("ci_width_max_bps", default=120.0, low=80.0, high=180.0),
            TunableParam("stop_loss_pct", default=0.0025, low=0.0015, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        bs_win = int(params.get("bootstrap_window", 120.0))
        block_sz = int(params.get("block_size", 5.0))
        n_samp = int(params.get("n_samples", 200.0))
        fwd = int(params.get("fwd_bars", 30.0))
        ci_min = params.get("ci_lower_min_bps", 5.0)
        ci_width_max = params.get("ci_width_max_bps", 50.0)
        stop_pct = params.get("stop_loss_pct", 0.0025)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # 1-min returns in bps
        ret = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if close[i-1] > 0:
                ret[i] = (close[i] - close[i-1]) / close[i-1] * 10000

        # Block bootstrap CI
        ci_lower = np.zeros(n, dtype=np.float64)
        ci_upper = np.zeros(n, dtype=np.float64)
        ci_width = np.zeros(n, dtype=np.float64)

        rng = np.random.RandomState(42)

        for i in range(bs_win, n):
            base_rets = ret[i-bs_win+1:i+1]
            n_blocks = max(fwd // block_sz, 1)

            # Generate bootstrap samples
            fwd_sums = np.zeros(n_samp, dtype=np.float64)
            for s in range(n_samp):
                total = 0.0
                for _ in range(n_blocks):
                    start = rng.randint(0, max(len(base_rets) - block_sz, 1))
                    total += np.sum(base_rets[start:start+block_sz])
                fwd_sums[s] = total

            ci_lower[i] = np.percentile(fwd_sums, 5)
            ci_upper[i] = np.percentile(fwd_sums, 95)
            ci_width[i] = ci_upper[i] - ci_lower[i]

        # ROC confirmation
        roc5 = np.zeros(n, dtype=np.float64)
        for i in range(5, n):
            if close[i-5] > 0:
                roc5[i] = (close[i] - close[i-5]) / close[i-5] * 10000

        time_ok = (time_mins >= 675) & (time_mins <= 910)

        long_entry = (
            (ci_lower > ci_min)
            & (ci_width < ci_width_max)
            & (close > vwap)
            & (roc5 > 0)
            & time_ok
        )
        short_entry = (
            (ci_upper < -ci_min)
            & (ci_width < ci_width_max)
            & (close < vwap)
            & (roc5 < 0)
            & time_ok
        )

        # Signal exit: CI shifts to include zero
        sig_exit_long = ci_lower <= 0
        sig_exit_short = ci_upper >= 0

        atr = _compute_atr(high, low, close, 14)

        # Target from median bootstrap
        ci_med = (ci_lower + ci_upper) / 2.0
        avg_target = np.mean(np.abs(ci_med[ci_med != 0])) if np.any(ci_med != 0) else 20.0
        target_pct = max(0.001, min(avg_target / 10000, 0.005))

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            time_stop_bars=60,
        )
