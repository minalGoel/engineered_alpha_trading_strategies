"""Entropy Regime v1 — cursor_opus46max_174

Thesis: Shannon entropy of binned 1-min returns (50-bar window, 10 bins)
measures predictability. Low entropy (< 30th rolling percentile) = structured
regime, trade directionally with EMA(10)/EMA(20). High entropy = random, flat.
ATR-based stops/targets.
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


def _compute_atr(high, low, close, period):
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    if n < period:
        return atr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _shannon_entropy(returns_window, n_bins=10):
    """Compute Shannon entropy of binned returns."""
    n = len(returns_window)
    if n < n_bins:
        return 1.0  # default high entropy
    rng = np.max(returns_window) - np.min(returns_window)
    if rng < 1e-20:
        return 0.0  # all returns identical = zero entropy
    # Histogram with fixed number of bins
    counts, _ = np.histogram(returns_window, bins=n_bins)
    # Normalize to probabilities
    probs = counts / n
    # Shannon entropy: -sum(p * log(p)) for p > 0
    entropy = 0.0
    for p in probs:
        if p > 1e-20:
            entropy -= p * np.log2(p)
    return entropy


class Strategy(BaseStrategy):
    name = "cursor_opus46max_174"
    is_long_only = False
    session_start = 605   # 10:05
    session_end = 910     # 15:10
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("entropy_window", default=50.0, low=30.0, high=70.0),
            TunableParam("low_entropy_pctile", default=30.0, low=20.0, high=40.0),
            TunableParam("high_entropy_exit_pctile", default=70.0, low=60.0, high=80.0),
            TunableParam("atr_target_mult", default=1.2, low=0.8, high=1.6),
            TunableParam("atr_stop_mult", default=0.8, low=0.5, high=1.1),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        ent_window = int(params.get("entropy_window", 50.0))
        low_pctile = params.get("low_entropy_pctile", 30.0)
        high_pctile = params.get("high_entropy_exit_pctile", 70.0)
        atr_t_mult = params.get("atr_target_mult", 1.2)
        atr_s_mult = params.get("atr_stop_mult", 0.8)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high_arr = df["high"].to_numpy().astype(np.float64)
        low_arr = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        atr = _compute_atr(high_arr, low_arr, close, 14)

        # 1-bar returns
        returns = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if close[i - 1] > 0:
                returns[i] = close[i] / close[i - 1] - 1.0

        # Rolling Shannon entropy
        entropy = np.full(n, 1.0, dtype=np.float64)
        for i in range(ent_window, n):
            window = returns[i - ent_window + 1:i + 1]
            entropy[i] = _shannon_entropy(window, n_bins=10)

        # Rolling percentile of entropy (120-bar window as proxy for recent history)
        ent_pctile = np.full(n, 50.0, dtype=np.float64)
        pctile_window = 120
        for i in range(ent_window + pctile_window, n):
            hist = entropy[i - pctile_window + 1:i + 1]
            rank = np.sum(hist <= entropy[i]) / len(hist) * 100.0
            ent_pctile[i] = rank

        is_low_entropy = ent_pctile < low_pctile
        is_high_entropy = ent_pctile > high_pctile

        # Dominant direction: sign of mean returns in window
        dom_dir = np.zeros(n, dtype=np.float64)
        for i in range(ent_window, n):
            dom_dir[i] = np.mean(returns[i - ent_window + 1:i + 1])

        # EMA(10)/EMA(20) for confirmation
        ema10 = _compute_ema(close, 10)
        ema20 = _compute_ema(close, 20)

        # Low entropy stability: 5 consecutive bars
        stable = np.zeros(n, dtype=np.bool_)
        for i in range(5, n):
            s = True
            for j in range(5):
                if not is_low_entropy[i - j]:
                    s = False
                    break
            stable[i] = s

        # Volume filter
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > avg_vol

        time_ok = (time_mins >= 605) & (time_mins <= 910)

        # Long: low entropy + positive dominant direction + EMA up + above VWAP
        long_entry = is_low_entropy & (dom_dir > 0) & (ema10 > ema20) & \
                     (close > vwap) & stable & vol_ok & time_ok

        # Short: low entropy + negative dominant direction + EMA down + below VWAP
        short_entry = is_low_entropy & (dom_dir < 0) & (ema10 < ema20) & \
                      (close < vwap) & stable & vol_ok & time_ok

        # Signal exit: entropy rises to high percentile
        sig_exit_long = is_high_entropy
        sig_exit_short = is_high_entropy

        # ATR-based exit params (convert to pct using median close)
        median_close = np.median(close[close > 0]) if np.any(close > 0) else 1.0
        median_atr = np.median(atr[atr > 0]) if np.any(atr > 0) else 0.01
        target_pct = (atr_t_mult * median_atr) / median_close
        stop_pct = (atr_s_mult * median_atr) / median_close
        target_pct = np.clip(target_pct, 0.002, 0.008)
        stop_pct = np.clip(stop_pct, 0.001, 0.005)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=float(stop_pct),
            target_pct=float(target_pct),
            trailing_stop_pct=float(stop_pct * 0.5),
            trailing_activate_pct=float(target_pct * 0.6),
            time_stop_bars=60,
        )
