"""Wavelet Decomposition v1 — cursor_opus46max_182

Thesis: Wavelet decomposition separates price series into multiple time scales.
By zeroing the level-1 detail coefficients (noise) and reconstructing, we get
a denoised price. Trade when raw price crosses above/below denoised with
positive/negative denoised slope and sufficient energy at tradeable frequency.
Simplified implementation uses rolling linear denoising as wavelet proxy.
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


def _moving_average_denoise(data, short_period, long_period):
    """Denoise by averaging a short and long EMA (approximates wavelet denoising)."""
    n = len(data)
    short_ema = np.zeros(n, dtype=np.float64)
    long_ema = np.zeros(n, dtype=np.float64)
    short_ema[0] = data[0]
    long_ema[0] = data[0]
    ks = 2.0 / (short_period + 1.0)
    kl = 2.0 / (long_period + 1.0)
    for i in range(1, n):
        short_ema[i] = data[i] * ks + short_ema[i-1] * (1.0 - ks)
        long_ema[i] = data[i] * kl + long_ema[i-1] * (1.0 - kl)
    # Denoised: weighted average emphasizing medium-term
    denoised = 0.6 * short_ema + 0.4 * long_ema
    return denoised


class Strategy(BaseStrategy):
    name = "cursor_opus46max_182"
    is_long_only = False
    session_start = 620   # ~10:20
    session_end = 910     # 15:10
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("denoise_short", default=8.0, low=4.0, high=12.0),
            TunableParam("denoise_long", default=16.0, low=10.0, high=24.0),
            TunableParam("slope_period", default=5.0, low=3.0, high=8.0),
            TunableParam("slope_thresh_bps", default=3.0, low=1.0, high=6.0),
            TunableParam("energy_thresh", default=0.20, low=0.10, high=0.35),
            TunableParam("confirm_bars", default=2.0, low=1.0, high=4.0),
            TunableParam("stop_loss_pct", default=0.0020, low=0.0012, high=0.003),
            TunableParam("target_pct", default=0.0035, low=0.002, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        dn_short = int(params.get("denoise_short", 8.0))
        dn_long = int(params.get("denoise_long", 16.0))
        slope_p = int(params.get("slope_period", 5.0))
        slope_th = params.get("slope_thresh_bps", 3.0)
        energy_th = params.get("energy_thresh", 0.20)
        confirm = int(params.get("confirm_bars", 2.0))
        stop_pct = params.get("stop_loss_pct", 0.0020)
        target_pct = params.get("target_pct", 0.0035)

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

        # Denoised price (wavelet proxy)
        denoised = _moving_average_denoise(close, dn_short, dn_long)

        # Denoised slope (bps per bar)
        dn_slope = np.zeros(n, dtype=np.float64)
        for i in range(slope_p, n):
            if denoised[i-slope_p] > 0:
                dn_slope[i] = (denoised[i] - denoised[i-slope_p]) / denoised[i-slope_p] * 10000 / slope_p

        # Energy proxy: ratio of variance of (close-denoised) to variance of close over 64-bar window
        energy = np.zeros(n, dtype=np.float64)
        noise = close - denoised
        for i in range(63, n):
            var_noise = np.var(noise[i-63:i+1])
            var_close = np.var(close[i-63:i+1])
            if var_close > 1e-15:
                energy[i] = 1.0 - var_noise / var_close  # signal energy ratio

        # Crossover detection
        above = close > denoised
        below = close < denoised

        # Consecutive bars above/below denoised
        consec_above = np.zeros(n, dtype=np.int32)
        consec_below = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            consec_above[i] = (consec_above[i-1] + 1) if above[i] else 0
            consec_below[i] = (consec_below[i-1] + 1) if below[i] else 0

        # Cross up/down
        cross_up = np.zeros(n, dtype=np.bool_)
        cross_down = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if not above[i-1] and above[i]:
                cross_up[i] = True
            if not below[i-1] and below[i]:
                cross_down[i] = True

        # Entry with confirmation (cross happened in last `confirm+1` bars and stayed)
        recent_cross_up = np.zeros(n, dtype=np.bool_)
        recent_cross_down = np.zeros(n, dtype=np.bool_)
        for i in range(confirm, n):
            recent_cross_up[i] = any(cross_up[i-j] for j in range(confirm+1)) and consec_above[i] >= confirm
            recent_cross_down[i] = any(cross_down[i-j] for j in range(confirm+1)) and consec_below[i] >= confirm

        time_ok = (time_mins >= 620) & (time_mins <= 910)

        long_entry = (
            recent_cross_up
            & (dn_slope > slope_th)
            & (energy > energy_th)
            & (close > vwap)
            & time_ok
        )
        short_entry = (
            recent_cross_down
            & (dn_slope < -slope_th)
            & (energy > energy_th)
            & (close < vwap)
            & time_ok
        )

        # Signal exit: denoised slope crosses zero against position
        sig_exit_long = dn_slope < 0
        sig_exit_short = dn_slope > 0

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
            trailing_activate_pct=0.0020,
            trailing_stop_pct=0.0012,
            time_stop_bars=45,
        )
