"""Spectral Analysis v1 — cursor_opus46max_183

Thesis: Intraday price has periodic components driven by institutional cycles.
Using FFT on detrended 128-bar close, identify dominant cycle period. Enter at
cycle trough (long) or peak (short) when spectral ratio > 3.0.
Simplified: use rolling autocorrelation to detect dominant cycle.
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
    name = "cursor_opus46max_183"
    is_long_only = False
    session_start = 680   # ~11:20
    session_end = 910     # 15:10
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("fft_window", default=128.0, low=64.0, high=192.0),
            TunableParam("spectral_ratio_thresh", default=3.0, low=2.0, high=5.0),
            TunableParam("min_dom_period", default=8.0, low=4.0, high=16.0),
            TunableParam("max_dom_period", default=60.0, low=30.0, high=90.0),
            TunableParam("stop_loss_pct", default=0.0025, low=0.0015, high=0.004),
            TunableParam("target_pct", default=0.0035, low=0.002, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        fft_win = int(params.get("fft_window", 128.0))
        spec_ratio_th = params.get("spectral_ratio_thresh", 3.0)
        min_period = int(params.get("min_dom_period", 8.0))
        max_period = int(params.get("max_dom_period", 60.0))
        stop_pct = params.get("stop_loss_pct", 0.0025)
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

        # FFT-based cycle detection
        dom_period = np.zeros(n, dtype=np.float64)
        spectral_ratio = np.zeros(n, dtype=np.float64)
        cycle_phase = np.zeros(n, dtype=np.float64)

        for i in range(fft_win, n):
            seg = close[i-fft_win+1:i+1]
            # Detrend (linear)
            x_idx = np.arange(fft_win, dtype=np.float64)
            slope = (seg[-1] - seg[0]) / (fft_win - 1)
            detrended = seg - (seg[0] + slope * x_idx)

            # FFT
            fft_vals = np.fft.rfft(detrended)
            power = np.abs(fft_vals[1:]) ** 2  # skip DC component

            if len(power) > 0 and np.mean(power) > 1e-15:
                # Convert index to period
                freqs = np.arange(1, len(power) + 1)
                periods = fft_win / freqs

                # Filter to tradeable range
                mask = (periods >= min_period) & (periods <= max_period)
                if np.any(mask):
                    filtered_power = power.copy()
                    filtered_power[~mask] = 0

                    peak_idx = np.argmax(filtered_power)
                    if periods[peak_idx] >= min_period and periods[peak_idx] <= max_period:
                        dom_period[i] = periods[peak_idx]
                        spectral_ratio[i] = power[peak_idx] / np.mean(power)

                        # Phase: angle of the dominant frequency component
                        cycle_phase[i] = np.angle(fft_vals[peak_idx + 1])

        # 2-bar ROC for confirmation
        roc2 = np.zeros(n, dtype=np.float64)
        for i in range(2, n):
            if close[i-2] > 0:
                roc2[i] = (close[i] - close[i-2]) / close[i-2] * 10000

        # Near trough: phase near 0 or 2*pi (within pi/6)
        near_trough = (np.abs(cycle_phase) < np.pi / 6) | (np.abs(cycle_phase) > 11 * np.pi / 6)
        # Near peak: phase near pi (within pi/6)
        near_peak = np.abs(np.abs(cycle_phase) - np.pi) < np.pi / 6

        time_ok = (time_mins >= 680) & (time_mins <= 910)
        tradeable_period = (dom_period >= min_period) & (dom_period <= max_period)

        long_entry = (
            (spectral_ratio > spec_ratio_th)
            & near_trough
            & tradeable_period
            & (close < vwap)
            & (roc2 > 0)
            & time_ok
        )
        short_entry = (
            (spectral_ratio > spec_ratio_th)
            & near_peak
            & tradeable_period
            & (close > vwap)
            & (roc2 < 0)
            & time_ok
        )

        # Signal exit: spectral ratio drops below 2.0
        sig_exit_long = spectral_ratio < 2.0
        sig_exit_short = spectral_ratio < 2.0

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
            time_stop_bars=60,
        )
