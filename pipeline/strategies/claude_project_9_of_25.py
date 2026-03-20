"""RSI Divergence + VWAP v1 — claude_project_9_of_25

Thesis: Bullish divergence (new price low but RSI higher) near VWAP
support signals a reversal. Target VWAP touch from below.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_rsi(close, period):
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    if n < period + 1:
        return rsi
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.zeros(n, dtype=np.float64)
    avg_loss = np.zeros(n, dtype=np.float64)
    avg_gain[period] = np.mean(gain[1:period + 1])
    avg_loss[period] = np.mean(loss[1:period + 1])
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain[i] / avg_loss[i])
    return rsi


class Strategy(BaseStrategy):
    name = "rsi_divergence_vwap_v1"
    is_long_only = False
    session_start = 585   # 09:45
    session_end = 900     # 15:00
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rsi_thresh", default=40.0, low=25.0, high=50.0),
            TunableParam("rsi_short_thresh", default=60.0, low=50.0, high=75.0),
            TunableParam("vwap_band_pct", default=0.01, low=0.003, high=0.02),
            TunableParam("divergence_lookback", default=20.0, low=10.0, high=40.0),
            TunableParam("stop_loss_pct", default=0.0035, low=0.002, high=0.006),
            TunableParam("breakeven_pct", default=0.002, low=0.001, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        rsi_thresh = params.get("rsi_thresh", 40.0)
        rsi_short = params.get("rsi_short_thresh", 60.0)
        vwap_band = params.get("vwap_band_pct", 0.01)
        div_lb = int(params.get("divergence_lookback", 20.0))
        stop_pct = params.get("stop_loss_pct", 0.0035)
        be_pct = params.get("breakeven_pct", 0.002)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        rsi = _compute_rsi(close, 14)

        # ── Bullish divergence: price making lower low but RSI not ──
        bull_div = np.zeros(n, dtype=np.bool_)
        bear_div = np.zeros(n, dtype=np.bool_)
        for i in range(div_lb, n):
            past_close_min = np.min(close[i - div_lb:i])
            past_rsi_at_min = rsi[i - div_lb + np.argmin(close[i - div_lb:i])]
            # Bullish: current close < past min (lower low) but RSI > past RSI at min
            if close[i] < past_close_min and rsi[i] > past_rsi_at_min:
                bull_div[i] = True
            # Bearish: current close > past max but RSI < past RSI at max
            past_close_max = np.max(close[i - div_lb:i])
            past_rsi_at_max = rsi[i - div_lb + np.argmax(close[i - div_lb:i])]
            if close[i] > past_close_max and rsi[i] < past_rsi_at_max:
                bear_div[i] = True

        # ── Near VWAP filter ──
        # Long: close between vwap * (1 - band) and vwap (below VWAP)
        near_vwap_below = (close >= vwap * (1 - vwap_band)) & (close <= vwap)
        near_vwap_above = (close <= vwap * (1 + vwap_band)) & (close >= vwap)

        # ── Session time filter ──
        # AUDIT FIX: missing session time filter caused entries outside session window
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        time_ok = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # ── Entry ──
        long_entry = bull_div & (rsi < rsi_thresh) & near_vwap_below & time_ok
        short_entry = bear_div & (rsi > rsi_short) & near_vwap_above & time_ok

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=vwap.copy(),
            stop_loss_pct=stop_pct,
            use_target_indicator=True,
            breakeven_pct=be_pct,
            time_stop_bars=60,
        )
