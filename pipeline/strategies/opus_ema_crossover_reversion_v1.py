"""EMA Crossover Reversion — Opus_5

Thesis: Fade EMA(9)/EMA(21) crossovers in range-bound (low ATR) markets.
When ATR is below 1.5x weekly median, crossovers are more likely to be
noise than real trends. Wait 2 bars after cross, enter if price recovers.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_ema(close, period):
    """EMA using standard multiplier."""
    n = len(close)
    ema = np.zeros(n, dtype=np.float64)
    if n == 0:
        return ema
    ema[0] = close[0]
    mult = 2.0 / (period + 1)
    for i in range(1, n):
        ema[i] = close[i] * mult + ema[i - 1] * (1.0 - mult)
    return ema


def _compute_atr(high, low, close, period):
    """Wilder's ATR."""
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    if n < period + 1:
        return atr
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr[period] = np.mean(tr[1:period + 1])
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


class Strategy(BaseStrategy):
    name = "opus_ema_crossover_reversion_v1"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 870     # 14:30
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("atr_mult_thresh", default=1.5, low=1.0, high=2.5),
            TunableParam("target_pct", default=0.003, low=0.002, high=0.006),
            TunableParam("stop_loss_pct", default=0.0035, low=0.002, high=0.006),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        atr_mult_thresh = params.get("atr_mult_thresh", 1.5)
        target_pct = params.get("target_pct", 0.003)
        stop_pct = params.get("stop_loss_pct", 0.0035)

        n = len(df)

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── EMAs ──
        ema9 = _compute_ema(close, 9)
        ema21 = _compute_ema(close, 21)

        # ── ATR(60) ──
        atr60 = _compute_atr(high, low, close, 60)

        # ── Weekly median ATR: rolling median over 375 bars (~1 day of 375 1-min bars) ──
        # Use 375-bar rolling median as proxy for weekly median
        atr60_series = pl.Series("atr60", atr60)
        atr_median = atr60_series.rolling_median(375).to_numpy().astype(np.float64)
        atr_median = np.nan_to_num(atr_median, nan=1e10)
        atr_median = np.clip(atr_median, 1e-10, None)

        low_atr = atr60 < atr_mult_thresh * atr_median

        # ── Detect crossovers ──
        # Bearish cross: ema9 crosses below ema21
        ema9_prev = np.roll(ema9, 1)
        ema9_prev[0] = ema9[0]
        ema21_prev = np.roll(ema21, 1)
        ema21_prev[0] = ema21[0]

        bearish_cross = (ema9_prev >= ema21_prev) & (ema9 < ema21)
        bullish_cross = (ema9_prev <= ema21_prev) & (ema9 > ema21)

        # ── Wait 2 bars after cross, then check recovery ──
        bearish_cross_2ago = np.roll(bearish_cross, 2)
        bearish_cross_2ago[:2] = False
        bullish_cross_2ago = np.roll(bullish_cross, 2)
        bullish_cross_2ago[:2] = False

        # ── Filters ──
        time_ok = (time_mins >= 570) & (time_mins <= 870)

        # Long: bearish cross happened 2 bars ago, low ATR, close recovered above EMA21,
        # and close > VWAP*0.995
        long_entry = (bearish_cross_2ago & low_atr & (close > ema21)
                      & (close > vwap * 0.995) & time_ok)

        # Short: bullish cross happened 2 bars ago, low ATR, close fell below EMA21,
        # and close < VWAP*1.005
        short_entry = (bullish_cross_2ago & low_atr & (close < ema21)
                       & (close < vwap * 1.005) & time_ok)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=target_pct,
            stop_loss_pct=stop_pct,
            time_stop_bars=40,
        )
