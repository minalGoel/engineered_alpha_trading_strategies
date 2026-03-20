"""Volatility Breakout (ATR Expansion) — Opus_24

Thesis: When the ratio of short-term ATR(10) to long-term ATR(60)
exceeds 2.0, a volatility expansion is underway.  Enter on the
breakout bar with volume and VWAP confirmation.  Exit when the
expansion ratio contracts below 1.5.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Wilder's ATR."""
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i - 1]),
                     abs(low[i] - close[i - 1]))
    if n < period:
        return atr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


class Strategy(BaseStrategy):
    name = "opus_volatility_breakout_v1"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 870     # 14:30
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("expansion_thresh", default=2.0, low=1.5, high=3.0),
            TunableParam("contraction_exit", default=1.5, low=1.0, high=2.0),
            TunableParam("vol_mult", default=2.0, low=1.5, high=3.0),
            TunableParam("target_atr_mult", default=1.5, low=1.0, high=2.5),
            TunableParam("stop_atr_mult", default=1.0, low=0.5, high=1.5),
            TunableParam("trailing_stop_pct", default=0.003, low=0.002, high=0.005),
            TunableParam("trailing_activate_pct", default=0.004, low=0.002, high=0.006),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        exp_thresh = params.get("expansion_thresh", 2.0)
        cont_exit = params.get("contraction_exit", 1.5)
        vol_mult = params.get("vol_mult", 2.0)
        tgt_mult = params.get("target_atr_mult", 1.5)
        stp_mult = params.get("stop_atr_mult", 1.0)
        trail_pct = params.get("trailing_stop_pct", 0.003)
        trail_act = params.get("trailing_activate_pct", 0.004)

        n = len(df)
        open_ = df["open"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        day_id = df["day_id"].to_numpy()

        # ── VWAP ──
        df_vwap = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns(
            (pl.col("_ctv") / pl.col("_cv")).alias("_vwap")
        )
        vwap = df_vwap["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── ATR(10) and ATR(60) ──
        atr10 = _compute_atr(high, low, close, 10)
        atr60 = _compute_atr(high, low, close, 60)

        # ── Vol expansion ratio ──
        safe_atr60 = np.where(atr60 > 1e-12, atr60, 1e-12)
        vol_expansion = atr10 / safe_atr60

        # ── Relative volume ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        rel_vol = volume / avg_vol

        is_bullish = close > open_
        is_bearish = close < open_

        expanding = vol_expansion > exp_thresh
        vol_ok = rel_vol > vol_mult

        long_entry = expanding & is_bullish & vol_ok & (close > vwap)
        short_entry = expanding & is_bearish & vol_ok & (close < vwap)

        # ── Signal exit: expansion contracts below threshold ──
        sig_exit_long = vol_expansion < cont_exit
        sig_exit_short = vol_expansion < cont_exit

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr10,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_atr_mult=tgt_mult,
            stop_loss_atr_mult=stp_mult,
            trailing_stop_pct=trail_pct,
            trailing_activate_pct=trail_act,
            time_stop_bars=45,
        )
