"""ATR Expansion Breakout — claude_project_19_of_25

Thesis: When a single bar's range massively exceeds the average range (ATR),
it signals an institutional-grade breakout. Direction is determined by the
bar's open-close relationship, confirmed by volume and VWAP alignment.

# AUDIT FIX: short_entry was missing vol_ok and (close < vwap) filters that
# long_entry had, causing 3x more short signals than long signals (1032 vs 347).
# Also removed redundant (close < open_) — bearish already means close < open_.
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
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    if n < period:
        return atr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


class Strategy(BaseStrategy):
    name = "atr_expansion_breakout_v1"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 870     # 14:30
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("atr_expansion_mult", default=2.0, low=1.5, high=3.5),
            TunableParam("vol_mult", default=2.0, low=1.3, high=3.0),
            TunableParam("stop_loss_pct", default=0.005, low=0.002, high=0.008),
            TunableParam("trailing_stop_pct", default=0.005, low=0.002, high=0.01),
            TunableParam("vix_max", default=26.0, low=20.0, high=32.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        atr_expansion_mult = params.get("atr_expansion_mult", 2.0)
        vol_mult = params.get("vol_mult", 2.0)
        stop_pct = params.get("stop_loss_pct", 0.005)
        trailing_pct = params.get("trailing_stop_pct", 0.005)
        vix_max = params.get("vix_max", 26.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)

        volume = np.nan_to_num(volume, nan=1.0)
        vix = np.nan_to_num(vix, nan=99.0)

        # ── VWAP (daily reset) ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── ATR(20) ──
        atr = _compute_atr(high, low, close, 20)

        # ── Bar range ──
        bar_range = high - low

        # ── Volume filter ──
        avg_vol_20 = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol_20 = np.nan_to_num(avg_vol_20, nan=1.0)
        avg_vol_20 = np.clip(avg_vol_20, 1.0, None)
        vol_ok = volume > vol_mult * avg_vol_20

        # ── ATR expansion condition ──
        atr_safe = np.where(atr > 0, atr, 1e10)
        expansion = bar_range > atr_expansion_mult * atr_safe

        # ── Direction ──
        bullish = close > open_
        bearish = close < open_

        # ── VIX filter ──
        vix_ok = vix < vix_max

        # ── Entry ──
        long_entry = expansion & bullish & vol_ok & (close > vwap) & vix_ok
        short_entry = expansion & bearish & vol_ok & (close < vwap) & vix_ok

        # ── Target: 1x expansion bar range as pct of close ──
        # Compute per-bar expansion range pct; use a running value
        expansion_range_pct = np.where(
            close > 0, bar_range / close, 0.005
        )
        # Use median expansion range pct as the target
        expansion_bars = expansion_range_pct[expansion & (close > 0)]
        if len(expansion_bars) > 0:
            target_pct_val = float(np.median(expansion_bars))
        else:
            target_pct_val = 0.005

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=target_pct_val,
            stop_loss_pct=stop_pct,
            trailing_stop_pct=trailing_pct,
            trailing_activate_pct=0.0,
            time_stop_bars=60,
        )
