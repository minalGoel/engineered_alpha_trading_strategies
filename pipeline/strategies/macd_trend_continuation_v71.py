"""MACD Trend Continuation (Pullback Resumption) on NIFTY 5s bars.

Converted from: trading_strategies/unique_strategies_all/Strategy_111.json
Original: MACD crossover + EMA(50) trend on nifty200 stocks, 15-min bars.

Mechanism:
  On NIFTY, institutional TWAP/VWAP algorithms create persistent 3-8 minute micro-trends
  (visible via EMA(60) at 5s bars). Within these trends, 20-40 second pullbacks push the
  5s MACD briefly negative. When the MACD histogram crosses back above its signal line
  while still negative, institutional buying has resumed before the full re-acceleration —
  resting buy orders just absorbed the pullback supply. Entering this resumption at 5s
  resolution captures 10-15 NIFTY spot points before it's visible on 1-min bars.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average with standard 2/(period+1) multiplier."""
    k = 2.0 / (period + 1)
    ema = np.empty_like(arr, dtype=np.float64)
    ema[0] = arr[0]
    for i in range(1, len(arr)):
        ema[i] = arr[i] * k + ema[i - 1] * (1.0 - k)
    return ema


class Strategy(BaseStrategy):
    name = "macd_trend_continuation_v71"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — skip first 10 min of pre-open noise
    session_end_minutes = 920     # 15:20 IST — flatten before expiry risk
    max_trades_per_day = 8
    max_lookback = 120            # 10-min warmup for EMA(60) to stabilise

    def tunable_params(self) -> list[TunableParam]:
        return [
            # Minimum histogram value at crossover (filters very weak crossovers)
            TunableParam("hist_min_cross", 0.0, -0.5, 1.0),
            # Price must be above trend EMA by at least this fraction
            TunableParam("trend_gap_pct", 0.0002, 0.0, 0.001),
            TunableParam("stop_pts", 3.0, 2.0, 6.0),
            TunableParam("target_pts", 5.0, 3.0, 10.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract spot data (forward-fill NaN before numpy) ──────────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()

        # ── Params ──────────────────────────────────────────────────────────────
        hist_min_cross = params.get("hist_min_cross", 0.0)
        trend_gap_pct = params.get("trend_gap_pct", 0.0002)
        stop_pts = params.get("stop_pts", 3.0)
        target_pts = params.get("target_pts", 5.0)

        # ── MACD(12, 26, 9) at 5s resolution ───────────────────────────────────
        # Fast EMA(12) = 1-min, Slow EMA(26) = 2.2-min.
        # Proportions preserved from original 15-min MACD — same fast/slow ratio
        # determines what counts as a "pullback" (macd dips negative) vs trend.
        ema_fast = _ema(close, 12)
        ema_slow = _ema(close, 26)
        macd_line = ema_fast - ema_slow
        macd_sig = _ema(macd_line, 9)
        macd_hist = macd_line - macd_sig

        # ── 5-minute trend filter: EMA(60) ──────────────────────────────────────
        # Replaces original EMA(50) at 15-min (=750-min context).
        # Compressed to 5-min because our 30-120s hold needs proportional context.
        ema_trend = _ema(close, 60)

        # ── Histogram crossover detection ───────────────────────────────────────
        # prev_hist: previous bar's histogram value
        prev_hist = np.empty(n, dtype=np.float64)
        prev_hist[0] = macd_hist[0]
        prev_hist[1:] = macd_hist[:-1]

        # Bullish: histogram crosses from negative to positive (pullback resuming up)
        # Key filter: macd_line < 0 — still in pullback zone, not chasing late entry
        bullish_cross = (
            (prev_hist <= hist_min_cross)
            & (macd_hist > hist_min_cross)
            & (macd_line < 0.0)
        )

        # Bearish: histogram crosses from positive to negative (bounce resuming down)
        bearish_cross = (
            (prev_hist >= -hist_min_cross)
            & (macd_hist < -hist_min_cross)
            & (macd_line > 0.0)
        )

        # ── Trend direction filter ───────────────────────────────────────────────
        above_trend = close > ema_trend * (1.0 + trend_gap_pct)
        below_trend = close < ema_trend * (1.0 - trend_gap_pct)

        # ── Session and warmup filters ───────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmup = np.zeros(n, dtype=bool)
        warmup[self.max_lookback:] = True

        # ── Final signals ────────────────────────────────────────────────────────
        buy_ce = in_session & warmup & bullish_cross & above_trend
        buy_pe = in_session & warmup & bearish_cross & below_trend

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts, dtype=np.float64),
            target_points=np.full(n, target_pts, dtype=np.float64),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,            # 120 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
