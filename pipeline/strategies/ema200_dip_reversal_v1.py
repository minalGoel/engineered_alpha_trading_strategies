"""EMA 60-Minute Dip Reversal — NIFTY Index Options

Original: EMA200-5min dip bounce on NIFTY200 stocks (equity, 1-min bars, 10-30 min hold).
Converted: EMA(720) at 5s = 60-min intraday support bounce on NIFTY, 15-90s CE hold.

Mechanism:
  When NIFTY touches its 60-minute EMA, resting buy orders from systematic mean-reversion
  algos and delta-hedging market makers absorb selling. At 5s resolution we detect the first
  recovery bar (positive 5s return) as price re-crosses the touch zone, entering CE options
  before the move develops fully.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Compute exponential moving average via Wilder/standard alpha=2/(N+1)."""
    alpha = 2.0 / (period + 1)
    ema = np.empty(len(prices))
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1.0 - alpha) * ema[i - 1]
    return ema


class Strategy(BaseStrategy):
    name = "ema200_dip_reversal_v1"
    underlying = "NIFTY"
    session_start_minutes = 575   # 09:35 IST — skip first 20 min of noisy pre-trend
    session_end_minutes = 920     # 15:20 IST — avoid EOD unwinding
    max_trades_per_day = 8
    max_lookback = 720            # 60-min warmup (720 × 5s = 3600s); signals from ~10:15

    def tunable_params(self) -> list[TunableParam]:
        return [
            # How close to EMA triggers the "touch zone" (fraction of price)
            TunableParam("ema_touch_pct", 0.0015, 0.0005, 0.004),
            # Minimum 5s return to confirm bounce started
            TunableParam("bounce_threshold", 0.0002, 0.00005, 0.001),
            # VIX ceiling: avoid high-vol regimes where EMA breaks become trends
            TunableParam("vix_max", 20.0, 15.0, 28.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Raw arrays (forward-fill NaN before numpy) ──────────────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        ema_touch_pct = params.get("ema_touch_pct", 0.0015)
        bounce_threshold = params.get("bounce_threshold", 0.0002)
        vix_max = params.get("vix_max", 20.0)

        # ── EMA(720) — 60-minute exponential moving average ─────────────────
        ema_720 = _compute_ema(close, 720)

        # ── Signed % distance from EMA ───────────────────────────────────────
        # positive = above EMA, negative = below EMA
        dist_pct = (close - ema_720) / ema_720

        # ── Single-bar 5s return (bounce detection) ──────────────────────────
        ret_1 = np.zeros(n)
        ret_1[1:] = (close[1:] - close[:-1]) / close[:-1]

        # ── 30-second momentum (6 bars × 5s) ─────────────────────────────────
        mom_6 = np.zeros(n)
        for i in range(6, n):
            mom_6[i] = (close[i] - close[i - 6]) / close[i - 6]

        # ── Prior EMA test: rolling-min |dist_pct| over last 20 min (240 bars) ──
        # Proxies original "multi-day support" — EMA must have been tested recently
        # this session before we trust it as support.
        abs_dist = np.abs(dist_pct)
        prior_min_dist = np.full(n, np.inf)
        prior_window = 240  # 20 minutes
        for i in range(prior_window, n):
            prior_min_dist[i] = np.min(abs_dist[i - prior_window:i])

        # EMA considered "tested support" if it was touched within 20 min
        prior_tested = prior_min_dist < (2.0 * ema_touch_pct)

        # ── VIX — aligned to spot bars via backward asof join ────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(
                    ["datetime", pl.col("close").alias("vix_close")]
                ).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Session and warmup masks ──────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )
        warmed_up = np.zeros(n, dtype=bool)
        warmed_up[720:] = True  # need 720 bars for EMA to stabilise

        # ── Touch zone: price within ema_touch_pct of EMA ────────────────────
        near_ema = abs_dist < ema_touch_pct

        # ── Approach direction: prior bar was at or above EMA (not already broken) ──
        coming_from_above = np.zeros(n, dtype=bool)
        coming_from_above[1:] = dist_pct[:-1] >= -ema_touch_pct

        # ── BUY CE: dip to EMA then first recovery bar ────────────────────────
        buy_ce = (
            in_session
            & warmed_up
            & near_ema
            & coming_from_above
            & (ret_1 > bounce_threshold)      # current 5s bar positive — bounce start
            & (mom_6 > -0.0003)               # not in a sharp 30s breakdown
            & prior_tested                     # EMA acted as support earlier this session
            & (vix_close < vix_max)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=np.zeros(n, dtype=bool),
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            # stop=4pts (~8 NIFTY spot pts): EMA definitively broken if price falls 8pts below touch
            # target=7pts (~14 NIFTY spot pts): conservative lower bound of typical EMA-bounce return
            stop_points=np.full(n, 3),
            target_points=np.full(n, 6),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,        # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
