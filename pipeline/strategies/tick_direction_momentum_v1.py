"""
tick_direction_momentum_v1 — 5-second NIFTY index options strategy.

Converts the original equity tick-direction strategy (uptick/downtick counts on
NIFTY50 stocks) to a 5-second NIFTY index options strategy.

Mechanism:
  On NIFTY, institutional TWAP and momentum algorithms imprint persistent direction
  bias in 5-second bar closes. When >62% of the last 36 bars (3 min) close higher
  and this ratio is accelerating, buy-side flow is continuously absorbing liquidity —
  market makers short gamma delta-hedge by buying the index, compounding the signal.
  We enter 30-60s into the dominance phase, before it shows on 1-min charts.

Original: trading_strategies/unique_strategies_all/Strategy_241.json
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "tick_direction_momentum_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 15
    max_lookback = 120            # 10 min warmup (36 ratio + 24 momentum + buffer)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("tick_ratio_threshold", 0.62, 0.55, 0.72),
            TunableParam("tick_momentum_threshold", 0.08, 0.04, 0.16),
            TunableParam("price_roc_threshold", 0.0003, 0.0001, 0.0008),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        tick_ratio_thresh = params.get("tick_ratio_threshold", 0.62)
        tick_mom_thresh = params.get("tick_momentum_threshold", 0.08)
        price_roc_thresh = params.get("price_roc_threshold", 0.0003)

        # ── Extract arrays from spot_df (forward-fill NaN before numpy) ──────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # ── Session filter ────────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (
            time_min < self.session_end_minutes
        )

        # ── Indicator 1: Tick direction ratio (rolling 36 bars = 3 min) ──────
        # Approximate tick direction: each 5s bar is an "uptick bar" if close > prev_close
        tick_up = np.zeros(n)
        tick_up[1:] = (close[1:] > close[:-1]).astype(float)

        tick_dir_ratio = np.full(n, 0.5)
        for i in range(36, n):
            tick_dir_ratio[i] = np.mean(tick_up[i - 36 : i])

        # ── Indicator 2: Tick direction momentum (ratio change over 24 bars = 2 min) ─
        tick_dir_momentum = np.zeros(n)
        for i in range(60, n):
            tick_dir_momentum[i] = tick_dir_ratio[i] - tick_dir_ratio[i - 24]

        # ── Indicator 3: 1-minute price ROC (12 bars) ─────────────────────────
        price_roc_12 = np.zeros(n)
        safe_close = np.where(close == 0, 1.0, close)
        price_roc_12[12:] = (close[12:] - close[:-12]) / safe_close[:-12]

        # ── Indicator 4: Session VWAP (cumulative, reset per day) ─────────────
        typical = (high + low + close) / 3.0
        vwap = np.zeros(n)
        cum_tp_vol = 0.0
        cum_vol = 0.0
        current_day = day_id[0] if n > 0 else -1
        for i in range(n):
            if day_id[i] != current_day:
                cum_tp_vol = 0.0
                cum_vol = 0.0
                current_day = day_id[i]
            cum_tp_vol += typical[i] * volume[i]
            cum_vol += volume[i]
            vwap[i] = cum_tp_vol / cum_vol if cum_vol > 0.0 else close[i]

        # ── Entry signals ──────────────────────────────────────────────────────
        # buy_ce: sustained + accelerating uptick dominance, price confirming, above VWAP
        buy_ce = (
            in_session
            & (tick_dir_ratio > tick_ratio_thresh)
            & (tick_dir_momentum > tick_mom_thresh)
            & (price_roc_12 > price_roc_thresh)
            & (close > vwap)
        )

        # buy_pe: sustained + accelerating downtick dominance, price confirming, below VWAP
        buy_pe = (
            in_session
            & (tick_dir_ratio < (1.0 - tick_ratio_thresh))
            & (tick_dir_momentum < -tick_mom_thresh)
            & (price_roc_12 < -price_roc_thresh)
            & (close < vwap)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, 4.0),
            target_points=np.full(n, 6.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,         # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
