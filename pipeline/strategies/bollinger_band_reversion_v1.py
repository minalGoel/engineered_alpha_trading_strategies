"""
bollinger_band_reversion_v1 — Bollinger Band Mean Reversion on NIFTY 5s bars

Mechanism:
    When NIFTY closes below its 5-minute Bollinger lower band (60-bar, 2σ),
    algorithmic sell programs have exhausted near-term sell-side liquidity.
    Market makers net-short gamma immediately begin delta-hedging by buying
    NIFTY futures, and VWAP-benchmarked institutional desks see attractive
    fills and start accumulating. Reversion to the 5-min SMA typically
    unfolds within 30-90 seconds (10-15 spot points). Upper band breaches
    trigger the converse: gamma-hedge selling and profit-taking pull price
    back to the midline.

Converted from: trading_strategies/unique_strategies_all/Strategy_5.json
Original: 20-bar 1-min BB on nifty500 equities, 1-10 min hold.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "bollinger_band_reversion_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 8
    max_lookback = 60             # 60 bars × 5s = 5 min warmup for BB

    def tunable_params(self) -> list[TunableParam]:
        return [
            # Volume surge threshold: how many times avg volume is required
            TunableParam("vol_surge_threshold", 1.5, 0.8, 2.5),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(strategy="forward").fill_null(0).to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()

        vol_surge_threshold = params.get("vol_surge_threshold", 1.5)

        # ── 60-bar (5-minute) Bollinger Bands on spot NIFTY ──────────────────
        window = 60
        bb_mid = np.zeros(n)
        bb_upper = np.zeros(n)
        bb_lower = np.zeros(n)
        vol_avg = np.zeros(n)

        for i in range(window, n):
            w = close[i - window:i]
            mu = np.mean(w)
            sigma = np.std(w)
            bb_mid[i] = mu
            bb_upper[i] = mu + 2.0 * sigma
            bb_lower[i] = mu - 2.0 * sigma
            vol_avg[i] = np.mean(volume[i - window:i])

        # Volume ratio (avoid div-by-zero)
        vol_ratio = np.where(vol_avg > 0.0, volume / vol_avg, 0.0)

        # ── Masks ─────────────────────────────────────────────────────────────
        initialized = np.arange(n) >= window
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )
        vol_surge = vol_ratio >= vol_surge_threshold

        # buy CE: close below lower band with volume confirmation (bullish reversion)
        buy_ce = in_session & initialized & (close < bb_lower) & vol_surge

        # buy PE: close above upper band with volume confirmation (bearish reversion)
        buy_pe = in_session & initialized & (close > bb_upper) & vol_surge

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            # stop: 4 option pts — ~8 spot pt continuation = breakout, not reversion
            stop_points=np.full(n, 4.0),
            # target: 6 option pts — ~60% of typical 12-15 spot pt midline reversion
            target_points=np.full(n, 6.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
