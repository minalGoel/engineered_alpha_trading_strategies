"""gap_momentum_continuation — 5-second NIFTY index options strategy.

Mechanism: When NIFTY gaps >0.3% at the open due to overnight global cues (SGX Nifty,
US futures, Asia markets), institutional desks carry a directional bias. The first 15
minutes (09:15-09:30) forms a price-discovery range. A breakout above that range on a
bullish gap (or below on a bearish gap), confirmed by a volume surge, signals that
resting institutional buy/sell interest has overwhelmed the supply ceiling and TWAP/VWAP
algorithms are chasing fills — creating a 30-90 second momentum continuation window.

Original: gap_momentum_continuation — nifty200 stocks, 1-min bars, hold 60-120 min.
Adapted: NIFTY index, 5-second bars, hold 30-90s, gap threshold 0.3% (vs 2% for stocks).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "gap_momentum_continuation"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — wait for ORB to form
    session_end_minutes = 690     # 11:30 IST — gap momentum fades by noon
    max_trades_per_day = 4
    max_lookback = 240            # 20 min warmup for volume MA

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_thresh", 0.003, 0.001, 0.008),   # 0.1% – 0.8% gap
            TunableParam("vol_mult", 1.5, 1.0, 3.0),           # volume surge multiplier
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill NaN in Polars before extracting numpy arrays
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        open_ = spot_df["open"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        gap_thresh = params.get("gap_thresh", 0.003)
        vol_mult = params.get("vol_mult", 1.5)

        # ── 1. Gap computation ─────────────────────────────────────────────────
        # gap_pct[d] = (day d first bar open - day d-1 last bar close) / last close
        # This is a per-day scalar, broadcast to all bars of that day.
        unique_days = np.unique(day_id)
        day_last_close: dict[int, float] = {}
        day_first_open: dict[int, float] = {}

        for d in unique_days:
            idxs = np.where(day_id == d)[0]
            day_last_close[d] = float(close[idxs[-1]])
            day_first_open[d] = float(open_[idxs[0]])

        gap_pct_arr = np.zeros(n, dtype=np.float64)
        for d in unique_days:
            prev_d = int(d) - 1
            if prev_d in day_last_close and day_last_close[prev_d] > 0:
                g = (day_first_open[d] - day_last_close[prev_d]) / day_last_close[prev_d]
            else:
                g = 0.0
            gap_pct_arr[day_id == d] = g

        # ── 2. Opening Range Breakout (ORB) ────────────────────────────────────
        # ORB = first 15 minutes: 09:15–09:29 IST = time_min in [555, 569]
        # This is a FIXED TIME WINDOW (same 15 min regardless of bar size).
        orb_high_arr = np.zeros(n, dtype=np.float64)       # 0 = no valid ORB data
        orb_low_arr = np.full(n, 1e9, dtype=np.float64)    # 1e9 = no valid ORB data

        for d in unique_days:
            mask_day = day_id == d
            mask_orb = mask_day & (time_min >= 555) & (time_min < 570)
            if np.any(mask_orb):
                orb_h = float(np.max(high[mask_orb]))
                orb_l = float(np.min(low[mask_orb]))
            else:
                # No ORB data for this day — disable signals
                orb_h = 0.0
                orb_l = 1e9
            orb_high_arr[mask_day] = orb_h
            orb_low_arr[mask_day] = orb_l

        # ── 3. Relative volume: current bar vs 20-min rolling average ──────────
        # 20 min = 240 bars at 5s (matches original's sma(volume, 20) on 1-min bars)
        vol_cumsum = np.concatenate([[0.0], np.cumsum(volume)])
        vol_ma = np.zeros(n, dtype=np.float64)
        for i in range(n):
            start = max(0, i - 240)
            length = i - start
            if length > 0:
                vol_ma[i] = (vol_cumsum[i] - vol_cumsum[start]) / length
            else:
                vol_ma[i] = volume[i] if volume[i] > 0 else 1.0
        # Avoid division by zero
        vol_ma = np.where(vol_ma > 0, vol_ma, 1.0)

        # ── 4. First-cross detection ───────────────────────────────────────────
        # Only fire on the bar where price FIRST crosses the ORB level.
        # At 5s resolution, a breakout spans many bars; the first-cross filter
        # prevents re-entering the same breakout on subsequent bars.
        prev_close = np.concatenate([[close[0]], close[:-1]])

        # ── 5. Session and signal masks ────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        valid_orb_high = orb_high_arr > 0.0
        valid_orb_low = orb_low_arr < 1e8

        bullish_gap = gap_pct_arr > gap_thresh
        bearish_gap = gap_pct_arr < -gap_thresh
        vol_surge = volume > (vol_ma * vol_mult)

        # Bullish gap + first breakout above ORB high + volume surge → buy CE
        ce_first_cross = (close > orb_high_arr) & (prev_close <= orb_high_arr)
        buy_ce = (
            in_session
            & bullish_gap
            & valid_orb_high
            & ce_first_cross
            & vol_surge
        )

        # Bearish gap + first breakout below ORB low + volume surge → buy PE
        pe_first_cross = (close < orb_low_arr) & (prev_close >= orb_low_arr)
        buy_pe = (
            in_session
            & bearish_gap
            & valid_orb_low
            & pe_first_cross
            & vol_surge
        )

        # No signal-based exits — rely on stop/target/time_stop
        sell_ce = np.zeros(n, dtype=bool)
        sell_pe = np.zeros(n, dtype=bool)

        # Stop: 5 pts — ORB breakout reversal of ~10 NIFTY spot pts invalidates thesis
        # Target: 8 pts — first continuation leg extends ~15-20 NIFTY spot pts (delta ~0.5)
        # Time stop: 18 bars = 90 seconds — gap momentum window
        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=sell_ce,
            sell_pe=sell_pe,
            stop_points=np.full(n, 5.0),
            target_points=np.full(n, 8.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,
            max_trades_per_day=self.max_trades_per_day,
        )
