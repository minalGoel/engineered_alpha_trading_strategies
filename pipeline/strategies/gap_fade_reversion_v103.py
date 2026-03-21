"""gap_fade_reversion_v103 — NIFTY opening gap fade strategy.

When NIFTY gaps >0.3% overnight, option market makers who sold gamma are immediately
delta-offside and begin hedging within 30 seconds. VWAP/TWAP algorithms benchmarked to
prev close aggressively fade the open, creating a 20-50 spot point reversion impulse in
the first 60-90 seconds. We detect the first 15-second confirmation of the fade and enter
before the bulk of the impulse occurs.

Trades: 1-3 per day. Window: 09:15-09:35 IST only.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "gap_fade_reversion_v103"
    underlying = "NIFTY"
    session_start_minutes = 555   # 09:15 IST — gap fade starts at open
    session_end_minutes = 575     # 09:35 IST — only first 20 min
    max_trades_per_day = 3
    max_lookback = 240            # 20 min warmup (to accumulate first day's data)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_threshold", 0.003, 0.001, 0.008),   # min gap fraction
            TunableParam("vol_mult", 1.2, 1.0, 2.0),              # volume surge multiplier
            TunableParam("stop_pts", 3.0, 2.0, 8.0),
            TunableParam("target_pts", 6.0, 4.0, 14.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract arrays (forward-fill nulls before numpy) ──────────────────
        close = (
            spot_df["close"]
            .fill_nan(None)
            .fill_null(strategy="forward")
            .to_numpy()
        )
        open_ = (
            spot_df["open"]
            .fill_nan(None)
            .fill_null(strategy="forward")
            .to_numpy()
        )
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        day_id = spot_df["day_id"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        gap_threshold = params.get("gap_threshold", 0.003)
        vol_mult = params.get("vol_mult", 1.2)
        stop_pts = params.get("stop_pts", 3.0)
        target_pts = params.get("target_pts", 6.0)

        # ── Compute day-level open and previous-day close ─────────────────────
        day_open = np.zeros(n)
        prev_day_close = np.zeros(n)

        current_day = -1
        this_day_open = np.nan
        last_day_close = 0.0  # 0.0 signals "no prior day yet" → has_gap_ref=False

        for i in range(n):
            if day_id[i] != current_day:
                # Transition to new day
                if current_day >= 0 and i > 0:
                    last_day_close = close[i - 1]  # last bar of previous day
                current_day = int(day_id[i])
                this_day_open = open_[i]

            day_open[i] = this_day_open if not np.isnan(this_day_open) else close[i]
            prev_day_close[i] = last_day_close

        # ── Gap percentage ────────────────────────────────────────────────────
        # gap_pct = (day_open - prev_day_close) / prev_day_close
        gap_pct = np.zeros(n)
        valid_ref = prev_day_close > 0.0
        gap_pct[valid_ref] = (
            (day_open[valid_ref] - prev_day_close[valid_ref])
            / prev_day_close[valid_ref]
        )

        # ── 15-second fade momentum (3-bar lookback) ──────────────────────────
        # Gap-up fade (bearish): price falling back below day_open
        # Gap-dn fade (bullish): price rising back above day_open
        fade_bars = 3
        fade_up_conf = np.zeros(n, dtype=bool)   # gap-up fading → buy PE
        fade_dn_conf = np.zeros(n, dtype=bool)   # gap-dn fading → buy CE

        for i in range(fade_bars, n):
            c0 = close[i]
            c3 = close[i - fade_bars]
            do = day_open[i]
            # Gap-up fade: current close below today's open AND falling over last 15s
            if c0 < do and c0 < c3:
                fade_up_conf[i] = True
            # Gap-dn fade: current close above today's open AND rising over last 15s
            if c0 > do and c0 > c3:
                fade_dn_conf[i] = True

        # ── Volume surge: current bar > vol_mult * rolling 12-bar mean ────────
        vol_ma = np.zeros(n)
        for i in range(12, n):
            vol_ma[i] = np.mean(volume[i - 12:i])

        vol_surge = (vol_ma > 0) & (volume > vol_mult * vol_ma)

        # ── VIX filter ────────────────────────────────────────────────────────
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

        # Skip extreme VIX: below 10 = no fear (microstructural gap, not behavioral)
        # above 25 = macro event (gap won't fade on behavioral timing)
        vix_ok = (vix_close >= 10.0) & (vix_close <= 25.0)

        # ── Gate: must have a valid prev-day reference ─────────────────────────
        has_gap_ref = prev_day_close > 0.0

        # ── Gap direction filters ─────────────────────────────────────────────
        gap_up = gap_pct > gap_threshold
        gap_dn = gap_pct < -gap_threshold

        # ── Time window: first 20 min only ────────────────────────────────────
        in_gap_window = (time_min >= self.session_start_minutes) & (
            time_min < self.session_end_minutes
        )

        # ── Entry signals ─────────────────────────────────────────────────────
        buy_ce = (
            in_gap_window & has_gap_ref & gap_dn & fade_dn_conf & vol_surge & vix_ok
        )
        buy_pe = (
            in_gap_window & has_gap_ref & gap_up & fade_up_conf & vol_surge & vix_ok
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=12,          # 60 seconds — gap fade impulse is fast
            max_trades_per_day=self.max_trades_per_day,
        )
