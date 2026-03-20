"""intraday_range_reversion_v1 — Session extreme rejection on NIFTY.

When NIFTY touches its session high/low and forms a rejection candle (hammer at
session low, shooting star at session high), VWAP-benchmarked institutional
algorithms aggressively fade the extreme within 30-90 seconds. We enter ATM
options on the rejection bar itself.

Original: intraday range reversion on NIFTY200 stocks, 1-min bars, 10-40 min hold.
Converted: NIFTY index options, 5-second bars, 30-90 second hold.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "intraday_range_reversion_v1"
    underlying = "NIFTY"
    session_start_minutes = 600   # 10:00 IST — 45 min after open for range to form
    session_end_minutes = 885     # 14:45 IST
    max_trades_per_day = 3
    max_lookback = 540            # 45 min = 540 bars of warmup (09:15 → 10:00)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("pin_bar_threshold", 0.65, 0.50, 0.82),
            TunableParam("range_pct_min", 0.5, 0.3, 1.0),
            TunableParam("proximity_pct", 0.002, 0.001, 0.005),
            TunableParam("vix_low", 13.0, 10.0, 17.0),
            TunableParam("vix_high", 23.0, 18.0, 28.0),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 6.0, 4.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        pin_bar_threshold = params.get("pin_bar_threshold", 0.65)
        range_pct_min = params.get("range_pct_min", 0.5)
        proximity_pct = params.get("proximity_pct", 0.002)
        vix_low = params.get("vix_low", 13.0)
        vix_high = params.get("vix_high", 23.0)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 6.0)

        # ── VIX close (aligned to spot bars) ──
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Session high/low (cumulative per trading day) ──
        spot_with_extremes = spot_df.with_columns([
            pl.col("high").cum_max().over("day_id").alias("session_high"),
            pl.col("low").cum_min().over("day_id").alias("session_low"),
        ])

        session_high = (
            spot_with_extremes["session_high"]
            .fill_null(strategy="forward")
            .to_numpy()
        )
        session_low = (
            spot_with_extremes["session_low"]
            .fill_null(strategy="forward")
            .to_numpy()
        )

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        # ── Intraday range percentage ──
        # Avoid division by zero if session_low is 0 (should not happen on NIFTY)
        sl_safe = np.where(session_low < 1.0, 1.0, session_low)
        range_pct = (session_high - session_low) / sl_safe * 100.0

        # ── Pin bar scores ──
        # Clamp bar range to avoid division by zero on flat bars
        bar_range = np.maximum(high - low, 0.1)
        pin_bar_long = (close - low) / bar_range    # hammer: close near top
        pin_bar_short = (high - close) / bar_range  # shooting star: close near bottom

        # ── Proximity to session extreme ──
        # Bar's low within proximity_pct ABOVE session low (bar touched near the extreme)
        # session_low <= low always (cummin), so low / session_low >= 1.0
        # We want low <= session_low * (1 + proximity_pct)
        near_session_low = low <= session_low * (1.0 + proximity_pct)

        # Bar's high within proximity_pct BELOW session high
        # session_high >= high always (cummax), so high / session_high <= 1.0
        # We want high >= session_high * (1 - proximity_pct)
        near_session_high = high >= session_high * (1.0 - proximity_pct)

        # ── Filters ──
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        vix_ok = (vix_close >= vix_low) & (vix_close <= vix_high)
        range_ok = range_pct >= range_pct_min

        # ── Entry signals ──
        # Buy CE: hammer at session low (bullish rejection)
        buy_ce = (
            in_session
            & vix_ok
            & range_ok
            & near_session_low
            & (close > session_low)          # bar closed above the extreme (genuine rejection)
            & (pin_bar_long >= pin_bar_threshold)
        )

        # Buy PE: shooting star at session high (bearish rejection)
        buy_pe = (
            in_session
            & vix_ok
            & range_ok
            & near_session_high
            & (close < session_high)         # bar closed below the extreme (genuine rejection)
            & (pin_bar_short >= pin_bar_threshold)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,              # 90 seconds (18 × 5s)
            max_trades_per_day=self.max_trades_per_day,
        )
