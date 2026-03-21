"""Vol Surface Signal (vol_surface_signal_v1)

Converted from: trading_strategies/unique_strategies_all/Strategy_223.json

Original: Equity/ETF strategy trading NIFTYBEES based on 25-delta put-call IV skew
          z-score over a 20-day rolling window. Hold 60-240 min. Fires 2-4x per month.

Converted: Intraday ATM PE/CE premium ratio z-score on NIFTY 5-second bars.
           When intraday put-buying (PE/CE ratio) reaches 2σ extreme, fade by buying CE.
           When call-buying extreme (ratio < -2σ), buy PE.
           Hold: 30-90 seconds (6-18 bars). Fires 3-6x per day.

Mechanism: On NIFTY, the ATM PE/CE premium ratio tracks real-time intraday order flow
imbalance. When put-buying spikes above its 20-min session baseline by 2σ, the demand
burst has exhausted and market-maker delta-hedging (index buying) drives a short rebound.
Symmetric fade applies when call-buying reaches 2σ extreme.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _rolling_zscore(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling z-score over a fixed window. Returns NaN for early bars."""
    n = len(arr)
    z = np.full(n, np.nan)
    for i in range(window, n):
        w = arr[i - window:i]
        valid = w[~np.isnan(w)]
        if len(valid) >= window // 2:
            mu = np.mean(valid)
            sigma = np.std(valid)
            if sigma > 1e-6:
                z[i] = (arr[i] - mu) / sigma
    return z


class Strategy(BaseStrategy):
    name = "vol_surface_signal_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — let post-open option pricing settle
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 5
    max_lookback = 360            # 30 min warmup (240 z-score window + buffer)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("z_threshold", 2.0, 1.5, 3.0),
            TunableParam("trend_filter_pct", 0.003, 0.001, 0.008),
            TunableParam("stop_pts", 5.0, 3.0, 8.0),
            TunableParam("target_pts", 8.0, 5.0, 15.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)
        close = spot_df["close"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        z_threshold = params.get("z_threshold", 2.0)
        trend_filter_pct = params.get("trend_filter_pct", 0.003)
        stop_pts = params.get("stop_pts", 5.0)
        target_pts = params.get("target_pts", 8.0)

        # ── 1-minute spot momentum (12 bars × 5s = 60s) ──
        mom_12 = np.zeros(n)
        for i in range(12, n):
            if close[i - 12] > 0:
                mom_12[i] = (close[i] - close[i - 12]) / close[i - 12]

        # ── VIX filter ──
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── ATM PE/CE premium from option_df ──
        atm_ce = np.full(n, np.nan)
        atm_pe = np.full(n, np.nan)

        if option_df is not None and not option_df.is_empty():
            # Use nearest expiry only — avoids mixing far-expiry pricing
            min_expiry_df = (
                option_df
                .group_by("session_date")
                .agg(pl.col("expiry").min().alias("min_expiry"))
            )
            option_near = (
                option_df
                .join(min_expiry_df, on="session_date")
                .filter(pl.col("expiry") == pl.col("min_expiry"))
            )

            # Attach ATM strike from spot to each option bar via asof join
            atm_ref = spot_df.select(["datetime", "atm_strike"]).sort("datetime")
            option_with_atm = (
                option_near
                .sort("datetime")
                .join_asof(atm_ref, on="datetime", strategy="backward")
            )

            # Time series of ATM CE closes
            ce_ts = (
                option_with_atm
                .filter(
                    (pl.col("option_type") == "CE")
                    & (pl.col("strike") == pl.col("atm_strike"))
                )
                .select(["datetime", pl.col("close").alias("ce_close")])
                .sort("datetime")
            )

            # Time series of ATM PE closes
            pe_ts = (
                option_with_atm
                .filter(
                    (pl.col("option_type") == "PE")
                    & (pl.col("strike") == pl.col("atm_strike"))
                )
                .select(["datetime", pl.col("close").alias("pe_close")])
                .sort("datetime")
            )

            # Asof-join CE and PE series back to spot bar timestamps
            if not ce_ts.is_empty():
                spot_ce = spot_df.select("datetime").join_asof(
                    ce_ts, on="datetime", strategy="backward"
                )
                if "ce_close" in spot_ce.columns:
                    raw = spot_ce["ce_close"].fill_null(strategy="forward").to_numpy()
                    atm_ce = raw.astype(float)

            if not pe_ts.is_empty():
                spot_pe = spot_df.select("datetime").join_asof(
                    pe_ts, on="datetime", strategy="backward"
                )
                if "pe_close" in spot_pe.columns:
                    raw = spot_pe["pe_close"].fill_null(strategy="forward").to_numpy()
                    atm_pe = raw.astype(float)

            # Treat zero as missing (no valid quote)
            atm_ce = np.where(atm_ce == 0.0, np.nan, atm_ce)
            atm_pe = np.where(atm_pe == 0.0, np.nan, atm_pe)

        # ── PE/CE ratio → rolling 20-min z-score (240 bars) ──
        # Valid only when both options have meaningful premium (> 5 pts)
        valid_options = (atm_ce > 5.0) & (atm_pe > 5.0)
        pe_ce_ratio = np.where(valid_options, atm_pe / atm_ce, np.nan)

        # 240 bars = 20 minutes: chosen for intraday mean-reversion timing on NIFTY
        ratio_z = _rolling_zscore(pe_ce_ratio, window=240)
        # Replace NaN in ratio_z with 0.0 so comparisons are safe
        ratio_z_safe = np.where(np.isnan(ratio_z), 0.0, ratio_z)

        # ── Composite filters ──
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        vix_ok = (vix_close >= 12.0) & (vix_close <= 25.0)
        has_signal = ~np.isnan(ratio_z)  # only trade when z-score is computed

        # ── Entry signals ──
        # buy_ce: extreme put-buying exhausted → NIFTY likely to rebound (fade fear)
        buy_ce = (
            in_session
            & vix_ok
            & valid_options
            & has_signal
            & (ratio_z_safe > z_threshold)
            & (mom_12 > -trend_filter_pct)
        )

        # buy_pe: extreme call-buying exhausted → NIFTY likely to pull back (fade greed)
        buy_pe = (
            in_session
            & vix_ok
            & valid_options
            & has_signal
            & (ratio_z_safe < -z_threshold)
            & (mom_12 < trend_filter_pct)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,           # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
