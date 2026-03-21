"""heikin_ashi_trend_v1 — Heikin Ashi micro-trend following on NIFTY 5s bars.

Mechanism:
    On NIFTY, institutional TWAP algorithms executing large directional orders create
    30-90 second micro-trends where each 5-second bar is consistently one-directional.
    Heikin Ashi candles detect this: when 3+ consecutive green HA bars have no lower
    wick (HA_low ≈ HA_open), NIFTY sellers are absent within every 5-second interval.
    Combined with price above VWAP and short-term EMA alignment, this identifies the
    continuation window before the institutional batch order completes.

Converted from: trading_strategies/unique_strategies_all/Strategy_287.json
Original: 1-min HA trend on NIFTY50 stocks, 5-20 min hold.
Adapted: 5s NIFTY index options, 30-90s hold, stops/targets in option premium points.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "heikin_ashi_trend_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — skip first 10 min of opening noise
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 8
    max_lookback = 120            # 10 min warmup for EMA(108) and VWAP stabilization

    def tunable_params(self) -> list[TunableParam]:
        return [
            # Minimum consecutive HA bars of same color before entry fires
            TunableParam("min_consecutive_bars", 3.0, 2.0, 6.0),
            # Max opposing wick as fraction of HA body to qualify as "strong" bar
            TunableParam("wick_tolerance_frac", 0.20, 0.05, 0.40),
            # Stop loss in option premium points
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            # Target in option premium points
            TunableParam("target_pts", 7.0, 4.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── Extract and forward-fill spot arrays ──────────────────────────────
        open_ = spot_df["open"].fill_null(strategy="forward").to_numpy()
        high  = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low   = spot_df["low"].fill_null(strategy="forward").to_numpy()
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id   = spot_df["day_id"].to_numpy()

        min_consec = int(params.get("min_consecutive_bars", 3))
        wick_tol   = params.get("wick_tolerance_frac", 0.20)
        stop_pts   = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 7.0)

        # ── Heikin Ashi construction ──────────────────────────────────────────
        ha_close = (open_ + high + low + close) / 4.0

        ha_open = np.empty(n)
        ha_open[0] = (open_[0] + close[0]) / 2.0
        for i in range(1, n):
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

        ha_high = np.maximum(high, np.maximum(ha_open, ha_close))
        ha_low  = np.minimum(low,  np.minimum(ha_open, ha_close))

        ha_body = np.abs(ha_close - ha_open)
        # Use a floor to avoid divide-by-near-zero on flat bars
        min_body = np.maximum(ha_body, 0.05)

        ha_green = ha_close > ha_open
        ha_red   = ha_close < ha_open

        # Lower wick for green bars: distance below HA_open (ideal = 0)
        lower_wick = ha_open - ha_low
        # Upper wick for red bars: distance above HA_open (ideal = 0)
        upper_wick = ha_high - ha_open

        # Strong green: green body AND lower wick is negligible AND body is real
        strong_green = ha_green & (ha_body > 0.1) & (lower_wick < wick_tol * min_body)
        # Strong red: red body AND upper wick is negligible AND body is real
        strong_red   = ha_red   & (ha_body > 0.1) & (upper_wick < wick_tol * min_body)

        # ── Consecutive HA bar count (reset at day boundaries) ────────────────
        consec_green = np.zeros(n, dtype=np.int32)
        consec_red   = np.zeros(n, dtype=np.int32)

        for i in range(n):
            new_day = (i == 0) or (day_id[i] != day_id[i - 1])
            if new_day:
                consec_green[i] = 1 if ha_green[i] else 0
                consec_red[i]   = 1 if ha_red[i]   else 0
            else:
                consec_green[i] = (consec_green[i - 1] + 1) if ha_green[i] else 0
                consec_red[i]   = (consec_red[i - 1]   + 1) if ha_red[i]   else 0

        # ── Session VWAP (cumulative from open, reset each day) ───────────────
        typical   = (high + low + close) / 3.0
        cum_tp_vol = np.zeros(n)
        cum_vol    = np.zeros(n)

        for i in range(n):
            if i == 0 or day_id[i] != day_id[i - 1]:
                cum_tp_vol[i] = typical[i] * volume[i]
                cum_vol[i]    = volume[i]
            else:
                cum_tp_vol[i] = cum_tp_vol[i - 1] + typical[i] * volume[i]
                cum_vol[i]    = cum_vol[i - 1]    + volume[i]

        vwap = np.where(cum_vol > 0, cum_tp_vol / cum_vol, close)

        # ── Dual EMA on actual close ──────────────────────────────────────────
        # EMA(36) = 3-min fast: trade-trigger timeframe (compressed from original 9-min)
        # EMA(108) = 9-min slow: trend context (compressed from original 21-min)
        alpha_fast = 2.0 / (36  + 1)
        alpha_slow = 2.0 / (108 + 1)

        ema_fast = np.empty(n)
        ema_slow = np.empty(n)
        ema_fast[0] = close[0]
        ema_slow[0] = close[0]

        for i in range(1, n):
            ema_fast[i] = alpha_fast * close[i] + (1.0 - alpha_fast) * ema_fast[i - 1]
            ema_slow[i] = alpha_slow * close[i] + (1.0 - alpha_slow) * ema_slow[i - 1]

        # ── VIX regime filter ─────────────────────────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # VIX 12-25: need trend regimes; below 12 = too calm (no HA trends),
        # above 25 = too volatile (HA lag amplifies whipsaws)
        vix_ok = (vix_close > 12.0) & (vix_close < 25.0)

        # ── Session filter ────────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── Entry signals ─────────────────────────────────────────────────────
        # Bullish: strong HA green streak + price above VWAP + EMA bullish aligned
        buy_ce = (
            in_session
            & vix_ok
            & strong_green
            & (consec_green >= min_consec)
            & (close > vwap)
            & (ema_fast > ema_slow)
        )

        # Bearish: strong HA red streak + price below VWAP + EMA bearish aligned
        buy_pe = (
            in_session
            & vix_ok
            & strong_red
            & (consec_red >= min_consec)
            & (close < vwap)
            & (ema_fast < ema_slow)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,                    # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
