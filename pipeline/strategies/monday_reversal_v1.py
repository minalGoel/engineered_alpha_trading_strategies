"""monday_reversal_v1 — Monday Morning Institutional Flow Reversal

When NIFTY falls >0.5% on Friday, DII/SIP rebalancing buy programs create discrete
30-60 second upward micro-bursts on Monday morning as institutional limit orders absorb
supply wave by wave. We detect each burst as RSI(8) crosses above 45 while price
reclaims VWAP, entering buy CE for 30-90 second holds. The symmetric bearish version
trades Friday rallies that get faded by FII profit-taking on Monday.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder's RSI. Returns array of length n, filled with 50.0 until warm."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi

    delta = np.zeros(n)
    delta[1:] = close[1:] - close[:-1]

    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    # Initialise with SMA of first `period` gains/losses
    avg_gain = float(np.mean(gain[1:period + 1]))
    avg_loss = float(np.mean(loss[1:period + 1]))

    if avg_loss == 0.0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        if avg_loss == 0.0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def _compute_vwap(close: np.ndarray, volume: np.ndarray,
                  day_id: np.ndarray) -> np.ndarray:
    """Intraday VWAP, reset at each session boundary."""
    n = len(close)
    vwap = np.zeros(n)
    for uid in np.unique(day_id):
        mask = day_id == uid
        idx = np.where(mask)[0]
        c = close[mask]
        v = volume[mask].astype(np.float64)
        cv_cum = np.cumsum(c * v)
        v_cum = np.cumsum(v)
        v_cum = np.where(v_cum == 0.0, 1.0, v_cum)
        vwap[idx] = cv_cum / v_cum
    return vwap


def _prior_session_returns(spot_df: pl.DataFrame) -> np.ndarray:
    """For each bar, return the prior session's (open-to-close) return.

    For Monday bars the prior session is Friday (the last recorded prior day_id).
    For non-Monday bars the value is still computed but ignored by the session filter.
    """
    # Build per-session summary: first open and last close, sorted by day_id
    daily = (
        spot_df.group_by("day_id")
        .agg([
            pl.col("open").first().alias("day_open"),
            pl.col("close").last().alias("day_close"),
        ])
        .sort("day_id")
    )

    day_ids_arr = daily["day_id"].to_numpy()
    day_opens_arr = daily["day_open"].to_numpy().astype(np.float64)
    day_closes_arr = daily["day_close"].to_numpy().astype(np.float64)

    # prior_return_map: day_id -> that session's PRIOR session return
    prior_return_map: dict[int, float] = {}
    for i, d in enumerate(day_ids_arr):
        if i == 0:
            prior_return_map[int(d)] = 0.0
        else:
            prev_o = day_opens_arr[i - 1]
            prev_c = day_closes_arr[i - 1]
            if prev_o != 0.0:
                prior_return_map[int(d)] = float((prev_c - prev_o) / prev_o)
            else:
                prior_return_map[int(d)] = 0.0

    day_id_bars = spot_df["day_id"].to_numpy()
    return np.array([prior_return_map.get(int(d), 0.0) for d in day_id_bars])


class Strategy(BaseStrategy):
    name = "monday_reversal_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 660     # 11:00 IST — Monday morning window only
    max_trades_per_day = 10
    max_lookback = 24             # 2 minutes (fast RSI warmup)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("friday_return_threshold", 0.005, 0.003, 0.015),
            TunableParam("vix_max", 22.0, 18.0, 28.0),
            TunableParam("rsi_buy_threshold", 45.0, 38.0, 52.0),
            TunableParam("rsi_sell_threshold", 55.0, 48.0, 62.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Raw arrays (forward-fill NaN) ──────────────────────────────────
        close = (
            spot_df["close"].cast(pl.Float64).fill_nan(None).forward_fill()
            .to_numpy()
        )
        volume = (
            spot_df["volume"].cast(pl.Float64).fill_nan(None).forward_fill()
            .fill_null(0.0).to_numpy()
        )
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # Day of week: ISO-8601 — Monday=1, Sunday=7
        # Using session_date (Date type) for reliability across Polars versions
        weekday = spot_df["session_date"].dt.weekday().to_numpy()
        is_monday = weekday == 1  # ISO Monday

        # ── Prior session return (Friday's index move) ──────────────────────
        prior_day_return = _prior_session_returns(spot_df)

        # ── Intraday VWAP (reset each session) ─────────────────────────────
        vwap = _compute_vwap(close, volume, day_id)

        # ── RSI(8) — 40-second fast momentum ───────────────────────────────
        rsi_8 = _compute_rsi(close, 8)

        # ── VIX (join asof onto spot bars) ──────────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select([
                    "datetime",
                    pl.col("close").cast(pl.Float64).fill_nan(None)
                    .forward_fill().alias("vix_close"),
                ]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Parameters ──────────────────────────────────────────────────────
        friday_thresh = float(params.get("friday_return_threshold", 0.005))
        vix_max = float(params.get("vix_max", 22.0))
        rsi_buy = float(params.get("rsi_buy_threshold", 45.0))
        rsi_sell = float(params.get("rsi_sell_threshold", 55.0))

        # ── Session filter ───────────────────────────────────────────────────
        in_session = (
            is_monday
            & (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── Directional filters ──────────────────────────────────────────────
        friday_down = prior_day_return < -friday_thresh   # Friday NIFTY fell
        friday_up = prior_day_return > friday_thresh      # Friday NIFTY rose
        low_vix = vix_close < vix_max

        # Price relative to VWAP
        above_vwap = close > vwap
        below_vwap = close < vwap

        # RSI conditions
        rsi_rising = rsi_8 > rsi_buy    # momentum turning up
        rsi_falling = rsi_8 < rsi_sell  # momentum turning down

        # ── Entry signals ────────────────────────────────────────────────────
        # Friday down → expect Monday morning institutional buying bursts → buy CE
        buy_ce = in_session & low_vix & friday_down & above_vwap & rsi_rising

        # Friday up → expect Monday morning institutional profit-taking → buy PE
        buy_pe = in_session & low_vix & friday_up & below_vwap & rsi_falling

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            # 3 pts stop = ~6 NIFTY spot pts; if institutional bid absent, retrace
            # exceeds this within 30s and we exit.
            # 5 pts target = ~10 NIFTY spot pts; captures body of one institutional
            # buying wave at delta ~0.5 (1:1.67 R:R).
            stop_points=np.full(n, 3.0),
            target_points=np.full(n, 5.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
