"""partial_gap_fill_scalp_v1 — NIFTY gap partial fill at open.

Thesis: When NIFTY gaps 0.5–1.5% at open, institutional VWAP algorithms
benchmarked to the previous close face adverse fills and start trading back
toward the gap midpoint within the first few minutes. We enter on the first
5-second bar where RSI(24) exhaustion is confirmed (oversold after gap-down,
overbought after gap-up) and the bar shows a reversal candle, targeting
~16 spot points of reversion (8 option pts) with a 5-pt stop.

Entry window: 09:20–09:45 IST only.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder's RSI. Returns array of same length, first `period` bars = 50.0."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi

    delta = np.diff(close)  # length n-1
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    # Seed: simple average over first `period` deltas
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0.0:
        rsi[period] = 100.0
    else:
        rsi[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    # Wilder smoothing for remaining bars
    for i in range(period, n - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0.0:
            rsi[i + 1] = 100.0
        else:
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    return rsi


class Strategy(BaseStrategy):
    name = "partial_gap_fill_scalp_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST — entry window open
    session_end_minutes = 585     # 09:45 IST — entry window close
    max_trades_per_day = 3
    max_lookback = 48             # 4-min warmup — enough for RSI(24) to seed

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_min_pct", 0.5, 0.3, 0.8),
            TunableParam("gap_max_pct", 1.5, 1.0, 2.5),
            TunableParam("rsi_ob", 70.0, 65.0, 80.0),
            TunableParam("rsi_os", 30.0, 20.0, 35.0),
            TunableParam("vix_max", 22.0, 16.0, 28.0),
            TunableParam("stop_pts", 5.0, 3.0, 8.0),
            TunableParam("target_pts", 8.0, 5.0, 15.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict,
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Raw arrays ──────────────────────────────────────────────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        open_ = spot_df["open"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # ── Params ──────────────────────────────────────────────────────────
        gap_min = params.get("gap_min_pct", 0.5)
        gap_max = params.get("gap_max_pct", 1.5)
        rsi_ob = params.get("rsi_ob", 70.0)
        rsi_os = params.get("rsi_os", 30.0)
        vix_max = params.get("vix_max", 22.0)
        stop_pts = params.get("stop_pts", 5.0)
        target_pts = params.get("target_pts", 8.0)

        # ── VIX filter ──────────────────────────────────────────────────────
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

        # ── RSI(24) — 2-minute exhaustion signal ────────────────────────────
        rsi_24 = _compute_rsi(close, 24)

        # ── Per-day gap computation ─────────────────────────────────────────
        # day_open: first bar's open for each day_id
        # prev_day_close: last bar's close from the immediately preceding day_id
        unique_days = np.unique(day_id)

        day_open_map: dict[int, float] = {}
        day_prev_close_map: dict[int, float] = {}

        for i, d in enumerate(unique_days):
            mask = day_id == d
            idx = np.where(mask)[0]
            day_open_map[d] = float(open_[idx[0]])
            if i == 0:
                day_prev_close_map[d] = np.nan
            else:
                prev_d = unique_days[i - 1]
                prev_mask = day_id == prev_d
                prev_idx = np.where(prev_mask)[0]
                day_prev_close_map[d] = float(close[prev_idx[-1]])

        # Broadcast to per-bar arrays
        day_open_arr = np.array(
            [day_open_map[int(d)] for d in day_id], dtype=np.float64
        )
        prev_close_arr = np.array(
            [day_prev_close_map.get(int(d), np.nan) for d in day_id],
            dtype=np.float64,
        )

        # Gap percent: positive = gapped up, negative = gapped down
        valid_prev = prev_close_arr > 0
        gap_pct = np.where(
            valid_prev,
            (day_open_arr - prev_close_arr) / prev_close_arr * 100.0,
            0.0,
        )

        # Fraction of gap already moved toward partial fill
        gap_size = np.abs(day_open_arr - prev_close_arr)
        # For gap-down (CE trade): positive fraction = price moved UP from open
        gap_moved_up = np.where(
            gap_size > 0,
            (close - day_open_arr) / gap_size,
            0.0,
        )
        # For gap-up (PE trade): positive fraction = price moved DOWN from open
        gap_moved_dn = np.where(
            gap_size > 0,
            (day_open_arr - close) / gap_size,
            0.0,
        )

        # ── Boolean masks ────────────────────────────────────────────────────
        in_entry_window = (
            (time_min >= self.session_start_minutes)
            & (time_min <= self.session_end_minutes)
        )
        vix_ok = vix_close < vix_max

        is_green = close > open_
        is_red = close < open_

        # Previous bar's RSI (for "just crossed" detection)
        rsi_prev = np.empty(n)
        rsi_prev[0] = 50.0
        rsi_prev[1:] = rsi_24[:-1]

        # Gap conditions
        gap_down_ok = (gap_pct < -gap_min) & (gap_pct > -gap_max) & valid_prev
        gap_up_ok = (gap_pct > gap_min) & (gap_pct < gap_max) & valid_prev

        # Price not already significantly into the fill (< 30% moved)
        not_too_far_up = gap_moved_up < 0.30
        not_too_far_dn = gap_moved_dn < 0.30

        # ── Entry signals ────────────────────────────────────────────────────
        # BUY CE: NIFTY gapped down → expect partial fill upward
        buy_ce = (
            in_entry_window
            & vix_ok
            & gap_down_ok
            & (rsi_prev < rsi_os)   # prior bar was oversold
            & is_green               # current bar turned green = reversal start
            & not_too_far_up
        )

        # BUY PE: NIFTY gapped up → expect partial fill downward
        buy_pe = (
            in_entry_window
            & vix_ok
            & gap_up_ok
            & (rsi_prev > rsi_ob)   # prior bar was overbought
            & is_red                 # current bar turned red = reversal start
            & not_too_far_dn
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=60,           # 5-minute max hold
            max_trades_per_day=self.max_trades_per_day,
        )
