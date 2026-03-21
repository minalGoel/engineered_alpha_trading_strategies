"""gap_fill_fade_v1 — Gap Fill Fade on NIFTY/BANKNIFTY

When NIFTY/BANKNIFTY gaps 0.3-1.5% at open (overnight noise from global cues),
institutional VWAP/arrival-price desks immediately buy the dip (gap-down) or sell
into strength (gap-up) to improve their benchmark fills. We detect the first 5-second
reversal bar within the opening 25 minutes and ride the initial institutional impulse.
Hold 30-90 seconds, target 8 option pts, stop 5 option pts.
"""
from __future__ import annotations
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_rsi(prices: np.ndarray, period: int) -> np.ndarray:
    """Wilder's RSI. Returns array of same length, pre-filled with 50.0."""
    n = len(prices)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, n - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss > 0 else 100.0
        rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def _compute_gap_pct(day_ids: np.ndarray, opens: np.ndarray, closes: np.ndarray) -> np.ndarray:
    """Compute per-bar gap_pct = (day's first open - prev day's last close) / prev day's last close * 100.

    Returns array of same length as inputs; bars with no prior day data get 0.0.
    """
    n = len(day_ids)
    gap_pct = np.zeros(n)

    unique_days = sorted(set(day_ids.tolist()))
    day_first_open: dict[int, float] = {}
    day_last_close: dict[int, float] = {}

    for d in unique_days:
        mask = day_ids == d
        indices = np.where(mask)[0]
        day_first_open[d] = opens[indices[0]]
        day_last_close[d] = closes[indices[-1]]

    for i in range(n):
        d = day_ids[i]
        prev_d = d - 1
        if prev_d in day_last_close and d in day_first_open:
            prev_close = day_last_close[prev_d]
            if prev_close > 0.0:
                gap_pct[i] = (day_first_open[d] - prev_close) / prev_close * 100.0

    return gap_pct


class Strategy(BaseStrategy):
    name = "gap_fill_fade_v1"
    underlying = "BOTH"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 3
    max_lookback = 60             # 5-min warmup — enough for RSI(36) to stabilise

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_min_pct", 0.3, 0.2, 0.6),    # minimum gap size to trade
            TunableParam("gap_max_pct", 1.5, 1.0, 2.0),    # maximum gap size (above = information)
            TunableParam("rsi_low",    42.0, 35.0, 48.0),  # RSI threshold for gap-down exhaustion
            TunableParam("rsi_high",   58.0, 52.0, 65.0),  # RSI threshold for gap-up exhaustion
            TunableParam("vix_max",    22.0, 18.0, 28.0),  # max VIX to permit entry
            TunableParam("stop_pts",    5.0,  3.0,  8.0),
            TunableParam("target_pts",  8.0,  5.0, 14.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── Extract arrays ──────────────────────────────────────────────────
        close    = spot_df["close"].fill_null(strategy="forward").to_numpy()
        open_    = spot_df["open"].fill_null(strategy="forward").to_numpy()
        day_ids  = spot_df["day_id"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        # ── Parameters ──────────────────────────────────────────────────────
        gap_min  = params.get("gap_min_pct",  0.3)
        gap_max  = params.get("gap_max_pct",  1.5)
        rsi_low  = params.get("rsi_low",     42.0)
        rsi_high = params.get("rsi_high",    58.0)
        vix_max  = params.get("vix_max",     22.0)
        stop_pts   = params.get("stop_pts",   5.0)
        target_pts = params.get("target_pts", 8.0)

        # ── VIX (align to spot timestamps) ──────────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Gap pct (per bar, constant within a day) ─────────────────────────
        gap_pct = _compute_gap_pct(day_ids, open_, close)

        # ── RSI(36) on spot close — 3-min RSI ────────────────────────────────
        rsi_36 = _compute_rsi(close, 36)

        # ── Session and entry-window filters ─────────────────────────────────
        # Entry only in first 25 min of session (09:20-09:45 = 560-585)
        entry_window = (time_min >= self.session_start_minutes) & (time_min <= 585)
        in_session   = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # ── Reversal bar confirmation ────────────────────────────────────────
        bar_green = close > open_   # bullish 5-second bar (gap-down fade)
        bar_red   = close < open_   # bearish 5-second bar (gap-up fade)

        # ── VIX filter ───────────────────────────────────────────────────────
        vix_ok = vix_close < vix_max

        # ── Entry signals ────────────────────────────────────────────────────
        # buy_ce: fade the gap-down — expect NIFTY to revert toward prev close
        gap_down = (gap_pct < -gap_min) & (gap_pct > -gap_max)
        buy_ce = entry_window & gap_down & (rsi_36 < rsi_low) & bar_green & vix_ok

        # buy_pe: fade the gap-up — expect NIFTY to revert toward prev close
        gap_up = (gap_pct > gap_min) & (gap_pct < gap_max)
        buy_pe = entry_window & gap_up & (rsi_36 > rsi_high) & bar_red & vix_ok

        # No signal-based exits — rely on stop/target/time_stop
        sell_ce = np.zeros(n, dtype=bool)
        sell_pe = np.zeros(n, dtype=bool)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=sell_ce,
            sell_pe=sell_pe,
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,        # 90 seconds — capture first institutional impulse
            max_trades_per_day=self.max_trades_per_day,
        )
