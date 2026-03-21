"""Gap Fill Full v1 — NIFTY 5-second index options strategy.

Overnight gaps of 0.3-1.0% attract institutional VWAP-benchmarked mean-reversion flow.
The first intraday VWAP cross (typically 15-45 min after open) marks the inflection where
institutional accumulation overwhelms retail gap-chasing. Enter at the VWAP cross bar and
hold 30-120 seconds to capture the first 15-25 NIFTY spot point impulse.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder's RSI over a close price array. Returns array of same length."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 2:
        return rsi
    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss == 0.0:
        rsi[period] = 100.0
    else:
        rsi[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    for i in range(period, n - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0.0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


class Strategy(BaseStrategy):
    name = "gap_fill_full_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 3
    max_lookback = 60             # 5 min warmup to allow RSI_36 to stabilise

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_min_pct", 0.003, 0.002, 0.006),   # minimum gap magnitude (0.3%)
            TunableParam("gap_max_pct", 0.010, 0.007, 0.015),   # maximum gap magnitude (1.0%)
            TunableParam("vix_max", 20.0, 15.0, 25.0),          # VIX ceiling
            TunableParam("rsi_ce_max", 60.0, 50.0, 70.0),       # RSI upper bound for CE entry
            TunableParam("rsi_pe_min", 40.0, 30.0, 50.0),       # RSI lower bound for PE entry
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Pull params ──────────────────────────────────────────────────────
        gap_min = params.get("gap_min_pct", 0.003)
        gap_max = params.get("gap_max_pct", 0.010)
        vix_max = params.get("vix_max", 20.0)
        rsi_ce_max = params.get("rsi_ce_max", 60.0)
        rsi_pe_min = params.get("rsi_pe_min", 40.0)

        # ── Base arrays ───────────────────────────────────────────────────────
        close_np = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(float)
        open_np = spot_df["open"].fill_null(strategy="forward").to_numpy().astype(float)
        high_np = spot_df["high"].fill_null(strategy="forward").to_numpy().astype(float)
        low_np = spot_df["low"].fill_null(strategy="forward").to_numpy().astype(float)
        volume_np = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        day_id_np = spot_df["day_id"].to_numpy().astype(int)
        time_min = spot_df["time_minutes"].to_numpy().astype(int)

        # ── VIX ──────────────────────────────────────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy().astype(float)

        # ── Per-day gap_pct and intraday VWAP ────────────────────────────────
        gap_pct = np.zeros(n)
        vwap = np.zeros(n)

        sorted_days = sorted(set(day_id_np.tolist()))
        # Build map: day_id → last close of that day
        day_close_map: dict[int, float] = {}
        for d in sorted_days:
            mask = np.where(day_id_np == d)[0]
            if len(mask) > 0:
                day_close_map[d] = close_np[mask[-1]]

        for i, d in enumerate(sorted_days):
            mask = np.where(day_id_np == d)[0]
            if len(mask) == 0:
                continue

            # Compute gap_pct for this day (vs previous day's close)
            if i > 0:
                prev_d = sorted_days[i - 1]
                prev_close = day_close_map.get(prev_d, np.nan)
                if prev_close > 0 and not np.isnan(prev_close):
                    day_open = open_np[mask[0]]
                    gp = (day_open - prev_close) / prev_close
                    gap_pct[mask] = gp
                # else gap_pct stays 0 → won't trigger any signal

            # Compute intraday VWAP from session start (cumulative)
            tp = (high_np[mask] + low_np[mask] + close_np[mask]) / 3.0
            vol = volume_np[mask]
            cum_vol = np.cumsum(vol)
            cum_tpv = np.cumsum(tp * vol)
            denom = np.where(cum_vol == 0.0, 1.0, cum_vol)
            vwap[mask] = cum_tpv / denom

        # ── RSI(36) ≈ 3-minute RSI ───────────────────────────────────────────
        rsi_np = _rsi(close_np, 36)

        # ── VWAP cross detection (1-bar lookback) ────────────────────────────
        close_prev = np.roll(close_np, 1)
        vwap_prev = np.roll(vwap, 1)
        # Day boundary: first bar of each day has no valid "previous"
        first_bar_mask = np.zeros(n, dtype=bool)
        for d in sorted_days:
            mask = np.where(day_id_np == d)[0]
            if len(mask) > 0:
                first_bar_mask[mask[0]] = True

        bullish_cross = (
            (close_prev < vwap_prev) &
            (close_np >= vwap) &
            ~first_bar_mask
        )
        bearish_cross = (
            (close_prev > vwap_prev) &
            (close_np <= vwap) &
            ~first_bar_mask
        )

        # ── Session and VIX filters ──────────────────────────────────────────
        # Entry window: 09:20–10:30 (560–630 minutes)
        in_entry_window = (time_min >= 560) & (time_min <= 630)
        vix_ok = vix_close < vix_max

        # ── Gap direction flags ───────────────────────────────────────────────
        gap_down = (gap_pct <= -gap_min) & (gap_pct >= -gap_max)
        gap_up = (gap_pct >= gap_min) & (gap_pct <= gap_max)

        # ── Entry signals ─────────────────────────────────────────────────────
        # buy_ce: gap-down day, first bullish VWAP cross, RSI not overbought
        buy_ce = (
            gap_down &
            bullish_cross &
            (rsi_np < rsi_ce_max) &
            in_entry_window &
            vix_ok
        )

        # buy_pe: gap-up day, first bearish VWAP cross, RSI not oversold
        buy_pe = (
            gap_up &
            bearish_cross &
            (rsi_np > rsi_pe_min) &
            in_entry_window &
            vix_ok
        )

        # No simultaneous positions — buy_ce and buy_pe cannot both be True
        # (gap_down and gap_up are mutually exclusive by construction, so this
        # is already guaranteed, but be explicit)
        buy_ce = buy_ce & ~buy_pe

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, 4),
            target_points=np.full(n, 8),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,          # 120 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
