"""gap_fade_large_v1 — Large Gap Fade on NIFTY

Thesis: NIFTY large overnight gaps (>0.3%) driven by SGX/US markets cause retail
market-on-open orders to over-extend price in the gap direction during the first
30-90 seconds. Institutional market makers with hedged overnight exposure absorb
this flow and fade the extension. We detect stall (RSI exhaustion + 30s momentum
reversal) and enter the first leg of the fade.

Session: 09:15-10:30 IST only. Gap fades are a morning phenomenon.
"""
from __future__ import annotations
import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder RSI on 1D array. Returns array same length, NaN for first `period` bars."""
    n = len(close)
    rsi = np.full(n, np.nan)
    if n < period + 1:
        return rsi

    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Seed with simple average
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, n):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        if avg_loss == 0.0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


class Strategy(BaseStrategy):
    name = "gap_fade_large_v1"
    underlying = "NIFTY"
    # Session: 09:15–10:30 IST (555–630 minutes); gap fades are morning-only
    session_start_minutes = 555   # 09:15 IST
    session_end_minutes = 630     # 10:30 IST
    max_lookback = 36             # 3 min warmup — enough for RSI(12) + mom(6) to stabilize
    max_trades_per_day = 3        # rare setup: large gap days only

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_threshold", 0.003, 0.002, 0.008),  # 0.3% NIFTY gap
            TunableParam("ext_threshold", 0.0005, 0.0002, 0.002),  # 0.05% intra-session extension
            TunableParam("rsi_oversold", 30.0, 20.0, 38.0),
            TunableParam("rsi_overbought", 70.0, 62.0, 80.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        open_ = spot_df["open"].fill_null(strategy="forward").to_numpy()
        day_id = spot_df["day_id"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        gap_threshold = params.get("gap_threshold", 0.003)
        ext_threshold = params.get("ext_threshold", 0.0005)
        rsi_oversold = params.get("rsi_oversold", 30.0)
        rsi_overbought = params.get("rsi_overbought", 70.0)

        # ── Compute day-level gap_pct and day_open per bar ──────────────────────
        # day_open: first bar's open for each day_id
        # prev_close: last bar's close for (day_id - 1)
        day_open = np.zeros(n)
        gap_pct = np.zeros(n)

        # Build lookup: day_id → first open, last close
        unique_days = np.unique(day_id)
        day_first_open: dict[int, float] = {}
        day_last_close: dict[int, float] = {}
        for d in unique_days:
            mask = day_id == d
            idxs = np.where(mask)[0]
            day_first_open[int(d)] = float(open_[idxs[0]])
            day_last_close[int(d)] = float(close[idxs[-1]])

        for i in range(n):
            d = int(day_id[i])
            dopen = day_first_open.get(d, 0.0)
            day_open[i] = dopen
            prev_d = d - 1
            if prev_d in day_last_close and dopen > 0:
                gap_pct[i] = (dopen - day_last_close[prev_d]) / day_last_close[prev_d]
            else:
                gap_pct[i] = 0.0  # no gap on first day

        # ── Extension from day open (intra-session drift in gap direction) ───────
        extension = np.zeros(n)
        valid_open = day_open > 0
        extension[valid_open] = (close[valid_open] - day_open[valid_open]) / day_open[valid_open]

        # ── RSI(12) — 12 bars = 60 seconds, detects short-term exhaustion ───────
        rsi = _compute_rsi(close, 12)
        # Forward-fill NaN (warmup) with neutral 50.0
        for i in range(n):
            if np.isnan(rsi[i]):
                rsi[i] = 50.0

        # ── 30-second momentum (mom_6) ────────────────────────────────────────────
        mom_6 = np.zeros(n)
        for i in range(6, n):
            if close[i - 6] > 0:
                mom_6[i] = (close[i] - close[i - 6]) / close[i - 6]

        # ── VIX filter: VIX < 25 ─────────────────────────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        vix_ok = vix_close < 25.0

        # ── Session filter: 09:15–10:30 IST ─────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # ── Gap-up day: price extends further above open → fade with PE ──────────
        gap_up_day = gap_pct > gap_threshold
        extending_up = extension > ext_threshold
        exhausted_up = rsi > rsi_overbought
        momentum_fading_up = mom_6 < 0.0

        buy_pe = in_session & vix_ok & gap_up_day & extending_up & exhausted_up & momentum_fading_up

        # ── Gap-down day: price extends further below open → fade with CE ─────────
        gap_down_day = gap_pct < -gap_threshold
        extending_down = extension < -ext_threshold
        exhausted_down = rsi < rsi_oversold
        momentum_fading_down = mom_6 > 0.0

        buy_ce = in_session & vix_ok & gap_down_day & extending_down & exhausted_down & momentum_fading_down

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            # Stop: 4 pts = ~8 NIFTY spot pts at delta 0.5
            # If extension deepens by 4 option pts, institutional flow is joining — exit
            stop_points=np.full(n, 4),
            # Target: 7 pts = ~14 NIFTY spot pts
            # First fade leg from an over-extended gap covers 15-25 spot pts; 7 pts captures ~50%
            target_points=np.full(n, 8),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
