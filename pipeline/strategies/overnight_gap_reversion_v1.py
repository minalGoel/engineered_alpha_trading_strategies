"""overnight_gap_reversion_v1 — NIFTY overnight gap reversion strategy.

When NIFTY opens with a 0.3-1.0% gap vs previous close (driven by SGX Nifty
futures and thin pre-market activity), VWAP-benchmarked institutional desks
initiate counter-directional orders within the first 30 minutes to close the
mispricing against their intraday VWAP benchmark.

We enter in the direction of the expected institutional gap-fill flow when:
  - Gap is in the 0.3-1.0% range (not noise, not macro/news-driven)
  - 1-min RSI confirms short-term exhaustion of the opening spike
  - Price is still away from session VWAP (gap not yet filled)
  - VIX is calm (< 22)
  - Time is in the 09:20-09:50 window (peak gap-fill pressure)

Hold 30-120 seconds (6-24 bars), targeting ~14 NIFTY spot points (~7 option pts).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder RSI on a numpy array. Returns 50.0 for warmup bars."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi

    delta = np.empty(n)
    delta[0] = 0.0
    delta[1:] = close[1:] - close[:-1]

    gain = np.where(delta > 0.0, delta, 0.0)
    loss = np.where(delta < 0.0, -delta, 0.0)

    # Seed with simple average of first `period` moves
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[period] = np.mean(gain[1:period + 1])
    avg_loss[period] = np.mean(loss[1:period + 1])

    # Wilder smoothing for subsequent bars
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period

    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


def _compute_session_vwap(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    day_ids: np.ndarray,
) -> np.ndarray:
    """Cumulative session VWAP, reset at each day boundary."""
    n = len(close)
    vwap = np.zeros(n)
    cum_tpv = 0.0
    cum_vol = 0.0
    for i in range(n):
        if i == 0 or day_ids[i] != day_ids[i - 1]:
            cum_tpv = 0.0
            cum_vol = 0.0
        tp = (high[i] + low[i] + close[i]) / 3.0
        cum_tpv += tp * volume[i]
        cum_vol += volume[i]
        vwap[i] = cum_tpv / (cum_vol if cum_vol > 0.0 else 1.0)
    return vwap


def _compute_gap_pct(
    opens_bar: np.ndarray,
    close: np.ndarray,
    day_ids: np.ndarray,
) -> np.ndarray:
    """Per-bar gap_pct = (day_open - prev_day_close) / prev_day_close * 100.

    Uses the first bar's open as day_open and the previous day's last close as
    prev_day_close. First day in dataset has no prev_close → gap_pct = 0.
    """
    n = len(close)
    unique_days = np.unique(day_ids)

    # Pass 1: record each day's first open and last close
    day_first_open: dict[int, float] = {}
    day_last_close: dict[int, float] = {}
    for d in unique_days:
        mask = day_ids == d
        idx = np.where(mask)[0]
        day_first_open[int(d)] = float(opens_bar[idx[0]])
        day_last_close[int(d)] = float(close[idx[-1]])

    # Pass 2: broadcast gap_pct to every bar
    gap_pct = np.zeros(n)
    for i in range(n):
        d = int(day_ids[i])
        d_open = day_first_open[d]
        prev_d = d - 1
        if prev_d in day_last_close:
            prev_close = day_last_close[prev_d]
            if prev_close > 0.0:
                gap_pct[i] = (d_open - prev_close) / prev_close * 100.0
    return gap_pct


class Strategy(BaseStrategy):
    name = "overnight_gap_reversion_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST — skip first 5 min opening noise
    session_end_minutes = 590     # 09:50 IST — gap fill pressure window
    max_trades_per_day = 2
    max_lookback = 36             # 3 min warmup (RSI-12 needs 12 bars + seed)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_min_pct",   0.3,  0.2,  0.5),
            TunableParam("gap_max_pct",   1.0,  0.7,  1.5),
            TunableParam("rsi_oversold",  40.0, 28.0, 50.0),
            TunableParam("rsi_overbought", 60.0, 50.0, 72.0),
            TunableParam("stop_pts",       4.0,  2.0,  8.0),
            TunableParam("target_pts",     8.0,  4.0, 12.0),
            TunableParam("vix_max",       22.0, 15.0, 28.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Raw arrays (forward-fill NaN in Polars before numpy) ──────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high  = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low   = spot_df["low"].fill_null(strategy="forward").to_numpy()
        opens_bar = spot_df["open"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        day_ids  = spot_df["day_id"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        # ── Parameters ────────────────────────────────────────────────────────
        gap_min  = params.get("gap_min_pct",    0.3)
        gap_max  = params.get("gap_max_pct",    1.0)
        rsi_os   = params.get("rsi_oversold",  40.0)
        rsi_ob   = params.get("rsi_overbought", 60.0)
        stop_pts = params.get("stop_pts",        4.0)
        tgt_pts  = params.get("target_pts",      8.0)
        vix_max  = params.get("vix_max",        22.0)

        # ── VIX filter ────────────────────────────────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Gap percentage (daily, broadcast to all bars) ─────────────────────
        gap_pct = _compute_gap_pct(opens_bar, close, day_ids)

        # ── Session VWAP ──────────────────────────────────────────────────────
        vwap = _compute_session_vwap(high, low, close, volume, day_ids)

        # ── 1-minute RSI (12 bars × 5s = 60s) ────────────────────────────────
        rsi = _compute_rsi(close, 12)

        # ── Filters ──────────────────────────────────────────────────────────
        in_window = (
            (time_min >= self.session_start_minutes) &
            (time_min <  self.session_end_minutes)
        )
        low_vix = vix_close < vix_max

        # ── Gap conditions ────────────────────────────────────────────────────
        # gap-down: NIFTY opened below prev close → expect reversion up → buy CE
        gap_down = (gap_pct < -gap_min) & (gap_pct > -gap_max)
        buy_ce = (
            in_window &
            low_vix &
            gap_down &
            (rsi < rsi_os) &        # 1-min exhaustion of initial down spike
            (close < vwap)          # price still below benchmark (gap not filled)
        )

        # gap-up: NIFTY opened above prev close → expect reversion down → buy PE
        gap_up = (gap_pct > gap_min) & (gap_pct < gap_max)
        buy_pe = (
            in_window &
            low_vix &
            gap_up &
            (rsi > rsi_ob) &        # 1-min exhaustion of initial up spike
            (close > vwap)          # price still above benchmark (gap not filled)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, tgt_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,           # 120 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
