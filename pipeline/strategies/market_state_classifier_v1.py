"""market_state_classifier_v1 — NIFTY micro-regime classification strategy.

Classifies NIFTY into TRENDING (institutional TWAP/VWAP flow) or RANGING
(market-maker delta-hedging) states using ADX(24), Bollinger Band width
percentile, and ATR acceleration. Applies state-appropriate entry logic:
EMA crossover in trending state, BB-extreme fade in ranging state.

Hold time: 30-90 seconds. Flat in VOLATILE state (ATR spike).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_adx(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 24
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ADX, +DI, -DI via Wilder's smoothing."""
    n = len(close)
    adx = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)

    if n < period * 2 + 1:
        return adx, plus_di, minus_di

    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

        up = high[i] - high[i - 1]
        dn = low[i - 1] - low[i]
        if up > dn and up > 0.0:
            plus_dm[i] = up
        elif dn > up and dn > 0.0:
            minus_dm[i] = dn

    # Wilder's initial smoothed values = sum of first `period` bars (index 1..period)
    s_tr = np.zeros(n)
    s_pdm = np.zeros(n)
    s_mdm = np.zeros(n)

    s_tr[period] = np.sum(tr[1 : period + 1])
    s_pdm[period] = np.sum(plus_dm[1 : period + 1])
    s_mdm[period] = np.sum(minus_dm[1 : period + 1])

    for i in range(period + 1, n):
        s_tr[i] = s_tr[i - 1] - s_tr[i - 1] / period + tr[i]
        s_pdm[i] = s_pdm[i - 1] - s_pdm[i - 1] / period + plus_dm[i]
        s_mdm[i] = s_mdm[i - 1] - s_mdm[i - 1] / period + minus_dm[i]

    dx = np.zeros(n)
    for i in range(period, n):
        if s_tr[i] > 0.0:
            pdi = 100.0 * s_pdm[i] / s_tr[i]
            mdi = 100.0 * s_mdm[i] / s_tr[i]
            plus_di[i] = pdi
            minus_di[i] = mdi
            di_sum = pdi + mdi
            if di_sum > 0.0:
                dx[i] = 100.0 * abs(pdi - mdi) / di_sum

    # ADX = Wilder-smoothed DX; seed = mean of first `period` DX values
    start = period * 2
    if start >= n:
        return adx, plus_di, minus_di

    adx[start - 1] = np.mean(dx[period:start])
    for i in range(start, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx, plus_di, minus_di


def _compute_atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 12
) -> np.ndarray:
    """ATR via Wilder's smoothing."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    atr = np.zeros(n)
    if n < period:
        return atr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Standard EMA with multiplier 2/(period+1)."""
    result = np.zeros(len(arr))
    k = 2.0 / (period + 1)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = arr[i] * k + result[i - 1] * (1.0 - k)
    return result


class Strategy(BaseStrategy):
    name = "market_state_classifier_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — skip opening range absorption
    session_end_minutes = 920     # 15:20 IST
    max_lookback = 120            # 10 min warmup: ADX(24) needs 48 bars, BB pctile 120
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("adx_trend_threshold", 22.0, 18.0, 30.0),
            TunableParam("adx_range_threshold", 18.0, 12.0, 22.0),
            TunableParam("bb_width_pctile_threshold", 40.0, 25.0, 55.0),
            TunableParam("atr_accel_threshold", 0.6, 0.3, 0.9),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill NaN before converting to numpy
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        adx_thresh = params.get("adx_trend_threshold", 22.0)
        range_thresh = params.get("adx_range_threshold", 18.0)
        bb_pctile_thresh = params.get("bb_width_pctile_threshold", 40.0)
        atr_accel_thresh = params.get("atr_accel_threshold", 0.6)

        # ── VIX ──────────────────────────────────────────────────────────────
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

        # ── Session VWAP (cumulative from open each day) ──────────────────────
        typical = (high + low + close) / 3.0
        vwap = np.zeros(n)
        cum_tp_vol = 0.0
        cum_vol = 0.0
        for i in range(n):
            if i == 0 or day_id[i] != day_id[i - 1]:
                cum_tp_vol = typical[i] * volume[i]
                cum_vol = volume[i]
            else:
                cum_tp_vol += typical[i] * volume[i]
                cum_vol += volume[i]
            vwap[i] = cum_tp_vol / cum_vol if cum_vol > 0.0 else close[i]

        # ── ADX(24) — 2-minute regime classifier ─────────────────────────────
        adx, _pdi, _mdi = _compute_adx(high, low, close, period=24)

        # ── ATR acceleration: 1-min ATR vs same metric 60s ago ────────────────
        atr_12 = _compute_atr(high, low, close, period=12)
        atr_lag = 12  # 60 seconds back
        atr_accel = np.zeros(n)
        for i in range(atr_lag, n):
            if atr_12[i - atr_lag] > 0.0:
                atr_accel[i] = atr_12[i] / atr_12[i - atr_lag] - 1.0

        # ── Bollinger Bands (5-min = 60 bars) + BB width percentile rank ─────
        bb_period = 60
        bb_upper = np.zeros(n)
        bb_lower = np.zeros(n)
        bb_width = np.zeros(n)

        for i in range(bb_period - 1, n):
            w = close[i - bb_period + 1 : i + 1]
            m = float(np.mean(w))
            s = float(np.std(w))
            bb_upper[i] = m + 2.0 * s
            bb_lower[i] = m - 2.0 * s
            bb_width[i] = (4.0 * s) / m if m > 0.0 else 0.0

        # Rolling 60-bar percentile rank of BB width
        pctile_window = 60
        bb_width_pctile = np.zeros(n)
        for i in range(bb_period + pctile_window - 2, n):
            recent = bb_width[i - pctile_window + 1 : i + 1]
            cur = recent[-1]
            if cur > 0.0:
                bb_width_pctile[i] = float(np.sum(recent < cur)) / pctile_window * 100.0

        # ── EMA crossover (EMA12 = 1-min fast, EMA36 = 3-min slow) ──────────
        ema_12 = _ema(close, 12)
        ema_36 = _ema(close, 36)

        ema_cross_up = np.zeros(n, dtype=bool)
        ema_cross_down = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if ema_12[i] > ema_36[i] and ema_12[i - 1] <= ema_36[i - 1]:
                ema_cross_up[i] = True
            if ema_12[i] < ema_36[i] and ema_12[i - 1] >= ema_36[i - 1]:
                ema_cross_down[i] = True

        # ── State classification ──────────────────────────────────────────────
        # TRENDING: ADX above threshold AND no volatility spike
        state_trending = (adx > adx_thresh) & (atr_accel < atr_accel_thresh)
        # RANGING: ADX below threshold AND BB width in squeeze territory (pctile > 0 means initialized)
        state_ranging = (
            (adx < range_thresh)
            & (bb_width_pctile < bb_pctile_thresh)
            & (bb_width_pctile > 0.0)
        )

        # ── Session and VIX filters ───────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (
            time_min < self.session_end_minutes
        )
        vix_ok = vix_close < 25.0
        base_filter = in_session & vix_ok

        # ── Entry signals ─────────────────────────────────────────────────────
        # TRENDING state: EMA crossover aligned with VWAP direction
        trend_buy_ce = state_trending & ema_cross_up & (close > vwap) & base_filter
        trend_buy_pe = state_trending & ema_cross_down & (close < vwap) & base_filter

        # RANGING state: price at BB extreme, aligned with VWAP (mean-reversion)
        range_buy_ce = (
            state_ranging & (close <= bb_lower) & (close < vwap) & base_filter
        )
        range_buy_pe = (
            state_ranging & (close >= bb_upper) & (close > vwap) & base_filter
        )

        buy_ce = trend_buy_ce | range_buy_ce
        buy_pe = trend_buy_pe | range_buy_pe

        # ── Per-bar stop/target: two-tier based on state ──────────────────────
        # Ranging entries: tighter (3 stop / 5 target) — smaller reversion moves
        # Trending entries: wider  (5 stop / 8 target) — larger institutional legs
        stop_pts = np.zeros(n, dtype=np.float64)
        target_pts = np.zeros(n, dtype=np.float64)

        stop_pts[range_buy_ce | range_buy_pe] = 3.0
        target_pts[range_buy_ce | range_buy_pe] = 5.0
        stop_pts[trend_buy_ce | trend_buy_pe] = 5.0   # overwrites if overlap
        target_pts[trend_buy_ce | trend_buy_pe] = 8.0

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=stop_pts,
            target_points=target_pts,
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,         # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
