"""
weighted_indicator_v1 — Dynamic Performance-Weighted Indicator Composite on NIFTY

Five indicators (VWAP position, RSI(24), MACD(24/52/18), EMA(18/42) crossover,
OBV-vs-SMA(60)) are combined via softmax-weighted composite score, where each
indicator's weight is proportional to its recent 50-signal directional hit rate.
The meta-learning layer shifts weight toward whichever indicator type (momentum
or mean-reversion) is currently predictive for NIFTY's intraday regime.

Entry: weighted_score > threshold AND ≥3 indicators agree AND VWAP aligned.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average. Returns NaN for bars before first valid value."""
    result = np.full(len(arr), np.nan)
    k = 2.0 / (period + 1.0)
    start = 0
    while start < len(arr) and np.isnan(arr[start]):
        start += 1
    if start >= len(arr):
        return result
    result[start] = arr[start]
    for i in range(start + 1, len(arr)):
        result[i] = arr[i] * k + result[i - 1] * (1.0 - k)
    return result


def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder's RSI. Returns 50.0 for warmup bars."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n <= period:
        return rsi

    delta = np.zeros(n)
    delta[1:] = close[1:] - close[:-1]
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[period] = np.mean(gain[1 : period + 1])
    avg_loss[period] = np.mean(loss[1 : period + 1])

    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period

    for i in range(period, n):
        if avg_loss[i] == 0.0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


def _rolling_hit_rates(
    indicators: list[np.ndarray],
    next_ret_sign: np.ndarray,
    window: int,
    min_signals: int = 10,
) -> np.ndarray:
    """Compute rolling hit rate for each indicator over last `window` non-neutral signals.

    At bar i, hit_rate[k, i] = fraction of last `window` bars where indicator k
    was non-zero and correctly predicted sign(close[i+1] - close[i]).

    Uses only information available at bar i (outcomes of bars 0..i-1).
    """
    K = len(indicators)
    n = len(next_ret_sign)
    hit_rates = np.full((K, n), 0.5)

    for k in range(K):
        ind_k = indicators[k]
        history: deque[float] = deque(maxlen=window)

        for i in range(1, n):
            # Record outcome of bar i-1: we now know sign(close[i] - close[i-1])
            prev_sig = ind_k[i - 1]
            if prev_sig != 0.0:
                hit = 1.0 if next_ret_sign[i - 1] == prev_sig else 0.0
                history.append(hit)

            if len(history) >= min_signals:
                hit_rates[k, i] = float(np.mean(history))

    return hit_rates


class Strategy(BaseStrategy):
    name = "weighted_indicator_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — allow 15 min for VWAP establishment
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 7
    max_lookback = 360            # 30 min warmup for MACD(52) + hit-rate seed

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("score_threshold", 0.55, 0.40, 0.75),
            TunableParam("min_indicators", 3.0, 2.0, 5.0),
            TunableParam("alpha", 2.0, 0.5, 4.0),
            TunableParam("stop_pts", 5.0, 3.0, 9.0),
            TunableParam("target_pts", 8.0, 5.0, 14.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract arrays ────────────────────────────────────────────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(float)
        high = spot_df["high"].fill_null(strategy="forward").to_numpy().astype(float)
        low = spot_df["low"].fill_null(strategy="forward").to_numpy().astype(float)
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        score_threshold = float(params.get("score_threshold", 0.55))
        min_indicators = int(params.get("min_indicators", 3))
        alpha = float(params.get("alpha", 2.0))
        stop_pts = float(params.get("stop_pts", 5.0))
        target_pts = float(params.get("target_pts", 8.0))

        # ── Indicator 1: Session VWAP position ───────────────────────────────
        # sign(close - vwap); +1 above VWAP (bullish bias), -1 below
        tp = (high + low + close) / 3.0
        vwap = np.zeros(n)
        cum_tp_vol = 0.0
        cum_vol = 0.0
        for i in range(n):
            if i == 0 or day_id[i] != day_id[i - 1]:
                cum_tp_vol = tp[i] * volume[i]
                cum_vol = volume[i]
            else:
                cum_tp_vol += tp[i] * volume[i]
                cum_vol += volume[i]
            vwap[i] = cum_tp_vol / cum_vol if cum_vol > 0.0 else close[i]

        ind1 = np.sign(close - vwap)  # {-1, 0, +1}

        # ── Indicator 2: RSI(24) thresholds ──────────────────────────────────
        # 24 bars × 5s = 2-min RSI; thresholds 40/60 for smoother NIFTY index
        rsi_vals = _rsi(close, period=24)
        ind2 = np.where(rsi_vals < 40.0, 1.0, np.where(rsi_vals > 60.0, -1.0, 0.0))

        # ── Indicator 3: MACD(24,52,18) histogram sign ───────────────────────
        # 2× scaling of MACD(12,26,9) from 1-min → 5s bars
        ema_fast = _ema(close, 24)
        ema_slow = _ema(close, 52)
        macd_line = np.where(
            np.isnan(ema_fast) | np.isnan(ema_slow), 0.0, ema_fast - ema_slow
        )
        macd_sig = _ema(macd_line, 18)
        macd_sig_filled = np.where(np.isnan(macd_sig), 0.0, macd_sig)
        macd_hist = macd_line - macd_sig_filled
        ind3 = np.sign(macd_hist)

        # ── Indicator 4: EMA(18,42) crossover sign ───────────────────────────
        ema18 = _ema(close, 18)
        ema42 = _ema(close, 42)
        ema_diff = np.where(
            np.isnan(ema18) | np.isnan(ema42), 0.0, ema18 - ema42
        )
        ind4 = np.sign(ema_diff)

        # ── Indicator 5: OBV vs SMA(OBV, 60) ────────────────────────────────
        # OBV resets at each new trading day; 60-bar (5-min) SMA
        obv = np.zeros(n)
        for i in range(1, n):
            if day_id[i] != day_id[i - 1]:
                # Reset at day boundary
                if close[i] > close[i - 1]:
                    obv[i] = volume[i]
                elif close[i] < close[i - 1]:
                    obv[i] = -volume[i]
                # else obv[i] = 0
            else:
                if close[i] > close[i - 1]:
                    obv[i] = obv[i - 1] + volume[i]
                elif close[i] < close[i - 1]:
                    obv[i] = obv[i - 1] - volume[i]
                else:
                    obv[i] = obv[i - 1]

        obv_sma_period = 60
        obv_sma = np.zeros(n)
        for i in range(obv_sma_period - 1, n):
            obv_sma[i] = np.mean(obv[i - obv_sma_period + 1 : i + 1])
        ind5 = np.sign(obv - obv_sma)

        # ── Dynamic performance weighting ─────────────────────────────────────
        indicators = [ind1, ind2, ind3, ind4, ind5]
        K = len(indicators)

        # next-bar return sign: at bar i, sign(close[i+1] - close[i])
        next_ret_sign = np.zeros(n)
        next_ret_sign[:-1] = np.sign(close[1:] - close[:-1])

        hit_rates = _rolling_hit_rates(
            indicators, next_ret_sign, window=50, min_signals=10
        )

        # ── Compute weighted composite score ──────────────────────────────────
        indicators_arr = np.array(indicators)  # (K, n)
        weighted_score = np.zeros(n)
        count_agree = np.zeros(n, dtype=np.int32)

        for i in range(n):
            hr = hit_rates[:, i]
            # Softmax weights
            ex = np.exp(alpha * hr)
            ex_sum = ex.sum()
            weights = ex / ex_sum if ex_sum > 0.0 else np.ones(K) / K

            score = float(np.dot(weights, indicators_arr[:, i]))
            weighted_score[i] = score

            # Count indicators agreeing with score direction
            sig_dir = 1.0 if score > 0.0 else -1.0
            count_agree[i] = int(np.sum(indicators_arr[:, i] == sig_dir))

        # ── Session filter ────────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (
            time_min < self.session_end_minutes
        )

        # ── Entry signals ─────────────────────────────────────────────────────
        # buy_ce: bullish — high weighted score, 3+ indicators agree, above VWAP
        buy_ce = (
            in_session
            & (weighted_score > score_threshold)
            & (count_agree >= min_indicators)
            & (ind1 > 0)
        )
        # buy_pe: bearish — low weighted score, 3+ indicators agree, below VWAP
        buy_pe = (
            in_session
            & (weighted_score < -score_threshold)
            & (count_agree >= min_indicators)
            & (ind1 < 0)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
