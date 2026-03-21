"""multi_indicator_consensus_v1 — NIFTY 5s index options strategy.

Mechanism: Detects pullback exhaustion within a trending NIFTY session by requiring
4+ of 5 independent indicators (VWAP position, RSI-3min, MACD-2/4min, Stoch-2min,
volume ROC) to agree simultaneously. The 2-bar (10s) persistence filter eliminates
noise spikes, leaving only genuine multi-timeframe convergence setups where
institutional algo re-entry is likely within 15-120 seconds.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average (Wilder-style k=2/(period+1))."""
    n = len(values)
    result = np.empty(n)
    k = 2.0 / (period + 1)
    result[0] = values[0]
    for i in range(1, n):
        result[i] = values[i] * k + result[i - 1] * (1.0 - k)
    return result


def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    """RSI with Wilder smoothing. Returns 50.0 for warm-up bars."""
    n = len(close)
    result = np.full(n, 50.0)
    if n < period + 1:
        return result
    delta = np.empty(n)
    delta[0] = 0.0
    delta[1:] = close[1:] - close[:-1]
    gains = np.where(delta > 0.0, delta, 0.0)
    losses = np.where(delta < 0.0, -delta, 0.0)
    avg_gain = np.mean(gains[1: period + 1])
    avg_loss = np.mean(losses[1: period + 1])
    for i in range(period, n):
        if i > period:
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0.0:
            result[i] = 100.0
        else:
            result[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return result


def _stoch_k(high: np.ndarray, low: np.ndarray, close: np.ndarray,
             period: int, smooth: int) -> np.ndarray:
    """Stochastic %K smoothed with SMA(smooth). Returns 50.0 for warm-up bars."""
    n = len(close)
    raw_k = np.full(n, 50.0)
    for i in range(period - 1, n):
        h = np.max(high[i - period + 1: i + 1])
        lo = np.min(low[i - period + 1: i + 1])
        if h > lo:
            raw_k[i] = 100.0 * (close[i] - lo) / (h - lo)
    result = np.full(n, 50.0)
    for i in range(n):
        start = max(0, i - smooth + 1)
        result[i] = np.mean(raw_k[start: i + 1])
    return result


class Strategy(BaseStrategy):
    name = "multi_indicator_consensus_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — 10-min buffer after open for warm-up
    session_end_minutes = 920     # 15:20 IST — flatten before expiry-day cutoff
    max_trades_per_day = 6
    max_lookback = 156            # 13 min = 52 bars (MACD slow EMA) + 104-bar buffer

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("consensus_threshold", 4.0, 3.0, 5.0),
            TunableParam("rsi_oversold",  40.0, 30.0, 50.0),
            TunableParam("rsi_overbought", 60.0, 50.0, 70.0),
            TunableParam("stoch_oversold", 30.0, 20.0, 45.0),
            TunableParam("stoch_overbought", 70.0, 55.0, 80.0),
            TunableParam("vroc_threshold", 50.0, 20.0, 100.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # --- Extract arrays (forward-fill NaN before to_numpy) ---
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high  = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low   = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = (
            spot_df["volume"]
            .cast(pl.Float64)
            .fill_null(0.0)
            .to_numpy()
        )
        time_min = spot_df["time_minutes"].to_numpy()
        day_id   = spot_df["day_id"].to_numpy()

        # --- Parameters ---
        consensus_threshold = int(params.get("consensus_threshold", 4))
        rsi_oversold        = params.get("rsi_oversold", 40.0)
        rsi_overbought      = params.get("rsi_overbought", 60.0)
        stoch_oversold      = params.get("stoch_oversold", 30.0)
        stoch_overbought    = params.get("stoch_overbought", 70.0)
        vroc_threshold      = params.get("vroc_threshold", 50.0)

        # ── Indicator 1: Session VWAP (cumulative, resets each day) ──────────
        vwap = np.empty(n)
        cum_pv  = 0.0
        cum_vol = 0.0
        prev_day = -1
        for i in range(n):
            if day_id[i] != prev_day:
                cum_pv  = close[i] * volume[i]
                cum_vol = volume[i]
                prev_day = day_id[i]
            else:
                cum_pv  += close[i] * volume[i]
                cum_vol += volume[i]
            vwap[i] = cum_pv / cum_vol if cum_vol > 0.0 else close[i]

        # VWAP signal: +1 above, -1 below, 0 at
        vwap_signal = np.where(close > vwap, 1,
                      np.where(close < vwap, -1, 0))

        # ── Indicator 2: RSI(36) — 3-minute RSI ─────────────────────────────
        rsi = _rsi(close, 36)
        rsi_signal = np.where(rsi < rsi_oversold,  1,
                     np.where(rsi > rsi_overbought, -1, 0))

        # ── Indicator 3: MACD(24, 52, 18) — 2 / 4.3 / 1.5 min ──────────────
        ema_fast   = _ema(close, 24)
        ema_slow   = _ema(close, 52)
        macd_line  = ema_fast - ema_slow
        macd_sig   = _ema(macd_line, 18)
        macd_diff  = macd_line - macd_sig
        macd_signal = np.where(macd_diff > 0.0, 1, -1)

        # ── Indicator 4: Stochastic K(24, 6) — 2-min range, 30s smooth ──────
        stoch = _stoch_k(high, low, close, 24, 6)
        stoch_signal = np.where(stoch < stoch_oversold,  1,
                       np.where(stoch > stoch_overbought, -1, 0))

        # ── Indicator 5: Directional VROC(12) — 1-minute volume ROC ─────────
        # Sign follows price direction so it contributes ±1 for both bull/bear
        vroc = np.zeros(n)
        for i in range(12, n):
            prev_vol = volume[i - 12]
            if prev_vol > 0.0:
                vroc[i] = (volume[i] - prev_vol) / prev_vol * 100.0

        # Price direction uses PRIOR close to avoid look-ahead
        price_dir = np.zeros(n, dtype=np.int32)
        price_dir[1:] = np.where(close[1:] > close[:-1], 1,
                        np.where(close[1:] < close[:-1], -1, 0))

        vroc_signal = np.where(vroc > vroc_threshold, price_dir, 0)

        # ── Consensus Score (max ±5) ─────────────────────────────────────────
        consensus = (vwap_signal + rsi_signal + macd_signal
                     + stoch_signal + vroc_signal)

        # ── 2-bar confirmation (10 seconds) ──────────────────────────────────
        buy_ce_raw = consensus >= consensus_threshold
        buy_pe_raw = consensus <= -consensus_threshold

        buy_ce = np.zeros(n, dtype=bool)
        buy_pe = np.zeros(n, dtype=bool)
        for i in range(1, n):
            buy_ce[i] = bool(buy_ce_raw[i]) and bool(buy_ce_raw[i - 1])
            buy_pe[i] = bool(buy_pe_raw[i]) and bool(buy_pe_raw[i - 1])

        # ── Session filter ────────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes) &
            (time_min <  self.session_end_minutes)
        )
        buy_ce = buy_ce & in_session
        buy_pe = buy_pe & in_session

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, 4.0),
            target_points=np.full(n, 6.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,
            max_trades_per_day=self.max_trades_per_day,
        )
