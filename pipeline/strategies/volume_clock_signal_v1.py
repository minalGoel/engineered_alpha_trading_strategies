"""volume_clock_signal_v1 — Volume-Clock RSI/BB Exhaustion on NIFTY

Mechanism: On NIFTY, institutional TWAP/algo executions create bursts of
high-volume 5-second bars where information content far exceeds low-volume
drift bars. When RSI drops below 30 during a volume surge (vol > 1.5x
rolling average), it signals that large sell programs have pushed price
aggressively below the 5-min Bollinger lower band AND below VWAP — a
multi-confirmation exhaustion signal. The moment RSI turns up while volume
remains elevated, market-makers absorb remaining sell flow, triggering a
10-20 NIFTY spot point bounce within 30-90 seconds. This approximates the
volume-clock bar concept (treating high-volume 5s bars as 'information-dense')
without dynamic bar regrouping.

Converted from: trading_strategies/unique_strategies_all/Strategy_242.json
Original: RSI(14) + BB(20) on volume-clock bars across NIFTY50 constituents.
"""
from __future__ import annotations
import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """RSI with Wilder smoothing. Returns 50.0 for warmup bars."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi

    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)

    # Seed with simple average over first `period` deltas
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    # Wilder smoothing: each subsequent bar
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period

    for i in range(period, n):
        if avg_loss[i] == 0.0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


def _compute_vwap(
    close: np.ndarray, volume: np.ndarray, day_id: np.ndarray
) -> np.ndarray:
    """Cumulative VWAP from session open, reset each day."""
    n = len(close)
    vwap = np.zeros(n)
    cum_pv = 0.0
    cum_vol = 0.0
    prev_day = -1

    for i in range(n):
        if day_id[i] != prev_day:
            cum_pv = 0.0
            cum_vol = 0.0
            prev_day = day_id[i]
        cum_pv += close[i] * volume[i]
        cum_vol += volume[i]
        vwap[i] = cum_pv / cum_vol if cum_vol > 0 else close[i]

    return vwap


class Strategy(BaseStrategy):
    name = "volume_clock_signal_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 10
    # 60 bars (5 min) for BB warmup + 12 bars for RSI
    max_lookback = 72

    def tunable_params(self) -> list[TunableParam]:
        return [
            # RSI oversold threshold
            TunableParam("rsi_low", 30.0, 25.0, 40.0),
            # RSI overbought threshold
            TunableParam("rsi_high", 70.0, 60.0, 75.0),
            # VWAP deviation threshold (fraction, e.g. 0.0015 = 15 bps)
            TunableParam("vwap_dev_threshold", 0.0015, 0.0008, 0.003),
            # Volume surge ratio vs. 5-min rolling average
            TunableParam("vol_ratio_threshold", 1.5, 1.2, 2.5),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict,
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract arrays (forward-fill NaN before converting) ──────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        volume = (
            spot_df["volume"]
            .fill_null(0)
            .cast(pl.Float64)
            .to_numpy()
        )
        day_id = spot_df["day_id"].fill_null(0).to_numpy()
        time_min = spot_df["time_minutes"].fill_null(0).to_numpy()

        # ── Parameters ───────────────────────────────────────────────────────
        rsi_low = params.get("rsi_low", 30.0)
        rsi_high = params.get("rsi_high", 70.0)
        vwap_dev_threshold = params.get("vwap_dev_threshold", 0.0015)
        vol_ratio_threshold = params.get("vol_ratio_threshold", 1.5)

        # ── RSI(12) — 60-second RSI on 5s bars ──────────────────────────────
        # Lookback 12: captures fast exhaustion within 1 minute, tuned to
        # 30-120s hold time (vs. original RSI(14) on ~21 min of volume bars)
        rsi = _compute_rsi(close, 12)

        # ── Bollinger Bands(60, 2.0) — 5-minute window ───────────────────────
        # Lookback 60: reflects current micro-session context. Original used
        # 20 volume-clock bars ≈ 30 min; compressed to 5 min for 30-120s hold.
        bb_period = 60
        bb_upper = np.full(n, np.inf)
        bb_lower = np.full(n, -np.inf)
        for i in range(bb_period, n):
            window = close[i - bb_period:i]
            sma = np.mean(window)
            std = np.std(window)
            bb_upper[i] = sma + 2.0 * std
            bb_lower[i] = sma - 2.0 * std

        # ── VWAP — cumulative from session open, reset daily ─────────────────
        vwap = _compute_vwap(close, volume, day_id)
        vwap_dev = np.where(vwap > 0, (close - vwap) / vwap, 0.0)

        # ── Volume ratio — current bar vs. 5-min rolling average ─────────────
        # Approximates volume-clock density: vol > 1.5x avg = information-dense bar
        vol_avg = np.ones(n)
        for i in range(bb_period, n):
            avg = np.mean(volume[i - bb_period:i])
            vol_avg[i] = avg if avg > 0 else 1.0
        vol_ratio = np.where(vol_avg > 0, volume / vol_avg, 0.0)

        # ── RSI direction (confirmation) ─────────────────────────────────────
        rsi_turning_up = np.zeros(n, dtype=bool)
        rsi_turning_down = np.zeros(n, dtype=bool)
        rsi_turning_up[1:] = rsi[1:] > rsi[:-1]
        rsi_turning_down[1:] = rsi[1:] < rsi[:-1]

        # ── Session filter ────────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── Signals ──────────────────────────────────────────────────────────
        # Bullish: oversold RSI + below BB lower + negative VWAP dev + volume surge + RSI turning up
        buy_ce = (
            in_session
            & (rsi < rsi_low)
            & (close <= bb_lower)
            & (vwap_dev < -vwap_dev_threshold)
            & (vol_ratio >= vol_ratio_threshold)
            & rsi_turning_up
        )

        # Bearish: overbought RSI + above BB upper + positive VWAP dev + volume surge + RSI turning down
        buy_pe = (
            in_session
            & (rsi > rsi_high)
            & (close >= bb_upper)
            & (vwap_dev > vwap_dev_threshold)
            & (vol_ratio >= vol_ratio_threshold)
            & rsi_turning_down
        )

        # Mask warmup region (no valid BB yet)
        buy_ce[:bb_period] = False
        buy_pe[:bb_period] = False

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            # Stop: 5 pts ≈ 10 NIFTY spot pts — if price continues past 10 pts
            # beyond entry, the exhaustion read was wrong; stop is at upper end
            # of 30s typical range (2-5 pts per calibration table).
            stop_points=np.full(n, 5.0),
            # Target: 8 pts ≈ 16 NIFTY spot pts — volume-surge bounces cover
            # 14-20 spot pts in 60-90s; 8 pts captures the median first-leg
            # reversion (1:1.6 R:R with 5pt stop).
            target_points=np.full(n, 8.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,        # 120 seconds max hold
            max_trades_per_day=10,
        )
