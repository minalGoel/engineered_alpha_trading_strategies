"""adverse_selection_filter_v1 — VPIN-based dual-regime strategy for NIFTY 5-second options.

Mechanism:
  On NIFTY at 5-second resolution, institutional TWAP/FII basket orders create measurable
  directional volume imbalance within each bar via bulk volume classification (BVC).
  When VPIN is high (informed flow dominant) → trade momentum continuation.
  When VPIN is low (noise flow dominant) → fade RSI extremes via mean-reversion.
  This avoids the classic adverse-selection error of mean-reverting into an institutional order.

Converted from: trading_strategies/unique_strategies_all/Strategy_245.json
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_vpin(volume: np.ndarray, close: np.ndarray,
                  high: np.ndarray, low: np.ndarray,
                  lookback: int = 60) -> np.ndarray:
    """Approximate VPIN using bulk volume classification (BVC).

    buy_vol estimate per bar = volume × (close - low) / (high - low)
    sell_vol estimate        = volume × (high - close) / (high - low)
    VPIN = |sum_buy_vol - sum_sell_vol| / sum_total_vol  over rolling `lookback` bars
    """
    n = len(volume)
    bar_range = high - low
    buy_frac = np.where(bar_range > 0.0, (close - low) / bar_range, 0.5)
    buy_vol = volume * buy_frac
    sell_vol = volume * (1.0 - buy_frac)

    vpin = np.full(n, 0.5)
    for i in range(lookback, n):
        total = np.sum(volume[i - lookback:i])
        if total > 0.0:
            net = abs(np.sum(buy_vol[i - lookback:i]) - np.sum(sell_vol[i - lookback:i]))
            vpin[i] = net / total
    return vpin


def _compute_rsi(close: np.ndarray, period: int = 24) -> np.ndarray:
    """Wilder RSI on price changes."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n <= period:
        return rsi

    delta = np.zeros(n)
    delta[1:] = close[1:] - close[:-1]
    gains = np.maximum(delta, 0.0)
    losses = np.maximum(-delta, 0.0)

    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[period] = np.mean(gains[1: period + 1])
    avg_loss[period] = np.mean(losses[1: period + 1])

    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period

    rs = np.where(avg_loss > 0.0, avg_gain / avg_loss, 100.0)
    rsi[period:] = 100.0 - 100.0 / (1.0 + rs[period:])
    return rsi


class Strategy(BaseStrategy):
    name = "adverse_selection_filter_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 8
    max_lookback = 300            # 25-min warmup: 240-bar VPIN context + RSI/momentum lead-in

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vpin_toxic_pctile", 0.70, 0.55, 0.85),
            TunableParam("vpin_benign_pctile", 0.35, 0.15, 0.50),
            TunableParam("rsi_oversold", 30.0, 20.0, 42.0),
            TunableParam("rsi_overbought", 70.0, 58.0, 80.0),
            TunableParam("momentum_threshold", 0.0003, 0.0001, 0.0010),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract spot arrays (forward-fill NaN before numpy) ──
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(float)
        high = spot_df["high"].fill_null(strategy="forward").to_numpy().astype(float)
        low = spot_df["low"].fill_null(strategy="forward").to_numpy().astype(float)
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        time_min = spot_df["time_minutes"].to_numpy()

        # ── Parameters ──
        vpin_toxic_pctile = float(params.get("vpin_toxic_pctile", 0.70))
        vpin_benign_pctile = float(params.get("vpin_benign_pctile", 0.35))
        rsi_oversold = float(params.get("rsi_oversold", 30.0))
        rsi_overbought = float(params.get("rsi_overbought", 70.0))
        mom_thresh = float(params.get("momentum_threshold", 0.0003))

        # ── Session filter ──
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # ── 1. VPIN (5-minute rolling, 60 × 5s bars) ──
        vpin_lookback = 60
        vpin = _compute_vpin(volume, close, high, low, lookback=vpin_lookback)

        # ── 2. Rolling VPIN regime: compare to 20-min (240-bar) rolling percentile ──
        vpin_context = 240
        is_toxic = np.zeros(n, dtype=bool)
        is_benign = np.zeros(n, dtype=bool)

        for i in range(vpin_context, n):
            window = vpin[i - vpin_context:i]
            toxic_thresh = np.quantile(window, vpin_toxic_pctile)
            benign_thresh = np.quantile(window, vpin_benign_pctile)
            if vpin[i] >= toxic_thresh:
                is_toxic[i] = True
            elif vpin[i] <= benign_thresh:
                is_benign[i] = True

        # ── 3. RSI(24) — 2-minute RSI for benign mean-reversion ──
        rsi = _compute_rsi(close, period=24)

        # ── 4. 1-minute momentum (12 bars) for toxic continuation ──
        mom_12 = np.zeros(n)
        denom = np.where(close[:-12] > 0.0, close[:-12], 1.0)
        mom_12[12:] = (close[12:] - close[:-12]) / denom

        # ── 5. Volume above rolling 3-min average (loose filter) ──
        vol_window = 36  # 3 minutes
        vol_sma = np.zeros(n)
        for i in range(vol_window, n):
            vol_sma[i] = np.mean(volume[i - vol_window:i])
        above_avg_vol = (volume >= vol_sma * 0.8) | (vol_sma == 0.0)

        # ── Signal generation ──
        # Benign mode: RSI mean-reversion
        benign_buy_ce = in_session & is_benign & (rsi < rsi_oversold) & above_avg_vol
        benign_buy_pe = in_session & is_benign & (rsi > rsi_overbought) & above_avg_vol

        # Toxic mode: momentum continuation
        toxic_buy_ce = in_session & is_toxic & (mom_12 > mom_thresh) & above_avg_vol
        toxic_buy_pe = in_session & is_toxic & (mom_12 < -mom_thresh) & above_avg_vol

        buy_ce = benign_buy_ce | toxic_buy_ce
        buy_pe = benign_buy_pe | toxic_buy_pe

        # ── Per-bar stop/target by regime ──
        # Toxic: tighter stop (3 pts), wider target (7 pts) — momentum fails fast or runs hard
        # Benign: wider stop (4 pts), moderate target (6 pts) — need room for final flush before bounce
        stop_arr = np.where(is_toxic, 3.0, 4.0).astype(np.float64)
        target_arr = np.where(is_toxic, 7.0, 6.0).astype(np.float64)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=stop_arr,
            target_points=target_arr,
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,           # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
