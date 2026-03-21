"""entropy_regime_v1 — Shannon Entropy Regime Detection for NIFTY 5s Options

Mechanism:
    When institutional TWAP/VWAP algorithms work large directional NIFTY orders, the
    5-second return distribution compresses into a tight cluster of small positive (buy
    program) or small negative (sell program) values. Shannon entropy over 60 bars (5 min)
    falls below ~1.8 bits vs. the 3.0-bit random-walk baseline. This compressed entropy
    signals an order program with unfilled lots remaining, creating 30-90s autocorrelation
    in NIFTY spot returns. We enter in the direction of the dominant mean return, confirmed
    by VWAP position and 1-min/3-min EMA alignment.

Original: entropy_regime_v1 — NIFTY 50 constituents, 1-min bars, 15-45 min hold.
Conversion: NIFTY index itself, 5s bars, 30-90s hold.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _shannon_entropy_bits(returns_window: np.ndarray, n_bins: int = 8) -> float:
    """Compute Shannon entropy of binned returns in bits.

    Returns log2(n_bins) (maximum entropy) when the window is too small
    or all returns are identical.
    """
    max_entropy = float(np.log2(n_bins))
    if len(returns_window) < n_bins:
        return max_entropy
    counts, _ = np.histogram(returns_window, bins=n_bins)
    total = counts.sum()
    if total == 0:
        return max_entropy
    probs = counts[counts > 0] / total
    return float(-np.sum(probs * np.log2(probs)))


class Strategy(BaseStrategy):
    name = "entropy_regime_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — need 60-bar warmup after open
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 6
    max_lookback = 120            # 10-min warmup (120 × 5s)

    def tunable_params(self) -> list[TunableParam]:
        return [
            # Primary regime filter: entropy below this = structured order flow
            TunableParam("entropy_threshold", 1.8, 1.0, 2.5),
            # Minimum |mean return| to confirm direction (not a flat-entropy regime)
            TunableParam("min_dominant_return", 0.00005, 0.00002, 0.00020),
            # Minimum return std to filter inactivity false signals
            TunableParam("min_return_vol", 0.00010, 0.00005, 0.00050),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # Extract arrays — forward-fill NaN before converting to numpy
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)

        entropy_threshold = params.get("entropy_threshold", 1.8)
        min_dom_ret = params.get("min_dominant_return", 0.00005)
        min_ret_vol = params.get("min_return_vol", 0.00010)

        # ── 5-second log returns ──────────────────────────────────────────────
        log_ret = np.zeros(n, dtype=np.float64)
        prev = np.where(close[:-1] > 0, close[:-1], 1.0)
        log_ret[1:] = np.log(close[1:] / prev)

        # ── Rolling 60-bar (5-min) metrics ───────────────────────────────────
        # entropy_60: Shannon entropy of binned returns (bits)
        # dom_ret:    mean return over window (direction of regime)
        # ret_vol:    std of returns (liveness guard vs. inactivity)
        ENTROPY_WINDOW = 60
        MAX_ENTROPY = float(np.log2(8))  # 3.0 bits

        entropy = np.full(n, MAX_ENTROPY, dtype=np.float64)
        dom_ret = np.zeros(n, dtype=np.float64)
        ret_vol = np.zeros(n, dtype=np.float64)

        for i in range(ENTROPY_WINDOW, n):
            window = log_ret[i - ENTROPY_WINDOW:i]
            entropy[i] = _shannon_entropy_bits(window, n_bins=8)
            dom_ret[i] = float(np.mean(window))
            ret_vol[i] = float(np.std(window))

        # ── EMA(12) = 1-min trigger, EMA(36) = 3-min context ─────────────────
        ema_12 = np.empty(n, dtype=np.float64)
        ema_36 = np.empty(n, dtype=np.float64)
        alpha_12 = 2.0 / (12 + 1)
        alpha_36 = 2.0 / (36 + 1)
        ema_12[0] = close[0]
        ema_36[0] = close[0]
        for i in range(1, n):
            ema_12[i] = alpha_12 * close[i] + (1.0 - alpha_12) * ema_12[i - 1]
            ema_36[i] = alpha_36 * close[i] + (1.0 - alpha_36) * ema_36[i - 1]

        # ── Session VWAP (cumulative, resets each day) ────────────────────────
        vwap = np.empty(n, dtype=np.float64)
        cum_pv = 0.0
        cum_v = 0.0
        cur_day = day_id[0]
        for i in range(n):
            if day_id[i] != cur_day:
                cum_pv = 0.0
                cum_v = 0.0
                cur_day = day_id[i]
            cum_pv += close[i] * volume[i]
            cum_v += volume[i]
            vwap[i] = cum_pv / cum_v if cum_v > 0.0 else close[i]

        # ── Masks ─────────────────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )
        warmed = np.arange(n) >= ENTROPY_WINDOW

        # Genuine structured regime: low entropy AND sufficient return volatility
        low_entropy = (entropy < entropy_threshold) & (ret_vol > min_ret_vol)

        # Bullish: low-entropy buy program + positive mean return + above VWAP + EMA aligned up
        buy_ce = (
            in_session
            & warmed
            & low_entropy
            & (dom_ret > min_dom_ret)
            & (close > vwap)
            & (ema_12 > ema_36)
        )

        # Bearish: low-entropy sell program + negative mean return + below VWAP + EMA aligned down
        buy_pe = (
            in_session
            & warmed
            & low_entropy
            & (dom_ret < -min_dom_ret)
            & (close < vwap)
            & (ema_12 < ema_36)
        )

        # Stop: 4 pts — 8 spot pts adverse move in <30s invalidates the algo-program thesis
        # Target: 7 pts — 14 spot pts, lower bound of 60-90s directed NIFTY program move
        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, 4.0),
            target_points=np.full(n, 7.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,        # 90 seconds
            max_trades_per_day=6,
        )
