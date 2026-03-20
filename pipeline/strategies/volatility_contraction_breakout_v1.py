"""
volatility_contraction_breakout_v1 — ATR Squeeze Breakout on NIFTY

Mechanism: NIFTY consolidates for 3-5 minute periods (institutional order pauses) with
ATR(12) dropping below the 20th percentile of its 10-minute rolling distribution. When
this equilibrium breaks — visible as a single 5-second bar with range > 1.5× ATR and
volume > 1.8× 5-min average — trapped participants create a 15-90 second directional
burst. VWAP alignment filters the direction.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """SMA-based ATR. Uses expanding window for initial bars."""
    n = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
    )
    atr = np.empty(n)
    for i in range(n):
        start = max(0, i - period + 1)
        atr[i] = np.mean(tr[start : i + 1])
    return atr


def _day_start_indices(day_id: np.ndarray) -> np.ndarray:
    """For each bar i, the index of the first bar in the same day."""
    n = len(day_id)
    day_starts = np.zeros(n, dtype=np.int32)
    for i in range(1, n):
        if day_id[i] != day_id[i - 1]:
            day_starts[i] = i
        else:
            day_starts[i] = day_starts[i - 1]
    return day_starts


def _rolling_atr_pctile(
    atr: np.ndarray, day_starts: np.ndarray, window: int
) -> np.ndarray:
    """Percentile rank of current ATR within the last `window` bars, clipped to day start."""
    n = len(atr)
    pctile = np.zeros(n)
    for i in range(1, n):
        start = max(day_starts[i], i - window + 1)
        w = atr[start : i + 1]
        pctile[i] = np.sum(w <= atr[i]) / len(w) * 100.0
    return pctile


class Strategy(BaseStrategy):
    name = "volatility_contraction_breakout_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — allow 15 min of session history for stable percentile
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 6
    max_lookback = 240            # 20-minute warmup (240 × 5s)

    def tunable_params(self) -> list[TunableParam]:
        return [
            # ATR percentile threshold for squeeze detection
            TunableParam("atr_pctile_threshold", 20.0, 10.0, 35.0),
            # Minimum consecutive squeeze bars required (float → cast to int in compute)
            TunableParam("squeeze_bars_min", 36.0, 20.0, 72.0),
            # Volume ratio threshold on the expansion bar
            TunableParam("volume_ratio_threshold", 1.8, 1.2, 3.0),
            # Range expansion multiplier: bar_range > atr * this
            TunableParam("expansion_ratio", 1.5, 1.0, 2.5),
            # Stop/target in option premium points
            TunableParam("stop_pts", 3.0, 2.0, 6.0),
            TunableParam("target_pts", 6.0, 4.0, 12.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict,
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Raw arrays (forward-fill NaN before conversion) ──────────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # ── Parameters ───────────────────────────────────────────────────────
        atr_pctile_threshold = params.get("atr_pctile_threshold", 20.0)
        squeeze_bars_min = int(params.get("squeeze_bars_min", 36.0))
        volume_ratio_threshold = params.get("volume_ratio_threshold", 1.8)
        expansion_ratio = params.get("expansion_ratio", 1.5)
        stop_pts = params.get("stop_pts", 3.0)
        target_pts = params.get("target_pts", 6.0)

        # ── ATR(12) — 60-second true range ───────────────────────────────────
        atr = _compute_atr(high, low, close, period=12)
        # Guard: very small ATR values → replace with median to avoid division issues
        atr_median = np.median(atr[atr > 0]) if np.any(atr > 0) else 1.0
        atr = np.where(atr > 0, atr, atr_median)

        # ── Rolling ATR percentile (10-min = 120 bars, within same day) ──────
        day_starts = _day_start_indices(day_id)
        atr_pctile = _rolling_atr_pctile(atr, day_starts, window=120)

        # ── Squeeze count: consecutive bars with ATR pctile < threshold ───────
        in_squeeze = atr_pctile < atr_pctile_threshold
        squeeze_count = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            if day_id[i] != day_id[i - 1]:
                # New day: reset count
                squeeze_count[i] = 1 if in_squeeze[i] else 0
            elif in_squeeze[i]:
                squeeze_count[i] = squeeze_count[i - 1] + 1
            else:
                squeeze_count[i] = 0

        # ── Sustained squeeze recently ended: within last 5 bars ─────────────
        prev_squeeze_sustained = np.zeros(n, dtype=bool)
        for i in range(1, n):
            lookback = min(5, i)
            for j in range(1, lookback + 1):
                if day_id[i] == day_id[i - j] and squeeze_count[i - j] >= squeeze_bars_min:
                    prev_squeeze_sustained[i] = True
                    break

        # ── Volume ratio: current / 5-min SMA ────────────────────────────────
        vol_sma = np.empty(n)
        for i in range(n):
            start = max(0, i - 60 + 1)
            vol_sma[i] = np.mean(volume[start : i + 1])
        vol_ratio = np.where(vol_sma > 0, volume / vol_sma, 1.0)

        # ── Expansion bar detection ───────────────────────────────────────────
        bar_range = high - low
        is_expansion = (bar_range > atr * expansion_ratio) & (vol_ratio > volume_ratio_threshold)

        # ── Cumulative VWAP per day ───────────────────────────────────────────
        vwap = np.empty(n)
        cum_vol = 0.0
        cum_tpv = 0.0
        for i in range(n):
            if i == 0 or day_id[i] != day_id[i - 1]:
                cum_vol = volume[i]
                cum_tpv = close[i] * volume[i]
            else:
                cum_vol += volume[i]
                cum_tpv += close[i] * volume[i]
            vwap[i] = cum_tpv / cum_vol if cum_vol > 0 else close[i]

        # ── Session filter ────────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # ── Entry signals ─────────────────────────────────────────────────────
        # Conditions: squeeze just ended (prev_squeeze_sustained) AND current bar is NOT
        # in squeeze AND expansion bar fires AND VWAP alignment
        base_signal = in_session & prev_squeeze_sustained & (~in_squeeze) & is_expansion

        buy_ce = base_signal & (close > vwap)   # bullish breakout above VWAP
        buy_pe = base_signal & (close < vwap)   # bearish breakout below VWAP

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,           # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
