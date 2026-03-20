"""supertrend_momentum_v1 — SuperTrend flip momentum on NIFTY 5-second bars.

Mechanism: On NIFTY, when the 2-minute SuperTrend (ATR period=24 bars, multiplier=3.0)
flips direction, algorithmic strategies — especially those using Zerodha Kite defaults —
cascade directional orders in the flip direction. Multiplier=3.0 is preserved from the
original because it is the Zerodha Kite platform default, maximising the crowd Schelling-
point effect. VWAP alignment and volume confirmation filter out low-conviction flips.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Wilder's smoothed ATR."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    atr = np.zeros(n)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _compute_supertrend(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int, multiplier: float
) -> np.ndarray:
    """Return SuperTrend direction array: 1=bullish, -1=bearish."""
    n = len(close)
    atr = _compute_atr(high, low, close, period)

    hl2 = (high + low) / 2.0
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    supertrend = np.zeros(n)
    direction = np.full(n, -1, dtype=np.int32)

    final_upper[0] = basic_upper[0]
    final_lower[0] = basic_lower[0]
    supertrend[0] = basic_upper[0]  # start bearish
    direction[0] = -1

    for i in range(1, n):
        # Final upper band: tighten only, reset when price breaks above
        if basic_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Final lower band: raise only, reset when price breaks below
        if basic_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # Direction: track which band is active
        if supertrend[i - 1] == final_upper[i - 1]:  # was bearish
            if close[i] > final_upper[i]:
                direction[i] = 1
                supertrend[i] = final_lower[i]
            else:
                direction[i] = -1
                supertrend[i] = final_upper[i]
        else:  # was bullish
            if close[i] < final_lower[i]:
                direction[i] = -1
                supertrend[i] = final_upper[i]
            else:
                direction[i] = 1
                supertrend[i] = final_lower[i]

    return direction


def _compute_vwap(close: np.ndarray, volume: np.ndarray, day_id: np.ndarray) -> np.ndarray:
    """Session VWAP, reset at each day boundary."""
    n = len(close)
    vwap = np.zeros(n)
    cum_pv = 0.0
    cum_v = 0.0
    current_day = -1

    for i in range(n):
        if day_id[i] != current_day:
            cum_pv = 0.0
            cum_v = 0.0
            current_day = day_id[i]
        vol = volume[i] if volume[i] > 0 else 0.0
        cum_pv += close[i] * vol
        cum_v += vol
        vwap[i] = cum_pv / cum_v if cum_v > 0 else close[i]

    return vwap


def _rolling_mean(arr: np.ndarray, period: int) -> np.ndarray:
    """Simple rolling mean; returns NaN for first (period-1) bars."""
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        result[i] = np.mean(arr[i - period + 1: i + 1])
    return result


class Strategy(BaseStrategy):
    name = "supertrend_momentum_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 8
    max_lookback = 120            # 10 min warmup — covers 24-bar ATR + 60-bar volume SMA

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_ratio_threshold", 1.3, 0.8, 2.5),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict,
    ) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill NaN in Polars before numpy conversion
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(float)
        high = spot_df["high"].fill_null(strategy="forward").to_numpy().astype(float)
        low = spot_df["low"].fill_null(strategy="forward").to_numpy().astype(float)
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        day_id = spot_df["day_id"].fill_null(0).to_numpy().astype(int)
        time_min = spot_df["time_minutes"].fill_null(0).to_numpy().astype(int)

        vol_ratio_threshold = params.get("vol_ratio_threshold", 1.3)

        # SuperTrend direction — ATR period=24 (2 min), multiplier=3.0 (Zerodha Kite default)
        st_direction = _compute_supertrend(high, low, close, period=24, multiplier=3.0)

        # Flip detection: bullish flip = direction goes -1 → 1
        flip_bullish = np.zeros(n, dtype=bool)
        flip_bearish = np.zeros(n, dtype=bool)
        flip_bullish[1:] = (st_direction[1:] == 1) & (st_direction[:-1] == -1)
        flip_bearish[1:] = (st_direction[1:] == -1) & (st_direction[:-1] == 1)

        # Session VWAP (cumulative, resets each day)
        vwap = _compute_vwap(close, volume, day_id)

        # Volume ratio: current bar vs 5-minute rolling mean (60 bars)
        vol_mean = _rolling_mean(volume, 60)
        vol_mean_safe = np.where(np.isnan(vol_mean) | (vol_mean <= 0), 1.0, vol_mean)
        volume_ratio = volume / vol_mean_safe
        # Where rolling mean was NaN (warmup), set ratio to 0 → no trade
        volume_ratio = np.where(np.isnan(vol_mean), 0.0, volume_ratio)

        # Session gate
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # Entry signals
        vol_ok = volume_ratio > vol_ratio_threshold
        buy_ce = in_session & flip_bullish & (close > vwap) & vol_ok
        buy_pe = in_session & flip_bearish & (close < vwap) & vol_ok

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, 4.0),    # 4 pts = ~8 spot pts reversal → flip was false
            target_points=np.full(n, 6.0),  # 6 pts = ~12 spot pts burst → initial momentum captured
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,              # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
