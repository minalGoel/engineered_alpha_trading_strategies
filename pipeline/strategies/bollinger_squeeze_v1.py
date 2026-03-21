"""Bollinger Squeeze Breakout Strategy — bollinger_squeeze_v1

Detects 4-minute Bollinger Band compressions inside the Keltner Channel on NIFTY.
Enters at the first bar when the squeeze releases (BB expands beyond KC) with
directional confirmation from the normalized momentum oscillator slope.

Hold time: 30-90 seconds (time_stop_bars=18).
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


def _sma_and_std(close: np.ndarray, period: int):
    """Rolling SMA and population std via cumulative sums — O(n)."""
    n = len(close)
    sma = np.full(n, np.nan)
    std = np.full(n, np.nan)
    cum_c = np.zeros(n + 1)
    cum_c2 = np.zeros(n + 1)
    for i in range(n):
        cum_c[i + 1] = cum_c[i] + close[i]
        cum_c2[i + 1] = cum_c2[i] + close[i] ** 2
    for i in range(period - 1, n):
        s = cum_c[i + 1] - cum_c[i + 1 - period]
        s2 = cum_c2[i + 1] - cum_c2[i + 1 - period]
        mean = s / period
        var = max(s2 / period - mean ** 2, 0.0)
        sma[i] = mean
        std[i] = np.sqrt(var)
    return sma, std


def _ema(close: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average — O(n)."""
    n = len(close)
    out = np.zeros(n)
    alpha = 2.0 / (period + 1)
    out[0] = close[0]
    for i in range(1, n):
        out[i] = alpha * close[i] + (1.0 - alpha) * out[i - 1]
    return out


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """ATR via EMA of true range — O(n)."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    return _ema(tr, period)


def _rolling_min(arr: np.ndarray, period: int) -> np.ndarray:
    """Rolling minimum — O(n) amortized via monotone deque."""
    n = len(arr)
    out = np.full(n, np.nan)
    # Deque stores indices; arr values are non-decreasing from front to back
    dq_front = 0
    dq = np.empty(n, dtype=np.int64)
    dq_back = -1  # exclusive back pointer

    for i in range(n):
        # Evict front if outside window
        while dq_front <= dq_back and dq[dq_front] <= i - period:
            dq_front += 1
        # Pop back while arr[back] >= arr[i] (they can never be minimum while i is in window)
        while dq_front <= dq_back and arr[dq[dq_back]] >= arr[i]:
            dq_back -= 1
        dq_back += 1
        dq[dq_back] = i
        if i >= period - 1:
            out[i] = arr[dq[dq_front]]
    return out


def _linreg_slope(arr: np.ndarray, period: int) -> np.ndarray:
    """Rolling linear regression slope — O(n × period), period is small (12)."""
    n = len(arr)
    out = np.zeros(n)
    x = np.arange(period, dtype=float) - (period - 1) / 2.0  # zero-centred x
    x_var = np.sum(x ** 2)
    if x_var < 1e-12:
        return out
    for i in range(period - 1, n):
        y = arr[i - period + 1: i + 1]
        if np.any(np.isnan(y)):
            continue
        out[i] = np.dot(x, y) / x_var
    return out


class Strategy(BaseStrategy):
    name = "bollinger_squeeze_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — avoid opening range noise
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 6
    max_lookback = 240            # 20-min warmup for rolling_min(240)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("mom_slope_threshold", 0.03, 0.005, 0.15),
            TunableParam("squeeze_min_bars", 6.0, 3.0, 15.0),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 8.0, 4.0, 16.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── Raw arrays (forward-fill NaN in Polars before numpy) ──
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        # ── Parameters ──
        mom_thresh = params.get("mom_slope_threshold", 0.03)
        squeeze_min = int(params.get("squeeze_min_bars", 6.0))
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 8.0)

        period = 48  # 4-minute BB / KC / momentum period

        # ── Bollinger Bands ──
        bb_mid, bb_std = _sma_and_std(close, period)
        bb_upper = bb_mid + 2.0 * bb_std
        bb_lower = bb_mid - 2.0 * bb_std
        # Safe denominator: replace nan/zero with current close
        bb_mid_safe = np.where(np.isnan(bb_mid) | (bb_mid <= 0), close, bb_mid)
        bb_width = (bb_upper - bb_lower) / bb_mid_safe * 100.0
        bb_width = np.nan_to_num(bb_width, nan=100.0)  # nan → large width (no squeeze)

        # ── Keltner Channel ──
        ema48 = _ema(close, period)
        atr48 = _atr(high, low, close, period)
        kc_upper = ema48 + 1.5 * atr48
        kc_lower = ema48 - 1.5 * atr48

        # ── Squeeze: BB inside KC ──
        # Fill nan before boolean comparison
        bb_upper_safe = np.nan_to_num(bb_upper, nan=np.inf)
        bb_lower_safe = np.nan_to_num(bb_lower, nan=-np.inf)
        squeeze_confirmed = (bb_upper_safe < kc_upper) & (bb_lower_safe > kc_lower)
        squeeze_confirmed[:period] = False  # mask warmup

        # ── Consecutive squeeze bar counter ──
        consec_squeeze = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            if squeeze_confirmed[i]:
                consec_squeeze[i] = consec_squeeze[i - 1] + 1
            # else: already 0

        # Squeeze release: first bar OUT of squeeze after sustained compression
        # prev_consec[i] = how many consecutive squeeze bars ended at bar i-1
        prev_consec = np.zeros(n, dtype=np.int32)
        prev_consec[1:] = consec_squeeze[:-1]
        squeeze_release = (~squeeze_confirmed) & (prev_consec >= squeeze_min)

        # ── Normalized momentum oscillator and 60s slope ──
        momentum_norm = (close - bb_mid_safe) / bb_mid_safe * 100.0  # % of price
        momentum_norm = np.nan_to_num(momentum_norm, nan=0.0)
        mom_slope = _linreg_slope(momentum_norm, 12)  # 60-second slope
        mom_slope = np.nan_to_num(mom_slope, nan=0.0)

        # ── Directional position within squeeze ──
        above_mid = close > bb_mid_safe
        below_mid = close < bb_mid_safe

        # ── Session filter ──
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # ── Entry signals ──
        buy_ce = in_session & squeeze_release & above_mid & (mom_slope > mom_thresh)
        buy_pe = in_session & squeeze_release & below_mid & (mom_slope < -mom_thresh)

        # Resolve simultaneous CE+PE (ambiguous momentum at squeeze boundary → skip)
        both = buy_ce & buy_pe
        buy_ce = buy_ce & (~both)
        buy_pe = buy_pe & (~both)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,           # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
