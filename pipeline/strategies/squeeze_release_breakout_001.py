"""Squeeze Release Breakout — NIFTY 5-second options.

Detects Bollinger Band squeeze inside Keltner Channel (2-min compression),
then enters on directional breakout above/below the 2-min range with EMA
alignment, VWAP filter, and volume confirmation.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average seeded with the first element."""
    alpha = 2.0 / (period + 1)
    out = arr.copy().astype(float)
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Average True Range via EMA of true range."""
    n = len(high)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    return _ema(tr, period)


def _rolling_sma(arr: np.ndarray, period: int) -> np.ndarray:
    """Rolling simple moving average (vectorised via cumsum)."""
    n = len(arr)
    out = np.zeros(n)
    padded = np.concatenate([[0.0], np.cumsum(arr)])
    out[period - 1 :] = (padded[period:] - padded[:n - period + 1]) / period
    return out


def _rolling_std(arr: np.ndarray, period: int) -> np.ndarray:
    """Rolling population standard deviation."""
    n = len(arr)
    out = np.zeros(n)
    for i in range(period - 1, n):
        out[i] = np.std(arr[i - period + 1 : i + 1], ddof=0)
    return out


def _rolling_max_prev(arr: np.ndarray, period: int) -> np.ndarray:
    """Rolling max over the PREVIOUS `period` bars (excludes current bar).

    out[i] = max(arr[i-period : i])  for i >= period, else nan.
    """
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(period, n):
        out[i] = np.max(arr[i - period : i])
    return out


def _rolling_min_prev(arr: np.ndarray, period: int) -> np.ndarray:
    """Rolling min over the PREVIOUS `period` bars (excludes current bar)."""
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(period, n):
        out[i] = np.min(arr[i - period : i])
    return out


def _session_vwap(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    day_id: np.ndarray,
) -> np.ndarray:
    """Session VWAP that resets at each new day_id."""
    n = len(close)
    typical = (high + low + close) / 3.0
    vwap = np.empty(n)
    cum_tpv = cum_v = 0.0
    prev_day = day_id[0]

    for i in range(n):
        if day_id[i] != prev_day:
            cum_tpv = cum_v = 0.0
            prev_day = day_id[i]
        cum_tpv += typical[i] * volume[i]
        cum_v += volume[i]
        vwap[i] = cum_tpv / cum_v if cum_v > 0.0 else typical[i]

    return vwap


class Strategy(BaseStrategy):
    name = "squeeze_release_breakout_001"
    underlying = "NIFTY"
    session_start_minutes = 600   # 10:00 IST — skip opening noise
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 6
    max_lookback = 72             # 6-min warmup (72 × 5s)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rel_vol_threshold", 1.3, 1.0, 2.0),
            TunableParam("squeeze_min_bars", 3.0, 2.0, 6.0),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 7.0, 4.0, 14.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── Extract arrays, forward-fill nulls in Polars ──────────────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(float)
        high = spot_df["high"].fill_null(strategy="forward").to_numpy().astype(float)
        low = spot_df["low"].fill_null(strategy="forward").to_numpy().astype(float)
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # ── Params ────────────────────────────────────────────────────────────
        rel_vol_thr = float(params.get("rel_vol_threshold", 1.3))
        squeeze_min = int(params.get("squeeze_min_bars", 3))
        stop_pts = float(params.get("stop_pts", 4.0))
        target_pts = float(params.get("target_pts", 7.0))

        period = 24  # 2-min squeeze window (24 × 5s bars)

        # ── Bollinger Bands (2-min) ────────────────────────────────────────────
        sma24 = _rolling_sma(close, period)
        std24 = _rolling_std(close, period)
        bb_upper = sma24 + 2.0 * std24
        bb_lower = sma24 - 2.0 * std24

        # ── Keltner Channels (2-min) ───────────────────────────────────────────
        ema24 = _ema(close, period)
        atr24 = _atr(high, low, close, period)
        kc_upper = ema24 + 1.5 * atr24
        kc_lower = ema24 - 1.5 * atr24

        # ── Squeeze state: BB entirely inside KC ──────────────────────────────
        squeeze_on = (bb_upper < kc_upper) & (bb_lower > kc_lower)

        # ── Recent squeeze: True if squeeze was on in >= squeeze_min of last 6 bars
        squeeze_recent = np.zeros(n, dtype=bool)
        for i in range(6, n):
            squeeze_recent[i] = int(squeeze_on[i - 6 : i].sum()) >= squeeze_min

        # ── Breakout levels (previous 24-bar range, excludes current bar) ─────
        sq_high = _rolling_max_prev(high, period)   # 2-min ceiling
        sq_low = _rolling_min_prev(low, period)     # 2-min floor

        # Replace NaN with values that never trigger (close can't exceed inf)
        sq_high = np.where(np.isnan(sq_high), np.inf, sq_high)
        sq_low = np.where(np.isnan(sq_low), -np.inf, sq_low)

        # ── Directional EMAs ──────────────────────────────────────────────────
        ema18 = _ema(close, 18)   # 90s fast — matches max hold time
        ema36 = _ema(close, 36)   # 3-min slow — micro-trend context

        # ── Session VWAP ──────────────────────────────────────────────────────
        vwap = _session_vwap(close, high, low, volume, day_id)

        # ── Relative volume ───────────────────────────────────────────────────
        sma_vol = _rolling_sma(volume, period)
        rel_vol = np.where(sma_vol > 0.0, volume / sma_vol, 1.0)

        # ── Session and warmup masks ──────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )
        warmup = np.zeros(n, dtype=bool)
        warmup[self.max_lookback :] = True

        # ── Entry signals ─────────────────────────────────────────────────────
        buy_ce = (
            in_session
            & warmup
            & squeeze_recent
            & (close > sq_high)
            & (ema18 > ema36)
            & (close > vwap)
            & (rel_vol >= rel_vol_thr)
        )

        buy_pe = (
            in_session
            & warmup
            & squeeze_recent
            & (close < sq_low)
            & (ema18 < ema36)
            & (close < vwap)
            & (rel_vol >= rel_vol_thr)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,           # 90-second max hold
            max_trades_per_day=self.max_trades_per_day,
        )
