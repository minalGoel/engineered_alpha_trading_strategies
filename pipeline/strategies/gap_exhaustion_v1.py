"""gap_exhaustion_v1 — Gap Exhaustion Reversal on NIFTY.

After a gap open (>0.5%), NIFTY extends the gap for 10-30 minutes before
exhausting. When price is near the session extreme but 5-second volume has
dried up AND MFI + RSI confirm overbought/oversold, VWAP-benchmarked
institutions and short-gamma market-makers initiate counter-trend flow.
We enter ATM options to capture the first reversal leg (15-25 spot pts).

Session: 09:25-11:00 IST only (exhaustion window).
Hold: 15-120 seconds (time_stop_bars=24).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple rolling mean with NaN for the first (window-1) bars."""
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        out[i] = arr[i - window + 1 : i + 1].mean()
    return out


def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder RSI. Returns NaN for first `period` bars."""
    n = len(close)
    rsi = np.full(n, np.nan)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    if n < period + 1:
        return rsi
    avg_gain = np.mean(gain[1 : period + 1])
    avg_loss = np.mean(loss[1 : period + 1])
    for i in range(period, n):
        if i == period:
            ag, al = avg_gain, avg_loss
        else:
            ag = (ag * (period - 1) + gain[i]) / period
            al = (al * (period - 1) + loss[i]) / period
        if al == 0:
            rsi[i] = 100.0
        else:
            rs = ag / al
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def _mfi(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    period: int,
) -> np.ndarray:
    """Money Flow Index over `period` bars."""
    n = len(close)
    mfi = np.full(n, np.nan)
    tp = (high + low + close) / 3.0
    raw_mf = tp * volume
    pos_mf = np.where(
        np.concatenate([[False], tp[1:] > tp[:-1]]), raw_mf, 0.0
    )
    neg_mf = np.where(
        np.concatenate([[False], tp[1:] < tp[:-1]]), raw_mf, 0.0
    )
    for i in range(period, n):
        pos_sum = pos_mf[i - period + 1 : i + 1].sum()
        neg_sum = neg_mf[i - period + 1 : i + 1].sum()
        if neg_sum == 0:
            mfi[i] = 100.0
        else:
            mfi[i] = 100.0 - 100.0 / (1.0 + pos_sum / neg_sum)
    return mfi


def _compute_gap_and_session_extremes(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    day_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-bar gap_pct, running session_high, running session_low.

    gap_pct: (session_open - prev_day_last_close) / prev_day_last_close * 100
             Same value for every bar within a day. 0.0 on the first day.
    session_high: running max of high from session open within each day.
    session_low:  running min of low from session open within each day.
    """
    n = len(close)
    gap_pct = np.zeros(n)
    session_high = np.zeros(n)
    session_low = np.zeros(n)

    # Track per-day first open and previous day's last close
    prev_last_close = np.nan
    current_day = day_id[0]
    sess_open = open_[0]
    sess_high_val = high[0]
    sess_low_val = low[0]
    day_gap = 0.0

    session_high[0] = sess_high_val
    session_low[0] = sess_low_val
    gap_pct[0] = 0.0  # No prev day on first bar

    for i in range(1, n):
        if day_id[i] != current_day:
            # Day boundary — record last close of old day
            prev_last_close = close[i - 1]
            current_day = day_id[i]
            sess_open = open_[i]
            sess_high_val = high[i]
            sess_low_val = low[i]
            if not np.isnan(prev_last_close) and prev_last_close > 0:
                day_gap = (sess_open - prev_last_close) / prev_last_close * 100.0
            else:
                day_gap = 0.0
        else:
            sess_high_val = max(sess_high_val, high[i])
            sess_low_val = min(sess_low_val, low[i])

        gap_pct[i] = day_gap
        session_high[i] = sess_high_val
        session_low[i] = sess_low_val

    return gap_pct, session_high, session_low


class Strategy(BaseStrategy):
    name = "gap_exhaustion_v1"
    underlying = "NIFTY"
    session_start_minutes = 565  # 09:25 IST — allow gap to establish before signalling
    session_end_minutes = 660    # 11:00 IST — exhaustion only in first 90 minutes
    max_trades_per_day = 3
    max_lookback = 240           # 20 min warmup (MFI/RSI need 36-bar window to stabilise)

    def tunable_params(self) -> list[TunableParam]:
        return [
            # Minimum absolute gap % to consider (both up and down)
            TunableParam("gap_threshold", 0.5, 0.3, 1.0),
            # Proximity to session extreme: close must be within this % of extreme
            TunableParam("near_pct", 0.0012, 0.0006, 0.0025),
            # Recent vol / baseline vol ratio threshold (below = exhaustion)
            TunableParam("vol_ratio", 0.72, 0.45, 0.90),
            # MFI overbought threshold (>this for gap-up exhaustion)
            TunableParam("mfi_ob", 72.0, 65.0, 82.0),
            # MFI oversold threshold (<this for gap-down exhaustion)
            TunableParam("mfi_os", 28.0, 18.0, 35.0),
            # RSI overbought threshold (>this for gap-up exhaustion)
            TunableParam("rsi_ob", 68.0, 60.0, 78.0),
            # RSI oversold threshold (<this for gap-down exhaustion)
            TunableParam("rsi_os", 32.0, 22.0, 40.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict,
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract spot arrays (forward-fill NaN in Polars first) ──
        spot_filled = spot_df.with_columns([
            pl.col("open").forward_fill(),
            pl.col("high").forward_fill(),
            pl.col("low").forward_fill(),
            pl.col("close").forward_fill(),
            pl.col("volume").forward_fill().fill_null(0),
            pl.col("time_minutes").forward_fill(),
            pl.col("day_id").forward_fill(),
        ])
        open_ = spot_filled["open"].to_numpy().astype(np.float64)
        high = spot_filled["high"].to_numpy().astype(np.float64)
        low = spot_filled["low"].to_numpy().astype(np.float64)
        close = spot_filled["close"].to_numpy().astype(np.float64)
        volume = spot_filled["volume"].to_numpy().astype(np.float64)
        time_min = spot_filled["time_minutes"].to_numpy()
        day_id = spot_filled["day_id"].to_numpy()

        # ── VIX ──
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

        # ── Parameters ──
        gap_threshold = params.get("gap_threshold", 0.5)
        near_pct = params.get("near_pct", 0.0012)
        vol_ratio = params.get("vol_ratio", 0.72)
        mfi_ob = params.get("mfi_ob", 72.0)
        mfi_os = params.get("mfi_os", 28.0)
        rsi_ob = params.get("rsi_ob", 68.0)
        rsi_os = params.get("rsi_os", 32.0)

        # ── Indicators ──
        gap_pct, session_high, session_low = _compute_gap_and_session_extremes(
            open_, high, low, close, day_id
        )

        # Rolling volume: 36-bar baseline (3 min), 6-bar recent (30 s)
        vol_ma_36 = _rolling_mean(volume, 36)
        vol_ma_6 = _rolling_mean(volume, 6)

        # Replace NaN in volume MAs with neutral (no exhaustion signal)
        vol_ma_36 = np.where(np.isnan(vol_ma_36), 1.0, vol_ma_36)
        vol_ma_6 = np.where(np.isnan(vol_ma_6), 1.0, vol_ma_6)

        # MFI(36) and RSI(24)
        mfi = _mfi(high, low, close, volume, 36)
        rsi = _rsi(close, 24)

        # Replace NaN in MFI/RSI with neutral (50)
        mfi = np.where(np.isnan(mfi), 50.0, mfi)
        rsi = np.where(np.isnan(rsi), 50.0, rsi)

        # ── Session and VIX filters ──
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )
        vix_ok = (vix_close >= 12.0) & (vix_close <= 22.0)

        # ── Gap filters ──
        gap_up = gap_pct > gap_threshold    # Gap-up day
        gap_dn = gap_pct < -gap_threshold   # Gap-down day

        # ── Proximity to session extreme ──
        near_session_high = close >= session_high * (1.0 - near_pct)
        near_session_low = close <= session_low * (1.0 + near_pct)

        # ── Volume exhaustion: recent 30s below 72% of 3-min baseline ──
        vol_exhausted = vol_ma_6 < vol_ma_36 * vol_ratio

        # ── Gap-up exhaustion → buy PE (bearish reversal) ──
        buy_pe = (
            in_session
            & vix_ok
            & gap_up
            & near_session_high
            & vol_exhausted
            & (mfi > mfi_ob)
            & (rsi > rsi_ob)
        )

        # ── Gap-down exhaustion → buy CE (bullish reversal) ──
        buy_ce = (
            in_session
            & vix_ok
            & gap_dn
            & near_session_low
            & vol_exhausted
            & (mfi < mfi_os)
            & (rsi < rsi_os)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            # Stop: 4 pts = ~8 spot pts adverse — if NIFTY extends further, thesis broken
            stop_points=np.full(n, 4.0),
            # Target: 7 pts = ~14 spot pts reversion — captures ~60% of first reversal leg
            target_points=np.full(n, 7.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,          # 120 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
