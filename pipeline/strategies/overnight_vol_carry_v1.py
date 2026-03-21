"""overnight_vol_carry_v1 — Overnight volatility carry regime strategy.

Daily regime classification based on NIFTY overnight gap vs rolling average:
- High overnight gap (>1.5x avg): mean-reversion regime — fade the gap direction
- Low overnight gap (<0.3x avg): momentum regime — follow the embedded directional flow
- Neutral gap: no trade

At 5-second resolution, the regime acts as a daily filter; RSI(36) and EMA crossovers
time precise 60-90 second entries within the regime's expected direction.
"""
from __future__ import annotations
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder RSI, returns array of same length with NaN at warmup bars."""
    n = len(close)
    rsi = np.full(n, np.nan)
    if n < period + 1:
        return rsi
    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    # Wilder smoothing
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, n - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0.0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def _compute_ema(close: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average."""
    n = len(close)
    ema = np.full(n, np.nan)
    if n < period:
        return ema
    k = 2.0 / (period + 1)
    ema[period - 1] = np.mean(close[:period])
    for i in range(period, n):
        ema[i] = close[i] * k + ema[i - 1] * (1.0 - k)
    return ema


class Strategy(BaseStrategy):
    name = "overnight_vol_carry_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 6
    max_lookback = 72             # 6 min warmup for EMA/RSI; regime computed from multi-day spot_df

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("high_ratio_threshold", 1.5, 1.1, 2.5),   # overnight_ratio above → mean-rev
            TunableParam("low_ratio_threshold", 0.3, 0.05, 0.6),    # overnight_ratio below → momentum
            TunableParam("rsi_oversold", 35.0, 25.0, 45.0),         # mean-rev buy CE threshold
            TunableParam("rsi_overbought", 65.0, 55.0, 75.0),       # mean-rev buy PE threshold
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        high_ratio = params.get("high_ratio_threshold", 1.5)
        low_ratio = params.get("low_ratio_threshold", 0.3)
        rsi_oversold = params.get("rsi_oversold", 35.0)
        rsi_overbought = params.get("rsi_overbought", 65.0)

        # ── Extract arrays ──
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(float)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # ── Compute per-day session info ──
        # For each day: session_open = first bar open, session_close = last bar close
        unique_days = np.unique(day_id)
        n_days = len(unique_days)

        # Build day_id → index mapping for first/last bar per day
        day_first_close = {}   # first bar close of each day (= effective open)
        day_last_close = {}    # last bar close of each day (= session close)
        for d in unique_days:
            mask = day_id == d
            idxs = np.where(mask)[0]
            day_first_close[d] = close[idxs[0]]
            day_last_close[d] = close[idxs[-1]]

        # Compute overnight gaps for each day (gap = |today_open - prev_close| / prev_close)
        day_gap = {}
        sorted_days = sorted(unique_days)
        for i in range(1, len(sorted_days)):
            today = sorted_days[i]
            prev = sorted_days[i - 1]
            prev_close = day_last_close[prev]
            today_open = day_first_close[today]
            if prev_close > 0:
                day_gap[today] = abs(today_open - prev_close) / prev_close
            else:
                day_gap[today] = 0.0

        # Rolling average of overnight gap (from all available days with gaps)
        gap_values = [v for v in day_gap.values() if v > 0]
        avg_overnight = np.mean(gap_values) if gap_values else 0.01

        # Per-day regime and gap direction
        day_ratio = {}
        day_gap_dir = {}
        for i in range(1, len(sorted_days)):
            today = sorted_days[i]
            prev = sorted_days[i - 1]
            g = day_gap.get(today, 0.0)
            day_ratio[today] = g / avg_overnight if avg_overnight > 0 else 0.0
            prev_close = day_last_close[prev]
            today_open = day_first_close[today]
            day_gap_dir[today] = 1 if today_open >= prev_close else -1

        # ── Per-bar regime arrays ──
        overnight_ratio_arr = np.zeros(n)
        gap_direction_arr = np.zeros(n, dtype=np.int32)
        for i in range(n):
            d = day_id[i]
            overnight_ratio_arr[i] = day_ratio.get(d, 0.0)
            gap_direction_arr[i] = day_gap_dir.get(d, 0)

        # ── Intraday indicators ──
        rsi_36 = _compute_rsi(close, 36)
        ema_12 = _compute_ema(close, 12)
        ema_36 = _compute_ema(close, 36)

        # Forward-fill NaN from warmup bars
        rsi_36 = np.where(np.isnan(rsi_36), 50.0, rsi_36)   # neutral RSI at warmup
        ema_12 = np.where(np.isnan(ema_12), close, ema_12)
        ema_36 = np.where(np.isnan(ema_36), close, ema_36)

        # ── Session filter ──
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # ── Regime masks ──
        is_mean_rev = overnight_ratio_arr > high_ratio
        is_momentum = overnight_ratio_arr < low_ratio
        gap_up = gap_direction_arr > 0
        gap_down = gap_direction_arr < 0
        ema_bullish = ema_12 > ema_36
        ema_bearish = ema_12 < ema_36

        # ── Entry signals ──
        # Mean-reversion regime: fade the gap
        #   gap DOWN → buy CE (fade: bounce back up)
        #   gap UP   → buy PE (fade: sell off from gap up)
        buy_ce_meanrev = (
            in_session & is_mean_rev & gap_down
            & (rsi_36 < rsi_oversold)
            & (close > ema_12)     # price already turning up
        )
        buy_pe_meanrev = (
            in_session & is_mean_rev & gap_up
            & (rsi_36 > rsi_overbought)
            & (close < ema_12)     # price already rolling over
        )

        # Momentum regime: follow the embedded institutional flow
        #   EMA bullish → buy CE (trend up, enter on RSI dip within bullish trend)
        #   EMA bearish → buy PE (trend down, enter on RSI pop within bearish trend)
        buy_ce_mom = (
            in_session & is_momentum & ema_bullish
            & (rsi_36 > 45.0) & (rsi_36 < rsi_overbought)
        )
        buy_pe_mom = (
            in_session & is_momentum & ema_bearish
            & (rsi_36 < 55.0) & (rsi_36 > rsi_oversold)
        )

        buy_ce = buy_ce_meanrev | buy_ce_mom
        buy_pe = buy_pe_meanrev | buy_pe_mom

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, 5.0),
            target_points=np.full(n, 8.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,
            max_trades_per_day=self.max_trades_per_day,
        )
