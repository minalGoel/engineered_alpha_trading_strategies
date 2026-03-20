"""ORB Narrow Range (NR4) v1 — cursor_opus46max_089

Thesis: When the 15-min ORB range is narrower than prior 4 sessions' ORB
ranges, subsequent breakout has higher follow-through. The compression signals
consensus before a directional move. Wider 3x range target since stop is tight.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_atr(high, low, close, period):
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    if n < period:
        return atr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


class Strategy(BaseStrategy):
    name = "cursor_opus46max_089"
    is_long_only = False
    session_start = 555   # 09:15
    session_end = 910     # 15:10
    max_trades_per_day = 2

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_mult", default=1.3, low=1.0, high=2.0),
            TunableParam("stop_loss_pct", default=0.003, low=0.002, high=0.005),
            TunableParam("target_atr_mult", default=3.0, low=2.0, high=5.0),
            TunableParam("trailing_stop_pct", default=0.003, low=0.001, high=0.005),
            TunableParam("vix_low", default=12.0, low=8.0, high=15.0),
            TunableParam("vix_high", default=20.0, low=17.0, high=25.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vol_mult = params.get("vol_mult", 1.3)
        stop_pct = params.get("stop_loss_pct", 0.003)
        target_atr_mult = params.get("target_atr_mult", 3.0)
        trailing_pct = params.get("trailing_stop_pct", 0.003)
        vix_lo = params.get("vix_low", 12.0)
        vix_hi = params.get("vix_high", 20.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_min = df["time_minutes"].to_numpy()
        day_id = df["day_id"].to_numpy()

        # ── ORB 15-min per day + track prior day ORB ranges ──
        orb_high = np.zeros(n, dtype=np.float64)
        orb_low = np.zeros(n, dtype=np.float64)
        orb_range = np.zeros(n, dtype=np.float64)
        orb_computed = np.zeros(n, dtype=np.bool_)
        nr4_condition = np.zeros(n, dtype=np.bool_)

        prior_ranges = []  # stores ORB ranges of prior sessions
        cur_h = 0.0
        cur_l = 1e18

        for i in range(n):
            if i == 0 or day_id[i] != day_id[i - 1]:
                # Save prior day's ORB range
                if i > 0 and orb_range[i - 1] > 0:
                    prior_ranges.append(orb_range[i - 1])
                    if len(prior_ranges) > 4:
                        prior_ranges = prior_ranges[-4:]
                cur_h = high[i]
                cur_l = low[i]
            if time_min[i] < 570:
                cur_h = max(cur_h, high[i])
                cur_l = min(cur_l, low[i])
            orb_high[i] = cur_h
            orb_low[i] = cur_l
            orb_range[i] = cur_h - cur_l
            if time_min[i] >= 570:
                orb_computed[i] = True
                # NR4: today's ORB range < min of prior 4
                if len(prior_ranges) >= 4:
                    if orb_range[i] < min(prior_ranges) and orb_range[i] > 0:
                        nr4_condition[i] = True

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > vol_mult * avg_vol

        # ── Bar position ──
        bar_range = high - low
        bar_range = np.clip(bar_range, 1e-10, None)
        bar_pos = (close - low) / bar_range
        bullish_confirm = bar_pos > 0.75
        bearish_confirm = bar_pos < 0.25

        vix_ok = (vix >= vix_lo) & (vix <= vix_hi)
        time_ok = (time_min >= 570) & (time_min <= 630)

        long_entry = nr4_condition & (close > orb_high) & vol_ok & bullish_confirm & vix_ok & time_ok
        short_entry = nr4_condition & (close < orb_low) & vol_ok & bearish_confirm & vix_ok & time_ok

        # ── Signal exit: whipsaw through ORB ──
        signal_exit_long = close < orb_low
        signal_exit_short = close > orb_high

        atr = _compute_atr(high, low, close, 14)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_atr_mult=target_atr_mult,
            trailing_stop_pct=trailing_pct,
            trailing_activate_pct=trailing_pct,
            time_stop_bars=180,
        )
