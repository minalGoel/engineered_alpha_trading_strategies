"""ORB Wide Range Fade v1 — cursor_opus46max_090

Thesis: When the 15-min ORB range is >2x the 20-day average, the initial
move is exhausted. Fade the wide range by buying near ORB_low and selling
near ORB_high, targeting the ORB midpoint (mean reversion).
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


def _compute_rsi(close, period):
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    if n < period + 1:
        return rsi
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])
    if avg_loss > 0:
        rsi[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    else:
        rsi[period] = 100.0
    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        if avg_loss > 0:
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
        else:
            rsi[i + 1] = 100.0
    return rsi


class Strategy(BaseStrategy):
    name = "cursor_opus46max_090"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 870     # 14:30
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("wide_range_mult", default=2.0, low=1.5, high=3.0),
            TunableParam("price_pos_long", default=0.15, low=0.05, high=0.25),
            TunableParam("price_pos_short", default=0.85, low=0.75, high=0.95),
            TunableParam("rsi_long", default=25.0, low=15.0, high=35.0),
            TunableParam("rsi_short", default=75.0, low=65.0, high=85.0),
            TunableParam("stop_loss_pct", default=0.004, low=0.002, high=0.006),
            TunableParam("vix_max", default=22.0, low=18.0, high=28.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        wide_mult = params.get("wide_range_mult", 2.0)
        pos_long = params.get("price_pos_long", 0.15)
        pos_short = params.get("price_pos_short", 0.85)
        rsi_long = params.get("rsi_long", 25.0)
        rsi_short = params.get("rsi_short", 75.0)
        stop_pct = params.get("stop_loss_pct", 0.004)
        vix_max = params.get("vix_max", 22.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_min = df["time_minutes"].to_numpy()
        day_id = df["day_id"].to_numpy()

        # ── ORB 15-min + track prior ORB ranges for average ──
        orb_high = np.zeros(n, dtype=np.float64)
        orb_low = np.zeros(n, dtype=np.float64)
        orb_range = np.zeros(n, dtype=np.float64)
        orb_mid = np.zeros(n, dtype=np.float64)
        orb_computed = np.zeros(n, dtype=np.bool_)
        wide_range = np.zeros(n, dtype=np.bool_)

        prior_ranges = []
        cur_h = 0.0
        cur_l = 1e18

        for i in range(n):
            if i == 0 or day_id[i] != day_id[i - 1]:
                if i > 0 and orb_range[i - 1] > 0:
                    prior_ranges.append(orb_range[i - 1])
                    if len(prior_ranges) > 20:
                        prior_ranges = prior_ranges[-20:]
                cur_h = high[i]
                cur_l = low[i]
            if time_min[i] < 570:
                cur_h = max(cur_h, high[i])
                cur_l = min(cur_l, low[i])
            orb_high[i] = cur_h
            orb_low[i] = cur_l
            orb_range[i] = cur_h - cur_l
            orb_mid[i] = (cur_h + cur_l) / 2.0
            if time_min[i] >= 570:
                orb_computed[i] = True
                if len(prior_ranges) >= 5:
                    avg_range = np.mean(prior_ranges)
                    if orb_range[i] > wide_mult * avg_range and avg_range > 0:
                        wide_range[i] = True

        # ── Price position within ORB range ──
        orb_r_safe = np.clip(orb_range, 1e-10, None)
        price_position = np.where(orb_r_safe > 0, (close - orb_low) / orb_r_safe, 0.5)

        # ── RSI(14) ──
        rsi = _compute_rsi(close, 14)

        # ── Bounce/rejection confirmation ──
        bounce = (close > opn) & (price_position > 0.0)
        reject = (close < opn) & (price_position < 1.0)

        vix_ok = vix < vix_max
        time_ok = (time_min >= 570) & (time_min <= 660)

        long_entry = wide_range & (price_position < pos_long) & (rsi < rsi_long) & bounce & vix_ok & time_ok
        short_entry = wide_range & (price_position > pos_short) & (rsi > rsi_short) & reject & vix_ok & time_ok

        # ── Signal exit: ORB range extension ──
        signal_exit_long = close < (orb_low - 0.3 * orb_range)
        signal_exit_short = close > (orb_high + 0.3 * orb_range)

        atr = _compute_atr(high, low, close, 14)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=orb_mid.copy(),
            stop_loss_pct=stop_pct,
            use_target_indicator=True,
            time_stop_bars=120,
        )
