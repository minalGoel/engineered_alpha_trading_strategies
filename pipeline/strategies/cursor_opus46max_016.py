"""Intraday Range Reversion v1 — cursor_opus46max_016

Thesis: When price touches session high/low and forms a pin bar (hammer/shooting
star), it signals range boundary defence. Enter on next-bar confirmation of
bounce. Target is 50% reversion toward range midpoint.
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
    name = "cursor_opus46max_016"
    is_long_only = False
    session_start = 600   # 10:00
    session_end = 885     # 14:45
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("proximity_pct", default=0.001, low=0.0005, high=0.003),
            TunableParam("range_min_pct", default=0.8, low=0.4, high=1.5),
            TunableParam("pin_thresh", default=0.7, low=0.5, high=0.85),
            TunableParam("stop_loss_pct", default=0.0015, low=0.0008, high=0.003),
            TunableParam("trailing_pct", default=0.001, low=0.0005, high=0.002),
            TunableParam("trailing_activate", default=0.002, low=0.001, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        prox = params.get("proximity_pct", 0.001)
        range_min = params.get("range_min_pct", 0.8)
        pin_thresh = params.get("pin_thresh", 0.7)
        stop_pct = params.get("stop_loss_pct", 0.0015)
        trail_pct = params.get("trailing_pct", 0.001)
        trail_act = params.get("trailing_activate", 0.002)

        n = len(df)

        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        day_id = df["day_id"].to_numpy()

        # ── Session high / low (running) ──
        sess_hi = np.zeros(n, dtype=np.float64)
        sess_lo = np.zeros(n, dtype=np.float64)
        bar_from_open = np.zeros(n, dtype=np.int32)
        unique_days = np.unique(day_id)
        for d in unique_days:
            mask = day_id == d
            idx = np.where(mask)[0]
            rhi = -np.inf
            rlo = np.inf
            for j, gi in enumerate(idx):
                rhi = max(rhi, high[gi])
                rlo = min(rlo, low[gi])
                sess_hi[gi] = rhi
                sess_lo[gi] = rlo
                bar_from_open[gi] = j

        safe_lo = np.where(sess_lo > 0, sess_lo, 1.0)
        range_pct = (sess_hi - sess_lo) / safe_lo * 100.0

        sess_range = sess_hi - sess_lo
        sess_range = np.where(sess_range > 0, sess_range, 1e-10)

        # ── Bar range and pin bar detection ──
        bar_range = high - low
        bar_range_safe = np.where(bar_range > 0, bar_range, 1e-10)
        lower_wick_ratio = (close - low) / bar_range_safe  # hammer
        upper_wick_ratio = (high - close) / bar_range_safe  # shooting star

        # ── Near session low / high ──
        near_low = low <= sess_lo * (1.0 + prox)
        near_high = high >= sess_hi * (1.0 - prox)

        # ── Range midpoint for target ──
        range_mid = (sess_hi + sess_lo) / 2.0

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > avg_vol

        # ── ATR ──
        atr = _compute_atr(high, low, close, 14)

        # ── Confirmation: next bar confirms bounce ──
        confirm_long = np.zeros(n, dtype=np.bool_)
        confirm_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            # Pin bar at low + next bar confirms
            if near_low[i - 1] and (close[i - 1] > sess_lo[i - 1]) and (lower_wick_ratio[i - 1] > pin_thresh):
                if close[i] > high[i - 1]:
                    confirm_long[i] = True
            # Shooting star at high + next bar confirms
            if near_high[i - 1] and (close[i - 1] < sess_hi[i - 1]) and (upper_wick_ratio[i - 1] > pin_thresh):
                if close[i] < low[i - 1]:
                    confirm_short[i] = True

        # ── Filters ──
        time_ok = (time_mins >= 600) & (time_mins <= 885)
        vix_ok = (vix >= 14) & (vix <= 22)
        range_ok = range_pct > range_min
        bars_ok = bar_from_open >= 60

        # ── Entries ──
        long_entry = confirm_long & range_ok & vol_ok & vix_ok & time_ok & bars_ok
        short_entry = confirm_short & range_ok & vol_ok & vix_ok & time_ok & bars_ok

        # ── Signal exit: session extreme breached on volume ──
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if close[i] < sess_lo[i - 1] and volume[i] > 1.5 * avg_vol[i]:
                sig_exit_long[i] = True
            if close[i] > sess_hi[i - 1] and volume[i] > 1.5 * avg_vol[i]:
                sig_exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=range_mid.copy(),
            stop_loss_pct=stop_pct,
            use_target_indicator=True,
            trailing_stop_pct=trail_pct,
            trailing_activate_pct=trail_act,
            time_stop_bars=40,
        )
