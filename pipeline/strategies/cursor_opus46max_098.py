"""ORB Pullback Entry v1 — cursor_opus46max_098

Thesis: Instead of entering at ORB breakout, wait for a pullback to ORB_high (long)
or ORB_low (short) after confirmed breakout. The pullback confirms the ORB level as
support/resistance, eliminates many false breakouts, and provides a tighter stop
(0.3 * ORB_range below the level). Tradeoff: misses ~30% of breakouts that never pull back.
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


def _ema(arr, period):
    n = len(arr)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    out[0] = arr[0]
    alpha = 2.0 / (period + 1)
    for i in range(1, n):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


class Strategy(BaseStrategy):
    name = "cursor_opus46max_098"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 870     # 14:30
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("confirm_bars", default=3.0, low=2.0, high=5.0),
            TunableParam("pullback_window", default=30.0, low=15.0, high=50.0),
            TunableParam("pullback_tol_pct", default=0.0005, low=0.0002, high=0.001),
            TunableParam("stop_loss_pct", default=0.003, low=0.002, high=0.005),
            TunableParam("target_atr_mult", default=2.5, low=1.5, high=4.0),
            TunableParam("vix_low", default=12.0, low=8.0, high=16.0),
            TunableParam("vix_high", default=22.0, low=18.0, high=28.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        confirm_bars = int(params.get("confirm_bars", 3.0))
        pullback_win = int(params.get("pullback_window", 30.0))
        pullback_tol = params.get("pullback_tol_pct", 0.0005)
        stop_pct = params.get("stop_loss_pct", 0.003)
        target_atr_mult = params.get("target_atr_mult", 2.5)
        vix_lo = params.get("vix_low", 12.0)
        vix_hi = params.get("vix_high", 22.0)

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

        # -- ORB 15-min --
        orb_high = np.zeros(n, dtype=np.float64)
        orb_low = np.zeros(n, dtype=np.float64)
        orb_computed = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            if i == 0 or day_id[i] != day_id[i - 1]:
                cur_h = high[i]
                cur_l = low[i]
            if time_min[i] < 570:
                cur_h = max(cur_h, high[i])
                cur_l = min(cur_l, low[i])
            orb_high[i] = cur_h
            orb_low[i] = cur_l
            if time_min[i] >= 570:
                orb_computed[i] = True

        # -- EMA(9) --
        ema9 = _ema(close, 9)

        # -- Track confirmed breakouts (3+ consecutive bars above/below ORB) --
        breakout_up_confirmed = np.zeros(n, dtype=np.bool_)
        breakout_down_confirmed = np.zeros(n, dtype=np.bool_)
        breakout_up_bar = np.full(n, -1, dtype=np.int64)
        breakout_down_bar = np.full(n, -1, dtype=np.int64)

        consec_above = 0
        consec_below = 0
        first_above_bar = -1
        first_below_bar = -1
        prev_day = -1

        for i in range(n):
            if day_id[i] != prev_day:
                consec_above = 0
                consec_below = 0
                first_above_bar = -1
                first_below_bar = -1
                prev_day = day_id[i]

            if not orb_computed[i]:
                continue

            # Track consecutive closes above ORB high
            if close[i] > orb_high[i]:
                if consec_above == 0:
                    first_above_bar = i
                consec_above += 1
            else:
                consec_above = 0
                first_above_bar = -1

            if close[i] < orb_low[i]:
                if consec_below == 0:
                    first_below_bar = i
                consec_below += 1
            else:
                consec_below = 0
                first_below_bar = -1

            if consec_above >= confirm_bars:
                breakout_up_confirmed[i] = True
                breakout_up_bar[i] = first_above_bar
            if consec_below >= confirm_bars:
                breakout_down_confirmed[i] = True
                breakout_down_bar[i] = first_below_bar

        # -- Propagate breakout state forward within day --
        had_breakout_up = np.zeros(n, dtype=np.bool_)
        had_breakout_down = np.zeros(n, dtype=np.bool_)
        last_bup_bar = np.full(n, -1, dtype=np.int64)
        last_bdn_bar = np.full(n, -1, dtype=np.int64)
        for i in range(n):
            if i == 0 or day_id[i] != day_id[i - 1]:
                had_breakout_up[i] = breakout_up_confirmed[i]
                had_breakout_down[i] = breakout_down_confirmed[i]
                last_bup_bar[i] = breakout_up_bar[i] if breakout_up_confirmed[i] else -1
                last_bdn_bar[i] = breakout_down_bar[i] if breakout_down_confirmed[i] else -1
            else:
                had_breakout_up[i] = had_breakout_up[i - 1] or breakout_up_confirmed[i]
                had_breakout_down[i] = had_breakout_down[i - 1] or breakout_down_confirmed[i]
                last_bup_bar[i] = breakout_up_bar[i] if breakout_up_confirmed[i] else last_bup_bar[i - 1]
                last_bdn_bar[i] = breakout_down_bar[i] if breakout_down_confirmed[i] else last_bdn_bar[i - 1]

        # -- Detect pullback to ORB level --
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)

        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)

        vix_ok = (vix >= vix_lo) & (vix <= vix_hi)
        time_ok = (time_min >= 575) & (time_min <= 660)

        for i in range(1, n):
            if not orb_computed[i] or not vix_ok[i] or not time_ok[i]:
                continue
            if day_id[i] != day_id[i - 1]:
                continue

            orb_h = orb_high[i]
            orb_l = orb_low[i]
            orb_h_safe = orb_h if orb_h > 0 else 1.0
            orb_l_safe = orb_l if orb_l > 0 else 1.0

            # Long pullback: had breakout up, now price pulled back near ORB_high
            if had_breakout_up[i] and last_bup_bar[i] > 0:
                bars_since = i - last_bup_bar[i]
                if 0 < bars_since <= pullback_win:
                    # Low touches ORB_high area but close stays above
                    near_orb_h = abs(low[i] - orb_h) / orb_h_safe <= pullback_tol
                    close_above = close[i] > orb_h
                    bounce = close[i] > ema9[i]
                    vol_ok = volume[i] > avg_vol[i]
                    if near_orb_h and close_above and bounce and vol_ok:
                        long_entry[i] = True

            # Short pullback: had breakout down, now price pulled back near ORB_low
            if had_breakout_down[i] and last_bdn_bar[i] > 0:
                bars_since = i - last_bdn_bar[i]
                if 0 < bars_since <= pullback_win:
                    near_orb_l = abs(high[i] - orb_l) / orb_l_safe <= pullback_tol
                    close_below = close[i] < orb_l
                    rejection = close[i] < ema9[i]
                    vol_ok = volume[i] > avg_vol[i]
                    if near_orb_l and close_below and rejection and vol_ok:
                        short_entry[i] = True

        # -- Signal exit: close breaks ORB level against position --
        signal_exit_long = np.zeros(n, dtype=np.bool_)
        signal_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            if orb_computed[i] and orb_high[i] > 0:
                if close[i] < orb_high[i] * (1.0 - 0.001):
                    signal_exit_long[i] = True
            if orb_computed[i] and orb_low[i] > 0:
                if close[i] > orb_low[i] * (1.0 + 0.001):
                    signal_exit_short[i] = True

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
            time_stop_bars=200,
        )
