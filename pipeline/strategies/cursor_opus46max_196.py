"""Post-Halt Momentum v1 — cursor_opus46max_196

Thesis: When a stock hits circuit limits and is halted, reopening shows strong
momentum continuation ~65% of the time as pent-up order flow resolves.
Adapted: detect large gap-up/gap-down opens (proxy for halt-like events)
with volume surge and trade the continuation after 2-bar confirmation.
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
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    if n < period:
        return atr
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr


class Strategy(BaseStrategy):
    name = "cursor_opus46max_196"
    is_long_only = False
    session_start = 555   # 09:15
    session_end = 915     # 15:15
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_thresh_bps", default=100.0, low=50.0, high=200.0),
            TunableParam("vol_surge_mult", default=3.0, low=2.0, high=5.0),
            TunableParam("confirm_bars", default=2.0, low=1.0, high=4.0),
            TunableParam("stop_loss_pct", default=0.003, low=0.002, high=0.005),
            TunableParam("target_pct", default=0.005, low=0.003, high=0.008),
            TunableParam("trailing_activate_pct", default=0.003, low=0.002, high=0.005),
            TunableParam("trailing_stop_pct", default=0.0015, low=0.001, high=0.003),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        gap_th = params.get("gap_thresh_bps", 100.0)
        vol_mult = params.get("vol_surge_mult", 3.0)
        confirm = int(params.get("confirm_bars", 2.0))
        stop_pct = params.get("stop_loss_pct", 0.003)
        target_pct = params.get("target_pct", 0.005)
        trail_act = params.get("trailing_activate_pct", 0.003)
        trail_pct = params.get("trailing_stop_pct", 0.0015)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        day_id = df["day_id"].to_numpy()

        # Detect large gaps at session open
        gap_up = np.zeros(n, dtype=np.bool_)
        gap_down = np.zeros(n, dtype=np.bool_)
        prev_close_at_open = np.zeros(n, dtype=np.float64)

        for i in range(1, n):
            if day_id[i] != day_id[i-1]:
                prev_close_at_open[i] = close[i-1]
                if close[i-1] > 0:
                    gap_bps = (opn[i] - close[i-1]) / close[i-1] * 10000
                    if gap_bps > gap_th:
                        gap_up[i] = True
                    elif gap_bps < -gap_th:
                        gap_down[i] = True

        # Propagate gap signal for first ~10 bars of the day
        gap_up_session = np.zeros(n, dtype=np.bool_)
        gap_down_session = np.zeros(n, dtype=np.bool_)
        in_gap_up = False
        in_gap_down = False
        gap_bar_count = 0

        for i in range(n):
            if day_id[i] != day_id[i-1] if i > 0 else True:
                in_gap_up = gap_up[i]
                in_gap_down = gap_down[i]
                gap_bar_count = 0
            gap_bar_count += 1
            if gap_bar_count <= 10:
                gap_up_session[i] = in_gap_up
                gap_down_session[i] = in_gap_down

        # Volume surge
        avg_vol = np.zeros(n, dtype=np.float64)
        for i in range(19, n):
            avg_vol[i] = np.mean(volume[i-19:i+1])
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_surge = volume > (vol_mult * avg_vol)

        # Confirmation: consecutive bars in gap direction
        confirm_up = np.zeros(n, dtype=np.bool_)
        confirm_down = np.zeros(n, dtype=np.bool_)
        consec_up = np.zeros(n, dtype=np.int32)
        consec_down = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            if day_id[i] != day_id[i-1]:
                consec_up[i] = 0
                consec_down[i] = 0
            else:
                consec_up[i] = (consec_up[i-1] + 1) if close[i] > close[i-1] else 0
                consec_down[i] = (consec_down[i-1] + 1) if close[i] < close[i-1] else 0
        confirm_up = consec_up >= confirm
        confirm_down = consec_down >= confirm

        vix_ok = vix < 25.0

        long_entry = (
            gap_up_session & vol_surge & confirm_up & vix_ok
        )
        short_entry = (
            gap_down_session & vol_surge & confirm_down & vix_ok
        )

        # Signal exit: momentum fades (price reverses)
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(2, n):
            if close[i] < close[i-1] and close[i-1] < close[i-2]:
                sig_exit_long[i] = True
            if close[i] > close[i-1] and close[i-1] > close[i-2]:
                sig_exit_short[i] = True

        atr = _compute_atr(high, low, close, 14)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            trailing_activate_pct=trail_act,
            trailing_stop_pct=trail_pct,
            time_stop_bars=30,
        )
