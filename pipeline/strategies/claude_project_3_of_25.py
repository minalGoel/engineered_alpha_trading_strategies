"""Gap Fill Fade v1 — claude_project_3_of_25

Thesis: Opening gaps of moderate size (-0.3% to -1.5%) tend to fill.
Fade the gap with RSI confirmation; exit on partial fill or time.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_rsi(close, period):
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    if n < period + 1:
        return rsi
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.zeros(n, dtype=np.float64)
    avg_loss = np.zeros(n, dtype=np.float64)
    avg_gain[period] = np.mean(gain[1:period + 1])
    avg_loss[period] = np.mean(loss[1:period + 1])
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain[i] / avg_loss[i])
    return rsi


class Strategy(BaseStrategy):
    name = "gap_fill_fade_v1"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 720     # 12:00
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_min", default=0.003, low=0.001, high=0.006),
            TunableParam("gap_max", default=0.015, low=0.008, high=0.025),
            TunableParam("rsi_long_thresh", default=40.0, low=25.0, high=50.0),
            TunableParam("rsi_short_thresh", default=60.0, low=50.0, high=75.0),
            TunableParam("vix_max", default=22.0, low=15.0, high=30.0),
            TunableParam("stop_loss_pct", default=0.005, low=0.002, high=0.008),
            TunableParam("target_pct", default=0.005, low=0.002, high=0.010),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        gap_min = params.get("gap_min", 0.003)
        gap_max = params.get("gap_max", 0.015)
        rsi_long_th = params.get("rsi_long_thresh", 40.0)
        rsi_short_th = params.get("rsi_short_thresh", 60.0)
        vix_max = params.get("vix_max", 22.0)
        stop_pct = params.get("stop_loss_pct", 0.005)
        target_pct = params.get("target_pct", 0.005)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        day_id = df["day_id"].to_numpy()

        rsi = _compute_rsi(close, 7)

        # ── Compute gap pct per day (today open vs prev day close) ──
        gap_pct = np.zeros(n, dtype=np.float64)
        prev_day_close = np.nan
        cur_day = -1
        day_open = np.nan
        for i in range(n):
            if day_id[i] != cur_day:
                if not np.isnan(prev_day_close):
                    day_open = opn[i]
                    gap_pct_val = (day_open - prev_day_close) / prev_day_close if prev_day_close > 0 else 0.0
                else:
                    gap_pct_val = 0.0
                cur_day = day_id[i]
            gap_pct[i] = gap_pct_val if not np.isnan(gap_pct_val) else 0.0
            prev_day_close = close[i]  # will be overwritten each bar; last bar of day is final

        # Fix: compute prev_day_close properly
        prev_close_arr = np.zeros(n, dtype=np.float64)
        cur_day = -1
        last_close = np.nan
        for i in range(n):
            if day_id[i] != cur_day:
                prev_close_arr[i] = last_close
                cur_day = day_id[i]
            else:
                prev_close_arr[i] = prev_close_arr[i - 1] if i > 0 else np.nan
            last_close = close[i]

        gap_pct = np.where(prev_close_arr > 0, (opn - prev_close_arr) / prev_close_arr, 0.0)
        # Propagate day's gap to all bars of that day
        cur_day = -1
        day_gap = 0.0
        for i in range(n):
            if day_id[i] != cur_day:
                cur_day = day_id[i]
                day_gap = gap_pct[i]
            gap_pct[i] = day_gap
        gap_pct = np.nan_to_num(gap_pct, nan=0.0)

        # ── Bar count within each day ──
        bar_in_day = np.zeros(n, dtype=np.int32)
        cur_day = -1
        cnt = 0
        for i in range(n):
            if day_id[i] != cur_day:
                cur_day = day_id[i]
                cnt = 0
            bar_in_day[i] = cnt
            cnt += 1

        # ── Filters ──
        first_30 = bar_in_day < 30
        vix_ok = vix < vix_max
        # AUDIT FIX: missing session time filter caused entries outside session window
        time_ok = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # ── Entry ──
        # Gap down fade (long): gap between -gap_max and -gap_min + RSI < thresh
        gap_down = (gap_pct < -gap_min) & (gap_pct > -gap_max)
        long_entry = gap_down & (rsi < rsi_long_th) & first_30 & vix_ok & time_ok

        # Gap up fade (short): gap between +gap_min and +gap_max + RSI > thresh
        gap_up = (gap_pct > gap_min) & (gap_pct < gap_max)
        short_entry = gap_up & (rsi > rsi_short_th) & first_30 & vix_ok & time_ok

        # Breakeven after 50% fill ~ 0.5 * gap_pct median
        breakeven_pct = 0.5 * (gap_min + gap_max) / 2.0

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            breakeven_pct=breakeven_pct,
            time_stop_bars=120,
        )
