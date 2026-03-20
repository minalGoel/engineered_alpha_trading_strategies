"""ATR Expansion Breakout v1 — cursor_opus46max_050

Thesis: When current bar's True Range exceeds 2.5x ATR(20), it signals a
volatility regime change. If the expansion bar also breaks the session
high/low, closes in the direction of the break (upper/lower 40% of bar),
and the next bar follows through, it represents a genuine volatility
breakout with high follow-through probability.
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
    name = "cursor_opus46max_050"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 840     # 14:00
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("tr_ratio_min", default=2.5, low=2.0, high=3.5),
            TunableParam("close_pos_thresh", default=0.6, low=0.5, high=0.75),
            TunableParam("vol_mult", default=2.0, low=1.5, high=3.0),
            TunableParam("vix_max", default=28.0, low=20.0, high=35.0),
            TunableParam("stop_loss_pct", default=0.003, low=0.002, high=0.005),
            TunableParam("target_pct", default=0.006, low=0.004, high=0.010),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        tr_ratio_min = params.get("tr_ratio_min", 2.5)
        close_pos_thresh = params.get("close_pos_thresh", 0.6)
        vol_mult = params.get("vol_mult", 2.0)
        vix_max = params.get("vix_max", 28.0)
        stop_pct = params.get("stop_loss_pct", 0.003)
        tgt_pct = params.get("target_pct", 0.006)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        day_id = df["day_id"].to_numpy()

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── ATR(20) ──
        atr20 = _compute_atr(high, low, close, 20)
        # Also compute ATR(14) for StrategySignals
        atr14 = _compute_atr(high, low, close, 14)

        # ── True Range ──
        tr = np.zeros(n, dtype=np.float64)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))

        # ── TR / ATR ratio ──
        atr20_safe = np.where(atr20 > 0, atr20, 1e-10)
        tr_ratio = tr / atr20_safe

        # ── Bar range and close position ──
        bar_range = high - low
        bar_range_safe = np.where(bar_range > 0, bar_range, 1e-10)
        close_pos = (close - low) / bar_range_safe  # 1.0 = closed at high, 0.0 = closed at low

        # ── Session high/low ──
        sess_high = np.zeros(n, dtype=np.float64)
        sess_low = np.zeros(n, dtype=np.float64)
        prev_day = -1
        cur_h = 0.0
        cur_l = 1e18
        for i in range(n):
            if day_id[i] != prev_day:
                prev_day = day_id[i]
                cur_h = high[i]
                cur_l = low[i]
            else:
                cur_h = max(cur_h, high[i])
                cur_l = min(cur_l, low[i])
            sess_high[i] = cur_h
            sess_low[i] = cur_l

        # ── New session high/low on current bar ──
        new_sess_high = np.zeros(n, dtype=np.bool_)
        new_sess_low = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if day_id[i] == day_id[i-1]:
                if high[i] > sess_high[i-1]:
                    new_sess_high[i] = True
                if low[i] < sess_low[i-1]:
                    new_sess_low[i] = True

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > vol_mult * avg_vol

        # ── Expansion bar detection ──
        expansion_long = (
            (tr_ratio > tr_ratio_min) &
            new_sess_high &
            (close > opn) &
            (close_pos > close_pos_thresh) &
            vol_ok &
            (close > vwap)
        )
        expansion_short = (
            (tr_ratio > tr_ratio_min) &
            new_sess_low &
            (close < opn) &
            (close_pos < (1.0 - close_pos_thresh)) &
            vol_ok &
            (close < vwap)
        )

        # ── Follow-through confirmation: next bar closes above/below expansion bar midpoint ──
        exp_mid = (high + low) / 2.0

        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)

        vix_ok = vix < vix_max
        time_ok = (time_mins >= 570) & (time_mins <= 840)
        # Skip first 5 bars and last 10 bars of session
        bar_in_day = np.zeros(n, dtype=np.int32)
        prev_day = -1
        cnt = 0
        for i in range(n):
            if day_id[i] != prev_day:
                prev_day = day_id[i]
                cnt = 0
            else:
                cnt += 1
            bar_in_day[i] = cnt

        for i in range(1, n):
            if day_id[i] == day_id[i-1] and bar_in_day[i] > 5:
                if (expansion_long[i-1] and close[i] > exp_mid[i-1] and
                        vix_ok[i] and time_ok[i]):
                    long_entry[i] = True
                if (expansion_short[i-1] and close[i] < exp_mid[i-1] and
                        vix_ok[i] and time_ok[i]):
                    short_entry[i] = True

        # ── Signal exit: next 3 bars all have TR < ATR (volatility contraction) ──
        low_vol_count = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            if atr20[i] > 0 and tr[i] < atr20[i]:
                low_vol_count[i] = low_vol_count[i-1] + 1
            else:
                low_vol_count[i] = 0

        signal_exit_long = low_vol_count >= 3
        signal_exit_short = low_vol_count >= 3

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr14,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=tgt_pct,
            trailing_stop_pct=0.002,
            trailing_activate_pct=0.004,
            time_stop_bars=45,
        )
