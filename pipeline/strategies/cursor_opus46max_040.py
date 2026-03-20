"""Volume Breakout v1 — cursor_opus46max_040

Thesis: Volume exceeding 5x SMA(20) AND price breaking to a new session
high/low signals institutional participation. The 5x threshold is strict,
catching only the most extreme volume events. Confirmation bar required.
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
    name = "cursor_opus46max_040"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 840     # 14:00
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_spike_mult", default=5.0, low=3.0, high=8.0),
            TunableParam("close_pos_min", default=0.7, low=0.5, high=0.85),
            TunableParam("stop_loss_pct", default=0.0025, low=0.001, high=0.004),
            TunableParam("target_pct", default=0.005, low=0.003, high=0.008),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        spike_mult = params.get("vol_spike_mult", 5.0)
        cp_min = params.get("close_pos_min", 0.7)
        stop_pct = params.get("stop_loss_pct", 0.0025)
        tgt_pct = params.get("target_pct", 0.005)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
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

        # ── ATR ──
        atr = _compute_atr(high, low, close, 14)

        # ── Volume spike ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_spike = volume / avg_vol

        # ── Session high/low ──
        session_high = np.zeros(n, dtype=np.float64)
        session_low = np.zeros(n, dtype=np.float64)
        prev_day = -1
        for i in range(n):
            if day_id[i] != prev_day:
                prev_day = day_id[i]
                session_high[i] = high[i]
                session_low[i] = low[i]
            else:
                session_high[i] = max(session_high[i-1], high[i])
                session_low[i] = min(session_low[i-1], low[i])

        # ── Close position in bar ──
        bar_range = high - low
        bar_range_safe = np.where(bar_range > 0, bar_range, 1e-10)
        close_pos = (close - low) / bar_range_safe

        # ── Detect spike bar + new session extreme + follow-through ──
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)

        for i in range(2, n):
            prev = i - 1
            if time_mins[prev] < 575 or time_mins[prev] > 840:
                continue
            if time_mins[i] < 575 or time_mins[i] > 840:
                continue
            if day_id[i] != day_id[prev]:
                continue

            # Bullish: spike bar makes new session high, green, close near high
            if (vol_spike[prev] > spike_mult and
                high[prev] > session_high[prev-1] and
                close[prev] > opn[prev] and
                close[prev] > vwap[prev] and
                close_pos[prev] > cp_min and
                close[i] > session_high[prev]):  # confirmation
                long_entry[i] = True

            # Bearish
            if (vol_spike[prev] > spike_mult and
                low[prev] < session_low[prev-1] and
                close[prev] < opn[prev] and
                close[prev] < vwap[prev] and
                (1.0 - close_pos[prev]) > cp_min and
                close[i] < session_low[prev]):
                short_entry[i] = True

        # ── Signal exit: reverse vol spike ──
        signal_exit_long = np.zeros(n, dtype=np.bool_)
        signal_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            if vol_spike[i] > 3.0 and close[i] < opn[i]:
                signal_exit_long[i] = True
            if vol_spike[i] > 3.0 and close[i] > opn[i]:
                signal_exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=tgt_pct,
            trailing_stop_pct=0.002,
            trailing_activate_pct=0.003,
            time_stop_bars=45,
        )
