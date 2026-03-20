# AUDIT FIX: Add session window filter to entry signals (entries were firing outside session_start/session_end)
"""VIX Mean Reversion v1 — cursor_opus46max_076

Thesis: India VIX exhibits strong intraday mean-reversion after spikes.
When VIX jumps >8% in 30 mins and starts declining, buy NIFTY constituents
for the recovery. Short when VIX crushes >6% and starts rising.
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
    name = "cursor_opus46max_076"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 915     # 15:15
    max_trades_per_day = 2

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vix_spike_pct", default=8.0, low=5.0, high=12.0),
            TunableParam("vix_crush_pct", default=6.0, low=3.0, high=9.0),
            TunableParam("vix_z_long", default=1.5, low=1.0, high=2.5),
            TunableParam("vix_z_short", default=-1.0, low=-2.0, high=-0.5),
            TunableParam("rsi_long_thresh", default=35.0, low=25.0, high=45.0),
            TunableParam("stop_loss_pct", default=0.005, low=0.003, high=0.008),
            TunableParam("target_pct", default=0.004, low=0.002, high=0.007),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vix_spike_pct = params.get("vix_spike_pct", 8.0)
        vix_crush_pct = params.get("vix_crush_pct", 6.0)
        vix_z_long = params.get("vix_z_long", 1.5)
        vix_z_short = params.get("vix_z_short", -1.0)
        rsi_long_thresh = params.get("rsi_long_thresh", 35.0)
        stop_pct = params.get("stop_loss_pct", 0.005)
        target_pct = params.get("target_pct", 0.004)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(index_close, nan=0.0)
        day_id = df["day_id"].to_numpy()
        time_min = df["time_minutes"].to_numpy()
        in_session = (time_min >= self.session_start) & (time_min <= self.session_end)

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── VIX 30-bar change ──
        vix_30_change = np.zeros(n, dtype=np.float64)
        for i in range(30, n):
            if vix[i - 30] > 0:
                vix_30_change[i] = (vix[i] - vix[i - 30]) / vix[i - 30] * 100.0

        # ── VIX z-score (rolling 120-bar as proxy for ~20d) ──
        vix_z = np.zeros(n, dtype=np.float64)
        zw = 120
        for i in range(zw, n):
            window = vix[i - zw:i + 1]
            mu = np.mean(window)
            sd = np.std(window)
            if sd > 0:
                vix_z[i] = (vix[i] - mu) / sd

        # ── VIX declining 3 bars (confirmation for long) ──
        vix_declining = np.zeros(n, dtype=np.bool_)
        for i in range(3, n):
            if vix[i] < vix[i - 1] and vix[i - 1] < vix[i - 2] and vix[i - 2] < vix[i - 3]:
                vix_declining[i] = True

        # ── VIX rising 3 bars (confirmation for short) ──
        vix_rising = np.zeros(n, dtype=np.bool_)
        for i in range(3, n):
            if vix[i] > vix[i - 1] and vix[i - 1] > vix[i - 2] and vix[i - 2] > vix[i - 3]:
                vix_rising[i] = True

        # ── Session high tracking for drawdown ──
        session_high = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if i == 0 or day_id[i] != day_id[i - 1]:
                session_high[i] = close[i]
            else:
                session_high[i] = max(session_high[i - 1], close[i])
        drawdown = np.where(session_high > 0, (close - session_high) / session_high * 100.0, 0.0)

        # ── RSI(14) on close ──
        rsi = _compute_rsi(close, 14)

        # ── Entry signals ──
        long_entry = (
            (vix_30_change > vix_spike_pct) &
            (vix_z > vix_z_long) &
            (drawdown < -0.5) &
            (rsi < rsi_long_thresh) &
            vix_declining &
            (vix < 25.0) &
            in_session
        )

        short_entry = (
            (vix_30_change < -vix_crush_pct) &
            (vix_z < vix_z_short) &
            vix_rising &
            in_session
        )

        # ── Signal exit: VIX makes new session high while long ──
        signal_exit_long = np.zeros(n, dtype=np.bool_)
        signal_exit_short = np.zeros(n, dtype=np.bool_)

        atr = _compute_atr(high, low, close, 14)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=vwap.copy(),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            breakeven_pct=0.003,
            time_stop_bars=150,
        )
