"""Parabolic SAR Momentum v1 — cursor_opus46max_033

Thesis: Parabolic SAR with AF 0.02/0.20 on 1-min bars provides mechanical
trend-following with built-in trailing stop. SAR flips confirmed by VWAP
alignment and ADX > 20 reduce whipsaws by ~40%. SAR itself acts as the
trailing stop for exits.
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


def _compute_psar(high, low, close, af_step=0.02, af_max=0.20):
    """Compute Parabolic SAR. Returns (psar, direction) arrays."""
    n = len(close)
    psar = np.zeros(n, dtype=np.float64)
    direction = np.zeros(n, dtype=np.int32)  # 1=bull, -1=bear

    if n < 2:
        return psar, direction

    # Initialize: assume bullish start
    bull = True
    af = af_step
    ep = high[0]
    psar[0] = low[0]
    direction[0] = 1

    for i in range(1, n):
        prev_psar = psar[i-1]

        if bull:
            psar[i] = prev_psar + af * (ep - prev_psar)
            psar[i] = min(psar[i], low[i-1])
            if i >= 2:
                psar[i] = min(psar[i], low[i-2])

            if low[i] < psar[i]:
                # Flip to bearish
                bull = False
                psar[i] = ep
                ep = low[i]
                af = af_step
                direction[i] = -1
            else:
                direction[i] = 1
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            psar[i] = prev_psar + af * (ep - prev_psar)
            psar[i] = max(psar[i], high[i-1])
            if i >= 2:
                psar[i] = max(psar[i], high[i-2])

            if high[i] > psar[i]:
                # Flip to bullish
                bull = True
                psar[i] = ep
                ep = high[i]
                af = af_step
                direction[i] = 1
            else:
                direction[i] = -1
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)

    return psar, direction


def _compute_adx(high, low, close, period):
    n = len(close)
    adx = np.zeros(n, dtype=np.float64)
    if n < period + 1:
        return adx
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0
    sm_tr = np.zeros(n, dtype=np.float64)
    sm_pdm = np.zeros(n, dtype=np.float64)
    sm_mdm = np.zeros(n, dtype=np.float64)
    sm_tr[period] = np.sum(tr[1:period+1])
    sm_pdm[period] = np.sum(plus_dm[1:period+1])
    sm_mdm[period] = np.sum(minus_dm[1:period+1])
    for i in range(period+1, n):
        sm_tr[i] = sm_tr[i-1] - sm_tr[i-1]/period + tr[i]
        sm_pdm[i] = sm_pdm[i-1] - sm_pdm[i-1]/period + plus_dm[i]
        sm_mdm[i] = sm_mdm[i-1] - sm_mdm[i-1]/period + minus_dm[i]
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    dx = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        if sm_tr[i] > 0:
            plus_di[i] = 100.0 * sm_pdm[i] / sm_tr[i]
            minus_di[i] = 100.0 * sm_mdm[i] / sm_tr[i]
        s = plus_di[i] + minus_di[i]
        if s > 0:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / s
    if n >= 2 * period:
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, n):
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    return adx


class Strategy(BaseStrategy):
    name = "cursor_opus46max_033"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 885     # 14:45
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("adx_min", default=20.0, low=15.0, high=30.0),
            TunableParam("af_step", default=0.02, low=0.01, high=0.04),
            TunableParam("af_max", default=0.20, low=0.10, high=0.30),
            TunableParam("stop_loss_pct", default=0.005, low=0.003, high=0.008),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        adx_min = params.get("adx_min", 20.0)
        af_step = params.get("af_step", 0.02)
        af_max = params.get("af_max", 0.20)
        stop_pct = params.get("stop_loss_pct", 0.005)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

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

        # ── Parabolic SAR ──
        psar, psar_dir = _compute_psar(high, low, close, af_step, af_max)

        # ── ADX ──
        adx = _compute_adx(high, low, close, 14)

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > avg_vol

        # ── SAR flip detection ──
        sar_flip_bull = np.zeros(n, dtype=np.bool_)
        sar_flip_bear = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if psar_dir[i] == 1 and psar_dir[i-1] == -1:
                sar_flip_bull[i] = True
            if psar_dir[i] == -1 and psar_dir[i-1] == 1:
                sar_flip_bear[i] = True

        # ── Flip bar quality: close in top/bottom 30% of range ──
        bar_range = high - low
        bar_range = np.where(bar_range > 0, bar_range, 1e-10)
        close_pos = (close - low) / bar_range

        # ── Whipsaw filter: count flips in last 30 bars ──
        flip_count_30 = np.zeros(n, dtype=np.int32)
        flips = (sar_flip_bull | sar_flip_bear).astype(np.int32)
        for i in range(n):
            start = max(0, i - 29)
            flip_count_30[i] = np.sum(flips[start:i+1])

        # ── Time filter ──
        time_ok = (time_mins >= 570) & (time_mins <= 885)

        # ── Entries ──
        long_entry = (
            sar_flip_bull & (adx > adx_min) & (close > vwap) &
            (close_pos > 0.70) & vol_ok & (flip_count_30 <= 4) & time_ok
        )
        short_entry = (
            sar_flip_bear & (adx > adx_min) & (close < vwap) &
            (close_pos < 0.30) & vol_ok & (flip_count_30 <= 4) & time_ok
        )

        # ── Signal exits: SAR re-flips ──
        signal_exit_long = sar_flip_bear
        signal_exit_short = sar_flip_bull

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            time_stop_bars=60,
        )
