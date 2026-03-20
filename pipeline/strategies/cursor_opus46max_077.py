"""Low Vol Momentum v1 — cursor_opus46max_077

Thesis: Low-volatility stocks with positive intraday momentum (above VWAP,
EMA alignment) outperform high-vol stocks. Go long low-vol momentum stocks,
short high-vol weak stocks. Approximated for single-stock backtest by checking
if the stock's own realized vol is low relative to its history.
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
    name = "cursor_opus46max_077"
    is_long_only = False
    session_start = 585   # 09:45
    session_end = 910     # 15:10
    max_trades_per_day = 2

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_low_pctile", default=33.0, low=20.0, high=45.0),
            TunableParam("vol_high_pctile", default=67.0, low=55.0, high=80.0),
            TunableParam("vix_max", default=20.0, low=15.0, high=25.0),
            TunableParam("stop_loss_pct", default=0.005, low=0.003, high=0.008),
            TunableParam("target_pct", default=0.004, low=0.002, high=0.007),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vol_low_pctile = params.get("vol_low_pctile", 33.0)
        vol_high_pctile = params.get("vol_high_pctile", 67.0)
        vix_max = params.get("vix_max", 20.0)
        stop_pct = params.get("stop_loss_pct", 0.005)
        target_pct = params.get("target_pct", 0.004)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── Realized vol: rolling std of 1-min returns over 120 bars ──
        returns = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if close[i - 1] > 0:
                returns[i] = (close[i] - close[i - 1]) / close[i - 1]

        realized_vol = np.zeros(n, dtype=np.float64)
        vol_window = 120
        for i in range(vol_window, n):
            realized_vol[i] = np.std(returns[i - vol_window:i])

        # ── Vol percentile rank over 500 bars (~2 days) ──
        vol_pctile = np.full(n, 50.0, dtype=np.float64)
        pctile_window = 500
        for i in range(pctile_window, n):
            window = realized_vol[i - pctile_window:i + 1]
            if window[-1] > 0:
                vol_pctile[i] = (np.sum(window < window[-1]) / len(window)) * 100.0

        # ── EMA(9) and EMA(21) ──
        ema9 = _ema(close, 9)
        ema21 = _ema(close, 21)

        # ── Session high tracking ──
        day_id = df["day_id"].to_numpy()
        session_high_arr = np.zeros(n, dtype=np.float64)
        session_low_arr = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if i == 0 or day_id[i] != day_id[i - 1]:
                session_high_arr[i] = high[i]
                session_low_arr[i] = low[i]
            else:
                session_high_arr[i] = max(session_high_arr[i - 1], high[i])
                session_low_arr[i] = min(session_low_arr[i - 1], low[i])

        vix_ok = vix < vix_max

        # ── Long: low vol + above VWAP + EMA aligned up + new session high ──
        low_vol = vol_pctile < vol_low_pctile
        long_entry = (
            low_vol &
            (close > vwap) &
            (ema9 > ema21) &
            (close >= session_high_arr) &
            vix_ok
        )

        # ── Short: high vol + below VWAP + EMA aligned down + new session low ──
        high_vol = vol_pctile > vol_high_pctile
        short_entry = (
            high_vol &
            (close < vwap) &
            (ema9 < ema21) &
            (close <= session_low_arr) &
            vix_ok
        )

        # ── Signal exit: VIX spikes above 20 ──
        signal_exit_long = vix > 20.0
        signal_exit_short = vix > 20.0

        atr = _compute_atr(high, low, close, 14)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            time_stop_bars=300,
        )
