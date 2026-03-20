# AUDIT FIX: Add session window filter to entry signals (entries were firing outside session_start/session_end)
"""Vol Clustering v1 — cursor_opus46max_075

Thesis: GARCH(1,1)-based vol forecast identifies high-vol clusters where
breakout follow-through probability is elevated. Trade momentum during clusters.
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


def _ema(data, period):
    n = len(data)
    out = np.zeros(n, dtype=np.float64)
    out[0] = data[0]
    alpha = 2.0 / (period + 1)
    for i in range(1, n):
        out[i] = alpha * data[i] + (1 - alpha) * out[i-1]
    return out


class Strategy(BaseStrategy):
    name = "cursor_opus46max_075"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 910     # 15:10
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("garch_alpha", default=0.10, low=0.05, high=0.20),
            TunableParam("garch_beta", default=0.85, low=0.75, high=0.92),
            TunableParam("vol_pctile_thresh", default=75.0, low=60.0, high=90.0),
            TunableParam("cluster_bars_min", default=5.0, low=3.0, high=10.0),
            TunableParam("vix_min", default=13.0, low=8.0, high=16.0),
            TunableParam("stop_loss_pct", default=0.003, low=0.001, high=0.006),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        g_alpha = params.get("garch_alpha", 0.10)
        g_beta = params.get("garch_beta", 0.85)
        vol_pct_thresh = params.get("vol_pctile_thresh", 75.0)
        cluster_min = int(params.get("cluster_bars_min", 5.0))
        vix_min = params.get("vix_min", 13.0)
        stop_pct = params.get("stop_loss_pct", 0.003)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high_ = df["high"].to_numpy().astype(np.float64)
        low_ = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=15.0)
        time_min = df["time_minutes"].to_numpy()
        in_session = (time_min >= self.session_start) & (time_min <= self.session_end)

        atr14 = _compute_atr(high_, low_, close, 14)
        ema8 = _ema(close, 8)
        ema21 = _ema(close, 21)

        # Log returns
        ret = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if close[i-1] > 0:
                ret[i] = np.log(close[i] / close[i-1])

        # GARCH(1,1) variance forecast
        omega = 0.00001
        sigma2 = np.zeros(n, dtype=np.float64)
        sigma2[0] = omega / (1.0 - g_alpha - g_beta) if (g_alpha + g_beta) < 1.0 else 0.0001
        for i in range(1, n):
            sigma2[i] = omega + g_alpha * ret[i-1]**2 + g_beta * sigma2[i-1]
        sigma_forecast = np.sqrt(np.clip(sigma2, 1e-12, None))

        # Percentile rank of sigma_forecast (rolling 120)
        vol_pctile = np.zeros(n, dtype=np.float64)
        pct_lb = 120
        for i in range(pct_lb, n):
            seg = sigma_forecast[i - pct_lb + 1:i + 1]
            vol_pctile[i] = np.sum(seg <= sigma_forecast[i]) / pct_lb * 100.0

        # High-vol cluster counter
        high_vol = vol_pctile > vol_pct_thresh
        cluster_count = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            if high_vol[i]:
                cluster_count[i] = cluster_count[i-1] + 1
            else:
                cluster_count[i] = 0

        in_cluster = cluster_count >= cluster_min

        # Bar range ratio
        bar_range = high_ - low_
        range_ratio = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if atr14[i] > 1e-8:
                range_ratio[i] = bar_range[i] / atr14[i]

        # Momentum
        momentum = np.zeros(n, dtype=np.float64)
        for i in range(10, n):
            momentum[i] = close[i] - close[i-10]

        # Volume ratio
        avg_vol = np.zeros(n, dtype=np.float64)
        for i in range(30, n):
            avg_vol[i] = np.mean(volume[i-29:i+1])
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ratio = volume / avg_vol

        vix_ok = vix >= vix_min

        # New 20-bar high/low during cluster
        high_20 = np.zeros(n, dtype=np.float64)
        low_20 = np.full(n, np.inf, dtype=np.float64)
        for i in range(20, n):
            high_20[i] = np.max(close[i-19:i+1])
            low_20[i] = np.min(close[i-19:i+1])

        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            if not in_cluster[i] or not vix_ok[i] or not in_session[i]:
                continue
            if (close[i] >= high_20[i] and close[i] > ema8[i] and ema8[i] > ema21[i]
                    and momentum[i] > 0 and range_ratio[i] > 1.2 and vol_ratio[i] > 1.3):
                long_entry[i] = True
            if (close[i] <= low_20[i] and close[i] < ema8[i] and ema8[i] < ema21[i]
                    and momentum[i] < 0 and range_ratio[i] > 1.2 and vol_ratio[i] > 1.3):
                short_entry[i] = True

        # Signal exit: vol cluster ends or EMA cross
        signal_exit_long = np.zeros(n, dtype=np.bool_)
        signal_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if vol_pctile[i] < 50.0 or (ema8[i] < ema21[i] and ema8[i-1] >= ema21[i-1]):
                signal_exit_long[i] = True
            if vol_pctile[i] < 50.0 or (ema8[i] > ema21[i] and ema8[i-1] <= ema21[i-1]):
                signal_exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr14,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_atr_mult=2.0,
            trailing_stop_pct=0.0015,
            trailing_activate_pct=0.003,
            time_stop_bars=40,
        )
