"""Adverse Selection Filter v1 — cursor_opus46max_107

Thesis: Flow toxicity varies throughout the day.  Use VPIN-like metric to
estimate when flow is informed vs benign.  Take mean-reversion when benign,
momentum when toxic.  Approximate VPIN using volume imbalance ratio.
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
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain[i] / avg_loss[i])
    return rsi


class Strategy(BaseStrategy):
    name = "cursor_opus46max_107"
    is_long_only = False
    session_start = 575   # 09:35
    session_end = 920     # 15:20
    max_trades_per_day = 12
    assumptions = ["VPIN approximated using absolute signed volume imbalance ratio"]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vpin_toxic_thresh", default=0.6, low=0.4, high=0.8),
            TunableParam("rsi_long_benign", default=30.0, low=20.0, high=40.0),
            TunableParam("rsi_short_benign", default=70.0, low=60.0, high=80.0),
            TunableParam("momentum_thresh_bps", default=10.0, low=5.0, high=20.0),
            TunableParam("vix_max", default=25.0, low=18.0, high=30.0),
            TunableParam("stop_loss_pct", default=0.001, low=0.0005, high=0.002),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vpin_thresh = params.get("vpin_toxic_thresh", 0.6)
        rsi_long_b = params.get("rsi_long_benign", 30.0)
        rsi_short_b = params.get("rsi_short_benign", 70.0)
        mom_thresh = params.get("momentum_thresh_bps", 10.0)
        vix_max = params.get("vix_max", 25.0)
        stop_pct = params.get("stop_loss_pct", 0.001)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(df["volume"].to_numpy().astype(np.float64), nan=1.0)
        volume = np.clip(volume, 1.0, None)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = np.nan_to_num(df_v["_vwap"].to_numpy().astype(np.float64), nan=close[0] if n > 0 else 0.0)

        # Signed volume using bar close position
        bar_range = high - low
        bar_range = np.where(bar_range < 1e-10, 1e-10, bar_range)
        sign_factor = 2.0 * (close - low) / bar_range - 1.0
        v_buy = np.where(sign_factor > 0, sign_factor * volume, 0.0)
        v_sell = np.where(sign_factor < 0, -sign_factor * volume, 0.0)

        # VPIN proxy: rolling abs(V_buy - V_sell) / V_total over 50 bars
        vpin_window = 50
        vpin = np.zeros(n, dtype=np.float64)
        for i in range(vpin_window - 1, n):
            total_buy = np.sum(v_buy[i - vpin_window + 1: i + 1])
            total_sell = np.sum(v_sell[i - vpin_window + 1: i + 1])
            total = total_buy + total_sell
            if total > 1e-10:
                vpin[i] = abs(total_buy - total_sell) / total

        is_toxic = vpin > vpin_thresh
        is_benign = ~is_toxic

        # RSI(14)
        rsi = _compute_rsi(close, 14)

        # Price momentum over 5 bars (bps)
        price_mom = np.zeros(n, dtype=np.float64)
        for i in range(5, n):
            if close[i-5] > 1e-10:
                price_mom[i] = (close[i] - close[i-5]) / close[i-5] * 10000.0

        # Volume filter
        avg_vol = np.ones(n, dtype=np.float64)
        for i in range(19, n):
            avg_vol[i] = np.mean(volume[i-19:i+1])
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > avg_vol

        vix_ok = vix < vix_max

        # RSI confirmation
        rsi_up = np.zeros(n, dtype=np.bool_)
        rsi_down = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            rsi_up[i] = rsi[i] > rsi[i-1]
            rsi_down[i] = rsi[i] < rsi[i-1]

        # Momentum sustained 2 bars
        mom_sustained_up = np.zeros(n, dtype=np.bool_)
        mom_sustained_down = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            mom_sustained_up[i] = price_mom[i] > mom_thresh and price_mom[i-1] > mom_thresh * 0.5
            mom_sustained_down[i] = price_mom[i] < -mom_thresh and price_mom[i-1] < -mom_thresh * 0.5

        # Benign long: RSI oversold, mean reversion
        benign_long = is_benign & (rsi < rsi_long_b) & (close > vwap * 0.998) & rsi_up & vol_ok & vix_ok
        # Toxic long: momentum continuation
        toxic_long = is_toxic & (price_mom > mom_thresh) & (close > vwap) & mom_sustained_up & vol_ok & vix_ok

        benign_short = is_benign & (rsi > rsi_short_b) & (close < vwap * 1.002) & rsi_down & vol_ok & vix_ok
        toxic_short = is_toxic & (price_mom < -mom_thresh) & (close < vwap) & mom_sustained_down & vol_ok & vix_ok

        long_entry = benign_long | toxic_long
        short_entry = benign_short | toxic_short

        # Signal exit: VPIN regime flips
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if is_toxic[i] != is_toxic[i-1]:
                sig_exit_long[i] = True
                sig_exit_short[i] = True

        atr = _compute_atr(high, low, close, 14)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=vwap.copy(),
            stop_loss_pct=stop_pct,
            target_pct=0.002,
            trailing_stop_pct=0.0007,
            trailing_activate_pct=0.0012,
            time_stop_bars=10,
        )
