"""LASSO Feature Selection v1 — cursor_opus46max_195

Thesis: LASSO regression automatically selects the most predictive features
by shrinking unimportant coefficients to zero. With 10 features (RSI, MACD,
BB, VWAP dist, volume ratio, ATR, OBV slope, NIFTY return, VIX, momentum),
the non-zero coefficients identify currently predictive indicators.
Simplified: use rolling OLS with L1-like feature trimming.
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
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i]) / period
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain[i] / avg_loss[i])
    return rsi


def _compute_ema(data, period):
    n = len(data)
    ema = np.zeros(n, dtype=np.float64)
    if n == 0:
        return ema
    ema[0] = data[0]
    k = 2.0 / (period + 1.0)
    for i in range(1, n):
        ema[i] = data[i] * k + ema[i-1] * (1.0 - k)
    return ema


class Strategy(BaseStrategy):
    name = "cursor_opus46max_195"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 910     # 15:10
    max_trades_per_day = 7

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("prediction_thresh_bps", default=15.0, low=8.0, high=25.0),
            TunableParam("confirm_bars", default=3.0, low=2.0, high=5.0),
            TunableParam("min_active_features", default=3.0, low=2.0, high=5.0),
            TunableParam("stop_loss_pct", default=0.0015, low=0.001, high=0.003),
            TunableParam("trailing_activate_pct", default=0.0015, low=0.001, high=0.0025),
            TunableParam("trailing_stop_pct", default=0.0008, low=0.0005, high=0.0015),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        pred_th = params.get("prediction_thresh_bps", 15.0)
        confirm = int(params.get("confirm_bars", 3.0))
        min_feat = int(params.get("min_active_features", 3.0))
        stop_pct = params.get("stop_loss_pct", 0.0015)
        trail_act = params.get("trailing_activate_pct", 0.0015)
        trail_pct = params.get("trailing_stop_pct", 0.0008)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(index_close, nan=0.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # Feature 1: VWAP distance
        safe_vwap = np.where(vwap > 0, vwap, 1.0)
        f_vwap_dist = (close - vwap) / safe_vwap * 10000

        # Feature 2: RSI
        f_rsi = _compute_rsi(close, 14)

        # Feature 3: Momentum 10
        f_mom = np.zeros(n, dtype=np.float64)
        for i in range(10, n):
            if close[i-10] > 0:
                f_mom[i] = (close[i] - close[i-10]) / close[i-10] * 10000

        # Feature 4: Volume ratio
        avg_vol = np.zeros(n, dtype=np.float64)
        for i in range(19, n):
            avg_vol[i] = np.mean(volume[i-19:i+1])
        avg_vol = np.clip(avg_vol, 1.0, None)
        f_vol_ratio = volume / avg_vol

        # Feature 5: ATR normalized
        atr = _compute_atr(high, low, close, 14)
        safe_close = np.where(close > 0, close, 1.0)
        f_atr_norm = atr / safe_close * 10000

        # Feature 6: Index return 20
        f_idx_ret = np.zeros(n, dtype=np.float64)
        for i in range(20, n):
            if index_close[i-20] > 0:
                f_idx_ret[i] = (index_close[i] - index_close[i-20]) / index_close[i-20] * 10000

        # Feature 7: EMA(12) - EMA(26) (MACD proxy)
        ema12 = _compute_ema(close, 12)
        ema26 = _compute_ema(close, 26)
        f_macd = (ema12 - ema26) / safe_close * 10000

        # Composite score using feature agreement (simplified LASSO)
        # Count how many features agree on direction
        prediction = np.zeros(n, dtype=np.float64)
        active_features = np.zeros(n, dtype=np.int32)

        for i in range(30, n):
            score = 0.0
            active = 0

            # VWAP distance: bullish if below vwap
            if abs(f_vwap_dist[i]) > 5:
                score += -np.sign(f_vwap_dist[i]) * min(abs(f_vwap_dist[i]), 30)
                active += 1

            # RSI: bullish if oversold
            if f_rsi[i] < 35:
                score += 15
                active += 1
            elif f_rsi[i] > 65:
                score += -15
                active += 1

            # Momentum: follow
            if abs(f_mom[i]) > 10:
                score += np.sign(f_mom[i]) * min(abs(f_mom[i]) * 0.5, 20)
                active += 1

            # Volume: amplifier (not directional by itself)
            if f_vol_ratio[i] > 1.5:
                score *= 1.2
                active += 1

            # Index return: follow
            if abs(f_idx_ret[i]) > 10:
                score += np.sign(f_idx_ret[i]) * min(abs(f_idx_ret[i]) * 0.3, 15)
                active += 1

            # MACD: follow
            if abs(f_macd[i]) > 5:
                score += np.sign(f_macd[i]) * min(abs(f_macd[i]) * 0.5, 15)
                active += 1

            # ATR: reduce in high vol
            if f_atr_norm[i] > 30:
                score *= 0.8
                active += 1

            prediction[i] = score
            active_features[i] = active

        # Prediction persistence
        pred_pos_consec = np.zeros(n, dtype=np.int32)
        pred_neg_consec = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            if prediction[i] > pred_th * 0.7:
                pred_pos_consec[i] = pred_pos_consec[i-1] + 1
            else:
                pred_pos_consec[i] = 0
            if prediction[i] < -pred_th * 0.7:
                pred_neg_consec[i] = pred_neg_consec[i-1] + 1
            else:
                pred_neg_consec[i] = 0

        time_ok = (time_mins >= 570) & (time_mins <= 910)

        long_entry = (
            (prediction > pred_th)
            & (active_features >= min_feat)
            & (active_features <= 7)
            & (close > vwap)
            & (pred_pos_consec >= confirm)
            & time_ok
        )
        short_entry = (
            (prediction < -pred_th)
            & (active_features >= min_feat)
            & (active_features <= 7)
            & (close < vwap)
            & (pred_neg_consec >= confirm)
            & time_ok
        )

        # Signal exit: prediction flips sign
        sig_exit_long = prediction < 0
        sig_exit_short = prediction > 0

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            trailing_activate_pct=trail_act,
            trailing_stop_pct=trail_pct,
            time_stop_bars=45,
        )
