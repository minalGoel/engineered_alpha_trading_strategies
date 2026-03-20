"""Correlation Regime v1 — cursor_opus46max_177

Thesis: When a stock's rolling 30-bar correlation with NIFTY 50 drops below 0.3
(decoupled), the stock is driven by idiosyncratic factors. If the stock is up
while the index is flat, the move tends to persist. Trade only in decoupled
regime, fading when alpha has been positive/negative for 10+ bars.
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
    name = "cursor_opus46max_177"
    is_long_only = False
    session_start = 585   # 09:45
    session_end = 910     # 15:10
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("corr_window", default=30.0, low=20.0, high=50.0),
            TunableParam("corr_decouple_thresh", default=0.30, low=0.15, high=0.45),
            TunableParam("alpha_thresh_bps", default=20.0, low=10.0, high=40.0),
            TunableParam("nifty_flat_thresh_bps", default=15.0, low=8.0, high=30.0),
            TunableParam("alpha_persist_bars", default=10.0, low=5.0, high=15.0),
            TunableParam("corr_recouple_thresh", default=0.50, low=0.35, high=0.65),
            TunableParam("stop_loss_pct", default=0.0025, low=0.0015, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        corr_win = int(params.get("corr_window", 30.0))
        decouple_th = params.get("corr_decouple_thresh", 0.30)
        alpha_th = params.get("alpha_thresh_bps", 20.0)
        nifty_flat_th = params.get("nifty_flat_thresh_bps", 15.0)
        persist_bars = int(params.get("alpha_persist_bars", 10.0))
        recouple_th = params.get("corr_recouple_thresh", 0.50)
        stop_pct = params.get("stop_loss_pct", 0.0025)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(index_close, nan=0.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # Stock and index returns
        stock_ret = np.zeros(n, dtype=np.float64)
        index_ret = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if close[i-1] > 0:
                stock_ret[i] = (close[i] - close[i-1]) / close[i-1]
            if index_close[i-1] > 0:
                index_ret[i] = (index_close[i] - index_close[i-1]) / index_close[i-1]

        # Rolling correlation
        rolling_corr = np.zeros(n, dtype=np.float64)
        for i in range(corr_win, n):
            sr = stock_ret[i-corr_win+1:i+1]
            ir = index_ret[i-corr_win+1:i+1]
            sr_m = sr - np.mean(sr)
            ir_m = ir - np.mean(ir)
            denom = np.sqrt(np.sum(sr_m**2) * np.sum(ir_m**2))
            if denom > 1e-15:
                rolling_corr[i] = np.sum(sr_m * ir_m) / denom

        decoupled = rolling_corr < decouple_th

        # NIFTY flat: abs(30-bar return) < threshold
        nifty_flat = np.zeros(n, dtype=np.bool_)
        for i in range(corr_win, n):
            if index_close[i-corr_win] > 0:
                nifty_ret = abs((index_close[i] - index_close[i-corr_win]) / index_close[i-corr_win]) * 10000
                nifty_flat[i] = nifty_ret < nifty_flat_th

        # Alpha: cumulative stock return - cumulative index return over window (bps)
        alpha_30 = np.zeros(n, dtype=np.float64)
        for i in range(corr_win, n):
            s_sum = np.sum(stock_ret[i-corr_win+1:i+1])
            i_sum = np.sum(index_ret[i-corr_win+1:i+1])
            alpha_30[i] = (s_sum - i_sum) * 10000

        # Persistence: alpha positive/negative for persist_bars
        alpha_pos_persist = np.zeros(n, dtype=np.bool_)
        alpha_neg_persist = np.zeros(n, dtype=np.bool_)
        for i in range(persist_bars, n):
            alpha_pos_persist[i] = all(alpha_30[i-j] > 0 for j in range(persist_bars))
            alpha_neg_persist[i] = all(alpha_30[i-j] < 0 for j in range(persist_bars))

        time_ok = (time_mins >= 585) & (time_mins <= 910)

        long_entry = (
            decoupled & (alpha_30 > alpha_th) & nifty_flat
            & (close > vwap) & alpha_pos_persist & time_ok
        )
        short_entry = (
            decoupled & (alpha_30 < -alpha_th) & nifty_flat
            & (close < vwap) & alpha_neg_persist & time_ok
        )

        # Signal exit: correlation rises above recouple threshold
        sig_exit_long = rolling_corr > recouple_th
        sig_exit_short = rolling_corr > recouple_th

        atr = _compute_atr(high, low, close, 14)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=0.0040,
            trailing_activate_pct=0.0025,
            trailing_stop_pct=0.0015,
            time_stop_bars=75,
        )
