"""Tata Group Pairs — cursor_opus46max_133

Thesis: Tata group stocks share brand-sentiment correlation. When one stock
diverges from the group, it reverts. Proxied by stock vs index spread z-score
with correlation filter.
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
    name = "cursor_opus46max_133"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 915     # 15:15
    max_trades_per_day = 4
    assumptions = [
        "Group stock data not available; using stock vs index as proxy",
    ]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_entry", default=2.0, low=1.5, high=3.0),
            TunableParam("zscore_confirm", default=1.7, low=1.2, high=2.5),
            TunableParam("corr_thresh", default=0.3, low=0.1, high=0.5),
            TunableParam("lookback", default=45.0, low=20.0, high=60.0),
            TunableParam("vix_max", default=22.0, low=15.0, high=28.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        zs_entry = params.get("zscore_entry", 2.0)
        zs_confirm = params.get("zscore_confirm", 1.7)
        corr_thresh = params.get("corr_thresh", 0.3)
        lb = int(params.get("lookback", 45.0))
        vix_max = params.get("vix_max", 22.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        day_id = df["day_id"].to_numpy()
        idx_close = df["index_close"].to_numpy().astype(np.float64)
        idx_close = np.nan_to_num(idx_close, nan=0.0)

        atr = _compute_atr(high, low, close, 14)

        # 1-bar returns
        stock_ret1 = np.zeros(n, dtype=np.float64)
        idx_ret1 = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if close[i-1] > 0:
                stock_ret1[i] = (close[i] - close[i-1]) / close[i-1]
            if idx_close[i-1] > 0:
                idx_ret1[i] = (idx_close[i] - idx_close[i-1]) / idx_close[i-1]

        # Rolling correlation (60-bar)
        corr_lb = 60
        rolling_corr = np.zeros(n, dtype=np.float64)
        for i in range(corr_lb, n):
            s = stock_ret1[i-corr_lb:i]
            x = idx_ret1[i-corr_lb:i]
            ss, xs = np.std(s), np.std(x)
            if ss > 1e-10 and xs > 1e-10:
                rolling_corr[i] = np.corrcoef(s, x)[0, 1]

        # Cumulative returns from daily open
        day_open = np.zeros(n, dtype=np.float64)
        idx_day_open = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if i == 0 or day_id[i] != day_id[i-1]:
                day_open[i] = open_[i]
                idx_day_open[i] = idx_close[i] if idx_close[i] > 0 else 1.0
            else:
                day_open[i] = day_open[i-1]
                idx_day_open[i] = idx_day_open[i-1]

        safe_do = np.clip(day_open, 1e-10, None)
        safe_ido = np.clip(idx_day_open, 1e-10, None)
        spread = (close - day_open) / safe_do * 10000.0 - (idx_close - idx_day_open) / safe_ido * 10000.0

        # Z-score
        spread_mean = np.zeros(n, dtype=np.float64)
        spread_std = np.zeros(n, dtype=np.float64)
        for i in range(lb, n):
            w = spread[i-lb:i]
            spread_mean[i] = np.mean(w)
            spread_std[i] = np.std(w)
        spread_std = np.clip(spread_std, 1e-10, None)
        zscore = (spread - spread_mean) / spread_std

        vix_ok = vix < vix_max
        corr_ok = rolling_corr > corr_thresh
        time_ok = (time_mins >= 585) & (time_mins <= 900)

        long_signal = np.zeros(n, dtype=np.bool_)
        short_signal = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if zscore[i-1] < -zs_entry and zscore[i] > -zs_confirm:
                long_signal[i] = True
            if zscore[i-1] > zs_entry and zscore[i] < zs_confirm:
                short_signal[i] = True

        long_entry = long_signal & corr_ok & vix_ok & time_ok
        short_entry = short_signal & corr_ok & vix_ok & time_ok

        sig_exit_long = (zscore > 0.3) | (zscore < -3.0)
        sig_exit_short = (zscore < -0.3) | (zscore > 3.0)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=0.0025,
            time_stop_bars=45,
        )
