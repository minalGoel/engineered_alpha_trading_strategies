"""Order Flow Imbalance v1 — cursor_opus46max_106

Thesis: Cumulative order flow imbalance (OFI) predicts short-term price
direction.  When OFI diverges from price (flow rising, price flat), it
signals latent buying pressure.  Proxy OFI with signed volume using
close position within bar.
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


def _ema(arr, period):
    n = len(arr)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    out[0] = arr[0]
    alpha = 2.0 / (period + 1)
    for i in range(1, n):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
    return out


class Strategy(BaseStrategy):
    name = "cursor_opus46max_106"
    is_long_only = False
    session_start = 575   # 09:35
    session_end = 920     # 15:20
    max_trades_per_day = 15
    assumptions = ["OFI proxied via signed volume using bar close position"]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vix_max", default=22.0, low=15.0, high=28.0),
            TunableParam("stop_loss_pct", default=0.001, low=0.0005, high=0.002),
            TunableParam("target_pct", default=0.002, low=0.001, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vix_max = params.get("vix_max", 22.0)
        stop_pct = params.get("stop_loss_pct", 0.001)
        target_pct = params.get("target_pct", 0.002)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(df["volume"].to_numpy().astype(np.float64), nan=1.0)
        volume = np.clip(volume, 1.0, None)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)
        day_id = df["day_id"].to_numpy()

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = np.nan_to_num(df_v["_vwap"].to_numpy().astype(np.float64), nan=close[0] if n > 0 else 0.0)

        # Signed volume: bar close position * volume
        bar_range = high - low
        bar_range = np.where(bar_range < 1e-10, 1e-10, bar_range)
        sign_factor = 2.0 * (close - low) / bar_range - 1.0  # -1 to +1
        ofi_1min = sign_factor * volume

        # Cumulative OFI (reset daily)
        cum_ofi = np.zeros(n, dtype=np.float64)
        cum_ofi[0] = ofi_1min[0]
        for i in range(1, n):
            if day_id[i] != day_id[i-1]:
                cum_ofi[i] = ofi_1min[i]
            else:
                cum_ofi[i] = cum_ofi[i-1] + ofi_1min[i]

        # OFI momentum: EMA(5) - EMA(15) of ofi_1min
        ofi_fast = _ema(ofi_1min, 5)
        ofi_slow = _ema(ofi_1min, 15)
        ofi_momentum = ofi_fast - ofi_slow

        # Price change over 5 bars
        price_chg = np.zeros(n, dtype=np.float64)
        for i in range(5, n):
            price_chg[i] = close[i] - close[i-5]

        # Divergence: OFI says buy but price flat/down (or vice versa)
        div_long = (ofi_momentum > 0) & (price_chg <= 0)
        div_short = (ofi_momentum < 0) & (price_chg >= 0)

        # OFI confirmation: ofi_1min positive for 2 of last 3 bars
        ofi_confirm_long = np.zeros(n, dtype=np.bool_)
        ofi_confirm_short = np.zeros(n, dtype=np.bool_)
        for i in range(2, n):
            pos_count = int(ofi_1min[i] > 0) + int(ofi_1min[i-1] > 0) + int(ofi_1min[i-2] > 0)
            neg_count = int(ofi_1min[i] < 0) + int(ofi_1min[i-1] < 0) + int(ofi_1min[i-2] < 0)
            ofi_confirm_long[i] = pos_count >= 2
            ofi_confirm_short[i] = neg_count >= 2

        vix_ok = vix < vix_max

        long_entry = ((ofi_momentum > 0) & (cum_ofi > 0) & div_long &
                      (close > vwap * 0.999) & ofi_confirm_long & vix_ok)
        short_entry = ((ofi_momentum < 0) & (cum_ofi < 0) & div_short &
                       (close < vwap * 1.001) & ofi_confirm_short & vix_ok)

        # Signal exit: OFI momentum crosses zero
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if ofi_momentum[i] <= 0 and ofi_momentum[i-1] > 0:
                sig_exit_long[i] = True
            if ofi_momentum[i] >= 0 and ofi_momentum[i-1] < 0:
                sig_exit_short[i] = True

        atr = _compute_atr(high, low, close, 14)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            trailing_stop_pct=0.0006,
            trailing_activate_pct=0.0012,
            time_stop_bars=8,
        )
