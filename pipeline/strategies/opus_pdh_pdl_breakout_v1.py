"""Previous Day High/Low Breakout — Opus_13

Thesis: Breakouts beyond the prior session's high or low, confirmed by
volume and VWAP, often trigger continuation moves.  We require two
consecutive closes above PDH (or below PDL) to reduce false breakouts.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Wilder's ATR."""
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i - 1]),
                     abs(low[i] - close[i - 1]))
    if n < period:
        return atr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


class Strategy(BaseStrategy):
    name = "opus_pdh_pdl_breakout_v1"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 870     # 14:30
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("breakout_bps", default=15.0, low=5.0, high=30.0),
            TunableParam("vol_surge_mult", default=2.0, low=1.5, high=3.0),
            TunableParam("vix_max", default=22.0, low=15.0, high=30.0),
            TunableParam("stop_loss_pct", default=0.003, low=0.002, high=0.005),
            TunableParam("target_pct", default=0.005, low=0.003, high=0.008),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        breakout_bps = params.get("breakout_bps", 15.0)
        vol_surge = params.get("vol_surge_mult", 2.0)
        vix_max = params.get("vix_max", 22.0)
        stop_pct = params.get("stop_loss_pct", 0.003)
        target_pct = params.get("target_pct", 0.005)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        day_id = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── VWAP ──
        df_vwap = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns(
            (pl.col("_ctv") / pl.col("_cv")).alias("_vwap")
        )
        vwap = df_vwap["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── PDH / PDL per day_id ──
        pdh = np.zeros(n, dtype=np.float64)
        pdl = np.zeros(n, dtype=np.float64)
        prev_day_high = np.nan
        prev_day_low = np.nan
        cur_day_high = high[0]
        cur_day_low = low[0]
        cur_day = day_id[0]
        for i in range(n):
            if day_id[i] != cur_day:
                prev_day_high = cur_day_high
                prev_day_low = cur_day_low
                cur_day = day_id[i]
                cur_day_high = high[i]
                cur_day_low = low[i]
            else:
                if high[i] > cur_day_high:
                    cur_day_high = high[i]
                if low[i] < cur_day_low:
                    cur_day_low = low[i]
            pdh[i] = prev_day_high
            pdl[i] = prev_day_low

        pdh = np.nan_to_num(pdh, nan=0.0)
        pdl = np.nan_to_num(pdl, nan=0.0)

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > vol_surge * avg_vol

        # ── ATR for output ──
        atr = _compute_atr(high, low, close, 14)

        # ── Breakout threshold ──
        bps_mult = breakout_bps / 10000.0

        # Close above PDH for 2 consecutive bars
        above_pdh = (pdh > 0) & (close > pdh * (1.0 + bps_mult))
        below_pdl = (pdl > 0) & (close < pdl * (1.0 - bps_mult))

        two_above = np.zeros(n, dtype=np.bool_)
        two_below = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if above_pdh[i] and above_pdh[i - 1] and day_id[i] == day_id[i - 1]:
                two_above[i] = True
            if below_pdl[i] and below_pdl[i - 1] and day_id[i] == day_id[i - 1]:
                two_below[i] = True

        vix_ok = vix < vix_max
        vwap_ok_long = close > vwap

        long_entry = two_above & vol_ok & vwap_ok_long & vix_ok
        short_entry = two_below & vol_ok & vix_ok

        # ── Signal exit: close crosses back inside PDH/PDL ──
        sig_exit_long = (pdh > 0) & (close < pdh)
        sig_exit_short = (pdl > 0) & (close > pdl)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            breakeven_pct=0.003,
            time_stop_bars=90,
        )
