"""Corporate Action Arbitrage — cursor_opus46max_126

Thesis: On ex-dividend/ex-split dates, the market's opening price overshoots or
undershoots the theoretical adjusted price. Fade the mispricing as it converges.
Since we lack corporate-action calendar data, we proxy with: large overnight gap
+ abnormally high opening volume + price converging toward VWAP (which approximates
the fair value on ex-dates).
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
    name = "cursor_opus46max_126"
    is_long_only = False
    session_start = 555   # 09:15
    session_end = 660     # 11:00
    max_trades_per_day = 3
    assumptions = [
        "Corporate-action calendar data not available; proxied by gap + volume surge",
        "Theoretical adjusted price proxied by VWAP",
    ]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_bps_thresh", default=20.0, low=10.0, high=50.0),
            TunableParam("vol_surge_thresh", default=1.5, low=1.2, high=2.5),
            TunableParam("convergence_bps", default=5.0, low=2.0, high=15.0),
            TunableParam("stop_bps", default=50.0, low=30.0, high=80.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        gap_thresh = params.get("gap_bps_thresh", 20.0)
        vol_surge = params.get("vol_surge_thresh", 1.5)
        convergence = params.get("convergence_bps", 5.0)
        stop_bps = params.get("stop_bps", 50.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        day_id = df["day_id"].to_numpy()

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ATR
        atr = _compute_atr(high, low, close, 14)

        # Rolling average volume
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ratio = volume / avg_vol

        # Daily open (first bar's open per day)
        day_open = np.zeros(n, dtype=np.float64)
        prev_close_arr = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if i == 0 or day_id[i] != day_id[i-1]:
                day_open[i] = open_[i]
                prev_close_arr[i] = close[i-1] if i > 0 else open_[i]
            else:
                day_open[i] = day_open[i-1]
                prev_close_arr[i] = prev_close_arr[i-1]

        # Gap from previous close (bps)
        safe_prev = np.clip(prev_close_arr, 1e-10, None)
        gap_bps = (day_open - prev_close_arr) / safe_prev * 10000.0

        # Adjustment error: (close - vwap) / vwap * 10000
        safe_vwap = np.clip(np.abs(vwap), 1e-10, None)
        adj_error = (close - vwap) / safe_vwap * 10000.0

        # Filters
        time_ok = (time_mins >= 555) & (time_mins <= 585)  # entry 09:15-09:45
        vol_ok = vol_ratio > vol_surge

        # Long: stock opened too low (gap_bps < -gap_thresh), converging up
        long_entry = (
            (gap_bps < -gap_thresh) &
            vol_ok &
            (close > day_open) &
            (adj_error < -convergence) &
            time_ok
        )

        # Short: stock opened too high (gap_bps > gap_thresh), converging down
        short_entry = (
            (gap_bps > gap_thresh) &
            vol_ok &
            (close < day_open) &
            (adj_error > convergence) &
            time_ok
        )

        # Signal exit: convergence to near VWAP
        sig_exit_long = adj_error > -convergence
        sig_exit_short = adj_error < convergence

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=vwap.copy(),
            stop_loss_pct=stop_bps / 10000.0,
            use_target_indicator=True,
            trailing_stop_pct=0.001,
            trailing_activate_pct=0.0015,
            time_stop_bars=60,
        )
