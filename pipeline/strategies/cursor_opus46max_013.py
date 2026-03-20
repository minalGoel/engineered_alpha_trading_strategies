"""Z-Score Reversion v1 — cursor_opus46max_013

Thesis: Pure statistical z-score with 60-bar lookback on close prices.
When z-score exceeds +/- 2.0 and starts reverting (crosses back through
1.8), enter for reversion to mean. Target is z-score crossing 0.
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


class Strategy(BaseStrategy):
    name = "cursor_opus46max_013"
    is_long_only = False
    session_start = 600   # 10:00
    session_end = 900     # 15:00
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_entry", default=2.0, low=1.5, high=3.0),
            TunableParam("zscore_confirm", default=1.8, low=1.2, high=2.5),
            TunableParam("zscore_stop", default=3.0, low=2.5, high=4.0),
            TunableParam("vix_max", default=25.0, low=16.0, high=30.0),
            TunableParam("stop_loss_pct", default=0.005, low=0.002, high=0.008),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        zs_entry = params.get("zscore_entry", 2.0)
        zs_confirm = params.get("zscore_confirm", 1.8)
        zs_stop = params.get("zscore_stop", 3.0)
        vix_max = params.get("vix_max", 25.0)
        stop_pct = params.get("stop_loss_pct", 0.005)

        n = len(df)

        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── Rolling mean and std (60 bars) ──
        rolling_mean = df["close"].rolling_mean(60).to_numpy().astype(np.float64)
        rolling_mean = np.nan_to_num(rolling_mean, nan=close[0] if n > 0 else 0.0)
        rolling_std = df["close"].rolling_std(60).to_numpy().astype(np.float64)
        rolling_std = np.nan_to_num(rolling_std, nan=1e10)
        rolling_std = np.clip(rolling_std, 1e-10, None)

        zscore = (close - rolling_mean) / rolling_std
        zscore = np.nan_to_num(zscore, nan=0.0)

        # ── Z-score velocity ──
        zs_vel = np.zeros(n, dtype=np.float64)
        for i in range(5, n):
            zs_vel[i] = zscore[i] - zscore[i - 5]

        # ── ATR ──
        atr = _compute_atr(high, low, close, 14)

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > 0.8 * avg_vol

        # ── Filters ──
        time_ok = (time_mins >= 600) & (time_mins <= 870)
        vix_ok = vix < vix_max

        # ── Entry: z-score was extreme, now crossing back through confirm level ──
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            # Long: z was below -entry, now crossing above -confirm
            if (zscore[i - 1] < -zs_confirm and zscore[i] >= -zs_confirm
                    and zscore[i] < 0 and zs_vel[i] > 0):
                long_entry[i] = True
            # Short: z was above +entry, now crossing below +confirm
            if (zscore[i - 1] > zs_confirm and zscore[i] <= zs_confirm
                    and zscore[i] > 0 and zs_vel[i] < 0):
                short_entry[i] = True

        long_entry = long_entry & vol_ok & vix_ok & time_ok
        short_entry = short_entry & vol_ok & vix_ok & time_ok

        # ── Signal exit: z-score crosses zero OR hits extreme stop ──
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if zscore[i] >= 0 and zscore[i - 1] < 0:
                sig_exit_long[i] = True
            if zscore[i] < -zs_stop:
                sig_exit_long[i] = True
            if zs_vel[i] < 0 and zscore[i] < -zs_confirm:
                sig_exit_long[i] = True
            if zscore[i] <= 0 and zscore[i - 1] > 0:
                sig_exit_short[i] = True
            if zscore[i] > zs_stop:
                sig_exit_short[i] = True
            if zs_vel[i] > 0 and zscore[i] > zs_confirm:
                sig_exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=rolling_mean.copy(),
            stop_loss_pct=stop_pct,
            use_target_indicator=True,
            time_stop_bars=40,
        )
