"""BB Squeeze Breakout — NIFTY 5-second index options strategy.

When NIFTY's 5-minute Bollinger Band width compresses below the 10th-15th percentile
of the prior 1-hour distribution, institutional algos stand aside. A close outside
the compressed band with above-average volume signals cascading FII-driven TWAP/VWAP
entries in the breakout direction — autocorrelated for 30-90 seconds.

Original: bb_squeeze_breakout_v1 (equity, 1-min bars, NIFTY200 stocks, 20-120 min hold)
Adapted:  5-second NIFTY index options, 30-90 second hold
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


def _sma(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        result[i] = np.mean(arr[i - window + 1 : i + 1])
    return result


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        result[i] = np.std(arr[i - window + 1 : i + 1], ddof=0)
    return result


def _rolling_pctile_rank(arr: np.ndarray, window: int) -> np.ndarray:
    """Percentile rank of arr[i] within arr[i-window:i] (prior window bars, not including i)."""
    n = len(arr)
    result = np.full(n, 50.0)
    for i in range(window, n):
        past = arr[i - window : i]
        result[i] = np.searchsorted(np.sort(past), arr[i], side="left") / window * 100.0
    return result


class Strategy(BaseStrategy):
    name = "bb_squeeze_breakout_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — skip first 15 min opening noise
    session_end_minutes = 920     # 15:20 IST
    max_lookback = 720            # 1-hour warmup for percentile rank
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("squeeze_pctile", 15.0, 5.0, 25.0),
            TunableParam("vol_mult", 1.5, 1.0, 2.5),
            TunableParam("vix_max", 24.0, 18.0, 30.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        squeeze_pctile = params.get("squeeze_pctile", 15.0)
        vol_mult = params.get("vol_mult", 1.5)
        vix_max = params.get("vix_max", 24.0)

        # ── BB(60) = 5-minute Bollinger Bands ──────────────────────────────────
        BB_PERIOD = 60
        bb_mid = _sma(close, BB_PERIOD)
        bb_std = _rolling_std(close, BB_PERIOD)

        bb_mid_safe = np.where(np.isnan(bb_mid) | (bb_mid == 0.0), 1.0, bb_mid)
        bb_std_safe = np.where(np.isnan(bb_std), 0.0, bb_std)

        bb_upper = bb_mid + 2.0 * bb_std_safe
        bb_lower = bb_mid - 2.0 * bb_std_safe
        bb_width = 4.0 * bb_std_safe / bb_mid_safe  # (upper - lower) / mid

        # ── Percentile rank of width vs prior 1-hour (720 bars) ───────────────
        PCTILE_WINDOW = 720
        bb_width_pctile = _rolling_pctile_rank(bb_width, PCTILE_WINDOW)

        # ── Volume SMA(60) for surge filter ───────────────────────────────────
        vol_sma = _sma(volume, BB_PERIOD)
        vol_sma_safe = np.where(np.isnan(vol_sma) | (vol_sma == 0.0), 1.0, vol_sma)

        # ── VIX filter ────────────────────────────────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Composite conditions ───────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        vix_ok = (vix_close >= 11.0) & (vix_close <= vix_max)
        bb_valid = ~np.isnan(bb_mid)

        squeeze = bb_width_pctile < squeeze_pctile
        vol_surge = volume > vol_sma_safe * vol_mult

        buy_ce = in_session & vix_ok & bb_valid & squeeze & vol_surge & (close > bb_upper)
        buy_pe = in_session & vix_ok & bb_valid & squeeze & vol_surge & (close < bb_lower)

        # ── OptionSignals ─────────────────────────────────────────────────────
        # Stop 4 pts: genuine squeeze breakout should not retrace 8 NIFTY spot pts
        #             within 30-90s; if it does, the breakout was absorbed immediately.
        # Target 7 pts: first FII TWAP wave on NIFTY moves 14-25 spot pts;
        #               7 option pts (14 spot at delta 0.5) captures the lower end (1:1.75 R:R).
        # Time stop 18 bars = 90 seconds: breakout autocorrelation window on NIFTY.
        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, 5),
            target_points=np.full(n, 8),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,
            max_trades_per_day=self.max_trades_per_day,
        )
