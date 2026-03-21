"""vwap_rsi_volume_confluence_v1 — Triple-confluence VWAP mean-reversion on NIFTY.

Signal: buy CE when NIFTY is >1.5 stddev below session VWAP AND 2-min RSI < 25
AND volume > 2x 20-min average (climax selloff exhaustion). Mirror for buy PE.

Key differentiator from vwap_rsi_confluence_v1: uses z-score normalization
(adaptive to session volatility) + volume surge as third factor, not % deviation
+ RSI slope + candle confirmation.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder smoothed RSI. Returns 50.0 before warmup."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n <= period:
        return rsi
    delta = np.diff(close, prepend=close[0])
    gain = np.maximum(delta, 0.0)
    loss = np.maximum(-delta, 0.0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[period] = np.mean(gain[1 : period + 1])
    avg_loss[period] = np.mean(loss[1 : period + 1])
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss[period:] > 0, avg_gain[period:] / avg_loss[period:], np.inf)
        rsi[period:] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def _rolling_vol_ratio(volume: np.ndarray, window: int) -> np.ndarray:
    """volume / rolling_mean(volume, window). Fully vectorized via cumsum."""
    vol_f = volume.astype(np.float64)
    n = len(vol_f)
    cs = np.zeros(n + 1)
    cs[1:] = np.cumsum(vol_f)
    start = np.maximum(0, np.arange(n) + 1 - window)
    window_sums = cs[1:] - cs[start]
    window_lens = np.minimum(np.arange(1, n + 1), window).astype(np.float64)
    vol_sma = np.maximum(window_sums / window_lens, 0.01)
    return vol_f / vol_sma


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling population std via variance decomposition. Vectorized."""
    n = len(arr)
    cs = np.zeros(n + 1)
    cs[1:] = np.cumsum(arr)
    cs2 = np.zeros(n + 1)
    cs2[1:] = np.cumsum(arr ** 2)
    start = np.maximum(0, np.arange(n) + 1 - window)
    window_sums = cs[1:] - cs[start]
    window_sq_sums = cs2[1:] - cs2[start]
    window_lens = np.minimum(np.arange(1, n + 1), window).astype(np.float64)
    window_means = window_sums / window_lens
    window_var = window_sq_sums / window_lens - window_means ** 2
    return np.sqrt(np.maximum(window_var, 1e-6))


class Strategy(BaseStrategy):
    name = "vwap_rsi_volume_confluence_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — skip opening noise; VWAP needs warmup
    session_end_minutes = 920     # 15:20 IST
    max_lookback = 240            # 20 min warmup (240 x 5s) for rolling std
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vwap_zscore_thresh", 1.5, 0.8, 2.5),
            TunableParam("rsi_oversold", 25.0, 15.0, 35.0),
            TunableParam("rsi_overbought", 75.0, 65.0, 85.0),
            TunableParam("vol_ratio_thresh", 2.0, 1.5, 4.0),
            TunableParam("vix_max", 22.0, 15.0, 30.0),
            TunableParam("stop_pts", 3.0, 2.0, 8.0),
            TunableParam("target_pts", 6.0, 4.0, 12.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Int64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # Tunable thresholds
        vwap_z_thresh = params.get("vwap_zscore_thresh", 1.5)
        rsi_oversold = params.get("rsi_oversold", 25.0)
        rsi_overbought = params.get("rsi_overbought", 75.0)
        vol_thresh = params.get("vol_ratio_thresh", 2.0)
        vix_max = params.get("vix_max", 22.0)
        stop_pts = params.get("stop_pts", 3.0)
        target_pts = params.get("target_pts", 6.0)

        # ── VIX aligned to spot bars ──
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Session VWAP (cumulative per day, resets at day open) ──
        df_vwap = spot_df.with_columns(
            (pl.col("close") * pl.col("volume").cast(pl.Float64)).alias("_pv"),
        ).with_columns(
            pl.col("_pv").cum_sum().over("day_id").alias("_cum_pv"),
            pl.col("volume").cast(pl.Float64).cum_sum().over("day_id").alias("_cum_v"),
        ).with_columns(
            (pl.col("_cum_pv") / pl.col("_cum_v").clip(lower_bound=1.0)).alias("_vwap"),
        )
        vwap = df_vwap["_vwap"].fill_null(strategy="forward").fill_null(close[0] if n > 0 else 0.0).to_numpy()

        # ── VWAP z-score: deviation normalized by 20-min rolling stddev ──
        dev = close - vwap
        dev_std = _rolling_std(dev, 240)
        vwap_zscore = dev / np.maximum(dev_std, 0.1)

        # ── RSI(24) — 2-minute RSI at 5-second bars ──
        rsi_arr = _rsi(close, 24)

        # ── Volume ratio: bar volume vs 20-min rolling average ──
        vol_ratio = _rolling_vol_ratio(volume, 240)

        # ── Session and regime filter ──
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        low_vix = vix_close < vix_max

        # ── Triple-confluence signals ──
        # BUY CE: price deeply below VWAP + RSI oversold + volume surge (climax selloff reversal)
        buy_ce = (
            in_session
            & low_vix
            & (vwap_zscore < -vwap_z_thresh)
            & (rsi_arr < rsi_oversold)
            & (vol_ratio > vol_thresh)
        )

        # BUY PE: price deeply above VWAP + RSI overbought + volume surge (climax buy reversal)
        buy_pe = (
            in_session
            & low_vix
            & (vwap_zscore > vwap_z_thresh)
            & (rsi_arr > rsi_overbought)
            & (vol_ratio > vol_thresh)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
