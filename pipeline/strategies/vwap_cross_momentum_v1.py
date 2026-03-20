"""VWAP Cross Momentum — 5-second NIFTY index options strategy.

After NIFTY trades below VWAP for 5+ minutes (60+ bars), VWAP-benchmarked
buy-side algorithms have accumulated benchmark underperformance. When price
crosses back above VWAP on elevated volume (>1.5x 10-min avg) and the bar
closes in the top 30% of its range, institutional algorithms reset from passive
to active buying. The resulting cohort drives a 15-25 spot pt continuation
over the next 60-90 seconds.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


def _compute_vwap(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                  volume: np.ndarray, day_ids: np.ndarray) -> np.ndarray:
    """Session-cumulative VWAP, reset each trading day."""
    n = len(close)
    vwap = np.empty(n)
    cum_tpv = 0.0
    cum_vol = 0.0
    prev_day = -999999
    for i in range(n):
        if day_ids[i] != prev_day:
            cum_tpv = 0.0
            cum_vol = 0.0
            prev_day = day_ids[i]
        tp = (high[i] + low[i] + close[i]) / 3.0
        cum_tpv += tp * volume[i]
        cum_vol += volume[i]
        vwap[i] = cum_tpv / cum_vol if cum_vol > 0 else close[i]
    return vwap


def _compute_dwell(close: np.ndarray, vwap: np.ndarray,
                   day_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Count consecutive bars where close is below/above VWAP, reset at day boundaries."""
    n = len(close)
    bars_below = np.zeros(n, dtype=np.int32)
    bars_above = np.zeros(n, dtype=np.int32)
    for i in range(1, n):
        if day_ids[i] != day_ids[i - 1]:
            bars_below[i] = 0
            bars_above[i] = 0
        else:
            bars_below[i] = (bars_below[i - 1] + 1) if close[i - 1] < vwap[i - 1] else 0
            bars_above[i] = (bars_above[i - 1] + 1) if close[i - 1] > vwap[i - 1] else 0
    return bars_below, bars_above


def _compute_ema(close: np.ndarray, day_ids: np.ndarray, period: int) -> np.ndarray:
    """EMA with reset at day boundaries."""
    n = len(close)
    ema = np.empty(n)
    alpha = 2.0 / (period + 1)
    ema[0] = close[0]
    for i in range(1, n):
        if day_ids[i] != day_ids[i - 1]:
            ema[i] = close[i]
        else:
            ema[i] = alpha * close[i] + (1.0 - alpha) * ema[i - 1]
    return ema


def _compute_vol_sma(volume: np.ndarray, period: int) -> np.ndarray:
    """Rolling SMA of volume over `period` bars (no day-boundary reset)."""
    n = len(volume)
    vol_sma = np.zeros(n)
    for i in range(period, n):
        vol_sma[i] = np.mean(volume[i - period:i])
    return vol_sma


class Strategy(BaseStrategy):
    name = "vwap_cross_momentum_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — allow 15 min for VWAP to stabilize
    session_end_minutes = 920     # 15:20 IST — avoid EOD dislocation
    max_trades_per_day = 8
    max_lookback = 240            # 20-min warmup: covers vol_sma_120 + VWAP stabilisation

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("dwell_min", 60, 30, 120),           # bars below/above VWAP before cross
            TunableParam("vol_surge_threshold", 1.5, 1.2, 2.5),  # volume_surge multiplier
            TunableParam("range_ratio_threshold", 0.70, 0.60, 0.85),  # close in top/bottom X%
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill before converting to numpy
        filled = spot_df.with_columns([
            pl.col("close").forward_fill(),
            pl.col("high").forward_fill(),
            pl.col("low").forward_fill(),
            pl.col("open").forward_fill(),
            pl.col("volume").fill_null(0),
        ])

        close = filled["close"].to_numpy()
        high = filled["high"].to_numpy()
        low = filled["low"].to_numpy()
        volume = filled["volume"].to_numpy().astype(float)
        day_ids = filled["day_id"].to_numpy()
        time_min = filled["time_minutes"].to_numpy()

        dwell_min = int(params.get("dwell_min", 60))
        vol_surge_thr = params.get("vol_surge_threshold", 1.5)
        range_ratio_thr = params.get("range_ratio_threshold", 0.70)

        # ── Core indicators ──────────────────────────────────────────────────
        vwap = _compute_vwap(close, high, low, volume, day_ids)
        bars_below, bars_above = _compute_dwell(close, vwap, day_ids)

        # Volume surge: bar volume vs 10-min (120 bars) rolling average
        vol_sma = _compute_vol_sma(volume, 120)
        volume_surge = np.where(vol_sma > 0, volume / vol_sma, 1.0)

        # 1-minute EMA (12 bars × 5s) for momentum direction
        ema12 = _compute_ema(close, day_ids, 12)

        # Bar range ratio: close position within bar's high-low range
        bar_range = high - low
        range_ratio = np.where(bar_range > 0.0, (close - low) / bar_range, 0.5)

        # ── Cross detection (vectorised) ─────────────────────────────────────
        same_day = np.empty(n, dtype=bool)
        same_day[0] = False
        same_day[1:] = day_ids[1:] == day_ids[:-1]

        cross_up = np.zeros(n, dtype=bool)
        cross_down = np.zeros(n, dtype=bool)
        if n > 1:
            cross_up[1:] = (same_day[1:]
                            & (close[1:] >= vwap[1:])
                            & (close[:-1] < vwap[:-1]))
            cross_down[1:] = (same_day[1:]
                              & (close[1:] <= vwap[1:])
                              & (close[:-1] > vwap[:-1]))

        # EMA direction
        ema12_rising = np.zeros(n, dtype=bool)
        ema12_falling = np.zeros(n, dtype=bool)
        if n > 1:
            ema12_rising[1:] = same_day[1:] & (ema12[1:] > ema12[:-1])
            ema12_falling[1:] = same_day[1:] & (ema12[1:] < ema12[:-1])

        # ── Session filter ────────────────────────────────────────────────────
        in_session = ((time_min >= self.session_start_minutes)
                      & (time_min < self.session_end_minutes))

        # ── Entry signals ─────────────────────────────────────────────────────
        buy_ce = (in_session
                  & cross_up
                  & (bars_below >= dwell_min)
                  & (volume_surge >= vol_surge_thr)
                  & (range_ratio >= range_ratio_thr)
                  & ema12_rising)

        buy_pe = (in_session
                  & cross_down
                  & (bars_above >= dwell_min)
                  & (volume_surge >= vol_surge_thr)
                  & (range_ratio <= (1.0 - range_ratio_thr))
                  & ema12_falling)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, 4.0),     # ~8 spot pts retrace threshold
            target_points=np.full(n, 7.0),   # ~14 spot pts continuation target
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,                # 120 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
