"""Elder Impulse System — NIFTY 5-second index options strategy.

Original: Elder's Impulse System on 1-min NIFTY50 stocks (EMA(9) + MACD(9,21,7)).
Converted: NIFTY index at 5s bars. EMA(24)=2min trend, MACD(12,36,9). Hold 30-90s.

Mechanism: When NIFTY's 2-minute EMA slope and MACD(12,36,9) histogram slope are
simultaneously rising (green impulse), institutional TWAP orders are absorbing offer-side
liquidity at multiple levels, creating 30-90s momentum continuation bursts. Conversely,
simultaneous EMA decline + MACD histogram decline (red impulse) signals sustained selling.
Elder's 'forbidden trade' rule — never long CE on red, never long PE on green — functions
as a regime filter against active institutional directional flow.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Compute EMA with alpha = 2 / (period + 1), initialised at arr[0]."""
    alpha = 2.0 / (period + 1)
    out = np.empty(len(arr), dtype=np.float64)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def _compute_vwap(hlc3: np.ndarray, vol: np.ndarray, day_id: np.ndarray) -> np.ndarray:
    """Session-cumulative VWAP, reset at each new day_id."""
    n = len(hlc3)
    vwap = np.empty(n, dtype=np.float64)
    cum_pv = 0.0
    cum_v = 0.0
    prev_day = -999999
    for i in range(n):
        if day_id[i] != prev_day:
            cum_pv = 0.0
            cum_v = 0.0
            prev_day = day_id[i]
        cum_pv += hlc3[i] * vol[i]
        cum_v += vol[i]
        vwap[i] = cum_pv / cum_v if cum_v > 0.0 else hlc3[i]
    return vwap


class Strategy(BaseStrategy):
    name = "elder_impulse_v1"
    underlying = "NIFTY"
    session_start_minutes = 562   # 09:22 IST — allow 7 min EMA warmup after open
    session_end_minutes = 920     # 15:20 IST — EOD flatten
    max_lookback = 120            # 10 min warmup (covers EMA36 + signal EMA9 + margin)
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            # Minimum slope magnitude to qualify as directional (in index points)
            TunableParam("ema_slope_min", 0.005, 0.001, 0.05),
            TunableParam("hist_slope_min", 0.005, 0.001, 0.05),
            TunableParam("stop_pts", 3.0, 2.0, 6.0),
            TunableParam("target_pts", 5.0, 3.0, 10.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Parameters ──────────────────────────────────────────────────────────
        ema_slope_min = params.get("ema_slope_min", 0.005)
        hist_slope_min = params.get("hist_slope_min", 0.005)
        stop_pts = params.get("stop_pts", 3.0)
        target_pts = params.get("target_pts", 5.0)

        # ── Raw arrays (forward-fill NaN before numpy) ───────────────────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        high = spot_df["high"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        low = spot_df["low"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        vol = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        day_id = spot_df["day_id"].to_numpy().astype(np.int32)
        time_min = spot_df["time_minutes"].to_numpy().astype(np.int32)

        # ── EMA(24) — 2-minute trend slope ───────────────────────────────────────
        # Not 12× original EMA(9); compressed to 2-min to match 30-90s hold time.
        ema24 = _ema(close, 24)

        # ── MACD(12, 36, 9) — 1min fast / 3min slow / 45s signal ─────────────────
        # Preserves original ~1:2.3 fast:slow ratio, compressed to match hold time.
        ema12 = _ema(close, 12)
        ema36 = _ema(close, 36)
        macd_line = ema12 - ema36
        macd_signal = _ema(macd_line, 9)
        macd_hist = macd_line - macd_signal

        # ── Session VWAP — cumulative from day open ───────────────────────────────
        hlc3 = (high + low + close) / 3.0
        vwap = _compute_vwap(hlc3, vol, day_id)

        # ── Impulse color (Elder's definition) ───────────────────────────────────
        # Green: EMA slope > +min AND histogram slope > +min
        # Red:   EMA slope < -min AND histogram slope < -min
        ema_slope = np.zeros(n, dtype=np.float64)
        hist_slope = np.zeros(n, dtype=np.float64)
        ema_slope[1:] = ema24[1:] - ema24[:-1]
        hist_slope[1:] = macd_hist[1:] - macd_hist[:-1]

        green = (ema_slope > ema_slope_min) & (hist_slope > hist_slope_min)
        red = (ema_slope < -ema_slope_min) & (hist_slope < -hist_slope_min)

        # ── 2-bar confirmation (filters 5-second noise spikes) ───────────────────
        green_conf = np.zeros(n, dtype=bool)
        red_conf = np.zeros(n, dtype=bool)
        green_conf[1:] = green[1:] & green[:-1]
        red_conf[1:] = red[1:] & red[:-1]

        # ── Session filter ────────────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # ── Entry signals ─────────────────────────────────────────────────────────
        # CE: confirmed green impulse + price above VWAP (institutional buy-side context)
        buy_ce = in_session & green_conf & (close > vwap)

        # PE: confirmed red impulse + price below VWAP (institutional sell-side context)
        buy_pe = in_session & red_conf & (close < vwap)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts, dtype=np.float64),
            target_points=np.full(n, target_pts, dtype=np.float64),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
