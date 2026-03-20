"""Connors RSI-2 Pullback — 5-second NIFTY/BANKNIFTY index options.

Mechanism: On NIFTY and BANKNIFTY, sharp 30-60 second micro-selloffs within the
prevailing intraday VWAP-trend attract delta-hedging buy flow from option market makers
and VWAP-benchmarked institutional algos. When 1-minute RSI drops below 20 while spot
trades above session VWAP, resting bid orders at VWAP-anchored levels absorb the
downtick and the index reverts within 30-90 seconds as TWAP algos resume accumulation.

Converted from Strategy_109.json (Connors RSI-2 pullback on nifty200 stocks, 15-min bars).
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


def _compute_vwap(close: np.ndarray, volume: np.ndarray, day_id: np.ndarray) -> np.ndarray:
    """Session VWAP reset per trading day."""
    n = len(close)
    vwap = np.empty(n)
    cum_vol = 0.0
    cum_pv = 0.0
    prev_day = -1
    for i in range(n):
        if day_id[i] != prev_day:
            cum_vol = 0.0
            cum_pv = 0.0
            prev_day = day_id[i]
        v = volume[i]
        cum_vol += v
        cum_pv += close[i] * v
        vwap[i] = cum_pv / cum_vol if cum_vol > 0 else close[i]
    return vwap


def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """RSI using Wilder's smoothing. Returns 50.0 for bars before warmup."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 2:
        return rsi

    delta = np.empty(n)
    delta[0] = 0.0
    delta[1:] = close[1:] - close[:-1]

    gain = np.where(delta > 0.0, delta, 0.0)
    loss = np.where(delta < 0.0, -delta, 0.0)

    # Seed with simple average of first `period` changes
    avg_gain = np.mean(gain[1: period + 1])
    avg_loss = np.mean(loss[1: period + 1])
    alpha = 1.0 / period

    for i in range(period, n):
        if i > period:
            avg_gain = avg_gain * (1.0 - alpha) + gain[i] * alpha
            avg_loss = avg_loss * (1.0 - alpha) + loss[i] * alpha
        if avg_loss == 0.0:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    return rsi


class Strategy(BaseStrategy):
    name = "rsi_connors_pullback_v50"
    underlying = "BOTH"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 8
    max_lookback = 36             # 3-minute warmup — enough for RSI(12) to stabilise

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rsi_oversold",  20.0, 10.0, 30.0),
            TunableParam("rsi_overbought", 80.0, 70.0, 90.0),
            TunableParam("stop_pts",       4.0,  2.0,  8.0),
            TunableParam("target_pts",     6.0,  3.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close   = spot_df["close"].fill_null(strategy="forward").to_numpy()
        volume  = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy() \
                  if "volume" in spot_df.columns else np.ones(n)
        day_id  = spot_df["day_id"].fill_null(0).to_numpy()
        time_min = spot_df["time_minutes"].fill_null(0).to_numpy()

        rsi_period = 12  # 60 seconds at 5s bars — 1-minute RSI
        rsi  = _compute_rsi(close, rsi_period)
        vwap = _compute_vwap(close, volume.astype(float), day_id)

        rsi_oversold  = params.get("rsi_oversold",  20.0)
        rsi_overbought = params.get("rsi_overbought", 80.0)
        stop_pts  = params.get("stop_pts",  4.0)
        target_pts = params.get("target_pts", 6.0)

        in_session = (time_min >= self.session_start_minutes) & \
                     (time_min < self.session_end_minutes)

        warmed_up = np.zeros(n, dtype=bool)
        if n > rsi_period + 2:
            warmed_up[rsi_period + 2:] = True

        # Connors RSI-2 logic: extreme short-term RSI exhaustion within VWAP trend
        # buy_ce: price above VWAP (intraday uptrend) AND RSI(12) extremely oversold
        buy_ce = in_session & warmed_up & (close > vwap) & (rsi < rsi_oversold)

        # buy_pe: price below VWAP (intraday downtrend) AND RSI(12) extremely overbought
        buy_pe = in_session & warmed_up & (close < vwap) & (rsi > rsi_overbought)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,             # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
