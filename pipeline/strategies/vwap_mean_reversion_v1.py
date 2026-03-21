"""VWAP Mean Reversion v1 — RSI + VWAP Dual Confirmation

Converts: trading_strategies/unique_strategies_all/Strategy_68.json
Original: VWAP z-score reversion on Nifty50 stocks, 1-min bars, RSI + volume confirmation.

Mechanism: On NIFTY, when a 10-min rolling z-score deviation from session VWAP drops below -1.5
AND RSI(36) simultaneously falls below 38 (3-min oversold), two independent exhaustion signals
align. The z-score catches statistical displacement from the institutional benchmark price;
RSI(36) confirms the sell-flow velocity has abated. VWAP-benchmarked algorithms now have
negative tracking error and begin accumulating. VIX < 20 guards against trending sessions
where VWAP gravitational pull fails. Differentiated from v18 (extreme deviation + deepening
momentum) and v19 (microstructure range contraction) by the RSI dual-confirmation gate.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Simple RSI using SMA of gains/losses (Wilder-style approximation via SMA)."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi
    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    for i in range(period, n):
        avg_gain = float(np.mean(gains[i - period:i]))
        avg_loss = float(np.mean(losses[i - period:i]))
        if avg_loss < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsi


class Strategy(BaseStrategy):
    name = "vwap_mean_reversion_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST — skip pre-open noise
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 6
    max_lookback = 120            # 10 min warmup for rolling std + RSI

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_threshold", 1.5, 1.0, 2.5),
            TunableParam("rsi_low", 38.0, 25.0, 45.0),
            TunableParam("rsi_high", 62.0, 55.0, 75.0),
            TunableParam("vix_max", 20.0, 15.0, 28.0),
            TunableParam("stop_pts", 3.0, 2.0, 7.0),
            TunableParam("target_pts", 6.0, 4.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill NaN in Polars before converting to numpy
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        time_min = spot_df["time_minutes"].fill_null(0).to_numpy()
        day_id = spot_df["day_id"].fill_null(0).to_numpy()

        zscore_threshold = params.get("zscore_threshold", 1.5)
        rsi_low = params.get("rsi_low", 38.0)
        rsi_high = params.get("rsi_high", 62.0)
        vix_max = params.get("vix_max", 20.0)
        stop_pts = params.get("stop_pts", 3.0)
        target_pts = params.get("target_pts", 6.0)

        # --- VWAP: cumulative from session open, reset each trading day ---
        vwap = np.empty(n)
        cum_tp_vol = 0.0
        cum_vol = 0.0
        prev_day = day_id[0] if n > 0 else -1

        for i in range(n):
            if day_id[i] != prev_day:
                cum_tp_vol = 0.0
                cum_vol = 0.0
                prev_day = day_id[i]
            v = volume[i]
            cum_tp_vol += close[i] * v
            cum_vol += v
            vwap[i] = cum_tp_vol / cum_vol if cum_vol > 0.0 else close[i]

        # --- Rolling std(close, 120) — 10-min window for z-score denominator ---
        std_window = 120
        close_std = np.empty(n)
        for i in range(n):
            start = max(0, i - std_window)
            s = float(np.std(close[start:i + 1]))
            close_std[i] = s if s > 0.5 else 0.5   # floor at 0.5 NIFTY pt to avoid division issues

        vwap_zscore = (close - vwap) / close_std

        # --- RSI(36): 3-minute RSI to detect momentum exhaustion ---
        rsi = _rsi(close, period=36)

        # --- VIX filter: align VIX close to spot bars ---
        vix_close = np.full(n, 15.0)   # neutral default
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime", strategy="backward"
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # --- Session filter ---
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # --- VIX regime filter: no trades in high-volatility trending sessions ---
        low_vix = vix_close < vix_max

        # --- Entry signals ---
        # buy_ce: VWAP stretched below + RSI oversold → sell-flow exhausted, reversion imminent
        buy_ce = (
            in_session
            & low_vix
            & (vwap_zscore < -zscore_threshold)
            & (rsi < rsi_low)
        )

        # buy_pe: VWAP stretched above + RSI overbought → buy-flow exhausted, reversion imminent
        buy_pe = (
            in_session
            & low_vix
            & (vwap_zscore > zscore_threshold)
            & (rsi > rsi_high)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,           # 90 seconds — dual confirmation gives more time to revert
            max_trades_per_day=self.max_trades_per_day,
        )
