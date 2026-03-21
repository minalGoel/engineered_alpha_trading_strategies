"""implied_vs_realized_vol_v1 — Variance Risk Premium Compression

On NIFTY, when VIX has spiked intraday but 20-minute realized volatility has not kept
pace, market makers who sold calls into the fear spike are over-hedged short-delta and
will buy the index to unwind as VIX begins to revert. We z-score the intraday VRP
(VIX/100 - RV_240) over a 60-minute rolling window; when VRP z-score > 1.5 and VIX's
5-minute rate-of-change turns negative, we buy CE (bullish). When RV >> IV with VIX
rising, we buy PE (bearish).

NIFTY only — India VIX is based on NIFTY options.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder's RSI on a 1-D close array. Returns array of same length."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 2:
        return rsi

    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Seed with simple average of first `period` moves
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, n - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss < 1e-12:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    # Backfill initial period with neutral 50
    rsi[: period + 1] = 50.0
    return rsi


def _rolling_rv(log_ret: np.ndarray, window: int, ann_factor: float) -> np.ndarray:
    """Annualized realized vol over a rolling `window` of log returns."""
    n = len(log_ret)
    rv = np.zeros(n)
    for i in range(window, n):
        rv[i] = np.std(log_ret[i - window : i]) * ann_factor
    return rv


def _rolling_mean_std(x: np.ndarray, window: int):
    """Returns (mean_arr, std_arr) for rolling window. Outputs 0-mean, 1-std before warmup."""
    n = len(x)
    mean = np.zeros(n)
    std = np.ones(n)
    for i in range(window, n):
        w = x[i - window : i]
        mean[i] = np.mean(w)
        s = np.std(w)
        std[i] = s if s > 1e-9 else 1e-9
    return mean, std


class Strategy(BaseStrategy):
    name = "implied_vs_realized_vol_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — need 720 bars (60 min) of VRP warmup
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 4
    max_lookback = 720            # 720 × 5s = 60 min warmup for VRP z-score baseline

    # Annualisation factor for 5-second log returns:
    # sqrt(252 trading days × 6.25 trading hours × 3600 s/hr / 5 s per bar)
    _ANN_FACTOR = float(np.sqrt(252 * 6.25 * 3600 / 5))

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vrp_z_threshold", 1.5, 1.0, 2.5),
            TunableParam("vix_decline_threshold", 0.02, 0.005, 0.05),
            TunableParam("rsi_oversold", 45.0, 35.0, 52.0),
            TunableParam("rsi_overbought", 55.0, 48.0, 65.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict,
    ) -> OptionSignals:
        n = len(spot_df)

        vrp_z_thresh = float(params.get("vrp_z_threshold", 1.5))
        vix_decline = float(params.get("vix_decline_threshold", 0.02))
        rsi_low = float(params.get("rsi_oversold", 45.0))
        rsi_high = float(params.get("rsi_overbought", 55.0))

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        # ── 1. Align VIX to spot bars ────────────────────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")])
                .sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── 2. 20-min realized vol (240-bar rolling, log returns) ─────────────────
        log_ret = np.zeros(n)
        log_ret[1:] = np.log(close[1:] / np.maximum(close[:-1], 1e-6))

        rv_240 = _rolling_rv(log_ret, window=240, ann_factor=self._ANN_FACTOR)

        # ── 3. Variance risk premium (VIX decimal - RV) ───────────────────────────
        iv = vix_close / 100.0          # VIX is in percentage points; convert to decimal
        vrp = iv - rv_240               # positive when IV > RV (fear premium elevated)

        # ── 4. VRP z-score over 60-min (720-bar) rolling window ───────────────────
        vrp_mean, vrp_std = _rolling_mean_std(vrp, window=720)
        vrp_z = (vrp - vrp_mean) / vrp_std

        # ── 5. VIX 5-min (60-bar) rate of change ──────────────────────────────────
        vix_roc_60 = np.zeros(n)
        for i in range(60, n):
            base = vix_close[i - 60]
            if base > 1e-6:
                vix_roc_60[i] = (vix_close[i] - base) / base
            # else stays 0 (neutral)

        # ── 6. 3-min RSI (36 bars) on NIFTY close ────────────────────────────────
        rsi_36 = _compute_rsi(close, period=36)

        # ── 7. Session filter ────────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (
            time_min < self.session_end_minutes
        )

        # ── 8. Entry signals ─────────────────────────────────────────────────────
        # buy_ce: VRP elevated + VIX declining (fear compressing) + spot oversold
        buy_ce = (
            in_session
            & (vrp_z > vrp_z_thresh)
            & (vix_roc_60 < -vix_decline)
            & (rsi_36 < rsi_low)
        )

        # buy_pe: RV > IV (complacency) + VIX rising + spot extended
        buy_pe = (
            in_session
            & (vrp_z < -vrp_z_thresh)
            & (vix_roc_60 > vix_decline)
            & (rsi_36 > rsi_high)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            # stop=5 pts: 10 NIFTY spot pts adverse — VRP compression thesis failed
            # target=8 pts: ~16 NIFTY spot pts, ~60% of 15-25 pt dealer hedge unwind
            stop_points=np.full(n, 5.0),
            target_points=np.full(n, 8.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,         # 2-minute time stop
            max_trades_per_day=self.max_trades_per_day,
        )
