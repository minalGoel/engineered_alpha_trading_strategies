"""vix_mean_reversion_v1 — NIFTY 5-second options strategy.

Thesis: When India VIX surges >4% in 10 minutes (panic put buying), the precise
moment it decelerates (3 consecutive declining 5s bars) signals put-buying saturation.
Option market makers who sold those panic puts immediately delta-hedge by buying NIFTY
futures, generating a 10-25 spot-point mean-reversion bounce within 30-90 seconds.
Entry: NIFTY CE on VIX deceleration after spike. NIFTY PE on VIX bottom after crush.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "vix_mean_reversion_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 840     # 14:00 IST — post-14:00 VIX spikes unlikely to revert intraday
    max_trades_per_day = 3
    max_lookback = 360            # 30 min warmup (360 × 5s) for vix_zscore_360

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vix_spike_threshold", 4.0, 2.0, 8.0),   # % VIX rise in 10 min = panic
            TunableParam("vix_crush_threshold", 3.0, 1.5, 6.0),   # % VIX drop in 10 min = complacency
            TunableParam("rsi_oversold", 35.0, 25.0, 45.0),       # NIFTY RSI below = oversold
            TunableParam("rsi_overbought", 65.0, 55.0, 75.0),     # NIFTY RSI above = overbought
            TunableParam("stop_pts", 3.0, 2.0, 8.0),
            TunableParam("target_pts", 6.0, 4.0, 14.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)
        close = spot_df["close"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        vix_spike_threshold = params.get("vix_spike_threshold", 4.0)
        vix_crush_threshold = params.get("vix_crush_threshold", 3.0)
        rsi_oversold = params.get("rsi_oversold", 35.0)
        rsi_overbought = params.get("rsi_overbought", 65.0)
        stop_pts = params.get("stop_pts", 3.0)
        target_pts = params.get("target_pts", 6.0)

        # ── VIX aligned to spot bars via asof join ──
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── VIX 10-minute change % (120 bars × 5s = 10 min) ──
        # Compressed from original's 30-min window to match 30-120s hold time
        vix_change_120 = np.zeros(n)
        for i in range(120, n):
            prev = vix_close[i - 120]
            if prev > 0.01:
                vix_change_120[i] = (vix_close[i] - prev) / prev * 100.0

        # ── VIX z-score over 30-min rolling window (360 bars) ──
        # Replaces original's 20-day z-score with intraday session baseline
        vix_zscore = np.zeros(n)
        for i in range(360, n):
            window = vix_close[i - 360:i]
            mu = np.mean(window)
            sigma = np.std(window)
            if sigma > 0.01:
                vix_zscore[i] = (vix_close[i] - mu) / sigma

        # ── VIX directional streaks: 3 consecutive declining / rising bars ──
        # Deceleration after panic = MM delta-hedge buying starts (entry timing)
        vix_declining = np.zeros(n, dtype=bool)
        vix_rising = np.zeros(n, dtype=bool)
        for i in range(3, n):
            vix_declining[i] = (
                vix_close[i] < vix_close[i - 1]
                and vix_close[i - 1] < vix_close[i - 2]
                and vix_close[i - 2] < vix_close[i - 3]
            )
            vix_rising[i] = (
                vix_close[i] > vix_close[i - 1]
                and vix_close[i - 1] > vix_close[i - 2]
                and vix_close[i - 2] > vix_close[i - 3]
            )

        # ── RSI(36) on NIFTY close — 3-minute RSI for oversold/overbought ──
        # Compressed from original's RSI(14) on 1-min = 14-min RSI
        # 3-min RSI aligns with our 10-min VIX spike lookback
        rsi_36 = _compute_rsi(close, 36)

        # ── Session filter ──
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── BUY CE: VIX spike topped out + NIFTY oversold → MM futures buying begins ──
        buy_ce = (
            in_session
            & (vix_change_120 > vix_spike_threshold)
            & (vix_zscore > 1.0)
            & vix_declining
            & (rsi_36 < rsi_oversold)
        )

        # ── BUY PE: VIX crushed (complacency) now reversing + NIFTY overbought ──
        buy_pe = (
            in_session
            & (vix_change_120 < -vix_crush_threshold)
            & (vix_zscore < -0.5)
            & vix_rising
            & (rsi_36 > rsi_overbought)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,         # 120 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )


def _compute_rsi(prices: np.ndarray, period: int) -> np.ndarray:
    """Wilder-smoothed RSI. Returns 50.0 (neutral) for warmup bars."""
    n = len(prices)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, n - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss < 1e-10:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    return rsi
