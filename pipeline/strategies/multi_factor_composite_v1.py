"""multi_factor_composite_v1 — Multi-signal consensus mean-reversion on NIFTY.

Combines VWAP z-score, 2-min RSI, 3-min momentum, and relative volume into a
weighted composite score. Fires when 3+ orthogonal signals simultaneously show
extreme oversold/overbought readings, indicating a short-term reversion setup.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


def _compute_vwap(close: np.ndarray, volume: np.ndarray, day_id: np.ndarray) -> np.ndarray:
    """Cumulative VWAP, reset each trading day."""
    n = len(close)
    vwap = np.zeros(n)
    cum_pv = 0.0
    cum_v = 0.0
    for i in range(n):
        if i == 0 or day_id[i] != day_id[i - 1]:
            cum_pv = close[i] * volume[i]
            cum_v = volume[i]
        else:
            cum_pv += close[i] * volume[i]
            cum_v += volume[i]
        vwap[i] = cum_pv / cum_v if cum_v > 0 else close[i]
    return vwap


def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder-smoothed RSI."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n <= period:
        return rsi
    gains = np.zeros(n)
    losses = np.zeros(n)
    for i in range(1, n):
        d = close[i] - close[i - 1]
        if d > 0:
            gains[i] = d
        else:
            losses[i] = -d
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[period] = np.mean(gains[1 : period + 1])
    avg_loss[period] = np.mean(losses[1 : period + 1])
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period
    for i in range(period, n):
        rs = avg_gain[i] / avg_loss[i] if avg_loss[i] > 0 else 100.0
        rsi[i] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


class Strategy(BaseStrategy):
    """Multi-factor composite mean-reversion on NIFTY 5s bars.

    Fires when VWAP z-score, 2-min RSI, and 3-min momentum simultaneously
    reach extreme oversold or overbought readings, filtered by relative volume.
    Requires consensus of at least 2 out of 3 directional sub-signals.
    """

    name = "multi_factor_composite_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — skip pre-open noise
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 8
    max_lookback = 240            # 20 min warmup for VWAP z-score and RSI

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("composite_threshold", 1.5, 0.8, 2.5),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 7.0, 4.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── Raw arrays ──
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        threshold = params.get("composite_threshold", 1.5)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 7.0)

        # ── Signal 1: VWAP z-score ──
        # Cumulative daily VWAP; z-score over 120-bar (10-min) rolling window.
        # 10-min normalisation window matches current session volatility regime.
        vwap = _compute_vwap(close, volume, day_id)
        vwap_dev = close - vwap
        zscore_window = 120
        vwap_zscore = np.zeros(n)
        for i in range(zscore_window, n):
            w = vwap_dev[i - zscore_window : i]
            s = np.std(w)
            vwap_zscore[i] = vwap_dev[i] / s if s > 0.01 else 0.0

        # ── Signal 2: 2-min RSI (24 bars) ──
        # Detects fast oversold/overbought within the 15-120s hold window.
        rsi = _compute_rsi(close, 24)
        rsi_scaled = (rsi - 50.0) / 50.0  # [-1, 1]: negative=oversold, positive=overbought

        # ── Signal 3: 3-min momentum (36 bars) ──
        # Directional impulse we are fading; 0.3% move normalised to ±1.
        mom_period = 36
        momentum_norm = np.zeros(n)
        for i in range(mom_period, n):
            if close[i - mom_period] > 0:
                raw = (close[i] - close[i - mom_period]) / close[i - mom_period]
                momentum_norm[i] = np.clip(raw / 0.003, -1.0, 1.0)

        # ── Signal 4: Relative volume (10-min baseline) ──
        # Amplifies composite when confirmed by elevated volume.
        vol_window = 120
        rel_vol = np.zeros(n)
        for i in range(vol_window, n):
            avg_vol = np.mean(volume[i - vol_window : i])
            if avg_vol > 0:
                rel_vol[i] = np.clip(volume[i] / avg_vol - 1.0, -1.0, 1.0)

        # ── VIX regime dampener ──
        vix_score = np.zeros(n)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_vals = vix_joined["vix_close"].fill_null(15.0).to_numpy()
            vix_score = np.clip((20.0 - vix_vals) / 10.0, -1.0, 1.0)

        # VIX > 25 (vix_score < -0.5): halve signal weight — panic sell-offs
        # can gap through any mean-reversion level
        vix_damp = np.where(vix_score < -0.5, 0.5, 1.0)

        # ── Composite score (mean-reversion framing) ──
        # Positive composite = oversold = buy CE (expect bounce)
        # Negative composite = overbought = buy PE (expect pullback)
        # All three signals inverted: price below VWAP / RSI oversold / falling = positive
        composite = (
            -0.35 * vwap_zscore     # positive when below VWAP
            - 0.30 * rsi_scaled     # positive when RSI < 50 (oversold)
            - 0.35 * momentum_norm  # positive when 3-min trend is down (selling climax)
        )

        # Volume amplifier: scale up signal during high-volume bars (1.0 – 1.3x)
        vol_amp = 1.0 + 0.3 * np.clip(rel_vol, 0.0, 1.0)
        composite = composite * vol_amp * vix_damp

        # ── Consensus check: require 2 of 3 directional sub-signals to agree ──
        # Bullish (oversold) sub-signals
        sig_vwap_bull = (vwap_zscore < -0.10).astype(np.int32)   # below VWAP
        sig_rsi_bull = (rsi_scaled < 0.0).astype(np.int32)        # RSI < 50
        sig_mom_bull = (momentum_norm < -0.05).astype(np.int32)   # recent down move
        bull_consensus = sig_vwap_bull + sig_rsi_bull + sig_mom_bull

        # Bearish (overbought) sub-signals
        sig_vwap_bear = (vwap_zscore > 0.10).astype(np.int32)    # above VWAP
        sig_rsi_bear = (rsi_scaled > 0.0).astype(np.int32)        # RSI > 50
        sig_mom_bear = (momentum_norm > 0.05).astype(np.int32)    # recent up move
        bear_consensus = sig_vwap_bear + sig_rsi_bear + sig_mom_bear

        # ── Session filter ──
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # ── Entry signals ──
        buy_ce = in_session & (composite > threshold) & (bull_consensus >= 2)
        buy_pe = in_session & (composite < -threshold) & (bear_consensus >= 2)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,
            max_trades_per_day=self.max_trades_per_day,
        )
