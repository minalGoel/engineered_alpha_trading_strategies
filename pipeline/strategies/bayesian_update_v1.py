"""bayesian_update_v1 — Bayesian multi-evidence directional conviction on NIFTY.

Aggregates three independent evidence streams (1-min price return, volume ratio,
VWAP position) into a running Bayesian posterior. Fires when posterior exceeds
threshold with 3-bar monotonic accumulation confirming evidence is building.

Adapted from Strategy_299.json: replaced pre-calibrated Gaussian likelihoods
with session-adaptive z-score normalization; compressed prior lookback from
30 min to 5 min to match 90s hold horizon; removed sector strength (not in pipeline).
"""
from __future__ import annotations
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _sigmoid(x: np.ndarray, k: float) -> np.ndarray:
    """Convert evidence z-score to probability in [0, 1]."""
    return 1.0 / (1.0 + np.exp(-k * np.clip(x, -10.0, 10.0)))


class Strategy(BaseStrategy):
    name = "bayesian_update_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — need 30-min warmup for rolling stats
    session_end_minutes = 920     # 15:20 IST
    max_lookback = 360            # 30 min warmup (360 × 5s) for rolling std stabilisation
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("posterior_threshold", 0.68, 0.60, 0.78),
            TunableParam("decay_factor", 0.90, 0.80, 0.97),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # --- Raw arrays (forward-fill before numpy) ---
        close = spot_df["close"].fill_null(strategy="forward").fill_null(0.0).to_numpy()
        volume = spot_df["volume"].cast(pl.Float64).fill_null(0.0).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        threshold = float(params.get("posterior_threshold", 0.68))
        decay = float(params.get("decay_factor", 0.90))

        # ── Evidence 1: 1-minute (12-bar) price return ──────────────────────
        ret_12 = np.zeros(n)
        for i in range(12, n):
            denom = close[i - 12] if close[i - 12] > 0 else 1.0
            ret_12[i] = (close[i] - close[i - 12]) / denom

        # ── Evidence 2: Volume ratio (current / 2-min rolling avg) ──────────
        vol_ratio = np.ones(n)
        for i in range(24, n):
            vmean = np.mean(volume[i - 24:i])
            if vmean > 0:
                vol_ratio[i] = volume[i] / vmean

        # ── Evidence 3: Session VWAP distance ───────────────────────────────
        vwap = np.zeros(n)
        cum_pv = 0.0
        cum_v = 0.0
        cur_day = -1
        for i in range(n):
            if day_id[i] != cur_day:
                cur_day = day_id[i]
                cum_pv = 0.0
                cum_v = 0.0
            cum_pv += close[i] * volume[i]
            cum_v += volume[i]
            vwap[i] = cum_pv / cum_v if cum_v > 0 else close[i]

        vwap_dist = np.zeros(n)
        for i in range(n):
            ref = vwap[i] if vwap[i] > 0 else 1.0
            vwap_dist[i] = (close[i] - vwap[i]) / ref

        # ── 5-min momentum for Bayesian prior ──────────────────────────────
        mom_60 = np.zeros(n)
        for i in range(60, n):
            denom = close[i - 60] if close[i - 60] > 0 else 1.0
            mom_60[i] = (close[i] - close[i - 60]) / denom

        # ── Rolling std for adaptive normalization (60-bar = 5 min) ─────────
        ret_std = np.full(n, 1e-6)
        vd_std = np.full(n, 1e-6)
        mom_std = np.full(n, 1e-6)
        for i in range(60, n):
            s = np.std(ret_12[i - 60:i])
            ret_std[i] = s if s > 1e-6 else 1e-6
            s = np.std(vwap_dist[i - 60:i])
            vd_std[i] = s if s > 1e-6 else 1e-6
        for i in range(120, n):
            s = np.std(mom_60[i - 120:i])
            mom_std[i] = s if s > 1e-6 else 1e-6

        # ── Bayesian running posterior ───────────────────────────────────────
        # Each bar: convert 3 evidence z-scores to P(up|evidence) via sigmoid,
        # then compute naive-Bayes posterior; decay toward 0.5 each bar.
        posterior = np.full(n, 0.5)
        for i in range(360, n):
            ret_z = ret_12[i] / ret_std[i]
            vd_z = vwap_dist[i] / vd_std[i]
            mom_z = mom_60[i] / mom_std[i]

            # Per-evidence probabilities P(up | evidence)
            p_ret = _sigmoid(np.array([ret_z]), k=3.0)[0]
            p_vwap = _sigmoid(np.array([vd_z]), k=2.0)[0]
            p_mom = _sigmoid(np.array([mom_z]), k=2.5)[0]

            # Volume amplification (above-avg volume confirms direction, cap at 3×)
            vol_amp = 1.0 + 0.1 * min(max(vol_ratio[i] - 1.0, 0.0), 2.0)

            # Prior: blend decayed previous posterior + 5-min momentum prior
            prior_decay = 0.5 + decay * (posterior[i - 1] - 0.5)
            mom_prior = _sigmoid(np.array([mom_z]), k=1.0)[0]
            prior = 0.7 * prior_decay + 0.3 * mom_prior
            prior = np.clip(prior, 0.01, 0.99)

            # Naive Bayes likelihood update
            l_up = p_ret * p_vwap * p_mom * vol_amp
            l_down = (1.0 - p_ret) * (1.0 - p_vwap) * (1.0 - p_mom) / max(vol_amp, 1e-6)
            denom = prior * l_up + (1.0 - prior) * l_down
            if denom > 1e-10:
                posterior[i] = (prior * l_up) / denom
            else:
                posterior[i] = 0.5

            posterior[i] = np.clip(posterior[i], 0.0, 1.0)

        # ── 3-bar monotonic confirmation ────────────────────────────────────
        post_increasing = np.zeros(n, dtype=bool)
        post_decreasing = np.zeros(n, dtype=bool)
        for i in range(3, n):
            post_increasing[i] = (
                posterior[i] > posterior[i - 1]
                and posterior[i - 1] > posterior[i - 2]
                and posterior[i - 2] > posterior[i - 3]
            )
            post_decreasing[i] = (
                posterior[i] < posterior[i - 1]
                and posterior[i - 1] < posterior[i - 2]
                and posterior[i - 2] < posterior[i - 3]
            )

        # ── VIX filter ──────────────────────────────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Entry signals ────────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        vix_ok = vix_close < 25.0
        above_vwap = close > vwap
        below_vwap = close < vwap

        buy_ce = (
            in_session
            & vix_ok
            & (posterior >= threshold)
            & post_increasing
            & above_vwap
        )
        buy_pe = (
            in_session
            & vix_ok
            & (posterior <= (1.0 - threshold))
            & post_decreasing
            & below_vwap
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, 4.0),
            target_points=np.full(n, 7.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,
            max_trades_per_day=self.max_trades_per_day,
        )
