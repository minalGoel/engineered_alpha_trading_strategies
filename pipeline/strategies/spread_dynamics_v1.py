"""spread_dynamics_v1 — NIFTY Range Expansion Fade Strategy

Thesis: When NIFTY's 5-second bar range expands to 2x the 5-minute rolling average
(microstructure stress event), a directional overshoot occurs. When the range contracts
back to normal (stress resolved), market-maker inventory rebalancing and VWAP-benchmark
algo flows drive a 40-60% reversion of the spike within the next 60 seconds.

Adapted from spread_dynamics_v1 equity strategy (Strategy_248.json):
  - Replaces per-stock bid-ask spread with NIFTY bar range (valid OHLCV proxy)
  - Replaces 30-min spread SMA with 5-min rolling range avg (regime-appropriate)
  - Replaces 1-5 min hold with 60s time stop (5s bar resolution)
"""
from __future__ import annotations
import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "spread_dynamics_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — skip first 10 min (natural spread widening at open)
    session_end_minutes = 920     # 15:20 IST
    max_lookback = 60             # 60-bar (5-min) warmup for avg_range baseline
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("expansion_threshold", 2.0, 1.5, 3.0),
            TunableParam("contraction_threshold", 1.5, 1.0, 2.0),
            TunableParam("momentum_threshold", 0.0005, 0.0002, 0.0012),
            TunableParam("stop_pts", 5.0, 3.0, 8.0),
            TunableParam("target_pts", 7.0, 4.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── Extract spot arrays (forward-fill before numpy) ──
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        high = spot_df["high"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        low = spot_df["low"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()

        # ── Parameters ──
        exp_thr = params.get("expansion_threshold", 2.0)
        con_thr = params.get("contraction_threshold", 1.5)
        mom_thr = params.get("momentum_threshold", 0.0005)
        stop_pts = params.get("stop_pts", 5.0)
        target_pts = params.get("target_pts", 7.0)

        # ── Indicator 1: bar range (proxy for bid-ask spread width) ──
        bar_range = high - low  # shape (n,)

        # ── Indicator 2: avg_range_60 — 5-min rolling average range ──
        # Proxy for "normal" spread level. Use cumulative mean for early bars
        # to avoid division by zero; full 60-bar window from bar 1 onward.
        avg_range = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            start = max(0, i - 60)
            avg_range[i] = np.mean(bar_range[start:i])
        avg_range[0] = bar_range[0] if bar_range[0] > 0 else 1.0

        # ── Indicator 3: range_ratio — analog of spread_ratio ──
        safe_avg = np.where(avg_range > 0, avg_range, 1.0)
        range_ratio = bar_range / safe_avg

        # ── Indicator 4: peak_ratio_12 — rolling max over 60s (12 bars) ──
        # Detects if a stress event (expansion) occurred recently
        peak_ratio_12 = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            start = max(0, i - 12)
            peak_ratio_12[i] = np.max(range_ratio[start:i])

        # ── Indicator 5: momentum_6 — 30-second (6-bar) price return ──
        # Directional displacement during / immediately after the stress event
        momentum_6 = np.zeros(n, dtype=np.float64)
        if n > 6:
            momentum_6[6:] = (close[6:] - close[:-6]) / np.where(close[:-6] > 0, close[:-6], 1.0)

        # ── VIX filter ──
        # High VIX → range expansions may be regime-level, not transient microstructure
        vix_ok = np.ones(n, dtype=bool)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()
            vix_ok = vix_close < 22.0

        # ── Session filter ──
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # ── Entry conditions ──
        # Stress event recently: peak range_ratio in last 60s exceeded expansion_threshold
        had_expansion = peak_ratio_12 > exp_thr

        # Range contracting now: current bar range ratio back below contraction_threshold
        contracting_now = range_ratio < con_thr

        # Base gate: stress happened AND is now resolving
        event_resolving = in_session & had_expansion & contracting_now & vix_ok

        # Fade down move — price dropped during stress event → expect bounce → buy CE
        buy_ce = event_resolving & (momentum_6 < -mom_thr)

        # Fade up move — price rose during stress event → expect fade → buy PE
        buy_pe = event_resolving & (momentum_6 > mom_thr)

        # Mutual exclusion guard (shouldn't trigger but prevents logic errors)
        conflict = buy_ce & buy_pe
        buy_ce = buy_ce & ~conflict
        buy_pe = buy_pe & ~conflict

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts, dtype=np.float64),
            target_points=np.full(n, target_pts, dtype=np.float64),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=12,       # 60 seconds — reversion should happen fast or not at all
            max_trades_per_day=self.max_trades_per_day,
        )
