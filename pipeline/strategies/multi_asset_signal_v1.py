"""multi_asset_signal_v1 — NIFTY/BANKNIFTY VIX divergence mean-reversion.

When NIFTY/BANKNIFTY and India VIX move in the SAME direction over a 2-minute
window, it signals a divergence from the normal negative NIFTY-VIX correlation.
Such co-movement indicates institutional hedging (spot up + VIX up → buy PE) or
complacent selling (spot down + VIX down → buy CE). The divergence typically
resolves within 30-90 seconds as resting orders absorb the directional flow.

Original: cross-asset macro signal on NIFTY 50 / BANKNIFTY constituent stocks
(1-min bars, 15-45 min hold). Adapted to: NIFTY/BANKNIFTY index options, 5s bars,
30-90 second hold. Stock-selection layer removed; trading index options directly.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "multi_asset_signal_v1"
    underlying = "BOTH"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 6
    max_lookback = 180            # 15 min warmup (covers 24-bar ROC + persistence buffer)

    def tunable_params(self) -> list[TunableParam]:
        return [
            # Minimum 2-min spot ROC magnitude to qualify as a directional move
            TunableParam("roc_threshold", 0.0010, 0.0005, 0.0025),
            # Minimum 2-min VIX ROC magnitude to qualify as a directional VIX move
            TunableParam("vix_roc_threshold", 0.005, 0.002, 0.015),
            # Stop loss in option premium points
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            # Profit target in option premium points
            TunableParam("target_pts", 6.0, 4.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        roc_threshold = params.get("roc_threshold", 0.0010)
        vix_roc_threshold = params.get("vix_roc_threshold", 0.005)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 6.0)

        # ── VIX close aligned to spot bars ────────────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(
                    ["datetime", pl.col("close").alias("vix_close")]
                ).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── 2-minute (24-bar) ROC on spot ────────────────────────────────────
        # Compressed from original ROC_15 on 1-min bars (= 15-min ROC).
        # 24 bars at 5s = 2 min — matches the setup horizon for a 30-90s hold.
        spot_roc_24 = np.zeros(n)
        for i in range(24, n):
            denom = close[i - 24]
            if denom != 0.0:
                spot_roc_24[i] = (close[i] - denom) / denom

        # ── 2-minute VIX ROC ─────────────────────────────────────────────────
        vix_roc_24 = np.zeros(n)
        for i in range(24, n):
            denom = vix_close[i - 24]
            if denom != 0.0:
                vix_roc_24[i] = (vix_close[i] - denom) / denom

        # ── Raw divergence flags ──────────────────────────────────────────────
        # Bull divergence: spot falling AND VIX also falling (complacent dip)
        raw_bull = (spot_roc_24 < -roc_threshold) & (vix_roc_24 < -vix_roc_threshold)
        # Bear divergence: spot rising AND VIX also rising (institutional hedging)
        raw_bear = (spot_roc_24 > roc_threshold) & (vix_roc_24 > vix_roc_threshold)

        # ── Persistence filter: must hold for ≥3 consecutive bars (15s) ──────
        # Reduces false positives from 5-second tick noise.
        bull_persistent = np.zeros(n, dtype=bool)
        bear_persistent = np.zeros(n, dtype=bool)
        for i in range(2, n):
            if raw_bull[i] and raw_bull[i - 1] and raw_bull[i - 2]:
                bull_persistent[i] = True
            if raw_bear[i] and raw_bear[i - 1] and raw_bear[i - 2]:
                bear_persistent[i] = True

        # ── VIX level filter ─────────────────────────────────────────────────
        # Outside 12-28, NIFTY-VIX relationship is non-linear (VIX floored or
        # crisis regime). From original strategy's filters.
        vix_in_range = (vix_close >= 12.0) & (vix_close <= 28.0)

        # ── Session filter ────────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── Final signals ─────────────────────────────────────────────────────
        buy_ce = in_session & vix_in_range & bull_persistent
        buy_pe = in_session & vix_in_range & bear_persistent

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
