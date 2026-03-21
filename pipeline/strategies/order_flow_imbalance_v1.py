"""Order Flow Imbalance strategy for NIFTY 5-second index options.

Thesis: When NIFTY's 5-second bars show sustained positive signed volume
(close > open) against flat/negative price change, it signals FII/DII
algorithmic TWAP accumulation. As sell inventory depletes, NIFTY breaks
out directionally in 15-120 seconds. The OFI/price divergence is the key
alpha signal.

Converted from: trading_strategies/unique_strategies_all/Strategy_244.json
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling sum over last `window` bars. Partial sums for early bars."""
    n = len(arr)
    padded = np.zeros(n + 1)
    padded[1:] = np.cumsum(arr)
    result = np.zeros(n)
    # Early bars: sum from 0 to i (partial)
    result[:window - 1] = padded[1:window]
    # Full window bars
    result[window - 1:] = padded[window:] - padded[:n - window + 1]
    return result


class Strategy(BaseStrategy):
    name = "order_flow_imbalance_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 920     # 15:20 IST
    max_lookback = 120            # 10 min warmup (covers ofi_fast=12 + ofi_slow=60)
    max_trades_per_day = 10

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("ofi_threshold", 0.30, 0.10, 0.70),
            TunableParam("stop_pts",      5.0,  3.0,  9.0),
            TunableParam("target_pts",    8.0,  5.0,  15.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── Extract OHLCV arrays ───────────────────────────────────────────
        close = (
            spot_df["close"]
            .fill_nan(None)
            .fill_null(strategy="forward")
            .to_numpy()
        )
        open_ = (
            spot_df["open"]
            .fill_nan(None)
            .fill_null(strategy="forward")
            .to_numpy()
        )
        time_min = spot_df["time_minutes"].to_numpy()

        # ── Parameters ────────────────────────────────────────────────────
        ofi_threshold = params.get("ofi_threshold", 0.30)
        stop_pts      = params.get("stop_pts",      5.0)
        target_pts    = params.get("target_pts",    8.0)

        # ── Signed volume direction proxy ─────────────────────────────────
        # +1 = bullish bar, -1 = bearish bar, 0 = doji
        # Proxy for Lee-Ready per-bar order flow direction.
        signed_dir = np.where(close > open_, 1.0,
                     np.where(close < open_, -1.0, 0.0))

        # ── Fast OFI: 12-bar rolling sum (1 min) ─────────────────────────
        # Detects short-term accumulation/distribution building up.
        # Range: [-12, +12]; normalise to [-1, +1] for threshold comparison.
        OFI_FAST_WINDOW = 12
        ofi_fast_raw = _rolling_sum(signed_dir, OFI_FAST_WINDOW)
        ofi_fast = ofi_fast_raw / OFI_FAST_WINDOW  # [-1, +1]

        # ── Slow OFI: 60-bar rolling sum (5 min) ─────────────────────────
        # Context filter — are we in an overall buying or selling regime?
        OFI_SLOW_WINDOW = 60
        ofi_slow = _rolling_sum(signed_dir, OFI_SLOW_WINDOW)  # raw direction sum

        # ── 1-minute price return (same window as fast OFI) ───────────────
        price_change_12 = np.zeros(n)
        price_change_12[OFI_FAST_WINDOW:] = (
            (close[OFI_FAST_WINDOW:] - close[:-OFI_FAST_WINDOW])
            / close[:-OFI_FAST_WINDOW]
        )

        # ── OFI / price divergence (primary alpha signal) ─────────────────
        # Bull divergence: flow is bullish but price flat/down → latent buying
        bull_divergence = (ofi_fast > ofi_threshold) & (price_change_12 <= 0.0)
        bear_divergence = (ofi_fast < -ofi_threshold) & (price_change_12 >= 0.0)

        # ── Trend continuation (secondary signal) ─────────────────────────
        # Both fast OFI and slow context agree with price direction
        bull_trend = (
            (ofi_fast > ofi_threshold)
            & (ofi_slow > 0.0)
            & (price_change_12 > 0.0)
        )
        bear_trend = (
            (ofi_fast < -ofi_threshold)
            & (ofi_slow < 0.0)
            & (price_change_12 < 0.0)
        )

        # ── VIX filter ────────────────────────────────────────────────────
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

        vix_ok = vix_close < 22.0

        # ── Session filter ────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── Final signals ─────────────────────────────────────────────────
        buy_ce = in_session & vix_ok & (bull_divergence | bull_trend)
        buy_pe = in_session & vix_ok & (bear_divergence | bear_trend)

        # Resolve simultaneous signals (divergence + trend can both fire on
        # the same bar in edge cases) — favour neither; skip that bar.
        both   = buy_ce & buy_pe
        buy_ce = buy_ce & ~both
        buy_pe = buy_pe & ~both

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,           # 120 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
