"""gap_continuation_v1 — Gap-and-go opening range breakout on NIFTY.

When NIFTY opens with a meaningful gap (>0.3%) driven by overnight global cues,
counter-trend traders post limit sell orders at the opening range high (ORH)
of the first 5 minutes. A 5-second bar closing above the ORH signals supply
absorption; the thin order book above the shelf drives 15-25 spot point
continuation within 30-90 seconds as institutional TWAP programs execute
unencumbered. Only trades the 09:20-10:15 window where gap momentum is active.

Differentiated from gap_continuation_momentum_v1 which uses session_open
gap-holding + mom_12 trigger after 09:35. This strategy uses the structural
ORH/ORL breakout signal immediately after the opening range forms.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average (no NaN-aware version needed; caller fills NaN)."""
    k = 2.0 / (period + 1)
    out = np.empty(len(arr), dtype=np.float64)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1.0 - k)
    return out


class Strategy(BaseStrategy):
    name = "gap_continuation_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST — entry starts after OR forms
    session_end_minutes = 615     # 10:15 IST — gap momentum window only
    max_lookback = 240            # 20 min warmup to seed EMA(156)
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_threshold", 0.003, 0.001, 0.008),  # min gap fraction
            TunableParam("stop_pts",    4.0, 2.0,  8.0),
            TunableParam("target_pts",  7.0, 4.0, 14.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── Raw arrays (forward-fill NaN before converting) ──
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        open_arr = spot_df["open"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        day_id = spot_df["day_id"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        gap_threshold = params.get("gap_threshold", 0.003)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 7.0)

        # ── EMAs on close ──
        ema_60 = _ema(close, 60)    # 5-min EMA — trade trigger
        ema_156 = _ema(close, 156)  # 13-min EMA — context filter

        # ── Daily first open and last close ──
        sorted_days = sorted(np.unique(day_id).tolist())
        day_first_open: dict[int, float] = {}
        day_last_close: dict[int, float] = {}
        for d in sorted_days:
            mask = day_id == d
            idxs = np.where(mask)[0]
            day_first_open[d] = float(open_arr[idxs[0]])
            day_last_close[d] = float(close[idxs[-1]])

        # ── Daily gap_pct: (first_open - prev_close) / prev_close ──
        day_gap: dict[int, float] = {}
        for i, d in enumerate(sorted_days):
            if i == 0:
                day_gap[d] = 0.0  # no prior day
            else:
                prev_d = sorted_days[i - 1]
                pc = day_last_close[prev_d]
                fo = day_first_open[d]
                day_gap[d] = (fo - pc) / pc if pc != 0.0 else 0.0

        gap_pct = np.array([day_gap[int(d)] for d in day_id], dtype=np.float64)

        # ── Opening range: first 5 min (09:15-09:20, time_minutes 555-559) ──
        day_orh: dict[int, float] = {}
        day_orl: dict[int, float] = {}
        for d in sorted_days:
            or_mask = (day_id == d) & (time_min >= 555) & (time_min < 560)
            if np.any(or_mask):
                day_orh[d] = float(np.max(high[or_mask]))
                day_orl[d] = float(np.min(low[or_mask]))
            else:
                # No opening range bars for this day (e.g. data starts late)
                # Use sentinel values that will never trigger
                day_orh[d] = 1e10
                day_orl[d] = -1e10

        orh = np.array([day_orh[int(d)] for d in day_id], dtype=np.float64)
        orl = np.array([day_orl[int(d)] for d in day_id], dtype=np.float64)

        # ── Bar close strength: fraction of bar range occupied by close ──
        bar_range = high - low
        close_strength = np.where(
            bar_range > 0.1,
            (close - low) / bar_range,
            0.5,  # neutral when bar is flat
        )

        # ── Session filter ──
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── VIX filter: 13-25 for gap continuation (calm enough to trend) ──
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

        vix_ok = (vix_close >= 13.0) & (vix_close <= 25.0)

        # ── Entry signals ──
        # buy_ce: gap-up + ORH breakout + EMA bullish + strong bar close
        buy_ce = (
            in_session
            & vix_ok
            & (gap_pct > gap_threshold)
            & (close > orh)
            & (ema_60 > ema_156)
            & (close_strength > 0.70)
        )

        # buy_pe: gap-down + ORL breakdown + EMA bearish + weak bar close
        buy_pe = (
            in_session
            & vix_ok
            & (gap_pct < -gap_threshold)
            & (close < orl)
            & (ema_60 < ema_156)
            & (close_strength < 0.30)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
