from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    """Last 30-minute closing momentum on NIFTY.

    Institutional VWAP-benchmarked funds must fill final allocations in the
    14:45-15:20 window. When NIFTY has a >0.3% directional day and sits on the
    right side of VWAP, closing order bursts reinforce the day trend in 30-90s
    micro-accelerations. We enter each burst dynamically using 1-min momentum
    rather than a single fixed-time entry.
    """

    name = "last_30_min_momentum_v1"
    underlying = "NIFTY"
    session_start_minutes = 555   # 09:15 IST — full day needed for VWAP and day_return
    session_end_minutes = 920     # 15:20 IST — no new entries after this
    max_trades_per_day = 6
    max_lookback = 60             # 5-min warmup; cumulative measures start from session open

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("day_return_threshold", 0.003, 0.001, 0.008),
            TunableParam("mom_1min_threshold", 0.0005, 0.0001, 0.002),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 6.0, 4.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill before numpy conversion
        close = spot_df.select(pl.col("close").forward_fill()).to_series().to_numpy()
        volume = (
            spot_df.select(pl.col("volume").fill_null(0)).to_series().to_numpy().astype(np.float64)
        )
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        day_ret_thr = params.get("day_return_threshold", 0.003)
        mom_thr = params.get("mom_1min_threshold", 0.0005)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 6.0)

        # ── Day return from session open (reset per day) ──────────────────────
        # Use first bar's close of each day as the reference open price.
        # Cumulative measure: no scaling needed, fixed-time window from 09:15.
        day_return = np.zeros(n)
        prev_day = -1
        session_open_price = close[0] if n > 0 else 1.0
        for i in range(n):
            if day_id[i] != prev_day:
                prev_day = day_id[i]
                session_open_price = close[i]
            if session_open_price > 0.0:
                day_return[i] = (close[i] - session_open_price) / session_open_price

        # ── Session VWAP from open (reset per day) ────────────────────────────
        vwap = np.zeros(n)
        cum_pv = 0.0
        cum_v = 0.0
        prev_day2 = -1
        for i in range(n):
            if day_id[i] != prev_day2:
                cum_pv = 0.0
                cum_v = 0.0
                prev_day2 = day_id[i]
            cum_pv += close[i] * volume[i]
            cum_v += volume[i]
            vwap[i] = cum_pv / cum_v if cum_v > 0.0 else close[i]

        # ── 1-minute momentum (12 x 5s bars) — trade trigger ─────────────────
        # Detects each institutional burst as it begins within the closing window.
        # 12 bars (1 min) matches our 30-90s hold: burst must be visible at
        # 1-min scale to carry. Shorter is too noisy; longer misses the entry.
        mom_12 = np.zeros(n)
        for i in range(12, n):
            ref = close[i - 12]
            if ref > 0.0:
                mom_12[i] = (close[i] - ref) / ref

        # ── Time filter: 14:45-15:20 IST (885-920 min from midnight) ─────────
        in_closing_window = (time_min >= 885) & (time_min < 920)

        # ── Entry signals ─────────────────────────────────────────────────────
        # Bullish: up day, above VWAP, positive 1-min burst
        buy_ce = (
            in_closing_window
            & (day_return > day_ret_thr)
            & (close > vwap)
            & (mom_12 > mom_thr)
        )

        # Bearish: down day, below VWAP, negative 1-min burst
        buy_pe = (
            in_closing_window
            & (day_return < -day_ret_thr)
            & (close < vwap)
            & (mom_12 < -mom_thr)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,            # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
