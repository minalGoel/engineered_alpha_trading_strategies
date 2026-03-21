"""
previous_day_high_low_breakout_v1

Converted from: trading_strategies/unique_strategies_all/Strategy_32.json
Original: PDH/PDL breakout on Nifty50 stocks, 1-min bars, 15-60 min hold.

Mechanism: When NIFTY/BANKNIFTY crosses its prior day's high or low, two forces
create a directional burst: (1) stop-loss cascades from traders who placed stops
just beyond PDH/PDL, and (2) option market-maker delta-hedging as strikes go
in-the-money. We enter after 3 consecutive 5s bars closing beyond the level
(15s confirmation) and ride the first 15-25 NIFTY spot points of continuation.
"""
from __future__ import annotations
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "previous_day_high_low_breakout_v1"
    underlying = "BOTH"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 3
    max_lookback = 300            # 25 min warmup for vol_surge SMA(60) + buffer

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_thresh", 1.5, 1.0, 3.0),
            TunableParam("vix_thresh", 22.0, 15.0, 28.0),
            TunableParam("break_min_bps", 3.0, 1.0, 8.0),
            TunableParam("break_max_bps", 20.0, 10.0, 35.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        vol_thresh = params.get("vol_thresh", 1.5)
        vix_thresh = params.get("vix_thresh", 22.0)
        break_min_bps = params.get("break_min_bps", 3.0)
        break_max_bps = params.get("break_max_bps", 20.0)

        # ── Raw arrays ──────────────────────────────────────────────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()

        # session_date as Python objects for grouping
        session_dates = spot_df["session_date"].to_list()

        # ── Compute prior-day high and low per bar ───────────────────────────
        # Group bars by session_date
        date_to_bars: dict = {}
        for i, d in enumerate(session_dates):
            if d not in date_to_bars:
                date_to_bars[d] = []
            date_to_bars[d].append(i)

        # Compute max high and min low for each trading day
        day_high: dict = {}
        day_low: dict = {}
        for d, idxs in date_to_bars.items():
            idxs_arr = np.array(idxs)
            day_high[d] = np.max(high[idxs_arr])
            day_low[d] = np.min(low[idxs_arr])

        # Sort unique dates to map each date to its prior
        unique_dates = sorted(date_to_bars.keys())
        prev_date_map = {unique_dates[i]: unique_dates[i - 1] for i in range(1, len(unique_dates))}

        # Assign PDH/PDL to each bar (NaN for first day — no prior day)
        pdh = np.full(n, np.nan)
        pdl = np.full(n, np.nan)
        for i, d in enumerate(session_dates):
            if d in prev_date_map:
                prior = prev_date_map[d]
                pdh[i] = day_high[prior]
                pdl[i] = day_low[prior]

        # ── Distance from PDH/PDL in basis points ───────────────────────────
        with np.errstate(invalid="ignore", divide="ignore"):
            dist_from_pdh_bps = np.where(
                ~np.isnan(pdh) & (pdh > 0),
                (close - pdh) / pdh * 10000.0,
                np.nan,
            )
            dist_from_pdl_bps = np.where(
                ~np.isnan(pdl) & (pdl > 0),
                (pdl - close) / pdl * 10000.0,
                np.nan,
            )

        # ── Consecutive bars above PDH / below PDL ──────────────────────────
        consec_above_pdh = np.zeros(n, dtype=np.float64)
        consec_below_pdl = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if not np.isnan(pdh[i]) and close[i] > pdh[i]:
                consec_above_pdh[i] = consec_above_pdh[i - 1] + 1.0
            else:
                consec_above_pdh[i] = 0.0
            if not np.isnan(pdl[i]) and close[i] < pdl[i]:
                consec_below_pdl[i] = consec_below_pdl[i - 1] + 1.0
            else:
                consec_below_pdl[i] = 0.0

        # ── Volume surge: current bar vs 5-minute (60-bar) rolling mean ─────
        vol_sma60 = np.zeros(n)
        for i in range(60, n):
            vol_sma60[i] = np.mean(volume[i - 60:i])
        vol_surge = np.where(vol_sma60 > 0, volume / vol_sma60, 0.0)

        # ── Session VWAP (cumulative, reset each day) ────────────────────────
        vwap = np.zeros(n)
        cum_pv = 0.0
        cum_vol = 0.0
        prev_date = None
        for i in range(n):
            d = session_dates[i]
            if d != prev_date:
                cum_pv = close[i] * max(volume[i], 1.0)
                cum_vol = max(volume[i], 1.0)
                prev_date = d
            else:
                v = max(volume[i], 1.0)
                cum_pv += close[i] * v
                cum_vol += v
            vwap[i] = cum_pv / cum_vol

        # ── VIX aligned to spot bars ─────────────────────────────────────────
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

        # ── Session filter ────────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # ── Entry masks ───────────────────────────────────────────────────────
        has_pdh = ~np.isnan(pdh)
        has_pdl = ~np.isnan(pdl)

        # buy CE: PDH breakout
        buy_ce = (
            in_session
            & has_pdh
            & (consec_above_pdh >= 3)
            & (dist_from_pdh_bps >= break_min_bps)
            & (dist_from_pdh_bps <= break_max_bps)
            & (vol_surge > vol_thresh)
            & (close > vwap)
            & (vix_close < vix_thresh)
        )

        # buy PE: PDL breakdown
        buy_pe = (
            in_session
            & has_pdl
            & (consec_below_pdl >= 3)
            & (dist_from_pdl_bps >= break_min_bps)
            & (dist_from_pdl_bps <= break_max_bps)
            & (vol_surge > vol_thresh)
            & (close < vwap)
            & (vix_close < vix_thresh)
        )

        # No signal-based exits (use stop/target/time-stop only)
        sell_ce = np.zeros(n, dtype=bool)
        sell_pe = np.zeros(n, dtype=bool)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=sell_ce,
            sell_pe=sell_pe,
            stop_points=np.full(n, 4.0),
            target_points=np.full(n, 7.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
