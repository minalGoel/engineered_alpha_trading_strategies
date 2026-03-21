"""gap_vwap_momentum_trifecta_v1 — Gap + VWAP bounce + 30s momentum on NIFTY/BANKNIFTY.

Mechanism: NIFTY/BANKNIFTY gaps at open due to overnight FII/DII imbalances. Institutional
VWAP execution desks treat the session VWAP as their benchmark and absorb counter-trend flow
when price approaches it. When the index pulls back to VWAP and a 5-second bar closes back in
the gap direction with 30-second momentum confirmation, multiple VWAP desks simultaneously
accelerate fills — compressing a 15-60 min stock-level bounce into 30-90 seconds on the index.

Session: 09:20-11:30 IST (gap information decays after 2h)
Hold: 30-120s (6-24 bars at 5s)
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "gap_vwap_momentum_trifecta_v1"
    underlying = "BOTH"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 690     # 11:30 IST — gap trades lose edge after 2h
    max_trades_per_day = 4
    max_lookback = 12             # VWAP is cumulative from open; minimal warmup needed

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_min_pct", 0.3, 0.15, 0.8),
            TunableParam("vwap_touch_threshold", 0.0005, 0.0002, 0.002),
            TunableParam("stop_pts", 5.0, 3.0, 9.0),
            TunableParam("target_pts", 9.0, 5.0, 15.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_nan(None).fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_nan(None).fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_nan(None).fill_null(strategy="forward").to_numpy()
        open_ = spot_df["open"].fill_nan(None).fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        day_id = spot_df["day_id"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        gap_min = params.get("gap_min_pct", 0.3)
        touch_thr = params.get("vwap_touch_threshold", 0.0005)
        stop_pts = params.get("stop_pts", 5.0)
        target_pts = params.get("target_pts", 9.0)

        # ── Daily gap computation ──────────────────────────────────────────────
        # Use first bar open as day_open and last bar close as day_close per day_id.
        day_stats = (
            spot_df
            .group_by("day_id")
            .agg([
                pl.col("open").first().alias("day_open"),
                pl.col("close").last().alias("day_close"),
            ])
            .sort("day_id")
        )
        day_ids_sorted = day_stats["day_id"].to_numpy()
        day_opens = day_stats["day_open"].fill_nan(None).fill_null(strategy="forward").to_numpy()
        day_closes = day_stats["day_close"].fill_nan(None).fill_null(strategy="forward").to_numpy()

        # Build {day_id: gap_pct} dict
        gap_by_day: dict[int, float] = {}
        for idx in range(len(day_ids_sorted)):
            if idx == 0:
                gap_by_day[int(day_ids_sorted[idx])] = 0.0
            else:
                prev_c = day_closes[idx - 1]
                cur_o = day_opens[idx]
                if prev_c > 0:
                    gap_by_day[int(day_ids_sorted[idx])] = (cur_o - prev_c) / prev_c * 100.0
                else:
                    gap_by_day[int(day_ids_sorted[idx])] = 0.0

        # ── Session VWAP (cumulative, reset per day) ───────────────────────────
        vwap = np.zeros(n)
        cum_pv = 0.0
        cum_v = 0.0
        prev_d = -99999
        for i in range(n):
            d = int(day_id[i])
            if d != prev_d:
                cum_pv = 0.0
                cum_v = 0.0
                prev_d = d
            v = volume[i]
            if v > 0.0:
                cum_pv += close[i] * v
                cum_v += v
            vwap[i] = cum_pv / cum_v if cum_v > 0.0 else close[i]

        # ── VIX filter ─────────────────────────────────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Signal generation ──────────────────────────────────────────────────
        buy_ce = np.zeros(n, dtype=bool)
        buy_pe = np.zeros(n, dtype=bool)

        prev_d = -99999
        gap_dir = 0           # +1 gap-up, -1 gap-down, 0 no gap
        touched_vwap = False  # True once price has come within touch_thr of VWAP today
        last_signal_bar = -999

        for i in range(n):
            d = int(day_id[i])
            t = int(time_min[i])

            # ── New day: reset per-day state ───────────────────────────────────
            if d != prev_d:
                g = gap_by_day.get(d, 0.0)
                if g >= gap_min:
                    gap_dir = 1
                elif g <= -gap_min:
                    gap_dir = -1
                else:
                    gap_dir = 0
                touched_vwap = False
                last_signal_bar = -999
                prev_d = d

            # Session and gap filter
            if gap_dir == 0:
                continue
            if t < self.session_start_minutes or t >= self.session_end_minutes:
                continue

            # VIX regime filter
            if vix_close[i] >= 22.0:
                continue

            vwap_val = vwap[i]
            if vwap_val <= 0.0:
                continue

            # Cooldown: no re-entry within 12 bars (60s) of last signal
            if i - last_signal_bar < 12:
                continue

            # ── VWAP touch detection ───────────────────────────────────────────
            if not touched_vwap:
                if gap_dir > 0:
                    # Gap-up: watch for pullback — low comes within touch_thr of VWAP or below
                    if low[i] <= vwap_val * (1.0 + touch_thr):
                        touched_vwap = True
                else:
                    # Gap-down: watch for bounce — high comes within touch_thr of VWAP or above
                    if high[i] >= vwap_val * (1.0 - touch_thr):
                        touched_vwap = True

            # ── Bounce/rejection signal after VWAP touch ──────────────────────
            if touched_vwap and i >= 6:
                mom_6 = close[i] - close[i - 6]  # 30-second momentum

                if gap_dir > 0:
                    # Bounce above VWAP: current close crosses from at/below to above
                    if close[i] > vwap_val and close[i - 1] <= vwap_val and mom_6 > 0.0:
                        buy_ce[i] = True
                        last_signal_bar = i
                        touched_vwap = False  # reset to avoid chasing repeated touches

                else:
                    # Rejection below VWAP: current close crosses from at/above to below
                    if close[i] < vwap_val and close[i - 1] >= vwap_val and mom_6 < 0.0:
                        buy_pe[i] = True
                        last_signal_bar = i
                        touched_vwap = False

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
