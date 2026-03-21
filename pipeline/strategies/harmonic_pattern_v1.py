"""harmonic_pattern_v1 — NIFTY ABCD Harmonic Pattern Reversal

Edge: On NIFTY, institutional TWAP execution creates measurable AB=CD wave structures
at the 30-second to 10-minute scale. When the CD leg completes at 1.0x or 1.272x of AB
(Fibonacci exhaustion point), resting limit orders at the extension level trigger a
10-25 spot point snap-back within 15-90 seconds. We enter at the D-pivot confirmation
bar (30-second causal lag) and hold for up to 120 seconds.

Universe: NIFTY
Session: 09:25-15:20 IST
Hold: 15-120 seconds (3-24 bars at 5s)
"""
from __future__ import annotations

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "harmonic_pattern_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST
    session_end_minutes = 920     # 15:20 IST
    max_lookback = 720            # 60 min warmup — slowest patterns span ~30 min
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            # Bars required on each side to confirm a swing pivot (30s default = 6 bars)
            TunableParam("min_bars_pivot", 6.0, 3.0, 15.0),
            # Tolerance on CD/AB ratio for matching 1.0 or 1.272 Fibonacci levels
            TunableParam("ratio_tol", 0.08, 0.04, 0.15),
            # Stop in option premium points
            TunableParam("stop_pts", 5.0, 3.0, 10.0),
            # Target in option premium points
            TunableParam("target_pts", 8.0, 5.0, 15.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        min_bars = max(3, int(params.get("min_bars_pivot", 6.0)))
        ratio_tol = float(params.get("ratio_tol", 0.08))
        stop_pts = float(params.get("stop_pts", 5.0))
        target_pts = float(params.get("target_pts", 8.0))

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # VIX filter — patterns unreliable in stressed regimes
        vix_arr = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_arr = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        buy_ce = np.zeros(n, dtype=bool)
        buy_pe = np.zeros(n, dtype=bool)

        # Confirmed pivots tracked as list of [bar_idx, price, direction]
        # direction: +1 = high pivot, -1 = low pivot
        # Causal detection: pivot at bar k is confirmed at bar i = k + min_bars
        # because we need min_bars subsequent bars to confirm the extremum.
        pivots: list[list] = []

        # Track last fired pattern key to avoid re-firing same ABCD pattern
        last_ce_key: tuple | None = None
        last_pe_key: tuple | None = None

        for i in range(2 * min_bars, n):
            k = i - min_bars  # candidate pivot bar — confirmed at bar i

            # ── Check high pivot at bar k ──────────────────────────────────────
            # k is a high pivot if high[k] is strictly greater than all bars
            # within ±min_bars around it (checking only causal bars from bar i).
            # Bars k+1 to k+min_bars = i are all in the past at bar i.
            is_high = True
            for j in range(1, min_bars + 1):
                if k - j < 0:
                    is_high = False
                    break
                if high[k - j] >= high[k] or high[k + j] >= high[k]:
                    is_high = False
                    break

            # ── Check low pivot at bar k ───────────────────────────────────────
            if not is_high:
                is_low = True
                for j in range(1, min_bars + 1):
                    if k - j < 0:
                        is_low = False
                        break
                    if low[k - j] <= low[k] or low[k + j] <= low[k]:
                        is_low = False
                        break
            else:
                is_low = False

            if not (is_high or is_low):
                continue

            direction = 1 if is_high else -1
            price = high[k] if is_high else low[k]

            # ── Merge consecutive same-direction pivots (keep extreme) ─────────
            if pivots and pivots[-1][2] == direction:
                if direction == 1 and price > pivots[-1][1]:
                    pivots[-1] = [k, price, direction]
                elif direction == -1 and price < pivots[-1][1]:
                    pivots[-1] = [k, price, direction]
                # else: current pivot is not more extreme; keep existing
            else:
                pivots.append([k, price, direction])

            # Keep only the 4 most recent pivots (A, B, C, D)
            if len(pivots) > 4:
                pivots = pivots[-4:]

            if len(pivots) < 4:
                continue

            A_bar, A_p, A_d = pivots[0]
            B_bar, B_p, B_d = pivots[1]
            C_bar, C_p, C_d = pivots[2]
            D_bar, D_p, D_d = pivots[3]

            AB = abs(B_p - A_p)
            if AB < 5.0:
                # Minimum AB = 5 NIFTY spot points — filters noise micro-patterns
                continue

            BC = abs(C_p - B_p)
            CD = abs(D_p - C_p)

            bc_ratio = BC / AB
            cd_ratio = CD / AB

            # BC: must retrace 38.2%-88.6% of AB (standard ABCD filter)
            valid_bc = 0.382 <= bc_ratio <= 0.886

            # CD: must be near 1.0x (AB=CD) or 1.272x (Fibonacci extension)
            valid_cd = (
                (abs(cd_ratio - 1.0) <= ratio_tol)
                or (abs(cd_ratio - 1.272) <= ratio_tol)
            )

            if not (valid_bc and valid_cd):
                continue

            if not in_session[i]:
                continue

            if vix_arr[i] >= 22.0:
                # High-VIX regime: institutional wave structure is disrupted
                continue

            pattern_key = (A_bar, B_bar, C_bar, D_bar)

            # ── Bearish ABCD → bullish reversal → buy CE ───────────────────────
            # A=high, B=low, C=high (C < A), D=low
            # CD goes down to Fibonacci extension → expect bounce back up
            if (
                A_d == 1 and B_d == -1 and C_d == 1 and D_d == -1
                and C_p < A_p
                and pattern_key != last_ce_key
            ):
                buy_ce[i] = True
                last_ce_key = pattern_key

            # ── Bullish ABCD → bearish reversal → buy PE ──────────────────────
            # A=low, B=high, C=low (C > A), D=high
            # CD goes up to Fibonacci extension → expect pullback down
            elif (
                A_d == -1 and B_d == 1 and C_d == -1 and D_d == 1
                and C_p > A_p
                and pattern_key != last_pe_key
            ):
                buy_pe[i] = True
                last_pe_key = pattern_key

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
