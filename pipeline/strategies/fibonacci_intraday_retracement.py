"""Fibonacci Intraday Retracement — NIFTY 5-second options strategy.

Original: fibonacci_intraday_retracement on nifty100 stocks, 1-min bars, 60-120 min hold.
Converted: NIFTY index 5s bars, 15-90 second hold.

Mechanism: NIFTY's 09:15-10:15 opening range crystallizes overnight positioning into a high-low
anchor. The 61.8% Fibonacci retracement of this range is pre-loaded as a standing limit-order
cluster by systematic algos — making it a self-fulfilling price magnet. When price wicks into this
level mid-session and closes back through it within a 5-second bar, the dense resting order queue
provides a microstructure impulse: VWAP-benchmarked funds absorb the dip aggressively (long side),
or delta-hedging from option sellers adds directional supply (short side).
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "fibonacci_intraday_retracement"
    underlying = "NIFTY"
    session_start_minutes = 660   # 11:00 IST (after morning range fully forms at 10:15 + buffer)
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 4
    max_lookback = 720            # 60 min warmup to compute morning range (720 × 5s)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("min_range_pct", 0.003, 0.001, 0.008),  # min morning range as fraction of price
            TunableParam("stop_pts", 3.0, 2.0, 8.0),
            TunableParam("target_pts", 6.0, 4.0, 14.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Extract spot arrays — forward-fill NaN before converting
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        min_range_pct = params.get("min_range_pct", 0.003)
        stop_pts = params.get("stop_pts", 3.0)
        target_pts = params.get("target_pts", 6.0)

        # ── Compute session VWAP (resets each day) ──────────────────────────────
        vwap = np.zeros(n)
        prev_day = -1
        running_tp_vol = 0.0
        running_vol = 0.0
        for i in range(n):
            d = day_id[i]
            if d != prev_day:
                running_tp_vol = 0.0
                running_vol = 0.0
                prev_day = d
            tp = (high[i] + low[i] + close[i]) / 3.0
            v = volume[i]
            running_tp_vol += tp * v
            running_vol += v
            vwap[i] = running_tp_vol / running_vol if running_vol > 0.0 else close[i]

        # ── Compute morning range per day (09:15-10:15 = minutes 555-615) ────────
        MORNING_START = 555   # 09:15 IST
        MORNING_END = 615     # 10:15 IST (exclusive)

        # Pass 1: accumulate morning high/low per day_id
        day_mh = {}  # day_id -> float
        day_ml = {}  # day_id -> float
        for i in range(n):
            d = day_id[i]
            t = time_min[i]
            if MORNING_START <= t < MORNING_END:
                if d not in day_mh:
                    day_mh[d] = high[i]
                    day_ml[d] = low[i]
                else:
                    if high[i] > day_mh[d]:
                        day_mh[d] = high[i]
                    if low[i] < day_ml[d]:
                        day_ml[d] = low[i]

        # Pass 2: broadcast morning values to every bar in that day
        morning_high = np.full(n, np.nan)
        morning_low = np.full(n, np.nan)
        for i in range(n):
            d = day_id[i]
            if d in day_mh:
                morning_high[i] = day_mh[d]
                morning_low[i] = day_ml[d]

        # ── Fibonacci 61.8% level ────────────────────────────────────────────────
        # Single level: morning_high - 0.618 * (morning_high - morning_low)
        # = the classic 61.8% retracement between the session high and low anchors
        morning_range = morning_high - morning_low
        fib_618 = morning_high - morning_range * 0.618

        # Morning range as fraction of price (to filter tiny ranges)
        morning_range_pct = np.where(
            morning_low > 0.0,
            morning_range / morning_low,
            0.0,
        )

        # ── Masks ────────────────────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        valid_range = morning_range_pct > min_range_pct
        valid_levels = np.isfinite(fib_618)

        # Buy CE: wick into / below fib_618 (support), close back above + VWAP bullish
        buy_ce = (
            in_session
            & valid_range
            & valid_levels
            & (low <= fib_618)    # candle touched or pierced support level
            & (close > fib_618)   # closed back above — bounce confirmed
            & (close > vwap)      # institutional regime: net buyers
        )

        # Buy PE: wick into / above fib_618 (resistance), close back below + VWAP bearish
        buy_pe = (
            in_session
            & valid_range
            & valid_levels
            & (high >= fib_618)   # candle touched or pierced resistance level
            & (close < fib_618)   # closed back below — rejection confirmed
            & (close < vwap)      # institutional regime: net sellers
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,        # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
