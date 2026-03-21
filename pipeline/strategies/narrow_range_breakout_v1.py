"""narrow_range_breakout_v1 — NR7 micro-coil breakout on NIFTY.

Thesis:
    When NIFTY prints a bar with the narrowest high-low range in 7 consecutive
    5-second bars (~35s of micro-consolidation), institutional TWAP/VWAP algos
    have reached equilibrium, coiling resting orders at the NR7 high and low.
    A 5-second close above/below the NR7 bar triggers momentum and delta-hedging
    flow, extending the move 15-25 spot points over the next 30-60 seconds.
    Trade only in the direction of the 5-minute EMA to filter counter-trend fakeouts.

Adapted from:
    Original NR7 equity strategy (Strategy_34.json) — 1-min bars, top-30 FnO stocks.
    NR7 lookback kept at 7 bars (not 12x-scaled) because relative-narrowness is
    bar-size agnostic. EMA compressed from 20-min to 5-min to match 15-60s hold.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


def _compute_ema(close: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average, initialised to close[0]."""
    n = len(close)
    ema = np.empty(n)
    if n == 0:
        return ema
    ema[0] = close[0]
    alpha = 2.0 / (period + 1)
    for i in range(1, n):
        ema[i] = close[i] * alpha + ema[i - 1] * (1.0 - alpha)
    return ema


class Strategy(BaseStrategy):
    name = "narrow_range_breakout_v1"
    underlying = "NIFTY"
    session_start_minutes = 560    # 09:20 IST
    session_end_minutes = 925      # 15:25 IST
    max_lookback = 120             # 10-min warmup covers EMA(60) + NR7(7)
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 6.0, 4.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill NaN before converting to numpy
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 6.0)

        # ── Per-bar range ────────────────────────────────────────────────────
        bar_range = high - low   # shape (n,)

        # ── NR7: True when current bar has the narrowest range in a 7-bar window ──
        # 7 bars at 5s = 35 seconds of micro-consolidation; lookback matches original
        # (not 12x-scaled) because NR7 is a relative, bar-size-agnostic criterion.
        nr7 = np.zeros(n, dtype=bool)
        for i in range(6, n):
            r = bar_range[i]
            if r > 0.0 and r == np.min(bar_range[i - 6 : i + 1]):
                nr7[i] = True

        # ── EMA(60) = 5-minute EMA — trend direction context ─────────────────
        # Original used EMA(20) on 1-min = 20-min; compressed to 5-min (EMA_60 on 5s)
        # because we hold 15-60 seconds and need a trend filter at that horizon.
        ema_60 = _compute_ema(close, 60)

        # ── Session filter ────────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── Entry signals ─────────────────────────────────────────────────────
        # Entry on the bar AFTER the NR7 bar:
        #   buy_ce: close[i] > high[i-1]  AND  close[i-1] > ema_60[i-1] (uptrend NR7)
        #   buy_pe: close[i] < low[i-1]   AND  close[i-1] < ema_60[i-1] (downtrend NR7)
        buy_ce = np.zeros(n, dtype=bool)
        buy_pe = np.zeros(n, dtype=bool)

        for i in range(7, n):
            if not in_session[i]:
                continue
            if not nr7[i - 1]:
                continue
            prev_high = high[i - 1]
            prev_low = low[i - 1]
            prev_close = close[i - 1]
            prev_ema = ema_60[i - 1]
            cur_close = close[i]
            # Bullish breakout above NR7 bar high, in uptrend
            if cur_close > prev_high and prev_close > prev_ema:
                buy_ce[i] = True
            # Bearish breakout below NR7 bar low, in downtrend
            elif cur_close < prev_low and prev_close < prev_ema:
                buy_pe[i] = True

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=12,              # 60 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
