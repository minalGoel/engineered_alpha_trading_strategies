"""
prev_day_hl_reversal — NIFTY PDH/PDL liquidity-hunt fade.

When NIFTY briefly pierces the Previous Day High (or Low) within a 5-second bar
but closes back inside the range, retail stop-order clusters have been absorbed,
trapping breakout participants. Institutional algos pre-positioned to fade the
liquidity grab then push the index 15-25 spot points back toward VWAP within
30-90 seconds.

Trade: buy CE on PDL false-breakdown, buy PE on PDH false-breakout.
Hold: 15-90 seconds (time_stop_bars=18).
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "prev_day_hl_reversal"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 6
    max_lookback = 300            # 25 min warmup for RSI stabilisation

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rsi_oversold",  35.0, 20.0, 48.0),
            TunableParam("rsi_overbought", 65.0, 52.0, 80.0),
            TunableParam("stop_pts",       5.0,  3.0, 10.0),
            TunableParam("target_pts",     8.0,  5.0, 15.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── raw arrays (forward-fill NaN) ──────────────────────────────────
        close   = spot_df["close"].fill_nan(None).fill_null(strategy="forward").to_numpy()
        high_arr = spot_df["high"].fill_nan(None).fill_null(strategy="forward").to_numpy()
        low_arr  = spot_df["low"].fill_nan(None).fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id   = spot_df["day_id"].to_numpy()

        # ── parameters ─────────────────────────────────────────────────────
        rsi_oversold  = params.get("rsi_oversold",  35.0)
        rsi_overbought = params.get("rsi_overbought", 65.0)
        stop_pts      = params.get("stop_pts",       5.0)
        target_pts    = params.get("target_pts",     8.0)

        # ── previous day high / low ─────────────────────────────────────────
        # Group by day_id to get each session's full high/low,
        # then look up (day_id - 1) for each bar.
        daily_hl = (
            spot_df.select(["day_id", "high", "low"])
            .group_by("day_id")
            .agg([
                pl.col("high").max().alias("day_high"),
                pl.col("low").min().alias("day_low"),
            ])
        )
        day_high_dict: dict[int, float] = {}
        day_low_dict:  dict[int, float] = {}
        for row in daily_hl.iter_rows(named=True):
            day_high_dict[int(row["day_id"])] = float(row["day_high"])
            day_low_dict[int(row["day_id"])]  = float(row["day_low"])

        # Sentinel values that will never satisfy probe conditions
        pdh = np.array([day_high_dict.get(int(d) - 1, 1e10) for d in day_id])
        pdl = np.array([day_low_dict.get(int(d) - 1, 0.0)   for d in day_id])

        # ── RSI(24) — 2-minute momentum confirmation ────────────────────────
        rsi = _compute_rsi(close, 24)

        # ── masks ───────────────────────────────────────────────────────────
        in_session   = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        has_prev_day = (pdh < 1e9) & (pdl > 0.0)

        # Bullish: PDL false-breakdown (low < pdl, close > pdl)
        buy_ce = (
            in_session
            & has_prev_day
            & (low_arr < pdl)
            & (close > pdl)
            & (rsi < rsi_oversold)
        )

        # Bearish: PDH false-breakout (high > pdh, close < pdh)
        buy_pe = (
            in_session
            & has_prev_day
            & (high_arr > pdh)
            & (close < pdh)
            & (rsi > rsi_overbought)
        )

        # Mutual exclusion — probing both levels simultaneously is impossible
        # in practice but guard defensively
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
            time_stop_bars=18,             # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )


# ── helpers ────────────────────────────────────────────────────────────────────

def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder's smoothed RSI. Returns neutral 50.0 during warmup period."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi

    delta  = np.diff(close)
    gains  = np.maximum(delta, 0.0)
    losses = np.maximum(-delta, 0.0)

    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    for i in range(period, n - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss < 1e-10:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    return rsi
