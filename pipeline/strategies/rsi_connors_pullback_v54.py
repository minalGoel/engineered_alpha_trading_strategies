"""rsi_connors_pullback_v54 — BANKNIFTY dual-RSI pullback strategy.

Dual-RSI approach: RSI(24) (2-min) as momentum-burst direction filter,
RSI(6) (30s) as micro-pullback entry trigger. Inspired by Connors RSI-2
'any dip within trend' concept applied to BANKNIFTY's institutional burst cycle.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


def _wilder_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder smoothed RSI. Returns 50.0 (neutral) before warmup."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi

    delta = np.empty(n)
    delta[0] = 0.0
    delta[1:] = close[1:] - close[:-1]

    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)

    # Seed: simple average of first `period` changes
    avg_gain[period] = gains[1:period + 1].mean()
    avg_loss[period] = losses[1:period + 1].mean()

    # Wilder smoothing forward
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period

    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.where(avg_loss[period:] > 1e-10,
                      avg_gain[period:] / avg_loss[period:],
                      np.inf)
        rsi[period:] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


class Strategy(BaseStrategy):
    """BANKNIFTY dual-RSI pullback: RSI(24) trend state + RSI(6) entry trigger.

    Fires buy_ce when the 2-minute RSI (RSI(24)) confirms a bullish burst
    (>55) and the 30-second RSI (RSI(6)) shows a micro-pullback (<45).
    Symmetric for buy_pe. No SMA/VWAP price-level anchor — purely
    momentum-state driven, exploiting BANKNIFTY's bursty institutional flow.
    """

    name = "rsi_connors_pullback_v54"
    underlying = "BANKNIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 8
    max_lookback = 60             # 5-min warmup — covers RSI(24)+2 initialization

    def tunable_params(self) -> list[TunableParam]:
        return [
            # RSI(24) threshold: burst is active above this level
            TunableParam("rsi_trend_threshold", 55.0, 50.0, 65.0),
            # RSI(6) threshold: micro-pullback below this level fires long entry
            TunableParam("rsi_entry_low", 45.0, 35.0, 52.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill NaN before converting to numpy
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        rsi_trend_threshold = params.get("rsi_trend_threshold", 55.0)
        rsi_entry_low = params.get("rsi_entry_low", 45.0)
        # Symmetric thresholds for short side
        rsi_anti_threshold = 100.0 - rsi_trend_threshold   # default 45.0
        rsi_entry_high = 100.0 - rsi_entry_low             # default 55.0

        # ── Indicators ──────────────────────────────────────────────────────
        # 2-minute RSI: momentum-burst direction filter (24 bars × 5s = 120s)
        rsi_24 = _wilder_rsi(close, 24)

        # 30-second RSI: micro-pullback entry trigger (6 bars × 5s = 30s)
        rsi_6 = _wilder_rsi(close, 6)

        # ── Filters ─────────────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes) &
            (time_min < self.session_end_minutes)
        )

        # RSI(24) needs 24+2 bars to be fully seeded
        warmed_up = np.zeros(n, dtype=bool)
        warmed_up[26:] = True

        # ── Signals ─────────────────────────────────────────────────────────
        # buy_ce: 2-min burst bullish AND 30s micro-pullback
        buy_ce = (
            in_session &
            warmed_up &
            (rsi_24 > rsi_trend_threshold) &
            (rsi_6 < rsi_entry_low)
        )

        # buy_pe: 2-min burst bearish AND 30s micro-bounce
        buy_pe = (
            in_session &
            warmed_up &
            (rsi_24 < rsi_anti_threshold) &
            (rsi_6 > rsi_entry_high)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, 5.0),
            target_points=np.full(n, 8.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,           # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
