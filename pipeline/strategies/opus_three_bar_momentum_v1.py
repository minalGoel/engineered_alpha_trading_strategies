"""Three-Bar Momentum — Opus_16

Thesis: Three consecutive bars in the same direction with expanding volume
and expanding range signal strong momentum.  Enter on completion of the
third bar; exit on the first reversal bar.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Wilder's ATR."""
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i - 1]),
                     abs(low[i] - close[i - 1]))
    if n < period:
        return atr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


class Strategy(BaseStrategy):
    name = "opus_three_bar_momentum_v1"
    is_long_only = False
    session_start = 565   # 09:25
    session_end = 870     # 14:30
    max_trades_per_day = 12

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("stop_loss_pct", default=0.0015, low=0.001, high=0.003),
            TunableParam("target_pct", default=0.002, low=0.001, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        stop_pct = params.get("stop_loss_pct", 0.0015)
        target_pct = params.get("target_pct", 0.002)

        n = len(df)
        open_ = df["open"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        day_id = df["day_id"].to_numpy()

        bar_range = high - low
        is_green = close > open_
        is_red = close < open_

        atr14 = _compute_atr(high, low, close, 14)

        # ── Three consecutive bars same direction, expanding vol & range ──
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)

        for i in range(2, n):
            # Must be same day for all 3 bars
            if day_id[i] != day_id[i - 1] or day_id[i] != day_id[i - 2]:
                continue

            # Long: 3 green bars
            if (is_green[i] and is_green[i - 1] and is_green[i - 2] and
                    volume[i] > volume[i - 1] > volume[i - 2] and
                    bar_range[i] > bar_range[i - 1] > bar_range[i - 2]):
                long_entry[i] = True

            # Short: 3 red bars
            if (is_red[i] and is_red[i - 1] and is_red[i - 2] and
                    volume[i] > volume[i - 1] > volume[i - 2] and
                    bar_range[i] > bar_range[i - 1] > bar_range[i - 2]):
                short_entry[i] = True

        # ── Signal exit: first bar opposite direction ──
        sig_exit_long = is_red.copy()
        sig_exit_short = is_green.copy()

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr14,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            time_stop_bars=20,
        )
