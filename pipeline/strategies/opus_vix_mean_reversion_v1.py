"""VIX Intraday Mean Reversion — Opus_25

Thesis: When VIX spikes >3% from its opening level but then starts
declining, the initial panic is fading and equities tend to recover.
Conversely, a VIX drop >3% with a rising reversal signals complacency
unwinding.
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
    name = "opus_vix_mean_reversion_v1"
    is_long_only = False
    session_start = 600   # 10:00
    session_end = 870     # 14:30
    max_trades_per_day = 2

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vix_change_thresh", default=0.03, low=0.02, high=0.05),
            TunableParam("stop_loss_pct", default=0.0035, low=0.002, high=0.005),
            TunableParam("target_pct", default=0.005, low=0.003, high=0.008),
            TunableParam("trailing_stop_pct", default=0.0025, low=0.001, high=0.004),
            TunableParam("trailing_activate_pct", default=0.0035, low=0.002, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vix_thresh = params.get("vix_change_thresh", 0.03)
        stop_pct = params.get("stop_loss_pct", 0.0035)
        target_pct = params.get("target_pct", 0.005)
        trail_pct = params.get("trailing_stop_pct", 0.0025)
        trail_act = params.get("trailing_activate_pct", 0.0035)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=20.0)
        day_id = df["day_id"].to_numpy()

        # ── VIX open per day (first bar's VIX) ──
        vix_open = np.zeros(n, dtype=np.float64)
        cur_day = day_id[0]
        cur_vix_open = vix[0]
        for i in range(n):
            if day_id[i] != cur_day:
                cur_day = day_id[i]
                cur_vix_open = vix[i]
            vix_open[i] = cur_vix_open

        # ── VIX change from open ──
        safe_vix_open = np.where(vix_open > 0.1, vix_open, 0.1)
        vix_change = (vix - vix_open) / safe_vix_open

        # ── VIX declining / rising (5-bar lookback) ──
        vix_declining = np.zeros(n, dtype=np.bool_)
        vix_rising = np.zeros(n, dtype=np.bool_)
        for i in range(5, n):
            if vix[i] < vix[i - 5]:
                vix_declining[i] = True
            if vix[i] > vix[i - 5]:
                vix_rising[i] = True

        atr14 = _compute_atr(high, low, close, 14)

        # ── Long: VIX spiked up >3% but now declining ──
        long_entry = (vix_change > vix_thresh) & vix_declining

        # ── Short: VIX dropped >3% but now rising ──
        short_entry = (vix_change < -vix_thresh) & vix_rising

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr14,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            trailing_stop_pct=trail_pct,
            trailing_activate_pct=trail_act,
            time_stop_bars=270,  # exit by 14:30 from 10:00
        )
