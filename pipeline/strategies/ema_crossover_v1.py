"""EMA Crossover — GPT_7_of_10

Thesis: EMA9/EMA21 crossover captures short-term momentum shifts.
Long when close > EMA9 > EMA21, short when close < EMA9 < EMA21.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


class Strategy(BaseStrategy):
    name = "ema_crossover_v1"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 915     # 15:15
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("stop_loss_pct", default=0.003, low=0.001, high=0.008),
            TunableParam("target_pct", default=0.005, low=0.002, high=0.015),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        stop_pct = params.get("stop_loss_pct", 0.003)
        target_pct = params.get("target_pct", 0.005)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)

        # ── EMA 9 and 21 ──
        ema9 = df["close"].ewm_mean(span=9).to_numpy().astype(np.float64)
        ema21 = df["close"].ewm_mean(span=21).to_numpy().astype(np.float64)
        ema9 = np.nan_to_num(ema9, nan=0.0)
        ema21 = np.nan_to_num(ema21, nan=0.0)

        # ── Volume filter ──
        avg_vol_10 = df["volume"].rolling_mean(10).to_numpy().astype(np.float64)
        avg_vol_10 = np.nan_to_num(avg_vol_10, nan=1.0)
        avg_vol_10 = np.clip(avg_vol_10, 1.0, None)
        vol_ok = volume > avg_vol_10

        # ── Entry ──
        long_entry = (close > ema9) & (ema9 > ema21) & vol_ok
        short_entry = (close < ema9) & (ema9 < ema21) & vol_ok

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            time_stop_bars=15,
        )
