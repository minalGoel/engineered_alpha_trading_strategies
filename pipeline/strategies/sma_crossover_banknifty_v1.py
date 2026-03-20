"""SMA Crossover BankNifty — GPT_1_of_10

Thesis: Aligned moving averages (SMA10 > SMA29 > SMA100) indicate strong
intraday bullish momentum. Long-only strategy on liquid instruments.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


class Strategy(BaseStrategy):
    name = "sma_crossover_banknifty_v1"
    is_long_only = True
    session_start = 560   # 09:20
    session_end = 915     # 15:15
    max_trades_per_day = 1

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vix_max", default=30.0, low=15.0, high=40.0),
            TunableParam("vol_mult", default=1.5, low=1.0, high=3.0),
            TunableParam("stop_loss_pct", default=0.005, low=0.002, high=0.01),
            TunableParam("target_pct", default=0.005, low=0.002, high=0.015),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vix_max = params.get("vix_max", 30.0)
        vol_mult = params.get("vol_mult", 1.5)
        stop_pct = params.get("stop_loss_pct", 0.005)
        target_pct = params.get("target_pct", 0.005)

        n = len(df)

        # ── Indicators ──
        close = df["close"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)

        sma_10 = df["close"].rolling_mean(10).to_numpy().astype(np.float64)
        sma_29 = df["close"].rolling_mean(29).to_numpy().astype(np.float64)
        sma_100 = df["close"].rolling_mean(100).to_numpy().astype(np.float64)
        avg_vol_30 = df["volume"].rolling_mean(30).to_numpy().astype(np.float64)

        # NaN → neutral
        sma_10 = np.nan_to_num(sma_10, nan=0.0)
        sma_29 = np.nan_to_num(sma_29, nan=0.0)
        sma_100 = np.nan_to_num(sma_100, nan=0.0)
        avg_vol_30 = np.nan_to_num(avg_vol_30, nan=1.0)
        vix = np.nan_to_num(vix, nan=99.0)

        # ── Entry: close > SMA10 > SMA29 > SMA100 + volume + VIX filter ──
        alignment = (close > sma_10) & (sma_10 > sma_29) & (sma_29 > sma_100)
        vol_ok = volume > vol_mult * avg_vol_30
        vix_ok = vix < vix_max

        long_entry = alignment & vol_ok & vix_ok
        short_entry = np.zeros(n, dtype=np.bool_)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            time_stop_bars=20,
        )
