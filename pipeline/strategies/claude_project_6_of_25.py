"""VIX Regime Breakout v1 — claude_project_6_of_25

Thesis: When VIX spikes > 10% then pulls back > 3%, fear is subsiding.
Go long on price recovery toward VWAP. Long only.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


class Strategy(BaseStrategy):
    name = "vix_regime_breakout_v1"
    is_long_only = True
    session_start = 600   # 10:00
    session_end = 870     # 14:30
    max_trades_per_day = 2

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vix_spike_pct", default=0.10, low=0.05, high=0.20),
            TunableParam("vix_pullback_pct", default=0.03, low=0.01, high=0.06),
            TunableParam("vix_lookback", default=20.0, low=10.0, high=40.0),
            TunableParam("target_pct", default=0.005, low=0.002, high=0.010),
            TunableParam("stop_loss_pct", default=0.007, low=0.003, high=0.012),
            TunableParam("breakeven_pct", default=0.003, low=0.001, high=0.006),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vix_spike = params.get("vix_spike_pct", 0.10)
        vix_pb = params.get("vix_pullback_pct", 0.03)
        vix_lb = int(params.get("vix_lookback", 20.0))
        target_pct = params.get("target_pct", 0.005)
        stop_pct = params.get("stop_loss_pct", 0.007)
        be_pct = params.get("breakeven_pct", 0.003)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── VIX spike and pullback detection ──
        # VIX change over lookback
        vix_change = np.zeros(n, dtype=np.float64)
        vix_peak = np.zeros(n, dtype=np.float64)
        vix_from_peak = np.zeros(n, dtype=np.float64)
        for i in range(vix_lb, n):
            base_vix = vix[i - vix_lb]
            if base_vix > 0:
                vix_change[i] = (np.max(vix[i - vix_lb:i + 1]) - base_vix) / base_vix
            peak = np.max(vix[i - vix_lb:i + 1])
            vix_peak[i] = peak
            if peak > 0:
                vix_from_peak[i] = (peak - vix[i]) / peak

        # ── Price recovering: close near or above VWAP ──
        price_near_vwap = (close >= vwap * 0.995)

        # ── Session time filter ──
        # AUDIT FIX: missing session time filter caused entries outside session window
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        time_ok = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # ── Entry ──
        spike_ok = vix_change > vix_spike
        pullback_ok = vix_from_peak > vix_pb
        long_entry = spike_ok & pullback_ok & price_near_vwap & time_ok

        return StrategySignals(
            long_entry=long_entry,
            short_entry=np.zeros(n, dtype=np.bool_),
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=vwap.copy(),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            use_target_indicator=True,
            breakeven_pct=be_pct,
            time_stop_bars=180,
        )
