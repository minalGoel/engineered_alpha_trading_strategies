"""quality_momentum_v1 — EMA Acceleration + ROC + VWAP Momentum Strategy

Converted from: trading_strategies/unique_strategies_all/Strategy_294.json
Original: Quality-filtered equity momentum (EMA crossover + ROC + VWAP on NIFTY100 stocks)

Adaptation: Strips equity-specific quality filter (ROE, D/E, earnings).
Preserves the core signal: EMA crossover with WIDENING GAP confirmation + 5-min ROC + VWAP.
The gap acceleration (ema_gap growing over 30s) is the key differentiator — it identifies
live institutional TWAP programs on NIFTY rather than decaying or random crossovers.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "quality_momentum_v1"
    underlying = "NIFTY"
    session_start_minutes = 570    # 09:30 IST — allow EMA-120 (10 min) warmup after open
    session_end_minutes = 920      # 15:20 IST
    max_trades_per_day = 8
    max_lookback = 240             # 20 min warmup (240 × 5s) for slow EMA to stabilise

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("roc_threshold", 0.0015, 0.0005, 0.004),  # 5-min ROC threshold (15 bps default)
            TunableParam("ema_gap_min", 0.5, 0.1, 3.0),            # minimum EMA gap in index points
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── Raw arrays ──────────────────────────────────────────────────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        roc_threshold = params.get("roc_threshold", 0.0015)
        ema_gap_min = params.get("ema_gap_min", 0.5)

        # ── EMA(36) — 3-minute fast EMA ─────────────────────────────────────────
        # 3-min horizon: trade trigger for fast momentum (holds 30-90s, not 20-60 min).
        # Original EMA(9) on 1-min bars compressed proportionally to hold-time ratio.
        alpha_fast = 2.0 / (36 + 1)
        ema_fast = np.empty(n)
        ema_fast[0] = close[0]
        for i in range(1, n):
            ema_fast[i] = close[i] * alpha_fast + ema_fast[i - 1] * (1.0 - alpha_fast)

        # ── EMA(120) — 10-minute slow EMA ───────────────────────────────────────
        # 10-min horizon: context filter for dominant institutional trend direction.
        # Original EMA(21) on 1-min compressed to 10-min (120 bars × 5s).
        alpha_slow = 2.0 / (120 + 1)
        ema_slow = np.empty(n)
        ema_slow[0] = close[0]
        for i in range(1, n):
            ema_slow[i] = close[i] * alpha_slow + ema_slow[i - 1] * (1.0 - alpha_slow)

        # ── EMA gap and 30-second acceleration ──────────────────────────────────
        # ema_gap > 0: 3-min EMA above 10-min EMA = bullish alignment
        # gap_accel_bull: gap is still widening (institutional buy program active)
        # gap_accel_bear: gap is still narrowing bearishly (sell program active)
        ema_gap = ema_fast - ema_slow

        gap_accel_bull = np.zeros(n, dtype=bool)   # ema_gap growing more positive
        gap_accel_bear = np.zeros(n, dtype=bool)   # ema_gap growing more negative
        for i in range(6, n):
            gap_accel_bull[i] = ema_gap[i] > ema_gap[i - 6]
            gap_accel_bear[i] = ema_gap[i] < ema_gap[i - 6]

        # ── ROC(60) — 5-minute rate of change ───────────────────────────────────
        # Confirms macro follow-through. Original: ROC(30) on 1-min = 30-min ROC.
        # Compressed to 5-min (60 bars) to match ~10x compression in hold time.
        roc_60 = np.zeros(n)
        for i in range(60, n):
            base = close[i - 60]
            if base > 0.0:
                roc_60[i] = (close[i] - base) / base

        # ── Session VWAP (daily reset) ───────────────────────────────────────────
        # Cumulative: no scaling needed. Reset on each new trading day.
        vwap = np.empty(n)
        cum_pv = 0.0
        cum_v = 0.0
        prev_day = -1
        for i in range(n):
            if day_id[i] != prev_day:
                cum_pv = 0.0
                cum_v = 0.0
                prev_day = day_id[i]
            cum_pv += close[i] * float(volume[i])
            cum_v += float(volume[i])
            vwap[i] = (cum_pv / cum_v) if cum_v > 0.0 else close[i]

        # ── VIX filter ───────────────────────────────────────────────────────────
        # Above VIX 25, EMA gap acceleration signals fail more frequently due to choppy
        # institutional flows; skip all entries in high-volatility regime.
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(
                    ["datetime", pl.col("close").alias("vix_close")]
                ).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Signal masks ─────────────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        vix_ok = vix_close < 25.0

        # BUY CE: 3-min EMA above 10-min EMA, gap accelerating bullishly over 30s,
        # 5-min positive ROC, price above VWAP, low-VIX regime
        buy_ce = (
            in_session
            & vix_ok
            & (ema_gap > ema_gap_min)
            & gap_accel_bull
            & (roc_60 > roc_threshold)
            & (close > vwap)
        )

        # BUY PE: mirror conditions for bearish alignment
        buy_pe = (
            in_session
            & vix_ok
            & (ema_gap < -ema_gap_min)
            & gap_accel_bear
            & (roc_60 < -roc_threshold)
            & (close < vwap)
        )

        stop_pts = 3.0   # ~6 NIFTY spot pts at delta 0.5 — outside 30s noise range
        target_pts = 5.0  # ~10 NIFTY spot pts — lower half of typical 1-min extension

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
