"""Support/Resistance Reversal — Grok_10_of_10

Thesis: Reversals at 2+ day support/resistance levels with higher-low/lower-high
pattern formation. Uses volume confirmation.

Assumptions:
- "S/R level" = price level where close was within 0.3% on 2+ previous days
- "hl_reversal" at support = higher low forming (current low > previous low)
- "hl_reversal" at resistance = lower high forming (current high < previous high)
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_atr(high, low, close, period):
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    if n < period:
        return atr
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr


class Strategy(BaseStrategy):
    name = "support_resistance_reversal_v1"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 915     # 15:15
    max_trades_per_day = 6
    assumptions = ["S/R level = close within 0.3% on 2+ prior days",
                   "hl_reversal = higher low at support / lower high at resistance"]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vix_max", default=20.0, low=12.0, high=30.0),
            TunableParam("rel_vol_thresh", default=1.5, low=1.0, high=3.0),
            TunableParam("sr_pct", default=0.003, low=0.001, high=0.008),
            TunableParam("stop_atr_mult", default=0.8, low=0.3, high=2.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vix_max = params.get("vix_max", 20.0)
        rv_thresh = params.get("rel_vol_thresh", 1.5)
        sr_pct = params.get("sr_pct", 0.003)
        stop_atr = params.get("stop_atr_mult", 0.8)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)
        day_ids = df["day_id"].to_numpy()

        atr20 = _compute_atr(high, low, close, 20)

        # Relative volume
        avg_vol_20 = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol_20 = np.nan_to_num(avg_vol_20, nan=1.0)
        avg_vol_20 = np.clip(avg_vol_20, 1.0, None)
        rel_vol = volume / avg_vol_20

        # Compute daily close levels and find S/R
        unique_days = np.unique(day_ids)
        daily_closes = {}
        daily_highs = {}
        daily_lows = {}
        for d in unique_days:
            day_idx = np.where(day_ids == d)[0]
            daily_closes[d] = close[day_idx[-1]]
            daily_highs[d] = np.max(high[day_idx])
            daily_lows[d] = np.min(low[day_idx])

        # For each day, check if current close is near a level seen 2+ prior days
        near_support = np.zeros(n, dtype=np.bool_)
        near_resistance = np.zeros(n, dtype=np.bool_)
        higher_low = np.zeros(n, dtype=np.bool_)
        lower_high = np.zeros(n, dtype=np.bool_)

        for i, d in enumerate(unique_days):
            day_idx = np.where(day_ids == d)[0]
            prev_days = unique_days[max(0, i-5):i]
            if len(prev_days) < 2:
                continue

            prev_lows = [daily_lows[pd] for pd in prev_days]
            prev_highs = [daily_highs[pd] for pd in prev_days]

            # Support: count how many prior days had low near current low
            for bar in day_idx:
                sup_count = sum(1 for pl_ in prev_lows if abs(pl_ - low[bar]) < sr_pct * close[bar])
                res_count = sum(1 for ph in prev_highs if abs(ph - high[bar]) < sr_pct * close[bar])
                if sup_count >= 2:
                    near_support[bar] = True
                if res_count >= 2:
                    near_resistance[bar] = True

            # Higher low / lower high within this day
            if i > 0:
                prev_d_low = daily_lows[unique_days[i-1]]
                prev_d_high = daily_highs[unique_days[i-1]]
                cur_low = daily_lows[d]
                cur_high = daily_highs[d]
                if cur_low > prev_d_low:
                    higher_low[day_idx] = True
                if cur_high < prev_d_high:
                    lower_high[day_idx] = True

        # Filters
        vix_ok = vix < vix_max
        vol_ok = rel_vol > rv_thresh

        # Entry
        long_entry = near_support & higher_low & vix_ok & vol_ok
        short_entry = near_resistance & lower_high & vix_ok & vol_ok

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr20,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_atr_mult=stop_atr,
            target_pct=0.008,
            time_stop_bars=40,
        )
