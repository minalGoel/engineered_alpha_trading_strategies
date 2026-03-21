"""gap_reversal_volume_v1 — Gap Reversal with Volume Confirmation

Converted from: trading_strategies/unique_strategies_all/Strategy_203.json
Original: 1-min equity strategy on NIFTY 50 stocks using tick-level buy/sell volume
          classification to confirm overnight gap reversals. Hold 30-120 min.

Adaptation: Trades NIFTY index options. Replaces tick-level buy/sell classification
with 5-second bar pressure proxy (close vs open direction). Enters after 1-minute
volume confirmation window (12 bars). Targets the initial reversal thrust (30-120s),
not the full gap fill.

Differentiated from existing gap strategies:
- gap_fade_reversion_v103: price-only, first 5 bars (09:15-09:35)
- gap_fade_large_v1: RSI extremes + 30s momentum stall
- gap_reversal_volume_v1 (this): volume microstructure confirmation, enters later
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


def _rolling_sum_12(arr: np.ndarray) -> np.ndarray:
    """Fast rolling sum over 12 bars using cumsum trick."""
    window = 12
    result = np.zeros(len(arr))
    cs = np.cumsum(arr)
    result[window:] = cs[window:] - cs[:-window]
    result[:window] = cs[:window]
    return result


class Strategy(BaseStrategy):
    name = "gap_reversal_volume_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST — need 5 min of opening data
    session_end_minutes = 600     # 10:00 IST — gap reversal is opening-hour only
    max_lookback = 144            # 12 min warmup (prev day close + signal window)
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_threshold_pct", 0.003, 0.002, 0.010),
            TunableParam("pressure_threshold", 0.15, 0.05, 0.40),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 8.0, 4.0, 14.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── Extract arrays (forward-fill NaN in Polars before numpy) ──
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        open_ = spot_df["open"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        gap_threshold = params.get("gap_threshold_pct", 0.003)
        pressure_threshold = params.get("pressure_threshold", 0.15)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 8.0)

        GAP_MAX = 0.015  # ignore extreme gaps >1.5% (news-driven)
        SIGNAL_WINDOW = 12  # 1 minute of 5-second bars

        # ── Session filter ──
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── VIX filter ──
        vix_ok = np.ones(n, dtype=bool)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(
                    ["datetime", pl.col("close").alias("vix_close")]
                ).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_vals = vix_joined["vix_close"].fill_null(15.0).to_numpy()
            vix_ok = vix_vals < 20.0

        # ── Compute day-level reference prices ──
        # day_last_close[d] = last close of day d (built in forward pass)
        day_last_close: dict[int, float] = {}
        for i in range(n):
            day_last_close[int(day_id[i])] = close[i]

        # day_first_open[d] = first open of day d
        day_first_open: dict[int, float] = {}
        for i in range(n):
            d = int(day_id[i])
            if d not in day_first_open:
                day_first_open[d] = open_[i]

        # For each bar: gap_pct = (day_first_open - prev_day_last_close) / prev_day_last_close
        gap_pct = np.zeros(n)
        for i in range(n):
            d = int(day_id[i])
            prev_close = day_last_close.get(d - 1, 0.0)
            if prev_close > 0:
                day_open = day_first_open.get(d, close[i])
                gap_pct[i] = (day_open - prev_close) / prev_close
            # else: 0 (first day or unknown prev close → no gap trade)

        # ── Gap flags (constant per bar within a day) ──
        gap_down_day = gap_pct < -gap_threshold        # gapped down >threshold
        gap_up_day = gap_pct > gap_threshold            # gapped up >threshold
        gap_not_extreme = np.abs(gap_pct) < GAP_MAX    # not a structural news gap

        # ── Bar pressure: proxy for buy/sell imbalance ──
        # Each 5s bar: close > open → buying pressure (+1), close < open → selling (-1)
        # Normalize by typical bar range to avoid close == open giving 0 unfairly
        bar_dir = close - open_
        bar_abs = np.maximum(np.abs(bar_dir), 0.01)
        bar_sign = np.clip(bar_dir / bar_abs, -1.0, 1.0)  # ~[-1, 1]

        pressure_sum = _rolling_sum_12(bar_sign)
        pressure_norm = pressure_sum / SIGNAL_WINDOW  # [-1, 1]

        # ── Session OBV — reset at day start ──
        obv = np.zeros(n)
        for i in range(1, n):
            if day_id[i] != day_id[i - 1]:
                obv[i] = 0.0
            elif close[i] > close[i - 1]:
                obv[i] = obv[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                obv[i] = obv[i - 1] - volume[i]
            else:
                obv[i] = obv[i - 1]

        # OBV slope over 12 bars (only within same day)
        obv_slope = np.zeros(n)
        for i in range(SIGNAL_WINDOW, n):
            if day_id[i] == day_id[i - SIGNAL_WINDOW]:
                denom = abs(obv[i - SIGNAL_WINDOW]) + 1.0
                obv_slope[i] = (obv[i] - obv[i - SIGNAL_WINDOW]) / denom

        # ── Session VWAP — reset at day start ──
        cum_tp_vol = np.zeros(n)
        cum_vol_sum = np.zeros(n)
        for i in range(n):
            tp = (open_[i] + close[i]) * 0.5
            if i == 0 or day_id[i] != day_id[i - 1]:
                cum_tp_vol[i] = tp * volume[i]
                cum_vol_sum[i] = volume[i]
            else:
                cum_tp_vol[i] = cum_tp_vol[i - 1] + tp * volume[i]
                cum_vol_sum[i] = cum_vol_sum[i - 1] + volume[i]

        vwap = np.where(cum_vol_sum > 0, cum_tp_vol / cum_vol_sum, close)
        vwap_dev = (close - vwap) / np.maximum(vwap, 1.0)

        # ── Bars since session open per day ──
        bars_since_open = np.zeros(n, dtype=int)
        for i in range(n):
            if i == 0 or day_id[i] != day_id[i - 1]:
                bars_since_open[i] = 0
            else:
                bars_since_open[i] = bars_since_open[i - 1] + 1

        has_signal_window = bars_since_open >= SIGNAL_WINDOW

        # ── Entry signals ──
        # BUY CE: gap-down day + buying pressure confirms reversal
        buy_ce = (
            in_session
            & has_signal_window
            & vix_ok
            & gap_down_day
            & gap_not_extreme
            & (pressure_norm > pressure_threshold)   # net buying in last 1 min
            & (obv_slope > 0)                        # OBV rising (accumulation)
            & (vwap_dev < 0)                         # price still below VWAP
        )

        # BUY PE: gap-up day + selling pressure confirms reversal
        buy_pe = (
            in_session
            & has_signal_window
            & vix_ok
            & gap_up_day
            & gap_not_extreme
            & (pressure_norm < -pressure_threshold)  # net selling in last 1 min
            & (obv_slope < 0)                        # OBV falling (distribution)
            & (vwap_dev > 0)                         # price still above VWAP
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,
            max_trades_per_day=self.max_trades_per_day,
        )
