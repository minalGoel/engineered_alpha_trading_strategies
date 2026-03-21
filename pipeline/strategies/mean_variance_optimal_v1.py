"""mean_variance_optimal_v1 — Instantaneous Sharpe Timing on NIFTY

Mechanism: On NIFTY, when the rolling Sharpe ratio (1-min EMA of 5s returns /
3-min realized volatility) ranks in the top 15% of its 20-minute history, it
signals coherent institutional directional order flow — a 'coiled spring'
state preceding 30-60 second continuation. VWAP alignment confirms the flow
is not a post-spike overextension.

Converted from: trading_strategies/unique_strategies_all/Strategy_297.json
Original: Markowitz Sharpe-timing on NIFTY 50 constituent stocks, 1-min bars.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "mean_variance_optimal_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — skip first 15 min for variance warmup
    session_end_minutes = 920     # 15:20 IST
    max_lookback = 240            # 20 min warmup for Sharpe percentile (240 × 5s)
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("sharpe_threshold", 2.0, 1.0, 3.5),
            TunableParam("sharpe_pct_thresh", 0.85, 0.70, 0.95),
            TunableParam("stop_pts", 3.0, 2.0, 6.0),
            TunableParam("target_pts", 5.0, 3.0, 10.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── Extract spot arrays (forward-fill NaN before numpy) ──────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        sharpe_threshold = params.get("sharpe_threshold", 2.0)
        sharpe_pct_thresh = params.get("sharpe_pct_thresh", 0.85)
        stop_pts = params.get("stop_pts", 3.0)
        target_pts = params.get("target_pts", 5.0)

        # ── 5-second log returns ──────────────────────────────────────────────
        ret = np.zeros(n)
        for i in range(1, n):
            if close[i - 1] > 0:
                ret[i] = np.log(close[i] / close[i - 1])

        # ── EMA of returns over 12 bars (1 minute) ────────────────────────────
        # Smoothed expected return — numerator of instantaneous Sharpe
        ema_ret = np.zeros(n)
        alpha = 2.0 / (12 + 1)
        ema_ret[0] = ret[0]
        for i in range(1, n):
            ema_ret[i] = alpha * ret[i] + (1.0 - alpha) * ema_ret[i - 1]

        # ── Rolling variance of returns over 36 bars (3 minutes) ─────────────
        roll_var = np.zeros(n)
        for i in range(36, n):
            window = ret[i - 36:i]
            roll_var[i] = np.var(window)

        # ── Instantaneous Sharpe = ema_ret / sqrt(roll_var) ──────────────────
        instant_sharpe = np.zeros(n)
        for i in range(36, n):
            if roll_var[i] > 1e-14:
                instant_sharpe[i] = ema_ret[i] / np.sqrt(roll_var[i])

        # ── Sharpe percentile over last 240 bars (20 minutes) ────────────────
        sharpe_pct = np.zeros(n)
        for i in range(240, n):
            window = instant_sharpe[i - 240:i]
            val = instant_sharpe[i]
            # percentile rank: fraction of window values strictly below current
            sharpe_pct[i] = np.sum(window < val) / 240.0

        # ── Session-cumulative VWAP (resets each day) ─────────────────────────
        typical_price = (high + low + close) / 3.0
        vwap = np.zeros(n)
        cum_tp_vol = 0.0
        cum_vol = 0.0
        prev_day = -1
        for i in range(n):
            if day_id[i] != prev_day:
                cum_tp_vol = 0.0
                cum_vol = 0.0
                prev_day = day_id[i]
            cum_tp_vol += typical_price[i] * volume[i]
            cum_vol += volume[i]
            vwap[i] = cum_tp_vol / cum_vol if cum_vol > 0 else close[i]

        # ── Session filter ────────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes) &
            (time_min < self.session_end_minutes)
        )

        # ── Entry signals ─────────────────────────────────────────────────────
        # Buy CE: high positive Sharpe at session-relative top + above VWAP
        buy_ce = (
            in_session &
            (instant_sharpe > sharpe_threshold) &
            (sharpe_pct > sharpe_pct_thresh) &
            (close > vwap) &
            (roll_var > 1e-14)
        )

        # Buy PE: deep negative Sharpe at session-relative bottom + below VWAP
        buy_pe = (
            in_session &
            (instant_sharpe < -sharpe_threshold) &
            (sharpe_pct < (1.0 - sharpe_pct_thresh)) &
            (close < vwap) &
            (roll_var > 1e-14)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=12,        # 60 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
