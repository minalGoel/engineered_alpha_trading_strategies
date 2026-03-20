"""volume_breakout_v1 — Extreme volume spike at new session high/low on NIFTY.

Mechanism:
  On NIFTY, session highs and lows are algorithmic trigger levels watched by
  institutional participants, index ETF arbitrageurs, and systematic strategies.
  When NIFTY volume on a 5-second bar exceeds 5x its 20-minute baseline while
  price simultaneously prints a new session high (or low), multiple large
  institutional programs were executed aggressively at the breakout level,
  overwhelming resting supply/demand. This breadth of participation creates
  30-120 second momentum persistence as trailing systematic strategies add to
  the confirmed breakout.

Adapted from: trading_strategies/unique_strategies_all/Strategy_178.json
Original: 1-min bars, NIFTY 200 stocks, 5x vol spike + session high/low, hold 15-45 min.
"""

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "volume_breakout_v1"
    underlying = "NIFTY"
    session_start_minutes = 570    # 09:30 IST (same as original)
    session_end_minutes = 870      # 14:30 IST
    max_trades_per_day = 6
    max_lookback = 240             # 20-minute warmup for vol SMA

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_spike_threshold", 5.0, 3.0, 8.0),
            TunableParam("close_pos_threshold", 0.70, 0.50, 0.90),
            TunableParam("stop_pts", 5.0, 3.0, 8.0),
            TunableParam("target_pts", 8.0, 5.0, 15.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Extract arrays — forward-fill NaN in Polars before .to_numpy()
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        open_ = spot_df["open"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = (
            spot_df["volume"]
            .cast(pl.Float64)
            .fill_null(0.0)
            .to_numpy()
        )
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # Parameters
        vol_spike_threshold = params.get("vol_spike_threshold", 5.0)
        close_pos_threshold = params.get("close_pos_threshold", 0.70)
        stop_pts = params.get("stop_pts", 5.0)
        target_pts = params.get("target_pts", 8.0)

        # --- Volume SMA (240 bars = 20-minute baseline, same time window as original SMA(20) on 1-min) ---
        vol_sma_window = 240
        vol_sma = np.zeros(n)
        for i in range(vol_sma_window, n):
            vol_sma[i] = np.mean(volume[i - vol_sma_window:i])

        # Volume ratio — 0 where baseline not yet established or vol_sma is zero
        vol_ratio = np.zeros(n)
        valid_sma = vol_sma > 0.0
        vol_ratio[valid_sma] = volume[valid_sma] / vol_sma[valid_sma]

        # --- Session VWAP and session high/low (reset per day) ---
        vwap = np.zeros(n)
        session_high = np.full(n, -np.inf)
        session_low = np.full(n, np.inf)

        cum_tp_vol = 0.0
        cum_vol_sum = 0.0
        sh = -np.inf
        sl = np.inf
        prev_day = -1

        for i in range(n):
            if day_id[i] != prev_day:
                # First bar of new day — reset accumulators
                cum_tp_vol = 0.0
                cum_vol_sum = 0.0
                sh = high[i]
                sl = low[i]
                prev_day = day_id[i]
            else:
                sh = max(sh, high[i])
                sl = min(sl, low[i])

            tp = (high[i] + low[i] + close[i]) / 3.0
            vol_i = volume[i]
            cum_tp_vol += tp * vol_i
            cum_vol_sum += vol_i
            vwap[i] = cum_tp_vol / cum_vol_sum if cum_vol_sum > 0.0 else close[i]
            session_high[i] = sh
            session_low[i] = sl

        # Previous bar's session high/low — new-high/new-low requires exceeding these
        prev_session_high = np.empty(n)
        prev_session_low = np.empty(n)
        prev_session_high[0] = session_high[0]
        prev_session_low[0] = session_low[0]
        prev_session_high[1:] = session_high[:-1]
        prev_session_low[1:] = session_low[:-1]

        # Bar close position in range [0=near low, 1=near high]
        bar_range = high - low
        close_pos = np.where(bar_range > 0.0, (close - low) / bar_range, 0.5)

        # Session filter
        in_session = (
            (time_min >= self.session_start_minutes) &
            (time_min < self.session_end_minutes)
        )

        # --- Bullish signal: vol spike + new session high + green bar + above VWAP + closed near high ---
        buy_ce = (
            in_session &
            (vol_ratio >= vol_spike_threshold) &
            (high > prev_session_high) &
            (close > open_) &
            (close > vwap) &
            (close_pos >= close_pos_threshold)
        )

        # --- Bearish signal: vol spike + new session low + red bar + below VWAP + closed near low ---
        buy_pe = (
            in_session &
            (vol_ratio >= vol_spike_threshold) &
            (low < prev_session_low) &
            (close < open_) &
            (close < vwap) &
            (close_pos <= (1.0 - close_pos_threshold))
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,               # 120 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
