"""Price Channel (Donchian) Breakout — NIFTY 5-second index options.

Thesis: When NIFTY closes above its 10-minute Donchian upper band with a volume
surge and close above VWAP, institutional TWAP/VWAP algorithms have overcome a
meaningful intraday resistance level. Stop-loss orders from short participants
clustered above the channel high are triggered, accelerating the initial push.
The strong close (top 30% of bar range) confirms the breakout is demand-driven,
making a 10-20 spot point continuation over 30-90 seconds structurally likely.
Symmetric setup applies for lower-band breaks.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view


class Strategy(BaseStrategy):
    name = "price_channel_breakout_v1"
    underlying = "NIFTY"
    session_start_minutes = 570    # 09:30 IST — need 10-min DC warmup after open
    session_end_minutes = 920      # 15:20 IST
    max_lookback = 150             # 125 bars (>10 min) + buffer
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_ratio_threshold", 1.3, 0.8, 2.5),
            TunableParam("dc_width_min_pct", 0.15, 0.05, 0.40),
            TunableParam("close_position_threshold", 0.70, 0.50, 0.90),
            TunableParam("stop_pts", 5.0, 2.0, 8.0),
            TunableParam("target_pts", 8.0, 4.0, 14.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # --- Extract and forward-fill arrays ---
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(float)
        high = spot_df["high"].fill_null(strategy="forward").to_numpy().astype(float)
        low = spot_df["low"].fill_null(strategy="forward").to_numpy().astype(float)
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # --- Parameters ---
        vol_ratio_threshold = params.get("vol_ratio_threshold", 1.3)
        dc_width_min_pct = params.get("dc_width_min_pct", 0.15)
        close_pos_thresh = params.get("close_position_threshold", 0.70)
        stop_pts = params.get("stop_pts", 5.0)
        target_pts = params.get("target_pts", 8.0)

        # --- Donchian Channel (10 minutes = 120 bars at 5s) ---
        # Uses the prior bar's channel to avoid look-ahead: at bar i, compare
        # close[i] against the highest/lowest of bars [i-120 .. i-1].
        dc_period = 120

        dc_upper = np.full(n, np.nan)
        dc_lower = np.full(n, np.nan)

        if n > dc_period:
            # sliding_window_view over high[0..n-1], window size dc_period
            # windows_h[k] = high[k .. k+dc_period-1]
            # We want dc_upper[i] = max(high[i-dc_period .. i-1])
            # = windows_h[i-dc_period] at index i-dc_period, but we want bar i-1 as endpoint
            # Shift by 1: dc_upper[i] = max(high[i-dc_period:i]) requires ending at i (exclusive)
            # Use roll: compute rolling max over high, assign to offset position
            pad = dc_period  # first valid at index dc_period
            if n > pad:
                win_h = sliding_window_view(high, dc_period)   # shape: (n - dc_period + 1, dc_period)
                win_l = sliding_window_view(low, dc_period)

                dc_upper_valid = np.max(win_h, axis=1)   # length: n - dc_period + 1
                dc_lower_valid = np.min(win_l, axis=1)

                # dc_upper_valid[k] = max(high[k..k+dc_period-1])
                # We want dc_upper[i] = max(high[i-dc_period..i-1]) = dc_upper_valid[i-dc_period]
                # valid for i in [dc_period, n]
                for i in range(dc_period, n):
                    dc_upper[i] = dc_upper_valid[i - dc_period]
                    dc_lower[i] = dc_lower_valid[i - dc_period]

        # Fill NaN at beginning with close price (neutral — won't fire due to session filter)
        dc_upper = np.where(np.isnan(dc_upper), close, dc_upper)
        dc_lower = np.where(np.isnan(dc_lower), close, dc_lower)

        # DC width as % of close
        dc_width_pct = np.where(close > 0, (dc_upper - dc_lower) / close * 100.0, 0.0)

        # --- Volume ratio: volume / SMA(volume, 20 bars = 100s) ---
        vol_sma_period = 20
        vol_sma = np.full(n, np.nan)
        if n > vol_sma_period:
            win_v = sliding_window_view(volume, vol_sma_period)
            vol_sma_valid = np.mean(win_v, axis=1)
            # vol_sma[i] = mean(volume[i-vol_sma_period+1 .. i]) → index i = vol_sma_period-1 onward
            start = vol_sma_period - 1
            vol_sma[start: start + len(vol_sma_valid)] = vol_sma_valid

        # Fill leading NaN and zeros with 1.0 to avoid division issues
        vol_sma[:vol_sma_period] = np.nanmean(volume[:vol_sma_period]) if vol_sma_period <= n else 1.0
        vol_sma = np.where(np.isnan(vol_sma) | (vol_sma <= 0), 1.0, vol_sma)

        volume_ratio = volume / vol_sma

        # --- Session VWAP (cumulative from each day's open) ---
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
            v = volume[i]
            cum_tp_vol += typical_price[i] * v
            cum_vol += v
            vwap[i] = cum_tp_vol / cum_vol if cum_vol > 0 else close[i]

        # --- Bar close-position ratio: 0=at low, 1=at high ---
        bar_range = high - low
        bar_range = np.where(bar_range <= 0, 1.0, bar_range)
        close_position = (close - low) / bar_range

        # --- Session filter ---
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # --- Entry signals ---
        # buy_ce: NIFTY breaks above 10-min Donchian upper
        buy_ce = (
            in_session
            & (close > dc_upper)          # close breaks above the prior 10-min channel high
            & (close > vwap)              # above VWAP — institutional demand alignment
            & (volume_ratio > vol_ratio_threshold)
            & (dc_width_pct > dc_width_min_pct)
            & (close_position > close_pos_thresh)  # strong close — top 30% of bar
        )

        # buy_pe: NIFTY breaks below 10-min Donchian lower
        buy_pe = (
            in_session
            & (close < dc_lower)          # close breaks below the prior 10-min channel low
            & (close < vwap)              # below VWAP — institutional supply alignment
            & (volume_ratio > vol_ratio_threshold)
            & (dc_width_pct > dc_width_min_pct)
            & (close_position < (1.0 - close_pos_thresh))  # weak close — bottom 30% of bar
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
