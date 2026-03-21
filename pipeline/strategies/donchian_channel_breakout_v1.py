"""Donchian Channel Breakout — 5-second NIFTY options strategy.

Mechanism: When NIFTY closes above its 5-minute Donchian channel high with a volume
surge, systematic CTA and momentum algorithms trigger simultaneous buy signals,
creating institutional order flow that sustains the breakout for 30-90 seconds.
VWAP alignment ensures the breakout is consistent with session-level directional bias.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "donchian_channel_breakout_v1"
    underlying = "NIFTY"
    session_start_minutes = 570    # 09:30 IST — skip opening 15-min auction noise
    session_end_minutes = 920      # 15:20 IST
    max_trades_per_day = 6
    max_lookback = 120             # 10-min warmup (120 × 5s)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_mult", 1.3, 0.8, 2.0),
            TunableParam("vix_max", 24.0, 16.0, 30.0),
            TunableParam("stop_pts", 3.0, 2.0, 6.0),
            TunableParam("target_pts", 6.0, 4.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        vol_mult = params.get("vol_mult", 1.3)
        vix_max = params.get("vix_max", 24.0)
        stop_pts = params.get("stop_pts", 3.0)
        target_pts = params.get("target_pts", 6.0)

        # --- Donchian Channel (60 bars = 5 minutes) ---
        # Uses the previous 60 bars only (excludes current bar) to avoid look-ahead
        don_high = np.full(n, np.inf)   # neutral: close > inf is always False
        don_low = np.full(n, -np.inf)   # neutral: close < -inf is always False
        for i in range(60, n):
            don_high[i] = np.max(high[i - 60:i])
            don_low[i] = np.min(low[i - 60:i])

        # --- Volume SMA (60 bars = 5 minutes) ---
        vol_sma = np.full(n, np.inf)    # neutral: volume > inf is always False
        for i in range(60, n):
            vol_sma[i] = np.mean(volume[i - 60:i])

        # --- Session VWAP (cumulative from day open) ---
        vwap = np.zeros(n)
        cum_tp_vol = 0.0
        cum_vol = 0.0
        current_day = -1
        for i in range(n):
            if day_id[i] != current_day:
                current_day = day_id[i]
                cum_tp_vol = 0.0
                cum_vol = 0.0
            tp = (high[i] + low[i] + close[i]) / 3.0
            cum_tp_vol += tp * max(volume[i], 0.0)
            cum_vol += max(volume[i], 0.0)
            vwap[i] = cum_tp_vol / cum_vol if cum_vol > 0 else close[i]

        # --- VIX (joined asof to spot bars) ---
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # --- Signal masks ---
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        vol_surge = volume > (vol_sma * vol_mult)
        low_vix = vix_close < vix_max

        # Bullish: new 5-min Donchian high, volume surge, above VWAP, VIX calm
        buy_ce = in_session & (close > don_high) & vol_surge & (close > vwap) & low_vix

        # Bearish: new 5-min Donchian low, volume surge, below VWAP, VIX calm
        buy_pe = in_session & (close < don_low) & vol_surge & (close < vwap) & low_vix

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
