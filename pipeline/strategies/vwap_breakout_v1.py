"""
vwap_breakout_v1 — NIFTY VWAP Multi-Test Breakout

Thesis: When NIFTY tests VWAP 3+ times in a session, each test partially exhausts
opposing resting limit orders defending that level. The decisive break — accompanied
by a volume surge 2x the recent 5-min average — signals VWAP-benchmarked institutional
algorithms overwhelming the defence. Stop-loss cascades from VWAP-fade traders create
self-reinforcing continuation. Enter within 5 seconds of the surge bar.

Original: Strategy_183.json (NIFTY 200 equity, 1-min bars, 20-60 min hold)
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "vwap_breakout_v1"
    underlying = "NIFTY"
    session_start_minutes = 600    # 10:00 IST — need ~45 min for 3 VWAP tests to accumulate
    session_end_minutes = 870      # 14:30 IST
    max_trades_per_day = 3
    max_lookback = 720             # 60 min warmup — need time for 3+ VWAP tests

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("touch_threshold_pct", 0.0005, 0.0002, 0.0015),  # 0.05% default
            TunableParam("break_threshold_pct", 0.001, 0.0005, 0.002),    # 0.10% default
            TunableParam("volume_ratio_thresh", 2.0, 1.5, 3.5),
            TunableParam("min_touch_count", 3.0, 2.0, 5.0),
            TunableParam("side_score_thresh", 0.3, 0.1, 0.6),
            TunableParam("stop_pts", 5.0, 3.0, 8.0),
            TunableParam("target_pts", 9.0, 6.0, 15.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # --- Extract arrays (forward-fill NaN in Polars before numpy) ---
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # --- Parameters ---
        touch_thr = params.get("touch_threshold_pct", 0.0005)
        break_thr = params.get("break_threshold_pct", 0.001)
        vol_ratio_thr = params.get("volume_ratio_thresh", 2.0)
        min_touches = int(params.get("min_touch_count", 3.0))
        side_thr = params.get("side_score_thresh", 0.3)
        stop_pts = params.get("stop_pts", 5.0)
        target_pts = params.get("target_pts", 9.0)

        # --- VWAP (cumulative from session open, reset per day) ---
        typical_price = (high + low + close) / 3.0
        vwap = np.zeros(n)
        cum_tp_vol = 0.0
        cum_vol = 0.0
        cur_day = -1

        for i in range(n):
            if day_id[i] != cur_day:
                # New day: reset accumulators
                cur_day = day_id[i]
                cum_tp_vol = 0.0
                cum_vol = 0.0
            cum_tp_vol += typical_price[i] * volume[i]
            cum_vol += volume[i]
            vwap[i] = cum_tp_vol / cum_vol if cum_vol > 0 else typical_price[i]

        # --- VWAP touch episode count (distinct tests per day) ---
        # A touch = |close - vwap| / vwap < touch_thr; consecutive touching bars = 1 episode
        in_touch = np.zeros(n, dtype=bool)
        for i in range(n):
            if vwap[i] > 0:
                in_touch[i] = abs(close[i] - vwap[i]) / vwap[i] < touch_thr

        touch_count = np.zeros(n, dtype=np.int32)
        ep_count = 0
        prev_touch = False
        prev_day = -1

        for i in range(n):
            if day_id[i] != prev_day:
                # New day: reset episode counter
                prev_day = day_id[i]
                ep_count = 0
                prev_touch = False
            if in_touch[i] and not prev_touch:
                ep_count += 1  # new touch episode begins
            touch_count[i] = ep_count
            prev_touch = in_touch[i]

        # --- Rolling 5-min volume SMA (60 bars) ---
        vol_sma = np.zeros(n)
        for i in range(1, n):
            start = max(0, i - 60)
            vol_sma[i] = np.mean(volume[start:i])
        # vol_sma[0] = 0 (no data before first bar)

        volume_ratio = np.zeros(n)
        for i in range(n):
            if vol_sma[i] > 0:
                volume_ratio[i] = volume[i] / vol_sma[i]

        # --- Side-before-break: mean(sign(close - vwap)) over last 40 bars ---
        # Positive = predominantly above VWAP, negative = predominantly below
        side_score = np.zeros(n)
        for i in range(1, n):
            look = min(40, i)
            w_start = i - look
            signs = np.sign(close[w_start:i] - vwap[w_start:i])
            side_score[i] = np.mean(signs) if look > 0 else 0.0

        # --- VIX filter ---
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # --- Build signal masks ---
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # Breakout conditions
        above_vwap = close > vwap * (1.0 + break_thr)   # decisive break above
        below_vwap = close < vwap * (1.0 - break_thr)   # decisive break below

        vol_surge = volume_ratio > vol_ratio_thr
        enough_touches = touch_count >= min_touches
        vix_ok = vix_close < 25.0

        # Price was predominantly below VWAP before break → bullish CE
        was_below = side_score < -side_thr
        # Price was predominantly above VWAP before break → bearish PE
        was_above = side_score > side_thr

        buy_ce = in_session & above_vwap & enough_touches & was_below & vol_surge & vix_ok
        buy_pe = in_session & below_vwap & enough_touches & was_above & vol_surge & vix_ok

        # No simultaneous signals
        buy_ce = buy_ce & ~buy_pe

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,           # 120 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
