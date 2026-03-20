"""index_correlated_breakout_v1 — Running session high/low breakout with multi-confirmation.

Adapted from: trading_strategies/unique_strategies_all/Strategy_186.json
Original: Index-correlated dual breakout (stock + NIFTY simultaneously).

Conversion: Trade NIFTY index directly. The "dual confirmation" from the original
(stock AND index both breaking session high/low) is replaced by three independent
confirmations on NIFTY itself: VWAP alignment, 1-min momentum buildup, and volume
surge. Fires throughout the day (09:30-14:30), not just at the open.

Mechanism: When NIFTY closes above its running intraday session high, institutional
VWAP-benchmarked algorithms flip from passive accumulation to active momentum mode,
typically extending 15-25 spot pts in the next 30-90 seconds. VWAP alignment and
1-min momentum confirm sustained institutional push vs a retail spike.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "index_correlated_breakout_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — skip first 15 min of opening noise
    session_end_minutes = 870     # 14:30 IST — stop entries; 120s max hold exits by 14:32
    max_trades_per_day = 8
    max_lookback = 240            # 20 min warmup for VWAP and volume baseline

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("momentum_threshold", 0.0005, 0.0001, 0.002),
            TunableParam("volume_ratio_threshold", 1.3, 1.0, 2.5),
            TunableParam("stop_pts", 5.0, 3.0, 9.0),
            TunableParam("target_pts", 8.0, 5.0, 15.0),
            TunableParam("vix_max", 22.0, 16.0, 28.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill price columns in Polars before converting to numpy
        close = spot_df["close"].fill_null(strategy="forward").fill_null(0.0).to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").fill_null(0.0).to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").fill_null(0.0).to_numpy()
        volume = spot_df["volume"].cast(pl.Float64).fill_null(0.0).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        momentum_threshold = params.get("momentum_threshold", 0.0005)
        volume_ratio_threshold = params.get("volume_ratio_threshold", 1.3)
        stop_pts = params.get("stop_pts", 5.0)
        target_pts = params.get("target_pts", 8.0)
        vix_max = params.get("vix_max", 22.0)

        # ── VIX filter (aligned to spot bars) ────────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Running session high/low and VWAP (reset each day) ───────────────
        session_high = np.zeros(n)
        session_low = np.zeros(n)
        vwap = np.zeros(n)

        cur_day = -1
        s_high = -np.inf
        s_low = np.inf
        cum_tp_vol = 0.0
        cum_vol = 0.0

        for i in range(n):
            d = int(day_id[i])
            if d != cur_day:
                cur_day = d
                s_high = -np.inf
                s_low = np.inf
                cum_tp_vol = 0.0
                cum_vol = 0.0

            if high[i] > s_high:
                s_high = high[i]
            if low[i] < s_low:
                s_low = low[i]
            session_high[i] = s_high
            session_low[i] = s_low

            tp = (high[i] + low[i] + close[i]) / 3.0
            v = volume[i]
            cum_tp_vol += tp * v
            cum_vol += v
            vwap[i] = cum_tp_vol / cum_vol if cum_vol > 0.0 else close[i]

        # ── Previous-bar session high/low (within the same day) ──────────────
        # At day boundaries, use inf/0 to prevent cross-day false triggers.
        prev_session_high = np.full(n, np.inf)
        prev_session_low = np.zeros(n)
        for i in range(1, n):
            if int(day_id[i]) == int(day_id[i - 1]):
                prev_session_high[i] = session_high[i - 1]
                prev_session_low[i] = session_low[i - 1]
            # else: first bar of new day — prev values stay inf/0 → no trigger

        # ── 10-min rolling volume baseline (120 bars) ─────────────────────────
        vol_avg_120 = np.zeros(n)
        for i in range(n):
            start = max(0, i - 120)
            window = volume[start:i]
            vol_avg_120[i] = np.mean(window) if len(window) > 0 else 1.0
        vol_avg_120 = np.where(vol_avg_120 > 0, vol_avg_120, 1.0)

        # ── 1-minute momentum (12 bars × 5s = 60s) ───────────────────────────
        mom_12 = np.zeros(n)
        for i in range(12, n):
            if close[i - 12] > 0:
                mom_12[i] = (close[i] - close[i - 12]) / close[i - 12]

        # ── Filters ──────────────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed_up = np.arange(n) >= self.max_lookback

        vol_ratio = volume / vol_avg_120
        vix_ok = vix_close < vix_max

        # ── Signals ──────────────────────────────────────────────────────────
        # BUY CE: new intraday session high with VWAP + momentum + volume confirmation
        buy_ce = (
            in_session
            & warmed_up
            & (close > prev_session_high)
            & (close > vwap)
            & (mom_12 > momentum_threshold)
            & (vol_ratio > volume_ratio_threshold)
            & vix_ok
        )

        # BUY PE: new intraday session low with VWAP + momentum + volume confirmation
        buy_pe = (
            in_session
            & warmed_up
            & (close < prev_session_low)
            & (close < vwap)
            & (mom_12 < -momentum_threshold)
            & (vol_ratio > volume_ratio_threshold)
            & vix_ok
        )

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
