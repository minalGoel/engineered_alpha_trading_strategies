"""bid_ask_imbalance_v1 — NIFTY Order Flow Imbalance (OHLCV Proxy)

Approximates bid-ask imbalance using bar-level OHLCV microstructure.
bar_imbalance = (close - low) / (high - low) - 0.5 captures net buy/sell
pressure per 5-second bar. When the 1-minute EMA of this proxy is unusually
one-sided (z-score > threshold), institutional directional orders are likely
still active, predicting NIFTY continuation over the next 30-90 seconds.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "bid_ask_imbalance_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 12
    max_lookback = 120            # 10-min warmup (need 60-bar z-score + ema warmup)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("imbalance_zscore_threshold", 1.5, 1.0, 2.5),
            TunableParam("vol_ratio_threshold", 1.2, 0.8, 2.0),
            TunableParam("vix_max", 22.0, 16.0, 28.0),
            TunableParam("stop_pts", 3.0, 2.0, 6.0),
            TunableParam("target_pts", 5.0, 3.0, 10.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # --- Extract spot arrays (forward-fill NaN in Polars first) ---
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # --- Parameters ---
        threshold = params.get("imbalance_zscore_threshold", 1.5)
        vol_thresh = params.get("vol_ratio_threshold", 1.2)
        vix_max = params.get("vix_max", 22.0)
        stop_pts = params.get("stop_pts", 3.0)
        target_pts = params.get("target_pts", 5.0)

        # --- VIX filter ---
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # --- Bar imbalance: close position within bar range, centred at 0 ---
        bar_range = high - low
        bar_imbalance = np.where(
            bar_range > 0.0,
            (close - low) / bar_range - 0.5,
            0.0,
        )

        # --- EMA(12) of bar imbalance = 1-minute order flow pressure ---
        alpha = 2.0 / (12 + 1)
        imb_ema = np.zeros(n)
        imb_ema[0] = bar_imbalance[0]
        for i in range(1, n):
            imb_ema[i] = alpha * bar_imbalance[i] + (1.0 - alpha) * imb_ema[i - 1]

        # --- Z-score of EMA over 60-bar (5-min) rolling window ---
        imb_zscore = np.zeros(n)
        for i in range(60, n):
            window = imb_ema[i - 60:i]
            m = np.mean(window)
            s = np.std(window)
            if s > 1e-6:
                imb_zscore[i] = (imb_ema[i] - m) / s

        # --- Relative volume: vol / 60-bar SMA ---
        vol_sma = np.zeros(n)
        for i in range(60, n):
            vol_sma[i] = np.mean(volume[i - 60:i])
        vol_ratio = np.where(vol_sma > 0.0, volume / vol_sma, 1.0)

        # --- Session VWAP (cumulative from day open) ---
        vwap = np.zeros(n)
        cum_vol = 0.0
        cum_pv = 0.0
        prev_day = -1
        for i in range(n):
            if day_id[i] != prev_day:
                cum_vol = 0.0
                cum_pv = 0.0
                prev_day = day_id[i]
            v = volume[i]
            cum_vol += v
            cum_pv += close[i] * v
            vwap[i] = cum_pv / cum_vol if cum_vol > 0.0 else close[i]

        # --- 2-bar confirmation: z-score sustained on prior 2 bars ---
        # Confirmation level = 0.67 * threshold (ensures it was already elevated,
        # not just a single-bar spike at the current bar)
        confirm_level = 0.67 * threshold
        confirmed_bull = np.zeros(n, dtype=bool)
        confirmed_bear = np.zeros(n, dtype=bool)
        for i in range(2, n):
            confirmed_bull[i] = (
                imb_zscore[i - 1] > confirm_level
                and imb_zscore[i - 2] > confirm_level
            )
            confirmed_bear[i] = (
                imb_zscore[i - 1] < -confirm_level
                and imb_zscore[i - 2] < -confirm_level
            )

        # --- Session and filter masks ---
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        vix_ok = vix_close < vix_max
        vol_ok = vol_ratio > vol_thresh

        # --- Entry signals ---
        buy_ce = (
            in_session
            & vix_ok
            & vol_ok
            & (imb_zscore > threshold)
            & confirmed_bull
            & (close > vwap)
        )

        buy_pe = (
            in_session
            & vix_ok
            & vol_ok
            & (imb_zscore < -threshold)
            & confirmed_bear
            & (close < vwap)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,       # 90-second maximum hold
            max_trades_per_day=self.max_trades_per_day,
        )
