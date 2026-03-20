"""
microstructure_reversion_v1 — CLV-based order flow imbalance exhaustion on NIFTY

Mechanism:
    Each 5-second bar's close location value (CLV) approximates intra-bar buy vs sell
    pressure. When 20 consecutive 5-second bars (100s) show >68% buy-side volume dominance
    AND NIFTY has risen >0.15% over those 100s, institutional market-order buyers have
    absorbed available limit-sell supply. When the rolling buy imbalance fraction begins
    declining (sellers returning), the one-sided flow has exhausted and NIFTY reverts
    toward session VWAP. Entry at the imbalance inflection point captures the first 10-20
    NIFTY spot points of reversion. Symmetric logic applies for sell-side exhaustion.

Original:
    Strategy_155.json (microstructure_reversion_v1) — NIFTY 100 stocks, 1-min bars,
    CLV buy/sell approximation, 10-bar imbalance, hold 5-20 min. Compressed to 5s index
    with 20-bar (100s) window to match 30-90s hold time.
"""

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "microstructure_reversion_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — skip first 10 min opening volatility
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 8
    max_lookback = 120            # 10 min warmup: 20-bar imbalance + VWAP accumulation

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("imbalance_high", 0.68, 0.60, 0.80),
            TunableParam("imbalance_low", 0.32, 0.20, 0.40),
            TunableParam("price_change_threshold", 0.0015, 0.0005, 0.0030),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 6.0, 3.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        imbalance_high = params.get("imbalance_high", 0.68)
        imbalance_low = params.get("imbalance_low", 0.32)
        price_chg_thr = params.get("price_change_threshold", 0.0015)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 6.0)

        # ── Extract raw arrays (forward-fill NaN in Polars before numpy) ──
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        # ── Session VWAP (cumulative per day via Polars groupby) ──
        spot_with_vwap = spot_df.with_columns(
            (pl.col("close") * pl.col("volume")).alias("_pv")
        ).with_columns([
            pl.col("_pv").cum_sum().over("day_id").alias("_cum_pv"),
            pl.col("volume").cast(pl.Float64).cum_sum().over("day_id").alias("_cum_vol"),
        ]).with_columns(
            pl.when(pl.col("_cum_vol") > 0)
            .then(pl.col("_cum_pv") / pl.col("_cum_vol"))
            .otherwise(pl.col("close"))
            .alias("vwap")
        )
        vwap = spot_with_vwap["vwap"].fill_null(strategy="forward").to_numpy()

        # ── CLV-based buy volume approximation (per 5-second bar) ──
        # buy_vol = volume * (close - low) / (high - low) if range > 0 else volume * 0.5
        bar_range = high - low
        buy_vol = np.where(
            bar_range > 0,
            volume * (close - low) / bar_range,
            volume * 0.5,
        )

        # ── 20-bar rolling buy imbalance (100 seconds ≈ 1.7 min) ──
        # Neutral value 0.5 (balanced) for warmup bars
        WINDOW = 20
        buy_imbalance = np.full(n, 0.5)
        for i in range(WINDOW, n):
            tot_vol = float(np.sum(volume[i - WINDOW:i]))
            if tot_vol > 0:
                buy_imbalance[i] = float(np.sum(buy_vol[i - WINDOW:i])) / tot_vol

        # ── 20-bar price change (same window as imbalance) ──
        price_change_20 = np.zeros(n)
        for i in range(WINDOW, n):
            ref = close[i - WINDOW]
            if ref > 0:
                price_change_20[i] = (close[i] - ref) / ref

        # ── Imbalance turn detection (1-bar lag comparison) ──
        imb_rising = np.zeros(n, dtype=bool)
        imb_falling = np.zeros(n, dtype=bool)
        imb_rising[1:] = buy_imbalance[1:] > buy_imbalance[:-1]
        imb_falling[1:] = buy_imbalance[1:] < buy_imbalance[:-1]

        # ── Session filter ──
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── Buy CE (bullish reversion from sell exhaustion) ──
        # Sell-side dominated last 100s → absorbed bids → sellers exhausted → bounce
        buy_ce = (
            in_session
            & (buy_imbalance < imbalance_low)      # sell flow dominated last 20 bars
            & (price_change_20 < -price_chg_thr)   # price was pushed down
            & (close < vwap)                        # below VWAP — fair-value gap exists
            & imb_rising                            # selling waning, buyers returning
        )

        # ── Buy PE (bearish reversion from buy exhaustion) ──
        # Buy-side dominated last 100s → absorbed offers → buyers exhausted → fade
        buy_pe = (
            in_session
            & (buy_imbalance > imbalance_high)     # buy flow dominated last 20 bars
            & (price_change_20 > price_chg_thr)    # price was pushed up
            & (close > vwap)                       # above VWAP — extended vs fair value
            & imb_falling                          # buying waning, sellers returning
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,              # 90s max hold (18 × 5s)
            max_trades_per_day=self.max_trades_per_day,
        )
