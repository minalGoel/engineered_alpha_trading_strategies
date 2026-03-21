"""VIX Regime Switch Strategy — vix_regime_switch_v1

Trades NIFTY ATM options at 5-second frequency by detecting the current India VIX
regime and applying regime-appropriate entry logic:
  - Low VIX (<14): Bollinger Band mean-reversion (BB + RSI + ADX<25)
  - Elevated VIX (18-25): EMA crossover momentum (EMA + ADX>25)
  - Crisis VIX (>=25): No new entries

Hold time: 30-90 seconds (time_stop_bars=18).
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "vix_regime_switch_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 8
    max_lookback = 360            # 30-min warmup for VIX stability filter

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vix_low_threshold", 14.0, 11.0, 17.0),
            TunableParam("vix_high_threshold", 18.0, 16.0, 24.0),
            TunableParam("rsi_oversold", 35.0, 25.0, 45.0),
            TunableParam("rsi_overbought", 65.0, 55.0, 75.0),
            TunableParam("adx_trend_threshold", 25.0, 18.0, 35.0),
            TunableParam("mean_rev_stop_pts", 3.0, 2.0, 6.0),
            TunableParam("mean_rev_target_pts", 5.0, 3.0, 9.0),
            TunableParam("momentum_stop_pts", 5.0, 3.0, 9.0),
            TunableParam("momentum_target_pts", 8.0, 5.0, 15.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── Raw arrays (forward-fill NaN in Polars before numpy) ──
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        # ── VIX aligned to spot bars ──
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Parameters ──
        vix_low = params.get("vix_low_threshold", 14.0)
        vix_high = params.get("vix_high_threshold", 18.0)
        rsi_oversold = params.get("rsi_oversold", 35.0)
        rsi_overbought = params.get("rsi_overbought", 65.0)
        adx_thresh = params.get("adx_trend_threshold", 25.0)
        mean_rev_stop = params.get("mean_rev_stop_pts", 3.0)
        mean_rev_target = params.get("mean_rev_target_pts", 5.0)
        mom_stop = params.get("momentum_stop_pts", 5.0)
        mom_target = params.get("momentum_target_pts", 8.0)

        # ── BB(240) — 20-minute Bollinger Bands ──
        bb_period = 240
        bb_upper = np.zeros(n)
        bb_lower = np.zeros(n)
        cum_c = np.cumsum(close)
        cum_c2 = np.cumsum(close ** 2)
        for i in range(bb_period, n):
            s = cum_c[i] - cum_c[i - bb_period]
            s2 = cum_c2[i] - cum_c2[i - bb_period]
            mean = s / bb_period
            var = s2 / bb_period - mean ** 2
            std = np.sqrt(max(var, 0.0))
            bb_upper[i] = mean + 2.0 * std
            bb_lower[i] = mean - 2.0 * std

        # ── RSI(36) — 3-minute RSI via rolling mean of gains/losses ──
        rsi_period = 36
        rsi = np.full(n, 50.0)
        gains = np.zeros(n)
        losses = np.zeros(n)
        diff = close[1:] - close[:-1]
        gains[1:] = np.where(diff > 0, diff, 0.0)
        losses[1:] = np.where(diff < 0, -diff, 0.0)
        cum_g = np.cumsum(gains)
        cum_l = np.cumsum(losses)
        for i in range(rsi_period, n):
            avg_gain = (cum_g[i] - cum_g[i - rsi_period]) / rsi_period
            avg_loss = (cum_l[i] - cum_l[i - rsi_period]) / rsi_period
            if avg_loss < 1e-10:
                rsi[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100.0 - (100.0 / (1.0 + rs))

        # ── ADX(60) — 5-minute ADX ──
        adx_period = 60
        adx = np.full(n, 20.0)
        tr = np.zeros(n)
        dm_plus = np.zeros(n)
        dm_minus = np.zeros(n)
        for i in range(1, n):
            hl = high[i] - low[i]
            hpc = abs(high[i] - close[i - 1])
            lpc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hpc, lpc)
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]
            dm_plus[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
            dm_minus[i] = down_move if (down_move > up_move and down_move > 0) else 0.0
        cum_tr = np.cumsum(tr)
        cum_dmp = np.cumsum(dm_plus)
        cum_dmm = np.cumsum(dm_minus)
        for i in range(adx_period, n):
            atr_sum = cum_tr[i] - cum_tr[i - adx_period]
            if atr_sum < 1e-10:
                continue
            pdi = 100.0 * (cum_dmp[i] - cum_dmp[i - adx_period]) / atr_sum
            mdi = 100.0 * (cum_dmm[i] - cum_dmm[i - adx_period]) / atr_sum
            denom = pdi + mdi
            adx[i] = abs(pdi - mdi) / denom * 100.0 if denom > 0 else 0.0

        # ── EMA(36) and EMA(120) ──
        ema36 = np.zeros(n)
        ema120 = np.zeros(n)
        a36 = 2.0 / (36 + 1)
        a120 = 2.0 / (120 + 1)
        ema36[0] = close[0]
        ema120[0] = close[0]
        for i in range(1, n):
            ema36[i] = a36 * close[i] + (1.0 - a36) * ema36[i - 1]
            ema120[i] = a120 * close[i] + (1.0 - a120) * ema120[i - 1]

        # EMA36 slope over last 3 bars (15 seconds)
        ema36_slope = np.zeros(n)
        ema36_slope[3:] = ema36[3:] - ema36[:-3]

        # ── VIX stability filter — 30-min rolling std of VIX < 1.5 ──
        vix_stable = np.ones(n, dtype=bool)
        vix_stab_period = 360
        cum_v = np.cumsum(vix_close)
        cum_v2 = np.cumsum(vix_close ** 2)
        for i in range(vix_stab_period, n):
            sv = cum_v[i] - cum_v[i - vix_stab_period]
            sv2 = cum_v2[i] - cum_v2[i - vix_stab_period]
            mean_v = sv / vix_stab_period
            var_v = sv2 / vix_stab_period - mean_v ** 2
            vix_stable[i] = np.sqrt(max(var_v, 0.0)) < 1.5

        # ── Session and warmup filters ──
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed = np.zeros(n, dtype=bool)
        warmed[bb_period:] = True   # BB needs the most warmup (240 bars)

        # ── VIX regime masks ──
        low_vix_regime = vix_close < vix_low
        elevated_vix_regime = (vix_close >= vix_high) & (vix_close < 25.0)
        crisis_vix = vix_close >= 25.0

        # ── Mean-reversion signals (low VIX) ──
        mr_buy_ce = (
            low_vix_regime
            & (close < bb_lower)
            & (rsi < rsi_oversold)
            & (adx < adx_thresh)
            & (bb_lower > 0.0)
        )
        mr_buy_pe = (
            low_vix_regime
            & (close > bb_upper)
            & (rsi > rsi_overbought)
            & (adx < adx_thresh)
            & (bb_upper > 0.0)
        )

        # ── Momentum signals (elevated VIX) ──
        mom_buy_ce = (
            elevated_vix_regime
            & (ema36 > ema120)
            & (ema36_slope > 0.0)
            & (adx > adx_thresh)
        )
        mom_buy_pe = (
            elevated_vix_regime
            & (ema36 < ema120)
            & (ema36_slope < 0.0)
            & (adx > adx_thresh)
        )

        # ── Combine: apply global filters ──
        active = in_session & warmed & (~crisis_vix) & vix_stable
        buy_ce = active & (mr_buy_ce | mom_buy_ce)
        buy_pe = active & (mr_buy_pe | mom_buy_pe)

        # Resolve simultaneous CE+PE (cancel both — ambiguous regime boundary)
        both = buy_ce & buy_pe
        buy_ce = buy_ce & (~both)
        buy_pe = buy_pe & (~both)

        # ── Per-bar stop/target: regime-conditional ──
        stop_arr = np.where(low_vix_regime, mean_rev_stop, mom_stop).astype(np.float64)
        target_arr = np.where(low_vix_regime, mean_rev_target, mom_target).astype(np.float64)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=stop_arr,
            target_points=target_arr,
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
