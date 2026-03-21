"""
intraday_vol_pattern_v1 — U-Shaped Intraday Volatility Pattern on NIFTY

Thesis: NIFTY exhibits a structural U-shaped intraday vol pattern driven by institutional
order scheduling. At open (09:20-10:00) and close (14:30-15:25), FII/DII TWAP flows create
directional momentum when vol rises 15%+ above session baseline — trade with EMA alignment.
During lunch (11:30-13:30) only market makers are active, range is narrow, and BB touches
revert within 60-90s when vol is 15%+ below baseline — fade Bollinger Band extremes.
Afternoon (13:30-14:30) catches vol expansion transitions toward the close.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "intraday_vol_pattern_v1"
    underlying = "NIFTY"
    session_start_minutes = 560    # 09:20 IST
    session_end_minutes = 925      # 15:25 IST
    max_trades_per_day = 6
    max_lookback = 360             # 30 min warmup for long vol baseline (360 × 5s)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_high_ratio", 1.15, 1.05, 1.40),  # vol surge threshold for momentum
            TunableParam("vol_low_ratio",  0.85, 0.65, 0.95),  # vol contraction threshold for mean-rev
            TunableParam("bb_mult",        2.0,  1.5,  2.5),   # Bollinger Band width multiplier
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill NaN in Polars before converting to numpy
        close = spot_df["close"].fill_null(strategy="forward").fill_null(0.0).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        vol_high_ratio = params.get("vol_high_ratio", 1.15)
        vol_low_ratio  = params.get("vol_low_ratio",  0.85)
        bb_mult        = params.get("bb_mult",        2.0)

        # ── Log returns (safe division) ──────────────────────────────────────
        ret = np.zeros(n)
        prev = np.where(close[:-1] > 0, close[:-1], 1.0)
        ret[1:] = np.log(np.where(close[1:] > 0, close[1:], 1.0) / prev)

        # ── Short vol: 36-bar (3-min) rolling std — current regime ───────────
        SHORT_WIN = 36
        vol_short = np.zeros(n)
        for i in range(SHORT_WIN, n):
            vol_short[i] = np.std(ret[i - SHORT_WIN:i])

        # ── Long vol: 360-bar (30-min) rolling std — session baseline ────────
        LONG_WIN = 360
        vol_long = np.zeros(n)
        for i in range(LONG_WIN, n):
            vol_long[i] = np.std(ret[i - LONG_WIN:i])

        # vol_ratio > 1 → vol expanding (momentum), < 1 → vol contracting (mean-rev)
        vol_ratio = np.where(vol_long > 1e-8, vol_short / vol_long, 1.0)

        # ── Bollinger Bands (120-bar = 10-min) for mean-reversion ────────────
        BB_WIN = 120
        bb_mid   = np.zeros(n)
        bb_upper = np.zeros(n)
        bb_lower = np.zeros(n)
        for i in range(BB_WIN, n):
            w = close[i - BB_WIN:i]
            m = np.mean(w)
            s = np.std(w)
            bb_mid[i]   = m
            bb_upper[i] = m + bb_mult * s
            bb_lower[i] = m - bb_mult * s

        # ── EMAs: fast 36-bar (3-min), slow 120-bar (10-min) ─────────────────
        EMA_FAST, EMA_SLOW = 36, 120
        alpha_f = 2.0 / (EMA_FAST + 1)
        alpha_s = 2.0 / (EMA_SLOW + 1)
        ema_fast = np.zeros(n)
        ema_slow = np.zeros(n)
        ema_fast[0] = close[0]
        ema_slow[0] = close[0]
        for i in range(1, n):
            ema_fast[i] = alpha_f * close[i] + (1.0 - alpha_f) * ema_fast[i - 1]
            ema_slow[i] = alpha_s * close[i] + (1.0 - alpha_s) * ema_slow[i - 1]

        # ── Session period masks (minutes from midnight) ──────────────────────
        # Open: 09:20-10:00 (560-600)  ← momentum regime
        # Lunch: 11:30-13:30 (690-810) ← mean-reversion regime
        # Afternoon: 13:30-14:30 (810-870) ← vol-expansion regime
        # Close: 14:30-15:25 (870-925) ← momentum regime
        is_open    = (time_min >= 560) & (time_min < 600)
        is_lunch   = (time_min >= 690) & (time_min < 810)
        is_aftnoon = (time_min >= 810) & (time_min < 870)
        is_close   = (time_min >= 870) & (time_min < 925)
        is_hi_vol  = is_open | is_close   # momentum periods

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # Warmed-up mask — need LONG_WIN bars for stable vol_ratio
        warmed = np.zeros(n, dtype=bool)
        if LONG_WIN < n:
            warmed[LONG_WIN:] = True

        # ── Signal generation ────────────────────────────────────────────────

        # 1) MOMENTUM regime (open / close): EMA aligned + vol surging
        mom_bull = (ema_fast > ema_slow) & (vol_ratio > vol_high_ratio)
        mom_bear = (ema_fast < ema_slow) & (vol_ratio > vol_high_ratio)
        buy_ce_mom = in_session & warmed & is_hi_vol & mom_bull
        buy_pe_mom = in_session & warmed & is_hi_vol & mom_bear

        # 2) MEAN-REVERSION regime (lunch): BB extreme + vol suppressed
        mr_bull = (close < bb_lower) & (vol_ratio < vol_low_ratio) & (bb_lower > 0)
        mr_bear = (close > bb_upper) & (vol_ratio < vol_low_ratio) & (bb_upper > 0)
        buy_ce_mr = in_session & warmed & is_lunch & mr_bull
        buy_pe_mr = in_session & warmed & is_lunch & mr_bear

        # 3) VOL-EXPANSION regime (afternoon): vol transitioning up + EMA direction + price side of mid
        vol_expanding = (vol_ratio > 1.0) & (vol_ratio <= vol_high_ratio)
        afx_bull = (ema_fast > ema_slow) & vol_expanding & (close > bb_mid) & (bb_mid > 0)
        afx_bear = (ema_fast < ema_slow) & vol_expanding & (close < bb_mid) & (bb_mid > 0)
        buy_ce_afx = in_session & warmed & is_aftnoon & afx_bull
        buy_pe_afx = in_session & warmed & is_aftnoon & afx_bear

        buy_ce = buy_ce_mom | buy_ce_mr | buy_ce_afx
        buy_pe = buy_pe_mom | buy_pe_mr | buy_pe_afx

        # ── Per-bar stops and targets (regime-dependent) ─────────────────────
        # Lunch mean-reversion: stop 4 pts (BB deviation can extend ~8 spot pts),
        #   target 7 pts (BB-to-mid reversion ~15-20 spot pts; 7 = conservative half).
        # Open/close momentum: stop 3 pts (stall within 30s = thesis wrong),
        #   target 5 pts (~50% of expected 30-60s NIFTY momentum push at delta 0.5).
        # Afternoon vol-expansion: stop 4 pts, target 6 pts (regime uncertainty).
        stop_pts   = np.where(is_lunch, 4.0, np.where(is_aftnoon, 4.0, 3.0))
        target_pts = np.where(is_lunch, 7.0, np.where(is_aftnoon, 6.0, 5.0))

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=stop_pts,
            target_points=target_pts,
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,           # 120-second max hold
            max_trades_per_day=self.max_trades_per_day,
        )
