"""
narrow_range_7_breakout — NR7 micro-squeeze breakout on NIFTY 5-second bars.

Mechanism: When a 5-second bar's range is smaller than every prior bar over
the last ~1.7 minutes (20 bars), MM quotes compress and resting orders cluster
tightly. A subsequent close above that NR bar's high triggers buy-stop activation
and momentum algo entries, releasing compressed energy into a 10-20 spot point
directional burst within 15-45 seconds. Volume expansion confirms genuine flow.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "narrow_range_7_breakout"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 8
    max_lookback = 240            # 20-min warmup (240 × 5s)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("nr_lookback", 20.0, 10.0, 40.0),
            TunableParam("vol_threshold", 1.2, 1.0, 2.5),
            TunableParam("nr_freshness_bars", 6.0, 3.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close  = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high   = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low    = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        nr_lookback    = int(params.get("nr_lookback", 20))
        vol_threshold  = float(params.get("vol_threshold", 1.2))
        freshness_bars = int(params.get("nr_freshness_bars", 6))
        vol_lookback   = 12  # fixed: 1-minute volume average

        # ── Bar range ─────────────────────────────────────────────────────────
        bar_range = high - low

        # ── Rolling minimum range (previous nr_lookback bars, excluding current) ──
        rolling_min_range = np.full(n, np.inf)
        for i in range(1, n):
            start = max(0, i - nr_lookback)
            rolling_min_range[i] = np.min(bar_range[start:i])

        # NR bar: current range <= minimum of previous nr_lookback bars
        is_nr_bar = (bar_range <= rolling_min_range) & np.isfinite(rolling_min_range)

        # ── Volume MA (1-minute rolling average) ─────────────────────────────
        vol_ma = np.zeros(n)
        for i in range(vol_lookback, n):
            vol_ma[i] = np.mean(volume[i - vol_lookback:i])
        # Warmup bars: large sentinel so vol_confirm = False
        vol_ma[:vol_lookback] = np.finfo(float).max

        # ── NR reference levels (valid only within freshness_bars after NR bar) ─
        # Outside the window: NaN → filled with close → close > close = False
        nr_ref_high = np.full(n, np.nan)
        nr_ref_low  = np.full(n, np.nan)
        last_nr_h   = np.nan
        last_nr_l   = np.nan
        last_nr_idx = -1

        for i in range(n):
            if is_nr_bar[i]:
                last_nr_h   = high[i]
                last_nr_l   = low[i]
                last_nr_idx = i
            # Assign reference for bars 1..freshness_bars after the NR bar
            if last_nr_idx >= 0 and 0 < (i - last_nr_idx) <= freshness_bars:
                nr_ref_high[i] = last_nr_h
                nr_ref_low[i]  = last_nr_l

        # NaN → close (neutral: close > close is always False → no stale signals)
        nr_ref_high = np.where(np.isnan(nr_ref_high), close, nr_ref_high)
        nr_ref_low  = np.where(np.isnan(nr_ref_low),  close, nr_ref_low)

        # ── Filters ───────────────────────────────────────────────────────────
        in_session  = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        vol_confirm = volume > (vol_ma * vol_threshold)
        warmup_done = np.arange(n) >= max(nr_lookback, vol_lookback)

        # ── Signals ───────────────────────────────────────────────────────────
        # Close above NR bar high → bullish → buy CE
        buy_ce = in_session & warmup_done & vol_confirm & (~is_nr_bar) & (close > nr_ref_high)
        # Close below NR bar low → bearish → buy PE
        buy_pe = in_session & warmup_done & vol_confirm & (~is_nr_bar) & (close < nr_ref_low)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, 3.0),
            target_points=np.full(n, 5.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=12,
            max_trades_per_day=self.max_trades_per_day,
        )
