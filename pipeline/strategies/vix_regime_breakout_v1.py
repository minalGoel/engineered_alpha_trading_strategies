"""vix_regime_breakout_v1 — VIX Spike Mean-Reversion on NIFTY

When India VIX spikes >10% above its intraday session baseline, FII desks and domestic risk
engines execute NIFTY futures sell programs as portfolio delta hedges. The instant VIX begins
retracing from its intraday peak (>3% pullback), these hedges unwind and market makers who
sold puts delta-hedge by buying NIFTY futures, producing a sharp CE premium expansion within
30-90 seconds. Conversely, while VIX is still rising, directional selling has not peaked and
PE purchases capture further NIFTY downside.

Converted from Strategy_73.json (equity, 1-min bars, 30-120 min hold).
Hold compressed to 30-90s: enters at the moment of VIX peak confirmation.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "vix_regime_breakout_v1"
    underlying = "NIFTY"
    session_start_minutes = 600   # 10:00 IST — let panic settle per original
    session_end_minutes = 870     # 14:30 IST
    max_lookback = 120            # 10 min warmup (120 bars × 5s)
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vix_spike_threshold", 10.0, 5.0, 20.0),
            TunableParam("vix_pullback_threshold", 3.0, 1.0, 7.0),
            TunableParam("nifty_drop_threshold", 0.3, 0.1, 0.8),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 8.0, 4.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        vix_spike_thr = params.get("vix_spike_threshold", 10.0)
        vix_pullback_thr = params.get("vix_pullback_threshold", 3.0)
        nifty_drop_thr = params.get("nifty_drop_threshold", 0.3)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 8.0)

        # ── VIX aligned to spot bars ──────────────────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Cumulative session VIX min and max (reset each day) ───────────────
        # vix_session_min = pre-spike baseline (equivalent to original's vix_prev_close)
        # vix_session_max = intraday peak (for pullback detection)
        vix_session_min = np.empty(n)
        vix_session_max = np.empty(n)
        cur_day = -1
        running_min = np.inf
        running_max = -np.inf
        for i in range(n):
            if day_id[i] != cur_day:
                cur_day = day_id[i]
                running_min = vix_close[i]
                running_max = vix_close[i]
            else:
                if vix_close[i] < running_min:
                    running_min = vix_close[i]
                if vix_close[i] > running_max:
                    running_max = vix_close[i]
            vix_session_min[i] = running_min
            vix_session_max[i] = running_max

        # ── VIX spike: rise from session baseline ─────────────────────────────
        safe_min = np.where(vix_session_min > 0, vix_session_min, 15.0)
        vix_spike_pct = (vix_close - safe_min) / safe_min * 100.0

        # ── VIX pullback: retracement from session peak ───────────────────────
        safe_max = np.where(vix_session_max > 0, vix_session_max, 15.0)
        vix_pullback_pct = (safe_max - vix_close) / safe_max * 100.0

        # ── VIX still rising: 30-second direction check (6 bars × 5s) ────────
        vix_rising = np.zeros(n, dtype=bool)
        vix_rising[6:] = vix_close[6:] > vix_close[:-6]

        # ── NIFTY 5-minute return (60 bars × 5s) ─────────────────────────────
        # Compressed from original's 60-min (60 1-min bars) to 5 min to capture
        # only the acute selldown accompanying the VIX spike.
        nifty_ret_60 = np.zeros(n)
        if n > 60:
            base = close[:n - 60]
            safe_base = np.where(base > 0, base, 1.0)
            nifty_ret_60[60:] = (close[60:] - base) / safe_base * 100.0

        # ── Session + warmup gate ─────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed = np.zeros(n, dtype=bool)
        if n > 60:
            warmed[60:] = True
        active = in_session & warmed

        # ── Buy CE: VIX spiked AND now reverting → NIFTY bounce ──────────────
        buy_ce = (
            active
            & (vix_spike_pct > vix_spike_thr)
            & (vix_pullback_pct > vix_pullback_thr)
            & (nifty_ret_60 < -nifty_drop_thr)
        )

        # ── Buy PE: VIX spiked AND still rising → NIFTY continued downside ───
        # Only when CE is not firing (CE confirmation takes priority)
        buy_pe = (
            active
            & (vix_spike_pct > vix_spike_thr)
            & vix_rising
            & (nifty_ret_60 < -nifty_drop_thr)
            & ~buy_ce
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,           # 90 seconds (18 × 5s)
            max_trades_per_day=self.max_trades_per_day,
        )
