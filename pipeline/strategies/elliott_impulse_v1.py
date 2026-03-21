"""Elliott Impulse Wave 3 Continuation — 5-second NIFTY index options.

Detects micro-Elliott-wave structure on NIFTY at 5-second resolution:
  Wave 1: 15-40 second institutional TWAP burst (>= min_w1_pts spot points)
  Wave 2: 38-62% Fibonacci retracement as counter-trend limits absorb flow
  Wave 3: Breakout above Wave 1 peak signals order flow resumption — BUY CE
          Breakdown below Wave 1 trough signals selling resumption — BUY PE

Hold 30-90 seconds (6-18 bars), targeting the first leg of the Wave 3 extension.
"""
from __future__ import annotations
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder smoothed RSI."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 2:
        return rsi
    delta = np.empty(n)
    delta[0] = 0.0
    delta[1:] = close[1:] - close[:-1]
    gain = np.where(delta > 0.0, delta, 0.0)
    loss = np.where(delta < 0.0, -delta, 0.0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[period] = np.mean(gain[1 : period + 1])
    avg_loss[period] = np.mean(loss[1 : period + 1])
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss > 1e-9, avg_gain / avg_loss, 100.0)
        rsi[period:] = np.where(avg_loss[period:] > 1e-9, 100.0 - 100.0 / (1.0 + rs[period:]), 100.0)
    return rsi


def _compute_ema(close: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average."""
    n = len(close)
    ema = np.empty(n)
    ema[0] = close[0]
    k = 2.0 / (period + 1)
    for i in range(1, n):
        ema[i] = close[i] * k + ema[i - 1] * (1.0 - k)
    return ema


def _detect_wave_signals(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    rsi: np.ndarray,
    ema: np.ndarray,
    in_session: np.ndarray,
    vix_ok: np.ndarray,
    wave_lb: int,
    min_w1_pts: float,
    retrace_lo: float,
    retrace_hi: float,
    rsi_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect bullish/bearish Elliott Wave 3 breakout signals."""
    n = len(close)
    buy_ce = np.zeros(n, dtype=bool)
    buy_pe = np.zeros(n, dtype=bool)

    # lb2: first 2/3 of window is where Wave 1 start must originate
    lb2 = max(4, wave_lb * 2 // 3)

    for i in range(wave_lb + 1, n):
        if not in_session[i] or not vix_ok[i]:
            continue

        # Slice window (excludes bar i — wave pattern must be fully formed before entry)
        w_high = high[i - wave_lb : i]
        w_low = low[i - wave_lb : i]
        cur_close = close[i]

        # ── BULLISH SETUP: W1 low → W1 high → W2 retrace → W3 breakout ──
        # Find Wave 1 low in first lb2 bars of window
        w1_low_idx = int(np.argmin(w_low[:lb2]))
        w1_low_val = w_low[w1_low_idx]

        # Find Wave 1 high AFTER w1_low_idx (exclude last 2 bars — need W2 room)
        post_low = w_high[w1_low_idx:]
        if len(post_low) >= 3:
            w1_high_offset = int(np.argmax(post_low[:-2]))
            w1_high_val = post_low[:-2][w1_high_offset]
            w1_high_idx = w1_low_idx + w1_high_offset
            w1_size = w1_high_val - w1_low_val

            if w1_size >= min_w1_pts and w1_high_idx > w1_low_idx:
                # Wave 2: deepest low after wave 1 high
                post_w1h = w_low[w1_high_idx:]
                if len(post_w1h) >= 1:
                    w2_low_val = float(np.min(post_w1h))
                    retrace = (w1_high_val - w2_low_val) / w1_size

                    # Wave 3 breakout: current bar closes above Wave 1 high
                    if (
                        retrace_lo <= retrace <= retrace_hi
                        and cur_close > w1_high_val
                        and rsi[i] > rsi_threshold
                        and rsi[i] > rsi[i - 1]
                        and cur_close > ema[i]
                    ):
                        buy_ce[i] = True

        # ── BEARISH SETUP: W1 high → W1 low → W2 retrace → W3 breakdown ──
        # Find Wave 1 high in first lb2 bars of window
        w1_high_idx_b = int(np.argmax(w_high[:lb2]))
        w1_high_val_b = w_high[w1_high_idx_b]

        # Find Wave 1 low AFTER w1_high_idx_b (exclude last 2 bars)
        post_high = w_low[w1_high_idx_b:]
        if len(post_high) >= 3:
            w1_low_offset_b = int(np.argmin(post_high[:-2]))
            w1_low_val_b = post_high[:-2][w1_low_offset_b]
            w1_low_idx_b = w1_high_idx_b + w1_low_offset_b
            w1_size_b = w1_high_val_b - w1_low_val_b

            if w1_size_b >= min_w1_pts and w1_low_idx_b > w1_high_idx_b:
                # Wave 2: highest high after wave 1 low
                post_w1l = w_high[w1_low_idx_b:]
                if len(post_w1l) >= 1:
                    w2_high_val = float(np.max(post_w1l))
                    retrace_b = (w2_high_val - w1_low_val_b) / w1_size_b

                    # Wave 3 breakdown: current bar closes below Wave 1 low
                    if (
                        retrace_lo <= retrace_b <= retrace_hi
                        and cur_close < w1_low_val_b
                        and rsi[i] < (100.0 - rsi_threshold)
                        and rsi[i] < rsi[i - 1]
                        and cur_close < ema[i]
                    ):
                        buy_pe[i] = True

    # Mutual exclusion — if both trigger, neither fires
    both = buy_ce & buy_pe
    buy_ce[both] = False
    buy_pe[both] = False

    return buy_ce, buy_pe


class Strategy(BaseStrategy):
    name = "elliott_impulse_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 8
    max_lookback = 120            # 10 min warmup for RSI stability

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("min_w1_pts", 8.0, 4.0, 20.0),
            TunableParam("retrace_lo", 0.35, 0.25, 0.50),
            TunableParam("retrace_hi", 0.65, 0.50, 0.80),
            TunableParam("rsi_threshold", 52.0, 48.0, 62.0),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 7.0, 4.0, 15.0),
            TunableParam("vix_low", 11.0, 9.0, 15.0),
            TunableParam("vix_high", 22.0, 18.0, 30.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict,
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract spot arrays (forward-fill NaN before numpy) ──
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        high = spot_df["high"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        low = spot_df["low"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()

        # ── Parameters ──
        min_w1_pts = float(params.get("min_w1_pts", 8.0))
        retrace_lo = float(params.get("retrace_lo", 0.35))
        retrace_hi = float(params.get("retrace_hi", 0.65))
        rsi_threshold = float(params.get("rsi_threshold", 52.0))
        stop_pts = float(params.get("stop_pts", 4.0))
        target_pts = float(params.get("target_pts", 7.0))
        vix_low = float(params.get("vix_low", 11.0))
        vix_high = float(params.get("vix_high", 22.0))

        # ── Indicators ──
        # RSI(60) = 5-minute RSI: captures momentum of micro-wave cycles
        rsi = _compute_rsi(close, 60)
        # EMA(36) = 3-minute trend context: only trade waves aligned with micro-trend
        ema36 = _compute_ema(close, 36)

        # ── VIX filter (aligned to spot bars) ──
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy().astype(np.float64)

        # ── Session and VIX masks ──
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        vix_ok = (vix_close >= vix_low) & (vix_close <= vix_high)

        # ── Wave signal detection ──
        wave_lb = 24  # 2-minute window: enough to hold W1 + W2 + detect W3 breakout
        buy_ce, buy_pe = _detect_wave_signals(
            close=close,
            high=high,
            low=low,
            rsi=rsi,
            ema=ema36,
            in_session=in_session,
            vix_ok=vix_ok,
            wave_lb=wave_lb,
            min_w1_pts=min_w1_pts,
            retrace_lo=retrace_lo,
            retrace_hi=retrace_hi,
            rsi_threshold=rsi_threshold,
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds — Wave 3 first leg should complete
            max_trades_per_day=self.max_trades_per_day,
        )
