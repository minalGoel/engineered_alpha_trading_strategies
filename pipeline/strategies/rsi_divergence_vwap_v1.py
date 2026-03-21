"""RSI Divergence + VWAP Proximity — NIFTY 5s options strategy.

Mechanism: On NIFTY, institutional TWAP selling creates price lower lows while
RSI fails to confirm (absorption pauses slow momentum velocity). When NIFTY is
within 0.15% below session VWAP, VWAP-benchmarked algos accelerate accumulation,
triggering a 30-90 second bounce. Symmetric for bearish divergence above VWAP.

Converted from: trading_strategies/unique_strategies_all/Strategy_76.json
Original: RSI(14) divergence + VWAP proximity on FnO stocks at 1-min, hold 15-60 min.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder smoothed RSI."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n <= period:
        return rsi

    delta = np.zeros(n)
    delta[1:] = close[1:] - close[:-1]
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.mean(gains[1:period + 1])
    avg_loss = np.mean(losses[1:period + 1])

    if avg_loss == 0.0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - 100.0 / (1.0 + rs)

    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0.0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


def _compute_vwap(close: np.ndarray, volume: np.ndarray, day_id: np.ndarray) -> np.ndarray:
    """Cumulative session VWAP, reset each trading day."""
    n = len(close)
    vwap = np.full(n, np.nan)
    cum_pv = 0.0
    cum_vol = 0.0
    current_day = -999

    for i in range(n):
        if day_id[i] != current_day:
            current_day = day_id[i]
            cum_pv = 0.0
            cum_vol = 0.0
        cum_pv += close[i] * volume[i]
        cum_vol += volume[i]
        if cum_vol > 0.0:
            vwap[i] = cum_pv / cum_vol

    return vwap


def _compute_divergence_windows(
    close: np.ndarray,
    rsi: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each bar i, compute:
      - recent window [i-W, i): price min/max and RSI at those extremes
      - prior window [i-2W, i-W): price min/max and RSI at those extremes

    Returns 8 arrays (all length n):
      recent_min_price, prior_min_price, recent_min_rsi, prior_min_rsi,
      recent_max_price, prior_max_price, recent_max_rsi, prior_max_rsi
    """
    n = len(close)
    W = window
    recent_min_price = np.full(n, np.nan)
    prior_min_price = np.full(n, np.nan)
    recent_min_rsi = np.full(n, np.nan)
    prior_min_rsi = np.full(n, np.nan)
    recent_max_price = np.full(n, np.nan)
    prior_max_price = np.full(n, np.nan)
    recent_max_rsi = np.full(n, np.nan)
    prior_max_rsi = np.full(n, np.nan)

    for i in range(2 * W, n):
        rc = close[i - W:i]
        pc = close[i - 2 * W:i - W]
        rr = rsi[i - W:i]
        pr = rsi[i - 2 * W:i - W]

        rmin_idx = int(np.argmin(rc))
        pmin_idx = int(np.argmin(pc))
        recent_min_price[i] = rc[rmin_idx]
        prior_min_price[i] = pc[pmin_idx]
        recent_min_rsi[i] = rr[rmin_idx]
        prior_min_rsi[i] = pr[pmin_idx]

        rmax_idx = int(np.argmax(rc))
        pmax_idx = int(np.argmax(pc))
        recent_max_price[i] = rc[rmax_idx]
        prior_max_price[i] = pc[pmax_idx]
        recent_max_rsi[i] = rr[rmax_idx]
        prior_max_rsi[i] = pr[pmax_idx]

    return (
        recent_min_price, prior_min_price, recent_min_rsi, prior_min_rsi,
        recent_max_price, prior_max_price, recent_max_rsi, prior_max_rsi,
    )


class Strategy(BaseStrategy):
    """RSI divergence (2-min micro-windows) + VWAP proximity on NIFTY 5s bars.

    Bullish: NIFTY makes lower price low but RSI makes higher low (absorption),
    price just below VWAP (-0.15% to -0.01%) → buy CE for 30-90s bounce.
    Bearish: NIFTY makes higher price high but RSI makes lower high (exhaustion),
    price just above VWAP (+0.01% to +0.15%) → buy PE for 30-90s fade.
    """

    name = "rsi_divergence_vwap_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — skip opening 15 min for VWAP stability
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 5
    max_lookback = 72             # 6 min warmup: RSI(24) + 2 × 12-bar divergence windows

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rsi_low_threshold", 45.0, 35.0, 55.0),
            TunableParam("rsi_high_threshold", 55.0, 45.0, 65.0),
            TunableParam("vwap_dist_max_pct", 0.15, 0.05, 0.40),
            TunableParam("vix_threshold", 22.0, 15.0, 30.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill before numpy conversion
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        day_id = spot_df["day_id"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        rsi_low_thresh = params.get("rsi_low_threshold", 45.0)
        rsi_high_thresh = params.get("rsi_high_threshold", 55.0)
        vwap_dist_max = params.get("vwap_dist_max_pct", 0.15) / 100.0
        vix_thresh = params.get("vix_threshold", 22.0)

        # VIX aligned to spot bars
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # RSI(24) — 2-minute RSI at 5s bars
        rsi = _compute_rsi(close, 24)

        # Session VWAP — cumulative, reset per day
        vwap = _compute_vwap(close, volume, day_id)
        # Replace NaN VWAP with close price (avoids false distance signals at session open)
        vwap = np.where(np.isnan(vwap), close, vwap)

        # Divergence windows: W=12 bars (60s recent), W=12 bars (60s prior)
        W = 12
        (
            recent_min_price, prior_min_price, recent_min_rsi, prior_min_rsi,
            recent_max_price, prior_max_price, recent_max_rsi, prior_max_rsi,
        ) = _compute_divergence_windows(close, rsi, W)

        # Replace NaN with neutral values for boolean comparisons
        recent_min_price = np.where(np.isnan(recent_min_price), close, recent_min_price)
        prior_min_price = np.where(np.isnan(prior_min_price), close, prior_min_price)
        recent_min_rsi = np.where(np.isnan(recent_min_rsi), 50.0, recent_min_rsi)
        prior_min_rsi = np.where(np.isnan(prior_min_rsi), 50.0, prior_min_rsi)
        recent_max_price = np.where(np.isnan(recent_max_price), close, recent_max_price)
        prior_max_price = np.where(np.isnan(prior_max_price), close, prior_max_price)
        recent_max_rsi = np.where(np.isnan(recent_max_rsi), 50.0, recent_max_rsi)
        prior_max_rsi = np.where(np.isnan(prior_max_rsi), 50.0, prior_max_rsi)

        # VWAP distance (fractional, signed)
        vwap_safe = np.where(vwap > 0.0, vwap, close)
        vwap_dist = (close - vwap_safe) / vwap_safe

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        vix_ok = vix_close < vix_thresh

        # Bullish RSI divergence: price lower low, RSI higher low
        bull_divergence = (
            (recent_min_price < prior_min_price)  # price lower low
            & (recent_min_rsi > prior_min_rsi)    # RSI higher low (momentum divergence)
        )

        # Bearish RSI divergence: price higher high, RSI lower high
        bear_divergence = (
            (recent_max_price > prior_max_price)  # price higher high
            & (recent_max_rsi < prior_max_rsi)    # RSI lower high (momentum divergence)
        )

        # buy CE: bullish divergence, price just below VWAP, RSI still below neutral
        buy_ce = (
            in_session
            & vix_ok
            & bull_divergence
            & (vwap_dist < -0.0001)           # below VWAP
            & (vwap_dist > -vwap_dist_max)    # not too far below
            & (rsi < rsi_low_thresh)
        )

        # buy PE: bearish divergence, price just above VWAP, RSI still above neutral
        buy_pe = (
            in_session
            & vix_ok
            & bear_divergence
            & (vwap_dist > 0.0001)            # above VWAP
            & (vwap_dist < vwap_dist_max)     # not too far above
            & (rsi > rsi_high_thresh)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, 4.0),
            target_points=np.full(n, 7.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,
            max_trades_per_day=self.max_trades_per_day,
        )
