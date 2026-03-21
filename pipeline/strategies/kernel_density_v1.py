"""kernel_density_v1 — KDE Support/Resistance Bounce on NIFTY

Mechanism:
    On NIFTY, institutional TWAP/VWAP algorithms executing multi-hour mandates
    repeatedly transact at the same price bands within any 10-minute window, creating
    high-density clusters in the Gaussian KDE of recent 5-second closes. These KDE
    modes act as mechanical support/resistance: when NIFTY approaches a high-density
    mode from above with RSI(3-min) < 40, residual buy orders absorb incoming sell
    flow and produce a 10-20 spot-point bounce. At 5-second resolution we enter as
    proximity tightens to within 0.1% of the mode.

Converted from: trading_strategies/unique_strategies_all/Strategy_331.json
Original: Gaussian KDE S/R on NIFTY50 constituents, 1-min bars, 10-30 min hold.
"""

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder RSI. Returns array of length n, filled with 50.0 during warmup."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi

    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)

    # Initial SMA seed
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    # Wilder smoothing
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period

    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


def _compute_kde_levels(
    close: np.ndarray,
    lookback: int = 120,
    n_grid: int = 60,
) -> tuple:
    """Compute Gaussian KDE modes on rolling `lookback` window for each bar.

    For each bar i >= lookback:
      - Fit KDE on close[i-lookback:i] with Silverman bandwidth
      - Find local maxima (modes) of density on a n_grid-point evaluation grid
      - Identify nearest mode below (support) and above (resistance) current close
      - Compute mode strength = density_at_mode / mean_density_over_grid

    Returns four arrays of length n (NaN/0.0 during warmup):
      nearest_support, nearest_resistance, support_strength, resistance_strength
    """
    n = len(close)
    nearest_support = np.full(n, np.nan)
    nearest_resistance = np.full(n, np.nan)
    support_strength = np.zeros(n)
    resistance_strength = np.zeros(n)

    sqrt2pi = np.sqrt(2.0 * np.pi)

    for i in range(lookback, n):
        window = close[i - lookback:i]
        std = np.std(window)
        if std < 0.5:
            # Pathological: NIFTY essentially flat — no meaningful KDE modes
            continue

        # Silverman's rule of thumb bandwidth
        h = 1.06 * std * lookback ** (-0.2)

        # Evaluation grid spanning [min - h, max + h]
        lo = window.min() - h
        hi = window.max() + h
        grid = np.linspace(lo, hi, n_grid)

        # Vectorized KDE: shape (n_grid, lookback)
        z = (grid[:, None] - window[None, :]) / h
        density = np.sum(np.exp(-0.5 * z * z), axis=1) / (lookback * h * sqrt2pi)

        # Local maxima = KDE modes
        is_mode = (density[1:-1] > density[:-2]) & (density[1:-1] > density[2:])
        mode_indices = np.where(is_mode)[0] + 1  # +1 for slice offset

        if len(mode_indices) == 0:
            continue

        mode_prices = grid[mode_indices]
        mode_dens = density[mode_indices]
        avg_dens = np.mean(density)

        cur = close[i]

        # Nearest support: highest KDE mode strictly below current price
        below_mask = mode_prices < cur
        if np.any(below_mask):
            below_prices = mode_prices[below_mask]
            below_dens = mode_dens[below_mask]
            idx = np.argmax(below_prices)  # highest = closest to cur from below
            nearest_support[i] = below_prices[idx]
            support_strength[i] = below_dens[idx] / (avg_dens + 1e-12)

        # Nearest resistance: lowest KDE mode strictly above current price
        above_mask = mode_prices > cur
        if np.any(above_mask):
            above_prices = mode_prices[above_mask]
            above_dens = mode_dens[above_mask]
            idx = np.argmin(above_prices)  # lowest = closest to cur from above
            nearest_resistance[i] = above_prices[idx]
            resistance_strength[i] = above_dens[idx] / (avg_dens + 1e-12)

    return nearest_support, nearest_resistance, support_strength, resistance_strength


class Strategy(BaseStrategy):
    """KDE Support/Resistance Bounce — NIFTY 5-second options.

    Buys ATM calls when NIFTY approaches a high-density KDE support mode from above
    with RSI(3-min) < 40. Buys ATM puts at resistance modes with RSI > 60.
    """

    name = "kernel_density_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — after 120-bar (10-min) warmup from 09:20
    session_end_minutes = 920     # 15:20 IST
    max_lookback = 120            # 10 min warmup (120 × 5s)
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            # How close (% of price) NIFTY must be to the KDE mode to trigger
            TunableParam("proximity_pct", 0.0010, 0.0003, 0.0025),
            # Minimum mode density relative to mean grid density (filters weak modes)
            TunableParam("strength_threshold", 1.5, 1.0, 3.0),
            # RSI(36) below this at support → confirming oversold bounce
            TunableParam("rsi_low", 40.0, 30.0, 50.0),
            # RSI(36) above this at resistance → confirming overbought rejection
            TunableParam("rsi_high", 60.0, 50.0, 70.0),
            # Max VIX for reliable S/R (above this, modes are overrun by momentum)
            TunableParam("vix_high", 22.0, 16.0, 28.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill NaN before numpy conversion
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        # Parameters
        proximity_pct = params.get("proximity_pct", 0.0010)
        strength_thresh = params.get("strength_threshold", 1.5)
        rsi_low = params.get("rsi_low", 40.0)
        rsi_high = params.get("rsi_high", 60.0)
        vix_ceil = params.get("vix_high", 22.0)

        # ── VIX filter ──────────────────────────────────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(
                    ["datetime", pl.col("close").alias("vix_close")]
                ).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── KDE support/resistance levels ────────────────────────────────────────
        sup, res, sup_str, res_str = _compute_kde_levels(
            close, lookback=120, n_grid=60
        )

        # ── RSI(36) = 3-minute momentum exhaustion ───────────────────────────────
        rsi = _rsi(close, period=36)

        # ── Session and VIX filters ──────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (
            time_min < self.session_end_minutes
        )
        vix_ok = vix_close < vix_ceil

        # ── Proximity ratios (fraction of current price) ─────────────────────────
        # proximity_to_support: (close - sup) / close  > 0 when close > sup
        # proximity_to_resistance: (res - close) / close  > 0 when res > close
        safe_close = np.where(close > 0, close, 1.0)

        prox_sup = np.where(
            np.isfinite(sup) & (sup > 0),
            (close - sup) / safe_close,
            np.inf,
        )
        prox_res = np.where(
            np.isfinite(res) & (res > 0),
            (res - close) / safe_close,
            np.inf,
        )

        # ── Entry signals ────────────────────────────────────────────────────────

        # Buy CE (bullish): NIFTY approaching KDE support from above, RSI oversold
        #   close > sup               → price still above support (not broken)
        #   prox_sup ∈ [0, pct)       → within proximity_pct of support
        #   sup_str > strength_thresh → high-density mode (not noise peak)
        #   rsi < rsi_low             → momentum exhausted into support
        buy_ce = (
            in_session
            & vix_ok
            & np.isfinite(sup)
            & (prox_sup >= 0.0)
            & (prox_sup < proximity_pct)
            & (sup_str > strength_thresh)
            & (rsi < rsi_low)
        )

        # Buy PE (bearish): NIFTY approaching KDE resistance from below, RSI overbought
        buy_pe = (
            in_session
            & vix_ok
            & np.isfinite(res)
            & (prox_res >= 0.0)
            & (prox_res < proximity_pct)
            & (res_str > strength_thresh)
            & (rsi > rsi_high)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            # Stop: 4 pts → ~8 NIFTY spot pts; KDE level confirmed failed if broken by this
            stop_points=np.full(n, 4.0),
            # Target: 6 pts → ~12 NIFTY spot pts; lower bound of expected mode bounce
            target_points=np.full(n, 6.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,        # 90 seconds — if no bounce in 90s, mode not holding
            max_trades_per_day=8,
        )
