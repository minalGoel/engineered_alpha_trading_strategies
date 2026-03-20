"""Range Day Band Fade — cursor_gpt54_range_day_band_fade_003

Thesis: Exploits overreaction inside intraday range regimes where price moves
too far from fair value without broad directional sponsorship.  Traders
mechanically chase local strength/weakness even when the tape is rotational.

Entry long: ADX(14) <= 20, flat VWAP slope, close <= VWAP*0.993, close > BB lower,
    MFI(5) <= 20, bullish bar.
Entry short: mirror with close >= VWAP*1.007, MFI(5) >= 80.
Target: VWAP (use_target_indicator).  Stop: 0.45%.  Time stop: 22 bars.
VIX: 12-24.  Rel volume >= 1.4.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_atr(high, low, close, period):
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    if n < period:
        return atr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _compute_adx(high, low, close, period):
    """ADX using Wilder's smoothing."""
    n = len(close)
    adx = np.zeros(n, dtype=np.float64)
    if n < period * 2:
        return adx

    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    # Wilder's smoothing for TR, +DM, -DM
    atr_s = np.zeros(n, dtype=np.float64)
    pdm_s = np.zeros(n, dtype=np.float64)
    mdm_s = np.zeros(n, dtype=np.float64)

    atr_s[period] = np.sum(tr[1:period + 1])
    pdm_s[period] = np.sum(plus_dm[1:period + 1])
    mdm_s[period] = np.sum(minus_dm[1:period + 1])

    for i in range(period + 1, n):
        atr_s[i] = atr_s[i - 1] - atr_s[i - 1] / period + tr[i]
        pdm_s[i] = pdm_s[i - 1] - pdm_s[i - 1] / period + plus_dm[i]
        mdm_s[i] = mdm_s[i - 1] - mdm_s[i - 1] / period + minus_dm[i]

    # DI+ and DI-
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    dx = np.zeros(n, dtype=np.float64)

    for i in range(period, n):
        if atr_s[i] > 0:
            plus_di[i] = 100.0 * pdm_s[i] / atr_s[i]
            minus_di[i] = 100.0 * mdm_s[i] / atr_s[i]
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum

    # ADX = Wilder smooth of DX
    start = 2 * period
    if start < n:
        adx[start] = np.mean(dx[period:start + 1])
        for i in range(start + 1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx


def _compute_mfi(high, low, close, volume, period):
    """Money Flow Index."""
    n = len(close)
    mfi = np.full(n, 50.0, dtype=np.float64)
    tp = (high + low + close) / 3.0
    mf = tp * volume

    for i in range(period, n):
        pos_mf = 0.0
        neg_mf = 0.0
        for j in range(i - period + 1, i + 1):
            if j > 0 and tp[j] > tp[j - 1]:
                pos_mf += mf[j]
            elif j > 0 and tp[j] < tp[j - 1]:
                neg_mf += mf[j]
        if neg_mf == 0:
            mfi[i] = 100.0
        else:
            ratio = pos_mf / neg_mf
            mfi[i] = 100.0 - 100.0 / (1.0 + ratio)
    return mfi


class Strategy(BaseStrategy):
    name = "range_day_band_fade_003"
    is_long_only = False
    session_start = 615   # 10:15
    session_end = 855     # 14:15
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("adx_max", default=20.0, low=15.0, high=25.0),
            TunableParam("vwap_dev_long", default=0.007, low=0.004, high=0.012),
            TunableParam("vwap_dev_short", default=0.007, low=0.004, high=0.012),
            TunableParam("mfi_long_thresh", default=20.0, low=10.0, high=30.0),
            TunableParam("mfi_short_thresh", default=80.0, low=70.0, high=90.0),
            TunableParam("rel_vol_min", default=1.4, low=1.0, high=2.0),
            TunableParam("stop_loss_pct", default=0.0045, low=0.003, high=0.007),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        adx_max = params.get("adx_max", 20.0)
        vwap_dev_l = params.get("vwap_dev_long", 0.007)
        vwap_dev_s = params.get("vwap_dev_short", 0.007)
        mfi_l = params.get("mfi_long_thresh", 20.0)
        mfi_s = params.get("mfi_short_thresh", 80.0)
        rel_vol_min = params.get("rel_vol_min", 1.4)
        stop_pct = params.get("stop_loss_pct", 0.0045)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        day_ids = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)

        atr = _compute_atr(high, low, close, 14)
        adx = _compute_adx(high, low, close, 14)

        # ── VWAP ──
        vwap_df = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume")).cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = vwap_df["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── VWAP slope 10 bars ──
        vwap_slope_10 = np.zeros(n, dtype=np.float64)
        for i in range(10, n):
            vwap_slope_10[i] = vwap[i] - vwap[i - 10]

        # ── Bollinger Bands (SMA 20, 2.2 std) ──
        close_sma20 = df["close"].rolling_mean(20).to_numpy().astype(np.float64)
        close_sma20 = np.nan_to_num(close_sma20, nan=0.0)
        close_std20 = df["close"].rolling_std(20).to_numpy().astype(np.float64)
        close_std20 = np.nan_to_num(close_std20, nan=0.0)
        bb_upper = close_sma20 + 2.2 * close_std20
        bb_lower = close_sma20 - 2.2 * close_std20

        # ── MFI(5) ──
        mfi = _compute_mfi(high, low, close, volume, 5)

        # ── Relative volume ──
        vol_sma20 = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        vol_sma20 = np.nan_to_num(vol_sma20, nan=1.0)
        vol_sma20 = np.clip(vol_sma20, 1.0, None)
        rel_vol = volume / vol_sma20

        # ── OR range filter ──
        # AUDIT FIX: The original threshold of 1.8 * ATR(14) was calibrated for a
        # 1-min ATR, but the opening range covers ~15 bars.  A 15-bar range is
        # naturally ~sqrt(15) ≈ 3.9x the 1-min ATR under random-walk assumptions.
        # Using 1.8 excluded every single day.  Corrected to 6.0 * ATR(14) which
        # only filters genuinely extreme gap/spike open days (>6σ equivalent).
        unique_days = np.unique(day_ids)
        or_range_ok = np.ones(n, dtype=np.bool_)
        for d in unique_days:
            idx = np.where(day_ids == d)[0]
            if len(idx) == 0:
                continue
            or_bars = idx[time_mins[idx] < 570]
            if len(or_bars) == 0:
                continue
            or_rng = np.max(high[or_bars]) - np.min(low[or_bars])
            atr_val = atr[or_bars[-1]] if atr[or_bars[-1]] > 0 else atr[idx[-1]]
            if atr_val > 0 and or_rng > 6.0 * atr_val:
                or_range_ok[idx] = False

        # ── Time filter: 10:15-14:15 (615-855) ──
        time_ok = (time_mins >= 615) & (time_mins <= 855)

        # ── VIX filter: 12-24 ──
        vix_ok = (vix >= 12.0) & (vix <= 24.0)

        flat_vwap = np.abs(vwap_slope_10) <= 0.08 * atr
        bullish = close > open_
        bearish = close < open_

        long_entry = (
            (adx <= adx_max) &
            flat_vwap &
            (close <= vwap * (1 - vwap_dev_l)) &
            (close > bb_lower) &
            (mfi <= mfi_l) &
            bullish &
            (rel_vol >= rel_vol_min) &
            time_ok &
            vix_ok &
            or_range_ok
        )

        short_entry = (
            (adx <= adx_max) &
            flat_vwap &
            (close >= vwap * (1 + vwap_dev_s)) &
            (close < bb_upper) &
            (mfi >= mfi_s) &
            bearish &
            (rel_vol >= rel_vol_min) &
            time_ok &
            vix_ok &
            or_range_ok
        )

        # ── Signal exit: ADX rises above 24 OR close crosses VWAP ──
        sig_exit_long = (adx > 24.0) | ((close >= vwap) & (vwap > 0))
        sig_exit_short = (adx > 24.0) | ((close <= vwap) & (vwap > 0))

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=vwap,
            stop_loss_pct=stop_pct,
            use_target_indicator=True,
            time_stop_bars=22,
        )
