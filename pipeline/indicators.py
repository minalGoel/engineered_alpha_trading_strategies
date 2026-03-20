"""Technical indicator computation engine.

All indicators are computed on Polars DataFrames.
VWAP is computed with daily reset using day_id.
Indicators are computed ONCE per strategy (not per Optuna trial).
"""
from __future__ import annotations
import polars as pl
import numpy as np
import re
import logging
from typing import Optional

log = logging.getLogger(__name__)

# ── Registry of indicator functions ─────────────────────────────────────────
# Each function takes (df, **params) and returns df with new column(s).


def compute_sma(df: pl.DataFrame, col: str, period: int, name: str) -> pl.DataFrame:
    """Simple Moving Average."""
    return df.with_columns(
        pl.col(col).rolling_mean(window_size=period).alias(name)
    )


def compute_ema(df: pl.DataFrame, col: str, period: int, name: str) -> pl.DataFrame:
    """Exponential Moving Average."""
    return df.with_columns(
        pl.col(col).ewm_mean(span=period, adjust=False).alias(name)
    )


def compute_rsi(df: pl.DataFrame, col: str, period: int, name: str) -> pl.DataFrame:
    """Relative Strength Index."""
    delta = pl.col(col).diff()
    gain = delta.clip(lower_bound=0)
    loss = (-delta).clip(lower_bound=0)
    avg_gain = gain.ewm_mean(span=period, adjust=False)
    avg_loss = loss.ewm_mean(span=period, adjust=False)
    rs = avg_gain / avg_loss
    rsi = pl.lit(100.0) - (pl.lit(100.0) / (pl.lit(1.0) + rs))
    return df.with_columns(rsi.alias(name))


def compute_atr(df: pl.DataFrame, period: int, name: str) -> pl.DataFrame:
    """Average True Range."""
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - pl.col("close").shift(1)).abs(),
        (pl.col("low") - pl.col("close").shift(1)).abs(),
    )
    return df.with_columns(
        tr.ewm_mean(span=period, adjust=False).alias(name)
    )


def compute_bollinger(df: pl.DataFrame, col: str, period: int, std_mult: float,
                      name_upper: str, name_lower: str, name_mid: str) -> pl.DataFrame:
    """Bollinger Bands."""
    mid = pl.col(col).rolling_mean(window_size=period)
    std = pl.col(col).rolling_std(window_size=period)
    return df.with_columns(
        mid.alias(name_mid),
        (mid + std_mult * std).alias(name_upper),
        (mid - std_mult * std).alias(name_lower),
    )


def compute_macd(df: pl.DataFrame, col: str, fast: int, slow: int, signal: int,
                 name_macd: str, name_signal: str, name_hist: str) -> pl.DataFrame:
    """MACD indicator."""
    ema_fast = pl.col(col).ewm_mean(span=fast, adjust=False)
    ema_slow = pl.col(col).ewm_mean(span=slow, adjust=False)
    macd_line = ema_fast - ema_slow
    df = df.with_columns(macd_line.alias(name_macd))
    df = df.with_columns(
        pl.col(name_macd).ewm_mean(span=signal, adjust=False).alias(name_signal)
    )
    df = df.with_columns(
        (pl.col(name_macd) - pl.col(name_signal)).alias(name_hist)
    )
    return df


def compute_adx(df: pl.DataFrame, period: int, name: str) -> pl.DataFrame:
    """Average Directional Index."""
    # +DM / -DM
    high_diff = pl.col("high") - pl.col("high").shift(1)
    low_diff = pl.col("low").shift(1) - pl.col("low")

    plus_dm = pl.when((high_diff > low_diff) & (high_diff > 0)).then(high_diff).otherwise(0.0)
    minus_dm = pl.when((low_diff > high_diff) & (low_diff > 0)).then(low_diff).otherwise(0.0)

    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - pl.col("close").shift(1)).abs(),
        (pl.col("low") - pl.col("close").shift(1)).abs(),
    )

    df = df.with_columns(
        tr.ewm_mean(span=period, adjust=False).alias("_atr_adx"),
        plus_dm.ewm_mean(span=period, adjust=False).alias("_plus_dm_smooth"),
        minus_dm.ewm_mean(span=period, adjust=False).alias("_minus_dm_smooth"),
    )
    df = df.with_columns(
        (pl.col("_plus_dm_smooth") / pl.col("_atr_adx") * 100).alias("_plus_di"),
        (pl.col("_minus_dm_smooth") / pl.col("_atr_adx") * 100).alias("_minus_di"),
    )
    df = df.with_columns(
        ((pl.col("_plus_di") - pl.col("_minus_di")).abs()
         / (pl.col("_plus_di") + pl.col("_minus_di")) * 100)
        .ewm_mean(span=period, adjust=False)
        .alias(name)
    )
    df = df.drop(["_atr_adx", "_plus_dm_smooth", "_minus_dm_smooth", "_plus_di", "_minus_di"])
    return df


def compute_vwap(df: pl.DataFrame, name: str = "vwap") -> pl.DataFrame:
    """VWAP with daily reset using day_id column.

    typical_price = (high + low + close) / 3
    VWAP = cumsum(tp * volume) / cumsum(volume), reset each day.
    """
    tp = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
    tp_vol = tp * pl.col("volume")

    df = df.with_columns(
        tp_vol.alias("_tp_vol"),
    )
    # Cumulative sums within each day
    df = df.with_columns(
        pl.col("_tp_vol").cum_sum().over("day_id").alias("_cum_tp_vol"),
        pl.col("volume").cum_sum().over("day_id").alias("_cum_vol"),
    )
    df = df.with_columns(
        pl.when(pl.col("_cum_vol") > 0)
        .then(pl.col("_cum_tp_vol") / pl.col("_cum_vol"))
        .otherwise(pl.col("close"))
        .alias(name)
    )
    df = df.drop(["_tp_vol", "_cum_tp_vol", "_cum_vol"])
    return df


def compute_vwap_bands(df: pl.DataFrame, period: int, std_mult: float,
                       name_upper: str, name_lower: str) -> pl.DataFrame:
    """VWAP standard deviation bands."""
    if "vwap" not in df.columns:
        df = compute_vwap(df)
    dev = pl.col("close") - pl.col("vwap")
    std = dev.rolling_std(window_size=period)
    df = df.with_columns(
        (pl.col("vwap") + std_mult * std).alias(name_upper),
        (pl.col("vwap") - std_mult * std).alias(name_lower),
    )
    return df


def compute_zscore(df: pl.DataFrame, col: str, period: int, name: str) -> pl.DataFrame:
    """Rolling z-score: (value - mean) / std over period bars."""
    mean = pl.col(col).rolling_mean(window_size=period)
    std = pl.col(col).rolling_std(window_size=period)
    return df.with_columns(
        ((pl.col(col) - mean) / std).alias(name)
    )


def compute_rolling_max(df: pl.DataFrame, col: str, period: int, name: str) -> pl.DataFrame:
    """Rolling maximum."""
    return df.with_columns(
        pl.col(col).rolling_max(window_size=period).alias(name)
    )


def compute_rolling_min(df: pl.DataFrame, col: str, period: int, name: str) -> pl.DataFrame:
    """Rolling minimum."""
    return df.with_columns(
        pl.col(col).rolling_min(window_size=period).alias(name)
    )


def compute_rolling_std(df: pl.DataFrame, col: str, period: int, name: str) -> pl.DataFrame:
    """Rolling standard deviation."""
    return df.with_columns(
        pl.col(col).rolling_std(window_size=period).alias(name)
    )


def compute_stdev(df: pl.DataFrame, col: str, period: int, name: str) -> pl.DataFrame:
    """Alias for rolling std."""
    return compute_rolling_std(df, col, period, name)


def compute_obv(df: pl.DataFrame, name: str = "obv") -> pl.DataFrame:
    """On-Balance Volume."""
    sign = pl.when(pl.col("close") > pl.col("close").shift(1)).then(1) \
             .when(pl.col("close") < pl.col("close").shift(1)).then(-1) \
             .otherwise(0)
    return df.with_columns(
        (sign * pl.col("volume")).cum_sum().alias(name)
    )


def compute_obv_slope(df: pl.DataFrame, period: int, name: str) -> pl.DataFrame:
    """OBV slope via linear regression over period bars."""
    if "obv" not in df.columns:
        df = compute_obv(df)
    # Approximate slope using (OBV - OBV[period]) / period
    return df.with_columns(
        ((pl.col("obv") - pl.col("obv").shift(period)) / period).alias(name)
    )


def compute_prev_day_high_low(df: pl.DataFrame) -> pl.DataFrame:
    """Previous day high (PDH) and previous day low (PDL)."""
    # Get daily high/low per day_id
    daily = df.group_by("day_id").agg(
        pl.col("high").max().alias("_day_high"),
        pl.col("low").min().alias("_day_low"),
    ).sort("day_id")

    # Shift to get previous day
    daily = daily.with_columns(
        pl.col("_day_high").shift(1).alias("pdh"),
        pl.col("_day_low").shift(1).alias("pdl"),
    )

    df = df.join(daily.select(["day_id", "pdh", "pdl"]), on="day_id", how="left")
    return df


def compute_prev_day_close(df: pl.DataFrame) -> pl.DataFrame:
    """Previous day's closing price."""
    daily = df.group_by("day_id").agg(
        pl.col("close").last().alias("_day_close"),
    ).sort("day_id")
    daily = daily.with_columns(
        pl.col("_day_close").shift(1).alias("prev_close")
    )
    df = df.join(daily.select(["day_id", "prev_close"]), on="day_id", how="left")
    return df


def compute_gap_pct(df: pl.DataFrame, name: str = "gap_pct") -> pl.DataFrame:
    """Gap percentage: (today's open - prev close) / prev close."""
    if "prev_close" not in df.columns:
        df = compute_prev_day_close(df)
    # Get day's open
    day_open = df.group_by("day_id").agg(
        pl.col("open").first().alias("_day_open")
    )
    df = df.join(day_open, on="day_id", how="left")
    df = df.with_columns(
        ((pl.col("_day_open") - pl.col("prev_close")) / pl.col("prev_close")).alias(name)
    )
    df = df.drop("_day_open")
    return df


def compute_opening_range(df: pl.DataFrame, minutes: int = 15) -> pl.DataFrame:
    """Opening range high/low for first N minutes of each day."""
    # Mark bars within first N minutes from day start
    first_bar_per_day = df.group_by("day_id").agg(
        pl.col("datetime").min().alias("_day_start"),
    )
    df = df.join(first_bar_per_day, on="day_id", how="left")

    from datetime import timedelta
    df = df.with_columns(
        (pl.col("datetime") < (pl.col("_day_start") + timedelta(minutes=minutes))).alias("_in_or")
    )

    or_stats = df.filter(pl.col("_in_or")).group_by("day_id").agg(
        pl.col("high").max().alias("or_high"),
        pl.col("low").min().alias("or_low"),
    )
    df = df.join(or_stats, on="day_id", how="left")
    df = df.drop(["_day_start", "_in_or"])
    return df


def compute_index_return(df: pl.DataFrame, period: int, name: str) -> pl.DataFrame:
    """Index return over N bars."""
    if "index_close" not in df.columns:
        return df.with_columns(pl.lit(0.0).alias(name))
    return df.with_columns(
        (pl.col("index_close") / pl.col("index_close").shift(period) - 1.0).alias(name)
    )


def compute_bar_count_from_open(df: pl.DataFrame, name: str = "bar_count_from_open") -> pl.DataFrame:
    """Number of bars since market open (first bar of each day_id)."""
    return df.with_columns(
        (pl.lit(1).cum_sum().over("day_id") - 1).cast(pl.Int32).alias(name)
    )


def compute_roc(df: pl.DataFrame, col: str, period: int, name: str) -> pl.DataFrame:
    """Rate of Change: (current - N bars ago) / N bars ago * 100."""
    return df.with_columns(
        ((pl.col(col) / pl.col(col).shift(period) - 1.0) * 100.0).alias(name)
    )


def compute_session_high(df: pl.DataFrame, name: str = "session_high") -> pl.DataFrame:
    """Running session high — cumulative max of high within each day_id."""
    return df.with_columns(
        pl.col("high").cum_max().over("day_id").alias(name)
    )


def compute_session_low(df: pl.DataFrame, name: str = "session_low") -> pl.DataFrame:
    """Running session low — cumulative min of low within each day_id."""
    return df.with_columns(
        pl.col("low").cum_min().over("day_id").alias(name)
    )


def compute_cci(df: pl.DataFrame, period: int, name: str) -> pl.DataFrame:
    """Commodity Channel Index."""
    tp = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
    tp_mean = tp.rolling_mean(window_size=period)
    # Mean absolute deviation
    mad = (tp - tp_mean).abs().rolling_mean(window_size=period)
    return df.with_columns(
        ((tp - tp_mean) / (0.015 * mad)).alias(name)
    )


def compute_williams_r(df: pl.DataFrame, period: int, name: str) -> pl.DataFrame:
    """Williams %R oscillator."""
    highest = pl.col("high").rolling_max(window_size=period)
    lowest = pl.col("low").rolling_min(window_size=period)
    return df.with_columns(
        ((highest - pl.col("close")) / (highest - lowest) * -100.0).alias(name)
    )


def compute_donchian(df: pl.DataFrame, period: int,
                     name_upper: str, name_lower: str, name_mid: str) -> pl.DataFrame:
    """Donchian Channel."""
    return df.with_columns(
        pl.col("high").rolling_max(window_size=period).alias(name_upper),
        pl.col("low").rolling_min(window_size=period).alias(name_lower),
        ((pl.col("high").rolling_max(window_size=period)
          + pl.col("low").rolling_min(window_size=period)) / 2.0).alias(name_mid),
    )


def compute_keltner(df: pl.DataFrame, ema_period: int, atr_period: int, atr_mult: float,
                    name_upper: str, name_lower: str, name_mid: str) -> pl.DataFrame:
    """Keltner Channel."""
    # EMA midline
    mid = pl.col("close").ewm_mean(span=ema_period, adjust=False)
    # ATR
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - pl.col("close").shift(1)).abs(),
        (pl.col("low") - pl.col("close").shift(1)).abs(),
    )
    atr = tr.ewm_mean(span=atr_period, adjust=False)
    return df.with_columns(
        mid.alias(name_mid),
        (mid + atr_mult * atr).alias(name_upper),
        (mid - atr_mult * atr).alias(name_lower),
    )


def compute_day_open(df: pl.DataFrame, name: str = "open_today") -> pl.DataFrame:
    """Today's opening price (first bar open per day_id)."""
    day_open = df.group_by("day_id").agg(
        pl.col("open").first().alias(name)
    )
    return df.join(day_open, on="day_id", how="left")


def compute_morning_return(df: pl.DataFrame, name: str = "morning_return") -> pl.DataFrame:
    """Intraday return from today's open: (close - open_today) / open_today."""
    if "open_today" not in df.columns:
        df = compute_day_open(df)
    return df.with_columns(
        ((pl.col("close") - pl.col("open_today")) / pl.col("open_today")).alias(name)
    )


def compute_rel_volume(df: pl.DataFrame, period: int = 20, name: str = "rel_volume") -> pl.DataFrame:
    """Relative volume: volume / SMA(volume, period)."""
    return df.with_columns(
        (pl.col("volume") / pl.col("volume").rolling_mean(window_size=period)).alias(name)
    )


def compute_vol_spike(df: pl.DataFrame, period: int = 20, threshold: float = 2.0,
                      name: str = "vol_spike") -> pl.DataFrame:
    """Volume spike boolean: volume > threshold * SMA(volume, period)."""
    ratio = pl.col("volume") / pl.col("volume").rolling_mean(window_size=period)
    return df.with_columns(
        (ratio > threshold).cast(pl.Float64).alias(name)
    )


# ── Formula parser ──────────────────────────────────────────────────────────

# Regex patterns for formula parsing
_RE_SMA = re.compile(r"SMA\(\s*(\w+)\s*,\s*(\d+)\s*\)", re.IGNORECASE)
_RE_EMA = re.compile(r"EMA\(\s*(\w+)\s*,\s*(\d+)\s*\)", re.IGNORECASE)
_RE_RSI = re.compile(r"RSI\(\s*(\w+)\s*,\s*(\d+)\s*\)", re.IGNORECASE)
_RE_ATR = re.compile(r"ATR\(\s*(?:\w+\s*,\s*\w+\s*,\s*\w+\s*,\s*)?(\d+)\s*\)", re.IGNORECASE)
_RE_BB = re.compile(r"BB_(upper|lower|mid)\(\s*(\w+)\s*,\s*(\d+)\s*,\s*([\d.]+)\s*\)", re.IGNORECASE)
_RE_MACD = re.compile(r"MACD\(\s*(\w+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", re.IGNORECASE)
_RE_ADX = re.compile(r"ADX\(\s*(?:\w+\s*,\s*\w+\s*,\s*\w+\s*,\s*)?(\d+)\s*\)", re.IGNORECASE)
_RE_ZSCORE = re.compile(r"zscore\(\s*(.+?)\s*,\s*(\d+)\s*\)", re.IGNORECASE)
_RE_MAX = re.compile(r"(?:MAX|HIGHEST)\(\s*(\w+)\s*,\s*(\d+)\s*\)", re.IGNORECASE)
_RE_MIN = re.compile(r"(?:MIN|LOWEST)\(\s*(\w+)\s*,\s*(\d+)\s*\)", re.IGNORECASE)
_RE_STDEV = re.compile(r"(?:STDEV|STDDEV|STD|rolling_std)\(\s*(.+?)\s*,\s*(\d+)\s*\)", re.IGNORECASE)
_RE_RATIO = re.compile(r"^(\w+)\s*/\s*SMA\(\s*(\w+)\s*,\s*(\d+)\s*\)$", re.IGNORECASE)
_RE_SIMPLE_RATIO = re.compile(r"^(\w+)\s*/\s*(\w+)$")
_RE_DIFF = re.compile(r"^(\w+)\s*-\s*(\w+)$")
_RE_DIFF_NORM = re.compile(r"^\(\s*(\w+)\s*-\s*(\w+)\s*\)\s*/\s*(\w+)\s*(?:\*\s*([\d.]+))?$")
_RE_OBV = re.compile(r"OBV\(\s*(\w+)\s*,\s*(\w+)\s*\)", re.IGNORECASE)
_RE_OBV_SLOPE = re.compile(r"linear_regression_slope\(\s*OBV\(.+?\)\s*,\s*(\d+)\s*\)", re.IGNORECASE)
_RE_VWAP_SIMPLE = re.compile(r"^(?:VWAP|vwap)\(\s*\)$|^(?:VWAP|vwap)$", re.IGNORECASE)
_RE_RANGE_WIDTH = re.compile(r"^\(\s*(\w+)\s*-\s*(\w+)\s*\)\s*/\s*(\w+)\s*\*\s*([\d.]+)$")
_RE_VWAP_DEV = re.compile(r"^\(\s*close\s*-\s*(?:vwap|VWAP)(?:\(close\))?\s*\)\s*/\s*(?:vwap|VWAP)(?:\(close\))?\s*\*\s*([\d.]+)$", re.IGNORECASE)
_RE_VWAP_BANDS = re.compile(r"vwap\s*[+-]\s*([\d.]+)\s*\*\s*vwap_sd", re.IGNORECASE)
# New patterns
_RE_ROC = re.compile(r"ROC\(\s*(\w+)\s*,\s*(\d+)\s*\)", re.IGNORECASE)
_RE_CCI = re.compile(r"CCI\(\s*(\d+)\s*\)", re.IGNORECASE)
_RE_WILLIAMS = re.compile(r"(?:WILLIAMS_?R?|WR)\(\s*(\d+)\s*\)", re.IGNORECASE)
_RE_DONCHIAN = re.compile(r"DONCHIAN_(upper|lower|mid)\(\s*(\d+)\s*\)", re.IGNORECASE)
_RE_KELTNER = re.compile(r"KELTNER_(upper|lower|mid)\(\s*(\d+)\s*,\s*(\d+)\s*,\s*([\d.]+)\s*\)", re.IGNORECASE)
_RE_BB_FORMULA = re.compile(r"^SMA\(\s*(\w+)\s*,\s*(\d+)\s*\)\s*([+-])\s*([\d.]+)\s*\*\s*(?:STDEV|STDDEV|STD)\(\s*\w+\s*,\s*\d+\s*\)$", re.IGNORECASE)
_RE_VWAP_ZSCORE = re.compile(r"^\(\s*close\s*-\s*(?:vwap|VWAP)(?:\(\s*close\s*\))?\s*\)\s*/\s*(?:rolling_std|STDDEV|STDEV|STD)\(\s*(?:close\s*-\s*vwap|close)\s*,\s*(\d+)\s*\)$", re.IGNORECASE)
_RE_RUNNING_MAX = re.compile(r"running\s+max\(\s*(\w+)\s*\)", re.IGNORECASE)
_RE_RUNNING_MIN = re.compile(r"running\s+min\(\s*(\w+)\s*\)", re.IGNORECASE)
_RE_LAGGED_COL = re.compile(r"^(\w+)\[(?:t-)?(\d+)\]$")
_RE_LAGGED_RETURN = re.compile(r"^\(\s*(\w+)\s*-\s*(\w+)\[(?:t-)?(\d+)\]\s*\)\s*/\s*\2\[(?:t-)?\3\]\s*\*\s*([\d.]+)$")
_RE_CUMULATIVE_VWAP = re.compile(r"cumulative\s+from", re.IGNORECASE)
_RE_REF_DAILY = re.compile(r"REF\(\s*(\w+)\s*,\s*(\d+)\s*,\s*'day'\s*\)", re.IGNORECASE)
_RE_SIGN = re.compile(r"^sign\(\s*(.+?)\s*\)$", re.IGNORECASE)
_RE_LINREG_SLOPE = re.compile(r"(?:linear_regression_slope|linreg_slope)\(\s*(\w+)\s*,\s*(\d+)\s*\)", re.IGNORECASE)
_RE_DIFF_LAGGED = re.compile(r"^(\w+)\s*-\s*(\w+)\[(?:t-)?(\d+)\]$")
_RE_DIFF_LAGGED_NORM = re.compile(
    r"^\(\s*(\w+)\s*-\s*(\w+)\[(?:t-)?(\d+)\]\s*\)\s*/\s*\2\[(?:t-)?\3\]\s*\*\s*([\d.]+)$"
)
_RE_COMPOSITE_ZSCORE = re.compile(
    r"^\(\s*(\w+)\s*-\s*SMA\(\s*\1\s*,\s*(\d+)\s*\)\s*\)\s*/\s*(?:STDEV|STDDEV|STD)\(\s*\1\s*,\s*\2\s*\)$",
    re.IGNORECASE
)


def parse_indicator_formula(indicator: dict, existing_cols: set[str]) -> list[dict]:
    """Parse an indicator formula and return a list of computation steps.

    Each step is a dict: {"func": callable, "kwargs": {...}, "output_col": str}
    Returns empty list if unparseable.
    """
    name = indicator.get("name", "")
    formula = indicator.get("formula", "")
    # Fallback: many strategy JSONs use "params" instead of "formula"
    if not formula:
        formula = indicator.get("params", "")
    lookback = indicator.get("lookback", 0)
    if lookback is None:
        lookback = 0

    steps = []
    formula_stripped = formula.strip()

    # ── VWAP ────────────────────────────────────────────────────────────
    if _RE_VWAP_SIMPLE.match(formula_stripped) or name.lower() == "vwap":
        steps.append({"func": compute_vwap, "kwargs": {"name": name}, "output_col": name})
        return steps

    # ── SMA ──────────────────────────────────────────────────────────────
    m = _RE_SMA.match(formula_stripped)
    if m:
        col, period = m.group(1), int(m.group(2))
        steps.append({"func": compute_sma, "kwargs": {"col": col, "period": period, "name": name},
                      "output_col": name})
        return steps

    # ── EMA ──────────────────────────────────────────────────────────────
    m = _RE_EMA.match(formula_stripped)
    if m:
        col, period = m.group(1), int(m.group(2))
        steps.append({"func": compute_ema, "kwargs": {"col": col, "period": period, "name": name},
                      "output_col": name})
        return steps

    # ── RSI ──────────────────────────────────────────────────────────────
    m = _RE_RSI.match(formula_stripped)
    if m:
        col, period = m.group(1), int(m.group(2))
        steps.append({"func": compute_rsi, "kwargs": {"col": col, "period": period, "name": name},
                      "output_col": name})
        return steps

    # ── ATR ──────────────────────────────────────────────────────────────
    m = _RE_ATR.match(formula_stripped)
    if m:
        period = int(m.group(1))
        steps.append({"func": compute_atr, "kwargs": {"period": period, "name": name},
                      "output_col": name})
        return steps

    # ── ADX ──────────────────────────────────────────────────────────────
    m = _RE_ADX.match(formula_stripped)
    if m:
        period = int(m.group(1))
        steps.append({"func": compute_adx, "kwargs": {"period": period, "name": name},
                      "output_col": name})
        return steps

    # ── Bollinger Bands ──────────────────────────────────────────────────
    m = _RE_BB.match(formula_stripped)
    if m:
        band_type, col, period, std = m.group(1).lower(), m.group(2), int(m.group(3)), float(m.group(4))
        steps.append({"func": compute_bollinger,
                      "kwargs": {"col": col, "period": period, "std_mult": std,
                                 "name_upper": f"bb_upper_{period}_{std}",
                                 "name_lower": f"bb_lower_{period}_{std}",
                                 "name_mid": f"bb_mid_{period}"},
                      "output_col": name})
        return steps

    # ── MACD ─────────────────────────────────────────────────────────────
    m = _RE_MACD.match(formula_stripped)
    if m:
        col, fast, slow, signal = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        steps.append({"func": compute_macd,
                      "kwargs": {"col": col, "fast": fast, "slow": slow, "signal": signal,
                                 "name_macd": name, "name_signal": f"{name}_signal",
                                 "name_hist": f"{name}_hist"},
                      "output_col": name})
        return steps

    # ── Z-score ──────────────────────────────────────────────────────────
    m = _RE_ZSCORE.match(formula_stripped)
    if m:
        inner_expr, period = m.group(1).strip(), int(m.group(2))
        # Parse inner expression
        inner_m = _RE_DIFF.match(inner_expr)
        if inner_m:
            col_a, col_b = inner_m.group(1), inner_m.group(2)
            temp_col = f"_diff_{col_a}_{col_b}"
            steps.append({"func": "_diff", "kwargs": {"col_a": col_a, "col_b": col_b, "name": temp_col},
                          "output_col": temp_col})
            steps.append({"func": compute_zscore, "kwargs": {"col": temp_col, "period": period, "name": name},
                          "output_col": name})
        else:
            # zscore of a single column
            steps.append({"func": compute_zscore, "kwargs": {"col": inner_expr, "period": period, "name": name},
                          "output_col": name})
        return steps

    # ── Rolling max (MAX/HIGHEST) ────────────────────────────────────────
    m = _RE_MAX.match(formula_stripped)
    if m:
        col, period = m.group(1), int(m.group(2))
        steps.append({"func": compute_rolling_max, "kwargs": {"col": col, "period": period, "name": name},
                      "output_col": name})
        return steps

    # ── Rolling min (MIN/LOWEST) ─────────────────────────────────────────
    m = _RE_MIN.match(formula_stripped)
    if m:
        col, period = m.group(1), int(m.group(2))
        steps.append({"func": compute_rolling_min, "kwargs": {"col": col, "period": period, "name": name},
                      "output_col": name})
        return steps

    # ── STDEV / rolling_std ──────────────────────────────────────────────
    m = _RE_STDEV.match(formula_stripped)
    if m:
        inner_expr, period = m.group(1).strip(), int(m.group(2))
        inner_m = _RE_DIFF.match(inner_expr)
        if inner_m:
            col_a, col_b = inner_m.group(1), inner_m.group(2)
            temp_col = f"_diff_{col_a}_{col_b}"
            steps.append({"func": "_diff", "kwargs": {"col_a": col_a, "col_b": col_b, "name": temp_col},
                          "output_col": temp_col})
            steps.append({"func": compute_rolling_std,
                          "kwargs": {"col": temp_col, "period": period, "name": name},
                          "output_col": name})
        else:
            steps.append({"func": compute_rolling_std,
                          "kwargs": {"col": inner_expr, "period": period, "name": name},
                          "output_col": name})
        return steps

    # ── Ratio: col / SMA(col, period) ────────────────────────────────────
    m = _RE_RATIO.match(formula_stripped)
    if m:
        num_col, denom_col, period = m.group(1), m.group(2), int(m.group(3))
        sma_name = f"_sma_{denom_col}_{period}"
        steps.append({"func": compute_sma, "kwargs": {"col": denom_col, "period": period, "name": sma_name},
                      "output_col": sma_name})
        steps.append({"func": "_ratio", "kwargs": {"col_a": num_col, "col_b": sma_name, "name": name},
                      "output_col": name})
        return steps

    # ── VWAP deviation % ─────────────────────────────────────────────────
    if _RE_VWAP_DEV.match(formula_stripped):
        steps.append({"func": "_vwap_dev_pct", "kwargs": {"name": name}, "output_col": name})
        return steps

    # ── Normalized diff: (a - b) / c [* k] ──────────────────────────────
    m = _RE_DIFF_NORM.match(formula_stripped)
    if m:
        a, b, c = m.group(1), m.group(2), m.group(3)
        k = float(m.group(4)) if m.group(4) else 1.0
        steps.append({"func": "_diff_norm", "kwargs": {"a": a, "b": b, "c": c, "k": k, "name": name},
                      "output_col": name})
        return steps

    # ── Range width: (a - b) / c * k ────────────────────────────────────
    m = _RE_RANGE_WIDTH.match(formula_stripped)
    if m:
        a, b, c, k = m.group(1), m.group(2), m.group(3), float(m.group(4))
        steps.append({"func": "_diff_norm", "kwargs": {"a": a, "b": b, "c": c, "k": k, "name": name},
                      "output_col": name})
        return steps

    # ── Simple difference: a - b ─────────────────────────────────────────
    m = _RE_DIFF.match(formula_stripped)
    if m:
        col_a, col_b = m.group(1), m.group(2)
        steps.append({"func": "_diff", "kwargs": {"col_a": col_a, "col_b": col_b, "name": name},
                      "output_col": name})
        return steps

    # ── Simple ratio: a / b ──────────────────────────────────────────────
    m = _RE_SIMPLE_RATIO.match(formula_stripped)
    if m:
        col_a, col_b = m.group(1), m.group(2)
        steps.append({"func": "_ratio", "kwargs": {"col_a": col_a, "col_b": col_b, "name": name},
                      "output_col": name})
        return steps

    # ── OBV ──────────────────────────────────────────────────────────────
    m = _RE_OBV.match(formula_stripped)
    if m:
        steps.append({"func": compute_obv, "kwargs": {"name": name}, "output_col": name})
        return steps

    # ── OBV slope ────────────────────────────────────────────────────────
    m = _RE_OBV_SLOPE.match(formula_stripped)
    if m:
        period = int(m.group(1))
        steps.append({"func": compute_obv_slope, "kwargs": {"period": period, "name": name},
                      "output_col": name})
        return steps

    # ── VWAP bands ───────────────────────────────────────────────────────
    m = _RE_VWAP_BANDS.match(formula_stripped)
    if m:
        std_mult = float(m.group(1))
        if "+" in formula_stripped:
            steps.append({"func": "_vwap_band_upper",
                          "kwargs": {"std_mult": std_mult, "name": name},
                          "output_col": name})
        else:
            steps.append({"func": "_vwap_band_lower",
                          "kwargs": {"std_mult": std_mult, "name": name},
                          "output_col": name})
        return steps

    # ── ROC (Rate of Change) ────────────────────────────────────────────
    m = _RE_ROC.match(formula_stripped)
    if m:
        col, period = m.group(1), int(m.group(2))
        # Normalize INDEX → index_close
        if col.upper() == "INDEX":
            col = "index_close"
        steps.append({"func": compute_roc, "kwargs": {"col": col, "period": period, "name": name},
                      "output_col": name})
        return steps

    # ── CCI ──────────────────────────────────────────────────────────────
    m = _RE_CCI.match(formula_stripped)
    if m:
        period = int(m.group(1))
        steps.append({"func": compute_cci, "kwargs": {"period": period, "name": name},
                      "output_col": name})
        return steps

    # ── Williams %R ──────────────────────────────────────────────────────
    m = _RE_WILLIAMS.match(formula_stripped)
    if m:
        period = int(m.group(1))
        steps.append({"func": compute_williams_r, "kwargs": {"period": period, "name": name},
                      "output_col": name})
        return steps

    # ── VWAP z-score: (close - VWAP) / rolling_std(close - vwap, N) ────
    m = _RE_VWAP_ZSCORE.match(formula_stripped)
    if m:
        period = int(m.group(1))
        steps.append({"func": "_vwap_zscore", "kwargs": {"period": period, "name": name},
                      "output_col": name})
        return steps

    # ── VWAP deviation with variable multiplier ──────────────────────────
    m = _RE_VWAP_DEV.match(formula_stripped)
    if m:
        mult = float(m.group(1))
        steps.append({"func": "_vwap_dev_scaled", "kwargs": {"mult": mult, "name": name},
                      "output_col": name})
        return steps

    # ── Bollinger from SMA +/- k * STDEV formula ────────────────────────
    m = _RE_BB_FORMULA.match(formula_stripped)
    if m:
        col, period, sign, std_mult = m.group(1), int(m.group(2)), m.group(3), float(m.group(4))
        steps.append({"func": compute_bollinger,
                      "kwargs": {"col": col, "period": period, "std_mult": std_mult,
                                 "name_upper": f"bb_upper_{period}_{std_mult}",
                                 "name_lower": f"bb_lower_{period}_{std_mult}",
                                 "name_mid": f"bb_mid_{period}"},
                      "output_col": name})
        return steps

    # ── Running session max/min ──────────────────────────────────────────
    m = _RE_RUNNING_MAX.search(formula_stripped)
    if m:
        col = m.group(1)
        steps.append({"func": "_session_cum_max", "kwargs": {"col": col, "name": name},
                      "output_col": name})
        return steps

    m = _RE_RUNNING_MIN.search(formula_stripped)
    if m:
        col = m.group(1)
        steps.append({"func": "_session_cum_min", "kwargs": {"col": col, "name": name},
                      "output_col": name})
        return steps

    # ── Cumulative VWAP (narrative "cumulative from 09:15 IST") ─────────
    if _RE_CUMULATIVE_VWAP.search(formula_stripped):
        steps.append({"func": compute_vwap, "kwargs": {"name": name}, "output_col": name})
        return steps

    # ── REF(col, N, 'day') — previous day's value ───────────────────────
    m = _RE_REF_DAILY.match(formula_stripped)
    if m:
        col, shift = m.group(1).lower(), int(m.group(2))
        if col == "close" and shift == 1:
            steps.append({"func": compute_prev_day_close, "kwargs": {}, "output_col": "prev_close"})
            if name != "prev_close":
                steps.append({"func": "_alias", "kwargs": {"src": "prev_close", "name": name},
                              "output_col": name})
        elif col == "high" and shift == 1:
            steps.append({"func": compute_prev_day_high_low, "kwargs": {}, "output_col": "pdh"})
            if name != "pdh":
                steps.append({"func": "_alias", "kwargs": {"src": "pdh", "name": name},
                              "output_col": name})
        elif col == "low" and shift == 1:
            steps.append({"func": compute_prev_day_high_low, "kwargs": {}, "output_col": "pdl"})
            if name != "pdl":
                steps.append({"func": "_alias", "kwargs": {"src": "pdl", "name": name},
                              "output_col": name})
        return steps

    # ── linreg_slope / linear_regression_slope ──────────────────────────
    m = _RE_LINREG_SLOPE.match(formula_stripped)
    if m:
        col, period = m.group(1), int(m.group(2))
        steps.append({"func": "_col_slope", "kwargs": {"col": col, "period": period, "name": name},
                      "output_col": name})
        return steps

    # ── Composite z-score: (col - SMA(col, N)) / STDEV(col, N) ──────────
    m = _RE_COMPOSITE_ZSCORE.match(formula_stripped)
    if m:
        col, period = m.group(1), int(m.group(2))
        steps.append({"func": compute_zscore, "kwargs": {"col": col, "period": period, "name": name},
                      "output_col": name})
        return steps

    # ── Lagged diff: col - col[t-N] ──────────────────────────────────────
    m = _RE_DIFF_LAGGED.match(formula_stripped)
    if m:
        col_a, col_b, lag = m.group(1), m.group(2), int(m.group(3))
        if col_a == col_b:
            steps.append({"func": "_diff_shift", "kwargs": {"col": col_a, "period": lag, "name": name},
                          "output_col": name})
            return steps

    # ── Lagged return: (col - col[t-N]) / col[t-N] * K ──────────────────
    m = _RE_DIFF_LAGGED_NORM.match(formula_stripped)
    if m:
        col, _, lag, mult = m.group(1), m.group(2), int(m.group(3)), float(m.group(4))
        steps.append({"func": "_lagged_return", "kwargs": {"col": col, "lag": lag, "mult": mult, "name": name},
                      "output_col": name})
        return steps

    # ── sign(expr) ───────────────────────────────────────────────────────
    m = _RE_SIGN.match(formula_stripped)
    if m:
        inner = m.group(1).strip()
        # sign(close - open) or sign(gap_pct)
        inner_diff = _RE_DIFF.match(inner)
        if inner_diff:
            col_a, col_b = inner_diff.group(1), inner_diff.group(2)
            steps.append({"func": "_sign_diff", "kwargs": {"col_a": col_a, "col_b": col_b, "name": name},
                          "output_col": name})
        else:
            steps.append({"func": "_sign", "kwargs": {"col": inner, "name": name},
                          "output_col": name})
        return steps

    # ── Lagged return: (col - col[t-N]) / col[t-N] * K ──────────────────
    m = _RE_LAGGED_RETURN.match(formula_stripped)
    if m:
        col, _, lag, mult = m.group(1), m.group(2), int(m.group(3)), float(m.group(4))
        steps.append({"func": "_lagged_return", "kwargs": {"col": col, "lag": lag, "mult": mult, "name": name},
                      "output_col": name})
        return steps

    # ── Name-based inference (last resort before giving up) ──────────────
    name_lower = name.lower()

    # ATR with period in the name: "ATR_14", "atr_10", "ATR_14d"
    m = re.match(r"^atr[_]?(\d+)d?$", name_lower)
    if m:
        period = int(m.group(1))
        steps.append({"func": compute_atr, "kwargs": {"period": period, "name": name},
                      "output_col": name})
        return steps

    # EMA with period in the name: "EMA_9", "ema_21"
    m = re.match(r"^ema[_]?(\d+)$", name_lower)
    if m:
        period = int(m.group(1))
        steps.append({"func": compute_ema, "kwargs": {"col": "close", "period": period, "name": name},
                      "output_col": name})
        return steps

    # SMA with period in the name: "SMA_20", "sma_50"
    m = re.match(r"^sma[_]?(\d+)$", name_lower)
    if m:
        period = int(m.group(1))
        steps.append({"func": compute_sma, "kwargs": {"col": "close", "period": period, "name": name},
                      "output_col": name})
        return steps

    # RSI with period in the name: "RSI_14", "rsi_14"
    m = re.match(r"^rsi[_]?(\d+)$", name_lower)
    if m:
        period = int(m.group(1))
        steps.append({"func": compute_rsi, "kwargs": {"col": "close", "period": period, "name": name},
                      "output_col": name})
        return steps

    # ADX with period in the name: "ADX_14"
    m = re.match(r"^adx[_]?(\d+)$", name_lower)
    if m:
        period = int(m.group(1))
        steps.append({"func": compute_adx, "kwargs": {"period": period, "name": name},
                      "output_col": name})
        return steps

    # BB_upper / BB_lower / BB_mid (default params)
    if name_lower in ("bb_upper", "bb_lower", "bb_mid"):
        period, std = 20, 2.0
        steps.append({"func": compute_bollinger,
                      "kwargs": {"col": "close", "period": period, "std_mult": std,
                                 "name_upper": "BB_upper", "name_lower": "BB_lower", "name_mid": "BB_mid"},
                      "output_col": name})
        return steps

    # volume_ratio / rel_volume
    if name_lower in ("volume_ratio", "rel_volume", "rel_volume_20"):
        period = 20
        steps.append({"func": compute_rel_volume, "kwargs": {"period": period, "name": name},
                      "output_col": name})
        return steps

    # vol_spike
    if name_lower == "vol_spike":
        steps.append({"func": compute_vol_spike, "kwargs": {"name": name},
                      "output_col": name})
        return steps

    # session_high / session_low
    if name_lower == "session_high":
        steps.append({"func": compute_session_high, "kwargs": {"name": name}, "output_col": name})
        return steps
    if name_lower == "session_low":
        steps.append({"func": compute_session_low, "kwargs": {"name": name}, "output_col": name})
        return steps

    # open_today / morning_return / stock_return
    if name_lower in ("open_today", "day_open"):
        steps.append({"func": compute_day_open, "kwargs": {"name": name}, "output_col": name})
        return steps
    if name_lower in ("morning_return", "stock_return", "intraday_return"):
        steps.append({"func": compute_morning_return, "kwargs": {"name": name}, "output_col": name})
        return steps

    # VWAP-related names
    if name_lower in ("vwap_dev_pct", "vwap_deviation", "vwap_distance"):
        steps.append({"func": "_vwap_dev_pct", "kwargs": {"name": name}, "output_col": name})
        return steps
    if name_lower in ("vwap_zscore", "vwap_z", "deviation_zscore"):
        steps.append({"func": "_vwap_zscore", "kwargs": {"period": 60, "name": name}, "output_col": name})
        return steps

    # Index return
    m = re.match(r"^index_return[_]?(\d+)?$", name_lower)
    if m:
        period = int(m.group(1)) if m.group(1) else 15
        steps.append({"func": compute_index_return, "kwargs": {"period": period, "name": name},
                      "output_col": name})
        return steps

    # gap_pct
    if name_lower == "gap_pct":
        steps.append({"func": compute_gap_pct, "kwargs": {"name": name}, "output_col": name})
        return steps

    # india_vix / vix (column alias — available from data merge)
    if name_lower in ("india_vix", "vix", "vix_level"):
        # Already available from merge_vix_index as 'vix'; just alias
        steps.append({"func": "_alias", "kwargs": {"src": "vix", "name": name}, "output_col": name})
        return steps

    # pdh / pdl — previous day high/low (structural, computed on demand)
    if name_lower in ("pdh", "prev_day_high", "previous_day_high"):
        steps.append({"func": compute_prev_day_high_low, "kwargs": {}, "output_col": "pdh"})
        if name != "pdh":
            steps.append({"func": "_alias", "kwargs": {"src": "pdh", "name": name}, "output_col": name})
        return steps
    if name_lower in ("pdl", "prev_day_low", "previous_day_low"):
        steps.append({"func": compute_prev_day_high_low, "kwargs": {}, "output_col": "pdl"})
        if name != "pdl":
            steps.append({"func": "_alias", "kwargs": {"src": "pdl", "name": name}, "output_col": name})
        return steps

    # prev_close
    if name_lower in ("prev_close", "previous_close", "prev_day_close"):
        steps.append({"func": compute_prev_day_close, "kwargs": {}, "output_col": "prev_close"})
        if name != "prev_close":
            steps.append({"func": "_alias", "kwargs": {"src": "prev_close", "name": name}, "output_col": name})
        return steps

    # or_high / or_low / orb_high / orb_low / orb_high_15 / orb_low_15
    if name_lower in ("or_high", "orb_high", "orb_high_15", "opening_range_high", "first_30_high"):
        minutes = 30 if "30" in name_lower else 15
        steps.append({"func": compute_opening_range, "kwargs": {"minutes": minutes}, "output_col": "or_high"})
        if name != "or_high":
            steps.append({"func": "_alias", "kwargs": {"src": "or_high", "name": name}, "output_col": name})
        return steps
    if name_lower in ("or_low", "orb_low", "orb_low_15", "opening_range_low", "first_30_low"):
        minutes = 30 if "30" in name_lower else 15
        steps.append({"func": compute_opening_range, "kwargs": {"minutes": minutes}, "output_col": "or_low"})
        if name != "or_low":
            steps.append({"func": "_alias", "kwargs": {"src": "or_low", "name": name}, "output_col": name})
        return steps

    # bb_width — Bollinger Band width
    if name_lower in ("bb_width", "bbwidth"):
        steps.append({"func": "_bb_width", "kwargs": {"name": name}, "output_col": name})
        return steps

    # OBV_slope / obv_slope
    m = re.match(r"^obv_?slope[_]?(\d+)?$", name_lower)
    if m:
        period = int(m.group(1)) if m.group(1) else 20
        steps.append({"func": compute_obv_slope, "kwargs": {"period": period, "name": name},
                      "output_col": name})
        return steps

    # vwap_slope
    if name_lower in ("vwap_slope",):
        steps.append({"func": "_col_slope", "kwargs": {"col": "vwap", "period": 20, "name": name},
                      "output_col": name})
        return steps

    # momentum_N
    m = re.match(r"^momentum[_]?(\d+)$", name_lower)
    if m:
        period = int(m.group(1))
        steps.append({"func": "_diff_shift", "kwargs": {"col": "close", "period": period, "name": name},
                      "output_col": name})
        return steps

    # nifty_vwap / nifty_VWAP
    if name_lower in ("nifty_vwap",):
        # Approximate with index_close (VWAP not available for index)
        steps.append({"func": "_alias", "kwargs": {"src": "index_close", "name": name}, "output_col": name})
        return steps

    # stock_deviation — typically (close - vwap) / vwap  or (close - ema) / ema
    if name_lower in ("stock_deviation",):
        steps.append({"func": "_vwap_dev_pct", "kwargs": {"name": name}, "output_col": name})
        return steps

    # price_above_vwap / vwap_touch — boolean indicators
    if name_lower in ("price_above_vwap",):
        steps.append({"func": "_bool_above", "kwargs": {"col": "close", "ref": "vwap", "name": name},
                      "output_col": name})
        return steps

    # bar_direction: sign(close - open)
    if name_lower in ("bar_direction",):
        steps.append({"func": "_sign_diff", "kwargs": {"col_a": "close", "col_b": "open", "name": name},
                      "output_col": name})
        return steps

    # range_position: (close - session_low) / (session_high - session_low)
    if name_lower in ("range_position",):
        steps.append({"func": "_range_position", "kwargs": {"name": name}, "output_col": name})
        return steps

    # BB_pct_b: (close - BB_lower) / (BB_upper - BB_lower)
    if name_lower in ("bb_pct_b", "bb_pctb"):
        steps.append({"func": "_bb_pct_b", "kwargs": {"name": name}, "output_col": name})
        return steps

    # TR (True Range)
    if name_lower in ("tr", "true_range"):
        steps.append({"func": "_true_range", "kwargs": {"name": name}, "output_col": name})
        return steps

    # RSI_slope: RSI_14[t] - RSI_14[t-N]
    m = re.match(r"^(\w+)_slope$", name_lower)
    if m and m.group(1) not in ("obv", "vwap", "col"):
        base = m.group(1)
        steps.append({"func": "_diff_shift", "kwargs": {"col": base, "period": 3, "name": name},
                      "output_col": name})
        return steps

    # hist_accel: MACD_hist[t] - MACD_hist[t-1]
    if name_lower in ("hist_accel", "macd_accel"):
        steps.append({"func": "_diff_shift", "kwargs": {"col": "macd_hist", "period": 1, "name": name},
                      "output_col": name})
        return steps

    # donchian_mid: (donchian_high_N + donchian_low_N) / 2
    m = re.match(r"^donchian_mid(?:_(\d+))?$", name_lower)
    if m:
        period = int(m.group(1)) if m.group(1) else 20
        steps.append({"func": compute_donchian,
                      "kwargs": {"period": period,
                                 "name_upper": f"donchian_high_{period}",
                                 "name_lower": f"donchian_low_{period}",
                                 "name_mid": name},
                      "output_col": name})
        return steps
    m = re.match(r"^donchian_(high|low)(?:_(\d+))?$", name_lower)
    if m:
        band = m.group(1)
        period = int(m.group(2)) if m.group(2) else 20
        steps.append({"func": compute_donchian,
                      "kwargs": {"period": period,
                                 "name_upper": f"donchian_high_{period}",
                                 "name_lower": f"donchian_low_{period}",
                                 "name_mid": f"donchian_mid_{period}"},
                      "output_col": name})
        return steps

    # plus_DI / minus_DI — part of ADX computation
    m = re.match(r"^(plus|minus)_?di[_]?(\d+)?$", name_lower)
    if m:
        sign = m.group(1)
        period = int(m.group(2)) if m.group(2) else 14
        steps.append({"func": "_directional_indicator",
                      "kwargs": {"sign": sign, "period": period, "name": name},
                      "output_col": name})
        return steps

    # VWAP_band_upper / VWAP_band_lower / vwap_upper / vwap_lower
    if name_lower in ("vwap_band_upper", "vwap_upper_band", "vwap_upper"):
        steps.append({"func": "_vwap_band_upper", "kwargs": {"std_mult": 2.0, "name": name},
                      "output_col": name})
        return steps
    if name_lower in ("vwap_band_lower", "vwap_lower_band", "vwap_lower"):
        steps.append({"func": "_vwap_band_lower", "kwargs": {"std_mult": 2.0, "name": name},
                      "output_col": name})
        return steps

    # bars_above_vwap / bars_below_vwap / above_vwap_count / below_vwap_count
    if name_lower in ("bars_above_vwap", "above_vwap_count"):
        steps.append({"func": "_cum_count_above", "kwargs": {"col": "close", "ref": "vwap", "name": name},
                      "output_col": name})
        return steps
    if name_lower in ("bars_below_vwap", "below_vwap_count"):
        steps.append({"func": "_cum_count_below", "kwargs": {"col": "close", "ref": "vwap", "name": name},
                      "output_col": name})
        return steps

    # stoch_K / stochastic
    m = re.match(r"^stoch(?:astic)?_?k?(?:_(\d+))?$", name_lower)
    if m:
        period = int(m.group(1)) if m.group(1) else 14
        steps.append({"func": "_stochastic_k", "kwargs": {"period": period, "name": name},
                      "output_col": name})
        return steps

    # williams_r
    m = re.match(r"^williams_?r?(?:_(\d+))?$", name_lower)
    if m:
        period = int(m.group(1)) if m.group(1) else 14
        steps.append({"func": compute_williams_r, "kwargs": {"period": period, "name": name},
                      "output_col": name})
        return steps

    # MFI (Money Flow Index)
    m = re.match(r"^mfi[_]?(\d+)?$", name_lower)
    if m:
        period = int(m.group(1)) if m.group(1) else 14
        steps.append({"func": "_mfi", "kwargs": {"period": period, "name": name},
                      "output_col": name})
        return steps

    # wick_up_ratio / wick_down_ratio
    if name_lower == "wick_up_ratio":
        steps.append({"func": "_wick_ratio", "kwargs": {"direction": "up", "name": name},
                      "output_col": name})
        return steps
    if name_lower == "wick_down_ratio":
        steps.append({"func": "_wick_ratio", "kwargs": {"direction": "down", "name": name},
                      "output_col": name})
        return steps

    # gap_fill_pct
    if name_lower in ("gap_fill_pct",):
        steps.append({"func": "_gap_fill_pct", "kwargs": {"name": name}, "output_col": name})
        return steps

    # fhr_high / fhr_low (first hour range)
    if name_lower in ("fhr_high", "first_hour_high"):
        steps.append({"func": compute_opening_range, "kwargs": {"minutes": 60}, "output_col": "or_high"})
        steps.append({"func": "_alias", "kwargs": {"src": "or_high", "name": name}, "output_col": name})
        return steps
    if name_lower in ("fhr_low", "first_hour_low"):
        steps.append({"func": compute_opening_range, "kwargs": {"minutes": 60}, "output_col": "or_low"})
        steps.append({"func": "_alias", "kwargs": {"src": "or_low", "name": name}, "output_col": name})
        return steps

    # KC_position: (close - KC_lower) / (KC_upper - KC_lower)
    if name_lower in ("kc_position",):
        steps.append({"func": "_kc_position", "kwargs": {"name": name}, "output_col": name})
        return steps

    # vwap_touch: boolean if price is within 0.1% of VWAP
    if name_lower in ("vwap_touch",):
        steps.append({"func": "_vwap_touch", "kwargs": {"name": name}, "output_col": name})
        return steps

    # vwap_cross_up / vwap_cross_down
    if name_lower in ("vwap_cross_up",):
        steps.append({"func": "_bool_cross_above", "kwargs": {"col": "close", "ref": "vwap", "name": name},
                      "output_col": name})
        return steps
    if name_lower in ("vwap_cross_down",):
        steps.append({"func": "_bool_cross_below", "kwargs": {"col": "close", "ref": "vwap", "name": name},
                      "output_col": name})
        return steps

    # zscore_velocity: zscore[t] - zscore[t-N]
    m = re.match(r"^(\w+)_velocity$", name_lower)
    if m:
        base = m.group(1)
        steps.append({"func": "_diff_shift", "kwargs": {"col": base, "period": 5, "name": name},
                      "output_col": name})
        return steps

    # breakout_volume_ratio
    if name_lower in ("breakout_volume_ratio",):
        steps.append({"func": compute_rel_volume, "kwargs": {"period": 20, "name": name},
                      "output_col": name})
        return steps

    # ── Fallback: try to interpret as column reference or constant ────────
    log.debug("Unparseable indicator formula for %s: %s", name, formula)
    return []


def execute_indicator_steps(df: pl.DataFrame, steps: list[dict]) -> pl.DataFrame:
    """Execute a list of indicator computation steps on a DataFrame."""
    for step in steps:
        func = step["func"]
        kwargs = step["kwargs"]

        if func == "_diff":
            col_a, col_b, n = kwargs["col_a"], kwargs["col_b"], kwargs["name"]
            if col_a in df.columns and col_b in df.columns:
                df = df.with_columns((pl.col(col_a) - pl.col(col_b)).alias(n))
            else:
                log.warning("Cannot compute diff: missing %s or %s", col_a, col_b)
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(n))

        elif func == "_ratio":
            col_a, col_b, n = kwargs["col_a"], kwargs["col_b"], kwargs["name"]
            if col_a in df.columns and col_b in df.columns:
                df = df.with_columns((pl.col(col_a) / pl.col(col_b)).alias(n))
            else:
                log.warning("Cannot compute ratio: missing %s or %s", col_a, col_b)
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(n))

        elif func == "_diff_norm":
            a, b, c, k, n = kwargs["a"], kwargs["b"], kwargs["c"], kwargs["k"], kwargs["name"]
            if all(col in df.columns for col in [a, b, c]):
                df = df.with_columns(
                    ((pl.col(a) - pl.col(b)) / pl.col(c) * k).alias(n)
                )
            else:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(n))

        elif func == "_vwap_dev_pct":
            n = kwargs["name"]
            if "vwap" in df.columns:
                df = df.with_columns(
                    ((pl.col("close") - pl.col("vwap")) / pl.col("vwap") * 100).alias(n)
                )
            else:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(n))

        elif func == "_vwap_band_upper":
            std_mult, n = kwargs["std_mult"], kwargs["name"]
            if "vwap" in df.columns and "vwap_sd" in df.columns:
                df = df.with_columns(
                    (pl.col("vwap") + std_mult * pl.col("vwap_sd")).alias(n)
                )
            elif "vwap" in df.columns:
                dev = pl.col("close") - pl.col("vwap")
                std = dev.rolling_std(window_size=60)
                df = df.with_columns((pl.col("vwap") + std_mult * std).alias(n))

        elif func == "_vwap_band_lower":
            std_mult, n = kwargs["std_mult"], kwargs["name"]
            if "vwap" in df.columns and "vwap_sd" in df.columns:
                df = df.with_columns(
                    (pl.col("vwap") - std_mult * pl.col("vwap_sd")).alias(n)
                )
            elif "vwap" in df.columns:
                dev = pl.col("close") - pl.col("vwap")
                std = dev.rolling_std(window_size=60)
                df = df.with_columns((pl.col("vwap") - std_mult * std).alias(n))

        elif func == "_vwap_zscore":
            period, n = kwargs["period"], kwargs["name"]
            if "vwap" in df.columns:
                dev = pl.col("close") - pl.col("vwap")
                std = dev.rolling_std(window_size=period)
                df = df.with_columns((dev / std).alias(n))
            else:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(n))

        elif func == "_vwap_dev_scaled":
            mult, n = kwargs["mult"], kwargs["name"]
            if "vwap" in df.columns:
                df = df.with_columns(
                    ((pl.col("close") - pl.col("vwap")) / pl.col("vwap") * mult).alias(n)
                )
            else:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(n))

        elif func == "_session_cum_max":
            col, n = kwargs["col"], kwargs["name"]
            if col in df.columns and "day_id" in df.columns:
                df = df.with_columns(pl.col(col).cum_max().over("day_id").alias(n))
            else:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(n))

        elif func == "_session_cum_min":
            col, n = kwargs["col"], kwargs["name"]
            if col in df.columns and "day_id" in df.columns:
                df = df.with_columns(pl.col(col).cum_min().over("day_id").alias(n))
            else:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(n))

        elif func == "_sign_diff":
            col_a, col_b, n = kwargs["col_a"], kwargs["col_b"], kwargs["name"]
            if col_a in df.columns and col_b in df.columns:
                diff = pl.col(col_a) - pl.col(col_b)
                df = df.with_columns(
                    pl.when(diff > 0).then(1.0)
                    .when(diff < 0).then(-1.0)
                    .otherwise(0.0).alias(n)
                )
            else:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(n))

        elif func == "_sign":
            col, n = kwargs["col"], kwargs["name"]
            if col in df.columns:
                df = df.with_columns(
                    pl.when(pl.col(col) > 0).then(1.0)
                    .when(pl.col(col) < 0).then(-1.0)
                    .otherwise(0.0).alias(n)
                )
            else:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(n))

        elif func == "_lagged_return":
            col, lag, mult, n = kwargs["col"], kwargs["lag"], kwargs["mult"], kwargs["name"]
            if col in df.columns:
                df = df.with_columns(
                    ((pl.col(col) - pl.col(col).shift(lag)) / pl.col(col).shift(lag) * mult).alias(n)
                )
            else:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(n))

        elif func == "_alias":
            src, n = kwargs["src"], kwargs["name"]
            if src in df.columns:
                df = df.with_columns(pl.col(src).alias(n))
            else:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(n))

        elif func == "_bb_width":
            n = kwargs["name"]
            # Try to find existing BB columns, else compute default
            upper_col = lower_col = mid_col = None
            for c in df.columns:
                cl = c.lower()
                if "bb_upper" in cl or cl == "bb_upper":
                    upper_col = c
                if "bb_lower" in cl or cl == "bb_lower":
                    lower_col = c
                if "bb_mid" in cl or cl == "bb_mid":
                    mid_col = c
            if upper_col and lower_col:
                df = df.with_columns(
                    (pl.col(upper_col) - pl.col(lower_col)).alias(n)
                )
            elif "close" in df.columns:
                mid = pl.col("close").rolling_mean(window_size=20)
                std = pl.col("close").rolling_std(window_size=20)
                df = df.with_columns((4.0 * std).alias(n))
            else:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(n))

        elif func == "_col_slope":
            col, period, n = kwargs["col"], kwargs["period"], kwargs["name"]
            if col in df.columns:
                df = df.with_columns(
                    ((pl.col(col) - pl.col(col).shift(period)) / period).alias(n)
                )
            else:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(n))

        elif func == "_diff_shift":
            col, period, n = kwargs["col"], kwargs["period"], kwargs["name"]
            if col in df.columns:
                df = df.with_columns(
                    (pl.col(col) - pl.col(col).shift(period)).alias(n)
                )
            else:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(n))

        elif func == "_bool_above":
            col, ref, n = kwargs["col"], kwargs["ref"], kwargs["name"]
            if col in df.columns and ref in df.columns:
                df = df.with_columns(
                    (pl.col(col) > pl.col(ref)).cast(pl.Float64).alias(n)
                )
            else:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(n))

        elif func == "_range_position":
            n = kwargs["name"]
            if "session_high" in df.columns and "session_low" in df.columns:
                rng = pl.col("session_high") - pl.col("session_low")
                df = df.with_columns(
                    pl.when(rng > 0)
                    .then((pl.col("close") - pl.col("session_low")) / rng)
                    .otherwise(0.5).alias(n)
                )
            elif "day_id" in df.columns:
                # Compute inline
                sh = pl.col("high").cum_max().over("day_id")
                sl = pl.col("low").cum_min().over("day_id")
                rng = sh - sl
                df = df.with_columns(
                    pl.when(rng > 0)
                    .then((pl.col("close") - sl) / rng)
                    .otherwise(0.5).alias(n)
                )
            else:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(n))

        elif func == "_bb_pct_b":
            n = kwargs["name"]
            # Find BB columns
            upper_col = lower_col = None
            for c in df.columns:
                cl = c.lower()
                if "bb_upper" in cl:
                    upper_col = c
                if "bb_lower" in cl:
                    lower_col = c
            if upper_col and lower_col:
                width = pl.col(upper_col) - pl.col(lower_col)
                df = df.with_columns(
                    pl.when(width > 0)
                    .then((pl.col("close") - pl.col(lower_col)) / width)
                    .otherwise(0.5).alias(n)
                )
            else:
                # Compute default BB inline
                mid = pl.col("close").rolling_mean(window_size=20)
                std = pl.col("close").rolling_std(window_size=20)
                lower = mid - 2.0 * std
                width = 4.0 * std
                df = df.with_columns(
                    pl.when(width > 0)
                    .then((pl.col("close") - lower) / width)
                    .otherwise(0.5).alias(n)
                )

        elif func == "_true_range":
            n = kwargs["name"]
            tr = pl.max_horizontal(
                pl.col("high") - pl.col("low"),
                (pl.col("high") - pl.col("close").shift(1)).abs(),
                (pl.col("low") - pl.col("close").shift(1)).abs(),
            )
            df = df.with_columns(tr.alias(n))

        elif func == "_cum_count_above":
            col, ref, n = kwargs["col"], kwargs["ref"], kwargs["name"]
            if col in df.columns and ref in df.columns and "day_id" in df.columns:
                above = (pl.col(col) > pl.col(ref)).cast(pl.Int32)
                df = df.with_columns(above.cum_sum().over("day_id").cast(pl.Float64).alias(n))
            else:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(n))

        elif func == "_cum_count_below":
            col, ref, n = kwargs["col"], kwargs["ref"], kwargs["name"]
            if col in df.columns and ref in df.columns and "day_id" in df.columns:
                below = (pl.col(col) < pl.col(ref)).cast(pl.Int32)
                df = df.with_columns(below.cum_sum().over("day_id").cast(pl.Float64).alias(n))
            else:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(n))

        elif func == "_stochastic_k":
            period, n = kwargs["period"], kwargs["name"]
            highest = pl.col("high").rolling_max(window_size=period)
            lowest = pl.col("low").rolling_min(window_size=period)
            rng = highest - lowest
            k = pl.when(rng > 0).then((pl.col("close") - lowest) / rng * 100.0).otherwise(50.0)
            # Smooth with SMA(3)
            df = df.with_columns(k.alias("_stoch_raw"))
            df = df.with_columns(pl.col("_stoch_raw").rolling_mean(window_size=3).alias(n))
            df = df.drop("_stoch_raw")

        elif func == "_mfi":
            period, n = kwargs["period"], kwargs["name"]
            tp = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
            raw_mf = tp * pl.col("volume")
            tp_diff = tp.diff()
            pos_mf = pl.when(tp_diff > 0).then(raw_mf).otherwise(0.0)
            neg_mf = pl.when(tp_diff < 0).then(raw_mf).otherwise(0.0)
            pos_sum = pos_mf.rolling_sum(window_size=period)
            neg_sum = neg_mf.rolling_sum(window_size=period)
            mfr = pos_sum / neg_sum
            mfi = pl.lit(100.0) - (pl.lit(100.0) / (pl.lit(1.0) + mfr))
            df = df.with_columns(mfi.alias(n))

        elif func == "_wick_ratio":
            direction, n = kwargs["direction"], kwargs["name"]
            body = (pl.col("close") - pl.col("open")).abs()
            full_range = pl.col("high") - pl.col("low")
            if direction == "up":
                wick = pl.col("high") - pl.max_horizontal(pl.col("close"), pl.col("open"))
            else:
                wick = pl.min_horizontal(pl.col("close"), pl.col("open")) - pl.col("low")
            df = df.with_columns(
                pl.when(full_range > 0).then(wick / full_range).otherwise(0.0).alias(n)
            )

        elif func == "_gap_fill_pct":
            n = kwargs["name"]
            if "prev_close" in df.columns and "day_id" in df.columns:
                # How much of the gap has been filled: 1.0 = fully filled
                gap = pl.col("open").first().over("day_id") - pl.col("prev_close")
                # Use session_high/low to approximate fill
                df = df.with_columns(
                    pl.when(gap.abs() < 0.001).then(1.0)
                    .when(gap > 0).then(
                        (pl.col("open").first().over("day_id") - pl.col("low").cum_min().over("day_id")) / gap
                    )
                    .otherwise(
                        (pl.col("high").cum_max().over("day_id") - pl.col("open").first().over("day_id")) / gap.abs()
                    ).alias(n)
                )
            else:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(n))

        elif func == "_kc_position":
            n = kwargs["name"]
            # Default Keltner: EMA(20), ATR(10), mult=1.5
            mid = pl.col("close").ewm_mean(span=20, adjust=False)
            tr = pl.max_horizontal(
                pl.col("high") - pl.col("low"),
                (pl.col("high") - pl.col("close").shift(1)).abs(),
                (pl.col("low") - pl.col("close").shift(1)).abs(),
            )
            atr = tr.ewm_mean(span=10, adjust=False)
            upper = mid + 1.5 * atr
            lower = mid - 1.5 * atr
            width = upper - lower
            df = df.with_columns(
                pl.when(width > 0).then((pl.col("close") - lower) / width).otherwise(0.5).alias(n)
            )

        elif func == "_vwap_touch":
            n = kwargs["name"]
            if "vwap" in df.columns:
                dev = ((pl.col("close") - pl.col("vwap")) / pl.col("vwap")).abs()
                df = df.with_columns((dev < 0.001).cast(pl.Float64).alias(n))
            else:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(n))

        elif func == "_bool_cross_above":
            col, ref, n = kwargs["col"], kwargs["ref"], kwargs["name"]
            if col in df.columns and ref in df.columns:
                above_now = pl.col(col) > pl.col(ref)
                above_prev = pl.col(col).shift(1) <= pl.col(ref).shift(1)
                df = df.with_columns((above_now & above_prev).cast(pl.Float64).alias(n))
            else:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(n))

        elif func == "_bool_cross_below":
            col, ref, n = kwargs["col"], kwargs["ref"], kwargs["name"]
            if col in df.columns and ref in df.columns:
                below_now = pl.col(col) < pl.col(ref)
                below_prev = pl.col(col).shift(1) >= pl.col(ref).shift(1)
                df = df.with_columns((below_now & below_prev).cast(pl.Float64).alias(n))
            else:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(n))

        elif func == "_directional_indicator":
            sign_type, period, n = kwargs["sign"], kwargs["period"], kwargs["name"]
            high_diff = pl.col("high") - pl.col("high").shift(1)
            low_diff = pl.col("low").shift(1) - pl.col("low")
            tr = pl.max_horizontal(
                pl.col("high") - pl.col("low"),
                (pl.col("high") - pl.col("close").shift(1)).abs(),
                (pl.col("low") - pl.col("close").shift(1)).abs(),
            )
            atr = tr.ewm_mean(span=period, adjust=False)
            if sign_type == "plus":
                dm = pl.when((high_diff > low_diff) & (high_diff > 0)).then(high_diff).otherwise(0.0)
                di = dm.ewm_mean(span=period, adjust=False) / atr * 100
            else:
                dm = pl.when((low_diff > high_diff) & (low_diff > 0)).then(low_diff).otherwise(0.0)
                di = dm.ewm_mean(span=period, adjust=False) / atr * 100
            df = df.with_columns(di.alias(n))

        else:
            # It's a real function
            try:
                df = func(df, **kwargs)
            except Exception as e:
                log.warning("Indicator computation failed for %s: %s", step["output_col"], e)
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(step["output_col"]))

    return df


def compute_all_indicators(
    df: pl.DataFrame,
    indicator_defs: list[dict],
    needs_vwap: bool = False,
    needs_pdh_pdl: bool = False,
    needs_prev_close: bool = False,
    needs_opening_range: bool = False,
    needs_gap: bool = False,
    needs_obv: bool = False,
    needs_bar_count: bool = False,
) -> tuple[pl.DataFrame, list[str], list[str]]:
    """Compute all indicators for a strategy.

    Returns (df, computed_cols, failed_cols).
    """
    computed = []
    failed = []
    existing = set(df.columns)

    # Pre-compute structural indicators if needed
    if needs_vwap and "vwap" not in existing:
        df = compute_vwap(df)
        existing.add("vwap")
        computed.append("vwap")

    if needs_pdh_pdl and "pdh" not in existing:
        df = compute_prev_day_high_low(df)
        existing.update(["pdh", "pdl"])
        computed.extend(["pdh", "pdl"])

    if needs_prev_close and "prev_close" not in existing:
        df = compute_prev_day_close(df)
        existing.add("prev_close")
        computed.append("prev_close")

    if needs_gap and "gap_pct" not in existing:
        df = compute_gap_pct(df)
        existing.add("gap_pct")
        computed.append("gap_pct")

    if needs_opening_range and "or_high" not in existing:
        df = compute_opening_range(df)
        existing.update(["or_high", "or_low"])
        computed.extend(["or_high", "or_low"])

    if needs_obv and "obv" not in existing:
        df = compute_obv(df)
        existing.add("obv")
        computed.append("obv")

    if needs_bar_count and "bar_count_from_open" not in existing:
        df = compute_bar_count_from_open(df)
        existing.add("bar_count_from_open")
        computed.append("bar_count_from_open")

    # Compute declared indicators (topological sort by dependencies)
    remaining = list(indicator_defs)
    max_passes = len(remaining) + 2  # prevent infinite loop
    pass_count = 0

    while remaining and pass_count < max_passes:
        pass_count += 1
        still_remaining = []
        for ind in remaining:
            ind_name = ind.get("name", "")
            if ind_name in existing:
                continue  # already computed

            steps = parse_indicator_formula(ind, existing)
            if not steps:
                # Check if it's a text-based / narrative formula we can't parse
                formula = ind.get("formula", "")
                if _is_narrative_formula(formula):
                    log.info("Skipping narrative indicator %s: %s", ind_name, formula)
                    failed.append(ind_name)
                    continue
                # Maybe dependencies not ready yet
                still_remaining.append(ind)
                continue

            # Check if all input columns are available
            deps_ready = True
            for step in steps:
                kwargs = step["kwargs"]
                for k, v in kwargs.items():
                    if k in ("col", "col_a", "col_b", "a", "b", "c") and isinstance(v, str):
                        if v not in existing and not v.startswith("_"):
                            deps_ready = False
                            break
                if not deps_ready:
                    break

            if not deps_ready:
                still_remaining.append(ind)
                continue

            try:
                df = execute_indicator_steps(df, steps)
                for step in steps:
                    existing.add(step["output_col"])
                existing.add(ind_name)
                computed.append(ind_name)
            except Exception as e:
                log.warning("Failed to compute indicator %s: %s", ind_name, e)
                failed.append(ind_name)

        if len(still_remaining) == len(remaining):
            # No progress — remaining indicators have unresolvable deps
            for ind in still_remaining:
                ind_name = ind.get("name", "")
                log.warning("Unresolvable indicator deps for %s: %s", ind_name, ind.get("formula", ""))
                failed.append(ind_name)
            break
        remaining = still_remaining

    return df, computed, failed


def _is_narrative_formula(formula: str) -> bool:
    """Check if a formula is plain English / narrative rather than computable."""
    narrative_markers = [
        "count of consecutive",
        "classify", "tick-level",
        "hedge_ratio",
        "log(close_A)",
        "close_A", "close_B",
        "cointegrat",
        "TERP",
        "buy_volume", "sell_volume",
        "both legs",
        "order book", "bid-ask",
        "option chain", "implied volatility",
        "tick data", "microstructure",
        "pair", "spread between",
    ]
    formula_lower = formula.lower()
    return any(marker.lower() in formula_lower for marker in narrative_markers)


def get_max_lookback(indicator_defs: list[dict]) -> int:
    """Get the maximum lookback period across all indicators."""
    max_lb = 0
    for ind in indicator_defs:
        lb = ind.get("lookback", 0)
        if lb is not None and isinstance(lb, (int, float)):
            max_lb = max(max_lb, int(lb))
    # At least 1 bar warmup
    return max(max_lb, 1)
