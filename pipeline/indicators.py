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
        (pl.col("_cum_tp_vol") / pl.col("_cum_vol")).alias(name)
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
        (pl.arange(0, pl.count()).over("day_id")).cast(pl.Int32).alias(name)
    )


# ── Formula parser ──────────────────────────────────────────────────────────

# Regex patterns for formula parsing
_RE_SMA = re.compile(r"SMA\(\s*(\w+)\s*,\s*(\d+)\s*\)", re.IGNORECASE)
_RE_EMA = re.compile(r"EMA\(\s*(\w+)\s*,\s*(\d+)\s*\)", re.IGNORECASE)
_RE_RSI = re.compile(r"RSI\(\s*(\w+)\s*,\s*(\d+)\s*\)", re.IGNORECASE)
_RE_ATR = re.compile(r"ATR\(\s*(?:\w+\s*,\s*\w+\s*,\s*\w+\s*,\s*)?(\d+)\s*\)", re.IGNORECASE)
_RE_BB = re.compile(r"BB_(upper|lower|mid)\(\s*(\w+)\s*,\s*(\d+)\s*,\s*([\d.]+)\s*\)", re.IGNORECASE)
_RE_MACD = re.compile(r"MACD\(\s*(\w+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", re.IGNORECASE)
_RE_ADX = re.compile(r"ADX\(\s*(\d+)\s*\)", re.IGNORECASE)
_RE_ZSCORE = re.compile(r"zscore\(\s*(.+?)\s*,\s*(\d+)\s*\)", re.IGNORECASE)
_RE_MAX = re.compile(r"(?:MAX|HIGHEST)\(\s*(\w+)\s*,\s*(\d+)\s*\)", re.IGNORECASE)
_RE_MIN = re.compile(r"(?:MIN|LOWEST)\(\s*(\w+)\s*,\s*(\d+)\s*\)", re.IGNORECASE)
_RE_STDEV = re.compile(r"(?:STDEV|STD|rolling_std)\(\s*(.+?)\s*,\s*(\d+)\s*\)", re.IGNORECASE)
_RE_RATIO = re.compile(r"^(\w+)\s*/\s*SMA\(\s*(\w+)\s*,\s*(\d+)\s*\)$", re.IGNORECASE)
_RE_SIMPLE_RATIO = re.compile(r"^(\w+)\s*/\s*(\w+)$")
_RE_DIFF = re.compile(r"^(\w+)\s*-\s*(\w+)$")
_RE_DIFF_NORM = re.compile(r"^\(\s*(\w+)\s*-\s*(\w+)\s*\)\s*/\s*(\w+)\s*(?:\*\s*([\d.]+))?$")
_RE_OBV = re.compile(r"OBV\(\s*(\w+)\s*,\s*(\w+)\s*\)", re.IGNORECASE)
_RE_OBV_SLOPE = re.compile(r"linear_regression_slope\(\s*OBV\(.+?\)\s*,\s*(\d+)\s*\)", re.IGNORECASE)
_RE_VWAP_SIMPLE = re.compile(r"^(?:VWAP|vwap)\(\s*\)$|^(?:VWAP|vwap)$", re.IGNORECASE)
_RE_RANGE_WIDTH = re.compile(r"^\(\s*(\w+)\s*-\s*(\w+)\s*\)\s*/\s*(\w+)\s*\*\s*([\d.]+)$")
_RE_VWAP_DEV = re.compile(r"^\(\s*close\s*-\s*vwap\s*\)\s*/\s*vwap\s*\*\s*100$", re.IGNORECASE)
_RE_VWAP_BANDS = re.compile(r"vwap\s*[+-]\s*([\d.]+)\s*\*\s*vwap_sd", re.IGNORECASE)


def parse_indicator_formula(indicator: dict, existing_cols: set[str]) -> list[dict]:
    """Parse an indicator formula and return a list of computation steps.

    Each step is a dict: {"func": callable, "kwargs": {...}, "output_col": str}
    Returns empty list if unparseable.
    """
    name = indicator.get("name", "")
    formula = indicator.get("formula", "")
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
        "linear_regression_slope",
        "hedge_ratio",
        "log(close_A)",
        "close_A", "close_B",
        "cointegrat",
        "TERP",
        "buy_volume", "sell_volume",
        "both legs",
        "first 15 market minutes",
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
