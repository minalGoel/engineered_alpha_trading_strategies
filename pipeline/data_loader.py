"""Data loading for 5-second index option trading.

Loads spot_candles.parquet and option_candles.parquet from 5second_data/.
Converts UTC timestamps to IST using convert_time_zone (NOT replace_time_zone).
"""
from __future__ import annotations
import polars as pl
import numpy as np
from pathlib import Path
from typing import Optional
import logging

from pipeline.config import (
    FIVE_SEC_DIR, SPOT_FILE, OPTION_FILE,
    NIFTY_SYMBOL, BANKNIFTY_SYMBOL, VIX_SYMBOL,
)

log = logging.getLogger(__name__)


def _utc_to_ist(df: pl.DataFrame, ts_col: str = "ts") -> pl.DataFrame:
    """Convert UTC timestamps to IST (naive) and rename to 'datetime'.

    CRITICAL: Uses convert_time_zone then strips timezone.
    NEVER use replace_time_zone alone — that was a prior bug.
    """
    dt_dtype = df[ts_col].dtype

    if hasattr(dt_dtype, "time_zone") and dt_dtype.time_zone is not None:
        # Already tz-aware: convert to IST then strip
        df = df.with_columns(
            pl.col(ts_col)
            .dt.convert_time_zone("Asia/Kolkata")
            .dt.replace_time_zone(None)
            .alias("datetime")
        )
    else:
        # Naive timestamps assumed UTC — attach UTC then convert
        df = df.with_columns(
            pl.col(ts_col)
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone("Asia/Kolkata")
            .dt.replace_time_zone(None)
            .alias("datetime")
        )

    if ts_col != "datetime":
        df = df.drop(ts_col)
    return df


def _add_time_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Add day_id and time_minutes columns."""
    # day_id from session_date if available, else from datetime
    if "session_date" in df.columns:
        dates = df["session_date"].unique().sort()
        date_map = {d: i for i, d in enumerate(dates.to_list())}
        df = df.with_columns(
            pl.col("session_date")
            .replace_strict(date_map, default=-1)
            .cast(pl.Int32)
            .alias("day_id")
        )
    else:
        df = df.with_columns(
            pl.col("datetime").dt.date().alias("_date")
        )
        dates = df["_date"].unique().sort()
        date_map = {d: i for i, d in enumerate(dates.to_list())}
        df = df.with_columns(
            pl.col("_date")
            .replace_strict(date_map, default=-1)
            .cast(pl.Int32)
            .alias("day_id")
        )
        df = df.drop("_date")

    # time_minutes = minutes from midnight IST
    df = df.with_columns(
        (pl.col("datetime").dt.hour() * 60 + pl.col("datetime").dt.minute())
        .cast(pl.Int32)
        .alias("time_minutes")
    )
    return df


def load_spot_data(symbol: str) -> Optional[pl.DataFrame]:
    """Load 5-second spot candles for one symbol (NIFTY/BANKNIFTY/VIX).

    Returns DataFrame with columns:
        datetime (IST naive), open, high, low, close, volume,
        session_date, day_id, time_minutes
    """
    if not SPOT_FILE.exists():
        log.error("Spot data file not found: %s", SPOT_FILE)
        return None

    df = pl.read_parquet(SPOT_FILE)
    df = df.filter(pl.col("symbol") == symbol)

    if df.is_empty():
        log.warning("No spot data for symbol: %s", symbol)
        return None

    df = _utc_to_ist(df)
    df = _add_time_columns(df)
    df = df.sort("datetime")

    # Drop epoch and symbol columns (no longer needed)
    drop_cols = [c for c in ("epoch", "symbol") if c in df.columns]
    if drop_cols:
        df = df.drop(drop_cols)

    return df


def load_option_data(
    underlying: str,
    expiry: Optional[str] = None,
) -> Optional[pl.DataFrame]:
    """Load 5-second option candles for one underlying.

    Args:
        underlying: "NSE:NIFTY50-INDEX" or "NSE:NIFTYBANK-INDEX"
        expiry: Optional expiry date filter (str YYYY-MM-DD)

    Returns DataFrame with columns:
        symbol, underlying, session_date, expiry, strike, option_type,
        datetime (IST naive), open, high, low, close, volume, open_interest,
        day_id, time_minutes
    """
    if not OPTION_FILE.exists():
        log.error("Option data file not found: %s", OPTION_FILE)
        return None

    lf = pl.scan_parquet(OPTION_FILE)
    lf = lf.filter(pl.col("underlying") == underlying)

    if expiry is not None:
        lf = lf.filter(pl.col("expiry") == pl.lit(expiry).str.to_date())

    df = lf.collect()
    if df.is_empty():
        log.warning("No option data for underlying: %s", underlying)
        return None

    df = _utc_to_ist(df)
    df = _add_time_columns(df)
    df = df.sort(["strike", "option_type", "expiry", "datetime"])

    # Drop epoch
    if "epoch" in df.columns:
        df = df.drop("epoch")

    return df


def load_all_data() -> dict:
    """Load all 5-second data, split by symbol.

    Returns dict with keys:
        nifty_spot, banknifty_spot, vix,
        nifty_options, banknifty_options,
        trading_days (int)
    """
    nifty_spot = load_spot_data(NIFTY_SYMBOL)
    banknifty_spot = load_spot_data(BANKNIFTY_SYMBOL)
    vix = load_spot_data(VIX_SYMBOL)
    nifty_options = load_option_data(NIFTY_SYMBOL)
    banknifty_options = load_option_data(BANKNIFTY_SYMBOL)

    # Count trading days
    trading_days = 0
    if nifty_spot is not None:
        trading_days = nifty_spot["day_id"].n_unique()
    elif banknifty_spot is not None:
        trading_days = banknifty_spot["day_id"].n_unique()

    return {
        "nifty_spot": nifty_spot,
        "banknifty_spot": banknifty_spot,
        "vix": vix,
        "nifty_options": nifty_options,
        "banknifty_options": banknifty_options,
        "trading_days": trading_days,
    }


def add_atm_strike(spot_df: pl.DataFrame, step: int) -> pl.DataFrame:
    """Add ATM strike column to spot data."""
    return spot_df.with_columns(
        (pl.col("close") / step).round(0).cast(pl.Float64).mul(step).alias("atm_strike")
    )


def get_nearest_expiry_option(
    option_df: pl.DataFrame,
    session_date,
    strike: float,
    option_type: str,
) -> Optional[pl.DataFrame]:
    """Get option chain for nearest expiry on a given date, strike, and type."""
    day_opts = option_df.filter(
        (pl.col("session_date") == session_date) &
        (pl.col("strike") == strike) &
        (pl.col("option_type") == option_type)
    )
    if day_opts.is_empty():
        return None

    # Pick nearest expiry
    nearest_expiry = day_opts["expiry"].min()
    return day_opts.filter(pl.col("expiry") == nearest_expiry)


def split_leave_one_out(
    spot_df: pl.DataFrame,
) -> list[tuple[pl.DataFrame, pl.DataFrame]]:
    """Leave-one-day-out cross-validation splits.

    Returns list of (train_df, test_df) tuples — one per trading day.
    """
    day_ids = sorted(spot_df["day_id"].unique().to_list())
    splits = []
    for test_day in day_ids:
        train = spot_df.filter(pl.col("day_id") != test_day)
        test = spot_df.filter(pl.col("day_id") == test_day)
        splits.append((train, test))
    return splits


def build_data_inventory() -> dict:
    """Discover and document available 5-second data."""
    inventory = {
        "spot_file": str(SPOT_FILE),
        "option_file": str(OPTION_FILE),
        "spot_exists": SPOT_FILE.exists(),
        "option_exists": OPTION_FILE.exists(),
    }

    if SPOT_FILE.exists():
        spot = pl.read_parquet(SPOT_FILE)
        inventory["spot_rows"] = len(spot)
        inventory["spot_symbols"] = spot["symbol"].unique().to_list()
        inventory["spot_date_range"] = [
            str(spot["session_date"].min()),
            str(spot["session_date"].max()),
        ]

    if OPTION_FILE.exists():
        opt = pl.read_parquet(OPTION_FILE)
        inventory["option_rows"] = len(opt)
        inventory["option_underlyings"] = opt["underlying"].unique().to_list()
        inventory["option_expiries"] = [str(d) for d in sorted(opt["expiry"].unique().to_list())]
        inventory["option_types"] = opt["option_type"].unique().to_list()

    return inventory


def integration_test():
    """Quick sanity check: first bar hour in IST should be 9."""
    spot = load_spot_data(NIFTY_SYMBOL)
    if spot is None:
        raise RuntimeError("Cannot load NIFTY spot data for integration test")

    first_hour = spot["datetime"].head(1).dt.hour().to_list()[0]
    assert first_hour == 9, (
        f"CRITICAL: First bar hour is {first_hour}, expected 9 (IST). "
        f"UTC→IST conversion is broken!"
    )
    log.info("Integration test passed: first bar hour = %d (IST)", first_hour)

    # Check time_minutes
    first_tm = spot["time_minutes"].head(1).to_list()[0]
    assert 555 <= first_tm <= 570, (
        f"First bar time_minutes={first_tm}, expected ~555-570 (09:15-09:30 IST)"
    )
    log.info("Integration test passed: first bar time_minutes = %d", first_tm)
