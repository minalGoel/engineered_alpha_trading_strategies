"""Data loading — read Parquet files, merge VIX/index, assign day_id."""
from __future__ import annotations
import polars as pl
from pathlib import Path
from typing import Optional
import logging
import orjson

from pipeline.config import (
    DATA_DIR, TIMEFRAME_FILES, VIX_FILES, INDEX_FILES, STOCKS_LIST_FILE,
    MARKET_OPEN_H, MARKET_OPEN_M, TRAIN_END, TEST_START,
)

log = logging.getLogger(__name__)


def load_stocks_list() -> list[str]:
    """Return sorted list of available stock symbols."""
    path = STOCKS_LIST_FILE
    if not path.exists():
        log.warning("stocks_list.parquet not found; will infer from data")
        return []
    df = pl.read_parquet(path)
    # Try common column names
    for col in ("symbol", "Symbol", "SYMBOL", "ticker", "Ticker"):
        if col in df.columns:
            return sorted(df[col].drop_nulls().unique().to_list())
    # Fallback: first string column
    for col in df.columns:
        if df[col].dtype == pl.Utf8:
            return sorted(df[col].drop_nulls().unique().to_list())
    return []


def _normalise_datetime_col(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure a 'datetime' column exists as pl.Datetime."""
    # Try common column names
    dt_candidates = ["datetime", "Datetime", "date", "Date", "timestamp", "Timestamp"]
    found = None
    for c in dt_candidates:
        if c in df.columns:
            found = c
            break
    if found is None:
        raise ValueError(f"No datetime column found in columns: {df.columns}")

    if found != "datetime":
        df = df.rename({found: "datetime"})

    # Cast to datetime if needed
    if df["datetime"].dtype == pl.Utf8:
        df = df.with_columns(pl.col("datetime").str.to_datetime().alias("datetime"))
    elif df["datetime"].dtype == pl.Date:
        df = df.with_columns(pl.col("datetime").cast(pl.Datetime).alias("datetime"))
    return df


def _normalise_symbol_col(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure a 'symbol' column exists."""
    for c in ("symbol", "Symbol", "SYMBOL", "ticker", "Ticker"):
        if c in df.columns:
            if c != "symbol":
                df = df.rename({c: "symbol"})
            return df
    return df  # No symbol column (e.g., VIX/index data)


def _normalise_ohlcv_cols(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure lowercase ohlcv column names."""
    rename_map = {}
    for expected in ("open", "high", "low", "close", "volume"):
        for c in df.columns:
            if c.lower() == expected and c != expected:
                rename_map[c] = expected
                break
    if rename_map:
        df = df.rename(rename_map)
    return df


def _strip_timezone(df: pl.DataFrame) -> pl.DataFrame:
    """Strip timezone from datetime column if present, for safe comparisons."""
    if "datetime" in df.columns:
        dt_dtype = df["datetime"].dtype
        if hasattr(dt_dtype, "time_zone") and dt_dtype.time_zone is not None:
            df = df.with_columns(
                pl.col("datetime").dt.replace_time_zone(None).alias("datetime")
            )
    return df


def assign_day_id(df: pl.DataFrame) -> pl.DataFrame:
    """Assign integer day_id based on calendar date of each bar."""
    df = df.with_columns(
        pl.col("datetime").dt.date().alias("_date")
    )
    # Map each unique date to an integer
    dates = df["_date"].unique().sort()
    date_map = {d: i for i, d in enumerate(dates.to_list())}
    df = df.with_columns(
        pl.col("_date").replace_strict(date_map, default=-1).cast(pl.Int32).alias("day_id")
    )
    df = df.drop("_date")
    return df


def _time_in_minutes(dt_col: pl.Expr) -> pl.Expr:
    """Convert datetime to minutes since midnight."""
    return dt_col.dt.hour() * 60 + dt_col.dt.minute()


def load_stock_data(
    timeframe: str,
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[pl.DataFrame]:
    """Load OHLCV data for one stock at the given timeframe.

    Returns DataFrame with columns: datetime, open, high, low, close, volume, day_id
    or None if data unavailable.
    """
    path = TIMEFRAME_FILES.get(timeframe)
    if path is None or not path.exists():
        log.warning("No data file for timeframe %s", timeframe)
        return None

    try:
        # Use scan for lazy filtering
        lf = pl.scan_parquet(path)

        # Normalise column names
        cols = lf.collect_schema().names()
        rename = {}
        for c in cols:
            low = c.lower()
            if low in ("symbol", "open", "high", "low", "close", "volume"):
                if c != low:
                    rename[c] = low
            elif low in ("datetime", "date", "timestamp"):
                if c != "datetime":
                    rename[c] = "datetime"
        if rename:
            lf = lf.rename(rename)

        # Filter by symbol
        schema_names = lf.collect_schema().names()
        if "symbol" in schema_names:
            lf = lf.filter(pl.col("symbol") == symbol)

        # Filter by date range (timezone-safe: cast to naive if needed)
        if start_date or end_date:
            dt_dtype = lf.collect_schema()["datetime"]
            # If the column is tz-aware, strip timezone for comparison
            if hasattr(dt_dtype, "time_zone") and dt_dtype.time_zone is not None:
                lf = lf.with_columns(
                    pl.col("datetime").dt.replace_time_zone(None).alias("datetime")
                )
        if start_date:
            lf = lf.filter(pl.col("datetime") >= pl.lit(start_date).str.to_datetime())
        if end_date:
            lf = lf.filter(pl.col("datetime") <= pl.lit(end_date + " 23:59:59").str.to_datetime())

        df = lf.collect()
        if df.is_empty():
            return None

        df = _normalise_ohlcv_cols(df)
        df = df.sort("datetime")
        df = assign_day_id(df)
        return df

    except Exception as e:
        log.error("Failed to load %s/%s: %s", timeframe, symbol, e)
        return None


def load_vix_data(
    timeframe: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[pl.DataFrame]:
    """Load India VIX data. Returns DataFrame with datetime, close (as vix)."""
    path = VIX_FILES.get(timeframe)
    if path is None or not path.exists():
        # Fallback: try closest available timeframe
        for tf in ("1min", "5min", "15min", "30min", "1h"):
            if VIX_FILES.get(tf, Path()).exists():
                path = VIX_FILES[tf]
                break
        if path is None or not path.exists():
            log.warning("No VIX data found")
            return None

    try:
        df = pl.read_parquet(path)
        df = _normalise_datetime_col(df)
        df = _normalise_ohlcv_cols(df)
        df = _strip_timezone(df)

        if start_date:
            df = df.filter(pl.col("datetime") >= pl.lit(start_date).str.to_datetime())
        if end_date:
            df = df.filter(pl.col("datetime") <= pl.lit(end_date + " 23:59:59").str.to_datetime())

        # Keep only datetime and close, rename close to vix
        if "close" in df.columns:
            df = df.select(["datetime", "close"]).rename({"close": "vix"})
        elif "Close" in df.columns:
            df = df.select(["datetime", "Close"]).rename({"Close": "vix"})
        else:
            # Use first numeric column
            for c in df.columns:
                if c != "datetime" and df[c].dtype in (pl.Float64, pl.Float32, pl.Int64):
                    df = df.select(["datetime", c]).rename({c: "vix"})
                    break

        df = df.sort("datetime")
        return df
    except Exception as e:
        log.error("Failed to load VIX: %s", e)
        return None


def load_index_data(
    timeframe: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[pl.DataFrame]:
    """Load NIFTY 50 index data."""
    path = INDEX_FILES.get(timeframe)
    if path is None or not path.exists():
        for tf in ("1min", "5min", "15min", "30min", "1h"):
            if INDEX_FILES.get(tf, Path()).exists():
                path = INDEX_FILES[tf]
                break
        if path is None or not path.exists():
            log.warning("No index data found")
            return None

    try:
        df = pl.read_parquet(path)
        df = _normalise_datetime_col(df)
        df = _normalise_ohlcv_cols(df)
        df = _strip_timezone(df)

        if start_date:
            df = df.filter(pl.col("datetime") >= pl.lit(start_date).str.to_datetime())
        if end_date:
            df = df.filter(pl.col("datetime") <= pl.lit(end_date + " 23:59:59").str.to_datetime())

        # Prefix columns to avoid clash
        rename_map = {}
        for c in ("open", "high", "low", "close", "volume"):
            if c in df.columns:
                rename_map[c] = f"index_{c}"
        if rename_map:
            df = df.rename(rename_map)

        df = df.sort("datetime")
        return df
    except Exception as e:
        log.error("Failed to load index: %s", e)
        return None


def merge_vix_index(
    stock_df: pl.DataFrame,
    vix_df: Optional[pl.DataFrame],
    index_df: Optional[pl.DataFrame],
) -> pl.DataFrame:
    """Join VIX and index data onto stock data using asof join (nearest prior bar)."""
    if vix_df is not None and not vix_df.is_empty():
        stock_df = stock_df.join_asof(
            vix_df.sort("datetime"),
            on="datetime",
            strategy="backward",
        )
    else:
        stock_df = stock_df.with_columns(pl.lit(None).cast(pl.Float64).alias("vix"))

    if index_df is not None and not index_df.is_empty():
        stock_df = stock_df.join_asof(
            index_df.sort("datetime"),
            on="datetime",
            strategy="backward",
        )
    else:
        for c in ("index_open", "index_high", "index_low", "index_close", "index_volume"):
            if c not in stock_df.columns:
                stock_df = stock_df.with_columns(pl.lit(None).cast(pl.Float64).alias(c))

    return stock_df


def split_train_test(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split into training (<=2024-12-31) and test (>=2025-01-01)."""
    df = _strip_timezone(df)
    train = df.filter(pl.col("datetime") <= pl.lit(TRAIN_END + " 23:59:59").str.to_datetime())
    test = df.filter(pl.col("datetime") >= pl.lit(TEST_START).str.to_datetime())
    return train, test


def get_available_symbols(timeframe: str) -> list[str]:
    """Get list of symbols available in the given timeframe data file."""
    path = TIMEFRAME_FILES.get(timeframe)
    if path is None or not path.exists():
        return []
    try:
        lf = pl.scan_parquet(path)
        cols = lf.collect_schema().names()
        sym_col = None
        for c in ("symbol", "Symbol", "SYMBOL", "ticker"):
            if c in cols:
                sym_col = c
                break
        if sym_col is None:
            return []
        syms = lf.select(pl.col(sym_col).unique()).collect()[sym_col].to_list()
        return sorted(syms)
    except Exception as e:
        log.error("Failed to get symbols for %s: %s", timeframe, e)
        return []


def build_data_inventory() -> dict:
    """Phase 1: Discover and document all available data."""
    inventory = {
        "timeframes": {},
        "vix_available": {},
        "index_available": {},
        "symbols": [],
    }

    for tf, path in TIMEFRAME_FILES.items():
        if path.exists():
            try:
                lf = pl.scan_parquet(path)
                schema = {name: str(dtype) for name, dtype in lf.collect_schema().items()}
                # Find datetime column
                dt_col = None
                for c in lf.collect_schema().names():
                    if c.lower() in ("datetime", "date", "timestamp"):
                        dt_col = c
                        break
                # Get row count and date range
                if dt_col is not None:
                    stats = lf.select(
                        pl.count().alias("rows"),
                        pl.col(dt_col).min().alias("min_dt"),
                        pl.col(dt_col).max().alias("max_dt"),
                    ).collect()
                else:
                    stats = lf.select(pl.count().alias("rows")).collect()
                    stats = stats.with_columns(
                        pl.lit(None).alias("min_dt"),
                        pl.lit(None).alias("max_dt"),
                    )
                symbols = get_available_symbols(tf)
                inventory["timeframes"][tf] = {
                    "file": str(path),
                    "columns": schema,
                    "rows": stats["rows"][0],
                    "date_range": [str(stats["min_dt"][0]), str(stats["max_dt"][0])],
                    "symbols_count": len(symbols),
                }
                if tf == "1min":  # Use 1min as primary symbol list
                    inventory["symbols"] = symbols
            except Exception as e:
                inventory["timeframes"][tf] = {"file": str(path), "error": str(e)}
        else:
            inventory["timeframes"][tf] = {"file": str(path), "exists": False}

    for tf, path in VIX_FILES.items():
        inventory["vix_available"][tf] = path.exists()

    for tf, path in INDEX_FILES.items():
        inventory["index_available"][tf] = path.exists()

    if not inventory["symbols"]:
        inventory["symbols"] = load_stocks_list()

    return inventory
