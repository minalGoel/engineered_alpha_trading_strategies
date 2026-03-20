# Data Schemas

## Summary
Two data regimes: 1-minute OHLCV for 206 equities and 5-second candles for NIFTY/BANKNIFTY spot + options. Both require UTC→IST conversion. This document is the canonical reference for column names, types, and data exploration rules.

## Rule Zero: Explore Before Coding

**NEVER assume column names, directory structure, or date ranges.** Load sample files, print schemas, check dates, verify timezone. The most destructive bug in the project (UTC→IST) existed because the data schema was assumed, not verified.

Always create `outputs/data_inventory.json` documenting what was actually found.

## Equity Data (1-Minute OHLCV)

### Location
`data/*.parquet` — one file per stock

### Schema
| Column | Type | Description |
|--------|------|-------------|
| datetime | Timestamp | **UTC** — must convert to IST |
| open | Float64 | Open price |
| high | Float64 | High price |
| low | Float64 | Low price |
| close | Float64 | Close price |
| volume | Int64 | Volume |

### Derived Columns (Computed by Pipeline)
| Column | Type | How |
|--------|------|-----|
| day_id | Int32 | Sequential day number from date |
| time_minutes | Int32 | Minutes from midnight IST. **Cast to Int32 BEFORE multiplication** — Polars dt.hour() returns Int8, and 9×60=540 overflows Int8 max of 127. |
| vix | Float64 | Joined from VIX data, forward-filled |
| index_close | Float64 | Joined from NIFTY/BANKNIFTY data |

### Coverage
- 206 stocks, all NSE F&O segment
- ~388,000 bars per stock (3 years of 1-min data)
- Date range: 2022-01-01 to 2025-present
- Train: 2022-01-01 to 2024-12-31
- Test: 2025-01-01 to present

## 5-Second Options Data (Current)

### Location
```
5second_data/spot_candles.parquet
5second_data/option_candles.parquet
```

### Spot Candles Schema
162,000 rows | 3 symbols | 12 dates

| Column | Type | Description |
|--------|------|-------------|
| symbol | Varchar | `NSE:NIFTY50-INDEX`, `NSE:NIFTYBANK-INDEX`, `NSE:INDIAVIX-INDEX` |
| session_date | Date | Trading date |
| epoch | BigInt | Unix epoch seconds (UTC) |
| ts | Timestamp | UTC timestamp from epoch |
| open | Float64 | Open |
| high | Float64 | High |
| low | Float64 | Low |
| close | Float64 | Close |
| volume | BigInt | Volume |

4,500 candles per symbol per session. 54,000 rows per symbol total.

### Option Candles Schema
2,671,248 rows | 284 contracts | 12 dates | 4 expiries

| Column | Type | Description |
|--------|------|-------------|
| symbol | Varchar | e.g., `NSE:NIFTY2631024250CE` |
| underlying | Varchar | `NSE:NIFTY50-INDEX` or `NSE:NIFTYBANK-INDEX` |
| session_date | Date | Trading date |
| expiry | Date | Option expiry date |
| strike | Float64 | Strike price |
| option_type | Varchar | `CE` or `PE` |
| epoch | BigInt | Unix epoch seconds (UTC) |
| ts | Timestamp | UTC timestamp |
| open | Float64 | Open premium |
| high | Float64 | High premium |
| low | Float64 | Low premium |
| close | Float64 | Close premium |
| volume | BigInt | Traded volume |
| open_interest | BigInt | Open interest |

**MISSING columns**: bid, ask, bid_qty, ask_qty, iv, delta, gamma, theta, vega. These are NOT in the data. Spread cost is estimated from H-L proxy. Greeks must be computed if needed (Black-Scholes with estimated IV).

### Coverage
- NIFTY: 160 contracts, 3 expiries, 43 strikes
- BANKNIFTY: 124 contracts, 1 expiry, 62 strikes
- Date range: 2026-03-04 to 2026-03-19 (12 trading days)

## UTC→IST Conversion (CRITICAL)

All timestamps in both datasets are in UTC. IST = UTC + 5:30.

Market hours 09:15-15:30 IST = 03:45-10:00 UTC.

### The Correct Way (Polars)
```python
df = df.with_columns(
    pl.col("ts")
      .dt.replace_time_zone("UTC")
      .dt.convert_time_zone("Asia/Kolkata")
      .dt.replace_time_zone(None)
      .alias("ts_ist")
)
```

### The WRONG Way
```python
# WRONG — strips UTC label without converting
df = df.with_columns(pl.col("ts").dt.replace_time_zone(None))
```

This keeps the numeric timestamp as-is but removes the timezone label. 09:15 IST appears as 03:45.

### Verification Test
After conversion, assert:
```python
first_bar_hour = df["ts_ist"].dt.hour()[0]
assert first_bar_hour == 9, f"Expected 9, got {first_bar_hour}"
```

This single assertion catches the timezone bug. It should be in every test suite.

### time_minutes Computation (Int8 Overflow)
```python
# WRONG — Int8 overflow
df = df.with_columns(
    (pl.col("ts_ist").dt.hour() * 60 + pl.col("ts_ist").dt.minute()).alias("time_minutes")
)
# dt.hour() returns Int8. 9 * 60 = 540 overflows Int8 max (127). Silently wraps to 28.

# CORRECT — cast first
df = df.with_columns(
    (pl.col("ts_ist").dt.hour().cast(pl.Int32) * 60 + pl.col("ts_ist").dt.minute()).alias("time_minutes")
)
```

## Joining Spot + Options

```sql
-- DuckDB example
SELECT s.ts, s.close AS spot, c.close AS call_prem
FROM spot_candles s
JOIN option_candles c
  ON c.epoch = s.epoch
 AND c.session_date = s.session_date
 AND c.underlying = 'NSE:NIFTY50-INDEX'
 AND c.strike = 24000
 AND c.option_type = 'CE'
WHERE s.symbol = 'NSE:NIFTY50-INDEX'
ORDER BY s.epoch;
```

## Decisions Made
- Parquet over CSV everywhere (5-10x faster reads)
- Forward-fill NaN in Polars BEFORE converting to numpy
- day_id computed from date, not from gaps in timestamps (handles market holidays correctly)
- VIX forward-filled to match 5-second bar frequency (VIX updates every 15 seconds)

## Pitfalls & Anti-patterns
- **Assuming column names match between datasets**: Equity has `datetime`, options has `ts`. Always read the actual parquet schema.
- **Int8 overflow in time_minutes**: Polars `dt.hour()` returns Int8. 9 × 60 = 540 > 127. Cast to Int32 first.
- **Using `replace_time_zone(None)` alone**: This strips without converting. Must `convert_time_zone` first.
- **Treating session_date as IST**: The `session_date` column IS in IST (it's the trading date). But `ts` is UTC. Don't mix them without conversion.

## Corrections Log
- ~~Data had bid/ask columns for spread computation~~ → No bid/ask in the data. Spread is estimated from H-L proxy or assumed.
- ~~time_minutes computed without Int32 cast~~ → Int8 overflow caught during options pipeline audit. Cast required.
- ~~NIFTY strikes at 50-point intervals assumed everywhere~~ → Actually varies. Strike step should be derived from data, not hardcoded.
