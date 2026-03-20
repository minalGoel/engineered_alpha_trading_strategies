"""Smoke test: Load each strategy, compute signals on synthetic data, verify shapes."""
import sys
import numpy as np
import polars as pl
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.strategies.loader import list_available, load_strategy


def make_synthetic_df(n_bars: int = 2000, n_days: int = 5) -> pl.DataFrame:
    """Create a synthetic OHLCV DataFrame for testing."""
    np.random.seed(42)
    bars_per_day = n_bars // n_days

    datetimes = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    vix_vals = []
    index_close_vals = []
    day_ids = []
    time_minutes_arr = []

    price = 1000.0
    idx_price = 18000.0

    for d in range(n_days):
        for b in range(bars_per_day):
            minute = 555 + b  # 09:15 + bar number
            if minute > 929:  # cap at 15:29
                minute = 929

            ret = np.random.normal(0, 0.001)
            price *= (1 + ret)
            idx_ret = np.random.normal(0, 0.0005)
            idx_price *= (1 + idx_ret)

            o = price * (1 + np.random.normal(0, 0.0003))
            h = max(o, price) * (1 + abs(np.random.normal(0, 0.0005)))
            l = min(o, price) * (1 - abs(np.random.normal(0, 0.0005)))
            c = price
            v = max(1, int(np.random.lognormal(10, 1)))

            from datetime import datetime, timedelta
            dt = datetime(2024, 1, 1 + d, minute // 60, minute % 60)

            datetimes.append(dt)
            opens.append(o)
            highs.append(h)
            lows.append(l)
            closes.append(c)
            volumes.append(v)
            vix_vals.append(15.0 + np.random.normal(0, 2))
            index_close_vals.append(idx_price)
            day_ids.append(d)
            time_minutes_arr.append(minute)

    return pl.DataFrame({
        "datetime": datetimes,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "vix": vix_vals,
        "index_close": index_close_vals,
        "day_id": day_ids,
        "time_minutes": time_minutes_arr,
    })


def smoke_test_all():
    """Run smoke test on all available strategies."""
    df = make_synthetic_df()
    n = len(df)
    available = list_available()

    print(f"Found {len(available)} strategy modules")
    print(f"Synthetic data: {n} bars, {df['day_id'].n_unique()} days")
    print()

    passed = 0
    failed = 0

    for name in available:
        strategy = load_strategy(name)
        if strategy is None:
            print(f"  FAIL  {name}: could not load")
            failed += 1
            continue

        try:
            params = {tp.name: tp.default for tp in strategy.tunable_params()}
            signals = strategy.compute(df, params)

            # Verify shapes
            assert len(signals.long_entry) == n, f"long_entry len {len(signals.long_entry)} != {n}"
            assert len(signals.short_entry) == n, f"short_entry len {len(signals.short_entry)} != {n}"
            assert len(signals.signal_exit_long) == n
            assert len(signals.signal_exit_short) == n
            assert len(signals.atr_arr) == n
            assert len(signals.target_indicator) == n

            # Check types
            assert signals.long_entry.dtype == np.bool_, f"long_entry dtype {signals.long_entry.dtype}"
            assert signals.short_entry.dtype == np.bool_

            # Count signals
            n_long = signals.long_entry.sum()
            n_short = signals.short_entry.sum()

            status = "PASS"
            if n_long == 0 and n_short == 0:
                status = "WARN (no signals)"
            elif n_long == n or n_short == n:
                status = "WARN (all-True)"

            print(f"  {status:20s} {name:40s} long={n_long:5d}  short={n_short:5d}")
            passed += 1

        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {len(available)} strategies")
    return failed == 0


if __name__ == "__main__":
    success = smoke_test_all()
    sys.exit(0 if success else 1)
