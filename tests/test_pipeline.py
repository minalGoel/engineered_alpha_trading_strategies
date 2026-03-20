"""Unit tests for the 5-second index option trading pipeline.

Tests the CURRENT pipeline (not the old equity pipeline).
Run with: python -m pytest tests/test_pipeline.py -v
Or simply: python tests/test_pipeline.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import polars as pl
from datetime import datetime, timedelta


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_synthetic_spot(n: int = 1000, seed: int = 42) -> pl.DataFrame:
    """Create synthetic spot data matching the option pipeline schema."""
    rng = np.random.RandomState(seed)
    base = 24000.0
    close = base + np.cumsum(rng.randn(n) * 2)
    close = np.maximum(close, 20000.0)

    bars_per_day = n // 5
    datetimes = []
    day_ids = []
    time_minutes = []
    session_dates = []

    for day in range(5):
        start = datetime(2026, 3, 4 + day, 9, 15)
        sd = start.date()
        for bar in range(min(bars_per_day, n - day * bars_per_day)):
            dt = start + timedelta(seconds=bar * 5)
            datetimes.append(dt)
            day_ids.append(day)
            time_minutes.append(dt.hour * 60 + dt.minute)
            session_dates.append(sd)

    datetimes = datetimes[:n]
    day_ids = day_ids[:n]
    time_minutes = time_minutes[:n]
    session_dates = session_dates[:n]

    return pl.DataFrame({
        "datetime": datetimes,
        "session_date": session_dates,
        "open": close + rng.randn(n) * 0.5,
        "high": close + np.abs(rng.randn(n) * 1),
        "low": close - np.abs(rng.randn(n) * 1),
        "close": close[:n],
        "volume": np.abs(rng.randn(n) * 1000 + 5000).astype(np.int64),
        "day_id": np.array(day_ids, dtype=np.int32),
        "time_minutes": np.array(time_minutes, dtype=np.int32),
        "atm_strike": np.floor(close[:n] / 50 + 0.5) * 50,
    })


# ── Tests ────────────────────────────────────────────────────────────────────

def test_atm_strike_no_bankers_rounding():
    """BUG #1 fix: ATM strike must not use banker's rounding."""
    from pipeline.option_utils import nearest_strike, atm_strike_series

    # Midpoint: 24025 is exactly between 24000 and 24050
    assert nearest_strike(24025, 50) == 24050, "24025 should round UP to 24050"
    assert nearest_strike(24075, 50) == 24100, "24075 should round UP to 24100"

    # Non-midpoints
    assert nearest_strike(24024, 50) == 24000
    assert nearest_strike(24026, 50) == 24050
    assert nearest_strike(24074, 50) == 24050
    assert nearest_strike(24076, 50) == 24100

    # Vectorised version
    arr = np.array([24025.0, 24075.0, 24024.0, 24026.0])
    result = atm_strike_series(arr, 50)
    expected = np.array([24050.0, 24100.0, 24000.0, 24050.0])
    assert np.array_equal(result, expected), f"Vectorised ATM mismatch: {result} vs {expected}"

    print("  ATM strike no banker's rounding: passed")
    return True


def test_state_machine_basic():
    """State machine: basic long CE trade with stop and target."""
    from pipeline.state_machine import run_state_machine, warmup_numba
    warmup_numba()

    n = 100
    spot = np.full(n, 24000.0, dtype=np.float64)
    ce_prem = np.full(n, 100.0, dtype=np.float64)
    pe_prem = np.full(n, 100.0, dtype=np.float64)
    day_ids = np.zeros(n, dtype=np.int32)
    time_mins = np.full(n, 600, dtype=np.int32)  # 10:00 IST
    buy_ce = np.zeros(n, dtype=np.bool_)
    buy_pe = np.zeros(n, dtype=np.bool_)
    sell_ce = np.zeros(n, dtype=np.bool_)
    sell_pe = np.zeros(n, dtype=np.bool_)
    stops = np.full(n, 5.0, dtype=np.float64)
    targets = np.full(n, 10.0, dtype=np.float64)
    is_exp = np.zeros(n, dtype=np.bool_)

    # Entry at bar 10, premium goes up to trigger target at bar 15
    buy_ce[10] = True
    ce_prem[10] = 100.0
    for i in range(11, 20):
        ce_prem[i] = 100.0 + (i - 10) * 2.5  # rises 2.5 pts per bar

    result = run_state_machine(
        spot, ce_prem, pe_prem, day_ids, time_mins,
        buy_ce, buy_pe, sell_ce, sell_pe, stops, targets,
        24, 925, 560, 925, 10, 75, 5, is_exp, 920,
    )
    entry_bars, exit_bars, sides, ep, xp, reasons, pnls, tc = result
    assert tc >= 1, f"Expected at least 1 trade, got {tc}"
    assert entry_bars[0] == 10
    assert sides[0] == 1  # CE
    assert ep[0] == 100.0

    print(f"  State machine basic: {tc} trades OK")
    return True


def test_state_machine_zero_signals():
    """State machine: zero signals → zero trades, no crash."""
    from pipeline.state_machine import run_state_machine, warmup_numba
    warmup_numba()

    n = 100
    result = run_state_machine(
        np.full(n, 24000.0), np.full(n, 100.0), np.full(n, 100.0),
        np.zeros(n, dtype=np.int32), np.full(n, 600, dtype=np.int32),
        np.zeros(n, dtype=np.bool_), np.zeros(n, dtype=np.bool_),
        np.zeros(n, dtype=np.bool_), np.zeros(n, dtype=np.bool_),
        np.full(n, 5.0), np.full(n, 10.0),
        24, 925, 560, 925, 10, 75, 5,
        np.zeros(n, dtype=np.bool_), 920,
    )
    assert result[7] == 0, "Zero signals should produce zero trades"
    print("  State machine zero signals: passed")
    return True


def test_state_machine_force_close():
    """State machine: open position at end of data is force-closed."""
    from pipeline.state_machine import run_state_machine, warmup_numba
    warmup_numba()

    n = 50
    spot = np.full(n, 24000.0, dtype=np.float64)
    ce_prem = np.full(n, 100.0, dtype=np.float64)
    buy_ce = np.zeros(n, dtype=np.bool_)
    buy_ce[10] = True

    result = run_state_machine(
        spot, ce_prem, np.full(n, 100.0),
        np.zeros(n, dtype=np.int32), np.full(n, 600, dtype=np.int32),
        buy_ce, np.zeros(n, dtype=np.bool_),
        np.zeros(n, dtype=np.bool_), np.zeros(n, dtype=np.bool_),
        np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64),  # no stop/target
        0,  # no time stop
        925, 560, 925, 10, 75, 5,
        np.zeros(n, dtype=np.bool_), 920,
    )
    assert result[7] == 1, f"Expected 1 force-closed trade, got {result[7]}"
    assert result[1][0] == n - 1, f"Exit should be at last bar"
    print("  State machine force-close: passed")
    return True


def test_state_machine_expiry_flatten():
    """State machine: position flattened at 15:20 on expiry day."""
    from pipeline.state_machine import run_state_machine, warmup_numba
    warmup_numba()

    n = 50
    spot = np.full(n, 24000.0, dtype=np.float64)
    ce_prem = np.full(n, 100.0, dtype=np.float64)
    buy_ce = np.zeros(n, dtype=np.bool_)
    buy_ce[10] = True

    time_mins = np.full(n, 600, dtype=np.int32)
    time_mins[20] = 920  # 15:20 IST
    is_exp = np.ones(n, dtype=np.bool_)  # all bars on expiry

    result = run_state_machine(
        spot, ce_prem, np.full(n, 100.0),
        np.zeros(n, dtype=np.int32), time_mins,
        buy_ce, np.zeros(n, dtype=np.bool_),
        np.zeros(n, dtype=np.bool_), np.zeros(n, dtype=np.bool_),
        np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64),
        0, 925, 560, 925, 10, 75, 5,
        is_exp, 920,
    )
    assert result[7] == 1
    assert result[1][0] == 20, f"Should flatten at bar 20 (15:20), got {result[1][0]}"
    print("  State machine expiry flatten: passed")
    return True


def test_state_machine_no_entry_before_session():
    """State machine: no entry before session start."""
    from pipeline.state_machine import run_state_machine, warmup_numba
    warmup_numba()

    n = 50
    buy_ce = np.zeros(n, dtype=np.bool_)
    buy_ce[5] = True  # bar 5: time=550 (before session start 560)

    time_mins = np.array([550 + i for i in range(n)], dtype=np.int32)

    result = run_state_machine(
        np.full(n, 24000.0), np.full(n, 100.0), np.full(n, 100.0),
        np.zeros(n, dtype=np.int32), time_mins,
        buy_ce, np.zeros(n, dtype=np.bool_),
        np.zeros(n, dtype=np.bool_), np.zeros(n, dtype=np.bool_),
        np.full(n, 5.0), np.full(n, 10.0),
        24, 925, 560, 925, 10, 75, 0,  # warmup=0 so it's not the warmup blocking
        np.zeros(n, dtype=np.bool_), 920,
    )
    # bar 5 has time_minutes=555 < 560 → should not enter
    if result[7] > 0:
        assert result[0][0] != 5, "Should not enter before session start"
    print("  State machine no entry before session: passed")
    return True


def test_cost_model():
    """Cost model: verify exact numbers for a sample trade."""
    from pipeline.cost_model import compute_trade_costs

    r = compute_trade_costs(150.0, 153.0, 75, lots=1)
    assert abs(r["gross_pnl"] - 225.0) < 0.01
    assert abs(r["spread_cost"] - 225.0) < 0.01
    assert abs(r["stt"] - 7.171875) < 0.01
    assert abs(r["brokerage"] - 40.0) < 0.01
    assert abs(r["exchange"] - 10.0) < 0.01
    assert abs(r["net_pnl"] - (-57.171875)) < 0.01

    print("  Cost model: passed")
    return True


def test_metrics_uses_net_pnl():
    """BUG #3/#4/#5 fix: Sharpe and drawdown must use NET PnL."""
    from pipeline.metrics import compute_metrics
    from pipeline.cost_model import net_pnl_quick

    trades = pl.DataFrame({
        "entry_premium": [100.0, 100.0, 100.0],
        "exit_premium": [110.0, 95.0, 108.0],
        "pnl": [750.0, -375.0, 600.0],  # gross
        "entry_time": [datetime(2026, 3, 4, 10, 0), datetime(2026, 3, 5, 10, 0),
                       datetime(2026, 3, 6, 10, 0)],
        "exit_time": [datetime(2026, 3, 4, 11, 0), datetime(2026, 3, 5, 11, 0),
                      datetime(2026, 3, 6, 11, 0)],
        "side": [np.int8(1), np.int8(1), np.int8(1)],
        "holding_bars": [10, 10, 10],
        "exit_reason": ["TARGET", "STOP", "TARGET"],
    })

    m = compute_metrics(trades, lot_size=75, total_trading_days=3)

    # Total PnL should be NET, not gross
    expected_net = sum(net_pnl_quick(ep, xp, 75) for ep, xp in
                       zip([100, 100, 100], [110, 95, 108]))
    assert abs(m["total_pnl"] - expected_net) < 1.0, (
        f"total_pnl should be net ({expected_net:.0f}), got {m['total_pnl']:.0f}"
    )

    print(f"  Metrics uses net PnL: total_pnl={m['total_pnl']:.0f} (expected {expected_net:.0f})")
    return True


def test_utc_to_ist():
    """Data loader: UTC timestamps correctly converted to IST."""
    from pipeline.data_loader import _utc_to_ist

    df = pl.DataFrame({
        "ts": [datetime(2026, 3, 4, 3, 45, 0)],  # 03:45 UTC
    })
    result = _utc_to_ist(df)
    ist_hour = result["datetime"][0].hour
    assert ist_hour == 9, f"Expected IST hour 9, got {ist_hour}"
    ist_minute = result["datetime"][0].minute
    assert ist_minute == 15, f"Expected IST minute 15, got {ist_minute}"
    print("  UTC to IST: passed")
    return True


def test_time_minutes_no_overflow():
    """Data loader: time_minutes uses Int32, not Int8 (overflow bug fix)."""
    from pipeline.data_loader import _add_time_columns

    df = pl.DataFrame({
        "datetime": [datetime(2026, 3, 4, 15, 25, 0)],  # 15:25 IST → 925 minutes
        "session_date": [datetime(2026, 3, 4).date()],
    })
    result = _add_time_columns(df)
    tm = result["time_minutes"][0]
    assert tm == 925, f"15:25 IST should be 925 minutes, got {tm}"
    # Int8 would overflow: 15*60=900 > 127
    assert result["time_minutes"].dtype == pl.Int32, (
        f"time_minutes should be Int32, got {result['time_minutes'].dtype}"
    )
    print("  time_minutes no overflow: passed")
    return True


def test_option_utils_time_to_expiry():
    """Time to expiry: correct for Wednesday and Thursday."""
    from pipeline.option_utils import time_to_expiry_hours

    bar_ts = np.array(['2026-03-09T10:00:00'], dtype='datetime64[us]')
    expiry = np.array(['2026-03-10'], dtype='datetime64[D]')
    hours = time_to_expiry_hours(bar_ts, expiry)
    assert abs(hours[0] - 29.5) < 0.01, f"Expected 29.5 hours, got {hours[0]}"

    bar_ts2 = np.array(['2026-03-10T14:00:00'], dtype='datetime64[us]')
    hours2 = time_to_expiry_hours(bar_ts2, expiry)
    assert abs(hours2[0] - 1.5) < 0.01, f"Expected 1.5 hours, got {hours2[0]}"

    print("  Time to expiry: passed")
    return True


def test_json_nan_handling():
    """orjson must not crash on NaN/Infinity values."""
    from pipeline.run_all import _sanitise_for_json
    import orjson

    data = {
        "sharpe": float("nan"),
        "pf": float("inf"),
        "nested": {"val": float("-inf"), "ok": 1.5},
        "list": [1.0, float("nan"), 3.0],
    }
    sanitised = _sanitise_for_json(data)
    assert sanitised["sharpe"] is None
    assert sanitised["pf"] is None
    assert sanitised["nested"]["val"] is None
    assert sanitised["nested"]["ok"] == 1.5
    assert sanitised["list"][1] is None

    result = orjson.dumps(sanitised)
    assert b"null" in result

    print("  JSON NaN handling: passed")
    return True


# ── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Pipeline Unit Tests (5-Second Option Trading)")
    print("=" * 60)

    tests = [
        ("ATM strike no banker's rounding", test_atm_strike_no_bankers_rounding),
        ("State machine basic", test_state_machine_basic),
        ("State machine zero signals", test_state_machine_zero_signals),
        ("State machine force-close", test_state_machine_force_close),
        ("State machine expiry flatten", test_state_machine_expiry_flatten),
        ("State machine no entry before session", test_state_machine_no_entry_before_session),
        ("Cost model", test_cost_model),
        ("Metrics uses net PnL", test_metrics_uses_net_pnl),
        ("UTC to IST", test_utc_to_ist),
        ("time_minutes no overflow", test_time_minutes_no_overflow),
        ("Time to expiry", test_option_utils_time_to_expiry),
        ("JSON NaN handling", test_json_nan_handling),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            print(f"\n[TEST] {name}...")
            if test_fn():
                print(f"  PASSED")
                passed += 1
            else:
                print(f"  FAILED")
                failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
    sys.exit(0 if failed == 0 else 1)
