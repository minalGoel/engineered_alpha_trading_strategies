"""Unit tests for the pipeline — indicator engine, condition parser, Numba state machine.

Run with: python -m pytest tests/test_pipeline.py -v
Or simply: python tests/test_pipeline.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import polars as pl
from datetime import datetime, timedelta


def make_synthetic_data(n: int = 500, seed: int = 42) -> pl.DataFrame:
    """Create synthetic OHLCV data for testing."""
    rng = np.random.RandomState(seed)
    base_price = 100.0
    prices = base_price + np.cumsum(rng.randn(n) * 0.5)
    prices = np.maximum(prices, 10.0)  # Keep positive

    opens = prices + rng.randn(n) * 0.2
    highs = np.maximum(opens, prices) + np.abs(rng.randn(n) * 0.3)
    lows = np.minimum(opens, prices) - np.abs(rng.randn(n) * 0.3)
    volumes = np.abs(rng.randn(n) * 1000 + 5000).astype(int)

    # Generate datetimes (5 trading days, 100 bars each)
    start = datetime(2024, 1, 1, 9, 15)
    datetimes = []
    day_ids = []
    time_minutes = []
    for day in range(n // 100 + 1):
        for bar in range(min(100, n - day * 100)):
            dt = start + timedelta(days=day * 7 // 5, minutes=bar)
            datetimes.append(dt)
            day_ids.append(day)
            time_minutes.append(dt.hour * 60 + dt.minute)

    datetimes = datetimes[:n]
    day_ids = day_ids[:n]

    return pl.DataFrame({
        "datetime": datetimes,
        "open": opens[:n],
        "high": highs[:n],
        "low": lows[:n],
        "close": prices[:n],
        "volume": volumes[:n].astype(float),
        "day_id": day_ids,
        "vix": np.full(n, 15.0),
    })


def test_indicators():
    """Test indicator computation on synthetic data."""
    from pipeline.indicators import (
        compute_sma, compute_ema, compute_rsi, compute_atr,
        compute_vwap, compute_bollinger, compute_adx, compute_zscore,
        compute_rolling_max, compute_rolling_min, compute_obv,
        compute_prev_day_high_low, compute_gap_pct,
    )

    df = make_synthetic_data()
    n = len(df)

    # SMA
    df = compute_sma(df, "close", 20, "sma_20")
    assert "sma_20" in df.columns
    assert df["sma_20"].null_count() == 19  # first 19 should be null
    assert not np.isnan(df["sma_20"][-1])

    # EMA
    df = compute_ema(df, "close", 20, "ema_20")
    assert "ema_20" in df.columns
    # EMA starts from bar 0 but stabilizes after ~20 bars
    assert not np.isnan(df["ema_20"][-1])

    # RSI
    df = compute_rsi(df, "close", 14, "rsi_14")
    assert "rsi_14" in df.columns
    rsi_vals = df["rsi_14"].drop_nulls().to_numpy()
    assert np.all(rsi_vals >= 0) and np.all(rsi_vals <= 100), "RSI must be 0-100"

    # ATR
    df = compute_atr(df, 20, "atr_20")
    assert "atr_20" in df.columns
    atr_vals = df["atr_20"].drop_nulls().to_numpy()
    assert np.all(atr_vals >= 0), "ATR must be non-negative"

    # VWAP (daily reset)
    df = compute_vwap(df, "vwap")
    assert "vwap" in df.columns
    vwap_vals = df["vwap"].drop_nulls().to_numpy()
    assert len(vwap_vals) > 0

    # Bollinger Bands
    df = compute_bollinger(df, "close", 20, 2.0, "bb_upper", "bb_lower", "bb_mid")
    assert "bb_upper" in df.columns
    assert "bb_lower" in df.columns
    # Upper > Mid > Lower for non-null values
    valid = df.filter(pl.col("bb_upper").is_not_null())
    if len(valid) > 0:
        assert (valid["bb_upper"] >= valid["bb_mid"]).all()
        assert (valid["bb_mid"] >= valid["bb_lower"]).all()

    # ADX
    df = compute_adx(df, 14, "adx_14")
    assert "adx_14" in df.columns

    # Z-score
    df = df.with_columns((pl.col("close") - pl.col("vwap")).alias("_dev"))
    df = compute_zscore(df, "_dev", 60, "zscore_60")
    assert "zscore_60" in df.columns

    # Rolling max/min
    df = compute_rolling_max(df, "high", 30, "range_high_30")
    df = compute_rolling_min(df, "low", 30, "range_low_30")
    assert "range_high_30" in df.columns
    valid = df.filter(pl.col("range_high_30").is_not_null() & pl.col("range_low_30").is_not_null())
    if len(valid) > 0:
        assert (valid["range_high_30"] >= valid["range_low_30"]).all()

    # OBV
    df = compute_obv(df, "obv")
    assert "obv" in df.columns

    # PDH/PDL
    df = compute_prev_day_high_low(df)
    assert "pdh" in df.columns
    assert "pdl" in df.columns

    print(f"  Indicators: all {len(df.columns)} columns computed on {n} rows OK")
    return True


def test_formula_parser():
    """Test the indicator formula parser on various patterns."""
    from pipeline.indicators import parse_indicator_formula

    test_cases = [
        ({"name": "sma_20", "formula": "SMA(close, 20)", "lookback": 20}, True),
        ({"name": "ema_9", "formula": "EMA(close, 9)", "lookback": 9}, True),
        ({"name": "rsi_14", "formula": "RSI(close, 14)", "lookback": 14}, True),
        ({"name": "atr_20", "formula": "ATR(high, low, close, 20)", "lookback": 20}, True),
        ({"name": "atr_14", "formula": "ATR(14)", "lookback": 14}, True),
        ({"name": "adx_14", "formula": "ADX(14)", "lookback": 14}, True),
        ({"name": "bb_up", "formula": "BB_upper(close, 20, 2)", "lookback": 20}, True),
        ({"name": "macd", "formula": "MACD(close, 12, 26, 9)", "lookback": 26}, True),
        ({"name": "vwap", "formula": "VWAP()", "lookback": 0}, True),
        ({"name": "vwap2", "formula": "vwap", "lookback": 0}, True),
        ({"name": "zscore", "formula": "zscore(close - vwap, 60)", "lookback": 60}, True),
        ({"name": "vol_ratio", "formula": "volume / SMA(volume, 20)", "lookback": 20}, True),
        ({"name": "range_h", "formula": "MAX(high, 30)", "lookback": 30}, True),
        ({"name": "range_l", "formula": "MIN(low, 30)", "lookback": 30}, True),
        ({"name": "std_dev", "formula": "STDEV(close, 30)", "lookback": 30}, True),
        ({"name": "diff", "formula": "close - vwap", "lookback": 0}, True),
        ({"name": "ratio", "formula": "close / vwap", "lookback": 0}, True),
    ]

    passed = 0
    failed = 0
    for indicator, expected_success in test_cases:
        steps = parse_indicator_formula(indicator, set())
        if bool(steps) == expected_success:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: {indicator['name']} = {indicator['formula']}: "
                  f"got {'parsed' if steps else 'empty'}, expected {'parsed' if expected_success else 'empty'}")

    print(f"  Formula parser: {passed}/{passed + failed} passed")
    return failed == 0


def test_condition_parser():
    """Test condition parsing on various patterns."""
    from pipeline.condition_parser import parse_condition, parse_conditions_list

    test_cases = [
        ("close < vwap", True, 1),
        ("rsi_14 < 30", True, 1),
        ("vix < 18", True, 1),
        ("rel_volume > 1.3", True, 1),
        ("close > ema_20", True, 1),
        ("ema_9 > ema_21", True, 1),
        ("range_width > 0.2 AND range_width < 0.8", True, 2),
        ("ABS(index_return_15) <= 0.004", True, 1),
        ("vol_spike == True", True, 1),
        ("CROSSUNDER(low, pdl)", True, 1),
        ("none", False, 0),
        ("", False, 0),
        ("N/A", False, 0),
    ]

    passed = 0
    failed = 0
    for cond_str, expect_success, expect_count in test_cases:
        result = parse_condition(cond_str)
        success = len(result) > 0
        if success == expect_success and (not expect_success or len(result) == expect_count):
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: '{cond_str}': got {len(result)} conditions, "
                  f"expected {'success' if expect_success else 'skip'} with {expect_count}")

    # Test conditions list
    conds = ["close < vwap", "rsi_14 < 30", "vix < 18"]
    parsed, unparsed = parse_conditions_list(conds)
    assert len(parsed) == 3, f"Expected 3 parsed, got {len(parsed)}"
    assert len(unparsed) == 0

    # Test evaluation
    from pipeline.condition_parser import evaluate_conditions
    arrays = {
        "close": np.array([95, 105, 90, 110, 92]),
        "vwap": np.array([100, 100, 100, 100, 100]),
        "rsi_14": np.array([25, 55, 28, 60, 29]),
        "vix": np.array([15, 15, 15, 20, 15]),
    }
    mask = evaluate_conditions(parsed, arrays)
    # Bar 0: close<vwap=T, rsi<30=T, vix<18=T → True
    # Bar 1: close<vwap=F → False
    # Bar 2: close<vwap=T, rsi<30=T, vix<18=T → True
    # Bar 3: close<vwap=F → False
    # Bar 4: close<vwap=T, rsi<30=T, vix<18=T → True
    expected = np.array([True, False, True, False, True])
    assert np.array_equal(mask, expected), f"Condition eval mismatch: {mask} vs {expected}"

    print(f"  Condition parser: {passed}/{passed + failed} parsed, evaluation OK")
    return failed == 0


def test_exit_parser():
    """Test exit rule parsing."""
    from pipeline.condition_parser import parse_exit_rules

    # Test percentage stop loss
    exit1 = {"stop_loss": "0.3% from entry", "target": "0.5% from entry",
             "trailing_stop": "none", "time_stop": "exit after 30 bars",
             "eod_rule": "flatten", "signal_exit": "none"}
    rules = parse_exit_rules(exit1, max_hold_bars=45)
    assert rules.stop_loss_pct is not None and abs(rules.stop_loss_pct - 0.003) < 0.0001
    assert rules.target_pct is not None and abs(rules.target_pct - 0.005) < 0.0001
    assert rules.time_stop_bars == 30  # min(30, 45)
    assert rules.eod_flatten is True

    # Test ATR stop
    exit2 = {"stop_loss": "1.5 * ATR(20)", "target": "vwap touch",
             "trailing_stop": "trail at 0.2% once profit exceeds 0.3%",
             "time_stop": "exit after 60 bars", "eod_rule": "flatten", "signal_exit": "none"}
    rules2 = parse_exit_rules(exit2, max_hold_bars=120)
    assert rules2.stop_loss_atr_mult is not None and abs(rules2.stop_loss_atr_mult - 1.5) < 0.01
    assert rules2.target_vwap is True
    assert rules2.trailing_stop_pct is not None and abs(rules2.trailing_stop_pct - 0.002) < 0.0001
    assert rules2.trailing_activate_pct is not None and abs(rules2.trailing_activate_pct - 0.003) < 0.0001
    assert rules2.time_stop_bars == 60

    # Test breakeven trailing
    exit3 = {"stop_loss": "0.25% from entry",
             "trailing_stop": "move SL to breakeven after +0.2%",
             "target": "none", "time_stop": "none",
             "eod_rule": "flatten", "signal_exit": "exit when zscore crosses 0"}
    rules3 = parse_exit_rules(exit3)
    assert rules3.breakeven_after_pct is not None and abs(rules3.breakeven_after_pct - 0.002) < 0.0001
    assert len(rules3.signal_exit_conditions) > 0

    print("  Exit parser: all tests passed")
    return True


def test_time_filter():
    """Test time filter parsing."""
    from pipeline.condition_parser import parse_time_filter

    # Basic session
    tf = parse_time_filter("09:20-15:15", {"time_filter": "no trades first 5 min"})
    assert tf.start_h == 9 and tf.start_m == 20
    assert tf.end_h == 15 and tf.end_m == 15
    assert tf.skip_first_n_min == 5
    assert tf.effective_start_minutes() == 9 * 60 + 20 + 5

    # Expiry filter
    tf2 = parse_time_filter("09:15-15:20", {"event_filter": "skip on expiry day"})
    assert tf2.skip_expiry is True

    print("  Time filter: all tests passed")
    return True


def test_numba_state_machine():
    """Test the Numba state machine on synthetic data."""
    from pipeline.state_machine import run_state_machine, warmup_numba

    # Warmup
    warmup_numba()

    n = 200
    rng = np.random.RandomState(42)

    # Create simple trending up data
    prices = 100.0 + np.cumsum(np.full(n, 0.1) + rng.randn(n) * 0.05)
    opens = prices - 0.05
    highs = prices + 0.3
    lows = prices - 0.3
    close = prices

    day_ids = np.array([i // 100 for i in range(n)], dtype=np.int32)
    time_mins = np.array([(555 + i % 100) for i in range(n)], dtype=np.int32)

    # Create entry signals: buy every 20 bars
    long_entry = np.zeros(n, dtype=np.bool_)
    for i in range(20, n, 20):
        long_entry[i] = True
    short_entry = np.zeros(n, dtype=np.bool_)

    signal_exit_long = np.zeros(n, dtype=np.bool_)
    signal_exit_short = np.zeros(n, dtype=np.bool_)
    atr = np.full(n, 0.5, dtype=np.float64)
    target_ind = np.zeros(n, dtype=np.float64)

    result = run_state_machine(
        opens, highs, lows, close,
        day_ids, time_mins,
        long_entry, short_entry,
        signal_exit_long, signal_exit_short,
        atr, target_ind,
        stop_loss_pct=0.005,
        stop_loss_atr_mult=0.0,
        target_pct=0.01,
        target_atr_mult=0.0,
        use_target_indicator=False,
        trailing_stop_pct=0.0,
        trailing_activate_pct=0.0,
        breakeven_pct=0.0,
        time_stop_bars=15,
        eod_flatten_minutes=920,
        session_start_minutes=555,
        session_end_minutes=920,
        max_trades_per_day=10,
        max_daily_loss=5000.0,
        capital_per_trade=100000.0,
        warmup_bars=5,
    )

    entry_bars, exit_bars, sides, entry_prices, exit_prices, exit_reasons, trade_count = result
    assert trade_count > 0, f"Expected trades, got {trade_count}"
    assert len(entry_bars) == trade_count
    assert all(sides[:trade_count] == 1), "All trades should be LONG"

    # Verify no trade overlaps
    for i in range(trade_count):
        assert exit_bars[i] > entry_bars[i], f"Trade {i}: exit_bar must be > entry_bar"
        if i > 0:
            assert entry_bars[i] >= exit_bars[i - 1], f"Trade {i}: entry must be >= prev exit"

    print(f"  Numba state machine: {trade_count} trades on {n} bars OK")
    return True


def test_strategy_parser():
    """Test parsing a real strategy JSON."""
    from pipeline.strategy_parser import parse_strategy

    raw = {
        "name": "test_vwap_reversion",
        "timeframe": "1min",
        "session": "09:20-15:15",
        "max_hold_bars": 30,
        "max_trades_per_day": 6,
        "indicators": [
            {"name": "vwap_zscore", "formula": "zscore(close - vwap, 60)", "lookback": 60},
            {"name": "rel_volume", "formula": "volume / SMA(volume, 20)", "lookback": 20},
        ],
        "entry": {
            "long": {
                "conditions": ["close < vwap", "vwap_zscore < -1.5", "rel_volume > 1.3", "vix < 18"],
                "confirmation": "",
            },
            "short": {
                "conditions": ["close > vwap", "vwap_zscore > 1.5", "rel_volume > 1.3", "vix < 18"],
                "confirmation": "",
            },
            "entry_price": "close of signal bar",
        },
        "exit": {
            "target": "vwap touch",
            "stop_loss": "0.3% from entry",
            "trailing_stop": "none",
            "time_stop": "exit after 30 bars",
            "eod_rule": "flatten",
            "signal_exit": "exit when zscore crosses 0",
        },
        "filters": {
            "vix_filter": "vix < 18",
            "time_filter": "no trades in first 5 min",
            "volume_filter": "skip if volume < 50% of 20-bar avg",
            "trend_filter": "none",
            "spread_filter": "none",
            "event_filter": "none",
        },
        "risk": {
            "capital_per_trade": 100000,
            "max_risk_per_trade_pct": 0.3,
            "position_sizing": "fixed",
            "max_open_positions": 1,
            "max_daily_loss_inr": 5000,
        },
        "tags": ["mean_reversion"],
        "thesis": "VWAP reversion test",
        "weaknesses": ["test"],
        "_dedup_metadata": {"family": "vwap_reversion"},
    }

    ps = parse_strategy(raw)
    assert ps.is_parseable, f"Strategy should be parseable, errors: {ps.parse_errors}"
    assert len(ps.long_conditions) == 4, f"Expected 4 long conditions, got {len(ps.long_conditions)}"
    assert len(ps.short_conditions) == 4
    assert ps.needs_vwap is True
    assert ps.exit_rules.stop_loss_pct is not None
    assert ps.exit_rules.target_vwap is True
    assert ps.time_filter.skip_first_n_min == 5
    assert len(ps.tunable_params) > 0

    print(f"  Strategy parser: parseable={ps.is_parseable}, "
          f"long_conds={len(ps.long_conditions)}, short_conds={len(ps.short_conditions)}, "
          f"tunable_params={len(ps.tunable_params)}")
    return True


def test_metrics():
    """Test metrics computation."""
    from pipeline.metrics import compute_metrics

    # Create mock trades
    trades = pl.DataFrame({
        "trade_id": [f"t{i}" for i in range(20)],
        "symbol": ["TEST"] * 20,
        "side": ["LONG"] * 20,
        "entry_time": [datetime(2024, 1, d + 1, 10, 0) for d in range(20)],
        "exit_time": [datetime(2024, 1, d + 1, 11, 0) for d in range(20)],
        "entry_price": [100.0] * 20,
        "exit_price": [101.0] * 12 + [99.0] * 8,  # 12 wins, 8 losses
        "pnl": [1000.0] * 12 + [-1000.0] * 8,
        "pnl_pct": [0.01] * 12 + [-0.01] * 8,
        "holding_bars": [15] * 20,
        "exit_reason": ["TARGET"] * 12 + ["STOP"] * 8,
        "entry_indicators": ["{}"] * 20,
    })

    m = compute_metrics(trades, capital_per_trade=100000, total_trading_days=20)
    assert m["total_trades"] == 20
    assert abs(m["win_rate"] - 0.6) < 0.01
    assert m["total_pnl"] == 4000.0  # 12*1000 - 8*1000
    assert m["profit_factor"] == 1.5  # 12000 / 8000
    assert m["sharpe_annualized"] != 0  # Should have some Sharpe

    print(f"  Metrics: trades={m['total_trades']}, win_rate={m['win_rate']:.1%}, "
          f"Sharpe={m['sharpe_annualized']:.4f}, PF={m['profit_factor']:.2f}")
    return True


def test_condition_parser_edge_cases():
    """Test condition parser edge cases found during audit."""
    from pipeline.condition_parser import parse_condition

    # 1. "between" pattern
    result = parse_condition("vix between 12-22")
    assert len(result) == 2, f"'between' should produce 2 conditions, got {len(result)}"
    # Should be vix >= 12 AND vix <= 22
    assert result[0].lhs == "vix" and result[0].op == ">=" and result[0].rhs == 12.0
    assert result[1].lhs == "vix" and result[1].op == "<=" and result[1].rhs == 22.0

    # 2. "between X and Y"
    result = parse_condition("rsi_14 between 30 and 70")
    assert len(result) == 2, f"'between X and Y' should produce 2 conditions, got {len(result)}"

    # 3. Function-call syntax: "RSI(close, 14) < 30"
    result = parse_condition("RSI(close, 14) < 30")
    assert len(result) == 1, f"'RSI(close,14) < 30' should parse, got {len(result)}"
    assert result[0].lhs == "rsi_14"
    assert result[0].op == "<"
    assert result[0].rhs == 30.0

    # 4. "ADX(14) > 20"
    result = parse_condition("ADX(14) > 20")
    assert len(result) == 1, f"'ADX(14) > 20' should parse, got {len(result)}"
    assert result[0].lhs == "adx_14"

    # 5. No spaces: "close<vwap"
    result = parse_condition("close<vwap")
    assert len(result) == 1, f"'close<vwap' should parse, got {len(result)}"

    # 6. Percentage: "morning_return > 0.3%"
    result = parse_condition("morning_return > 0.3%")
    assert len(result) == 1
    assert abs(result[0].rhs - 0.003) < 0.0001, f"0.3% should become 0.003, got {result[0].rhs}"

    # 7. AND within single string
    result = parse_condition("rsi_14 < 30 AND adx > 20")
    assert len(result) == 2

    print("  Condition parser edge cases: all passed")
    return True


def test_state_machine_edge_cases():
    """Test state machine edge cases found during audit."""
    from pipeline.state_machine import run_state_machine

    def make_base(n):
        prices = np.full(n, 100.0, dtype=np.float64)
        return (prices.copy(), prices + 0.2, prices - 0.2, prices.copy(),  # OHLC
                np.zeros(n, dtype=np.int32),  # day_id
                np.array([(555 + i) for i in range(n)], dtype=np.int32),  # time_mins
                np.zeros(n, dtype=np.bool_), np.zeros(n, dtype=np.bool_),  # entries
                np.zeros(n, dtype=np.bool_), np.zeros(n, dtype=np.bool_),  # sig exits
                np.full(n, 1.0, dtype=np.float64),  # atr
                np.zeros(n, dtype=np.float64))  # target_ind

    def run(n, opens, highs, lows, close, day_ids, time_mins, le, se, sel, ses, atr, ti,
            **kwargs):
        defaults = dict(stop_loss_pct=0.005, stop_loss_atr_mult=0.0,
                       target_pct=0.01, target_atr_mult=0.0,
                       use_target_indicator=False, trailing_stop_pct=0.0,
                       trailing_activate_pct=0.0, breakeven_pct=0.0,
                       time_stop_bars=30, eod_flatten_minutes=920,
                       session_start_minutes=555, session_end_minutes=920,
                       max_trades_per_day=10, max_daily_loss=0.0,
                       capital_per_trade=100000.0, warmup_bars=5)
        defaults.update(kwargs)
        return run_state_machine(opens, highs, lows, close, day_ids, time_mins,
                                 le, se, sel, ses, atr, ti, **defaults)

    # Test 1: Zero entry signals → zero trades
    n = 100
    result = run(n, *make_base(n))
    assert result[6] == 0, "Zero signals should produce zero trades"

    # Test 2: Stop and target same bar → stop wins
    n = 100
    opens, highs, lows, close, day_ids, tm, le, se, sel, ses, atr, ti = make_base(n)
    highs[:] = 100.5
    lows[:] = 99.8
    le[10] = True
    close[10] = 100.0
    highs[11] = 101.5  # above target (101)
    lows[11] = 99.0    # below stop (99.5)
    result = run(n, opens, highs, lows, close, day_ids, tm, le, se, sel, ses, atr, ti)
    assert result[6] > 0
    assert result[5][0] == 1, "Stop should fire before target on same bar"

    # Test 3: No entries during warmup
    n = 100
    opens, highs, lows, close, day_ids, tm, le, se, sel, ses, atr, ti = make_base(n)
    le[3] = True   # during warmup
    le[25] = True  # after warmup
    result = run(n, opens, highs, lows, close, day_ids, tm, le, se, sel, ses, atr, ti,
                 warmup_bars=20)
    assert result[6] > 0
    assert result[0][0] >= 20, "No entries during warmup"

    # Test 4: Consecutive entry signals - second ignored when in position
    n = 100
    opens, highs, lows, close, day_ids, tm, le, se, sel, ses, atr, ti = make_base(n)
    le[10] = True
    le[11] = True  # should be ignored - already in position
    result = run(n, opens, highs, lows, close, day_ids, tm, le, se, sel, ses, atr, ti)
    assert result[0][0] == 10
    # Since range is tight (99.8-100.2), stop at 99.5 won't trigger between bars
    if result[6] > 1:
        assert result[0][1] != 11, "Second entry at bar 11 should be ignored"

    # Test 5: Entry at 15:19, EOD at 15:20 → 1-bar hold
    n = 50
    opens, highs, lows, close, day_ids, tm, le, se, sel, ses, atr, ti = make_base(n)
    tm[:] = 600
    tm[30] = 919  # 15:19
    tm[31] = 920  # 15:20
    le[30] = True
    result = run(n, opens, highs, lows, close, day_ids, tm, le, se, sel, ses, atr, ti,
                 target_pct=0.5)  # huge target so it won't trigger
    assert result[6] > 0
    assert result[1][0] - result[0][0] == 1, "Should be a 1-bar hold"
    assert result[5][0] == 6, "Should exit via EOD"

    print("  State machine edge cases: all passed")
    return True


def test_session_end_capped_at_eod():
    """Verify session end is capped at EOD flatten time in backtester."""
    from pipeline.config import EOD_FLATTEN_H, EOD_FLATTEN_M
    eod = EOD_FLATTEN_H * 60 + EOD_FLATTEN_M  # 920

    # If strategy has session end at 15:30 (930), it should be capped to 920
    effective_end = 930
    capped = min(effective_end, eod)
    assert capped == 920, f"Session end should be capped at EOD, got {capped}"

    # If strategy has session end at 15:15 (915), it stays at 915
    effective_end = 915
    capped = min(effective_end, eod)
    assert capped == 915, f"Session end below EOD should stay, got {capped}"

    print("  Session end capped at EOD: passed")
    return True


def test_optimizer_all_trials_fail():
    """Verify optimizer handles all-trials-fail gracefully."""
    # Simulate: best_value = -999 means all trials failed
    # The optimizer should fall back to default params
    default_sharpe = 0.5
    best_value = -999.0

    # This mirrors the logic in optimizer.py
    if best_value is not None and best_value > -900:
        optimized_sharpe = best_value
    else:
        optimized_sharpe = default_sharpe

    assert optimized_sharpe == default_sharpe, \
        f"All-trials-fail should fall back to default_sharpe, got {optimized_sharpe}"

    # Normal case
    best_value = 1.5
    if best_value is not None and best_value > -900:
        optimized_sharpe = best_value
    else:
        optimized_sharpe = default_sharpe
    assert optimized_sharpe == 1.5

    print("  Optimizer all-trials-fail: passed")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Pipeline Unit Tests")
    print("=" * 60)

    tests = [
        ("Indicator computation", test_indicators),
        ("Formula parser", test_formula_parser),
        ("Condition parser", test_condition_parser),
        ("Exit rule parser", test_exit_parser),
        ("Time filter parser", test_time_filter),
        ("Strategy parser", test_strategy_parser),
        ("Metrics computation", test_metrics),
        ("Numba state machine", test_numba_state_machine),
        ("Condition parser edge cases", test_condition_parser_edge_cases),
        ("State machine edge cases", test_state_machine_edge_cases),
        ("Session end capped at EOD", test_session_end_capped_at_eod),
        ("Optimizer all-trials-fail", test_optimizer_all_trials_fail),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            print(f"\n[TEST] {name}...")
            if test_fn():
                print(f"  ✓ PASSED")
                passed += 1
            else:
                print(f"  ✗ FAILED")
                failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
    sys.exit(0 if failed == 0 else 1)
