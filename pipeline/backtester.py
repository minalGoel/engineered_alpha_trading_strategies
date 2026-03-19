"""Backtester — run a single strategy on a single stock.

Computes indicators, evaluates conditions, runs the Numba state machine,
and returns trades + signals DataFrames.
"""
from __future__ import annotations
import numpy as np
import polars as pl
import logging
from typing import Optional

from pipeline.config import (
    DEFAULT_MAX_HOLD_BARS, EOD_FLATTEN_H, EOD_FLATTEN_M,
    STATE_FLAT, STATE_LONG, STATE_SHORT,
    ACTION_BUY, ACTION_SELL, ACTION_SHORT, ACTION_COVER,
)
from pipeline.strategy_parser import ParsedStrategy
from pipeline.indicators import compute_all_indicators
from pipeline.condition_parser import (
    evaluate_conditions, ParsedCondition, CrossCondition,
)
from pipeline.state_machine import run_state_machine, EXIT_REASON_MAP

log = logging.getLogger(__name__)


def _build_time_minutes(df: pl.DataFrame) -> np.ndarray:
    """Extract time-of-day in minutes from datetime column."""
    hours = df["datetime"].dt.hour().to_numpy().astype(np.int32)
    minutes = df["datetime"].dt.minute().to_numpy().astype(np.int32)
    return hours * 60 + minutes


def _df_to_arrays(df: pl.DataFrame) -> dict[str, np.ndarray]:
    """Convert DataFrame columns to numpy arrays for condition evaluation."""
    arrays = {}
    for col in df.columns:
        if df[col].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8,
                              pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8):
            arrays[col] = df[col].to_numpy().astype(np.float64)
        elif df[col].dtype == pl.Boolean:
            arrays[col] = df[col].to_numpy().astype(np.float64)
    # Add _time_minutes
    if "datetime" in df.columns:
        arrays["_time_minutes"] = _build_time_minutes(df).astype(np.float64)
    return arrays


def _apply_threshold_overrides(
    conditions: list,
    param_overrides: dict[str, float],
    prefix: str,
) -> list:
    """Apply Optuna threshold overrides to conditions."""
    new_conditions = []
    for cond in conditions:
        if isinstance(cond, ParsedCondition) and cond.rhs_is_numeric:
            param_name = f"{prefix}{cond.lhs}_threshold"
            # Also try with operator disambiguation
            param_name_op = f"{prefix}{cond.lhs}_{cond.op.replace('>', 'gt').replace('<', 'lt').replace('=', 'eq')}_threshold"
            if param_name in param_overrides:
                cond = cond.with_threshold(param_overrides[param_name])
            elif param_name_op in param_overrides:
                cond = cond.with_threshold(param_overrides[param_name_op])
        new_conditions.append(cond)
    return new_conditions


def backtest_single(
    df: pl.DataFrame,
    strategy: ParsedStrategy,
    symbol: str,
    param_overrides: Optional[dict[str, float]] = None,
    skip_indicators: bool = False,
) -> Optional[pl.DataFrame]:
    """Run backtest for one strategy on one stock's data.

    Args:
        df: OHLCV DataFrame with day_id, VIX, index data already merged.
        strategy: Parsed strategy specification.
        symbol: Stock symbol name.
        param_overrides: Optional dict of parameter overrides from Optuna.
        skip_indicators: If True, assume indicators are already computed on df.

    Returns:
        trades DataFrame or None if no trades / error.
    """
    if df is None or df.is_empty():
        return None

    param_overrides = param_overrides or {}

    try:
        # ── Compute indicators ──────────────────────────────────────────
        if not skip_indicators:
            df, computed, failed = compute_all_indicators(
                df,
                strategy.indicator_defs,
                needs_vwap=strategy.needs_vwap,
                needs_pdh_pdl=strategy.needs_pdh_pdl,
                needs_prev_close=strategy.needs_prev_close,
                needs_opening_range=strategy.needs_opening_range,
                needs_gap=strategy.needs_gap,
                needs_obv=strategy.needs_obv,
                needs_bar_count=strategy.needs_bar_count,
            )

        # Convert to numpy arrays for condition evaluation
        arrays = _df_to_arrays(df)

        # ── Apply parameter overrides to conditions ─────────────────────
        long_conds = _apply_threshold_overrides(strategy.long_conditions, param_overrides, "long_")
        short_conds = _apply_threshold_overrides(strategy.short_conditions, param_overrides, "short_")
        vix_conds = _apply_threshold_overrides(strategy.vix_conditions, param_overrides, "vix_")
        vol_conds = _apply_threshold_overrides(strategy.volume_conditions, param_overrides, "vol_")

        # ── Evaluate entry conditions ───────────────────────────────────
        n = len(df)
        long_mask = np.ones(n, dtype=np.bool_)
        short_mask = np.ones(n, dtype=np.bool_) if not strategy.is_long_only else np.zeros(n, dtype=np.bool_)

        if long_conds:
            long_mask = evaluate_conditions(long_conds, arrays)
        else:
            long_mask = np.zeros(n, dtype=np.bool_)

        if not strategy.is_long_only and short_conds:
            short_mask = evaluate_conditions(short_conds, arrays)

        # Apply VIX filter
        if vix_conds:
            vix_mask = evaluate_conditions(vix_conds, arrays)
            long_mask &= vix_mask
            short_mask &= vix_mask

        # Apply volume filter
        if vol_conds:
            vol_mask = evaluate_conditions(vol_conds, arrays)
            long_mask &= vol_mask
            short_mask &= vol_mask

        # Apply time filter — only generate signals within session window
        time_mins = _build_time_minutes(df)
        effective_start = strategy.time_filter.effective_start_minutes()
        effective_end = strategy.time_filter.effective_end_minutes()
        time_mask = (time_mins >= effective_start) & (time_mins < effective_end)
        long_mask &= time_mask
        short_mask &= time_mask

        # Apply expiry day filter
        if strategy.time_filter.skip_expiry:
            # Skip Thursdays (approximate expiry detection)
            if "datetime" in df.columns:
                weekday = df["datetime"].dt.weekday().to_numpy()
                expiry_mask = weekday != 3  # 3 = Thursday
                long_mask &= expiry_mask
                short_mask &= expiry_mask

        # ── Evaluate signal exit conditions ─────────────────────────────
        signal_exit_long = np.zeros(n, dtype=np.bool_)
        signal_exit_short = np.zeros(n, dtype=np.bool_)
        if strategy.exit_rules.signal_exit_conditions:
            sig_mask = evaluate_conditions(strategy.exit_rules.signal_exit_conditions, arrays)
            signal_exit_long = sig_mask
            signal_exit_short = sig_mask

        # ── Get ATR array for ATR-based stops ───────────────────────────
        atr_col = None
        for ind in strategy.indicator_defs:
            if "atr" in ind.get("name", "").lower():
                atr_col = ind["name"]
                break
        if atr_col and atr_col in arrays:
            atr_arr = arrays[atr_col]
        else:
            # Compute a default ATR(20) if needed
            atr_period = strategy.exit_rules.stop_loss_atr_period
            if atr_period > 0 and (strategy.exit_rules.stop_loss_atr_mult or strategy.exit_rules.target_atr_mult):
                from pipeline.indicators import compute_atr
                df_temp = compute_atr(df, atr_period, "_atr_temp")
                atr_arr = df_temp["_atr_temp"].to_numpy().astype(np.float64)
            else:
                atr_arr = np.zeros(n, dtype=np.float64)

        # ── Get target indicator array ──────────────────────────────────
        target_indicator = np.zeros(n, dtype=np.float64)
        use_target_indicator = False
        if strategy.exit_rules.target_vwap and "vwap" in arrays:
            target_indicator = arrays["vwap"]
            use_target_indicator = True
        elif strategy.exit_rules.target_indicator:
            ti = strategy.exit_rules.target_indicator
            if ti in arrays:
                target_indicator = arrays[ti]
                use_target_indicator = True

        # ── Resolve exit parameters with overrides ──────────────────────
        sl_pct = param_overrides.get("stop_loss_pct",
                    (strategy.exit_rules.stop_loss_pct or 0) * 100) / 100.0
        sl_atr = param_overrides.get("stop_loss_atr_mult",
                    strategy.exit_rules.stop_loss_atr_mult or 0)
        tgt_pct = param_overrides.get("target_pct",
                    (strategy.exit_rules.target_pct or 0) * 100) / 100.0
        tgt_atr = param_overrides.get("target_atr_mult",
                    strategy.exit_rules.target_atr_mult or 0)
        trail_pct = param_overrides.get("trailing_stop_pct",
                    (strategy.exit_rules.trailing_stop_pct or 0) * 100) / 100.0
        trail_activate = strategy.exit_rules.trailing_activate_pct or 0
        be_pct = strategy.exit_rules.breakeven_after_pct or 0
        time_stop = int(param_overrides.get("time_stop_bars",
                    strategy.exit_rules.time_stop_bars or DEFAULT_MAX_HOLD_BARS))

        # NaN-safe: replace NaN with 0 in atr array
        atr_arr = np.nan_to_num(atr_arr, nan=0.0)
        target_indicator = np.nan_to_num(target_indicator, nan=0.0)

        # ── Run state machine ───────────────────────────────────────────
        result = run_state_machine(
            open_arr=arrays.get("open", np.zeros(n)),
            high_arr=arrays.get("high", np.zeros(n)),
            low_arr=arrays.get("low", np.zeros(n)),
            close_arr=arrays.get("close", np.zeros(n)),
            day_id=df["day_id"].to_numpy().astype(np.int32),
            time_minutes=time_mins.astype(np.int32),
            long_entry=long_mask,
            short_entry=short_mask,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr_arr,
            target_indicator=target_indicator,
            stop_loss_pct=sl_pct,
            stop_loss_atr_mult=sl_atr,
            target_pct=tgt_pct,
            target_atr_mult=tgt_atr,
            use_target_indicator=use_target_indicator,
            trailing_stop_pct=trail_pct,
            trailing_activate_pct=trail_activate,
            breakeven_pct=be_pct,
            time_stop_bars=time_stop,
            eod_flatten_minutes=EOD_FLATTEN_H * 60 + EOD_FLATTEN_M,
            session_start_minutes=effective_start,
            session_end_minutes=effective_end,
            max_trades_per_day=strategy.max_trades_per_day,
            max_daily_loss=strategy.max_daily_loss_inr,
            capital_per_trade=strategy.capital_per_trade,
            warmup_bars=strategy.max_lookback,
        )

        entry_bars, exit_bars, sides, entry_prices, exit_prices, exit_reasons, trade_count = result

        if trade_count == 0:
            return None

        # ── Build trades DataFrame ──────────────────────────────────────
        datetimes = df["datetime"].to_list()
        capital = strategy.capital_per_trade

        trade_ids = []
        symbols = []
        side_labels = []
        entry_times = []
        exit_times = []
        entry_px = []
        exit_px = []
        pnls = []
        pnl_pcts = []
        holding_bars_list = []
        exit_reason_labels = []
        entry_indicators_list = []

        for t in range(trade_count):
            entry_bar = int(entry_bars[t])
            exit_bar = int(exit_bars[t])
            side = int(sides[t])
            ep = float(entry_prices[t])
            xp = float(exit_prices[t])

            if side == STATE_LONG:
                pnl = (xp - ep) * (capital / ep)
                pnl_pct = (xp - ep) / ep
                side_label = "LONG"
            else:
                pnl = (ep - xp) * (capital / ep)
                pnl_pct = (ep - xp) / ep
                side_label = "SHORT"

            trade_ids.append(f"{symbol}_{strategy.name}_{t}")
            symbols.append(symbol)
            side_labels.append(side_label)
            entry_times.append(datetimes[entry_bar])
            exit_times.append(datetimes[exit_bar])
            entry_px.append(ep)
            exit_px.append(xp)
            pnls.append(pnl)
            pnl_pcts.append(pnl_pct)
            holding_bars_list.append(exit_bar - entry_bar)
            exit_reason_labels.append(EXIT_REASON_MAP.get(int(exit_reasons[t]), "UNKNOWN"))

            # Capture entry indicators (for audit)
            ind_dict = {}
            for ind in strategy.indicator_defs:
                ind_name = ind.get("name", "")
                if ind_name in arrays:
                    val = float(arrays[ind_name][entry_bar])
                    if val == val:  # not NaN
                        ind_dict[ind_name] = round(val, 4)
            if "vix" in arrays:
                val = float(arrays["vix"][entry_bar])
                if val == val:
                    ind_dict["vix"] = round(val, 2)
            import orjson
            entry_indicators_list.append(orjson.dumps(ind_dict).decode())

        trades_df = pl.DataFrame({
            "trade_id": trade_ids,
            "symbol": symbols,
            "side": side_labels,
            "entry_time": entry_times,
            "exit_time": exit_times,
            "entry_price": entry_px,
            "exit_price": exit_px,
            "pnl": pnls,
            "pnl_pct": pnl_pcts,
            "holding_bars": holding_bars_list,
            "exit_reason": exit_reason_labels,
            "entry_indicators": entry_indicators_list,
        })

        return trades_df

    except Exception as e:
        log.error("Backtest failed for %s/%s: %s", strategy.name, symbol, e, exc_info=True)
        return None


def build_signals_from_trades(
    trades_df: pl.DataFrame,
    strategy_name: str,
) -> pl.DataFrame:
    """Convert trades DataFrame into signals DataFrame (BUY/SELL/SHORT/COVER).

    Each trade generates two signals: an entry and an exit.
    """
    if trades_df is None or trades_df.is_empty():
        return pl.DataFrame(schema={
            "timestamp": pl.Datetime, "symbol": pl.Utf8, "action": pl.Utf8,
            "price": pl.Float64, "stop_loss": pl.Float64, "target": pl.Float64,
            "strategy": pl.Utf8, "confidence": pl.Float64,
            "reason": pl.Utf8, "entry_indicators": pl.Utf8,
        })

    rows = []
    for row in trades_df.iter_rows(named=True):
        side = row["side"]

        # Entry signal
        entry_action = ACTION_BUY if side == "LONG" else ACTION_SHORT
        rows.append({
            "timestamp": row["entry_time"],
            "symbol": row["symbol"],
            "action": entry_action,
            "price": row["entry_price"],
            "stop_loss": None,  # Will be filled from exit rules
            "target": None,
            "strategy": strategy_name,
            "confidence": 1.0,
            "reason": "",
            "entry_indicators": row["entry_indicators"],
        })

        # Exit signal
        exit_action = ACTION_SELL if side == "LONG" else ACTION_COVER
        rows.append({
            "timestamp": row["exit_time"],
            "symbol": row["symbol"],
            "action": exit_action,
            "price": row["exit_price"],
            "stop_loss": None,
            "target": None,
            "strategy": strategy_name,
            "confidence": 1.0,
            "reason": row["exit_reason"],
            "entry_indicators": "",
        })

    return pl.DataFrame(rows)
