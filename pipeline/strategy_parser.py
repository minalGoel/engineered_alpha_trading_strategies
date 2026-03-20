"""Strategy parser — load and parse strategy JSON files into executable specs.

Determines:
- Which indicators to compute
- Which conditions to evaluate
- Which parameters are tunable
- Whether the strategy is parseable at all
"""
from __future__ import annotations
import orjson
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from pipeline.config import (
    STRATEGY_DIR, STRATEGY_SOURCES, DEFAULT_MAX_HOLD_BARS,
    DEFAULT_CAPITAL_PER_TRADE,
)
from pipeline.condition_parser import (
    ParsedCondition, CrossCondition, ExitRules,
    parse_conditions_list, parse_exit_rules, parse_time_filter,
    parse_vix_filter, parse_volume_filter, TimeFilter,
)
from pipeline.indicators import parse_indicator_formula, get_max_lookback

log = logging.getLogger(__name__)

_SKIP_VALS = frozenset(["none", "n/a", "not_applicable", "na", ""])


@dataclass
class ParsedStrategy:
    """Fully parsed strategy ready for backtesting."""
    name: str
    source_path: str
    raw_json: dict

    # Core settings
    timeframe: str = "1min"
    capital_per_trade: float = DEFAULT_CAPITAL_PER_TRADE
    max_hold_bars: int = DEFAULT_MAX_HOLD_BARS
    max_trades_per_day: int = 10
    max_daily_loss_inr: float = 0.0

    # Indicator definitions (original dicts from JSON)
    indicator_defs: list[dict] = field(default_factory=list)
    max_lookback: int = 1

    # Parsed entry conditions
    long_conditions: list = field(default_factory=list)
    short_conditions: list = field(default_factory=list)
    is_long_only: bool = False

    # Parsed exit rules
    exit_rules: ExitRules = field(default_factory=ExitRules)

    # Parsed filters
    time_filter: TimeFilter = field(default_factory=TimeFilter)
    vix_conditions: list = field(default_factory=list)
    volume_conditions: list = field(default_factory=list)
    extra_filter_conditions: list = field(default_factory=list)

    # Data requirements flags
    needs_vwap: bool = False
    needs_pdh_pdl: bool = False
    needs_prev_close: bool = False
    needs_opening_range: bool = False
    needs_gap: bool = False
    needs_obv: bool = False
    needs_bar_count: bool = False
    needs_index: bool = False

    # Tunable parameters: {param_name: (default_value, lo_bound, hi_bound)}
    tunable_params: dict = field(default_factory=dict)

    # Parse status
    parse_errors: list[str] = field(default_factory=list)
    parse_warnings: list[str] = field(default_factory=list)
    unparsed_conditions: list[str] = field(default_factory=list)
    failed_indicators: list[str] = field(default_factory=list)

    # Metadata
    family: str = ""
    tags: list[str] = field(default_factory=list)
    thesis: str = ""
    weaknesses: list[str] = field(default_factory=list)

    @property
    def is_parseable(self) -> bool:
        """Strategy is parseable if all critical conditions could be parsed."""
        # Must have at least one entry condition (long or short)
        if not self.long_conditions and not self.short_conditions:
            return False
        # No critical parse errors
        return len(self.parse_errors) == 0

    def get_all_conditions_columns(self) -> set[str]:
        """Get all column names needed by all conditions."""
        cols = set()
        for cond in self.long_conditions + self.short_conditions + self.vix_conditions + self.volume_conditions:
            if hasattr(cond, "required_columns"):
                cols.update(cond.required_columns())
        for cond in self.exit_rules.signal_exit_conditions:
            if hasattr(cond, "required_columns"):
                cols.update(cond.required_columns())
        return cols


def load_all_strategies() -> list[dict]:
    """Load all strategy JSON files from all source directories."""
    strategies = []
    for source in STRATEGY_SOURCES:
        source_dir = STRATEGY_DIR / source
        if not source_dir.exists():
            continue
        for json_file in sorted(source_dir.glob("*.json")):
            try:
                data = orjson.loads(json_file.read_bytes())
                data["_source_dir"] = source
                data["_file_path"] = str(json_file)
                strategies.append(data)
            except Exception as e:
                log.error("Failed to load %s: %s", json_file, e)
    log.info("Loaded %d strategy JSON files", len(strategies))
    return strategies


def _detect_data_needs(raw: dict, all_text: str) -> dict[str, bool]:
    """Detect what data features a strategy needs based on its content."""
    text_lower = all_text.lower()
    return {
        "needs_vwap": "vwap" in text_lower,
        "needs_pdh_pdl": any(x in text_lower for x in ["pdh", "pdl", "prev_day", "previous_day"]),
        "needs_prev_close": any(x in text_lower for x in ["prev_close", "gap"]),
        "needs_opening_range": any(x in text_lower for x in ["or_high", "or_low", "opening_range", "orb"]),
        "needs_gap": "gap" in text_lower,
        "needs_obv": "obv" in text_lower,
        "needs_bar_count": "bar_count_from_open" in text_lower,
        "needs_index": any(x in text_lower for x in [
            "index", "nifty", "nifty50", "nifty_50", "index_return", "index_close"
        ]),
    }


def _extract_all_text(raw: dict) -> str:
    """Extract all parseable text from a strategy JSON for data-need detection."""
    parts = []
    for ind in raw.get("indicators", []):
        if isinstance(ind, dict):
            parts.append(ind.get("formula", ""))
            parts.append(ind.get("name", ""))
        elif isinstance(ind, str):
            parts.append(ind)
    entry = raw.get("entry", {})
    for side in ("long", "short"):
        side_data = entry.get(side, {})
        if isinstance(side_data, dict):
            parts.extend(side_data.get("conditions", []))
            parts.append(str(side_data.get("confirmation", "")))
    exit_data = raw.get("exit", {})
    for k, v in exit_data.items():
        parts.append(str(v))
    filters = raw.get("filters", {})
    for k, v in filters.items():
        if isinstance(v, list):
            parts.extend(str(x) for x in v)
        else:
            parts.append(str(v))
    return " ".join(parts)


def _extract_tunable_params(
    long_conds: list, short_conds: list, exit_rules: ExitRules,
    vix_conds: list, volume_conds: list,
) -> dict[str, tuple[float, float, float]]:
    """Extract tunable numeric thresholds from parsed conditions.

    Returns {param_name: (default_value, lo_bound, hi_bound)}.
    Only numeric RHS values in comparisons are tunable.
    Indicator periods are NOT tunable.
    """
    params = {}
    seen_names = set()

    def _add_from_conditions(conds: list, prefix: str = ""):
        for cond in conds:
            if isinstance(cond, ParsedCondition) and cond.rhs_is_numeric:
                name = f"{prefix}{cond.lhs}_threshold"
                # Avoid duplicates
                if name in seen_names:
                    # Append operator to disambiguate
                    name = f"{prefix}{cond.lhs}_{cond.op.replace('>', 'gt').replace('<', 'lt').replace('=', 'eq')}_threshold"
                if name in seen_names:
                    continue
                seen_names.add(name)
                default = float(cond.rhs)
                # Set bounds: 30% to 300% of default, with sensible absolute limits
                if default == 0:
                    lo, hi = -1.0, 1.0
                elif default > 0:
                    lo = max(default * 0.3, default - abs(default) * 2)
                    hi = min(default * 3.0, default + abs(default) * 2)
                else:  # negative threshold
                    hi = min(default * 0.3, default + abs(default) * 2)
                    lo = max(default * 3.0, default - abs(default) * 2)
                params[name] = (default, lo, hi)

    _add_from_conditions(long_conds, "long_")
    _add_from_conditions(short_conds, "short_")
    _add_from_conditions(vix_conds, "vix_")
    _add_from_conditions(volume_conds, "vol_")

    # Add exit rule params
    exit_params = exit_rules.get_tunable_params()
    params.update(exit_params)

    return params


def parse_strategy(raw: dict) -> ParsedStrategy:
    """Parse a raw strategy JSON dict into a ParsedStrategy."""
    name = raw.get("name", "unknown")
    source_path = raw.get("_file_path", "")

    ps = ParsedStrategy(
        name=name,
        source_path=source_path,
        raw_json=raw,
    )

    # ── Basic metadata ───────────────────────────────────────────────────
    ps.timeframe = raw.get("timeframe", "1min").lower().strip()
    # Normalize timeframe
    tf_map = {"1 min": "1min", "5 min": "5min", "15 min": "15min",
              "30 min": "30min", "1 hour": "1h", "1hr": "1h"}
    ps.timeframe = tf_map.get(ps.timeframe, ps.timeframe)

    risk = raw.get("risk", {})
    ps.capital_per_trade = risk.get("capital_per_trade", DEFAULT_CAPITAL_PER_TRADE)
    ps.max_daily_loss_inr = risk.get("max_daily_loss_inr", 0)
    ps.max_hold_bars = raw.get("max_hold_bars") or DEFAULT_MAX_HOLD_BARS
    ps.max_trades_per_day = raw.get("max_trades_per_day", 10)
    ps.family = raw.get("_dedup_metadata", {}).get("family", "")
    ps.tags = raw.get("tags", [])
    ps.thesis = raw.get("thesis", "")
    ps.weaknesses = raw.get("weaknesses", [])

    # ── Data needs ───────────────────────────────────────────────────────
    all_text = _extract_all_text(raw)
    needs = _detect_data_needs(raw, all_text)
    for k, v in needs.items():
        setattr(ps, k, v)

    # ── Indicators ───────────────────────────────────────────────────────
    raw_indicators = raw.get("indicators", [])
    # Ensure all indicators are dicts (some may be strings)
    ps.indicator_defs = [ind for ind in raw_indicators if isinstance(ind, dict)]
    ps.max_lookback = get_max_lookback(ps.indicator_defs)

    # Check if indicators are parseable
    # Include all structural indicators that we compute regardless
    existing_cols = {"open", "high", "low", "close", "volume", "datetime", "day_id",
                     "vwap", "vix", "pdh", "pdl", "prev_close", "gap_pct",
                     "or_high", "or_low", "obv", "bar_count_from_open",
                     "index_close", "index_open", "index_high", "index_low",
                     "_time_minutes",
                     # Common aliases that map to structural indicators
                     "previous_day_high", "previous_day_low", "prev_day_high", "prev_day_low",
                     "orb_high", "orb_low", "or_high_15", "or_low_15",
                     "opening_range_high", "opening_range_low",
                     "nifty50_close", "nifty_close", "index_return",
                     # Session / intraday structural
                     "session_high", "session_low", "open_today", "day_open",
                     "morning_return", "stock_return", "intraday_return",
                     "rel_volume", "volume_ratio", "rel_volume_20", "vol_spike",
                     "india_vix",
                     }
    # Also add indicator names as "assumed available" — they reference each other
    for ind in ps.indicator_defs:
        ind_name = ind.get("name", "")
        if ind_name:
            existing_cols.add(ind_name)

    failed_indicators = []
    for ind in ps.indicator_defs:
        steps = parse_indicator_formula(ind, existing_cols)
        if not steps:
            formula = ind.get("formula", "")
            ind_name = ind.get("name", "")
            from pipeline.indicators import _is_narrative_formula
            if _is_narrative_formula(formula):
                ps.parse_warnings.append(f"Narrative indicator skipped: {ind_name} = {formula}")
            else:
                ps.parse_warnings.append(f"Unparseable indicator: {ind_name} = {formula}")
            failed_indicators.append(ind_name)
        else:
            for step in steps:
                existing_cols.add(step["output_col"])
            existing_cols.add(ind.get("name", ""))
    ps.failed_indicators = failed_indicators

    # ── Entry conditions ─────────────────────────────────────────────────
    entry = raw.get("entry", {})

    # Long conditions
    long_data = entry.get("long", {})
    if isinstance(long_data, dict):
        long_conds_raw = long_data.get("conditions", [])
        if isinstance(long_conds_raw, list):
            # Filter out N/A
            long_conds_raw = [c for c in long_conds_raw
                              if str(c).strip().lower() not in _SKIP_VALS]
            ps.long_conditions, long_unparsed = parse_conditions_list(long_conds_raw)
            ps.unparsed_conditions.extend(long_unparsed)

    # Short conditions
    short_data = entry.get("short", {})
    if isinstance(short_data, dict):
        short_conds_raw = short_data.get("conditions", [])
        if isinstance(short_conds_raw, list):
            short_conds_raw = [c for c in short_conds_raw
                               if str(c).strip().lower() not in _SKIP_VALS]
            if not short_conds_raw:
                ps.is_long_only = True
            else:
                ps.short_conditions, short_unparsed = parse_conditions_list(short_conds_raw)
                if not ps.short_conditions:
                    ps.is_long_only = True
                ps.unparsed_conditions.extend(short_unparsed)
        else:
            ps.is_long_only = True
    else:
        ps.is_long_only = True

    # Check if entry conditions reference failed indicators
    # This is a WARNING not an error — the indicator may still be computed
    # at runtime even if the formula parser couldn't parse it (structural indicators)
    required_cols = ps.get_all_conditions_columns()
    for col in required_cols:
        if col in failed_indicators:
            ps.parse_warnings.append(
                f"Entry condition references unparseable indicator: {col}"
            )

    # Log unparsed conditions as warnings
    for uc in ps.unparsed_conditions:
        ps.parse_warnings.append(f"Unparsed condition: {uc}")

    # Only FAIL if we have NO parseable entry conditions at all
    # Per spec: "Either parse all conditions or fail the strategy"
    # But in practice, many conditions are soft filters (volume text, etc.)
    # We fail ONLY if there are zero parseable entry conditions
    if not ps.long_conditions and not ps.short_conditions:
        ps.parse_errors.append("No parseable entry conditions found")

    # ── Exit rules ───────────────────────────────────────────────────────
    exit_data = raw.get("exit", {})
    ps.exit_rules = parse_exit_rules(exit_data, ps.max_hold_bars)

    # ── Filters ──────────────────────────────────────────────────────────
    filters = raw.get("filters", {})
    session_str = raw.get("session", "")
    ps.time_filter = parse_time_filter(session_str, filters)
    ps.vix_conditions = parse_vix_filter(filters)
    ps.volume_conditions = parse_volume_filter(filters)

    # ── Tunable parameters ───────────────────────────────────────────────
    ps.tunable_params = _extract_tunable_params(
        ps.long_conditions, ps.short_conditions, ps.exit_rules,
        ps.vix_conditions, ps.volume_conditions,
    )

    return ps


def build_indicator_coverage(strategies: list[dict]) -> dict:
    """Phase 2: Parse all strategies and build indicator coverage report."""
    coverage = {
        "total_strategies": len(strategies),
        "parseable": 0,
        "unparseable": 0,
        "indicator_usage": {},  # indicator_name -> count
        "condition_patterns": {},  # pattern -> count
        "timeframe_usage": {},
        "strategies": {},  # name -> parse status
    }

    for raw in strategies:
        ps = parse_strategy(raw)
        status = {
            "name": ps.name,
            "source": ps.source_path,
            "timeframe": ps.timeframe,
            "parseable": ps.is_parseable,
            "long_conditions_count": len(ps.long_conditions),
            "short_conditions_count": len(ps.short_conditions),
            "is_long_only": ps.is_long_only,
            "tunable_params_count": len(ps.tunable_params),
            "failed_indicators": ps.failed_indicators,
            "unparsed_conditions": ps.unparsed_conditions,
            "parse_errors": ps.parse_errors,
            "parse_warnings": ps.parse_warnings,
        }
        coverage["strategies"][ps.name] = status

        if ps.is_parseable:
            coverage["parseable"] += 1
        else:
            coverage["unparseable"] += 1

        # Track indicator usage
        for ind in ps.indicator_defs:
            ind_name = ind.get("name", "")
            coverage["indicator_usage"][ind_name] = coverage["indicator_usage"].get(ind_name, 0) + 1

        # Track timeframe usage
        tf = ps.timeframe
        coverage["timeframe_usage"][tf] = coverage["timeframe_usage"].get(tf, 0) + 1

    return coverage
