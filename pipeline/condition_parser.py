"""Condition parser — convert strategy JSON conditions to executable logic.

Entry/exit/filter conditions are written in pseudo-code / English.
This module parses them into Polars expressions or numpy boolean arrays.
"""
from __future__ import annotations
import re
import logging
from typing import Optional
import numpy as np

log = logging.getLogger(__name__)

# ── Comparison operators ────────────────────────────────────────────────────
_OPS = {
    ">=": ">=", "<=": "<=", "!=": "!=",
    ">": ">", "<": "<", "==": "==", "=": "==",
}

# Strip inline comments:  "close > vwap (some explanation here)"
_RE_COMMENT = re.compile(r"\s*\(.*?\)\s*$")
_RE_INLINE_COMMENT = re.compile(r"\s+—\s+.+$")  # em-dash comments

# Pattern: "LHS OP RHS"
_RE_COMPARISON = re.compile(
    r"^\s*([A-Za-z_][\w.]*(?:\[[\w\-]+\])?)\s*"  # LHS
    r"(>=|<=|!=|>|<|==|=)\s*"                       # operator
    r"([\-+]?[\d.]+|[A-Za-z_][\w.]*(?:\[[\w\-]+\])?)\s*$"  # RHS
)

# Pattern for "A AND B" within a single condition string
_RE_AND = re.compile(r"\s+AND\s+", re.IGNORECASE)
_RE_OR = re.compile(r"\s+OR\s+", re.IGNORECASE)

# Pattern: "ABS(expr) OP value"
_RE_ABS = re.compile(
    r"^\s*ABS\(\s*([A-Za-z_][\w.]*)\s*\)\s*(>=|<=|!=|>|<|==)\s*([\-+]?[\d.]+)\s*$",
    re.IGNORECASE
)

# Pattern: column[t-1] or column[-1]
_RE_LAGGED = re.compile(r"^(\w+)\[(?:t-)?(\d+)\]$")

# Pattern: "CROSSOVER(a, b)" / "CROSSUNDER(a, b)"
_RE_CROSSOVER = re.compile(r"CROSS(?:OVER|ABOVE)\(\s*(\w+)\s*,\s*(\w+)\s*\)", re.IGNORECASE)
_RE_CROSSUNDER = re.compile(r"CROSS(?:UNDER|BELOW)\(\s*(\w+)\s*,\s*(\w+)\s*\)", re.IGNORECASE)

# None / skip patterns
_SKIP_PATTERNS = frozenset([
    "none", "n/a", "not_applicable", "na", "",
    "no filter", "no filter needed",
])

# ── Time filter patterns ────────────────────────────────────────────────────
_RE_TIME_NO_FIRST = re.compile(r"no (?:trades?|entries?) (?:in )?first (\d+) min", re.IGNORECASE)
_RE_TIME_NO_LAST = re.compile(r"no (?:trades?|entries?) (?:in )?last (\d+) min", re.IGNORECASE)
_RE_TIME_WINDOW = re.compile(r"(?:only|entry window)\s*(\d{2}):(\d{2})\s*[-–]\s*(\d{2}):(\d{2})", re.IGNORECASE)
_RE_TIME_NO_BEFORE = re.compile(r"no (?:trades?|entries?) before (\d{2}):(\d{2})", re.IGNORECASE)
_RE_TIME_NO_AFTER = re.compile(r"no (?:trades?|entries?) after (\d{2}):(\d{2})", re.IGNORECASE)
_RE_TIME_SKIP_FIRST_N = re.compile(r"skip first (\d+) bars?", re.IGNORECASE)
_RE_TIME_ONLY_FIRST_N = re.compile(r"only first (\d+) min", re.IGNORECASE)
_RE_TIME_BEFORE = re.compile(r"no entries? before (\d{2}):(\d{2})", re.IGNORECASE)
_RE_TIME_AFTER = re.compile(r"no entries? after (\d{2}):(\d{2})", re.IGNORECASE)
_RE_TIME_PREFER_BEFORE = re.compile(r"prefer (?:breaks?|entries?) before (\d{2}):(\d{2})", re.IGNORECASE)

# ── Exit rule patterns ──────────────────────────────────────────────────────
_RE_EXIT_PCT = re.compile(r"([\d.]+)%\s*(?:from entry)?", re.IGNORECASE)
_RE_EXIT_ATR = re.compile(r"([\d.]+)\s*\*?\s*(?:ATR|atr)[_(\s]*(\d+)", re.IGNORECASE)
_RE_EXIT_BARS = re.compile(r"(?:exit )?after (\d+) bars?", re.IGNORECASE)
_RE_EXIT_TIME = re.compile(r"(\d+)\s*bars", re.IGNORECASE)
_RE_EXIT_VWAP_TOUCH = re.compile(r"vwap\s*(?:touch|cross)", re.IGNORECASE)
_RE_EXIT_ZSCORE_CROSS = re.compile(r"(?:exit\s+)?(?:when\s+)?(\w+)\s+cross(?:es)?\s+([\d.]+|zero|0)", re.IGNORECASE)

# Trailing stop patterns
_RE_TRAIL_PCT = re.compile(r"trail(?:ing)?\s+(?:at|by)\s+([\d.]+)%", re.IGNORECASE)
_RE_TRAIL_ATR = re.compile(r"trail(?:ing)?\s+(?:at|by)\s+([\d.]+)\s*\*?\s*ATR", re.IGNORECASE)
_RE_TRAIL_BE = re.compile(r"(?:move\s+)?(?:SL|stop)\s+to\s+breakeven\s+after\s+\+?([\d.]+)%", re.IGNORECASE)
_RE_TRAIL_ACTIVATE = re.compile(r"once (?:profit|past)\s+(?:exceeds?\s+)?([\d.]+)%", re.IGNORECASE)

# ── VIX filter patterns ────────────────────────────────────────────────────
_RE_VIX = re.compile(r"(?:vix|india_vix|India_VIX)\s*(>=|<=|>|<)\s*([\d.]+)", re.IGNORECASE)
_RE_VIX_RANGE = re.compile(
    r"(?:vix|india_vix)\s*>\s*([\d.]+)\s*AND\s*(?:vix|india_vix)\s*<\s*([\d.]+)",
    re.IGNORECASE
)


class ParsedCondition:
    """A parsed entry/exit condition that can be evaluated on numpy arrays."""

    def __init__(self, lhs: str, op: str, rhs: str | float, is_abs: bool = False):
        self.lhs = lhs
        self.op = op
        self.rhs = rhs  # Either a float or a column name
        self.is_abs = is_abs
        self.lhs_lag = 0
        self.rhs_lag = 0

        # Check for lagged references
        m = _RE_LAGGED.match(lhs)
        if m:
            self.lhs = m.group(1)
            self.lhs_lag = int(m.group(2))
        if isinstance(rhs, str):
            m = _RE_LAGGED.match(rhs)
            if m:
                self.rhs = m.group(1)
                self.rhs_lag = int(m.group(2))

    @property
    def rhs_is_numeric(self) -> bool:
        return isinstance(self.rhs, (int, float))

    @property
    def threshold_value(self) -> Optional[float]:
        """Return the numeric threshold if RHS is numeric, else None."""
        if self.rhs_is_numeric:
            return float(self.rhs)
        return None

    @property
    def threshold_param_name(self) -> Optional[str]:
        """Generate a parameter name for optimization if RHS is numeric."""
        if self.rhs_is_numeric:
            return f"{self.lhs}_threshold"
        return None

    def required_columns(self) -> set[str]:
        """Return set of column names needed to evaluate this condition."""
        cols = {self.lhs}
        if not self.rhs_is_numeric:
            cols.add(self.rhs)
        return cols

    def evaluate(self, arrays: dict[str, np.ndarray]) -> np.ndarray:
        """Evaluate condition on numpy arrays. Returns boolean array."""
        lhs = arrays.get(self.lhs)
        if lhs is None:
            log.warning("Missing column for condition: %s", self.lhs)
            return np.zeros(len(next(iter(arrays.values()))), dtype=np.bool_)

        if self.lhs_lag > 0:
            lhs = np.roll(lhs, self.lhs_lag)
            lhs[:self.lhs_lag] = np.nan

        if self.is_abs:
            lhs = np.abs(lhs)

        if self.rhs_is_numeric:
            rhs = float(self.rhs)
        else:
            rhs = arrays.get(self.rhs)
            if rhs is None:
                log.warning("Missing column for condition RHS: %s", self.rhs)
                return np.zeros(len(lhs), dtype=np.bool_)
            if self.rhs_lag > 0:
                rhs = np.roll(rhs, self.rhs_lag)
                rhs[:self.rhs_lag] = np.nan

        if self.op == ">":
            return lhs > rhs
        elif self.op == "<":
            return lhs < rhs
        elif self.op == ">=":
            return lhs >= rhs
        elif self.op == "<=":
            return lhs <= rhs
        elif self.op == "==":
            return lhs == rhs
        elif self.op == "!=":
            return lhs != rhs
        else:
            return np.ones(len(lhs), dtype=np.bool_)

    def with_threshold(self, new_rhs: float) -> "ParsedCondition":
        """Return a copy with a new RHS threshold (for optimization)."""
        pc = ParsedCondition(self.lhs, self.op, new_rhs, self.is_abs)
        pc.lhs_lag = self.lhs_lag
        pc.rhs_lag = self.rhs_lag
        return pc

    def __repr__(self):
        abs_str = "ABS(" if self.is_abs else ""
        abs_end = ")" if self.is_abs else ""
        lag_l = f"[t-{self.lhs_lag}]" if self.lhs_lag > 0 else ""
        lag_r = f"[t-{self.rhs_lag}]" if self.rhs_lag > 0 else ""
        return f"{abs_str}{self.lhs}{lag_l}{abs_end} {self.op} {self.rhs}{lag_r}"


class CrossCondition:
    """Crossover/crossunder condition."""

    def __init__(self, col_a: str, col_b: str, direction: str):
        self.col_a = col_a
        self.col_b = col_b
        self.direction = direction  # "over" or "under"

    def required_columns(self) -> set[str]:
        cols = {self.col_a}
        try:
            float(self.col_b)
        except ValueError:
            cols.add(self.col_b)
        return cols

    def evaluate(self, arrays: dict[str, np.ndarray]) -> np.ndarray:
        a = arrays.get(self.col_a)
        if a is None:
            return np.zeros(len(next(iter(arrays.values()))), dtype=np.bool_)
        try:
            b = float(self.col_b)
            b = np.full_like(a, b)
        except ValueError:
            b = arrays.get(self.col_b)
            if b is None:
                return np.zeros(len(a), dtype=np.bool_)

        if self.direction == "over":
            result = (a > b) & (np.roll(a, 1) <= np.roll(b, 1))
        else:
            result = (a < b) & (np.roll(a, 1) >= np.roll(b, 1))
        # Bar 0 has no prior bar — np.roll wraps last element, so suppress it
        result[0] = False
        return result

    def __repr__(self):
        return f"CROSS{'OVER' if self.direction == 'over' else 'UNDER'}({self.col_a}, {self.col_b})"


def _clean_condition(cond: str) -> str:
    """Strip inline comments and whitespace from a condition string.

    Careful not to strip function call parentheses like CROSSUNDER(low, pdl).
    Only strip trailing parenthesized text that looks like a comment.
    """
    cond = cond.strip()
    cond = _RE_INLINE_COMMENT.sub("", cond)
    # Only strip trailing parens if they don't look like a function call
    # A function call has the form WORD(... at the start
    if not re.match(r"^[A-Za-z_]+\(", cond):
        cond = _RE_COMMENT.sub("", cond)
    else:
        # For function calls, strip only trailing comments after closing paren
        # e.g., "CROSSUNDER(low, pdl) (some comment)" → "CROSSUNDER(low, pdl)"
        pass
    return cond.strip()


def parse_condition(cond_str: str) -> list[ParsedCondition | CrossCondition]:
    """Parse a single condition string into ParsedCondition(s).

    Returns list because one string may contain AND clauses.
    Returns empty list if unparseable.
    """
    cond_str = _clean_condition(cond_str)

    if cond_str.lower() in _SKIP_PATTERNS:
        return []

    results = []

    # Check for AND compound conditions first
    if _RE_AND.search(cond_str):
        parts = _RE_AND.split(cond_str)
        for part in parts:
            sub = parse_condition(part.strip())
            results.extend(sub)
        return results

    # Check for OR conditions — take the first parseable one
    if _RE_OR.search(cond_str) and not _RE_VIX_RANGE.search(cond_str):
        parts = _RE_OR.split(cond_str)
        for part in parts:
            sub = parse_condition(part.strip())
            if sub:
                return sub  # Use first parseable OR branch
        return []

    # Crossover/crossunder
    m = _RE_CROSSOVER.search(cond_str)
    if m:
        return [CrossCondition(m.group(1), m.group(2), "over")]
    m = _RE_CROSSUNDER.search(cond_str)
    if m:
        return [CrossCondition(m.group(1), m.group(2), "under")]

    # ABS(expr) comparison
    m = _RE_ABS.match(cond_str)
    if m:
        lhs, op, rhs = m.group(1), m.group(2), float(m.group(3))
        return [ParsedCondition(lhs, op, rhs, is_abs=True)]

    # VIX range: "vix > X AND vix < Y"
    m = _RE_VIX_RANGE.search(cond_str)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return [
            ParsedCondition("vix", ">", lo),
            ParsedCondition("vix", "<", hi),
        ]

    # Standard comparison
    m = _RE_COMPARISON.match(cond_str)
    if m:
        lhs, op, rhs = m.group(1), m.group(2), m.group(3)
        op = _OPS.get(op, op)
        # Try to parse RHS as number
        try:
            rhs_val = float(rhs)
            return [ParsedCondition(lhs, op, rhs_val)]
        except ValueError:
            return [ParsedCondition(lhs, op, rhs)]

    # Boolean conditions: "vol_spike == True"
    m = re.match(r"^\s*(\w+)\s*==\s*(True|true|1)\s*$", cond_str)
    if m:
        return [ParsedCondition(m.group(1), ">", 0.5)]

    # Time-based conditions: "time > 09:30 IST"
    m = re.match(r"time\s*>\s*(\d{2}):(\d{2})", cond_str)
    if m:
        minutes = int(m.group(1)) * 60 + int(m.group(2))
        return [ParsedCondition("_time_minutes", ">", float(minutes))]

    # Bar count: "bar_count_from_open < 180"
    m = re.match(r"bar_count_from_open\s*(>=|<=|>|<)\s*(\d+)", cond_str)
    if m:
        return [ParsedCondition("bar_count_from_open", m.group(1), float(m.group(2)))]

    # ── Extended patterns for better coverage ───────────────────────────

    # "volume > SMA(volume, 20) * 1.5" or "volume on breakout bar > SMA(volume, 20) * 1.3"
    m = re.search(r"volume\s*(>=|<=|>|<)\s*(?:SMA\(\s*volume\s*,\s*\d+\s*\)\s*\*?\s*)?([\d.]+)", cond_str, re.IGNORECASE)
    if m and "sma" in cond_str.lower():
        op = m.group(1)
        mult = float(m.group(2))
        return [ParsedCondition("rel_volume", op, mult)]

    # "volume > 2 * avgvol30" or "volume > 2x 20-bar SMA"
    m = re.search(r"volume\s*(>=|<=|>|<)\s*([\d.]+)\s*(?:\*|x|×)\s*(?:avg|SMA|sma)", cond_str, re.IGNORECASE)
    if m:
        op = m.group(1)
        mult = float(m.group(2))
        return [ParsedCondition("rel_volume", op, mult)]

    # "close > vwap - 0.5%" or "close < vwap + 0.5%"
    m = re.match(r"(\w+)\s*(>|<|>=|<=)\s*(\w+)\s*([+-])\s*([\d.]+)%", cond_str)
    if m:
        lhs, op, ref, sign, pct = m.group(1), m.group(2), m.group(3), m.group(4), float(m.group(5))
        # This is a deviation condition: close > vwap * (1 - pct/100)
        # Simplify: treat as "close > vwap" (the pct is a buffer)
        return [ParsedCondition(lhs, op, ref)]

    # "morning_return > 0.3%" → treat column name as parseable
    m = re.match(r"(\w+)\s*(>=|<=|>|<|==)\s*([\-+]?[\d.]+)%?", cond_str)
    if m:
        lhs, op, rhs = m.group(1), m.group(2), m.group(3)
        rhs_val = float(rhs)
        # If it ends with %, convert
        if cond_str.rstrip().endswith("%") and abs(rhs_val) > 0.001 and abs(rhs_val) < 100:
            rhs_val = rhs_val / 100.0
        return [ParsedCondition(lhs, op, rhs_val)]

    # "close stays above PDH for 3 consecutive bars" → simplify to close > pdh
    m = re.search(r"(\w+)\s+(?:stays?\s+)?(?:above|below)\s+(\w+)", cond_str, re.IGNORECASE)
    if m:
        lhs, rhs = m.group(1), m.group(2).lower()
        if "above" in cond_str.lower():
            return [ParsedCondition(lhs, ">", rhs)]
        else:
            return [ParsedCondition(lhs, "<", rhs)]

    # Last resort: try to extract any comparison pattern from within the text
    m = re.search(r"(\w+)\s*(>=|<=|!=|>|<|==)\s*([\-+]?[\d.]+|[A-Za-z_]\w*)", cond_str)
    if m:
        lhs, op, rhs = m.group(1), m.group(2), m.group(3)
        op = _OPS.get(op, op)
        try:
            rhs_val = float(rhs)
            return [ParsedCondition(lhs, op, rhs_val)]
        except ValueError:
            return [ParsedCondition(lhs, op, rhs)]

    log.debug("Unparseable condition: %s", cond_str)
    return []


def parse_conditions_list(conditions: list[str]) -> tuple[list[ParsedCondition | CrossCondition], list[str]]:
    """Parse a list of condition strings. Returns (parsed, unparseable)."""
    parsed = []
    unparseable = []
    for cond in conditions:
        if not cond or cond.lower().strip() in _SKIP_PATTERNS:
            continue
        result = parse_condition(cond)
        if result:
            parsed.extend(result)
        else:
            unparseable.append(cond)
    return parsed, unparseable


def evaluate_conditions(
    conditions: list[ParsedCondition | CrossCondition],
    arrays: dict[str, np.ndarray],
) -> np.ndarray:
    """Evaluate all conditions ANDed together. Returns boolean mask."""
    n = len(next(iter(arrays.values())))
    mask = np.ones(n, dtype=np.bool_)
    for cond in conditions:
        result = cond.evaluate(arrays)
        mask &= result
    return mask


# ── Exit rule parsing ───────────────────────────────────────────────────────

class ExitRules:
    """Parsed exit rules for a strategy."""

    def __init__(self):
        self.stop_loss_pct: Optional[float] = None       # as decimal (0.003 = 0.3%)
        self.stop_loss_atr_mult: Optional[float] = None
        self.stop_loss_atr_period: int = 20
        self.target_pct: Optional[float] = None           # as decimal
        self.target_atr_mult: Optional[float] = None
        self.target_atr_period: int = 20
        self.target_vwap: bool = False
        self.target_indicator: Optional[str] = None       # e.g., "ema_50"
        self.trailing_stop_pct: Optional[float] = None    # as decimal
        self.trailing_stop_atr_mult: Optional[float] = None
        self.trailing_activate_pct: Optional[float] = None  # activate after this profit %
        self.breakeven_after_pct: Optional[float] = None
        self.time_stop_bars: Optional[int] = None
        self.signal_exit_conditions: list[ParsedCondition | CrossCondition] = []
        self.eod_flatten: bool = True  # always True per spec

    def get_tunable_params(self) -> dict[str, tuple[float, float, float]]:
        """Return tunable parameters: {name: (default, lo, hi)}."""
        params = {}
        if self.stop_loss_pct is not None:
            default = self.stop_loss_pct * 100  # store as percentage for readability
            params["stop_loss_pct"] = (default, max(0.05, default * 0.3), min(3.0, default * 3.0))
        if self.stop_loss_atr_mult is not None:
            params["stop_loss_atr_mult"] = (self.stop_loss_atr_mult,
                                            max(0.3, self.stop_loss_atr_mult * 0.3),
                                            min(5.0, self.stop_loss_atr_mult * 3.0))
        if self.target_pct is not None:
            default = self.target_pct * 100
            params["target_pct"] = (default, max(0.05, default * 0.3), min(5.0, default * 3.0))
        if self.target_atr_mult is not None:
            params["target_atr_mult"] = (self.target_atr_mult,
                                         max(0.3, self.target_atr_mult * 0.3),
                                         min(5.0, self.target_atr_mult * 3.0))
        if self.trailing_stop_pct is not None:
            default = self.trailing_stop_pct * 100
            params["trailing_stop_pct"] = (default, max(0.05, default * 0.3), min(3.0, default * 3.0))
        if self.time_stop_bars is not None:
            params["time_stop_bars"] = (float(self.time_stop_bars),
                                        max(5.0, self.time_stop_bars * 0.3),
                                        min(300.0, self.time_stop_bars * 3.0))
        return params

    def __repr__(self):
        parts = []
        if self.stop_loss_pct:
            parts.append(f"SL={self.stop_loss_pct*100:.2f}%")
        if self.stop_loss_atr_mult:
            parts.append(f"SL={self.stop_loss_atr_mult}*ATR({self.stop_loss_atr_period})")
        if self.target_pct:
            parts.append(f"TP={self.target_pct*100:.2f}%")
        if self.target_vwap:
            parts.append("TP=VWAP")
        if self.trailing_stop_pct:
            parts.append(f"Trail={self.trailing_stop_pct*100:.2f}%")
        if self.time_stop_bars:
            parts.append(f"Time={self.time_stop_bars}bars")
        return f"ExitRules({', '.join(parts)})"


def parse_exit_rules(exit_dict: dict, max_hold_bars: Optional[int] = None) -> ExitRules:
    """Parse the exit section of a strategy JSON."""
    rules = ExitRules()

    # ── Stop loss ────────────────────────────────────────────────────────
    sl_str = str(exit_dict.get("stop_loss", "")).strip()
    if sl_str and sl_str.lower() not in _SKIP_PATTERNS:
        # Try percentage first
        m = _RE_EXIT_PCT.search(sl_str)
        if m:
            rules.stop_loss_pct = float(m.group(1)) / 100.0

        # Try ATR multiple
        m = _RE_EXIT_ATR.search(sl_str)
        if m:
            rules.stop_loss_atr_mult = float(m.group(1))
            rules.stop_loss_atr_period = int(m.group(2))

        # If both found (OR condition), use percentage as primary
        if rules.stop_loss_pct is None and rules.stop_loss_atr_mult is None:
            log.debug("Unparseable stop loss: %s", sl_str)

    # ── Target ───────────────────────────────────────────────────────────
    tgt_str = str(exit_dict.get("target", "")).strip()
    if tgt_str and tgt_str.lower() not in _SKIP_PATTERNS:
        if _RE_EXIT_VWAP_TOUCH.search(tgt_str):
            rules.target_vwap = True
        m = _RE_EXIT_PCT.search(tgt_str)
        if m:
            rules.target_pct = float(m.group(1)) / 100.0
        m = _RE_EXIT_ATR.search(tgt_str)
        if m:
            rules.target_atr_mult = float(m.group(1))
            rules.target_atr_period = int(m.group(2))
        # Check for indicator-based target
        m = re.search(r"(ema_\d+|EMA\(\w+,\s*\d+\))", tgt_str, re.IGNORECASE)
        if m:
            rules.target_indicator = m.group(1).lower()

        # zscore cross
        m = _RE_EXIT_ZSCORE_CROSS.search(tgt_str)
        if m:
            col = m.group(1)
            val = 0.0 if m.group(2).lower() in ("zero", "0") else float(m.group(2))
            rules.signal_exit_conditions.append(CrossCondition(col, str(val), "over"))

    # ── Trailing stop ────────────────────────────────────────────────────
    trail_str = str(exit_dict.get("trailing_stop", "")).strip()
    if trail_str and trail_str.lower() not in _SKIP_PATTERNS:
        # Breakeven after X%
        m = _RE_TRAIL_BE.search(trail_str)
        if m:
            rules.breakeven_after_pct = float(m.group(1)) / 100.0

        # Trail percentage
        m = _RE_TRAIL_PCT.search(trail_str)
        if m:
            rules.trailing_stop_pct = float(m.group(1)) / 100.0

        # Trail ATR
        m = _RE_TRAIL_ATR.search(trail_str)
        if m:
            rules.trailing_stop_atr_mult = float(m.group(1))

        # Activation threshold
        m = _RE_TRAIL_ACTIVATE.search(trail_str)
        if m:
            rules.trailing_activate_pct = float(m.group(1)) / 100.0

    # ── Time stop ────────────────────────────────────────────────────────
    ts_str = str(exit_dict.get("time_stop", "")).strip()
    if ts_str and ts_str.lower() not in _SKIP_PATTERNS:
        m = _RE_EXIT_BARS.search(ts_str)
        if m:
            rules.time_stop_bars = int(m.group(1))
        else:
            m = _RE_EXIT_TIME.search(ts_str)
            if m:
                rules.time_stop_bars = int(m.group(1))

    # Reconcile with max_hold_bars
    if max_hold_bars is not None and max_hold_bars > 0:
        if rules.time_stop_bars is not None:
            rules.time_stop_bars = min(rules.time_stop_bars, max_hold_bars)
        else:
            rules.time_stop_bars = max_hold_bars

    # ── Signal exit ──────────────────────────────────────────────────────
    sig_str = str(exit_dict.get("signal_exit", "")).strip()
    if sig_str and sig_str.lower() not in _SKIP_PATTERNS:
        m = _RE_EXIT_ZSCORE_CROSS.search(sig_str)
        if m:
            col = m.group(1)
            val = 0.0 if m.group(2).lower() in ("zero", "0") else float(m.group(2))
            rules.signal_exit_conditions.append(CrossCondition(col, str(val), "over"))
        else:
            # Try parsing as regular condition
            parsed = parse_condition(sig_str)
            if parsed:
                rules.signal_exit_conditions.extend(parsed)

    # Default stop loss if nothing specified
    if rules.stop_loss_pct is None and rules.stop_loss_atr_mult is None:
        rules.stop_loss_pct = 0.005  # 0.5% default

    return rules


# ── Time filter parsing ─────────────────────────────────────────────────────

class TimeFilter:
    """Parsed time window for trading."""

    def __init__(self):
        self.start_h: int = 9
        self.start_m: int = 15
        self.end_h: int = 15
        self.end_m: int = 20
        self.skip_first_n_min: int = 0
        self.skip_last_n_min: int = 0
        self.skip_expiry: bool = False

    @property
    def start_minutes(self) -> int:
        return self.start_h * 60 + self.start_m

    @property
    def end_minutes(self) -> int:
        return self.end_h * 60 + self.end_m

    def effective_start_minutes(self) -> int:
        """Start time with skip_first_n_min applied."""
        return self.start_minutes + self.skip_first_n_min

    def effective_end_minutes(self) -> int:
        """End time with skip_last_n_min applied."""
        return self.end_minutes - self.skip_last_n_min


def parse_session(session_str: str) -> tuple[int, int, int, int]:
    """Parse session string like '09:20-15:15' into (start_h, start_m, end_h, end_m)."""
    m = re.match(r"(\d{2}):(\d{2})\s*[-–]\s*(\d{2}):(\d{2})", session_str.strip())
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return 9, 15, 15, 20  # default


def parse_time_filter(session_str: str, filters: dict) -> TimeFilter:
    """Parse session and filter dict into a TimeFilter."""
    tf = TimeFilter()

    # Parse session
    if session_str and session_str.strip():
        tf.start_h, tf.start_m, tf.end_h, tf.end_m = parse_session(session_str)

    # Parse time_filter
    time_str = str(filters.get("time_filter", "")).strip()
    if time_str and time_str.lower() not in _SKIP_PATTERNS:
        m = _RE_TIME_NO_FIRST.search(time_str)
        if m:
            tf.skip_first_n_min = int(m.group(1))

        m = _RE_TIME_NO_LAST.search(time_str)
        if m:
            tf.skip_last_n_min = int(m.group(1))

        m = _RE_TIME_WINDOW.search(time_str)
        if m:
            # Override with explicit window
            start_h, start_m = int(m.group(1)), int(m.group(2))
            end_h, end_m = int(m.group(3)), int(m.group(4))
            new_start = max(tf.start_h * 60 + tf.start_m, start_h * 60 + start_m)
            tf.start_h, tf.start_m = new_start // 60, new_start % 60
            new_end = min(tf.end_h * 60 + tf.end_m, end_h * 60 + end_m)
            tf.end_h, tf.end_m = new_end // 60, new_end % 60

        m = _RE_TIME_BEFORE.search(time_str)
        if m:
            h, mi = int(m.group(1)), int(m.group(2))
            new_start = h * 60 + mi
            if new_start > tf.start_minutes:
                tf.start_h, tf.start_m = h, mi

        m = _RE_TIME_AFTER.search(time_str)
        if m:
            h, mi = int(m.group(1)), int(m.group(2))
            new_end = h * 60 + mi
            if new_end < tf.end_minutes:
                tf.end_h, tf.end_m = h, mi

        m = _RE_TIME_SKIP_FIRST_N.search(time_str)
        if m:
            tf.skip_first_n_min = max(tf.skip_first_n_min, int(m.group(1)))

        m = _RE_TIME_ONLY_FIRST_N.search(time_str)
        if m:
            only_min = int(m.group(1))
            tf.end_h = (tf.start_minutes + only_min) // 60
            tf.end_m = (tf.start_minutes + only_min) % 60

    # Check event_filter for expiry day
    event_str = str(filters.get("event_filter", "")).strip()
    if "expiry" in event_str.lower():
        tf.skip_expiry = True

    return tf


# ── VIX filter parsing ──────────────────────────────────────────────────────

def parse_vix_filter(filters: dict) -> list[ParsedCondition]:
    """Parse vix_filter from filters dict."""
    vix_str = str(filters.get("vix_filter", "")).strip()
    if not vix_str or vix_str.lower() in _SKIP_PATTERNS:
        return []

    # Check "no filter" patterns
    if "no filter" in vix_str.lower() or "all" in vix_str.lower():
        return []

    conditions = []
    # Range: "vix > 12 AND vix < 25"
    m = _RE_VIX_RANGE.search(vix_str)
    if m:
        conditions.append(ParsedCondition("vix", ">", float(m.group(1))))
        conditions.append(ParsedCondition("vix", "<", float(m.group(2))))
        return conditions

    # Single comparison
    m = _RE_VIX.search(vix_str)
    if m:
        conditions.append(ParsedCondition("vix", m.group(1), float(m.group(2))))
        return conditions

    return []


def parse_volume_filter(filters: dict) -> list[ParsedCondition]:
    """Parse volume_filter from filters dict."""
    vol_str = str(filters.get("volume_filter", "")).strip()
    if not vol_str or vol_str.lower() in _SKIP_PATTERNS:
        return []

    # "skip if volume < 50% of 20-bar avg" → volume > 0.5 * SMA(volume, 20)
    m = re.search(r"(?:skip if\s+)?(?:rel_)?volume\s*[<>]\s*([\d.]+)%?\s*of\s*(\d+)[\s-]*bar", vol_str, re.IGNORECASE)
    if m:
        pct = float(m.group(1))
        if pct > 1:
            pct = pct / 100.0
        return [ParsedCondition("rel_volume", ">", pct)]

    # "skip if rel_volume_20 < 1.2"
    m = re.search(r"(?:skip if\s+)?(\w*volume\w*)\s*(>=|<=|>|<)\s*([\d.]+)", vol_str, re.IGNORECASE)
    if m:
        col, op, val = m.group(1), m.group(2), float(m.group(3))
        # "skip if X < Y" → X >= Y
        if "skip" in vol_str.lower():
            if op == "<":
                op = ">="
            elif op == "<=":
                op = ">"
        return [ParsedCondition(col, op, val)]

    return []
