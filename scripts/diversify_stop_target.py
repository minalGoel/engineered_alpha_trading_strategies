"""
diversify_stop_target.py
------------------------
Fixes the stop/target homogeneity problem: 113+ strategies all use stop=4, target=7
(a codegen default). This script diversifies them based on mechanism and hold time.

Only strategies with EXACTLY stop_points=4.0 AND target_points=7.0 are touched.
"""

import json
import os
import re
import sys
from collections import Counter

STRAT_DIR = "unique_strategies"
PY_DIR = "pipeline/strategies"


def assign_stop_target(mechanism: str, hold_min: int, hold_max: int, tags: list) -> tuple:
    """
    Returns (stop, target) based on mechanism type and hold time.
    Rules:
    - Scalp / very fast (hold_max <= 15s): stop=2.5, target=4.0
    - Fast mean reversion (hold_max <= 30s OR "mean_reversion" in tags AND hold_max <= 60s): stop=3.0, target=5.0
    - Standard mean reversion (hold 30-60s, mean_reversion): stop=3.0, target=6.0
    - Breakout/momentum fast (breakout/momentum, hold <= 60s): stop=4.0, target=6.0  [slight variation]
    - Breakout/momentum standard (hold 60-120s): stop=5.0, target=8.0
    - Momentum scalp (momentum, hold <= 30s): stop=3.0, target=5.0
    - VWAP reversion: stop=3.0, target=6.0
    - Gap strategies: stop=4.0, target=8.0
    - Default (unclassified): keep 4.0/7.0 — do NOT change
    """
    tags_lower = [t.lower() for t in tags]
    mech_lower = mechanism.lower() if mechanism else ""

    is_momentum = any(t in tags_lower for t in ['momentum', 'breakout', 'trend_following', 'trend-following'])
    is_reversion = any(t in tags_lower for t in ['mean_reversion', 'mean-reversion', 'reversion', 'vwap_reversion'])
    is_vwap = 'vwap' in tags_lower or 'vwap' in mech_lower
    is_gap = 'gap' in tags_lower or 'gap' in mech_lower

    if hold_max <= 15:
        return 2.5, 4.0
    elif is_reversion and hold_max <= 30:
        return 3.0, 5.0
    elif is_reversion and hold_max <= 60:
        return 3.0, 6.0
    elif is_gap:
        return 4.0, 8.0
    elif is_vwap and is_reversion:
        return 3.0, 6.0
    elif is_momentum and hold_max <= 60:
        return 4.0, 6.0
    elif is_momentum and hold_max > 60:
        return 5.0, 8.0
    elif is_reversion:
        return 3.0, 6.0
    else:
        return None, None  # No change — unclassified


def parse_hold_time(hold_time_seconds) -> tuple:
    """Parse hold_time_seconds field, returns (hold_min, hold_max) in seconds."""
    if isinstance(hold_time_seconds, dict):
        hold_min = hold_time_seconds.get("min", hold_time_seconds.get("minimum", 0))
        hold_max = hold_time_seconds.get("max", hold_time_seconds.get("maximum", 60))
    elif isinstance(hold_time_seconds, list):
        if len(hold_time_seconds) >= 2:
            hold_min = hold_time_seconds[0]
            hold_max = hold_time_seconds[1]
        elif len(hold_time_seconds) == 1:
            hold_min = 0
            hold_max = hold_time_seconds[0]
        else:
            hold_min = 0
            hold_max = 60
    elif isinstance(hold_time_seconds, (int, float)):
        hold_min = 0
        hold_max = int(hold_time_seconds)
    else:
        hold_min = 0
        hold_max = 60
    return int(hold_min), int(hold_max)


def update_python_file(py_path: str, new_stop: float, new_target: float) -> bool:
    """
    Update stop_points and target_points in the Python strategy file.
    Handles two patterns:
    1. Hardcoded: stop_points=np.full(n, 4.0)
    2. Tunable param: TunableParam("stop_pts", 4.0, ...) and params.get("stop_pts", 4.0)
    Returns True if changes were made.
    """
    if not os.path.exists(py_path):
        return False

    with open(py_path, "r") as f:
        content = f.read()

    original = content

    # Format numbers: always use float notation for clarity
    def fmt(v):
        return str(float(v))

    new_stop_str = fmt(new_stop)
    new_target_str = fmt(new_target)

    # --- Pattern 1: Hardcoded np.full(n, 4.0) ---
    # stop_points=np.full(n, 4.0)  (with or without trailing comma)
    content = re.sub(
        r'(stop_points\s*=\s*np\.full\s*\(\s*n\s*,\s*)4\.0(\s*\)[,]?)',
        lambda m: m.group(1) + new_stop_str + m.group(2),
        content
    )
    # target_points=np.full(n, 7.0)
    content = re.sub(
        r'(target_points\s*=\s*np\.full\s*\(\s*n\s*,\s*)7\.0(\s*\)[,]?)',
        lambda m: m.group(1) + new_target_str + m.group(2),
        content
    )

    # --- Pattern 2: TunableParam("stop_pts", 4.0, ...) ---
    # Update the default value (second argument) in TunableParam for stop_pts
    content = re.sub(
        r'(TunableParam\s*\(\s*["\']stop_pts["\']\s*,\s*)4\.0(\s*,)',
        lambda m: m.group(1) + new_stop_str + m.group(2),
        content
    )
    # Update the default value in TunableParam for target_pts
    content = re.sub(
        r'(TunableParam\s*\(\s*["\']target_pts["\']\s*,\s*)7\.0(\s*,)',
        lambda m: m.group(1) + new_target_str + m.group(2),
        content
    )

    # --- Pattern 3: params.get("stop_pts", 4.0) ---
    content = re.sub(
        r'(params\.get\s*\(\s*["\']stop_pts["\']\s*,\s*)4\.0(\s*\))',
        lambda m: m.group(1) + new_stop_str + m.group(2),
        content
    )
    content = re.sub(
        r'(params\.get\s*\(\s*["\']target_pts["\']\s*,\s*)7\.0(\s*\))',
        lambda m: m.group(1) + new_target_str + m.group(2),
        content
    )

    if content != original:
        with open(py_path, "w") as f:
            f.write(content)
        return True
    return False


def main():
    output_lines = []

    def log(msg=""):
        print(msg)
        output_lines.append(msg)

    log("=" * 70)
    log("Stop/Target Diversification Script")
    log("=" * 70)
    log()

    changes = []
    skipped_unclassified = []
    skipped_same = []
    skipped_not_47 = 0
    json_missing_py = []

    # Collect all strategies
    strat_files = sorted(
        f for f in os.listdir(STRAT_DIR) if f.endswith(".json")
    )
    log(f"Total JSON files: {len(strat_files)}")

    # First pass: count strategies with stop=4, target=7
    candidate_count = 0
    for fname in strat_files:
        json_path = os.path.join(STRAT_DIR, fname)
        with open(json_path) as f:
            d = json.load(f)
        exit_ = d.get("exit", {})
        sp = exit_.get("stop_points")
        tp = exit_.get("target_points")
        if (sp == 4.0 or sp == 4) and (tp == 7.0 or tp == 7):
            candidate_count += 1

    log(f"Strategies with stop=4, target=7 (candidates): {candidate_count}")
    log()

    # Distribution before
    before_dist = Counter()
    for fname in strat_files:
        json_path = os.path.join(STRAT_DIR, fname)
        with open(json_path) as f:
            d = json.load(f)
        exit_ = d.get("exit", {})
        sp = exit_.get("stop_points")
        tp = exit_.get("target_points")
        if sp is not None and tp is not None and not isinstance(sp, dict) and not isinstance(tp, dict):
            before_dist[(sp, tp)] += 1

    log("Distribution BEFORE:")
    for pair, cnt in sorted(before_dist.items(), key=lambda x: -x[1]):
        log(f"  stop={pair[0]}, target={pair[1]}: {cnt} strategies")
    log()

    # Main processing loop
    log("Processing candidates...")
    log("-" * 70)

    for fname in strat_files:
        json_path = os.path.join(STRAT_DIR, fname)
        strat_name = fname.replace(".json", "")
        py_path = os.path.join(PY_DIR, strat_name + ".py")

        with open(json_path) as f:
            d = json.load(f)

        exit_ = d.get("exit", {})
        sp = exit_.get("stop_points")
        tp = exit_.get("target_points")

        # Only touch strategies with EXACTLY stop=4 AND target=7
        if not ((sp == 4.0 or sp == 4) and (tp == 7.0 or tp == 7)):
            skipped_not_47 += 1
            continue

        # Parse fields
        hold_time_seconds = d.get("hold_time_seconds")
        hold_min, hold_max = parse_hold_time(hold_time_seconds)

        mechanism = d.get("mechanism", "")
        tags = d.get("tags", [])

        # Compute new stop/target
        new_stop, new_target = assign_stop_target(mechanism, hold_min, hold_max, tags)

        # If unclassified, skip
        if new_stop is None:
            skipped_unclassified.append(strat_name)
            continue

        # If same as current (4/7 again), skip
        if new_stop == 4.0 and new_target == 7.0:
            skipped_same.append(strat_name)
            continue

        # --- Update JSON ---
        d["exit"]["stop_points"] = new_stop
        d["exit"]["target_points"] = new_target

        # Update avg_winner_to_loser
        new_awl = round(new_target / new_stop, 2)
        if "edge" in d:
            d["edge"]["avg_winner_to_loser"] = new_awl

            # Recompute profit_factor_estimate
            win_rate = d["edge"].get("win_rate_estimate")
            if win_rate is not None and win_rate > 0 and win_rate < 1:
                new_pf = round((win_rate * new_awl) / (1.0 - win_rate), 2)
                d["edge"]["profit_factor_estimate"] = new_pf

        # Save JSON
        with open(json_path, "w") as f:
            json.dump(d, f, indent=2)
            f.write("\n")

        # --- Update Python file ---
        py_updated = update_python_file(py_path, new_stop, new_target)
        if not py_updated and os.path.exists(py_path):
            # Python file exists but no 4.0/7.0 pattern found — log it
            json_missing_py.append(strat_name)

        changes.append({
            "name": strat_name,
            "old_stop": 4.0,
            "old_target": 7.0,
            "new_stop": new_stop,
            "new_target": new_target,
            "hold_min": hold_min,
            "hold_max": hold_max,
            "tags": tags[:4],
        })

        log(f"  {strat_name}: (4, 7) → ({new_stop}, {new_target})"
            f"  [hold={hold_min}-{hold_max}s, tags={tags[:3]}]")

    log()
    log("=" * 70)
    log(f"SUMMARY")
    log("=" * 70)
    log(f"Total changes made: {len(changes)}")
    log(f"Skipped (already different stop/target): {skipped_not_47}")
    log(f"Skipped (unclassified — kept 4/7): {len(skipped_unclassified)}")
    log(f"Skipped (assignment returned 4/7 — e.g. gap strategies already at 4/7): {len(skipped_same)}")
    log(f"Python files where no np.full(n, 4.0) pattern found: {len(json_missing_py)}")

    if json_missing_py:
        log()
        log("Python files without np.full(n, 4.0) pattern (manual review may be needed):")
        for s in json_missing_py:
            log(f"  {s}")

    if skipped_unclassified:
        log()
        log(f"Unclassified strategies (kept 4/7):")
        for s in skipped_unclassified:
            log(f"  {s}")

    # Distribution after
    log()
    after_dist = Counter()
    for fname in strat_files:
        json_path = os.path.join(STRAT_DIR, fname)
        with open(json_path) as f:
            d = json.load(f)
        exit_ = d.get("exit", {})
        sp_val = exit_.get("stop_points")
        tp_val = exit_.get("target_points")
        if sp_val is not None and tp_val is not None and not isinstance(sp_val, dict) and not isinstance(tp_val, dict):
            after_dist[(sp_val, tp_val)] += 1

    log("Distribution AFTER:")
    for pair, cnt in sorted(after_dist.items(), key=lambda x: -x[1]):
        log(f"  stop={pair[0]}, target={pair[1]}: {cnt} strategies")

    # Verification
    log()
    log("=" * 70)
    log("VERIFICATION")
    log("=" * 70)
    # Normalize distribution for verification (convert int keys to float)
    norm_dist = Counter()
    for (sp_k, tp_k), cnt_k in after_dist.items():
        try:
            norm_dist[(float(sp_k), float(tp_k))] += cnt_k
        except (ValueError, TypeError):
            norm_dist[(sp_k, tp_k)] += cnt_k

    unique_pairs = len(norm_dist)
    remaining_47 = norm_dist.get((4.0, 7.0), 0)
    total_strats = sum(norm_dist.values())
    pct_47 = round(100.0 * remaining_47 / total_strats, 1) if total_strats > 0 else 0

    log(f"Unique (stop, target) pairs: {unique_pairs} (must be >= 10)")
    log(f"Strategies still using (4, 7): {remaining_47} / {total_strats} = {pct_47}% (must be < 25%)")

    if unique_pairs >= 10:
        log("  PASS: >= 10 distinct pairs")
    else:
        log("  FAIL: < 10 distinct pairs")

    if pct_47 < 25.0:
        log(f"  PASS: (4,7) is {pct_47}% < 25%")
    else:
        log(f"  FAIL: (4,7) is {pct_47}% >= 25%")

    # Histogram of pairs
    log()
    log("Histogram of (stop, target) pairs:")
    for pair, cnt in sorted(after_dist.items(), key=lambda x: -x[1]):
        bar = "#" * min(cnt, 80)
        try:
            pair_str = f"({float(pair[0]):4.1f}, {float(pair[1]):4.1f})"
        except (ValueError, TypeError):
            pair_str = f"({str(pair[0])[:10]}, {str(pair[1])[:10]})"
        log(f"  {pair_str}: {cnt:3d}  {bar}")

    # Save output to /tmp
    output_text = "\n".join(output_lines)
    with open("/tmp/diversify_output.txt", "w") as f:
        f.write(output_text)
    print("\nOutput saved to /tmp/diversify_output.txt")

    # Build audit log entry
    top_pairs = sorted(after_dist.items(), key=lambda x: -x[1])[:8]
    dist_str = ", ".join(f"({p[0]},{p[1]}):{c}" for p, c in top_pairs)

    audit_entry = f"""
## Tier 3A — Stop/Target Diversification
Changed {len(changes)} strategies from (4,7) to diverse values.
Final distribution: {dist_str}
(4,7) remaining: {remaining_47} ({pct_47}%)
"""

    audit_log_path = "outputs/audit_fix_log.md"
    if os.path.exists(audit_log_path):
        with open(audit_log_path, "a") as f:
            f.write(audit_entry)
        print(f"\nAppended to {audit_log_path}")
    else:
        print(f"\nWARNING: {audit_log_path} not found — could not append audit entry")

    return changes, after_dist, remaining_47, total_strats


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
