"""
1A: Profit Factor Recalculation
Fixes profit_factor_estimate for strategies listed in math_failures.check8_pf
"""
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE12 = os.path.join(BASE, "outputs", "phase12_results.json")
STRAT_DIR = os.path.join(BASE, "unique_strategies")

with open(PHASE12) as f:
    phase12 = json.load(f)

entries = phase12["math_failures"]["check8_pf"]
print(f"Strategies to fix: {len(entries)}")

fixed = 0
errors = []
changes = []

for entry in entries:
    name = entry["name"]
    path = os.path.join(STRAT_DIR, f"{name}.json")
    if not os.path.exists(path):
        errors.append(f"File not found: {path}")
        continue
    try:
        with open(path) as f:
            data = json.load(f)
        edge = data.get("edge", {})
        win_rate = edge.get("win_rate_estimate")
        avg_wl = edge.get("avg_winner_to_loser")
        if win_rate is None or avg_wl is None:
            errors.append(f"{name}: missing win_rate_estimate or avg_winner_to_loser")
            continue
        if win_rate >= 1.0 or win_rate <= 0.0:
            errors.append(f"{name}: invalid win_rate {win_rate}")
            continue
        old_pf = edge.get("profit_factor_estimate")
        correct_pf = round((win_rate * avg_wl) / (1 - win_rate), 2)
        data["edge"]["profit_factor_estimate"] = correct_pf
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        changes.append((name, old_pf, correct_pf))
        fixed += 1
    except Exception as e:
        errors.append(f"{name}: {e}")

print(f"Fixed: {fixed}")
if errors:
    print(f"Errors ({len(errors)}):")
    for e in errors:
        print(f"  {e}")

# Write changes to a temp file for the audit log
changes_path = os.path.join(BASE, "outputs", "fix_1a_changes.json")
with open(changes_path, "w") as f:
    json.dump(changes, f, indent=2)
print(f"Changes saved to {changes_path}")
