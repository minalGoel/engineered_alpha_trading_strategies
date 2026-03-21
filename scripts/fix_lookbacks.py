"""
1E: Lookback type normalization
For each indicator where `lookback` is a string (not int):
- "real_time" → keep as-is
- all others ("session", "session_cumulative", "prev_day", "recursive", "derived", "dynamic",
  or any other non-"real_time" string) → convert to 0
"""
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRAT_DIR = os.path.join(BASE, "unique_strategies")

KEEP_AS_IS = {"real_time"}

files_changed = 0
total_indicators_fixed = 0
change_details = []

for fname in sorted(os.listdir(STRAT_DIR)):
    if not fname.endswith(".json"):
        continue
    path = os.path.join(STRAT_DIR, fname)
    try:
        with open(path) as f:
            data = json.load(f)
        indicators = data.get("indicators", [])
        if not isinstance(indicators, list):
            continue
        file_changed = False
        for ind in indicators:
            lb = ind.get("lookback")
            if isinstance(lb, str) and lb not in KEEP_AS_IS:
                ind_name = ind.get("name", "unknown")
                change_details.append({
                    "file": fname[:-5],
                    "indicator": ind_name,
                    "old": lb,
                    "new": 0,
                })
                ind["lookback"] = 0
                total_indicators_fixed += 1
                file_changed = True
        if file_changed:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            files_changed += 1
    except Exception as e:
        print(f"Error processing {fname}: {e}")

print(f"Files changed: {files_changed}")
print(f"Total indicators fixed: {total_indicators_fixed}")

# Save for audit log
changes_path = os.path.join(BASE, "outputs", "fix_1e_changes.json")
with open(changes_path, "w") as f:
    json.dump(change_details, f, indent=2)
print(f"Changes saved to {changes_path}")
