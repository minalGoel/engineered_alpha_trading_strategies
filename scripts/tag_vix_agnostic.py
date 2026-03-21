import json
import os

PHASE4_PATH = "outputs/phase4_results.json"
STRATEGIES_DIR = "unique_strategies"

with open(PHASE4_PATH) as f:
    phase4 = json.load(f)

no_vix_entries = phase4.get("check19_no_vix", [])

tagged = 0
for entry in no_vix_entries:
    name = entry["name"]
    strategy_path = os.path.join(STRATEGIES_DIR, f"{name}.json")
    if not os.path.exists(strategy_path):
        print(f"WARNING: {strategy_path} not found, skipping")
        continue
    with open(strategy_path) as f:
        data = json.load(f)
    data["vix_agnostic"] = True
    with open(strategy_path, "w") as f:
        json.dump(data, f, indent=2)
    tagged += 1

print(f"Tagged {tagged} strategies with vix_agnostic: true")
