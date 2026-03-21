"""
1D: Tag normalization
Replace "mean-reversion" with "mean_reversion" in tags arrays across all strategy JSONs.
"""
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRAT_DIR = os.path.join(BASE, "unique_strategies")

files_changed = 0
changed_files = []

for fname in sorted(os.listdir(STRAT_DIR)):
    if not fname.endswith(".json"):
        continue
    path = os.path.join(STRAT_DIR, fname)
    try:
        with open(path) as f:
            data = json.load(f)
        tags = data.get("tags", [])
        if not isinstance(tags, list):
            continue
        new_tags = ["mean_reversion" if t == "mean-reversion" else t for t in tags]
        if new_tags != tags:
            data["tags"] = new_tags
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            files_changed += 1
            changed_files.append(fname[:-5])
    except Exception as e:
        print(f"Error processing {fname}: {e}")

print(f"Files changed: {files_changed}")
for name in changed_files:
    print(f"  {name}")

# Save for audit log
changes_path = os.path.join(BASE, "outputs", "fix_1d_changes.json")
with open(changes_path, "w") as f:
    json.dump(changed_files, f, indent=2)
