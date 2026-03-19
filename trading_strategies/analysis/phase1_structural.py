#!/usr/bin/env python3
"""
Phase 1: Load, validate, and structurally fingerprint all trading strategies.
Groups by mechanics hash, picks best-specified canonical per group.
"""

import json
import hashlib
import re
import os
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
INPUT_DIR = ROOT / "all_trading_strategies"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ─── Normalization helpers ────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation noise."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()
    # collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    # remove unicode dashes / special chars → plain text
    text = text.replace('\u2014', '-').replace('\u2019', "'").replace('\u20b9', 'inr')
    return text


def normalize_list(lst) -> list:
    """Normalize a list of strings."""
    if not isinstance(lst, list):
        return []
    return sorted(set(normalize_text(str(x)) for x in lst if x))


def extract_entry_conditions(entry: dict) -> list:
    """Extract all conditions from entry.long and entry.short."""
    conditions = []
    for side in ('long', 'short'):
        side_data = entry.get(side, {})
        if isinstance(side_data, dict):
            conds = side_data.get('conditions', [])
            if isinstance(conds, list):
                conditions.extend(str(c) for c in conds)
            conf = side_data.get('confirmation', '')
            if conf:
                conditions.append(str(conf))
    return sorted(normalize_text(c) for c in conditions if c)


def extract_exit_fields(exit_data: dict) -> list:
    """Extract all exit rule values."""
    fields = []
    for key in ('target', 'stop_loss', 'trailing_stop', 'time_stop', 'signal_exit'):
        val = exit_data.get(key, '')
        if val:
            fields.append(normalize_text(str(val)))
    return sorted(fields)


def extract_filters(filters: dict) -> list:
    """Extract all filter values."""
    fields = []
    if not isinstance(filters, dict):
        return fields
    for key, val in filters.items():
        if key == 'other_filters':
            if isinstance(val, list):
                fields.extend(normalize_text(str(v)) for v in val if v)
        elif val:
            fields.append(normalize_text(str(val)))
    return sorted(fields)


def extract_indicator_names(indicators) -> list:
    """Extract sorted lowercase indicator names."""
    if not isinstance(indicators, list):
        return []
    names = []
    for ind in indicators:
        if isinstance(ind, dict):
            name = ind.get('name', '')
            if name:
                names.append(normalize_text(str(name)))
    return sorted(set(names))


def make_hash(items: list) -> str:
    """SHA-256 hash of joined sorted items."""
    joined = '||'.join(items)
    return hashlib.sha256(joined.encode('utf-8')).hexdigest()[:16]


# ─── Completeness scoring ─────────────────────────────────────────────────────

def completeness_score(s: dict) -> float:
    """Score how complete/detailed a strategy is (higher = better)."""
    score = 0.0

    # Non-empty string fields
    for field in ('name', 'thesis', 'universe', 'timeframe', 'prior_art', 'notes'):
        val = s.get(field, '')
        if isinstance(val, str) and len(val.strip()) > 10:
            score += 1.0
        if isinstance(val, str):
            score += min(len(val) / 500, 2.0)  # up to 2 bonus for length

    # Tags count
    tags = s.get('tags', [])
    score += min(len(tags) if isinstance(tags, list) else 0, 5) * 0.3

    # Indicators count and detail
    indicators = s.get('indicators', [])
    if isinstance(indicators, list):
        score += min(len(indicators), 8) * 0.5
        for ind in indicators:
            if isinstance(ind, dict):
                if ind.get('formula', ''):
                    score += 0.2
                if ind.get('purpose', ''):
                    score += 0.1

    # Entry conditions count
    entry = s.get('entry', {})
    if isinstance(entry, dict):
        for side in ('long', 'short'):
            side_data = entry.get(side, {})
            if isinstance(side_data, dict):
                conds = side_data.get('conditions', [])
                score += min(len(conds) if isinstance(conds, list) else 0, 6) * 0.3

    # Weaknesses count (more = more thoughtful)
    weaknesses = s.get('weaknesses', [])
    score += min(len(weaknesses) if isinstance(weaknesses, list) else 0, 5) * 0.4

    # Edge data completeness
    edge = s.get('edge', {})
    if isinstance(edge, dict):
        for f in ('expected_bps', 'win_rate_estimate', 'avg_winner_to_loser', 'profit_factor_estimate'):
            if edge.get(f) is not None:
                score += 0.3

    # Risk data
    risk = s.get('risk', {})
    if isinstance(risk, dict):
        for f in ('capital_per_trade', 'max_risk_per_trade_pct', 'position_sizing'):
            if risk.get(f) is not None:
                score += 0.2

    return score


# ─── Load all strategies ──────────────────────────────────────────────────────

def load_all_strategies() -> list:
    """Load all JSON files from all_trading_strategies/ recursively."""
    strategies = []
    errors = []
    all_files = sorted(INPUT_DIR.rglob("*.json"))
    print(f"Found {len(all_files)} JSON files in {INPUT_DIR}")

    for fpath in all_files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data['_filepath'] = str(fpath.relative_to(ROOT))
            strategies.append(data)
        except Exception as e:
            errors.append({'file': str(fpath), 'error': str(e)})
            print(f"  ERROR loading {fpath}: {e}")

    print(f"Loaded {len(strategies)} strategies ({len(errors)} errors)")
    return strategies, errors


# ─── Fingerprinting ───────────────────────────────────────────────────────────

def fingerprint(s: dict) -> dict:
    """Compute all fingerprint hashes for a strategy."""
    # Tag set
    tags_raw = s.get('tags', [])
    tag_set = normalize_list(tags_raw)

    # Indicator names
    indicator_names = extract_indicator_names(s.get('indicators', []))

    # Entry hash
    entry_conditions = extract_entry_conditions(s.get('entry', {}))
    entry_hash = make_hash(entry_conditions) if entry_conditions else 'empty'

    # Exit hash
    exit_fields = extract_exit_fields(s.get('exit', {}))
    exit_hash = make_hash(exit_fields) if exit_fields else 'empty'

    # Mechanics hash: entry + exit + filters + indicator names
    filter_fields = extract_filters(s.get('filters', {}))
    mechanics_parts = entry_conditions + exit_fields + filter_fields + indicator_names
    mechanics_hash = make_hash(mechanics_parts) if mechanics_parts else 'empty'

    # Thesis hash
    thesis = normalize_text(s.get('thesis', ''))
    # Use first 200 chars of thesis for hash (ignore minor edits at end)
    thesis_hash = make_hash([thesis[:200]]) if thesis else 'empty'

    return {
        'filepath': s.get('_filepath', ''),
        'name': s.get('name', ''),
        'tag_set': tag_set,
        'indicator_names': indicator_names,
        'entry_hash': entry_hash,
        'exit_hash': exit_hash,
        'mechanics_hash': mechanics_hash,
        'thesis_hash': thesis_hash,
        'universe': normalize_text(str(s.get('universe', ''))),
        'timeframe': normalize_text(str(s.get('timeframe', ''))),
        'completeness': completeness_score(s),
    }


# ─── Main Phase 1 logic ───────────────────────────────────────────────────────

def run_phase1():
    print("\n" + "="*60)
    print("PHASE 1: Structural fingerprinting and deduplication")
    print("="*60)

    strategies, load_errors = load_all_strategies()
    total_loaded = len(strategies)

    # Compute fingerprints
    print("\nComputing fingerprints...")
    fp_list = []
    for s in strategies:
        fp = fingerprint(s)
        fp['_raw'] = s  # keep reference
        fp_list.append(fp)

    # Group by mechanics hash
    groups = defaultdict(list)
    for fp in fp_list:
        groups[fp['mechanics_hash']].append(fp)

    dupe_groups = {h: members for h, members in groups.items() if len(members) >= 2}
    singleton_groups = {h: members for h, members in groups.items() if len(members) == 1}

    print(f"\nGrouping results:")
    print(f"  Unique mechanics hashes: {len(groups)}")
    print(f"  Groups with 2+ members (structural dupes): {len(dupe_groups)}")
    print(f"  Singleton groups (unique mechanics): {len(singleton_groups)}")

    # Within each dupe group, pick best canonical
    survivors = []
    killed = []
    phase1_groups_output = []

    # Singletons all survive
    for h, members in singleton_groups.items():
        fp = members[0]
        survivors.append(fp)

    # Dupe groups: pick best
    for h, members in dupe_groups.items():
        members_sorted = sorted(members, key=lambda x: x['completeness'], reverse=True)
        canonical = members_sorted[0]
        dupes = members_sorted[1:]

        survivors.append(canonical)
        for d in dupes:
            killed.append({
                'filepath': d['filepath'],
                'name': d['name'],
                'reason': 'structural_duplicate',
                'mechanics_hash': h,
                'kept_instead': canonical['filepath'],
                'completeness_score': d['completeness'],
                'canonical_completeness': canonical['completeness'],
            })

        phase1_groups_output.append({
            'mechanics_hash': h,
            'canonical': {
                'filepath': canonical['filepath'],
                'name': canonical['name'],
                'completeness': canonical['completeness'],
            },
            'duplicates': [
                {
                    'filepath': d['filepath'],
                    'name': d['name'],
                    'completeness': d['completeness'],
                }
                for d in dupes
            ],
            'total_in_group': len(members),
        })

    # Also: check thesis hash groups (same thesis = same idea even if mechanics differ slightly)
    thesis_groups = defaultdict(list)
    for fp in fp_list:
        if fp['thesis_hash'] != 'empty':
            thesis_groups[fp['thesis_hash']].append(fp)

    thesis_dupe_groups = {h: members for h, members in thesis_groups.items() if len(members) >= 2}
    print(f"\nThesis hash groups with 2+ members: {len(thesis_dupe_groups)}")

    # Additional kill: if same thesis AND same universe AND same timeframe,
    # and not already killed by mechanics hash, kill the less complete one
    already_killed = {k['filepath'] for k in killed}
    extra_thesis_kills = []

    for h, members in thesis_dupe_groups.items():
        # Only consider members not already killed
        active = [m for m in members if m['filepath'] not in already_killed]
        if len(active) < 2:
            continue
        # Check if same universe + timeframe
        universes = set(m['universe'] for m in active)
        timeframes = set(m['timeframe'] for m in active)
        if len(universes) == 1 and len(timeframes) == 1:
            # Same thesis, same universe, same timeframe → structural near-dupe
            sorted_by_completeness = sorted(active, key=lambda x: x['completeness'], reverse=True)
            canonical = sorted_by_completeness[0]
            for d in sorted_by_completeness[1:]:
                if d['filepath'] not in already_killed:
                    extra_thesis_kills.append({
                        'filepath': d['filepath'],
                        'name': d['name'],
                        'reason': 'thesis_universe_timeframe_duplicate',
                        'thesis_hash': h,
                        'kept_instead': canonical['filepath'],
                        'completeness_score': d['completeness'],
                        'canonical_completeness': canonical['completeness'],
                    })
                    already_killed.add(d['filepath'])

    print(f"Additional thesis+universe+timeframe duplicates killed: {len(extra_thesis_kills)}")

    # Re-filter survivors after extra kills
    survivors_final = [fp for fp in survivors if fp['filepath'] not in already_killed]
    all_killed = killed + extra_thesis_kills

    # Save output
    output = {
        'summary': {
            'total_loaded': total_loaded,
            'load_errors': len(load_errors),
            'structural_dupe_groups': len(dupe_groups),
            'thesis_dupe_extra_kills': len(extra_thesis_kills),
            'total_killed': len(all_killed),
            'survivors': len(survivors_final),
        },
        'structural_dupe_groups': phase1_groups_output,
        'killed_strategies': all_killed,
        'load_errors': load_errors,
    }

    output_path = OUTPUT_DIR / "phase1_structural_dupes.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {output_path}")

    # Save survivor list for Phase 2
    survivors_data = []
    killed_fps = {k['filepath'] for k in all_killed}
    for s in strategies:
        fp_rel = s.get('_filepath', '')
        if fp_rel not in killed_fps:
            survivors_data.append(s)

    survivor_path = OUTPUT_DIR / "phase1_survivors.json"
    with open(survivor_path, 'w') as f:
        # Save just the raw strategy data list (not fingerprints)
        json.dump(survivors_data, f, indent=2)
    print(f"Saved survivors: {survivor_path}")

    print("\n" + "─"*60)
    print("PHASE 1 SUMMARY")
    print("─"*60)
    print(f"  Total strategies loaded  : {total_loaded}")
    print(f"  Structural dupe groups   : {len(dupe_groups)}")
    print(f"  Killed (mechanics hash)  : {len(killed)}")
    print(f"  Killed (thesis+univ+tf)  : {len(extra_thesis_kills)}")
    print(f"  Total killed             : {len(all_killed)}")
    print(f"  SURVIVING after Phase 1  : {len(survivors_final)}")
    print("─"*60)

    return survivors_data, output


if __name__ == '__main__':
    survivors, output = run_phase1()
