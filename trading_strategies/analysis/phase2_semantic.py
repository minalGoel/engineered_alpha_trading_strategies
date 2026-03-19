#!/usr/bin/env python3
"""
Phase 2: Semantic similarity via embeddings.
Uses all-MiniLM-L6-v2 to embed strategies and find near-duplicates (cosine > 0.90).
"""

import json
import re
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "outputs"


# ─── Text representation ──────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r'\s+', ' ', text.lower().strip())


def build_text_repr(s: dict) -> str:
    """Build a single text representation of a strategy for embedding."""
    parts = []

    # Name
    parts.append("name: " + str(s.get('name', '')))

    # Thesis
    thesis = s.get('thesis', '')
    if thesis:
        parts.append("thesis: " + str(thesis))

    # Tags
    tags = s.get('tags', [])
    if isinstance(tags, list) and tags:
        parts.append("tags: " + " ".join(str(t) for t in tags))

    # Prior art
    prior_art = s.get('prior_art', '')
    if prior_art:
        parts.append("prior_art: " + str(prior_art))

    # Universe and timeframe
    parts.append("universe: " + str(s.get('universe', '')))
    parts.append("timeframe: " + str(s.get('timeframe', '')))

    # Indicators
    indicators = s.get('indicators', [])
    if isinstance(indicators, list):
        ind_parts = []
        for ind in indicators:
            if isinstance(ind, dict):
                name = ind.get('name', '')
                formula = ind.get('formula', '')
                purpose = ind.get('purpose', '')
                ind_parts.append(f"{name} {formula} {purpose}".strip())
        if ind_parts:
            parts.append("indicators: " + "; ".join(ind_parts))

    # Entry conditions
    entry = s.get('entry', {})
    if isinstance(entry, dict):
        entry_parts = []
        for side in ('long', 'short'):
            side_data = entry.get(side, {})
            if isinstance(side_data, dict):
                conds = side_data.get('conditions', [])
                if isinstance(conds, list):
                    entry_parts.extend(str(c) for c in conds)
                conf = side_data.get('confirmation', '')
                if conf:
                    entry_parts.append(str(conf))
        if entry_parts:
            parts.append("entry: " + "; ".join(entry_parts))

    # Exit rules
    exit_data = s.get('exit', {})
    if isinstance(exit_data, dict):
        exit_parts = []
        for key in ('target', 'stop_loss', 'trailing_stop', 'time_stop', 'signal_exit'):
            val = exit_data.get(key, '')
            if val:
                exit_parts.append(f"{key}: {val}")
        if exit_parts:
            parts.append("exit: " + "; ".join(exit_parts))

    # Filters
    filters = s.get('filters', {})
    if isinstance(filters, dict):
        filter_parts = []
        for key, val in filters.items():
            if key == 'other_filters' and isinstance(val, list):
                filter_parts.extend(str(v) for v in val if v)
            elif val:
                filter_parts.append(str(val))
        if filter_parts:
            parts.append("filters: " + "; ".join(filter_parts))

    # Weaknesses
    weaknesses = s.get('weaknesses', [])
    if isinstance(weaknesses, list) and weaknesses:
        parts.append("weaknesses: " + "; ".join(str(w) for w in weaknesses))

    # Notes
    notes = s.get('notes', '')
    if notes:
        parts.append("notes: " + str(notes))

    return " | ".join(parts)


# ─── Structural comparison helpers ───────────────────────────────────────────

def get_indicator_names(s: dict) -> set:
    indicators = s.get('indicators', [])
    names = set()
    if isinstance(indicators, list):
        for ind in indicators:
            if isinstance(ind, dict):
                name = ind.get('name', '').lower().strip()
                if name:
                    names.add(name)
    return names


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    intersection = set_a & set_b
    return len(intersection) / len(union)


def tag_overlap_pct(s_a: dict, s_b: dict) -> float:
    tags_a = set(t.lower().strip() for t in s_a.get('tags', []) if t)
    tags_b = set(t.lower().strip() for t in s_b.get('tags', []) if t)
    if not tags_a and not tags_b:
        return 1.0
    if not tags_a or not tags_b:
        return 0.0
    return len(tags_a & tags_b) / max(len(tags_a), len(tags_b))


def structural_checks(s_a: dict, s_b: dict) -> dict:
    """Run 4 structural checks on a pair. Return scores and pass count."""
    checks = {}

    # 1. Same universe and timeframe?
    same_universe = (
        str(s_a.get('universe', '')).lower().strip() ==
        str(s_b.get('universe', '')).lower().strip()
    )
    same_timeframe = (
        str(s_a.get('timeframe', '')).lower().strip() ==
        str(s_b.get('timeframe', '')).lower().strip()
    )
    checks['same_universe_timeframe'] = same_universe and same_timeframe

    # 2. Indicator Jaccard similarity >= 0.5
    ind_a = get_indicator_names(s_a)
    ind_b = get_indicator_names(s_b)
    ind_jacc = jaccard(ind_a, ind_b)
    checks['indicator_jaccard'] = ind_jacc
    checks['indicator_overlap'] = ind_jacc >= 0.5

    # 3. Tag overlap >= 50%
    tag_ov = tag_overlap_pct(s_a, s_b)
    checks['tag_overlap_pct'] = tag_ov
    checks['tag_overlap'] = tag_ov >= 0.5

    # 4. Entry conditions logical equivalence (simplified: normalize + overlap)
    def get_entry_conds(s):
        conds = []
        entry = s.get('entry', {})
        if isinstance(entry, dict):
            for side in ('long', 'short'):
                side_data = entry.get(side, {})
                if isinstance(side_data, dict):
                    for c in side_data.get('conditions', []):
                        conds.append(normalize_text(str(c)))
        return set(conds)

    conds_a = get_entry_conds(s_a)
    conds_b = get_entry_conds(s_b)
    entry_jacc = jaccard(conds_a, conds_b)
    checks['entry_jaccard'] = entry_jacc
    checks['entry_overlap'] = entry_jacc >= 0.4

    checks['pass_count'] = sum([
        checks['same_universe_timeframe'],
        checks['indicator_overlap'],
        checks['tag_overlap'],
        checks['entry_overlap'],
    ])

    return checks


def completeness_score(s: dict) -> float:
    """Quick completeness score."""
    score = 0.0
    for field in ('name', 'thesis', 'universe', 'timeframe', 'prior_art', 'notes'):
        val = s.get(field, '')
        if isinstance(val, str) and len(val.strip()) > 10:
            score += 1.0
        if isinstance(val, str):
            score += min(len(val) / 500, 2.0)
    weaknesses = s.get('weaknesses', [])
    score += min(len(weaknesses) if isinstance(weaknesses, list) else 0, 5) * 0.4
    indicators = s.get('indicators', [])
    score += min(len(indicators) if isinstance(indicators, list) else 0, 8) * 0.5
    return score


# ─── Main Phase 2 logic ───────────────────────────────────────────────────────

def run_phase2():
    print("\n" + "="*60)
    print("PHASE 2: Semantic similarity via embeddings")
    print("="*60)

    # Load Phase 1 survivors
    survivor_path = OUTPUT_DIR / "phase1_survivors.json"
    with open(survivor_path, 'r') as f:
        strategies = json.load(f)

    print(f"Loaded {len(strategies)} Phase 1 survivors")

    # Build text representations
    print("Building text representations...")
    texts = [build_text_repr(s) for s in strategies]
    filepaths = [s.get('_filepath', '') for s in strategies]

    # Load model and embed
    print("Loading all-MiniLM-L6-v2 model...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print(f"Embedding {len(texts)} strategies (this may take a moment)...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    print(f"Embeddings shape: {embeddings.shape}")

    # Save embeddings for Phase 3
    np.save(OUTPUT_DIR / "phase2_embeddings.npy", embeddings)
    with open(OUTPUT_DIR / "phase2_filepaths.json", 'w') as f:
        json.dump(filepaths, f)
    print("Saved embeddings and filepath index")

    # Compute cosine similarity (dot product since embeddings are normalized)
    print("Computing pairwise cosine similarity (upper triangle)...")
    n = len(embeddings)
    total_pairs = n * (n - 1) // 2
    print(f"  Total pairs to check: {total_pairs:,}")

    # Use batched matrix multiplication for efficiency
    sim_matrix = embeddings @ embeddings.T  # shape (n, n)

    # Find pairs with cosine > 0.90
    THRESHOLD = 0.90
    near_dupe_pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sim_matrix[i, j])
            if sim > THRESHOLD:
                near_dupe_pairs.append((i, j, sim))

    print(f"Found {len(near_dupe_pairs)} pairs with cosine > {THRESHOLD}")

    # For each near-dupe pair, run structural checks
    killed = {}  # filepath → kill reason
    kill_decisions = []
    pairs_output = []

    # Sort by similarity descending so we process strongest dupes first
    near_dupe_pairs.sort(key=lambda x: x[2], reverse=True)

    for i, j, sim in near_dupe_pairs:
        s_a = strategies[i]
        s_b = strategies[j]
        fp_a = filepaths[i]
        fp_b = filepaths[j]

        checks = structural_checks(s_a, s_b)
        pass_count = checks['pass_count']

        pair_record = {
            'strategy_a': fp_a,
            'name_a': s_a.get('name', ''),
            'strategy_b': fp_b,
            'name_b': s_b.get('name', ''),
            'cosine_similarity': round(sim, 4),
            'structural_checks': {
                'same_universe_timeframe': checks['same_universe_timeframe'],
                'indicator_jaccard': round(checks['indicator_jaccard'], 3),
                'indicator_overlap_pass': checks['indicator_overlap'],
                'tag_overlap_pct': round(checks['tag_overlap_pct'], 3),
                'tag_overlap_pass': checks['tag_overlap'],
                'entry_jaccard': round(checks['entry_jaccard'], 3),
                'entry_overlap_pass': checks['entry_overlap'],
                'pass_count': pass_count,
            },
        }

        # Decision: kill if cosine > 0.90 AND pass_count >= 3
        if pass_count >= 3 and sim > THRESHOLD:
            # Determine which to kill (less complete)
            comp_a = completeness_score(s_a)
            comp_b = completeness_score(s_b)

            if fp_a in killed and fp_b not in killed:
                pass  # A already killed, B survives
            elif fp_b in killed and fp_a not in killed:
                pass  # B already killed, A survives
            elif fp_a in killed and fp_b in killed:
                pass  # Both already killed
            else:
                # Neither killed yet: kill less complete
                if comp_b <= comp_a:
                    killed[fp_b] = {'reason': 'semantic_near_duplicate', 'of': fp_a, 'cosine': sim}
                    pair_record['decision'] = 'KILL_B'
                    pair_record['reason'] = f'cosine={sim:.3f}, pass_count={pass_count}/4, B less complete'
                else:
                    killed[fp_a] = {'reason': 'semantic_near_duplicate', 'of': fp_b, 'cosine': sim}
                    pair_record['decision'] = 'KILL_A'
                    pair_record['reason'] = f'cosine={sim:.3f}, pass_count={pass_count}/4, A less complete'
        else:
            pair_record['decision'] = 'KEEP_BOTH'
            pair_record['reason'] = f'cosine={sim:.3f} but pass_count={pass_count}/4 (need 3+) or sim too low'

        pairs_output.append(pair_record)

    print(f"Semantic near-duplicates killed: {len(killed)}")

    # Build survivors
    survivors = [s for s in strategies if s.get('_filepath', '') not in killed]

    # Save output
    killed_list = [
        {
            'filepath': fp,
            'reason': info['reason'],
            'duplicate_of': info['of'],
            'cosine_similarity': round(info['cosine'], 4),
        }
        for fp, info in killed.items()
    ]

    output = {
        'summary': {
            'phase1_survivors': len(strategies),
            'pairs_checked': len(near_dupe_pairs),
            'semantic_duplicates_killed': len(killed),
            'survivors_after_phase2': len(survivors),
            'threshold_used': THRESHOLD,
        },
        'near_duplicate_pairs': pairs_output,
        'killed_strategies': killed_list,
    }

    output_path = OUTPUT_DIR / "phase2_semantic_dupes.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {output_path}")

    # Save survivors for Phase 3
    survivor_path2 = OUTPUT_DIR / "phase2_survivors.json"
    with open(survivor_path2, 'w') as f:
        json.dump(survivors, f, indent=2)
    print(f"Saved survivors: {survivor_path2}")

    # Also save embeddings indexed to survivors for Phase 3
    survivor_fps = [s.get('_filepath', '') for s in survivors]
    all_fps = filepaths
    survivor_indices = [all_fps.index(fp) for fp in survivor_fps if fp in all_fps]
    survivor_embeddings = embeddings[survivor_indices]
    np.save(OUTPUT_DIR / "phase2_survivor_embeddings.npy", survivor_embeddings)
    with open(OUTPUT_DIR / "phase2_survivor_filepaths.json", 'w') as f:
        json.dump(survivor_fps, f)
    print(f"Saved survivor embeddings: shape {survivor_embeddings.shape}")

    print("\n" + "─"*60)
    print("PHASE 2 SUMMARY")
    print("─"*60)
    print(f"  Phase 1 survivors        : {len(strategies)}")
    print(f"  Pairs checked (cos>0.90) : {len(near_dupe_pairs)}")
    print(f"  Semantic dupes killed    : {len(killed)}")
    print(f"  SURVIVING after Phase 2  : {len(survivors)}")
    print("─"*60)

    return survivors, output


if __name__ == '__main__':
    survivors, output = run_phase2()
