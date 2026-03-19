#!/usr/bin/env python3
"""
Phase 3: Family clustering, LLM-judged merges, and final dedup.
Step 3a: Cluster into families (agglomerative, distance threshold 0.35)
Step 3b: LLM judge on within-family pairs (claude-sonnet-4-6)
Step 3c: Execute merges
Step 3d: Final family cleanup (limit families with 5+ VARIANT members)
"""

import json
import re
import os
import asyncio
import time
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "outputs"

# ─── Helpers ──────────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r'\s+', ' ', text.lower().strip())


def completeness_score(s: dict) -> float:
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


def slugify(text: str) -> str:
    """Create a filesystem-safe slug."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '_', text)
    text = text.strip('_')
    return text[:40]


# ─── Step 3a: Family clustering ───────────────────────────────────────────────

def cluster_strategies(strategies, embeddings):
    """Cluster using agglomerative clustering, distance threshold 0.35 (sim > 0.65)."""
    from sklearn.cluster import AgglomerativeClustering

    n = len(strategies)
    print(f"Clustering {n} strategies with distance threshold 0.35...")

    # Convert cosine similarity to distance: dist = 1 - sim
    # embeddings are already normalized, so cosine sim = dot product
    sim_matrix = embeddings @ embeddings.T
    dist_matrix = 1.0 - sim_matrix
    dist_matrix = np.clip(dist_matrix, 0, 2)  # numerical safety

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='precomputed',
        linkage='average',
        distance_threshold=0.35,
    )
    labels = clustering.fit_predict(dist_matrix)

    n_clusters = len(set(labels))
    print(f"Found {n_clusters} clusters")
    return labels.tolist()


def derive_family_name(members: list) -> str:
    """Derive a family name from the most common tags and prior_art."""
    tag_counter = Counter()
    for s in members:
        tags = s.get('tags', [])
        if isinstance(tags, list):
            for t in tags:
                if isinstance(t, str) and len(t) > 2:
                    tag_counter[t.lower().strip()] += 1

    prior_art_words = Counter()
    for s in members:
        pa = s.get('prior_art', '')
        if isinstance(pa, str):
            words = re.findall(r'\b[a-z]{3,}\b', pa.lower())
            for w in words:
                if w not in ('the', 'and', 'for', 'are', 'with', 'from', 'this', 'that', 'has', 'was'):
                    prior_art_words[w] += 1

    # Build name from top 3 tags
    top_tags = [t for t, _ in tag_counter.most_common(3)]
    if top_tags:
        return " ".join(top_tags)
    elif prior_art_words:
        top_words = [w for w, _ in prior_art_words.most_common(2)]
        return " ".join(top_words)
    else:
        # Fall back to most common universe + timeframe
        universes = [s.get('universe', '') for s in members]
        most_common_universe = Counter(universes).most_common(1)[0][0] if universes else 'unknown'
        return most_common_universe + " strategy"


# ─── Step 3b: LLM Judge ───────────────────────────────────────────────────────

JUDGE_PROMPT_TEMPLATE = """You are a quant researcher comparing two trading strategies that were flagged as potentially similar.

STRATEGY A:
{json_a}

STRATEGY B:
{json_b}

Compare them on these dimensions:
1. Core thesis — are they exploiting the same inefficiency?
2. Entry logic — same signals or meaningfully different triggers?
3. Exit logic — same stop/target/trailing mechanics?
4. Filters — do they avoid the same or different market regimes?
5. Indicators — overlapping or distinct computation?
6. Risk management — same sizing and limits?

Respond with ONLY this JSON, no markdown fences, no preamble:
{{
    "verdict": "DUPLICATE | VARIANT | DIFFERENT",
    "similarity_pct": <0-100>,
    "shared_family": "<family name>",
    "key_differences": ["<diff 1>", "<diff 2>", ...],
    "merge_possible": true/false,
    "merge_plan": "<if merge_possible: one paragraph on how to combine the best of both, else 'N/A'>",
    "keep_recommendation": "A | B | BOTH | MERGE",
    "reason": "<one sentence justification>"
}}

Definitions:
- DUPLICATE: Same thesis, same entry/exit logic, same filters. Only cosmetic or wording differences. One can be safely deleted.
- VARIANT: Same core idea but meaningfully different in at least one of: parameters, filters, risk management, indicator choice, or side (long vs short). Both add value.
- DIFFERENT: Fundamentally different strategies that happen to share surface features like tags or universe. Both are clearly distinct."""


MERGE_PROMPT_TEMPLATE = """You are a quant researcher creating a merged trading strategy from two complementary strategies.

STRATEGY A:
{json_a}

STRATEGY B:
{json_b}

MERGE RATIONALE: {merge_plan}

Create a single merged strategy JSON that:
- Takes the more detailed thesis (or combines both if they cover different aspects)
- Unions the indicator sets (no duplicates)
- Combines entry conditions: if they cover different sides (long/short) or different regimes, include both as separate condition sets
- Takes the MORE conservative risk parameters (tighter stops, lower max trades, lower capital)
- Unions the filter sets
- Unions the weaknesses lists
- Sets author to "merged:{author_a}+{author_b}"
- Adds a notes field: "Merged from {name_a} and {name_b}. {merge_plan}"

Return ONLY the complete merged strategy JSON. Same schema as the inputs. No markdown fences."""


async def call_llm_judge(client, s_a: dict, s_b: dict, semaphore, retries: int = 1):
    """Call Claude to judge if two strategies are DUPLICATE, VARIANT, or DIFFERENT."""
    # Strip internal metadata fields before sending to LLM
    def clean(s):
        return {k: v for k, v in s.items() if not k.startswith('_')}

    json_a = json.dumps(clean(s_a), indent=2)
    json_b = json.dumps(clean(s_b), indent=2)

    # Truncate if too long (keep within token limits)
    max_chars = 8000
    if len(json_a) > max_chars:
        json_a = json_a[:max_chars] + "\n... [truncated]"
    if len(json_b) > max_chars:
        json_b = json_b[:max_chars] + "\n... [truncated]"

    prompt = JUDGE_PROMPT_TEMPLATE.format(json_a=json_a, json_b=json_b)

    async with semaphore:
        for attempt in range(retries + 1):
            try:
                response = await asyncio.to_thread(
                    lambda: client.messages.create(
                        model="claude-sonnet-4-6",
                        max_tokens=1024,
                        messages=[{"role": "user", "content": prompt}]
                    )
                )
                content = response.content[0].text.strip()
                # Strip markdown fences if present
                content = re.sub(r'^```(?:json)?\s*', '', content)
                content = re.sub(r'\s*```$', '', content)
                result = json.loads(content)
                return result
            except json.JSONDecodeError as e:
                if attempt < retries:
                    await asyncio.sleep(2)
                else:
                    return {'verdict': 'ERROR', 'error': f'JSON parse error: {e}', 'raw': content[:500]}
            except Exception as e:
                if attempt < retries:
                    await asyncio.sleep(5)
                else:
                    return {'verdict': 'ERROR', 'error': str(e)}


async def call_llm_merge(client, s_a: dict, s_b: dict, merge_plan: str, semaphore, retries: int = 1):
    """Call Claude to merge two strategies."""
    def clean(s):
        return {k: v for k, v in s.items() if not k.startswith('_')}

    json_a = json.dumps(clean(s_a), indent=2)
    json_b = json.dumps(clean(s_b), indent=2)

    max_chars = 7000
    if len(json_a) > max_chars:
        json_a = json_a[:max_chars] + "\n... [truncated]"
    if len(json_b) > max_chars:
        json_b = json_b[:max_chars] + "\n... [truncated]"

    prompt = MERGE_PROMPT_TEMPLATE.format(
        json_a=json_a,
        json_b=json_b,
        merge_plan=merge_plan,
        author_a=s_a.get('author', 'unknown'),
        author_b=s_b.get('author', 'unknown'),
        name_a=s_a.get('name', 'strategy_a'),
        name_b=s_b.get('name', 'strategy_b'),
    )

    async with semaphore:
        for attempt in range(retries + 1):
            try:
                response = await asyncio.to_thread(
                    lambda: client.messages.create(
                        model="claude-sonnet-4-6",
                        max_tokens=4096,
                        messages=[{"role": "user", "content": prompt}]
                    )
                )
                content = response.content[0].text.strip()
                content = re.sub(r'^```(?:json)?\s*', '', content)
                content = re.sub(r'\s*```$', '', content)
                result = json.loads(content)
                return result
            except json.JSONDecodeError as e:
                if attempt < retries:
                    await asyncio.sleep(2)
                else:
                    return None
            except Exception as e:
                if attempt < retries:
                    await asyncio.sleep(5)
                else:
                    return None


# ─── Main Phase 3 logic ───────────────────────────────────────────────────────

async def run_phase3_async():
    import anthropic

    print("\n" + "="*60)
    print("PHASE 3: Family clustering + LLM judge + merges")
    print("="*60)

    # Load Phase 2 survivors and embeddings
    with open(OUTPUT_DIR / "phase2_survivors.json", 'r') as f:
        strategies = json.load(f)

    embeddings = np.load(OUTPUT_DIR / "phase2_survivor_embeddings.npy")
    with open(OUTPUT_DIR / "phase2_survivor_filepaths.json", 'r') as f:
        survivor_fps = json.load(f)

    print(f"Loaded {len(strategies)} Phase 2 survivors")
    print(f"Embeddings shape: {embeddings.shape}")

    # Reindex strategies by filepath for quick lookup
    fp_to_strategy = {s.get('_filepath', ''): s for s in strategies}

    # ─── Step 3a: Cluster ─────────────────────────────────────────────────────
    labels = cluster_strategies(strategies, embeddings)

    # Build clusters
    cluster_map = defaultdict(list)
    for i, label in enumerate(labels):
        cluster_map[label].append(i)

    n_families = len(cluster_map)
    n_singletons = sum(1 for members in cluster_map.values() if len(members) == 1)
    n_multi = n_families - n_singletons
    print(f"  Families: {n_families} ({n_singletons} singletons, {n_multi} multi-member)")

    # Assign family names and build cluster records
    clusters = {}
    sim_matrix = embeddings @ embeddings.T  # for pairwise sims

    for label, indices in cluster_map.items():
        members = [strategies[i] for i in indices]
        family_name = derive_family_name(members)
        clusters[label] = {
            'family_name': family_name,
            'family_slug': slugify(family_name),
            'member_indices': indices,
            'member_filepaths': [strategies[i].get('_filepath', '') for i in indices],
            'member_names': [strategies[i].get('name', '') for i in indices],
            'size': len(indices),
        }

    # Save cluster info
    clusters_output = list(clusters.values())
    with open(OUTPUT_DIR / "phase3_clusters.json", 'w') as f:
        json.dump(clusters_output, f, indent=2)
    print(f"Saved: {OUTPUT_DIR}/phase3_clusters.json")

    # ─── Step 3b: Collect within-family pairs for LLM judging ─────────────────
    print("\nCollecting within-family pairs for LLM judgment...")

    all_pairs = []  # (label, i, j, cosine_sim)
    for label, info in clusters.items():
        indices = info['member_indices']
        if len(indices) < 2:
            continue
        # Collect all pairs within cluster, sorted by similarity desc
        pair_sims = []
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                ia, ib = indices[a], indices[b]
                sim = float(sim_matrix[ia, ib])
                pair_sims.append((sim, ia, ib))
        pair_sims.sort(reverse=True)

        # Cap per family: process top N pairs
        # For small families process all; for large ones cap at 30
        max_pairs = min(len(pair_sims), 30)
        for sim, ia, ib in pair_sims[:max_pairs]:
            all_pairs.append((label, ia, ib, sim))

    print(f"Total pairs for LLM judgment: {len(all_pairs)}")

    # Initialize Anthropic client
    client = anthropic.Anthropic()
    semaphore = asyncio.Semaphore(4)  # 4 concurrent calls

    # ─── Step 3b: Run LLM judge ───────────────────────────────────────────────
    print("Running LLM judgment calls...")
    verdict_records = []
    llm_calls_made = 0
    llm_errors = 0

    async def judge_pair(label, ia, ib, sim):
        nonlocal llm_calls_made, llm_errors
        s_a = strategies[ia]
        s_b = strategies[ib]
        verdict = await call_llm_judge(client, s_a, s_b, semaphore)
        llm_calls_made += 1
        if verdict.get('verdict') == 'ERROR':
            llm_errors += 1
        return {
            'cluster_label': int(label),
            'family_name': clusters[label]['family_name'],
            'index_a': int(ia),
            'index_b': int(ib),
            'filepath_a': s_a.get('_filepath', ''),
            'name_a': s_a.get('name', ''),
            'filepath_b': s_b.get('_filepath', ''),
            'name_b': s_b.get('name', ''),
            'cosine_sim': round(sim, 4),
            'verdict': verdict,
        }

    tasks = [judge_pair(label, ia, ib, sim) for label, ia, ib, sim in all_pairs]

    # Process with progress tracking
    verdict_records = []
    batch_size = 20
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        results = await asyncio.gather(*batch)
        verdict_records.extend(results)
        completed = min(i + batch_size, len(tasks))
        print(f"  Judged {completed}/{len(tasks)} pairs "
              f"({llm_errors} errors so far)...")

    print(f"LLM calls made: {llm_calls_made}, errors: {llm_errors}")

    # Save verdicts
    with open(OUTPUT_DIR / "phase3_llm_verdicts.json", 'w') as f:
        json.dump(verdict_records, f, indent=2)
    print(f"Saved: {OUTPUT_DIR}/phase3_llm_verdicts.json")

    # ─── Process verdicts: build kill set ─────────────────────────────────────
    # Track kills from LLM verdicts
    lm_killed = {}  # filepath → kill info
    merge_pairs = []  # pairs to merge

    # Group verdicts by cluster
    cluster_verdicts = defaultdict(list)
    for vr in verdict_records:
        cluster_verdicts[vr['cluster_label']].append(vr)

    for label, verdicts in cluster_verdicts.items():
        for vr in verdicts:
            v = vr.get('verdict', {})
            if isinstance(v, dict):
                verdict_str = v.get('verdict', 'ERROR')
                keep_rec = v.get('keep_recommendation', 'BOTH')
                merge_possible = v.get('merge_possible', False)
                merge_plan = v.get('merge_plan', 'N/A')

                fp_a = vr['filepath_a']
                fp_b = vr['filepath_b']

                if verdict_str == 'DUPLICATE':
                    # Kill one
                    if keep_rec == 'A':
                        if fp_b not in lm_killed:
                            lm_killed[fp_b] = {
                                'reason': 'llm_duplicate',
                                'of': fp_a,
                                'verdict': verdict_str,
                                'similarity_pct': v.get('similarity_pct', 0),
                            }
                    elif keep_rec == 'B':
                        if fp_a not in lm_killed:
                            lm_killed[fp_a] = {
                                'reason': 'llm_duplicate',
                                'of': fp_b,
                                'verdict': verdict_str,
                                'similarity_pct': v.get('similarity_pct', 0),
                            }
                    else:
                        # BOTH or MERGE for a duplicate — pick less complete to kill
                        s_a = strategies[vr['index_a']]
                        s_b = strategies[vr['index_b']]
                        if completeness_score(s_b) <= completeness_score(s_a):
                            if fp_b not in lm_killed:
                                lm_killed[fp_b] = {
                                    'reason': 'llm_duplicate_auto',
                                    'of': fp_a,
                                    'verdict': verdict_str,
                                    'similarity_pct': v.get('similarity_pct', 0),
                                }
                        else:
                            if fp_a not in lm_killed:
                                lm_killed[fp_a] = {
                                    'reason': 'llm_duplicate_auto',
                                    'of': fp_b,
                                    'verdict': verdict_str,
                                    'similarity_pct': v.get('similarity_pct', 0),
                                }

                elif verdict_str == 'VARIANT' and keep_rec == 'MERGE' and merge_possible:
                    # Queue for merging (only if neither already killed or queued for merge)
                    if fp_a not in lm_killed and fp_b not in lm_killed:
                        merge_pairs.append({
                            'index_a': vr['index_a'],
                            'index_b': vr['index_b'],
                            'filepath_a': fp_a,
                            'filepath_b': fp_b,
                            'merge_plan': merge_plan,
                            'family': clusters[label]['family_name'],
                        })

    # Deduplicate merge pairs (same pair might appear multiple times)
    seen_merge_pairs = set()
    unique_merge_pairs = []
    for mp in merge_pairs:
        key = tuple(sorted([mp['filepath_a'], mp['filepath_b']]))
        if key not in seen_merge_pairs:
            seen_merge_pairs.add(key)
            unique_merge_pairs.append(mp)
    merge_pairs = unique_merge_pairs

    print(f"\nLLM verdict summary:")
    verdict_counts = Counter()
    for vr in verdict_records:
        v = vr.get('verdict', {})
        if isinstance(v, dict):
            verdict_counts[v.get('verdict', 'ERROR')] += 1
    for k, count in verdict_counts.most_common():
        print(f"  {k}: {count}")
    print(f"LLM duplicates to kill: {len(lm_killed)}")
    print(f"Pairs queued for merge: {len(merge_pairs)}")

    # ─── Step 3c: Execute merges ───────────────────────────────────────────────
    print(f"\nExecuting {len(merge_pairs)} merges...")
    merges_output = []
    merged_strategies = []
    merge_killed = set()  # filepaths consumed by merges

    async def do_merge(mp):
        s_a = strategies[mp['index_a']]
        s_b = strategies[mp['index_b']]
        result = await call_llm_merge(client, s_a, s_b, mp['merge_plan'], semaphore)
        return mp, result

    merge_tasks = [do_merge(mp) for mp in merge_pairs]
    merge_results = await asyncio.gather(*merge_tasks)

    for mp, merged in merge_results:
        fp_a = mp['filepath_a']
        fp_b = mp['filepath_b']

        merge_record = {
            'source_a': fp_a,
            'source_b': fp_b,
            'name_a': strategies[mp['index_a']].get('name', ''),
            'name_b': strategies[mp['index_b']].get('name', ''),
            'merge_plan': mp['merge_plan'],
            'family': mp['family'],
            'success': merged is not None,
        }

        if merged is not None:
            # Add internal tracking metadata
            merged['_filepath'] = f'merged/{merged.get("name", "merged_strategy")}.json'
            merged['_is_merged'] = True
            merged['_merged_from'] = [fp_a, fp_b]
            merged['_merge_family'] = mp['family']
            merged_strategies.append(merged)
            merge_record['merged_name'] = merged.get('name', '')
            merge_record['merged_result'] = merged

            # Kill originals
            merge_killed.add(fp_a)
            merge_killed.add(fp_b)
        else:
            # Merge failed — keep both originals
            merge_record['merged_result'] = None
            print(f"  Merge failed: {mp['filepath_a']} + {mp['filepath_b']}, keeping both")

        merges_output.append(merge_record)

    print(f"Successful merges: {sum(1 for m in merges_output if m['success'])}")
    print(f"Failed merges: {sum(1 for m in merges_output if not m['success'])}")
    print(f"Original strategies consumed by merges: {len(merge_killed)}")

    # Save merges
    merges_save = []
    for m in merges_output:
        rec = {k: v for k, v in m.items() if k != 'merged_result'}
        if m.get('merged_result'):
            rec['merged_result_name'] = m['merged_result'].get('name', '')
        merges_save.append(rec)
    with open(OUTPUT_DIR / "phase3_merges.json", 'w') as f:
        json.dump(merges_save, f, indent=2)
    print(f"Saved: {OUTPUT_DIR}/phase3_merges.json")

    # ─── Step 3d: Final family cleanup ────────────────────────────────────────
    print("\nStep 3d: Final family cleanup...")

    # Build current survivor set (after LLM kills and merge kills)
    all_killed_fps = set(lm_killed.keys()) | merge_killed
    current_survivors = [s for s in strategies if s.get('_filepath', '') not in all_killed_fps]
    current_survivors += merged_strategies  # add merged ones

    # Re-cluster with updated survivors to identify large families
    # Use the original cluster assignments, update for current survivors
    family_survivors = defaultdict(list)
    for s in current_survivors:
        fp = s.get('_filepath', '')
        # Find which cluster this was in
        found_cluster = None
        for label, info in clusters.items():
            if fp in info['member_filepaths']:
                found_cluster = label
                break
        if found_cluster is not None:
            family_survivors[found_cluster].append(s)
        else:
            # Merged strategies or unclustered — singleton
            family_survivors[f'merged_{fp}'].append(s)

    cleanup_kills = set()
    for label, members in family_survivors.items():
        if len(members) < 5:
            continue  # Only apply cleanup to families with 5+

        # Check if all pairs in this family got VARIANT with sim_pct > 80
        family_verdicts = cluster_verdicts.get(label, [])
        variant_high_sim = [
            vr for vr in family_verdicts
            if isinstance(vr.get('verdict'), dict)
            and vr['verdict'].get('verdict') == 'VARIANT'
            and vr['verdict'].get('similarity_pct', 0) > 80
        ]
        # Only apply if all verdicts for this family were high-sim VARIANTs
        if family_verdicts and len(variant_high_sim) == len(family_verdicts):
            # Keep only top 3 most distinct (lowest pairwise avg sim)
            if len(members) <= 3:
                continue
            fps_in_family = [m.get('_filepath', '') for m in members]
            # Build sub-embedding matrix for these members
            # Get indices in the full survivor embeddings array
            sub_indices = []
            for fp in fps_in_family:
                if fp in survivor_fps:
                    sub_indices.append(survivor_fps.index(fp))

            if len(sub_indices) < 3:
                continue

            sub_embs = embeddings[sub_indices]
            sub_sim = sub_embs @ sub_embs.T

            # Greedy selection of 3 most distinct members
            n_sub = len(sub_indices)
            avg_sims = []
            for i in range(n_sub):
                sims_to_others = [sub_sim[i, j] for j in range(n_sub) if j != i]
                avg_sim = np.mean(sims_to_others) if sims_to_others else 1.0
                avg_sims.append((avg_sim, i))
            avg_sims.sort()  # lowest avg sim = most distinct

            keep_indices = set(idx for _, idx in avg_sims[:3])
            for i in range(n_sub):
                if i not in keep_indices:
                    fp = fps_in_family[sub_indices.index(sub_indices[i]) if sub_indices[i] < len(sub_indices) else i] if i < len(fps_in_family) else ''
                    if fp:
                        cleanup_kills.add(fp)
                        print(f"  Family cleanup: killed {fp} (family={label}, all high-sim VARIANTs)")

    print(f"Family cleanup kills: {len(cleanup_kills)}")

    # ─── Build final survivors ─────────────────────────────────────────────────
    final_killed = all_killed_fps | cleanup_kills
    final_survivors = [s for s in current_survivors if s.get('_filepath', '') not in cleanup_kills]

    # Update cluster info with final family assignments
    final_cluster_info = {}
    for label, info in clusters.items():
        surviving_members = [fp for fp in info['member_filepaths'] if fp not in final_killed]
        # Add merged strategy filepaths if they came from this family
        for ms in merged_strategies:
            from_fps = ms.get('_merged_from', [])
            if any(fp in info['member_filepaths'] for fp in from_fps):
                surviving_members.append(ms.get('_filepath', ''))
        final_cluster_info[label] = {
            **info,
            'final_surviving_members': surviving_members,
            'final_size': len(surviving_members),
        }

    # Save updated clusters
    with open(OUTPUT_DIR / "phase3_clusters.json", 'w') as f:
        json.dump(list(final_cluster_info.values()), f, indent=2)

    # Build kill log for phase3
    phase3_kills = []
    for fp, info in lm_killed.items():
        phase3_kills.append({'filepath': fp, **info})
    for fp in merge_killed:
        phase3_kills.append({'filepath': fp, 'reason': 'consumed_by_merge'})
    for fp in cleanup_kills:
        phase3_kills.append({'filepath': fp, 'reason': 'family_cleanup_variant_excess'})

    # ─── Save final survivors ──────────────────────────────────────────────────
    phase3_survivor_path = OUTPUT_DIR / "phase3_survivors.json"
    with open(phase3_survivor_path, 'w') as f:
        json.dump(final_survivors, f, indent=2)
    print(f"Saved survivors: {phase3_survivor_path}")

    # Save cluster-to-family mapping for Phase 4
    fp_to_family = {}
    for label, info in final_cluster_info.items():
        for fp in info['member_filepaths']:
            fp_to_family[fp] = {
                'family_name': info['family_name'],
                'family_slug': info['family_slug'],
                'cluster_label': int(label),
            }
    # Add merged strategies
    for ms in merged_strategies:
        ms_fp = ms.get('_filepath', '')
        family = ms.get('_merge_family', 'merged')
        fp_to_family[ms_fp] = {
            'family_name': family,
            'family_slug': slugify(family),
            'cluster_label': -1,
        }

    with open(OUTPUT_DIR / "phase3_fp_to_family.json", 'w') as f:
        json.dump(fp_to_family, f, indent=2)

    # Summary stats
    dupe_count = verdict_counts.get('DUPLICATE', 0)
    variant_count = verdict_counts.get('VARIANT', 0)
    different_count = verdict_counts.get('DIFFERENT', 0)
    error_count = verdict_counts.get('ERROR', 0)

    print("\n" + "─"*60)
    print("PHASE 3 SUMMARY")
    print("─"*60)
    print(f"  Families found           : {n_families}")
    print(f"  Multi-member families    : {n_multi}")
    print(f"  LLM calls made           : {llm_calls_made}")
    print(f"  LLM errors               : {llm_errors}")
    print(f"  Verdicts — DUPLICATE     : {dupe_count}")
    print(f"  Verdicts — VARIANT       : {variant_count}")
    print(f"  Verdicts — DIFFERENT     : {different_count}")
    print(f"  Verdicts — ERROR         : {error_count}")
    print(f"  LLM DUPLICATEs killed    : {len(lm_killed)}")
    print(f"  Merges created           : {len([m for m in merges_output if m['success']])}")
    print(f"  Family cleanup kills     : {len(cleanup_kills)}")
    print(f"  SURVIVING after Phase 3  : {len(final_survivors)}")
    print("─"*60)

    return final_survivors, {
        'families': n_families,
        'multi_member_families': n_multi,
        'llm_calls': llm_calls_made,
        'llm_errors': llm_errors,
        'verdict_counts': dict(verdict_counts),
        'lm_duplicates_killed': len(lm_killed),
        'merges_created': len([m for m in merges_output if m['success']]),
        'family_cleanup_kills': len(cleanup_kills),
        'final_survivors': len(final_survivors),
        'phase3_kills': phase3_kills,
        'fp_to_family': fp_to_family,
        'clusters': final_cluster_info,
    }


def run_phase3():
    return asyncio.run(run_phase3_async())


if __name__ == '__main__':
    survivors, info = run_phase3()
