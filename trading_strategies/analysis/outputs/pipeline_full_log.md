# Strategy Deduplication Pipeline — Full Session Log

## Overview

Complete 5-phase deduplication of ~750 JSON trading strategy files for Indian equity intraday strategies.

---

## Phase 1: Structural Fingerprinting

**Method:** MD5 hash of normalized JSON content (keys sorted, whitespace stripped).

**Result:**
- Input: 715 strategy files
- Exact duplicates killed: 150
- Survivors: 565 (`outputs/phase1_survivors.json`)

**Output:** `outputs/phase1_structural_dupes.json`

---

## Phase 2: Semantic Embedding Deduplication

**Method:** sentence-transformers embeddings on concatenated strategy text fields (thesis + tags + indicators + entry/exit). Cosine similarity > 0.9 = near-duplicate → kill one.

**Result:**
- Input: 565
- Semantic duplicates killed: 204
- Survivors: 361 (`outputs/phase2_survivors.json`)
- Also produced: 68 strategy families via agglomerative clustering (distance_threshold=0.35), 424 within-family pairs for Phase 3 review

**Outputs:**
- `outputs/phase2_embeddings.npy` — embeddings matrix
- `outputs/phase2_filepaths.json` — filepath index
- `outputs/phase2_semantic_dupes.json` — 204 killed pairs with cosines
- `outputs/phase2_survivors.json` — 361 full strategy objects
- `outputs/phase3_clusters.json` — 68 cluster family assignments
- `outputs/phase3_pair_list.json` — 424 within-family pairs
- `outputs/family_*.json` — 68 family detail files (members + pairs with cosines)

---

## Phase 3: LLM Judge (Claude Sonnet 4.6 inline)

**Method:** For each of 424 within-family pairs (cosine 0.65–0.999), read both strategy JSONs and produce a verdict:
- **DUPLICATE** — effectively identical mechanics → kill one
- **VARIANT** — same thesis, different universe/timeframe/threshold → keep both (each covers a distinct market segment)
- **DIFFERENT** — distinct edge hypothesis, different indicator family, or opposite direction → keep both

**Batches written:**
| File | Families | Pairs | VARIANT | DIFFERENT | DUPLICATE |
|------|----------|-------|---------|-----------|-----------|
| `verdicts_batch1.json` | 6 large (≥9 members) | 180 | 140 | 40 | 0 |
| `verdicts_batch2.json` | 7 medium (5-8 members) | 128 | 89 | 39 | 0 |
| `verdicts_batch3.json` | 10 medium (4 members) | 45 | 35 | 10 | 0 |
| `verdicts_batch4.json` | small 3-member families | 43 | ~30 | ~13 | 0 |
| `verdicts_batch5.json` | 2-member families | 28 | 21 | 5 | 2 |
| **Total** | **68 families** | **424** | **289** | **132** | **3** |

**3 DUPLICATE kills:**
1. `all_trading_strategies/opus/Opus_42_of_50.json` — ADX regime switch, duplicate of `cursor_opus46max_strategy_152.json`
2. `all_trading_strategies/opus/Opus_21_of_50.json` — NSE-BSE cross-listed arb, duplicate of `cursor_opus46max_strategy_060.json`
3. `all_trading_strategies/grok/Grok_8_of_10.json` — NR7 breakout, duplicate of `gemini_14_of_20.json`

**Key analytical findings:**

- **Gap strategies** split cleanly: gap-fade vs gap-continuation = DIFFERENT (opposite direction on same trigger). Within each camp = VARIANT.
- **VWAP mean reversion** (30 pairs, all VARIANT): cursor_gemini31pro generated a parameterized sweep across 9 universe/TF/threshold combos.
- **EMA trend** contains Opus_5 (EMA *fade* = reversion, opposite direction to the rest) → correctly DIFFERENT.
- **PDH/PDL** splits: fakeout/reversal vs breakout continuation = DIFFERENT. Within each direction = VARIANT.
- **Relative strength**: cursor_gemini31pro (price rank vs benchmark) vs cursor_gpt54 (VWAP/ATR/base levels) = DIFFERENT despite same broad thesis.
- **Pairs trading**: all distinct implementations (different stock pairs, different leg structure, single-leg vs two-leg) → all DIFFERENT from each other.
- **Very low Phase 3 duplication rate** (3/361 = 0.8%): Phase 2 already eliminated near-identical strategies. Remaining within-family pairs at 0.70–0.99 cosine are mostly genuine variants or genuinely distinct strategies.

**Output:** `outputs/phase3_llm_verdicts.json` (424 verdicts)

---

## Phase 4: Build unique_strategies/ Folder

**Method:** Take 361 Phase 2 survivors, remove 3 Phase 3 kills = **358 unique strategies**. Each file augmented with `_dedup_metadata`:
- `original_path` — source file in `all_trading_strategies/`
- `family` — cluster family assignment from Phase 2/3
- `variant_pairs` — list of strategies that are VARIANT to this one
- `different_pairs` — list of strategies that are DIFFERENT from this one

**Verification:** MD5 hashes of all 358 files confirmed zero content duplicates.

**Output:** `unique_strategies/` — 358 files

---

## Phase 5: Final Report

| Metric | Value |
|--------|-------|
| Raw input | 715 |
| Phase 1 kills (exact) | 150 |
| Phase 2 kills (semantic, cosine > 0.9) | 204 |
| Phase 3 kills (LLM judge) | 3 |
| **Unique survivors** | **358** |
| **Total reduction** | **357 strategies (49.9%)** |

**Outputs:**
- `outputs/dedup_summary.json` — machine-readable summary
- `outputs/dedup_report.md` — formatted report
- `outputs/pipeline_full_log.md` — this file

---

## File Index

| File | Description |
|------|-------------|
| `outputs/phase1_structural_dupes.json` | Exact duplicate pairs from Phase 1 |
| `outputs/phase1_survivors.json` | 565 Phase 1 survivors |
| `outputs/phase2_embeddings.npy` | Sentence-transformer embeddings |
| `outputs/phase2_semantic_dupes.json` | 204 semantic duplicate pairs |
| `outputs/phase2_survivors.json` | 361 Phase 2 survivors (full JSON objects) |
| `outputs/phase3_clusters.json` | 68 cluster family assignments |
| `outputs/phase3_pair_list.json` | 424 within-family pairs |
| `outputs/family_*.json` | 68 family detail files |
| `outputs/verdicts_batch1-5.json` | Raw verdict batches |
| `outputs/phase3_llm_verdicts.json` | Consolidated 424 verdicts |
| `outputs/phase3_survivors.json` | Survivor manifest |
| `outputs/dedup_summary.json` | Pipeline summary stats |
| `outputs/dedup_report.md` | Formatted report |
| `unique_strategies/` | **358 unique strategies** (final output) |
