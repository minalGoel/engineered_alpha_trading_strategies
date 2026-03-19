# Strategy Deduplication Report

Generated: 2026-03-19 19:29

## Executive Summary

| Phase | Action | Before | After | Killed |
|-------|--------|--------|-------|--------|
| Phase 1 | Structural hash exact dedup | 715 | 565 | 150 |
| Phase 2 | Semantic embedding dedup (cosine > 0.9) | 565 | 361 | 204 |
| Phase 3 | LLM judge within-family pairs | 361 | 358 | 3 |
| **Total** | | **715** | **358** | **357 (49.9%)** |

## Phase 3 Verdict Breakdown (424 within-family pairs)

| Verdict | Count | Meaning |
|---------|-------|---------|
| VARIANT | 289 | Same thesis, different params — keep both |
| DIFFERENT | 132 | Distinct edge — keep both |
| DUPLICATE | 3 | Identical mechanics — kill one |

## Phase 3 Eliminated Files

- 
- 
- 

## Key Insight

Phase 3 LLM judging found very few true duplicates (3/361 = 0.8%) among strategies that survived Phases 1 & 2. This is expected: Phase 2 already eliminated near-identical strategies at cosine > 0.9. The remaining within-family pairs at 0.70-0.99 cosine are mostly VARIANT (same thesis, different universe/timeframe/threshold) or DIFFERENT (same category, opposite direction or different indicator family).
