# Deduplication Pipeline

## Summary
A 3-phase approach for deduplicating 750+ AI-generated trading strategies: structural fingerprinting (free, instant) → embedding similarity (cheap, fast) → LLM judging (targeted, expensive). The key insight: cheap methods eliminate the obvious duplicates so the expensive LLM judge only reviews ambiguous cases.

## Architecture

### Phase 1: Structural Fingerprinting (No ML)
For each strategy JSON, compute:
- **Tag set**: lowercased, deduplicated → `frozenset`
- **Indicator names**: extracted from the indicators list
- **Entry hash**: MD5 of normalized entry condition text
- **Exit hash**: MD5 of normalized exit rule text
- **Mechanics hash**: Combined hash of entry + exit + filters + indicator names
- **Thesis hash**: Hash of thesis text + prior_art

**Grouping**: Strategies with identical mechanics hash = structurally identical (same logic, different wording). Within each group, keep the most complete version (fewest empty fields, most detailed thesis, most weaknesses listed).

**Result**: In the actual project, Phase 1 killed 150 of 715 strategies.

### Phase 2: Embedding Similarity
- Model: `all-MiniLM-L6-v2` (sentence-transformers)
- Build text representation per strategy: concatenate name, thesis, tags, prior_art, indicators, entry, exit, filters, weaknesses
- Compute pairwise cosine similarity (upper triangle only — don't store the mirror)
- Near-duplicate threshold: cosine > 0.90

For each near-duplicate pair, structural cross-checks:
- Same universe and timeframe?
- Overlapping indicators (Jaccard similarity)?
- 50%+ tag overlap?
- Entry conditions logically equivalent?

If 3+ cross-checks pass AND cosine > 0.90 → mark the less-complete one as duplicate.

**Result**: Phase 2 reduced 565 → 361 strategies.

### Phase 3: LLM Judge (Targeted)
Only for **within-family pairs** after clustering. Do NOT run on all pairs.

Clustering: Agglomerative clustering with distance threshold 0.35 (similarity > 0.65 to be in same cluster).

For each within-family pair, LLM prompt forces a structured verdict:
```json
{
    "verdict": "DUPLICATE | VARIANT | DIFFERENT",
    "similarity_pct": 0-100,
    "key_differences": ["diff 1", "diff 2"],
    "merge_possible": true/false,
    "merge_plan": "how to combine best of both",
    "keep_recommendation": "A | B | BOTH | MERGE",
    "reason": "one sentence"
}
```

**Definitions**:
- DUPLICATE: Same thesis + same logic + same filters. Cosmetic differences only.
- VARIANT: Same core idea but meaningfully different parameters, filters, or risk management.
- DIFFERENT: Fundamentally different strategies sharing surface features.

For MERGE verdicts, a second LLM call produces the merged strategy JSON.

**Family size cap**: Families with 5+ surviving members after LLM pass → keep only the 3 most distinct.

## Decisions Made
- Store only upper triangle of similarity matrix (N×(N-1)/2 pairs, not N²)
- Phase order is load-bearing: structural hashing is free and catches the obvious clones before paying for embeddings
- Bias toward keeping: false negatives (killing unique) worse than false positives (keeping near-dupes)
- Dedup metadata (`_dedup_metadata`) attached to every surviving strategy for full provenance

## Pitfalls & Anti-patterns
- **Don't use the Anthropic API for all pairs**: 100K+ pair comparisons is unnecessary. Embeddings handle 95% of the work. LLM judge only for the ambiguous within-family pairs.
- **Key by file path, not name**: Strategy names repeat across files. Every record uses file_path + name as its unique identifier.
- **Claude Code tried to call the API from a Python script**: On Max plan, Claude Code's own reasoning uses the subscription, but a Python `anthropic.Client()` call needs a separate API key. The solution: have Claude Code do the judging directly in its own reasoning, not via a separate script.

## Corrections Log
- ~~Original plan stored full 511,225-row flattened table~~ → Store only upper triangle (255,255 pairs), discard diagonal and mirror
- ~~LLM judge was planned for "important subsets" (vague)~~ → Precisely defined: within-family pairs only, structured verdict schema, merge protocol
- ~~API key issue caused Phase 3 to fail silently~~ → Fix: Claude Code IS the LLM judge (no external API call needed on Max plan)
