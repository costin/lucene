<!--
  Licensed to the Apache Software Foundation (ASF) under one or more
  contributor license agreements.  See the NOTICE file distributed with
  this work for additional information regarding copyright ownership.
  The ASF licenses this file to You under the Apache License, Version 2.0
  (the "License"); you may not use this file except in compliance with
  the License.  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
-->

## lucene-13b: Deferred applyMask in MaxScoreBulkScorer Multi-Essential Path

**Status**: DEFERRED — fold into TermScorer bulk intoBitSet work (see below)

**Goal**: Replace per-doc `WindowFilterBits.get()` with a single bulk `applyMask()` AND per inner window in the multi-essential clause path.

**Depends on**: lucene-13 (filter bitset materialization, merged)
**Scope**: `MaxScoreBulkScorer.scoreInnerWindowMultipleEssentialClauses` only
**LOC estimate**: ~20

---

## Decision: Not worth shipping standalone

Benchmarked with 75% density gate and with no gate. Results vs lucene-13 baseline:

| filter | 75% gate (cc=12) | 75% gate (cc=45) | no gate (cc=12) | no gate (cc=45) |
|-------:|-----------------:|-----------------:|----------------:|----------------:|
| 0.05 | -1% | +1% | **-26%** | **-26%** |
| 0.1 | +0% | +21% (noise) | **-22%** | **-25%** |
| 0.2 | +2% | +10% (noise) | **-20%** | **-7%** |
| 0.5 | +1% | -1% | -3% | **-11%** |
| 0.8 | -5% | -5% | -4% | +1% |
| 0.95 | +1% | **+12%** | +2% | **+10%** |

**Findings**:
- Only sel=0.95 shows consistent improvement (+10-12%)
- sel=0.8 regresses -5% even with the 75% gate
- Without the gate, regressions are 20-27% at sel ≤ 0.2
- The cc=45/sel=0.1/0.2 gated results (+21%, +10%) are noise — deferred doesn't activate below 75%

**Root cause**: passing `null` as `acceptDocs` to `nextDocsAndScores` scores ALL docs then discards filtered ones. The wasted `scorer.score()` cost at medium densities overwhelms the `applyMask()` savings. Only at ≥95% density is the waste negligible enough for bulk AND to win.

**Path forward**: this optimization becomes viable when `TermScorer.nextDocsAndScores` gets a bulk `intoBitSet` mode that collects doc IDs into a bitset without scoring. Then the deferred path collects docs (no score waste) and applies `applyMask` in bulk. That eliminates the fundamental tradeoff — no wasted scores at any density.

---

## Context

lucene-13 materializes the filter into a per-window `FixedBitSet` via `intoBitSet()`, wraps it as `WindowFilterBits`, then delegates to the non-filter scoring paths. Those paths call `TermScorer.nextDocsAndScores(upTo, acceptDocs, buffer)` which applies the filter per-doc via `DocAndFloatFeatureBuffer.apply(Bits)` → `WindowFilterBits.get()`.

`WindowFilterBits.applyMask()` is implemented (bulk `long[] AND`) but never called from the current code path. This plan wires it up for the multi-essential clause path specifically.

### Current performance (lucene-13)

The per-doc `get()` approach already delivers strong wins by eliminating the zigzag join:

| scenario | AMD | AVX-512 |
|----------|-----|---------|
| 12 clauses, sel=0.2 | 1.44x | 1.45x |
| 45 clauses, sel=0.2 | 4.40x | 4.86x |
| 45 clauses, sel=0.8 | 3.58x | 3.93x |

This plan targets incremental improvement on top of these numbers — estimated 5-15% additional gain on the multi-essential path.

### Why multi-essential path only

The three non-filter scoring paths in `MaxScoreBulkScorer`:

1. **`scoreInnerWindowSingleEssentialClause`**: iterates one scorer's docs in batches of 64 via `nextDocsAndScores`. No window-level bitset to `applyMask` against — docs go directly to `scoreNonEssentialClauses`. Not a candidate for bulk AND.

2. **`scoreInnerWindowMultipleEssentialClauses`**: collects ALL essential clause docs into `windowMatches` (a `FixedBitSet`), accumulates scores in `windowScores[]`, then iterates set bits. This path **already has a bitset** (`windowMatches`) that can be AND-ed with the filter bitset in one operation.

3. **`scoreInnerWindowSingleEssentialClause` (half-window variant)**: same as #1.

Only path #2 has the right structure for deferred `applyMask`.

---

## Implementation

### Current code (scoreInnerWindowMultipleEssentialClauses)

```java
do {
    for (top.scorer.nextDocsAndScores(innerWindowMax, acceptDocs, docAndScoreBuffer); ...) {
        // acceptDocs filters per-doc inside nextDocsAndScores → buffer.apply(liveDocs)
        for (int index = 0; index < docAndScoreBuffer.size; ++index) {
            windowMatches.set(i);
            windowScores[i] += score;
        }
    }
    top = essentialQueue.updateTop();
} while (top.doc < innerWindowMax);

// iterate windowMatches → scoreNonEssentialClauses
```

### Proposed change

```java
// Pass null instead of acceptDocs — collect ALL docs, skip per-doc filter
do {
    for (top.scorer.nextDocsAndScores(innerWindowMax, null, docAndScoreBuffer); ...) {
        for (int index = 0; index < docAndScoreBuffer.size; ++index) {
            windowMatches.set(i);
            windowScores[i] += score;
        }
    }
    top = essentialQueue.updateTop();
} while (top.doc < innerWindowMax);

// One bulk AND — filter out non-matching docs
if (acceptDocs != null) {
    acceptDocs.applyMask(windowMatches, innerWindowMin);
}
// Zero out scores for cleared bits
windowMatches.forEach(0, innerWindowSize, 0, index -> { /* keep */ });
// OR: iterate windowMatches normally — cleared bits are already excluded
```

### Score cleanup concern

After `applyMask` clears bits in `windowMatches`, the corresponding `windowScores[i]` entries have stale values. Two options:

**Option A**: Don't clean up scores. The `windowMatches.forEach` loop only visits set bits, so stale `windowScores` entries are never read. They get zeroed in the existing cleanup: `windowScores[index] = 0d`. This works because the loop already zeros every visited entry — unvisited entries were either never set (zero from array init) or zeroed in a previous window.

Wait — entries set in this window but cleared by `applyMask` will NOT be visited and NOT be zeroed. They'll carry stale values into the next window. **This is a bug if the same window offset is reused.**

`windowScores` is `INNER_WINDOW_SIZE` (4096) entries. Each window uses offsets `[0, innerWindowSize)`. If the next window starts at a different `innerWindowMin`, the same physical indices are reused with different doc-to-index mappings. Stale scores from a previous window at the same index would corrupt results.

**Option B**: After `applyMask`, iterate the bits that were cleared and zero their scores:

```java
if (acceptDocs != null) {
    // Save a copy before masking
    FixedBitSet beforeMask = windowMatches.clone(); // or track cleared bits
    acceptDocs.applyMask(windowMatches, innerWindowMin);
    // Zero scores for cleared bits
    beforeMask.andNot(windowMatches);
    beforeMask.forEach(0, innerWindowSize, 0, index -> {
        windowScores[index] = 0d;
    });
}
```

This adds allocation (clone) and iteration overhead. Defeats some of the purpose.

**Option C (recommended)**: Zero ALL `windowScores` entries for the window at the start, not just visited ones at the end:

```java
Arrays.fill(windowScores, 0, innerWindowSize, 0d);
```

Then stale entries are always zero. The current code avoids this fill by zeroing only visited entries — but `Arrays.fill` on 4096 doubles (~32KB) is a single `memset` that the JIT will SIMD-ify. Cost: ~50ns. Negligible compared to the scoring work.

---

## Pros

1. **Replaces N per-doc `WindowFilterBits.get()` calls with one bulk AND** (~64 docs per `long` operation). For a 4096-doc window with 80% filter selectivity, that's ~3277 `get()` calls → ~64 `long AND` operations.

2. **Enables future `TermScorer` optimization**: if `TermScorer.nextDocsAndScores` later learns to skip the `buffer.apply(liveDocs)` when `liveDocs` is null, the batch decode becomes even faster (no filtering overhead at all inside the scorer).

3. **Small, isolated change**: only touches `scoreInnerWindowMultipleEssentialClauses`. No API changes, no new abstractions.

## Cons

1. **Scores non-matching docs then discards**: `scorer.score()` is called for docs that fail the filter. For sel=0.2, ~80% of window docs pass the filter, so ~20% of score computations are wasted. For sel=0.05 (sparse filter), ~95% are wasted — but sparse filters don't hit this path (below `maxDoc >>> 5` threshold).

2. **Score cleanup complexity**: need to handle stale `windowScores` entries (see Options A-C above). Option C (`Arrays.fill`) is simplest but changes the existing zeroing pattern.

3. **Marginal gain on top of 3-5x wins**: lucene-13 already delivers massive improvements. The additional gain from bulk AND is estimated at 5-15% — real but not transformative. Could be noise in some configurations.

4. **Only helps multi-essential path**: single-essential clause path (common for few-clause queries after competitive scoring kicks in) is unaffected. The biggest wins from lucene-13 are already in the multi-essential path.

---

## When this becomes more valuable

- **If `TermScorer.nextDocsAndScores` gets an `intoBitSet` mode**: instead of per-doc iteration + `buffer.apply(liveDocs)`, the scorer could bulk-decode into a bitset directly. Then both essential clause collection AND filter application use bulk operations. This is the "Option C" from the original analysis — broadest scope, separate PR, benefits all Bits-based filtering.

- **If filter selectivities in production are commonly 10-30%**: then the wasted-score overhead at these selectivities is small (70-90% of scored docs pass), and the bulk AND saves the most relative to per-doc `get()`.

---

## Benchmark plan

1. Implement Option C (Arrays.fill + null acceptDocs + deferred applyMask)
2. Run A/B against lucene-13 (not main — measure the incremental gain)
3. Focus on cc=45 where multi-essential path dominates
4. Key selectivities: 0.05, 0.2, 0.5, 0.8 (all materialized, varying filter density)

Expected: 5-15% improvement at cc=45, neutral at cc=12 (single-essential path dominates after warmup).

---

## Files to modify

| File | Change |
|------|--------|
| `MaxScoreBulkScorer.java` | `scoreInnerWindowMultipleEssentialClauses`: pass null to nextDocsAndScores, add applyMask + Arrays.fill |

No new files. No API changes.
