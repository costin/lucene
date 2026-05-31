## lucene-13: MaxScoreBulkScorer Filter Bitset Integration

**Goal**: Eliminate 50-60% overhead of filtered disjunctions by pre-loading FILTER clause into FixedBitSet, enabling the optimized non-filter scoring paths.

**Motivation**: Benchmark data shows adding a FILTER clause flips Lucene from winning (0.94x) to losing (1.35x) vs TurboPuffer. The per-doc `Bits.get()` + two-phase matching in `scoreInnerWindowWithFilter()` adds ~50-60% overhead. TurboPuffer handles filters with ~10% overhead. jpountz left a TODO at line 170-171.

**Branch**: `lucene/maxscore-filter-bitset` (to create)
**Depends on**: None (standalone)
**LOC estimate**: ~150

---

## Stage 1: Pre-load Filter Into Segment-Wide FixedBitSet
**Goal**: Load filter into bitset at `score()` entry, pass as `acceptDocs` to non-filter paths
**Success Criteria**: Filtered disjunctions use `scoreInnerWindowSingleEssentialClause` / `scoreInnerWindowMultipleEssentialClauses` instead of `scoreInnerWindowWithFilter`
**Tests**: Existing `testFilteredDisjunction()` and `testFilteredDisjunctionWithSkipping()` must still pass

### Changes

**MaxScoreBulkScorer.java** — `score()` method (line 85):
1. Before the main loop, if `filter != null`:
   - Evaluate heuristic: `filter.cost < maxDoc / 8` (sparse) or always for no-two-phase filters
   - Create `FixedBitSet filterBits = new FixedBitSet(maxDoc)`
   - Iterate filter: `filter.iterator.intoBitSet(NO_MORE_DOCS, filterBits, 0)`
   - If filter has twoPhaseView: verify each set bit with `matches()`, clear non-matches
   - Combine with `acceptDocs`: if both non-null, AND them; if only filter, use filter bitset
   - Set `filter = null` so `scoreInnerWindow()` dispatches to optimized paths
   - Pass combined bitset as `acceptDocs` to existing paths

2. Alternative (per-window, lower memory): materialize in `scoreInnerWindow()` dispatch
   - Uses only INNER_WINDOW_SIZE bits (512 bytes) instead of maxDoc/8
   - Requires intoBitSet per window — amortizes well for selective filters

### Decision: Segment-wide vs per-window
- Segment-wide: simpler, one-time cost, maxDoc/8 bytes (625KB at 5M). Better for sparse filters.
- Per-window: lower memory, repeated intoBitSet calls. Better if filter is dense (many matches).
- **Start with segment-wide for simplicity.** Can refine with per-window later.

### Heuristic at score() entry:
```java
if (filter != null && shouldMaterializeFilter(filter, maxDoc)) {
  FixedBitSet filterBits = materializeFilter(filter, maxDoc, acceptDocs);
  // Now route to non-filter paths with filterBits as acceptDocs
  acceptDocs = filterBits;
  filter = null; // clear filter — no longer needed
}
```

### Key code locations:
- `MaxScoreBulkScorer.java:56-73` — constructor, filter field
- `MaxScoreBulkScorer.java:85` — score() entry
- `MaxScoreBulkScorer.java:148-165` — scoreInnerWindow() dispatch (filter vs no-filter)
- `MaxScoreBulkScorer.java:167-222` — scoreInnerWindowWithFilter() (bypass target)
- `MaxScoreBulkScorer.java:224-240` — scoreInnerWindowSingleEssentialClause() (use with filterBits)
- `MaxScoreBulkScorer.java:242-282` — scoreInnerWindowMultipleEssentialClauses() (use with filterBits)
- `BooleanScorerSupplier.java:313-351` — filteredOptionalBulkScorer() creates MaxScoreBulkScorer with filter

### Pattern to follow:
- `ConstantScoreBulkScorer.java:113-114` — `iterator.intoBitSet(windowMax, windowMatches, windowBase)` + `acceptDocs.applyMask()`
- `DenseConjunctionBulkScorer.java:233-252` — AND-ing clause bitsets

**Status**: Not Started

---

## Stage 2: Add Tests for Bitset Filter Path
**Goal**: Verify correctness of filter bitset materialization across edge cases
**Success Criteria**: All new tests pass; randomized testing finds no discrepancies
**Tests**:
- `testFilteredDisjunctionBitsetPath()` — Force bitset path, verify same results
- `testFilteredDisjunctionWithTwoPhase()` — Two-phase filter (e.g., RandomApproximationQuery)
- `testFilteredDisjunctionWithAcceptDocsAndFilter()` — Both acceptDocs + filter bitset
- `testFilteredDisjunctionSparseVsDense()` — Sparse filter (should use bitset) vs dense
- Extend `TestMaxScoreBulkScorer.java` (existing tests at lines 139-280)

### Side-effects to verify:
- `acceptDocs` correctness when combined with filter bitset (AND semantics)
- Filter position state after `intoBitSet()` — iterator must be exhausted or past window
- Two-phase iterator state — `matches()` must be called at correct position
- `minCompetitiveScore` feedback loop still functions (TopScoreDocCollector → partitionScorers)
- No double-counting of docs when filter and acceptDocs overlap

**Status**: Not Started

---

## Stage 3: Benchmark Validation
**Goal**: Measure improvement on filtered disjunctions
**Success Criteria**: 15-30% latency reduction on filtered queries; no regression on unfiltered
**Tests**: JMH micro-benchmark comparing before/after

### Benchmark scenarios:
- 12-term disjunction + 80% selectivity filter (Regime 2)
- 45-term disjunction + 20% selectivity filter (Regime 3)
- 4-term disjunction + 5% selectivity filter (Regime 1, check no regression)
- Unfiltered baselines (verify no regression)
- With/without two-phase filter

### Expected results:
- Filtered: 20-35% improvement (closing gap with TurboPuffer)
- Unfiltered: no change (bitset path not triggered)

**Status**: Not Started

---

## Benchmark command:
```bash
./gradlew -p lucene/benchmark-jmh jmh -Pjmh.includes="MaxScoreBulkScorerBenchmark" -Pjmh.prof="perfnorm"
```

## PR checklist:
- [ ] `./gradlew check` passes
- [ ] Existing TestMaxScoreBulkScorer tests pass
- [ ] New tests for bitset filter path
- [ ] JMH benchmark shows improvement
- [ ] No regression on unfiltered queries
- [ ] PR title: "Optimize filtered disjunction scoring via filter bitset pre-loading"
- [ ] Reference jpountz TODO and related benchmark data
