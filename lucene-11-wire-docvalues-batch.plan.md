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

## lucene-11: Wire Doc-Values Queries to Batch Pipeline

**Goal**: Route all doc-values range query types through the batch bulk scorer pipeline (`DocValuesRangeBulkScorer`), closing the wiring gap where some query types fall through to per-doc iteration.

**Motivation**: `SortedNumericDocValuesRangeQuery` already wires singleton (NumericDocValues) fields to `BatchDocValuesRangeIterator` → `DocValuesRangeBulkScorer` via `ConstantScoreScorerSupplier.bulkScorer()`. Multi-valued fields use `BatchSortedNumericDocValuesRangeIterator` but this iterator lacks `bulkScorer()` — it only gets the `DenseConjunctionBulkScorer` path via `intoBitSet()`. The dedicated `DocValuesRangeBulkScorer` is faster for standalone range queries because it combines skip-index navigation with bulk rangeIntoBitSet evaluation.

**Branch**: `lucene/wire-docvalues-batch` (to create)
**Depends on**: lucene-08 (BatchSortedNumericDocValuesRangeIterator), lucene-10 (rangeIntoBitSet overrides for full benefit)
**LOC estimate**: ~200

---

## Stage 1: Add bulkScorer() to BatchSortedNumericDocValuesRangeIterator
**Goal**: Multi-valued range queries get DocValuesRangeBulkScorer path
**Success Criteria**: Multi-valued SortedNumericDocValues range queries with skip index use DocValuesRangeBulkScorer
**Tests**: Existing TestSortedNumericDocValuesRangeQuery must pass; add multi-valued specific benchmark

### Current wiring gap:

`BatchDocValuesRangeIterator` (singleton):
```java
BulkScorer bulkScorer(float score) {
  return new DocValuesRangeBulkScorer(score, blockIterator, values, minValue, maxValue);
}
```

`BatchSortedNumericDocValuesRangeIterator` (multi-valued):
- **No bulkScorer() method** — falls through to ConstantScoreBulkScorer or DenseConjunctionBulkScorer

### Fix:
1. Add `bulkScorer(float score)` to `BatchSortedNumericDocValuesRangeIterator`
2. Create `DocValuesRangeBulkScorer` variant (or generalize existing) that accepts `SortedNumericDocValues`
3. Route in `ConstantScoreScorerSupplier.bulkScorer()`:

```java
// Current (line 82):
if (scoreMode.needsScores() == false
    && iterator instanceof BatchDocValuesRangeIterator batchIterator) {
  return batchIterator.bulkScorer(score);
}

// Add:
if (scoreMode.needsScores() == false
    && iterator instanceof BatchSortedNumericDocValuesRangeIterator batchIterator) {
  return batchIterator.bulkScorer(score);
}
```

### DocValuesRangeBulkScorer generalization:
Currently uses `NumericDocValues.rangeIntoBitSet()`. Need to also accept `SortedNumericDocValues.rangeIntoBitSet()`.

Options:
a. **Two constructors** — one for NumericDocValues, one for SortedNumericDocValues
b. **Functional interface** — pass `rangeIntoBitSet` as a lambda/method reference
c. **Common interface** — extract `RangeIntoBitSetCapable` interface

Option (a) is simplest and matches Lucene style. Keep both paths explicit.

### Code locations:
- `BatchSortedNumericDocValuesRangeIterator.java` — add bulkScorer() method
- `DocValuesRangeBulkScorer.java` — add SortedNumericDocValues constructor/path
- `ConstantScoreScorerSupplier.java:78-84` — add multi-valued routing

### Side-effects to verify:
- Skip index navigation works identically for both value types
- rangeIntoBitSet semantics match (multi-valued: any value in range → set bit)
- Score is constant (1.0) — same as singleton path
- Interaction with DenseConjunctionBulkScorer — when used in conjunction, the DISI path should still work

**Status**: Not Started

---

## Stage 2: IndexSortSortedNumericDocValuesRangeQuery Batch Wiring
**Goal**: When index sort optimization isn't applicable, fall through to batch path
**Success Criteria**: Non-index-sorted segments use batch pipeline for range queries via IntField/LongField
**Tests**: Add test with non-sorted index using IntField/LongField range queries

### Current state (IndexSortSortedNumericDocValuesRangeQuery.java:155-170):
When `getDocIdSetIteratorOrNull()` returns null (no index sort), falls back to `fallbackWeight.scorerSupplier(context)`. The fallback is `SortedNumericDocValuesRangeQuery` which already has batch wiring. **This path already works correctly.**

When index sort IS used (line 157-169): Returns raw `ScorerSupplier` without `ConstantScoreScorerSupplier` — no bulk scorer routing. But index-sorted queries use `SortedSkipperScorerSupplier` which has its own optimized path. **No change needed here.**

### Action: Verify and document — no code change needed for this query type.

**Status**: Not Started (verification only)

---

## Stage 3: SortedSetDocValuesRangeQuery Assessment
**Goal**: Evaluate if batch path is feasible for ordinal-based range queries
**Success Criteria**: Decision documented with rationale

### Current state (SortedSetDocValuesRangeQuery.java:154-159):
Uses `DocValuesRangeIterator.forOrdinalRange()` → `TwoPhaseIterator.asDocIdSetIterator()`. No batch evaluation.

### Analysis:
- SortedSetDocValues doesn't have `rangeIntoBitSet()` — ordinal ranges are different from value ranges
- Would need `ordinalRangeIntoBitSet(fromDoc, toDoc, minOrd, maxOrd, bitSet, offset)` on SortedSetDocValues
- Lucene90DocValuesProducer's SortedSet implementation uses `TermsEnum` + ordinal lookup — more complex than numeric
- **Recommendation**: Out of scope for lucene-11. If needed, create lucene-15 for ordinal batch evaluation.

**Status**: Not Started (assessment only)

---

## Stage 4: Tests and Benchmark
**Goal**: Verify correctness and measure improvement for multi-valued fields
**Success Criteria**: Multi-valued range queries 30-50% faster (matching singleton improvement)
**Tests**:
- Multi-valued field with 2-4 values per doc, various selectivities
- Dense vs sparse multi-valued fields
- Conjunction of multi-valued range query + term query
- Compare results vs per-doc baseline

### Benchmark:
```bash
./gradlew -p lucene/benchmark-jmh jmh \
  -Pjmh.includes="SparseDocValuesRangeQueryBenchmark" \
  -Pjmh.params="fieldType=SORTED_NUMERIC_MULTI" \
  -Pjmh.prof="perfnorm"
```

### Expected improvement:
- Multi-valued with skip index: 30-50% faster (matching singleton perf)
- Without skip index: no change (BatchSortedNumericDocValuesRangeIterator requires skipper)

**Status**: Not Started

---

## Dependency chain:
```
lucene-03 (MERGED) → lucene-08 (WIP) → lucene-11 (this)
                                     ↗
lucene-10 (rangeIntoBitSet overrides)
```

lucene-11 can start once lucene-08 is merged. lucene-10 improves rangeIntoBitSet quality but isn't blocking — the base class default works correctly, just slower.

## PR checklist:
- [ ] `./gradlew check` passes
- [ ] TestSortedNumericDocValuesRangeQuery passes with multi-valued fields
- [ ] New tests for multi-valued bulk scorer routing
- [ ] Benchmark shows multi-valued matches singleton perf
- [ ] No regression on singleton fields
- [ ] PR title: "Route multi-valued range queries through batch bulk scorer"
- [ ] Reference lucene-08 PR for context
- [ ] Keep scope to wiring — no rangeIntoBitSet changes (that's lucene-10)
