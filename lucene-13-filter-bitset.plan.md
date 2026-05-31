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

## lucene-13: MaxScoreBulkScorer Filter Bitset Integration

**Goal**: Eliminate overhead of filtered disjunctions by pre-loading FILTER clause into per-window FixedBitSet, enabling the optimized non-filter scoring paths.

**Motivation**: Benchmark data shows adding a FILTER clause flips Lucene from winning (0.94x) to losing (1.35x) vs TurboPuffer. The per-doc zigzag join in `scoreInnerWindowWithFilter()` adds massive overhead. TurboPuffer handles filters with ~10% overhead. jpountz left a TODO at line 170-171.

**Branch**: `lucene/maxscore-filter-bitset`
**Depends on**: None (standalone)
**LOC estimate**: ~150

---

## Implementation Status: DONE — Benchmark Validated

Implementation is on branch `lucene/maxscore-filter-bitset` @ `79da00f8ea4`. Per-window bitset materialization via `intoBitSet()` + `WindowFilterBits` wrapper. Two issues found during benchmarking need fixing before PR.

---

## REQUIRED FIX 1: Benchmark bug — `minShouldMatch`

The benchmark query does NOT trigger MaxScoreBulkScorer. `BooleanScorerSupplier.booleanScorer()` line 217 requires `minShouldMatch >= 1` to route to `filteredOptionalBulkScorer()` → `MaxScoreBulkScorer`. Without it, queries route through `DefaultBulkScorer(ReqOptSumScorer)`.

**Fix**: In `MaxScoreBulkScorerBenchmark.buildQuery()`, add:
```java
builder.setMinimumNumberShouldMatch(1);
```

Also add selectivity sweep support:
```java
@Param({"none", "0.01", "0.03", "0.05", "0.1", "0.2", "0.5", "0.8", "0.95"})
public String filterSelectivity;
```
With matching `matchesFilter()` cases.

---

## REQUIRED FIX 2: Widen heuristic bounds

Original heuristic (based on broken benchmark data):
```java
private boolean shouldMaterializeFilter(DisiWrapper filter) {
    return filter.twoPhaseView == null
        && filter.cost >= Math.max(1, maxDoc >>> 3)   // >= 12.5%
        && filter.cost <= Math.max(1, maxDoc >>> 1);   // <= 50%
}
```

**Replace with**:
```java
private boolean shouldMaterializeFilter(DisiWrapper filter) {
    return filter.twoPhaseView == null
        && filter.cost >= Math.max(1, maxDoc >>> 5);   // >= ~3%
}
```

Upper bound removed entirely. Lower bound widened from 12.5% to ~3%.

---

## Benchmark Results

Platform: JDK 25, 1M docs, 1 fork, 1w+2i, 2s. Heuristic: `cost >= maxDoc >>> 5` (~3.125%), no upper bound.

### AMD EPYC (c5a.2xlarge, AVX2) — 12 clauses

| filter sel. | baseline (ops/s) | candidate (ops/s) | speedup | bitset path? |
|------------:|-----------------:|------------------:|--------:|:-------------|
| none | 123.1 | 124.0 | 1.01x | |
| 0.01 | 401.3 | 401.8 | 1.00x | no |
| 0.015 | 516.7 | 516.8 | 1.00x | no |
| 0.02 | 254.7 | 258.1 | 1.01x | no |
| 0.03 | 353.4 | 366.0 | 1.04x | no |
| 0.03125 | 224.9 | 286.4 | **1.27x** | yes |
| 0.05 | 204.7 | 275.7 | **1.35x** | yes |
| 0.1 | 166.3 | 240.3 | **1.44x** | yes |
| 0.2 | 140.6 | 201.9 | **1.44x** | yes |
| 0.5 | 118.0 | 156.5 | **1.33x** | yes |
| 0.8 | 100.3 | 146.2 | **1.46x** | yes |
| 0.95 | 90.9 | 121.7 | **1.34x** | yes |

### AMD EPYC (c5a.2xlarge, AVX2) — 45 clauses

| filter sel. | baseline (ops/s) | candidate (ops/s) | speedup | bitset path? |
|------------:|-----------------:|------------------:|--------:|:-------------|
| none | 27.8 | 27.4 | 0.99x | |
| 0.01 | 35.7 | 37.9 | 1.06x | no |
| 0.015 | 59.8 | 58.4 | 0.98x | no |
| 0.02 | 20.3 | 20.4 | 1.01x | no |
| 0.03 | 35.0 | 34.7 | 0.99x | no |
| 0.03125 | 16.0 | 55.9 | **3.50x** | yes |
| 0.05 | 12.1 | 43.1 | **3.56x** | yes |
| 0.1 | 9.4 | 39.4 | **4.21x** | yes |
| 0.2 | 9.1 | 39.8 | **4.40x** | yes |
| 0.5 | 8.8 | 35.2 | **3.98x** | yes |
| 0.8 | 8.1 | 28.9 | **3.58x** | yes |
| 0.95 | 8.1 | 24.7 | **3.05x** | yes |

### Intel Ice Lake (c6i.2xlarge, AVX-512) — 12 clauses

| filter sel. | baseline (ops/s) | candidate (ops/s) | speedup | bitset path? |
|------------:|-----------------:|------------------:|--------:|:-------------|
| none | 138.1 | 134.4 | 0.97x | |
| 0.01 | 467.8 | 462.5 | 0.99x | no |
| 0.015 | 604.5 | 604.1 | 1.00x | no |
| 0.02 | 288.0 | 287.6 | 1.00x | no |
| 0.03 | 434.4 | 431.2 | 0.99x | no |
| 0.03125 | 258.6 | 340.9 | **1.32x** | yes |
| 0.05 | 248.0 | 315.3 | **1.27x** | yes |
| 0.1 | 200.8 | 279.4 | **1.39x** | yes |
| 0.2 | 161.7 | 234.4 | **1.45x** | yes |
| 0.5 | 127.9 | 177.3 | **1.39x** | yes |
| 0.8 | 125.6 | 162.0 | **1.29x** | yes |
| 0.95 | 104.7 | 133.8 | **1.28x** | yes |

### Intel Ice Lake (c6i.2xlarge, AVX-512) — 45 clauses

| filter sel. | baseline (ops/s) | candidate (ops/s) | speedup | bitset path? |
|------------:|-----------------:|------------------:|--------:|:-------------|
| none | 26.7 | 29.6 | 1.11x | |
| 0.01 | 40.2 | 41.1 | 1.02x | no |
| 0.015 | 65.3 | 64.5 | 0.99x | no |
| 0.02 | 22.7 | 22.6 | 1.00x | no |
| 0.03 | 38.3 | 38.0 | 0.99x | no |
| 0.03125 | 17.8 | 64.4 | **3.63x** | yes |
| 0.05 | 13.1 | 62.0 | **4.73x** | yes |
| 0.1 | 10.3 | 47.1 | **4.56x** | yes |
| 0.2 | 9.7 | 47.2 | **4.86x** | yes |
| 0.5 | 9.9 | 38.3 | **3.87x** | yes |
| 0.8 | 8.4 | 32.8 | **3.93x** | yes |
| 0.95 | 9.2 | 27.6 | **3.01x** | yes |

### Key findings

- **Clean threshold**: `maxDoc >>> 5` (3.125%) gives sharp activation. Below threshold: 0.98-1.06x (no regressions). At threshold: 1.27-3.63x. Above: 1.27-4.86x.
- **Dense filters win big**: sel=0.8 gives 1.29-1.46x (cc=12), 3.58-3.93x (cc=45). The original upper bound was wrong — it was based on a broken benchmark.
- **45-clause queries benefit most**: 3-5x across the board due to multi-essential clause path eliminating zigzag overhead.
- **Consistent across architectures**: AMD and AVX-512 track closely, confirming the optimization is algorithmic not hardware-dependent.
- **Heuristic formula**: `maxDoc >>> 5` = `maxDoc / 32` = 3.125% selectivity. For 1M docs, filter cost must be ≥ 31,250.
- **cc=45 dominance**: multi-essential path (`scoreInnerWindowMultipleEssentialClauses`) benefits most — eliminates zigzag + enables batch decode. 3-5x wins across the board.

---

## Architecture Notes

### Why it works (even without applyMask)

The win comes from eliminating the **zigzag join** in `scoreInnerWindowWithFilter()`, NOT from bulk bitwise AND:

**Baseline**: per-doc `filter.advance(essentialDoc)` ↔ `essential.advance(filterDoc)` back-and-forth with queue maintenance per doc.

**Candidate**: one-time `intoBitSet()` per window, then forward-only iteration via `TermScorer.nextDocsAndScores()` (batch 64-doc decode) + per-doc `WindowFilterBits.get()` (array lookup + bit test).

### `applyMask()` is implemented but unreachable

`WindowFilterBits.applyMask()` exists but is never called — `TermScorer.nextDocsAndScores()` reaches the filter via `DocAndFloatFeatureBuffer.apply(Bits)` which calls per-doc `get()`. This is a future optimization opportunity, not a blocker.

---

## PR Checklist

- [ ] Fix benchmark: `setMinimumNumberShouldMatch(1)` + selectivity sweep
- [ ] Widen heuristic: `cost >= maxDoc >>> 5`, no upper bound
- [ ] `./gradlew check` passes
- [ ] Existing TestMaxScoreBulkScorer tests pass
- [ ] 137 new test lines in TestMaxScoreBulkScorer
- [ ] No regression on unfiltered queries (confirmed: 0.98-1.13x noise)
- [ ] Reference jpountz TODO and benchmark data in PR description
