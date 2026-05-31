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

## lucene-10: rangeIntoBitSet Remaining Overrides

**Goal**: Optimize `rangeIntoBitSet()` for encoding paths that still use the default per-doc loop: sparse NumericDocValues and GCD/delta encoded dense fields with table encoding.

**Motivation**: `BatchDocValuesRangeIterator` calls `values.rangeIntoBitSet()` on MAYBE blocks. Dense raw-encoded fields dispatch to `DocValuesRangeSupport` (SIMD-capable). GCD/delta fields use a per-doc scalar loop. Sparse fields fall through to the base class `advanceExact()` + `longValue()` loop. These are the remaining hot paths.

**Branch**: `lucene/range-into-bitset-overrides` (to create)
**Depends on**: lucene-08 (BatchDocValuesRangeIterator infrastructure), lucene-09 (GCD/delta already handled separately)
**LOC estimate**: ~200
**Relationship to lucene-09**: lucene-09 handles GCD/delta bound transform specifically. lucene-10 handles everything else: sparse fields, table-encoded fields, and ensures the base class default is only hit for truly generic cases.

---

## Stage 1: Sparse NumericDocValues rangeIntoBitSet
**Goal**: Override `rangeIntoBitSet` for sparse numeric doc values using bulk operations
**Success Criteria**: Sparse numeric range queries use bulk path instead of per-doc advanceExact
**Tests**: Extend TestDocValuesRangeQuery with sparse field configurations

### Current state (Lucene90DocValuesProducer, ~line 1036):
Sparse NumericDocValues uses `IndexedDISI` for doc existence. No `rangeIntoBitSet` override — falls through to `NumericDocValues.rangeIntoBitSet()` default which calls `advanceExact(d)` for every doc in [fromDoc, toDoc), even docs without values.

### Optimized path:
```java
@Override
public void rangeIntoBitSet(
    int fromDoc, int toDoc, long minValue, long maxValue,
    FixedBitSet bitSet, int offset) throws IOException {
  int currentDoc = disi.docID();
  if (currentDoc < fromDoc) {
    currentDoc = disi.advance(fromDoc);
  }
  while (currentDoc < toDoc) {
    long v = values.get(/* position for currentDoc */);
    // Apply GCD/delta transform if needed
    if (v >= minValue && v <= maxValue) {
      bitSet.set(currentDoc - offset);
    }
    currentDoc = disi.nextDoc();
  }
}
```

Key improvement: Only visits docs that HAVE values (via IndexedDISI), not all docs in range. For a 10% dense field over a 4096-doc window, this visits ~410 docs instead of 4096.

### Code location:
- `Lucene90DocValuesProducer.java:1036-1097` — sparse NumericDocValues creation
- Add override after `longValue()` / `longValues()` methods

### Side-effects to verify:
- IndexedDISI position state after rangeIntoBitSet — must leave iterator past `toDoc`
- Interaction with subsequent calls (iterator is forward-only)
- GCD/delta transform on sparse values (sparse + GCD is valid combo)
- Table-encoded values (sparse + table is valid combo)

**Status**: Not Started

---

## Stage 2: Table-Encoded Dense NumericDocValues
**Goal**: Override rangeIntoBitSet for table-encoded fields
**Success Criteria**: Table-encoded fields use lookup-based range check
**Tests**: Create fields with small value cardinality to trigger table encoding

### Current state:
Table-encoded fields store ordinals instead of values. `longValue()` looks up `table[ordinal]`. The rangeIntoBitSet default calls `advanceExact` + `longValue` per doc.

### Optimized path:
Pre-compute which table ordinals fall in [minValue, maxValue]. Then check ordinals directly without the table lookup per doc:
```java
// Pre-compute ordinal set once
boolean[] matchingOrdinals = new boolean[tableSize];
for (int i = 0; i < tableSize; i++) {
  matchingOrdinals[i] = (table[i] >= minValue && table[i] <= maxValue);
}
// Then iterate and check ordinal
for (int d = fromDoc; d < toDoc; d++) {
  int ord = (int) values.get(d);
  if (matchingOrdinals[ord]) {
    bitSet.set(d - offset);
  }
}
```

### Code location:
- `Lucene90DocValuesProducer.java` — find table-encoded path (where `entry.table != null`)
- Table lookup is in the same getNumeric method chain

**Status**: Not Started

---

## Stage 3: Tests and Benchmark
**Goal**: Verify correctness and measure improvement
**Success Criteria**: No regression on dense raw fields; measurable improvement on sparse/table fields
**Tests**:
- Sparse numeric field range queries (various densities: 1%, 10%, 50%)
- Table-encoded field range queries (cardinality 2, 8, 64)
- Mixed segment with dense + sparse fields
- Verify results match per-doc baseline

### Benchmark:
```bash
./gradlew -p lucene/benchmark-jmh jmh \
  -Pjmh.includes="SparseDocValuesRangeQueryBenchmark" \
  -Pjmh.prof="perfnorm"
```

### Expected improvement:
- Sparse fields: 2-5x faster for low-density fields (skip non-existent docs)
- Table-encoded: 10-30% faster (no per-doc table lookup, pre-computed ordinal set)
- Dense raw: no change (already optimized via DocValuesRangeSupport)

**Status**: Not Started

---

## PR checklist:
- [ ] `./gradlew check` passes
- [ ] Existing tests pass (TestDocValuesRangeQuery, TestSortedNumericDocValuesRangeQuery)
- [ ] New tests for sparse and table-encoded fields
- [ ] Benchmark shows improvement on target field types
- [ ] No regression on dense raw fields
- [ ] PR title: "Optimize rangeIntoBitSet for sparse and table-encoded doc values"
- [ ] Keep scope to rangeIntoBitSet overrides only — no query wiring changes (that's lucene-11)
