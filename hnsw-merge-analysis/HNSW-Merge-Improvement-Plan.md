# HNSW Segment-Merge — Improvement Plan

**Repository:** apache/lucene (fork `costin/lucene`) · **Date:** 2026-09-08
**Companion:** see `HNSW-Merge-Analysis.md` + `results.csv` in this folder for the data this plan is built on.

## Guiding principle

The benchmark study showed the three published merge algorithms (IGTM, FGIM, Lazy) and a faithful HNSW-Merger "Combined" design **do not beat Lucene's existing `MergingHnswGraphBuilder`**, because they optimize a *contention* problem Lucene does not have (its concurrent merge already scales ~6.5× on 8 workers). The **real, data-supported gaps are throughput gaps in Lucene's own merge**, with **no recall cost**:

1. Deletions collapse concurrent scaling from **6.5× → 2.8×** (a ~7 s serial tail appears at 20% deletes).
2. A **serial base-init / ordinal-mapping prefix** caps efficiency at ~81% even with no deletes.
3. The reverse-link write lock is minor now (≤5% at 8 workers) but **grows** with worker count.

This plan targets those, in priority order. Every item is measured with the **existing fair-baseline harness** (`HnswMergeResearch` / `HnswMergeBenchmark`) so improvements are apples-to-apples against today's `SMART_MERGE` and `FULL_REBUILD`.

## Global success criteria & method

- **Primary metric:** merge wall-time and parallel scaling (speedup vs 1 worker) under `ConcurrentHnswMerger`, plus deterministic **distance-computation count**.
- **Quality gate:** SIFT `recall@100` must not regress vs current `SMART_MERGE` (use SIFT, not RANDOM — RANDOM recall is concentration noise; confirm with MDR ≈ 1.0). Report MDR + ε-recall alongside recall.
- **Fairness gate:** baseline re-run through the *same* harness/worker-counts; report `writeLockWaitMs`, `effectiveConcurrency`, `blockedPct` to prove any win is real, not a strangled baseline.
- **Datasets/scales:** SIFT (primary) + RANDOM; n ∈ {200k, 512k}; segments ∈ {2, 5}; deleteRatio ∈ {0.0, 0.2}; workers ∈ {1,2,4,8}; beamWidth ∈ {100,64,48}.
- **Correctness:** all `org.apache.lucene.util.hnsw.*` tests + `TestHnswMergeAbort` green; add targeted unit tests per workstream.

---

## Workstream 1 — Partial reuse of deletion-bearing graphs (repair, don't discard) [HIGHEST VALUE]

### Problem (from data)
At `deleteRatio=0.2`, the parallel build still scales (35.4 s → 5.7 s at 8 workers) but wall-clock only improves 2.8× because ~6.9 s becomes serial; distance work also rises (SIFT 512k: 381M vs 313M). Recall is unaffected — this is purely speed.

### Current behavior
`IncrementalHnswGraphMerger.addReader` only adds a graph to the reusable-source set when `candidateVectorCount == graphSize` (zero deletes). Only the **largest** graph may be a base with deletes ≤ 40% (`DELETE_PCT_THRESHOLD`), repaired via `InitializedHnswGraphBuilder.initGraph`. Every other segment with *any* deletion is discarded and its live vectors are re-inserted from scratch (the expensive, partly-serial path).

### Proposed algorithm — local "bridge repair" of holes
A deleted node leaves (a) dangling in-edges from live neighbors and (b) potentially broken paths. Instead of discarding the graph:
1. **Tombstone** deleted ordinals — keep graph structure; exclude tombstones as search results and as connection targets; remap live ordinals to merged space.
2. **Bridge repair (local, no search):** for each live node `v` whose neighbor `d` is deleted, replace edge `v→d` with edges from `v` to `d`'s surviving out-neighbors ("friends of the deleted friend"), then apply the existing diversity heuristic (`NeighborArray.addAndEnsureDiversity` / RobustPrune) to cap degree at `M` (2M on level 0).
3. **Bounded top-up (only stragglers):** if tombstoning drops a node's live degree below a threshold (e.g. `< M/2`) or risks disconnection, run a *small-ef bounded* local search from that node to add a few edges — for the few under-connected nodes only, not all.
4. **Feed the repaired graph into the smart merge** as a normal source (join-set + reduced-beam entry points), so its structure is reused instead of rebuilt.

This generalizes what `initGraph` already does for the base graph to *all* source graphs. At 20% deletes ~80% of edges survive, so local repair is far cheaper than full re-insertion.

### Implementation sketch
- `IncrementalHnswGraphMerger.addReader`: relax the `== graphSize` gate to admit graphs with deletes up to a (tunable) threshold as **repairable sources**; track their liveDocs.
- New `HnswGraphDeletionRepair` utility (or extend `InitializedHnswGraphBuilder`): implements tombstone + bridge-repair + bounded top-up, producing an `OnHeapHnswGraph` over live ordinals.
- `MergingHnswGraphBuilder.fromGraphs` / `IncrementalHnswGraphMerger.createBuilder`: accept repaired graphs alongside clean ones.
- **Concurrency:** repair must be per-node and parallelizable (striped `HnswLock`, or repair each source graph independently across workers) so it does not reintroduce a serial tail.

### Risks & mitigations
- *Edge quality lower than fresh insert* → gate on SIFT recall vs current reconstruct path; fall back to reconstruction if a segment's deletion rate exceeds a threshold.
- *Degree overflow / duplicate edges* → reuse `addAndEnsureDiversity` (already dedup+prunes).
- *Disconnection at high deletion rates* → the bounded top-up step; keep the 40%-style ceiling.

### Validation & success
Re-run the `deleteRatio=0.2` parallel sweep. **Target:** move 20%-delete scaling from 2.8× toward the ~6.5× clean-case curve and cut distance work, with SIFT recall within ~1 pt of current SMART_MERGE.

---

## Workstream 2 — Parallelize the serial init prefix [HIGH VALUE]

### Problem
Even with no deletes, 8-worker efficiency is ~81% (6.55×, not 8×). `getNewOrdMapping` (ordinal remap) and `InitializedHnswGraphBuilder.initGraph` (base-graph copy + delete repair) run single-threaded before the parallel region. With deletes this prefix grows (Workstream 1 interacts).

### Approach
- Parallelize the **base-graph copy/repair** in `initGraph`: partition base nodes across workers, copy neighbor arrays concurrently (they are independent reads of the source graph), synchronize only on shared write structures.
- Parallelize / streamline `getNewOrdMapping`: the doc→ordinal map build is embarrassingly parallel per source; consider a single pass with concurrent hash maps or per-worker partial maps merged at the end.
- Measure the serial fraction directly (the harness already reports `buildMs` vs `wallMs`; extend to time init vs merge phases separately).

### Risks & mitigations
- *Concurrency bugs in graph copy* → strong unit tests + deterministic seeds; compare copied graph to serial copy for equality.

### Success
Reduce `wall − build` serial time to near-zero at 8 workers for both delete ratios; push clean-case scaling above 6.5× and unblock scaling past 8 cores.

---

## Workstream 3 — Correct warm-start for reconstructed nodes [MEDIUM, CHEAP]

### Idea
When a node from a deleted (or excluded) segment *is* re-inserted from scratch, seed its level-0 search with **its own old neighbors' new ordinals** (from the discarded source graph), instead of a cold search from the entry point. This is the warm-start idea IGTM implemented incorrectly (IGTM used the *previous, unrelated* node's result); applied to a node's *own* prior neighborhood it is both correct and free (no extra distance computations to build the entry set).

### Implementation
- In the reconstruct path (uninitialized nodes in `MergingHnswGraphBuilder.build` / the concurrent worker), when the node has a known prior neighborhood in a source graph, pass those remapped ordinals as `eps` to `addGraphNode(node, eps)`.
- Guard: only use entry points already present in the merged graph (same invariant as the existing neighbor-of-neighbor `eps`).

### Success
Fewer distance computations per reconstructed node at equal recall; complements Workstream 1 for the residual reconstruct cases.

---

## Workstream 4 — Revisit backlink elimination at high core counts [CONDITIONAL / FUTURE]

### Rationale
The reverse-link write lock is not the bottleneck at ≤8 workers (≤5% wait), but `writeLockWaitMs` grew 103 → 362 ms across 1 → 8 workers. At **16–32+ workers** it may become material. The `Combined` design (search-free backward-connect) drove it to zero — the mechanism works; it was simply premature and paired with an expensive cross-query.

### Approach (only if a high-core host is available)
- Re-measure the current `ConcurrentHnswMerger` scaling at 16/32/64 workers; quantify lock-wait growth.
- If lock-wait becomes a real ceiling, prototype backlink elimination **without** the expensive cross-query, paired with a *cheap* faithful repair (record incoming edges during the forward pass + a single RobustPrune, as in HNSW-Merger — not the "thorough re-search" variant that was slower than a rebuild).
- Strict quality gate: SIFT recall must hold (the earlier Combined design lost 8–16 points — that is the bar to clear).

### Success
A backlink-free concurrent build that scales past the lock-bound baseline at 16+ cores **with no recall regression**. If the recall bar cannot be met cheaply, shelve it.

---

## Workstream 5 — Adaptive construction beam [OPTIONAL / QUALITY]

### Rationale
Non-join nodes search with a reduced beam (`min(beamWidth/2, M*3) = 48`), which is where merge recall sits ~1–2 pts under a full rebuild (SIFT 512k: 0.933 vs 0.945).

### Approach
Use the full beam only for "low-confidence" insertions (e.g. when the reduced-beam result's top similarity is below a threshold, or the candidate set is small/degenerate); keep the cheap beam otherwise.

### Success
Close most of the recall gap vs full rebuild at negligible extra cost; ship only if it is net-positive on the time/recall frontier.

---

## Sequencing

1. **Phase 1 — instrumentation & baseline** (small): split init-vs-merge timing in the harness; lock in current baseline curves (done data exists; formalize).
2. **Phase 2 — Workstream 1** (largest win): partial-reuse repair, single-threaded correctness first, then concurrency.
3. **Phase 3 — Workstream 2**: parallelize the init prefix (compounds with WS1).
4. **Phase 4 — Workstream 3**: warm-start reconstructed nodes (cheap add-on).
5. **Phase 5 — Workstreams 4/5**: conditional, only if a high-core host (WS4) or a recall need (WS5) justifies them.

## Non-goals

- Re-landing IGTM / FGIM / Lazy / Combined as-is — the data shows they do not help Lucene's merge.
- Optimizing the RANDOM dataset's absolute recall (it is a distance-concentration artifact, not a defect).

## Key files

- `lucene/core/src/java/org/apache/lucene/util/hnsw/IncrementalHnswGraphMerger.java` — reader admission / delete gate, ord mapping, base selection.
- `lucene/core/src/java/org/apache/lucene/util/hnsw/InitializedHnswGraphBuilder.java` — base copy + delete repair (extend for WS1/WS2).
- `lucene/core/src/java/org/apache/lucene/util/hnsw/MergingHnswGraphBuilder.java` — smart merge (`updateGraph`, reconstruct path).
- `lucene/core/src/java/org/apache/lucene/util/hnsw/ConcurrentHnswMerger.java` / `HnswConcurrentMergeBuilder.java` — concurrent orchestration.
- `lucene/core/src/java/org/apache/lucene/util/hnsw/HnswLock.java` — striped locks + contention counters.
- `lucene/core/src/java/org/apache/lucene/util/hnsw/NeighborArray.java` — `addAndEnsureDiversity` (repair primitive).
- Harness: `lucene/benchmark-jmh/.../HnswMergeResearch.java`, `HnswMergeBenchmark.java`.

## Baseline reference (SIFT, from the study)

| condition | SMART_MERGE | scaling (1→8w) |
|---|---|---|
| 512k, 2 seg, del 0.0 | 30.98 s single-thread; 6.90 s @ 8w | **6.55×** (target for delete case) |
| 512k, 2 seg, del 0.2 | 35.6 s @ 1w; 12.6 s @ 8w | **2.82×** (Workstream 1 target) |
| full rebuild (512k) | 92.6 s | — |
