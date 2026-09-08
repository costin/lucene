# HNSW Segment-Merge Optimization — Analysis & Results

**Repository:** apache/lucene (fork `costin/lucene`) · **Date:** 2026-09-08
**Machine:** 8 vCPU, 47 GiB RAM, x86_64 · **JDK:** Temurin 25 (`--add-modules=jdk.incubator.vector`, 512-bit SIMD + FMA) · **Lucene:** 11.0.0-SNAPSHOT (`main` @ 3a63f5e)

This document records an end-to-end evaluation of whether three recent HNSW graph-merge papers — plus a faithful combined design — can improve Lucene's existing HNSW segment-merge (`MergingHnswGraphBuilder` / `IncrementalHnswGraphMerger` / `ConcurrentHnswMerger`). All raw data is in `results.csv` and the `*.log` files alongside this report.

---

## 1. Executive summary

- **Lucene's existing "smart merge" (`MergingHnswGraphBuilder`) is already strong.** On SIFT it is ~3× faster than a full rebuild at ≤1.1 recall points lower, and under `ConcurrentHnswMerger` it scales **near-linearly (6.5× on 8 workers)**.
- **None of the evaluated algorithms beat it** — IGTM (warm-start), FGIM (cross-query), Lazy backward-connect (light/thorough), and the faithful HNSW-Merger **Combined** design all add cost and/or lose recall.
- The papers' headline speedups are measured **against naive rebuild / incremental insertion baselines that Lucene already surpasses** (FGIM's own paper shows Lucene only ~1.3× behind full FGIM, in C++).
- The **reverse-link write lock is NOT the bottleneck** at ≤8 workers (≤5% of wall time); eliminating it (Combined) yields no speedup.
- Low recall on uniform-**RANDOM** vectors is a **measurement artifact of distance concentration** (Mean Distance Ratio ≈ 1.02 — answers are near-optimal); **SIFT** is the meaningful dataset.
- **The real, addressable gaps are in Lucene's own merge:** (1) all-or-nothing deletion handling collapses concurrent scaling to 2.8× at 20% deletes; (2) a serial base-init/ordinal-mapping prefix caps efficiency; (3) the reverse-link lock, while minor now, grows with worker count.

**Recommendation:** keep `MergingHnswGraphBuilder` + `ConcurrentHnswMerger`. Do not land any of the variants. Pursue the deletion-path and serial-prefix improvements instead.

---

## 2. Method

### Datasets
- **RANDOM** — uniform `float[128]` from `Random(42)` (deterministic).
- **SIFT** — SIFT1M 128-dim descriptors (`sift-128-euclidean`), the standard realistic ANN dataset.
- Similarity: **EUCLIDEAN**.

### Parameters swept
- `vectorCount ∈ {2000 (smoke), 200000, 512000}`
- `numSegments ∈ {2, 5}` — segment 0 (base) holds 60%; the rest split the remainder equally.
- `deleteRatio ∈ {0.0, 0.2}` — **see §5 for exact deletion semantics.**
- `beamWidth ∈ {100, 64, 48}` (single-thread combined runs).
- `numWorkers ∈ {1, 2, 4, 8}` (parallel runs).
- Fixed: `dim=128`, `M=16`, `beamWidth=100` (unless swept).

### Merge modes
- **FULL_REBUILD** — build one HNSW over all merged vectors from scratch (`HnswGraphBuilder`). The reference "just rebuild it" baseline.
- **SMART_MERGE** — Lucene's production merge (`MergingHnswGraphBuilder.fromGraphs`, base-graph reuse + join-set + reduced-beam entry points).
- **COMBINED** — faithful HNSW-Merger design: forward-only insertion + cross-query of source graphs + incoming-edge recording + a single search-free `RobustPrune` (no reverse-link search, no reverse-link lock).

### Metrics
- **wallMs** — merge wall-clock (mean of 3 iterations; 2 for rebuild).
- **distanceComputations** — deterministic count via a wrapped `RandomVectorScorerSupplier`.
- **recall@100** — overlap of merged-graph top-100 vs brute-force exact top-100 over 100 queries.
- **MDR** (Mean Distance Ratio) — mean of `trueDist(approx_i)/trueDist(exact_i)`; ≥1.0, where 1.0 = answers exactly as close as truth.
- **ε-recall@100** (ε=0.01) — fraction of returned neighbors within `(1+ε)·trueDist(exact_100)`.
- Parallel instrumentation: `writeLockWaitMs`, `writeLockCount`, `effectiveConcurrency`, `blockedPct`, `workerThreads`.

---

## 3. Baseline: full rebuild vs smart merge

### SIFT (the meaningful dataset)

| n | seg | del | mode | merge s | distances | recall@100 |
|---:|---:|---:|---|---:|---:|---:|
| 200k | 2 | 0.0 | FULL_REBUILD | 28.71 | 385.7M | 0.9649 |
| 200k | 2 | 0.0 | SMART_MERGE | **9.54** | 114.9M | 0.9569 |
| 200k | 5 | 0.0 | FULL_REBUILD | 27.97 | 385.7M | 0.9649 |
| 200k | 5 | 0.0 | SMART_MERGE | **9.11** | 114.2M | 0.9542 |
| 512k | 2 | 0.0 | FULL_REBUILD | 92.61 | 1.09B | 0.9446 |
| 512k | 2 | 0.0 | SMART_MERGE | **30.98** | 312.9M | 0.9331 |
| 512k | 5 | 0.0 | FULL_REBUILD | 94.45 | 1.09B | 0.9446 |
| 512k | 5 | 0.0 | SMART_MERGE | **30.30** | 309.8M | 0.9341 |

**Smart merge is ~3× faster than full rebuild at ≤1.1 recall points lower.** Merging works: the purpose of reusing the base graph is validated.

### RANDOM (recall is concentration-limited — see §7)
SMART is likewise ~2.5–3× faster than FULL; recall ≈ 0.18–0.29 for *both* (not a merge defect — it's the data).

---

## 4. The algorithms vs smart merge (single-thread)

### SIFT, deleteRatio = 0.0

| n | seg | variant | merge s | Δ time vs SMART | distances | recall@100 |
|---:|---:|---|---:|---:|---:|---:|
| 200k | 2 | SMART_MERGE | 9.54 | — | 114.9M | 0.9569 |
| 200k | 2 | IGTM | 9.78 | +2.5% | 121.0M | 0.9405 |
| 200k | 2 | FGIM | 12.73 | +33% | 141.3M | 0.9570 |
| 200k | 2 | Lazy-light | 8.96 | −6.2% | 105.7M | 0.9333 |
| 200k | 2 | Lazy-thorough | 35.48 | +272% | 482.5M | 0.9424 |
| 512k | 2 | SMART_MERGE | 30.98 | — | 312.9M | 0.9331 |
| 512k | 2 | IGTM | 30.22 | −2.5% | 322.1M | 0.8836 |
| 512k | 2 | FGIM | 41.87 | +35% | 383.7M | 0.9339 |
| 512k | 2 | Lazy-light | 30.00 | −3.2% | 285.9M | 0.9038 |
| 512k | 2 | Lazy-thorough | 107.64 | +247% | 1.28B | 0.9194 |

**Reading:** IGTM costs recall for no speed; FGIM costs +33–76% time (more with more segments) for no recall gain; Lazy-light saves ~3–9% distances but loses 3–8 recall points; Lazy-thorough is *slower than a full rebuild* and still below SMART recall.

### Combined design (COMBINED), SIFT 512k / 2 seg, single-thread

| del | beam | mode | merge s | distances | recall@100 |
|---:|---:|---|---:|---:|---:|
| 0.0 | 100 | SMART | 32.63 | 313M | 0.9331 |
| 0.0 | 100 | COMBINED | 103.06 | 980M | 0.8523 |
| 0.0 | 64 | SMART | 24.22 | 217M | 0.9229 |
| 0.0 | 64 | COMBINED | 70.57 | 670M | 0.8474 |
| 0.0 | 48 | SMART | 20.41 | 170M | 0.9145 |
| 0.0 | 48 | COMBINED | 51.99 | 524M | 0.8436 |
| 0.2 | 100 | SMART | 36.63 | 382M | 0.9466 |
| 0.2 | 100 | COMBINED | 38.50 | 436M | 0.7840 |

Cross-querying every source graph is ~3× the distance work; the search-free prune produces a lower-quality graph. Reducing `beamWidth` scales both down but COMBINED never catches SMART.

---

## 5. Deletion semantics (exactly what was tested)

Two conditions were tested for every configuration:

- **`deleteRatio = 0.0`** — **no deleted documents.** All segments are fully live and eligible as merge sources.
- **`deleteRatio = 0.2`** — **20% of documents deleted in every non-base segment.** The base (largest, segment 0, ~60% of vectors) is kept delete-free, mirroring how `IncrementalHnswGraphMerger` selects the largest low-deletion graph as the base. Deleted vectors are removed from the merged output (their ordinals map to −1).

At 20% deletes the merged live count drops (e.g. 512000 → 471107; 200000 → 184081). This faithfully mirrors Lucene's real merge: **a segment containing any deletions is excluded from the reusable-source set** (`IncrementalHnswGraphMerger.addReader` only adds a graph when `candidateVectorCount == graphSize`), so its live vectors are re-inserted from scratch.

---

## 6. Parallel scaling (SIFT 512k / 2 seg) — the decisive experiment

Same thread pool, same `TaskExecutor`, same warmup/iterations for **both** baseline and combined (fair comparison). Baseline uses the real `ConcurrentHnswMerger` / `HnswConcurrentMergeBuilder` path.

### deleteRatio = 0.0

| workers | SMART s | SMART speedup | eff.conc | writeLockWait | blocked | COMBINED s | COMBINED speedup | COMBINED lockWait |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 45.18 | 1.00× | 1.00 | 103 ms (2.75M acq) | 0 | 115.89 | 1.00× | 0 |
| 2 | 22.24 | 2.03× | 1.995 | 145 ms | 0 | 59.13 | 1.96× | 0 |
| 4 | 11.50 | 3.93× | 3.952 | 182 ms | 0 | 33.44 | 3.47× | 0 |
| 8 | **6.90** | **6.55×** | 7.725 | 362 ms | 0 | 25.54 | 4.54× | 0 |

- SMART recall stable 0.944–0.946; COMBINED recall stuck at **0.859**.
- **The baseline is NOT strangled:** near-linear scaling (6.55× @ 8w, effectiveConcurrency 7.73, `blockedPct=0`). The reverse-link lock costs ≤5% of wall time.
- COMBINED eliminates the lock (`writeLockCount=0`) — but that buys nothing, and it is **3.7× slower than SMART at 8 workers** with lower recall.

### deleteRatio = 0.2 — reveals a serial bottleneck

| workers | SMART wall s | SMART **build** s | serial (wall−build) | SMART speedup | COMBINED s | COMBINED recall |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 35.61 | 35.39 | 0.2 | 1.00× | 35.00 | 0.784 |
| 2 | 17.91 | 18.49 | ~0 | 1.99× | 17.81 | 0.783 |
| 4 | 17.45 | 16.57 | 0.9 | 2.04× | 10.46 | 0.783 |
| 8 | 12.61 | **5.69** | **6.92** | **2.82×** | 18.56 | 0.783 |

**Key finding:** with deletes, the *parallel build* still scales (35.4s → 5.7s), but wall-clock only improves **2.8×** because a **~6.9 s serial phase** appears at 8 workers. This is Lucene's real scaling gap in the delete case. (COMBINED at 4 workers is faster but at a catastrophic recall cost of 0.783 vs 0.947, and regresses at 8 workers.)

---

## 7. Recall vs answer quality (why RANDOM recall is low)

`quality-mdr` runs (n=200k, SMART, no deletes, raw-L2 distances):

| dataset | seg | recall@100 | MDR | ε-recall@100 |
|---|---:|---:|---:|---:|
| RANDOM | 2 | 0.2784 | **1.0238** | 0.4377 |
| RANDOM | 5 | 0.2747 | **1.0239** | 0.4362 |
| SIFT | 2 | 0.9569 | **1.0011** | 0.9992 |
| SIFT | 5 | 0.9542 | **1.0011** | 0.9991 |

On RANDOM, recall ≈ 0.28 but **MDR ≈ 1.02** — the returned neighbors are only ~2.4% farther than the true ones. Uniform high-dim L2 distances concentrate, so many points are interchangeable and ordinal-overlap counting treats near-ties as misses. On SIFT distances are separated: recall high, MDR ≈ 1.001. **Draw recall conclusions from SIFT only.**

---

## 8. Why the papers don't win here

1. **Baseline already optimized.** Lucene's smart merge already implements the papers' core ideas: base-graph reuse (`InitializedHnswGraphBuilder.initGraph`), join-set coverage (`UpdateGraphsUtils.computeJoinSet`), and neighbor-derived **reduced-beam** entry points (`beamCandidates0 = min(beamWidth/2, M*3) = 48`). The papers' gains are vs *naive* merge / incremental insertion, which Lucene beats.
2. **They lose on the papers' own metric too.** IGTM/FGIM do *more* distance computations than SMART, so it is not merely a SIMD wall-time story.
3. **Contention is not the limiter.** The reverse-link lock is ≤5% at 8 workers; removing it (Combined) does nothing.
4. **Faithful-port caveat.** The three earlier reimplementations grafted paper ideas onto Lucene's insert loop rather than replacing construction (Lazy-thorough re-searches every node — the opposite of the paper). The Combined design *is* faithful, and it still loses.

---

## 9. Lucene's real weak points (recommended work)

1. **All-or-nothing deletion handling (biggest).** Any deletion disqualifies a segment from reuse (`addReader`: `candidateVectorCount == graphSize`), forcing full re-insertion of its live nodes and a serial tail that drops concurrent scaling to 2.8× at 20% deletes. Recall is unaffected — this is purely a speed/scaling defect. **Fix:** partial reuse with local repair of deletion holes (see report §"Partial reuse").
2. **Serial base-init / ordinal-mapping prefix.** `getNewOrdMapping` + `initGraph` run single-threaded before the parallel region, capping efficiency (~81% even with no deletes). **Fix:** parallelize base copy/repair and ord-mapping.
3. **Reverse-link lock grows with workers** (103→362 ms, 1→8w). Negligible now but a future ceiling at 16–32+ cores — revisit backlink-elimination there, not at ≤8 workers.
4. **~1–2 pt recall gap vs full rebuild** from the reduced-beam shortcut on non-join nodes; an adaptive beam could close it if desired.

---

## 10. Appendix — full raw data

Full per-configuration results are in `results.csv` (columns: `label,dataset,mergeMode,vectorCount,numSegments,deleteRatio,mergedSize,wallMs,distanceComputations,recall,iters,mdr,epsRecall`). Per-run logs: `baseline.log`, `igtm.log`, `fgim.log`, `lazy-light.log`, `lazy-thorough.log`, `quality-mdr.log`, `combined-st-*.log`, `combined-par-sift512k.log`.

Labels: `baseline` (FULL_REBUILD + SMART_MERGE), `igtm`, `fgim`, `lazy-light`, `lazy-thorough`, `quality-mdr` (MDR/ε-recall), `combined-st-*` (single-thread combined), `combined-par-*` (parallel scaling), `smoke` (2k validation).
