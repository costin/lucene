# HNSW Merge Research — Phase 2 Results

## Environment

- Host: 8 vCPU, 47 GiB RAM, Linux 6.12
- JDK: Temurin 25 (`--add-modules=jdk.incubator.vector`; research heap `-Xms8g -Xmx24g`, 2M HFEMB `-Xmx28g`)
- Base: `merge-algo-combined` @ `daaea04d1c2` (Phase-1 harness + Combined/Concurrent path)
- Work branch (local only): `hnsw-merge-phase2-code`
- Primary metric: **distance-computation count** (deterministic). Wall-clock is directional.
- Quality: recall@100, MDR, ε-recall@100 (ε=0.01) on raw L2
- RANDOM is sanity-only (do not use for quality comparison)
- HNSW unit tests: **111 tests, 2 skipped, SUCCESS** (`org.apache.lucene.util.hnsw.*`)
- Machine-readable rows: `results.csv` (58 data rows)

## Datasets

| Name | Dim | Max N | Notes |
|---|---|---|---|
| SIFT | 128 | 1,000,000 | ann-benchmarks `sift-128-euclidean` via HuggingFace `hhy3/ann-datasets` (ann-benchmarks.com returned 403). |
| HFEMB | 384 | 2,000,000 | Public clustered Wikipedia embeddings: `NotHotTryHard/wikipedia-en-harrier-270m-emb` (native 384-d, L2-normalized). Serves as both the clustered-distribution test and the 2M scale test. GloVe dropped. |
| RANDOM | 128 | harness | Uniform `float[128]`, seed 42. Sanity / verify-counter only. |

## Verification-counter table

| Algorithm | Counter | First successful value | OK? |
|---|---|---|---|
| IGTM_FIXED | warmStartUsed / coldFallback | RANDOM 5218/0; SIFT 512k 162931/0; SIFT 1M 319317/0; HFEMB 200k 62844/0 | YES |
| FGIM_P1 | crossQuerySearches | RANDOM 5218; SIFT 512k 162931 | YES |
| FGIM_P1P2 | nnDescentRounds / edgeUpdates / convergedAfter | RANDOM 5 / 446214 / 5 | YES (path ran; quality/cost bad — gated off) |
| LAZY_LIGHT | backwardLinksSkipped / added | RANDOM 178411/33423; SIFT 512k 2349594/1555386 | YES |
| REPAIR @ delete=0.2 | repairedGraphs / nodesRepaired | SIFT 512k 1/163907; SIFT 1M/2 1/320076; SIFT 1M/5 **4**/320076; HFEMB 200k 1/64081; HFEMB 2M 1/639765 | YES |

At delete=0.2 **without** repair, IGTM/FGIM extra-source counters are 0 because production reuse discards the dirty non-base graph. That is expected. The gate requires nonzero counters only when `segmentGraphs.length > 1`.

## Distance-count comparison (primary)

### Sequential delete=0.2: SMART vs REPAIR

This is the Phase B question. Ratio is stable across scale, dimension, and clustering.

| dataset | n | segs | SMART dist | REPAIR dist | Δ dist | SMART recall | REPAIR recall |
|---|---|---|---|---|---|---|---|
| RANDOM | 20k | 2 | 27,152,561 | 20,130,704 | −25.9% | 0.653 | 0.648 |
| SIFT | 512k | 2 | 381,583,074 | 251,138,777 | **−34.2%** | 0.947 | 0.940 |
| SIFT | 1M | 2 | 762,710,640 | 501,619,116 | **−34.2%** | 0.934 | 0.924 |
| SIFT | 1M | 5 | 762,710,640 | 498,848,666 | **−34.6%** | 0.934 | 0.924 |
| HFEMB 384-d | 200k | 2 | 137,519,756 | 90,557,298 | **−34.1%** | 0.974 | 0.973 |
| HFEMB 384-d | **2M** | 2 | 1,623,019,688 | 1,072,332,185 | **−33.9%** | 0.970 | 0.970 |

Repair reuses the dirty non-base graph after compacting it (`InitializedHnswGraphBuilder.initGraph`). Sequential SMART at delete=0.2 has `joinSetMs=0` (scratch-insert of live non-base vectors). After repair, `joinSetMs` is nonzero again and merge wall returns toward the clean-case curve.

### Sequential clean (delete=0.0): IGTM / FGIM / Lazy vs SMART

| dataset | n | SMART | IGTM (Δ) | FGIM_P1 (Δ) | LAZY_LIGHT (Δ) |
|---|---|---|---|---|---|
| RANDOM | 20k | 25.17M | 24.34M (−3.3%) | 28.72M (+14%) | 19.81M (−21%, recall 0.48) |
| SIFT | 512k | 312.88M | 299.71M (−4.2%) | 383.72M (+23%) | 285.85M (−9%, recall 0.904) |
| SIFT | 1M | 623.00M | 598.29M (−4.0%) | — gated | — |
| HFEMB | 200k | 113.87M | 108.79M (−4.5%) | — gated | — |

IGTM recall matches SMART at every point (within 0.002). FGIM_P1 recall matches SMART but costs more. Lazy cheaper, recall down.

FGIM_P1P2 (RANDOM 20k): 151.5M distances (**+502%**), recall 0.50 vs 0.63. **Not run on SIFT/HFEMB.**

## Wall-clock (directional)

| config | SMART w=1 | REPAIR w=1 merge | repairMs | SMART w=8 | REPAIR w=8 |
|---|---|---|---|---|---|
| SIFT 512k/2/0.0 | 31.3s | 37.2s (no-op) | 0 | 6.6s | 6.9s |
| SIFT 512k/2/0.2 | 41.3s | **31.3s** | 18.5s | 6.7s | 5.9s |
| SIFT 1M/2/0.0 | 83.8s | 97.3s (no-op) | 0 | 42.2s | 48.2s |
| SIFT 1M/2/0.2 | 106.6s | **64.4s** | 38.2s | 33.1s | 13.4s* |
| SIFT 1M/5/0.2 | 93.8s | **62.2s** | 26.2s (4 graphs) | 35.4s | 34.2s |
| HFEMB 200k/2/0.2 | 20.0s | **14.6s** | 9.7s | 3.2s | 5.3s |
| HFEMB 2M/2/0.0 | 385s | 380s | 0 | 130s (3.0×) | 122s |
| HFEMB 2M/2/0.2 | 440s | **312s** | 222s | 98s | 98s |

\*13.4s at SIFT 1M/2/0.2 REPAIR w=8 has the **same distances** as SMART w=8 (763M); treat as wall noise, not an algorithmic concurrent win.

Repair overhead (`repairMs`) is a compact `initGraph` on a separate scorer and is **not** in `distanceCount`. Four 100k repairs (1M/5) were cheaper than one 320k repair (1M/2): 26s vs 38s.

## Per-phase timing

Every RESULT line records `initMs`, `repairMs`, `joinSetMs`, `insertMs`, `totalMs`.

- Clean sequential SMART: init ~copy of base, joinSet ~0.5–3s, insert dominates.
- Delete=0.2 sequential SMART: `joinSetMs=0`, insert is a full HNSW build of the live non-base set.
- Delete=0.2 sequential REPAIR: joinSet returns (0.5–2s); insert looks like a clean smart-merge of the repaired source.

## Memory

Used-heap before / after / peak during merge (sampler ~20 ms).

| config | before | after | peak |
|---|---|---|---|
| SIFT 512k sequential | ~420 MB | ~2.4 GB | ~2.4–3.2 GB |
| SIFT 1M sequential | ~820 MB | ~4.7 GB | ~4.8 GB |
| HFEMB 200k sequential | ~360 MB | ~1.2 GB | ~1.2 GB |
| HFEMB 2M sequential | ~3.6 GB | ~6.4 GB | **~9.1 GB** |

## Concurrent scaling (workers=8)

`workers=8` is **ConcurrentHnswMerger-style copy-largest + insert-rest**, not a parallelized `MergingHnswGraphBuilder`. Distances are therefore *higher* than sequential SMART on clean data (SIFT 512k: 477M vs 313M; SIFT 1M: 952M vs 623M; HFEMB 2M: 2.03B vs 1.35B).

| config | seq SMART wall | conc SMART wall | wall ratio | seq SMART dist | conc SMART dist |
|---|---|---|---|---|---|
| SIFT 512k/2/0.0 | 31.3s | 6.6s | 4.7× | 313M | 477M |
| SIFT 512k/2/0.2 | 41.3s | 6.7s | 6.2× | 382M | 382M |
| SIFT 1M/2/0.0 | 83.8s | 42.2s | 2.0× | 623M | 952M |
| SIFT 1M/2/0.2 | 106.6s | 33.1s | 3.2× | 763M | 763M |
| HFEMB 2M/2/0.0 | 385s | 130s | 3.0× | 1.35B | 2.03B |
| HFEMB 2M/2/0.2 | 440s | 98s | 4.5× | 1.62B | 1.62B |

The Phase-1 “2.82× at 8 workers vs ~6.5× clean” gap does **not** appear as a concurrent-vs-concurrent delete penalty here: harness deletes are **non-base only**, so concurrent always copies a clean base. Concurrent walls at 0.0 vs 0.2 are similar or *better* at 0.2 (fewer inserts). The real delete penalty is **sequential SMART discarding the non-base graph**. Repair fixes that sequential path. Concurrent REPAIR distances equal concurrent SMART (insert set unchanged).

## Per-algorithm analysis

### IGTM vs Lucene’s existing `updateGraph` (required argument)

Lucene’s production `MergingHnswGraphBuilder.updateGraph` already seeds each non-join node `u` with:

1. **u’s own gS-neighbors**, remapped into merged space (`ordMapS[v]`), **and**
2. a **neighbor-of-neighbor expansion in gL** (neighbors of those remapped nodes).

Quoted from `MergingHnswGraphBuilder.updateGraph` on `merge-algo-combined`:

```java
// for each node outside of j set:
// form the entry points set for the node
// by joining the node's neighbours in gS with
// the node's neighbours' neighbours in gL
for (int u = 0; u < size; u++) {
  if (j.contains(u)) {
    continue;
  }
  IntHashSet eps = new IntHashSet();
  gS.seek(0, u);
  for (int v = gS.nextNeighbor(); v != NO_MORE_DOCS; v = gS.nextNeighbor()) {
    // if u's neighbour v is in the join set, or already added to gL (v < u),
    // then we add v's neighbours from gL to the candidate list
    if (v < u || j.contains(v)) {
      int newv = ordMapS[v];
      eps.add(newv);

      hnsw.seek(0, newv);
      int friendOrd;
      while ((friendOrd = hnsw.nextNeighbor()) != NO_MORE_DOCS) {
        eps.add(friendOrd);
      }
    }
  }
  addGraphNode(ordMapS[u], eps);
}
```

**Fixed IGTM** is a *subset*: seed only with u’s own remapped gS-neighbors already present in the merged graph (`nodeExistAtLevel`); if empty, cold-insert. No gL NoN expansion.

**Measured:** IGTM does **3–4.5% fewer** distances than SMART at delete=0.0 (RANDOM −3.3%, SIFT 512k −4.2%, SIFT 1M −4.0%, HFEMB 200k −4.5%) with **no recall gain** and mixed/worse wall. At delete=0.2 without repair it is **identical** to SMART (no `gS` to seed from). **Fixed IGTM is not a new algorithm relative to Lucene’s existing logic.** Phase-1 IGTM on `merge-algo-igtm` (reuse previous insertion’s search result) is a different idea and was not kept.

### FGIM

P1 cross-queries every **already-inserted** source graph with `RemappingVectorScorer` (deleted source ords score −∞) and unions hits into `eps`. Searching not-yet-inserted graphs would produce eps ordinals that are not in the merged HNSW.

P1 is **more expensive** than SMART (RANDOM +14%, SIFT 512k **+23%**) with a 0.0008 recall tick. P2 NN-Descent (≤5 rounds, `NeighborArray.addAndEnsureDiversity`) on RANDOM: 5 rounds, 446k edge updates, **+502% distances, recall 0.50 vs 0.63**. **P1P2 gated off.** Not run at 1M/2M. **FGIM’s advantage does not appear at 512k and was not given a 2M budget because P1 never showed promise.**

### Lazy

Light: fewer distances, lower recall (SIFT 512k clean 0.904 vs 0.933; delete=0.2 0.863 vs 0.947). Thorough (RANDOM only): more distances than SMART, still worse recall. Kept as-is from Phase 1. Not a production default.

### Non-base repair (highest value)

`IncrementalHnswGraphMerger.addReader`:

- clean graphs (`candidateVectorCount == graphSize`) → `graphReaders`
- deletes ≤ 40% → `repairableReaders`, repaired via `InitializedHnswGraphBuilder.initGraph` into a compact `OnHeapHnswGraph`
- ordinal map: `compactOrd → oldOrd → mergedOrd`
- repair is parallelizable (`repairExecutor`; `ConcurrentHnswMerger` sets it to the merge `TaskExecutor`)
- `ConcurrentHnswMerger` copies a **repaired largest** graph when the base itself was dirty (`compactOrd → mergedOrd`)

**Sequential effect:** delete=0.2 becomes a smart-merge of repaired sources. Distance count drops **~34%** at every real-data scale from 200k to 2M; recall stays within ~0.01 of SMART and always ≥ 0.92 on SIFT/HFEMB.

**Concurrent effect:** non-base repair does **not** shrink the insert set. Concurrent REPAIR distances == concurrent SMART. Closing a concurrent scaling gap would require a multi-source concurrent merger (e.g. CombinedConcurrent), which is a different algorithm.

## Conclusions

1. **Does non-base repair close the delete-scaling gap?** **Yes, sequentially.** The sequential delete penalty is “discard dirty non-base, scratch-insert live vectors.” Repair removes that penalty: **−34% distances** at SIFT 512k, SIFT 1M (2- and 5-seg), HFEMB 200k, and HFEMB 2M, with recall held. **No, concurrently** — `ConcurrentHnswMerger` still copies only the largest graph. The Phase-1 2.82× vs 6.5× concurrent curve is not the gap this change closes; with non-base-only deletes, concurrent 0.0 and 0.2 walls are already similar.

2. **Does FGIM’s advantage appear at 2M?** **No evidence it would.** P1 is more expensive than SMART at every scale we ran (including clustered HFEMB 200k we skipped P1 on, because SIFT 512k was already +23%). P2 is a cost bomb and can degrade the graph. P1P2 was gated off; 2M FGIM was not run.

3. **Is fixed-IGTM distinct from Lucene’s existing logic?** **No.** It is a subset of `updateGraph` (own-neighbor seed, no NoN). Data: 3–4.5% fewer distances, no recall gain. That is an acceptable, valuable negative result.

4. **Recommendation:** **Land non-base repair** in `IncrementalHnswGraphMerger` (40% threshold, parallel repair, compact ordinal maps). **Do not land fixed-IGTM or FGIM.** Keep Lazy experimental. Concurrent multi-source merge is a separate follow-up if 8-worker insert-set scaling is the goal.

## Recommendation

Ship **non-base segment repair**. Do not ship fixed-IGTM or FGIM. Treat Lazy as an optional quality/cost tradeoff, not a default.
