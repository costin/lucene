# HNSW Merge Research — Phase 2 · Progress Log

Updated ~hourly while the experiment runs. Final `RESULTS-hnsw-merge-phase2.md` will be added here on completion.

**Repository:** costin/lucene · **Machine:** 8 vCPU, 47 GiB RAM · **JDK:** Temurin 25 · **Started:** 2026-09-08 ~16:55 UTC

## Status
| Time (UTC) | Phase | Status |
|---|---|---|
| 16:55 | Setup | Loop launched. |
| 17:58 | A+B done; C running | All variants implemented + `[VERIFY]` counters live. SIFT 512k done. |
| 19:04 | C running | SIFT **1M** (2 & 5 seg) done; **HFEMB 200k** (clustered real embeddings) done; **HFEMB 2M** running. 51 result rows. |

## Non-base repair holds across scale AND clustered data (single-thread)

**SIFT 1M, 5 segments, deleteRatio=0.2** (multi-segment deletes, ES-realistic):
| variant | dist comps | wall @1w | recall@100 |
|---|---:|---:|---:|
| SMART_MERGE | 762.7M | 93.8 s | 0.9343 |
| **NON_BASE_REPAIR** | **498.8M (−35%)** | **62.2 s (−34%)** | 0.9238 |

**HFEMB 200k, deleteRatio=0.2** (clustered real embeddings — repair-quality test):
| variant | dist comps | wall @1w | recall@100 |
|---|---:|---:|---:|
| SMART_MERGE | 137.5M | 20.0 s | 0.9742 |
| **NON_BASE_REPAIR** | **90.6M (−34%)** | **14.6 s (−27%)** | 0.9728 |

Repair delivers a consistent **~34–35% distance reduction with recall preserved** on both SIFT and clustered embeddings — the concern that repair might degrade on clustered data does **not** materialize.

## Important gap found: repair not yet wired into the concurrent path
At **8 workers**, REPAIR ≈ SMART_MERGE (e.g. SIFT 1M/5seg/0.2: both ~763M dist, ~34 s). The non-base repair currently engages only on the single-threaded `IncrementalHnswGraphMerger` path; `ConcurrentHnswMerger` falls back to baseline. **Wiring repair into the concurrent merger is the key follow-up** — it should carry the single-thread −34% distance win into the parallel path (and directly attack the Phase-1 delete-scaling gap).

## Other reads
- **IGTM_FIXED**: consistently ≈ baseline (SIFT 1M/2/0.0: 598M vs 623M dist, ~equal recall, ~equal/slower wall) — confirms it is a subset of Lucene's existing own-neighbor + neighbor-of-neighbor seeding. Full argument in RESULTS.
- **REPAIR @ deleteRatio=0.0 == SMART_MERGE exactly** (no repairable graphs) — sanity holds.
- HFEMB recall ~0.974 (real embeddings, discriminative).

## Next
HFEMB **2M** scale test (does FGIM's cross-query advantage finally appear at 2M? does repair hold at 2M?), then Phase D write-up. _Next update in ~1h._
