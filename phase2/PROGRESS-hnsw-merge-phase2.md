# HNSW Merge Research — Phase 2 · Progress Log

Updated ~hourly while the experiment runs. Final `RESULTS-hnsw-merge-phase2.md` will be added here on completion.

**Repository:** costin/lucene · **Machine:** 8 vCPU, 47 GiB RAM · **JDK:** Temurin 25 · **Started:** 2026-09-08 ~16:55 UTC

## Status
| Time (UTC) | Phase | Status |
|---|---|---|
| 16:55 | Setup | Loop launched. |
| 17:58 | A+B done; C running | All variants implemented + `[VERIFY]` counters live. SIFT 512k done. |
| 19:04 | C running | SIFT 1M (2 & 5 seg) + HFEMB 200k (clustered) done. |
| 20:06 | C running | **HFEMB 2M** underway (deleteRatio=0.0 done; 0.2 + FGIM-at-2M pending). 2M merge ≈ 6.4 min single-thread / 2.2 min @ 8w. |

## Headline: non-base repair — consistent ~34% distance cut (single-thread), recall preserved

| config | baseline dist | repair dist | Δ | recall (base→repair) |
|---|---:|---:|---:|---:|
| SIFT 512k, 2 seg, 20% del | 381.6M | 251.1M | −34% | 0.9466 → 0.9475 |
| SIFT 1M, 5 seg, 20% del | 762.7M | 498.8M | −35% | 0.9343 → 0.9238 |
| HFEMB 200k (clustered), 20% del | 137.5M | 90.6M | −34% | 0.9742 → 0.9728 |

Holds across scale and on clustered real embeddings, with recall essentially preserved.

## HFEMB 2M scale point (deleteRatio=0.0 so far)
| variant | workers | dist | wall | recall@100 |
|---|---:|---:|---:|---:|
| SMART_MERGE | 1 | 1.35B | 384.7 s | 0.9604 |
| REPAIR | 1 | 1.35B | 380.3 s | 0.9604 |
| SMART_MERGE | 8 | 2.03B | 129.5 s | 0.9703 |
| REPAIR | 8 | 2.04B | 122.5 s | 0.9705 |

At 0% deletes REPAIR == baseline (expected). The **2M / 20% deletes** point — where repair should help — and **FGIM at 2M** (the "does cross-query pay off at scale" test) are still running.

## Open gap (unchanged): repair not yet wired into the concurrent path
At 8 workers REPAIR ≈ baseline; the non-base repair engages only on the single-threaded merger, while `ConcurrentHnswMerger` falls back to baseline (RESULTS notes "concurrent insert set unchanged"). **Wiring repair into the concurrent merger is the key follow-up** to carry the −34% into the parallel path.

## Other reads
- **IGTM_FIXED** ≈ baseline everywhere (a subset of Lucene's existing own-neighbor + neighbor-of-neighbor seeding; argument documented in RESULTS).
- REPAIR == baseline exactly at 0% deletes (sanity).

## Next
Finish HFEMB 2M (deletes + FGIM-at-scale), Lazy at scale, then Phase D. _Next update in ~1h._
