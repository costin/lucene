# HNSW Merge Research — Phase 2 · Progress Log

Updated ~hourly while the experiment runs. Final `RESULTS-hnsw-merge-phase2.md` will be added here on completion.

**Repository:** costin/lucene · **Machine:** 8 vCPU, 47 GiB RAM · **JDK:** Temurin 25 · **Started:** 2026-09-08 ~16:55 UTC

## Status
| Time (UTC) | Phase | Status |
|---|---|---|
| 16:55 | Setup | Loop launched. |
| 17:58 | A+B done; C running | **All variants implemented + `[VERIFY]` counters live.** SIFT 512k sweep (C1–C6, workers 1 & 8) complete; SIFT 1M running; HFEMB (~2M) downloaded + converted. |

## Implementation complete
- **IGTM_FIXED** — own-neighborhood warm-start (`igtm=162931/0` warm/cold verified).
- **FGIM_P1** (cross-query, `fgimCQ=162931`) and **FGIM_P1P2** (+NN-Descent, `nnD=5/446214/5`).
- **Non-base REPAIR** — deletion-bearing segments repaired instead of discarded (`repair=1/0/163907/18526` = 1 graph, 163,907 nodes repaired).
- **LAZY_LIGHT / LAZY_THOROUGH** carried forward. All HNSW tests green.
- Datasets parameterized (dataset + dim): SIFT (≤1M), HFEMB (HuggingFace real embeddings ~2M), RANDOM (sanity).

## Headline early result — non-base repair works (SIFT 512k, 2 seg, deleteRatio=0.2)

| variant | distance comps | Δ vs SMART | recall@100 | wall @1w | wall @8w |
|---|---:|---:|---:|---:|---:|
| SMART_MERGE (baseline) | 381,583,074 | — | 0.9466 | 41.3 s | 6.66 s |
| **NON_BASE_REPAIR** | **251,138,777** | **−34%** | 0.9475 | 34.0 s | **5.88 s** |

Repairing the 20%-deleted segment (163,907 nodes) instead of re-inserting from scratch cuts **~34% of distance computations** and is **faster at 8 workers with recall preserved (0.9475 vs 0.9475)**. This is the Phase 1 → Phase 2 target result. (Single-thread repair still costs ~18.5 s serial → further parallelization is the obvious follow-up.) At deleteRatio=0.0 REPAIR == SMART exactly (no repairable graphs), a good sanity check.

## Other early reads (SIFT 512k, no deletes)
- **IGTM_FIXED**: ~4% fewer distances (299.7M vs 312.9M) but **slower wall** and ~equal recall — consistent with the prediction that it is a *subset* of Lucene's existing own-neighbor + neighbor-of-neighbor seeding (full argument will be in RESULTS).
- **FGIM_P1**: more distances (+23%), slower, no recall gain. **FGIM_P1P2**: NN-Descent is very expensive and hurt recall on the smoke set — under scrutiny at scale.
- **LAZY_LIGHT**: fewer distances, lower recall (as Phase 1).

## Next
SIFT 1M sweep, then HFEMB ~2M (scale + clustered test — does FGIM's advantage finally appear, and does repair hold on clustered data?). _Next update in ~1h._
