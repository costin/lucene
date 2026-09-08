# HNSW Merge Research — Phase 2 · Progress Log (COMPLETE)

**Status: COMPLETE (2026-09-08 ~20:38 UTC).** Full write-up: [`RESULTS-hnsw-merge-phase2.md`](RESULTS-hnsw-merge-phase2.md) · raw rows: [`phase2-results.csv`](phase2-results.csv).

**Repository:** costin/lucene · **Machine:** 8 vCPU, 47 GiB RAM · **JDK:** Temurin 25 · HNSW tests: 111 passed, 2 skipped.

## Timeline
| Time (UTC) | Status |
|---|---|
| 16:55 | Loop launched. |
| 17:58 | All variants implemented + `[VERIFY]` counters live; SIFT 512k done. |
| 19:04 | SIFT 1M + HFEMB 200k (clustered) done. |
| 20:06 | HFEMB 2M underway. |
| 20:38 | **Complete.** 58 data rows; results + csv published. |

## Bottom line

**Land non-base repair** (in `IncrementalHnswGraphMerger`). **Do not land fixed-IGTM or FGIM.** Keep Lazy experimental.

### Non-base repair — consistent ~34% distance cut, recall preserved (single-thread)
| dataset | n | SMART dist | REPAIR dist | Δ | recall SMART→REPAIR |
|---|---|---:|---:|---:|---:|
| SIFT | 512k | 381.6M | 251.1M | **−34.2%** | 0.947 → 0.940 |
| SIFT | 1M / 2 seg | 762.7M | 501.6M | **−34.2%** | 0.934 → 0.924 |
| SIFT | 1M / 5 seg | 762.7M | 498.8M | **−34.6%** | 0.934 → 0.924 |
| HFEMB | 200k | 137.5M | 90.6M | **−34.1%** | 0.974 → 0.973 |
| HFEMB | **2M** | 1.623B | 1.072B | **−33.9%** | 0.970 → 0.970 |

Stable across scale, dimensionality, and clustering. **Concurrent caveat:** `ConcurrentHnswMerger` still copies only the largest graph + inserts the rest, so concurrent REPAIR == concurrent SMART — wiring repair into the concurrent path is the follow-up.

### FGIM — no advantage at any scale
FGIM_P1 costs more than SMART everywhere (SIFT 512k +23% distances, no recall gain). FGIM_P1P2 (NN-Descent) was far worse (+502% distances on RANDOM, recall collapse) → gated off; 2M not run.

### Fixed-IGTM — a subset of Lucene's existing logic
Lucene's `updateGraph` already seeds non-join nodes with their own remapped gS-neighbors **plus** a neighbor-of-neighbor expansion; fixed-IGTM keeps only the first → 3–4.5% fewer distances, no recall gain, mixed/worse wall. Negative result confirmed (full argument + code citation in `RESULTS-hnsw-merge-phase2.md`).

### Lazy — cheaper but recall down (unchanged from Phase 1).

## Recommendation
Land non-base repair (40% threshold, parallel repair executor, compact ordinal maps). Concurrent multi-source merge is a separate follow-up for 8-worker insert-set scaling.
