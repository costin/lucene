/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.lucene.benchmark.jmh;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import org.apache.lucene.benchmark.jmh.HnswMergeBenchmark.ConcurrentRun;
import org.apache.lucene.benchmark.jmh.HnswMergeBenchmark.MergeMode;
import org.apache.lucene.benchmark.jmh.HnswMergeBenchmark.PreparedMerge;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.util.hnsw.HnswGraphBuilder;
import org.apache.lucene.util.hnsw.OnHeapHnswGraph;

/**
 * Non-JMH sweep driver that shares segment-graph construction across delete ratios and merge modes.
 * Prefer this for the research tables: it still times each variant with multiple iterations, but
 * does not rebuild 200k/512k segment graphs for every JMH trial.
 *
 * <p>Usage (from repo root, after {@code :lucene:benchmark-jmh:assemble}):
 *
 * <pre>
 *   java --add-modules=jdk.incubator.vector -Xms8g -Xmx20g \
 *     -cp lucene/benchmark-jmh/build/benchmarks/'*' \
 *     org.apache.lucene.benchmark.jmh.HnswMergeResearch \
 *     --label baseline --vectorCount 20000 --dataset RANDOM --iters 2
 * </pre>
 */
public final class HnswMergeResearch {

  private HnswMergeResearch() {}

  public static void main(String[] args) throws Exception {
    String label = "run";
    int[] vectorCounts = new int[] {200000, 512000};
    int[] numSegments = new int[] {2, 5};
    double[] deleteRatios = new double[] {0.0, 0.2};
    String[] datasets = new String[] {"RANDOM", "SIFT"};
    MergeMode[] modes = MergeMode.values();
    int[] beamWidths = new int[] {HnswMergeBenchmark.BEAM_WIDTH};
    int[] numWorkers = new int[] {0}; // 0 = sequential builders (axis A)
    int mergeIters = 3;
    int rebuildIters = 2;
    int warmup = 1;

    for (int i = 0; i < args.length; i++) {
      switch (args[i]) {
        case "--label" -> label = args[++i];
        case "--vectorCount" -> vectorCounts = parseInts(args[++i]);
        case "--numSegments" -> numSegments = parseInts(args[++i]);
        case "--deleteRatio" -> deleteRatios = parseDoubles(args[++i]);
        case "--dataset" -> datasets = args[++i].split(",");
        case "--mergeMode" -> modes = parseModes(args[++i]);
        case "--beamWidth" -> beamWidths = parseInts(args[++i]);
        case "--numWorkers" -> numWorkers = parseInts(args[++i]);
        case "--warmup" -> warmup = Integer.parseInt(args[++i]);
        case "--iters" -> {
          mergeIters = Integer.parseInt(args[++i]);
          rebuildIters = Math.max(1, mergeIters);
        }
        case "--mergeIters" -> mergeIters = Integer.parseInt(args[++i]);
        case "--rebuildIters" -> rebuildIters = Integer.parseInt(args[++i]);
        default -> throw new IllegalArgumentException("unknown arg: " + args[i]);
      }
    }

    System.out.printf(
        Locale.ROOT,
        "HnswMergeResearch label=%s datasets=%s vectorCounts=%s segs=%s deletes=%s modes=%s beamWidths=%s numWorkers=%s warmup=%d mergeIters=%d rebuildIters=%d%n",
        label,
        String.join(",", datasets),
        join(vectorCounts),
        join(numSegments),
        join(deleteRatios),
        join(modes),
        join(beamWidths),
        join(numWorkers),
        warmup,
        mergeIters,
        rebuildIters);

    for (String dataset : datasets) {
      for (int vc : vectorCounts) {
        for (int segs : numSegments) {
          for (double dr : deleteRatios) {
            PreparedMerge prepared = HnswMergeBenchmark.prepare(dataset, vc, segs, dr);
            for (int beam : beamWidths) {
              HnswMergeBenchmark.mergeBeamWidth = beam;
              for (int workers : numWorkers) {
                for (MergeMode mode : modes) {
                  if (mode == MergeMode.FULL_REBUILD && workers > 0) {
                    continue;
                  }
                  int iters = mode == MergeMode.FULL_REBUILD ? rebuildIters : mergeIters;
                  measure(
                      label, dataset, mode, vc, segs, dr, beam, workers, prepared, iters, warmup);
                }
              }
            }
          }
        }
      }
    }
  }

  static void measure(
      String label,
      String dataset,
      MergeMode mode,
      int vectorCount,
      int numSegments,
      double deleteRatio,
      int beamWidth,
      int numWorkers,
      PreparedMerge prepared,
      int iters,
      int warmup)
      throws Exception {
    HnswMergeBenchmark.CountingScorerSupplier counting =
        new HnswMergeBenchmark.CountingScorerSupplier(
            DefaultFlatVectorScorer.INSTANCE.getRandomVectorScorerSupplier(
                HnswMergeBenchmark.SIMILARITY,
                FloatVectorValues.fromFloats(prepared.mergedList, HnswMergeBenchmark.DIM)));

    boolean parallel = numWorkers > 0 && mode != MergeMode.FULL_REBUILD;
    for (int w = 0; w < warmup; w++) {
      HnswGraphBuilder.randSeed = HnswMergeBenchmark.DATA_SEED;
      counting.reset();
      run(mode, prepared, counting, beamWidth, numWorkers);
      System.out.printf(
          Locale.ROOT,
          "  warmup %d/%d %s %s n=%d segs=%d del=%.1f beam=%d workers=%d dist=%d%n",
          w + 1,
          warmup,
          dataset,
          mode,
          vectorCount,
          numSegments,
          deleteRatio,
          beamWidth,
          parallel ? numWorkers : 1,
          counting.count());
    }

    double[] times = new double[iters];
    long distances = -1;
    HnswMergeBenchmark.QualityMetrics quality = null;
    ConcurrentRun lastParallel = null;
    for (int i = 0; i < iters; i++) {
      HnswGraphBuilder.randSeed = HnswMergeBenchmark.DATA_SEED;
      counting.reset();
      long start = System.nanoTime();
      Object out = run(mode, prepared, counting, beamWidth, numWorkers);
      times[i] = (System.nanoTime() - start) / 1_000_000.0;
      OnHeapHnswGraph graph;
      if (out instanceof ConcurrentRun cr) {
        lastParallel = cr;
        graph = cr.graph;
        // Keep the outer timer: initGraph + executor + build. lastWallNanos is
        // build-only (lock/contention window) and is printed in the RESULT extras.
      } else {
        graph = (OnHeapHnswGraph) out;
      }
      if (i == 0) {
        distances = counting.count();
        quality = HnswMergeBenchmark.computeQuality(graph, prepared.mergedList, prepared.queries);
      }
      System.out.printf(
          Locale.ROOT,
          "  iter %d/%d %s %s n=%d segs=%d del=%.1f beam=%d workers=%d wallMs=%.2f dist=%d%n",
          i + 1,
          iters,
          dataset,
          mode,
          vectorCount,
          numSegments,
          deleteRatio,
          beamWidth,
          parallel ? numWorkers : 1,
          times[i],
          counting.count());
    }
    double mean = 0;
    for (double t : times) {
      mean += t;
    }
    mean /= iters;
    if (lastParallel != null) {
      double blockedPct =
          lastParallel.samplerTicks == 0
              ? Double.NaN
              : 100.0
                  * lastParallel.blockedSamples
                  / (double)
                      Math.max(
                          1, lastParallel.blockedSamples + lastParallel.runnableSamples);
      System.out.printf(
          Locale.ROOT,
          "RESULT label=%s dataset=%s mergeMode=%s vectorCount=%d numSegments=%d deleteRatio=%.1f mergedSize=%d "
              + "wallMs=%.2f distanceComputations=%d recall@100=%.4f mdr=%.4f epsRecall@100=%.4f iters=%d "
              + "beamWidth=%d numWorkers=%d buildMs=%.2f writeLockWaitMs=%.3f writeLockCount=%d incomingWaitMs=%.3f incomingCount=%d "
              + "effectiveConcurrency=%.3f workerThreads=%d blockedPct=%.2f runnableSamples=%d blockedSamples=%d%n",
          label,
          dataset,
          mode,
          vectorCount,
          numSegments,
          deleteRatio,
          prepared.mergedSize,
          mean,
          distances,
          quality == null ? Double.NaN : quality.recall,
          quality == null ? Double.NaN : quality.mdr,
          quality == null ? Double.NaN : quality.epsRecall,
          iters,
          beamWidth,
          numWorkers,
          lastParallel.wallNanos / 1_000_000.0,
          lastParallel.writeLockWaitNanos / 1_000_000.0,
          lastParallel.writeLockCount,
          lastParallel.incomingWaitNanos / 1_000_000.0,
          lastParallel.incomingCount,
          lastParallel.effectiveConcurrency,
          lastParallel.workerThreads,
          blockedPct,
          lastParallel.runnableSamples,
          lastParallel.blockedSamples);
    } else {
      System.out.printf(
          Locale.ROOT,
          "RESULT label=%s dataset=%s mergeMode=%s vectorCount=%d numSegments=%d deleteRatio=%.1f mergedSize=%d "
              + "wallMs=%.2f distanceComputations=%d recall@100=%.4f mdr=%.4f epsRecall@100=%.4f iters=%d "
              + "beamWidth=%d numWorkers=%d%n",
          label,
          dataset,
          mode,
          vectorCount,
          numSegments,
          deleteRatio,
          prepared.mergedSize,
          mean,
          distances,
          quality == null ? Double.NaN : quality.recall,
          quality == null ? Double.NaN : quality.mdr,
          quality == null ? Double.NaN : quality.epsRecall,
          iters,
          beamWidth,
          1);
    }
  }

  private static Object run(
      MergeMode mode,
      PreparedMerge prepared,
      HnswMergeBenchmark.CountingScorerSupplier counting,
      int beamWidth,
      int numWorkers)
      throws Exception {
    if (numWorkers > 0 && mode != MergeMode.FULL_REBUILD) {
      return switch (mode) {
        case SMART_MERGE ->
            HnswMergeBenchmark.runConcurrentSmart(prepared, counting, beamWidth, numWorkers);
        case COMBINED ->
            HnswMergeBenchmark.runConcurrentCombined(prepared, counting, beamWidth, numWorkers);
        case FULL_REBUILD -> throw new IllegalStateException();
      };
    }
    return switch (mode) {
      case FULL_REBUILD ->
          HnswGraphBuilder.create(
                  counting, HnswMergeBenchmark.M, beamWidth, HnswMergeBenchmark.DATA_SEED, prepared.mergedSize)
              .build(prepared.mergedSize);
      case SMART_MERGE ->
          HnswMergeBenchmark.runSmartSequential(prepared, counting, beamWidth);
      case COMBINED -> HnswMergeBenchmark.runCombinedSequential(prepared, counting, beamWidth);
    };
  }

  private static int[] parseInts(String csv) {
    String[] p = csv.split(",");
    int[] out = new int[p.length];
    for (int i = 0; i < p.length; i++) {
      out[i] = Integer.parseInt(p[i].trim());
    }
    return out;
  }

  private static double[] parseDoubles(String csv) {
    String[] p = csv.split(",");
    double[] out = new double[p.length];
    for (int i = 0; i < p.length; i++) {
      out[i] = Double.parseDouble(p[i].trim());
    }
    return out;
  }

  private static MergeMode[] parseModes(String csv) {
    String[] p = csv.split(",");
    MergeMode[] out = new MergeMode[p.length];
    for (int i = 0; i < p.length; i++) {
      out[i] = MergeMode.valueOf(p[i].trim());
    }
    return out;
  }

  private static String join(int[] xs) {
    List<String> s = new ArrayList<>();
    for (int x : xs) {
      s.add(Integer.toString(x));
    }
    return String.join(",", s);
  }

  private static String join(double[] xs) {
    List<String> s = new ArrayList<>();
    for (double x : xs) {
      s.add(Double.toString(x));
    }
    return String.join(",", s);
  }

  private static String join(MergeMode[] xs) {
    List<String> s = new ArrayList<>();
    for (MergeMode x : xs) {
      s.add(x.name());
    }
    return String.join(",", s);
  }
}
