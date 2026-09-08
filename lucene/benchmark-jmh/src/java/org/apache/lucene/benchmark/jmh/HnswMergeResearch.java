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
import org.apache.lucene.benchmark.jmh.HnswMergeBenchmark.MergeMode;
import org.apache.lucene.benchmark.jmh.HnswMergeBenchmark.PreparedMerge;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.hnsw.HnswGraphBuilder;
import org.apache.lucene.util.hnsw.MergingHnswGraphBuilder;
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
    int mergeIters = 3;
    int rebuildIters = 2;

    for (int i = 0; i < args.length; i++) {
      switch (args[i]) {
        case "--label" -> label = args[++i];
        case "--vectorCount" -> vectorCounts = parseInts(args[++i]);
        case "--numSegments" -> numSegments = parseInts(args[++i]);
        case "--deleteRatio" -> deleteRatios = parseDoubles(args[++i]);
        case "--dataset" -> datasets = args[++i].split(",");
        case "--mergeMode" -> modes = parseModes(args[++i]);
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
        "HnswMergeResearch label=%s datasets=%s vectorCounts=%s segs=%s deletes=%s modes=%s mergeIters=%d rebuildIters=%d%n",
        label,
        String.join(",", datasets),
        join(vectorCounts),
        join(numSegments),
        join(deleteRatios),
        join(modes),
        mergeIters,
        rebuildIters);

    for (String dataset : datasets) {
      for (int vc : vectorCounts) {
        for (int segs : numSegments) {
          for (double dr : deleteRatios) {
            PreparedMerge prepared = HnswMergeBenchmark.prepare(dataset, vc, segs, dr);
            for (MergeMode mode : modes) {
              int iters = mode == MergeMode.FULL_REBUILD ? rebuildIters : mergeIters;
              measure(label, dataset, mode, vc, segs, dr, prepared, iters);
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
      PreparedMerge prepared,
      int iters)
      throws Exception {
    HnswMergeBenchmark.CountingScorerSupplier counting =
        new HnswMergeBenchmark.CountingScorerSupplier(
            DefaultFlatVectorScorer.INSTANCE.getRandomVectorScorerSupplier(
                HnswMergeBenchmark.SIMILARITY,
                FloatVectorValues.fromFloats(prepared.mergedList, HnswMergeBenchmark.DIM)));

    double[] times = new double[iters];
    long distances = -1;
    double recall = Double.NaN;
    for (int i = 0; i < iters; i++) {
      HnswGraphBuilder.randSeed = HnswMergeBenchmark.DATA_SEED;
      counting.reset();
      long start = System.nanoTime();
      OnHeapHnswGraph graph = run(mode, prepared, counting);
      times[i] = (System.nanoTime() - start) / 1_000_000.0;
      if (i == 0) {
        distances = counting.count();
        recall = HnswMergeBenchmark.computeRecall(graph, prepared.mergedList, prepared.queries);
      }
      System.out.printf(
          Locale.ROOT,
          "  iter %d/%d %s %s n=%d segs=%d del=%.1f wallMs=%.2f dist=%d%n",
          i + 1,
          iters,
          dataset,
          mode,
          vectorCount,
          numSegments,
          deleteRatio,
          times[i],
          counting.count());
    }
    double mean = 0;
    for (double t : times) {
      mean += t;
    }
    mean /= iters;
    System.out.printf(
        Locale.ROOT,
        "RESULT label=%s dataset=%s mergeMode=%s vectorCount=%d numSegments=%d deleteRatio=%.1f mergedSize=%d "
            + "wallMs=%.2f distanceComputations=%d recall@100=%.4f iters=%d%n",
        label,
        dataset,
        mode,
        vectorCount,
        numSegments,
        deleteRatio,
        prepared.mergedSize,
        mean,
        distances,
        recall,
        iters);
  }

  private static OnHeapHnswGraph run(
      MergeMode mode, PreparedMerge prepared, HnswMergeBenchmark.CountingScorerSupplier counting)
      throws Exception {
    return switch (mode) {
      case FULL_REBUILD -> {
        HnswGraphBuilder builder =
            HnswGraphBuilder.create(
                counting,
                HnswMergeBenchmark.M,
                HnswMergeBenchmark.BEAM_WIDTH,
                HnswMergeBenchmark.DATA_SEED,
                prepared.mergedSize);
        yield builder.build(prepared.mergedSize);
      }
      case SMART_MERGE -> {
        if (prepared.segmentGraphs.length == 0) {
          HnswGraphBuilder builder =
              HnswGraphBuilder.create(
                  counting,
                  HnswMergeBenchmark.M,
                  HnswMergeBenchmark.BEAM_WIDTH,
                  HnswMergeBenchmark.DATA_SEED,
                  prepared.mergedSize);
          yield builder.build(prepared.mergedSize);
        }
        MergingHnswGraphBuilder merger =
            MergingHnswGraphBuilder.fromGraphs(
                counting,
                HnswMergeBenchmark.BEAM_WIDTH,
                HnswMergeBenchmark.DATA_SEED,
                prepared.segmentGraphs,
                prepared.ordMaps,
                prepared.mergedSize,
                prepared.initializedNodes == null
                    ? null
                    : ((FixedBitSet) prepared.initializedNodes).clone());
        yield merger.build(prepared.mergedSize);
      }
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
