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

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.util.NamedThreadFactory;
import org.apache.lucene.util.hnsw.CombinedConcurrentHnswGraphBuilder;
import org.apache.lucene.util.hnsw.CombinedHnswGraphBuilder;
import org.apache.lucene.util.hnsw.HnswConcurrentMergeBuilder;
import org.apache.lucene.util.hnsw.HnswGraphBuilder;
import org.apache.lucene.util.hnsw.HnswGraphSearcher;
import org.apache.lucene.util.hnsw.InitializedHnswGraphBuilder;
import org.apache.lucene.util.hnsw.MergingHnswGraphBuilder;
import org.apache.lucene.util.hnsw.NeighborQueue;
import org.apache.lucene.util.hnsw.OnHeapHnswGraph;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.apache.lucene.util.hnsw.UpdateableRandomVectorScorer;
import org.openjdk.jmh.annotations.AuxCounters;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

/**
 * Benchmarks HNSW graph construction via (a) a full rebuild with {@link HnswGraphBuilder} and (b)
 * the production smart-merge path ({@link MergingHnswGraphBuilder}, the algorithm behind {@code
 * IncrementalHnswGraphMerger}).
 *
 * <p>Reported per configuration:
 *
 * <ul>
 *   <li><b>wall-clock time</b> of the rebuild or merge (JMH {@code SingleShotTime}, plus a
 *       multi-iteration mean printed by {@link #summary()});
 *   <li><b>distance computations</b> during that build, counted by wrapping {@link
 *       RandomVectorScorerSupplier} ({@code score} + {@code bulkScore});
 *   <li><b>recall@100</b> versus brute-force exact kNN on 100 deterministic queries (ordinal
 *       overlap);
 *   <li><b>mean distance ratio (MDR)</b> and <b>ε-recall@100</b> ({@code ε=0.01}) on raw L2, so
 *       near-ties on concentrated RANDOM data are not mistaken for bad answers.
 * </ul>
 *
 * <h2>Deletion model vs {@code IncrementalHnswGraphMerger}</h2>
 *
 * <p>Production merge reuse is conservative:
 *
 * <ul>
 *   <li>A graph is added as a merge <em>source</em> only if it has <em>no</em> deleted vectors.
 *   <li>The largest graph may still be chosen as the <em>base</em> if its delete rate is {@code <=
 *       40%}.
 *   <li>Live vectors from excluded graphs are inserted from scratch ({@code initializedNodes}).
 * </ul>
 *
 * This harness applies {@code deleteRatio} only to non-base segments (the research plan). At {@code
 * 0.2}, every non-base segment almost surely has at least one delete, so only the base graph is
 * reused — matching production. The base itself is never deleted here, so we do not exercise {@code
 * InitializedHnswGraphBuilder}'s hole-repair path. Deleted source ordinals are mapped to {@code -1}
 * and must be skipped by the merger.
 */
@BenchmarkMode(Mode.SingleShotTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 1)
@Measurement(iterations = 3)
@Fork(
    value = 1,
    jvmArgsAppend = {"-Xmx20g", "-Xms8g", "--add-modules=jdk.incubator.vector"})
@State(Scope.Benchmark)
public class HnswMergeBenchmark {

  static final int DIM = 128;
  static final int M = 16;
  static final int BEAM_WIDTH = 100;
  static final long DATA_SEED = 42L;
  static final int RECALL_QUERIES = 100;
  static final int RECALL_K = 100;

  /** Relative-distance slack for ε-recall@100. */
  static final double EPS_RECALL = 0.01;

  static final VectorSimilarityFunction SIMILARITY = VectorSimilarityFunction.EUCLIDEAN;

  /**
   * How the merged graph is produced. Paper branches keep these two modes; SMART_MERGE is the
   * paper.
   */
  public enum MergeMode {
    FULL_REBUILD,
    SMART_MERGE,
    COMBINED
  }

  /** Merge construction beam width. Segment graphs stay at {@link #BEAM_WIDTH}. */
  static int mergeBeamWidth = BEAM_WIDTH;

  @Param({"200000", "512000"})
  public int vectorCount;

  @Param({"2", "5"})
  public int numSegments;

  @Param({"0.0", "0.2"})
  public double deleteRatio;

  @Param({"RANDOM", "SIFT"})
  public String dataset;

  @Param({"FULL_REBUILD", "SMART_MERGE"})
  public String mergeMode;

  private OnHeapHnswGraph[] segmentGraphs;
  private int[][] ordMaps;
  private BitSet initializedNodes;
  private int mergedSize;
  private List<float[]> mergedList;
  private List<float[]> queryList;
  private FloatVectorValues mergedValues;
  private CountingScorerSupplier countingSupplier;

  private long refDistanceComputations;
  private double refRecall;
  private double refMdr;
  private double refEpsRecall;
  private double refWallMillis;
  private double timedMeanMillis;
  private int timedIters;

  // Segment graphs depend only on (dataset, vectorCount, numSegments). Reuse across
  // deleteRatio and mergeMode in the same JVM (JMH cartesian product + research driver).
  private static final Object CACHE_LOCK = new Object();
  private static String cacheKey;
  private static float[][] cacheVectors;
  private static float[][] cacheQueries;
  private static OnHeapHnswGraph[] cacheGraphs;

  @Setup(Level.Trial)
  public void setup() throws IOException {
    PreparedMerge prepared = prepare(dataset, vectorCount, numSegments, deleteRatio);
    segmentGraphs = prepared.segmentGraphs;
    ordMaps = prepared.ordMaps;
    initializedNodes = prepared.initializedNodes;
    mergedSize = prepared.mergedSize;
    mergedList = prepared.mergedList;
    queryList = prepared.queries;
    mergedValues = FloatVectorValues.fromFloats(mergedList, DIM);
    countingSupplier =
        new CountingScorerSupplier(
            DefaultFlatVectorScorer.INSTANCE.getRandomVectorScorerSupplier(
                SIMILARITY, mergedValues));

    RunResult ref = runOnce(MergeMode.valueOf(mergeMode));
    refWallMillis = ref.millis;
    refDistanceComputations = ref.distances;
    QualityMetrics quality = computeQuality(ref.graph, mergedList, queryList);
    refRecall = quality.recall;
    refMdr = quality.mdr;
    refEpsRecall = quality.epsRecall;
    timedMeanMillis = refWallMillis;
    timedIters = 1;
  }

  @Benchmark
  public void buildOrMerge(Blackhole bh, Metrics metrics) throws IOException {
    RunResult result = runOnce(MergeMode.valueOf(mergeMode));
    metrics.distanceComputations = result.distances;
    metrics.recallBasisPoints = Math.round(refRecall * 10000.0);
    bh.consume(result.graph);
  }

  @TearDown(Level.Trial)
  public void summary() {
    System.out.printf(
        Locale.ROOT,
        "RESULT dataset=%s mergeMode=%s vectorCount=%d numSegments=%d deleteRatio=%.1f mergedSize=%d "
            + "wallMs=%.2f distanceComputations=%d recall@100=%.4f mdr=%.4f epsRecall@100=%.4f%n",
        dataset,
        mergeMode,
        vectorCount,
        numSegments,
        deleteRatio,
        mergedSize,
        refWallMillis,
        refDistanceComputations,
        refRecall,
        refMdr,
        refEpsRecall);
  }

  RunResult runOnce(MergeMode mode) throws IOException {
    HnswGraphBuilder.randSeed = DATA_SEED;
    countingSupplier.reset();
    long start = System.nanoTime();
    OnHeapHnswGraph graph =
        switch (mode) {
          case FULL_REBUILD -> doFullRebuild();
          case SMART_MERGE -> doMerge();
          case COMBINED -> doCombined();
        };
    double millis = (System.nanoTime() - start) / 1_000_000.0;
    return new RunResult(graph, millis, countingSupplier.count());
  }

  private OnHeapHnswGraph doFullRebuild() throws IOException {
    HnswGraphBuilder builder =
        HnswGraphBuilder.create(countingSupplier, M, mergeBeamWidth, DATA_SEED, mergedSize);
    return builder.build(mergedSize);
  }

  private OnHeapHnswGraph doMerge() throws IOException {
    if (segmentGraphs.length == 0) {
      return doFullRebuild();
    }
    BitSet init = initializedNodes == null ? null : ((FixedBitSet) initializedNodes).clone();
    MergingHnswGraphBuilder merger =
        MergingHnswGraphBuilder.fromGraphs(
            countingSupplier, mergeBeamWidth, DATA_SEED, segmentGraphs, ordMaps, mergedSize, init);
    return merger.build(mergedSize);
  }

  private OnHeapHnswGraph doCombined() throws IOException {
    if (segmentGraphs.length == 0) {
      return doFullRebuild();
    }
    BitSet init = initializedNodes == null ? null : ((FixedBitSet) initializedNodes).clone();
    CombinedHnswGraphBuilder merger =
        CombinedHnswGraphBuilder.fromGraphs(
            countingSupplier, mergeBeamWidth, DATA_SEED, segmentGraphs, ordMaps, mergedSize, init);
    return merger.build(mergedSize);
  }

  /**
   * Sequential SMART MERGE ({@link MergingHnswGraphBuilder}). Used for the single-thread quality
   * axis.
   */
  static OnHeapHnswGraph runSmartSequential(
      PreparedMerge prepared, CountingScorerSupplier counting, int beamWidth) throws IOException {
    if (prepared.segmentGraphs.length == 0) {
      return HnswGraphBuilder.create(counting, M, beamWidth, DATA_SEED, prepared.mergedSize)
          .build(prepared.mergedSize);
    }
    BitSet init =
        prepared.initializedNodes == null
            ? null
            : ((FixedBitSet) prepared.initializedNodes).clone();
    return MergingHnswGraphBuilder.fromGraphs(
            counting,
            beamWidth,
            DATA_SEED,
            prepared.segmentGraphs,
            prepared.ordMaps,
            prepared.mergedSize,
            init)
        .build(prepared.mergedSize);
  }

  /** Sequential combined (HNSW-Merger forward + incoming + prune). */
  static OnHeapHnswGraph runCombinedSequential(
      PreparedMerge prepared, CountingScorerSupplier counting, int beamWidth) throws IOException {
    if (prepared.segmentGraphs.length == 0) {
      return HnswGraphBuilder.create(counting, M, beamWidth, DATA_SEED, prepared.mergedSize)
          .build(prepared.mergedSize);
    }
    BitSet init =
        prepared.initializedNodes == null
            ? null
            : ((FixedBitSet) prepared.initializedNodes).clone();
    return CombinedHnswGraphBuilder.fromGraphs(
            counting,
            beamWidth,
            DATA_SEED,
            prepared.segmentGraphs,
            prepared.ordMaps,
            prepared.mergedSize,
            init)
        .build(prepared.mergedSize);
  }

  /**
   * Production concurrent merge: {@link HnswConcurrentMergeBuilder} — the builder {@link
   * org.apache.lucene.util.hnsw.ConcurrentHnswMerger} installs. Pool size is {@code numWorkers} so
   * {@link TaskExecutor} (N−1 on the pool + 1 on the caller) is not starved.
   */
  static ConcurrentRun runConcurrentSmart(
      PreparedMerge prepared, CountingScorerSupplier counting, int beamWidth, int numWorkers)
      throws Exception {
    return runConcurrent(prepared, counting, beamWidth, numWorkers, false);
  }

  /** Combined concurrent merge: forward-only + incoming + prune, same insert set as SMART. */
  static ConcurrentRun runConcurrentCombined(
      PreparedMerge prepared, CountingScorerSupplier counting, int beamWidth, int numWorkers)
      throws Exception {
    return runConcurrent(prepared, counting, beamWidth, numWorkers, true);
  }

  private static ConcurrentRun runConcurrent(
      PreparedMerge prepared,
      CountingScorerSupplier counting,
      int beamWidth,
      int numWorkers,
      boolean combined)
      throws Exception {
    if (prepared.segmentGraphs.length == 0) {
      long t0 = System.nanoTime();
      OnHeapHnswGraph g =
          HnswGraphBuilder.create(counting, M, beamWidth, DATA_SEED, prepared.mergedSize)
              .build(prepared.mergedSize);
      return new ConcurrentRun(g, System.nanoTime() - t0, 0, 0, 0, 0, 0, 1, Set.of(), 0, 0, 0);
    }
    OnHeapHnswGraph graph =
        InitializedHnswGraphBuilder.initGraph(
            prepared.segmentGraphs[0],
            prepared.ordMaps[0],
            prepared.mergedSize,
            beamWidth,
            counting);
    FixedBitSet init = baseInitialized(prepared);
    ExecutorService exec =
        Executors.newFixedThreadPool(numWorkers, new NamedThreadFactory("hnsw-merge"));
    TaskExecutor taskExecutor = new TaskExecutor(exec);
    ThreadSampler sampler = new ThreadSampler("hnsw-merge");
    Thread caller = Thread.currentThread();
    String oldName = caller.getName();
    caller.setName("hnsw-merge-caller");
    try {
      if (combined) {
        CombinedConcurrentHnswGraphBuilder builder =
            new CombinedConcurrentHnswGraphBuilder(
                taskExecutor,
                numWorkers,
                counting,
                beamWidth,
                graph,
                init,
                prepared.segmentGraphs,
                prepared.ordMaps);
        OnHeapHnswGraph out = builder.build(prepared.mergedSize);
        sampler.stop();
        return new ConcurrentRun(
            out,
            builder.lastWallNanos(),
            builder.writeLockWaitNanos(),
            builder.writeLockCount(),
            builder.incomingWaitNanos(),
            builder.incomingCount(),
            builder.lastEffectiveConcurrency(),
            builder.workerThreadIds().size(),
            builder.workerThreadIds(),
            sampler.blockedSamples(),
            sampler.runnableSamples(),
            sampler.totalSamples());
      } else {
        HnswConcurrentMergeBuilder builder =
            new HnswConcurrentMergeBuilder(
                taskExecutor, numWorkers, counting, beamWidth, graph, init);
        OnHeapHnswGraph out = builder.build(prepared.mergedSize);
        sampler.stop();
        return new ConcurrentRun(
            out,
            builder.lastWallNanos(),
            builder.writeLockWaitNanos(),
            builder.writeLockCount(),
            0,
            0,
            builder.lastEffectiveConcurrency(),
            builder.workerThreadIds().size(),
            builder.workerThreadIds(),
            sampler.blockedSamples(),
            sampler.runnableSamples(),
            sampler.totalSamples());
      }
    } finally {
      caller.setName(oldName);
      sampler.close();
      exec.shutdown();
      if (exec.awaitTermination(2, TimeUnit.HOURS) == false) {
        exec.shutdownNow();
      }
    }
  }

  static FixedBitSet baseInitialized(PreparedMerge prepared) {
    FixedBitSet bits = new FixedBitSet(Math.max(1, prepared.mergedSize));
    if (prepared.ordMaps.length == 0) {
      return bits;
    }
    for (int o : prepared.ordMaps[0]) {
      if (o >= 0 && o < prepared.mergedSize) {
        bits.set(o);
      }
    }
    return bits;
  }

  /**
   * Samples thread states whose names start with {@code namePrefix}. Used to show whether the
   * baseline's workers are actually RUNNABLE or sitting BLOCKED on the reverse-link write lock.
   */
  static final class ThreadSampler implements AutoCloseable {
    private final String namePrefix;
    private final AtomicInteger blocked = new AtomicInteger();
    private final AtomicInteger runnable = new AtomicInteger();
    private final AtomicInteger samples = new AtomicInteger();
    private final ExecutorService probe =
        Executors.newSingleThreadExecutor(new NamedThreadFactory("hnsw-sampler"));
    private volatile boolean running = true;

    ThreadSampler(String namePrefix) {
      this.namePrefix = namePrefix;
      probe.execute(this::loop);
    }

    private void loop() {
      while (running) {
        int b = 0;
        int r = 0;
        int n = 0;
        for (Thread t : Thread.getAllStackTraces().keySet()) {
          if (t.getName().startsWith(namePrefix) == false) {
            continue;
          }
          n++;
          switch (t.getState()) {
            case BLOCKED -> b++;
            case RUNNABLE -> r++;
            default -> {}
          }
        }
        if (n > 0) {
          blocked.addAndGet(b);
          runnable.addAndGet(r);
          samples.incrementAndGet();
        }
        try {
          Thread.sleep(20);
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          return;
        }
      }
    }

    void stop() {
      running = false;
    }

    int blockedSamples() {
      return blocked.get();
    }

    int runnableSamples() {
      return runnable.get();
    }

    int totalSamples() {
      return samples.get();
    }

    @Override
    public void close() {
      running = false;
      probe.shutdownNow();
    }
  }

  static final class ConcurrentRun {
    final OnHeapHnswGraph graph;
    final long wallNanos;
    final long writeLockWaitNanos;
    final long writeLockCount;
    final long incomingWaitNanos;
    final long incomingCount;
    final double effectiveConcurrency;
    final int workerThreads;
    final Set<Long> workerThreadIds;
    final int blockedSamples;
    final int runnableSamples;
    final int samplerTicks;

    ConcurrentRun(
        OnHeapHnswGraph graph,
        long wallNanos,
        long writeLockWaitNanos,
        long writeLockCount,
        long incomingWaitNanos,
        long incomingCount,
        double effectiveConcurrency,
        int workerThreads,
        Set<Long> workerThreadIds,
        int blockedSamples,
        int runnableSamples,
        int samplerTicks) {
      this.graph = graph;
      this.wallNanos = wallNanos;
      this.writeLockWaitNanos = writeLockWaitNanos;
      this.writeLockCount = writeLockCount;
      this.incomingWaitNanos = incomingWaitNanos;
      this.incomingCount = incomingCount;
      this.effectiveConcurrency = effectiveConcurrency;
      this.workerThreads = workerThreads;
      this.workerThreadIds = workerThreadIds;
      this.blockedSamples = blockedSamples;
      this.runnableSamples = runnableSamples;
      this.samplerTicks = samplerTicks;
    }
  }

  static PreparedMerge prepare(String dataset, int vectorCount, int numSegments, double deleteRatio)
      throws IOException {
    HnswGraphBuilder.randSeed = DATA_SEED;
    CachedSegments cached = loadOrBuildSegments(dataset, vectorCount, numSegments);
    int[] segStart = partitionStarts(vectorCount, numSegments);

    // Deletions: non-base only. A non-base segment with any delete is excluded from the
    // source-graph set (IncrementalHnswGraphMerger). Live vectors become uninitialized nodes.
    Random deleteRandom = new Random(DATA_SEED + 1);
    boolean[][] deleted = new boolean[numSegments][];
    boolean[] segmentHasDeletes = new boolean[numSegments];
    for (int s = 0; s < numSegments; s++) {
      int segSize = segStart[s + 1] - segStart[s];
      deleted[s] = new boolean[segSize];
      if (s == 0 || deleteRatio <= 0.0) {
        continue;
      }
      for (int i = 0; i < segSize; i++) {
        if (deleteRandom.nextDouble() < deleteRatio) {
          deleted[s][i] = true;
          segmentHasDeletes[s] = true;
        }
      }
    }

    List<float[]> mergedList = new ArrayList<>(vectorCount);
    int[][] maps = new int[numSegments][];
    List<OnHeapHnswGraph> sourceGraphs = new ArrayList<>();
    List<int[]> sourceMaps = new ArrayList<>();
    int nextOrd = 0;
    for (int s = 0; s < numSegments; s++) {
      int from = segStart[s];
      int segSize = segStart[s + 1] - from;
      maps[s] = new int[segSize];
      for (int i = 0; i < segSize; i++) {
        if (deleted[s][i]) {
          maps[s][i] = -1;
        } else {
          maps[s][i] = nextOrd++;
          mergedList.add(cached.vectors[from + i]);
        }
      }
      if (s == 0 || !segmentHasDeletes[s]) {
        sourceGraphs.add(cached.graphs[s]);
        sourceMaps.add(maps[s]);
      }
    }

    boolean allNodesFromSources =
        sourceGraphs.size() == numSegments && !anyDeletes(segmentHasDeletes);
    BitSet initializedNodes;
    if (allNodesFromSources) {
      initializedNodes = null;
    } else {
      FixedBitSet bits = new FixedBitSet(nextOrd);
      for (int[] map : sourceMaps) {
        for (int newOrd : map) {
          if (newOrd >= 0) {
            bits.set(newOrd);
          }
        }
      }
      initializedNodes = bits;
    }

    List<float[]> queries = new ArrayList<>(cached.queries.length);
    queries.addAll(Arrays.asList(cached.queries));
    return new PreparedMerge(
        sourceGraphs.toArray(new OnHeapHnswGraph[0]),
        sourceMaps.toArray(new int[0][]),
        initializedNodes,
        nextOrd,
        mergedList,
        queries);
  }

  private static CachedSegments loadOrBuildSegments(
      String dataset, int vectorCount, int numSegments) throws IOException {
    String key = dataset + "|" + vectorCount + "|" + numSegments;
    synchronized (CACHE_LOCK) {
      if (key.equals(cacheKey) && cacheGraphs != null && cacheGraphs.length == numSegments) {
        return new CachedSegments(cacheVectors, cacheQueries, cacheGraphs);
      }
      System.out.printf(
          Locale.ROOT,
          "CACHE miss dataset=%s vectorCount=%d numSegments=%d — building segment graphs%n",
          dataset,
          vectorCount,
          numSegments);
      float[][] vectors = loadVectors(dataset, vectorCount);
      float[][] queries = loadQueries(dataset);
      int[] segStart = partitionStarts(vectorCount, numSegments);
      OnHeapHnswGraph[] graphs = new OnHeapHnswGraph[numSegments];
      for (int s = 0; s < numSegments; s++) {
        int from = segStart[s];
        int to = segStart[s + 1];
        System.out.printf(
            Locale.ROOT, "  building segment %d/%d size=%d%n", s + 1, numSegments, to - from);
        List<float[]> segList = new ArrayList<>(to - from);
        for (int i = from; i < to; i++) {
          segList.add(vectors[i]);
        }
        FloatVectorValues segValues = FloatVectorValues.fromFloats(segList, DIM);
        RandomVectorScorerSupplier segSupplier =
            DefaultFlatVectorScorer.INSTANCE.getRandomVectorScorerSupplier(SIMILARITY, segValues);
        HnswGraphBuilder builder =
            HnswGraphBuilder.create(segSupplier, M, BEAM_WIDTH, DATA_SEED, to - from);
        graphs[s] = builder.build(to - from);
      }
      cacheKey = key;
      cacheVectors = vectors;
      cacheQueries = queries;
      cacheGraphs = graphs;
      return new CachedSegments(vectors, queries, graphs);
    }
  }

  /** Segment 0 gets ~60%; the remainder is split equally (research plan). */
  static int[] partitionStarts(int vectorCount, int numSegments) {
    int[] segStart = new int[numSegments + 1];
    int baseSize = numSegments == 1 ? vectorCount : (int) Math.round(vectorCount * 0.6);
    segStart[0] = 0;
    segStart[1] = Math.min(baseSize, vectorCount);
    int remaining = vectorCount - segStart[1];
    for (int s = 1; s < numSegments; s++) {
      int take = remaining / (numSegments - s);
      remaining -= take;
      segStart[s + 1] = segStart[s] + take;
    }
    segStart[numSegments] = vectorCount;
    return segStart;
  }

  static float[][] loadVectors(String dataset, int vectorCount) throws IOException {
    if ("SIFT".equals(dataset)) {
      float[][] all = readFvecs(siftBasePath(), vectorCount, DIM);
      if (all.length < vectorCount) {
        throw new IOException(
            "SIFT base has only "
                + all.length
                + " vectors; need "
                + vectorCount
                + ". Expected "
                + siftBasePath());
      }
      return all;
    }
    Random random = new Random(DATA_SEED);
    float[][] vectors = new float[vectorCount][DIM];
    for (int i = 0; i < vectorCount; i++) {
      for (int d = 0; d < DIM; d++) {
        vectors[i][d] = random.nextFloat();
      }
    }
    return vectors;
  }

  static float[][] loadQueries(String dataset) throws IOException {
    if ("SIFT".equals(dataset)) {
      float[][] q = readFvecs(siftQueryPath(), RECALL_QUERIES, DIM);
      if (q.length >= RECALL_QUERIES) {
        return q;
      }
    }
    Random queryRandom = new Random(DATA_SEED + 2);
    float[][] queries = new float[RECALL_QUERIES][DIM];
    for (int i = 0; i < RECALL_QUERIES; i++) {
      for (int d = 0; d < DIM; d++) {
        queries[i][d] = queryRandom.nextFloat();
      }
    }
    return queries;
  }

  static Path siftDir() {
    String env = System.getenv("HNSW_SIFT_DIR");
    if (env != null && !env.isBlank()) {
      return Path.of(env);
    }
    Path cwd = Path.of(System.getProperty("user.dir", "."));
    Path direct = cwd.resolve("hnsw-bench-data").resolve("sift");
    if (Files.isDirectory(direct)) {
      return direct;
    }
    Path parent = cwd.getParent();
    if (parent != null) {
      Path up = parent.resolve("hnsw-bench-data").resolve("sift");
      if (Files.isDirectory(up)) {
        return up;
      }
    }
    return direct;
  }

  static Path siftBasePath() {
    return siftDir().resolve("sift_base.fvecs");
  }

  static Path siftQueryPath() {
    return siftDir().resolve("sift_query.fvecs");
  }

  /**
   * Reads up to {@code maxCount} little-endian texmex {@code .fvecs} vectors (int32 dim + dim
   * floats per record).
   */
  static float[][] readFvecs(Path path, int maxCount, int expectedDim) throws IOException {
    if (!Files.isRegularFile(path)) {
      throw new IOException("Missing SIFT file: " + path);
    }
    try (FileChannel ch = FileChannel.open(path)) {
      long size = ch.size();
      if (size > Integer.MAX_VALUE) {
        throw new IOException("fvecs too large: " + path);
      }
      ByteBuffer buf = ByteBuffer.allocate((int) size).order(ByteOrder.LITTLE_ENDIAN);
      while (buf.hasRemaining()) {
        if (ch.read(buf) < 0) {
          break;
        }
      }
      buf.flip();
      List<float[]> out = new ArrayList<>(Math.min(maxCount, 1024));
      while (buf.remaining() >= 4 && out.size() < maxCount) {
        int dim = buf.getInt();
        if (dim != expectedDim) {
          throw new IOException("expected dim=" + expectedDim + " but file has dim=" + dim);
        }
        if (buf.remaining() < dim * Float.BYTES) {
          throw new IOException("truncated fvecs at vector " + out.size());
        }
        float[] v = new float[dim];
        buf.asFloatBuffer().get(v);
        buf.position(buf.position() + dim * Float.BYTES);
        out.add(v);
      }
      return out.toArray(new float[0][]);
    }
  }

  /**
   * Answer-quality metrics versus brute-force exact kNN. {@code recall} is ordinal overlap. {@code
   * mdr} and {@code epsRecall} use <em>raw Euclidean L2</em> (not Lucene's {@code EUCLIDEAN}
   * similarity transform).
   */
  static final class QualityMetrics {
    final double recall;
    final double mdr;
    final double epsRecall;

    QualityMetrics(double recall, double mdr, double epsRecall) {
      this.recall = recall;
      this.mdr = mdr;
      this.epsRecall = epsRecall;
    }
  }

  static double computeRecall(
      OnHeapHnswGraph graph, List<float[]> mergedList, List<float[]> queries) throws IOException {
    return computeQuality(graph, mergedList, queries).recall;
  }

  static QualityMetrics computeQuality(
      OnHeapHnswGraph graph, List<float[]> mergedList, List<float[]> queries) throws IOException {
    FloatVectorValues values = FloatVectorValues.fromFloats(mergedList, DIM);
    int size = mergedList.size();
    int k = Math.min(RECALL_K, size);
    long totalOverlap = 0;
    double mdrSum = 0.0;
    long mdrTerms = 0;
    long epsHits = 0;
    long returned = 0;
    for (float[] query : queries) {
      RandomVectorScorer queryScorer =
          DefaultFlatVectorScorer.INSTANCE.getRandomVectorScorer(SIMILARITY, values, query);
      KnnCollector nn = HnswGraphSearcher.search(queryScorer, k, graph, null, Integer.MAX_VALUE);
      TopDocs topDocs = nn.topDocs();

      NeighborQueue expected = new NeighborQueue(k, false);
      for (int ord = 0; ord < size; ord++) {
        float score = SIMILARITY.compare(query, mergedList.get(ord));
        expected.add(ord, score);
        if (expected.size() > k) {
          expected.pop();
        }
      }
      int[] actual = new int[topDocs.scoreDocs.length];
      for (int i = 0; i < actual.length; i++) {
        actual[i] = topDocs.scoreDocs[i].doc;
      }
      int[] exactOrds = expected.nodes();
      totalOverlap += overlap(actual, exactOrds);

      double[] approxDist = new double[actual.length];
      for (int i = 0; i < actual.length; i++) {
        approxDist[i] = rawL2(query, mergedList.get(actual[i]));
      }
      double[] exactDist = new double[exactOrds.length];
      for (int i = 0; i < exactOrds.length; i++) {
        exactDist[i] = rawL2(query, mergedList.get(exactOrds[i]));
      }
      Arrays.sort(exactDist);
      if (exactDist.length > 0) {
        double thresh = (1.0 + EPS_RECALL) * exactDist[exactDist.length - 1];
        for (int i = 0; i < approxDist.length; i++) {
          if (approxDist[i] <= thresh) {
            epsHits++;
          }
          returned++;
        }
      }
      Arrays.sort(approxDist);
      int ranks = Math.min(approxDist.length, exactDist.length);
      for (int i = 0; i < ranks; i++) {
        if (exactDist[i] != 0.0) {
          mdrSum += approxDist[i] / exactDist[i];
          mdrTerms++;
        }
      }
    }
    double recall = totalOverlap / (double) (queries.size() * k);
    double mdr = mdrTerms == 0 ? Double.NaN : mdrSum / mdrTerms;
    double epsRecall = returned == 0 ? Double.NaN : epsHits / (double) returned;
    return new QualityMetrics(recall, mdr, epsRecall);
  }

  /** Raw Euclidean L2. Do not use {@link VectorSimilarityFunction#EUCLIDEAN} here. */
  static double rawL2(float[] a, float[] b) {
    return Math.sqrt(VectorUtil.squareDistance(a, b));
  }

  private static int overlap(int[] a, int[] b) {
    int[] x = a.clone();
    int[] y = b.clone();
    Arrays.sort(x);
    Arrays.sort(y);
    int count = 0;
    for (int i = 0, j = 0; i < x.length && j < y.length; ) {
      if (x[i] == y[j]) {
        count++;
        i++;
        j++;
      } else if (x[i] > y[j]) {
        j++;
      } else {
        i++;
      }
    }
    return count;
  }

  private static boolean anyDeletes(boolean[] segmentHasDeletes) {
    for (boolean b : segmentHasDeletes) {
      if (b) {
        return true;
      }
    }
    return false;
  }

  @AuxCounters(AuxCounters.Type.EVENTS)
  @State(Scope.Thread)
  public static class Metrics {
    public long distanceComputations;
    public long recallBasisPoints;

    @Setup(Level.Iteration)
    public void clean() {
      distanceComputations = 0;
      recallBasisPoints = 0;
    }
  }

  static final class RunResult {
    final OnHeapHnswGraph graph;
    final double millis;
    final long distances;

    RunResult(OnHeapHnswGraph graph, double millis, long distances) {
      this.graph = graph;
      this.millis = millis;
      this.distances = distances;
    }
  }

  static final class PreparedMerge {
    final OnHeapHnswGraph[] segmentGraphs;
    final int[][] ordMaps;
    final BitSet initializedNodes;
    final int mergedSize;
    final List<float[]> mergedList;
    final List<float[]> queries;

    PreparedMerge(
        OnHeapHnswGraph[] segmentGraphs,
        int[][] ordMaps,
        BitSet initializedNodes,
        int mergedSize,
        List<float[]> mergedList,
        List<float[]> queries) {
      this.segmentGraphs = segmentGraphs;
      this.ordMaps = ordMaps;
      this.initializedNodes = initializedNodes;
      this.mergedSize = mergedSize;
      this.mergedList = mergedList;
      this.queries = queries;
    }
  }

  private static final class CachedSegments {
    final float[][] vectors;
    final float[][] queries;
    final OnHeapHnswGraph[] graphs;

    CachedSegments(float[][] vectors, float[][] queries, OnHeapHnswGraph[] graphs) {
      this.vectors = vectors;
      this.queries = queries;
      this.graphs = graphs;
    }
  }

  /**
   * Wraps a {@link RandomVectorScorerSupplier} to count distance computations ({@code score} and
   * {@code bulkScore}). Copies share the counter.
   */
  static final class CountingScorerSupplier implements RandomVectorScorerSupplier {
    private final RandomVectorScorerSupplier delegate;
    private final AtomicLong counter;

    CountingScorerSupplier(RandomVectorScorerSupplier delegate) {
      this(delegate, new AtomicLong());
    }

    private CountingScorerSupplier(RandomVectorScorerSupplier delegate, AtomicLong counter) {
      this.delegate = delegate;
      this.counter = counter;
    }

    void reset() {
      counter.set(0);
    }

    long count() {
      return counter.get();
    }

    @Override
    public UpdateableRandomVectorScorer scorer() throws IOException {
      return new CountingScorer(delegate.scorer(), counter);
    }

    @Override
    public RandomVectorScorerSupplier copy() throws IOException {
      return new CountingScorerSupplier(delegate.copy(), counter);
    }
  }

  private static final class CountingScorer implements UpdateableRandomVectorScorer {
    private final UpdateableRandomVectorScorer delegate;
    private final AtomicLong counter;

    CountingScorer(UpdateableRandomVectorScorer delegate, AtomicLong counter) {
      this.delegate = delegate;
      this.counter = counter;
    }

    @Override
    public void setScoringOrdinal(int node) throws IOException {
      delegate.setScoringOrdinal(node);
    }

    @Override
    public float score(int node) throws IOException {
      counter.incrementAndGet();
      return delegate.score(node);
    }

    @Override
    public float bulkScore(int[] nodes, float[] scores, int numNodes) throws IOException {
      counter.addAndGet(numNodes);
      return delegate.bulkScore(nodes, scores, numNodes);
    }

    @Override
    public int maxOrd() {
      return delegate.maxOrd();
    }

    @Override
    public int ordToDoc(int ord) {
      return delegate.ordToDoc(ord);
    }

    @Override
    public Bits getAcceptOrds(Bits acceptDocs) {
      return delegate.getAcceptOrds(acceptDocs);
    }
  }
}
