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
package org.apache.lucene.util.hnsw;

import static org.apache.lucene.util.hnsw.HnswGraphBuilder.HNSW_COMPONENT;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import org.apache.lucene.internal.hppc.IntHashSet;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.InfoStream;

/**
 * Concurrent counterpart of {@link CombinedHnswGraphBuilder}. Same insert set as {@link
 * HnswConcurrentMergeBuilder} / {@link ConcurrentHnswMerger}: copy the largest graph, then insert
 * every other live node. Reverse links are not installed under {@link HnswLock#write}; workers only
 * take the (shared) read lock while snapshotting neighbor lists for search. Incoming edges are
 * appended under a striped lock, then a search-free level-0 prune runs in parallel.
 *
 * @lucene.experimental
 */
public final class CombinedConcurrentHnswGraphBuilder implements HnswBuilder {

  private static final int DEFAULT_BATCH_SIZE = 2048;

  private final TaskExecutor taskExecutor;
  private final Worker[] workers;
  private final HnswLock hnswLock;
  private final IncomingEdges incoming;
  private final IncomingEdges extras;
  private final Set<Long> workerThreadIds = ConcurrentHashMap.newKeySet();
  private volatile double lastEffectiveConcurrency;
  private volatile long lastWallNanos;
  private volatile long lastPruneNanos;
  private InfoStream infoStream = InfoStream.getDefault();
  private boolean frozen;

  /**
   * @param initializedNodes nodes already copied from the base graph (skipped by workers)
   * @param sourceGraphs all merge source graphs (index 0 is the base already copied into {@code
   *     hnsw}); used for cross-query. May be a single-element array.
   */
  public CombinedConcurrentHnswGraphBuilder(
      TaskExecutor taskExecutor,
      int numWorker,
      RandomVectorScorerSupplier scorerSupplier,
      int beamWidth,
      OnHeapHnswGraph hnsw,
      BitSet initializedNodes,
      HnswGraph[] sourceGraphs,
      int[][] ordMaps)
      throws IOException {
    if (numWorker < 1) {
      throw new IllegalArgumentException("numWorker must be >= 1");
    }
    this.taskExecutor = taskExecutor;
    this.hnswLock = new HnswLock();
    int n = Math.max(1, hnsw.maxNodeId() + 1);
    this.incoming = new IncomingEdges(n);
    this.extras = new IncomingEdges(n);
    int[] homeGraph = new int[n];
    int[] homeOldOrd = new int[n];
    Arrays.fill(homeGraph, -1);
    if (sourceGraphs != null && sourceGraphs.length > 0) {
      CombinedHnswGraphBuilder.fillHomeMaps(sourceGraphs, ordMaps, homeGraph, homeOldOrd);
    }
    AtomicInteger workProgress = new AtomicInteger(0);
    workers = new Worker[numWorker];
    for (int i = 0; i < numWorker; i++) {
      workers[i] =
          new Worker(
              scorerSupplier.copy(),
              beamWidth,
              HnswGraphBuilder.randSeed,
              hnsw,
              hnswLock,
              incoming,
              extras,
              sourceGraphs,
              ordMaps,
              homeGraph,
              homeOldOrd,
              initializedNodes,
              workProgress,
              workerThreadIds);
    }
  }

  @Override
  public OnHeapHnswGraph build(int maxOrd) throws IOException {
    if (frozen) {
      throw new IllegalStateException("graph has already been built");
    }
    long mergeStartTimeNs = System.nanoTime();
    if (infoStream.isEnabled(HNSW_COMPONENT)) {
      infoStream.message(
          HNSW_COMPONENT,
          "build combined graph from "
              + maxOrd
              + " vectors, with "
              + workers.length
              + " workers (forward-only + prune)");
    }
    AtomicLong cumulativeWorkTimeNs = new AtomicLong();
    for (Worker worker : workers) {
      worker.setMergeStartTimeNs(mergeStartTimeNs);
      worker.setCumulativeWorkTimeNs(cumulativeWorkTimeNs);
    }
    List<Callable<Void>> futures = new ArrayList<>();
    for (int i = 0; i < workers.length; i++) {
      int finalI = i;
      futures.add(
          () -> {
            workers[finalI].runInsert(maxOrd);
            return null;
          });
    }
    taskExecutor.invokeAll(futures);

    long pruneStart = System.nanoTime();
    AtomicInteger pruneProgress = new AtomicInteger(0);
    List<Callable<Void>> pruneFutures = new ArrayList<>();
    for (int i = 0; i < workers.length; i++) {
      int finalI = i;
      pruneFutures.add(
          () -> {
            workers[finalI].runPrune(maxOrd, pruneProgress);
            return null;
          });
    }
    taskExecutor.invokeAll(pruneFutures);
    lastPruneNanos = System.nanoTime() - pruneStart;

    lastWallNanos = System.nanoTime() - mergeStartTimeNs;
    double wallClockMs = lastWallNanos / 1_000_000.0;
    double totalWorkerMs = cumulativeWorkTimeNs.get() / 1_000_000.0;
    lastEffectiveConcurrency = wallClockMs > 0 ? totalWorkerMs / wallClockMs : 0;
    if (infoStream.isEnabled(HNSW_COMPONENT)) {
      infoStream.message(
          HNSW_COMPONENT,
          String.format(
              Locale.ROOT,
              "combined merge completed: %d vectors, %.2f ms wall, %.2f ms prune, %.2f ms worker, %.2fx effective concurrency",
              maxOrd,
              wallClockMs,
              lastPruneNanos / 1_000_000.0,
              totalWorkerMs,
              lastEffectiveConcurrency));
    }
    return getCompletedGraph();
  }

  @Override
  public void addGraphNode(int node) {
    throw new UnsupportedOperationException("This builder is for merge only");
  }

  @Override
  public void addGraphNode(int node, IntHashSet eps) {
    throw new UnsupportedOperationException("This builder is for merge only");
  }

  @Override
  public void setInfoStream(InfoStream infoStream) {
    this.infoStream = infoStream;
    for (Worker worker : workers) {
      worker.setInfoStream(infoStream);
    }
  }

  @Override
  public OnHeapHnswGraph getCompletedGraph() throws IOException {
    if (frozen == false) {
      workers[0].getCompletedGraph();
      frozen = true;
    }
    return getGraph();
  }

  @Override
  public OnHeapHnswGraph getGraph() {
    return workers[0].getGraph();
  }

  public long writeLockWaitNanos() {
    return hnswLock.writeWaitNanos();
  }

  public long writeLockCount() {
    return hnswLock.writeCount();
  }

  public long incomingWaitNanos() {
    return incoming.addWaitNanos();
  }

  public long incomingCount() {
    return incoming.addCount();
  }

  public Set<Long> workerThreadIds() {
    return workerThreadIds;
  }

  public double lastEffectiveConcurrency() {
    return lastEffectiveConcurrency;
  }

  public long lastWallNanos() {
    return lastWallNanos;
  }

  public long lastPruneNanos() {
    return lastPruneNanos;
  }

  public int numWorkers() {
    return workers.length;
  }

  private static final class Worker extends CombinedHnswGraphBuilder {
    private final BitSet initializedNodes;
    private final AtomicInteger workProgress;
    private final Set<Long> workerThreadIds;
    private int batchSize = DEFAULT_BATCH_SIZE;

    private Worker(
        RandomVectorScorerSupplier scorerSupplier,
        int beamWidth,
        long seed,
        OnHeapHnswGraph hnsw,
        HnswLock hnswLock,
        IncomingEdges incoming,
        IncomingEdges extras,
        HnswGraph[] graphs,
        int[][] ordMaps,
        int[] homeGraph,
        int[] homeOldOrd,
        BitSet initializedNodes,
        AtomicInteger workProgress,
        Set<Long> workerThreadIds)
        throws IOException {
      super(
          scorerSupplier,
          beamWidth,
          seed,
          hnsw,
          hnswLock,
          new LockedHnswGraphSearcher(
              new NeighborQueue(beamWidth, true), hnswLock, new FixedBitSet(hnsw.maxNodeId() + 1)),
          incoming,
          extras,
          graphs,
          ordMaps,
          homeGraph,
          homeOldOrd,
          initializedNodes);
      this.initializedNodes = initializedNodes;
      this.workProgress = workProgress;
      this.workerThreadIds = workerThreadIds;
    }

    private void runInsert(int maxOrd) throws IOException {
      workerThreadIds.add(Thread.currentThread().threadId());
      int start = getStartPos(maxOrd, workProgress, batchSize);
      while (start != -1) {
        int end = Math.min(maxOrd, start + batchSize);
        addVectors(start, end);
        start = getStartPos(maxOrd, workProgress, batchSize);
      }
    }

    private void runPrune(int maxOrd, AtomicInteger progress) throws IOException {
      workerThreadIds.add(Thread.currentThread().threadId());
      long pruneStart = System.nanoTime();
      int start = getStartPos(maxOrd, progress, batchSize);
      int maxConn = M * 2;
      while (start != -1) {
        int end = Math.min(maxOrd, start + batchSize);
        for (int v = start; v < end; v++) {
          if (needsPrune(v)) {
            pruneNode(v, maxConn);
          }
        }
        start = getStartPos(maxOrd, progress, batchSize);
      }
      addWorkTimeNs(System.nanoTime() - pruneStart);
    }

    @Override
    public void addGraphNode(int node) throws IOException {
      if (initializedNodes != null && initializedNodes.get(node)) {
        return;
      }
      super.addGraphNode(node);
    }
  }

  private static int getStartPos(int maxOrd, AtomicInteger workProgress, int batchSize) {
    int start = workProgress.getAndAdd(batchSize);
    return start < maxOrd ? start : -1;
  }
}
