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

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import java.io.IOException;
import java.util.Arrays;
import org.apache.lucene.internal.hppc.IntHashSet;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;

/**
 * HNSW-Merger-style merge builder: forward-only insertion (no inline reverse links), cross-query of
 * source graphs (Lucene #15504), incoming-edge recording, then a single search-free level-0
 * RobustPrune of {@code incoming ∪ forward ∪ originalSourceNeighbors}.
 *
 * <p>This is the mechanism Lucene's {@link MergingHnswGraphBuilder} does not have: the current
 * smart merge still installs reverse links during insert (and the concurrent builder takes {@link
 * HnswLock#write} per reverse link).
 *
 * @lucene.experimental
 */
public class CombinedHnswGraphBuilder extends HnswGraphBuilder {
  private final HnswGraph[] graphs;
  private final int[][] ordMaps;
  private final BitSet initializedNodes;
  private final IncomingEdges extras;
  private final int[] homeGraph;
  private final int[] homeOldOrd;

  private HnswGraphSearcher sourceSearcher;
  private GraphBuilderKnnCollector sourceCollector;

  /**
   * Create a sequential combined merger initialized from {@code graphs[0]} (largest / base).
   *
   * @param initializedNodes nodes already present via source graphs; remaining live ords are
   *     inserted from scratch. Null means every live node comes from a source graph.
   */
  public static CombinedHnswGraphBuilder fromGraphs(
      RandomVectorScorerSupplier scorerSupplier,
      int beamWidth,
      long seed,
      HnswGraph[] graphs,
      int[][] ordMaps,
      int totalNumberOfVectors,
      BitSet initializedNodes)
      throws IOException {
    if (graphs == null || graphs.length == 0) {
      throw new IllegalArgumentException("graphs must be non-empty");
    }
    OnHeapHnswGraph graph =
        InitializedHnswGraphBuilder.initGraph(
            graphs[0], ordMaps[0], totalNumberOfVectors, beamWidth, scorerSupplier);
    IncomingEdges incoming = new IncomingEdges(totalNumberOfVectors);
    IncomingEdges extras = new IncomingEdges(totalNumberOfVectors);
    int[] homeGraph = new int[totalNumberOfVectors];
    int[] homeOldOrd = new int[totalNumberOfVectors];
    Arrays.fill(homeGraph, -1);
    fillHomeMaps(graphs, ordMaps, homeGraph, homeOldOrd);
    return new CombinedHnswGraphBuilder(
        scorerSupplier,
        beamWidth,
        seed,
        graph,
        null,
        null,
        incoming,
        extras,
        graphs,
        ordMaps,
        homeGraph,
        homeOldOrd,
        initializedNodes);
  }

  CombinedHnswGraphBuilder(
      RandomVectorScorerSupplier scorerSupplier,
      int beamWidth,
      long seed,
      OnHeapHnswGraph hnsw,
      HnswLock hnswLock,
      HnswGraphSearcher graphSearcher,
      IncomingEdges incoming,
      IncomingEdges extras,
      HnswGraph[] graphs,
      int[][] ordMaps,
      int[] homeGraph,
      int[] homeOldOrd,
      BitSet initializedNodes)
      throws IOException {
    super(
        scorerSupplier,
        beamWidth,
        seed,
        hnsw,
        hnswLock,
        graphSearcher != null
            ? graphSearcher
            : new HnswGraphSearcher(
                new NeighborQueue(beamWidth, true), new FixedBitSet(Math.max(1, hnsw.size()))));
    this.skipReverseLinks = true;
    this.incomingEdges = incoming;
    this.extras = extras;
    this.graphs = graphs;
    this.ordMaps = ordMaps;
    this.homeGraph = homeGraph;
    this.homeOldOrd = homeOldOrd;
    this.initializedNodes = initializedNodes;
  }

  static void fillHomeMaps(HnswGraph[] graphs, int[][] ordMaps, int[] homeGraph, int[] homeOldOrd) {
    for (int g = 0; g < graphs.length; g++) {
      int[] map = ordMaps[g];
      for (int old = 0; old < map.length; old++) {
        int neu = map[old];
        if (neu >= 0 && neu < homeGraph.length && homeGraph[neu] == -1) {
          homeGraph[neu] = g;
          homeOldOrd[neu] = old;
        }
      }
    }
  }

  IncomingEdges incomingEdges() {
    return incomingEdges;
  }

  IncomingEdges extras() {
    return extras;
  }

  @Override
  public OnHeapHnswGraph build(int maxOrd) throws IOException {
    if (frozen) {
      throw new IllegalStateException("This HnswGraphBuilder is frozen and cannot be updated");
    }
    if (infoStream.isEnabled(HNSW_COMPONENT)) {
      infoStream.message(
          HNSW_COMPONENT,
          "build combined HNSW-Merger graph from "
              + graphs.length
              + " graphs, "
              + maxOrd
              + " vectors");
    }
    for (int i = 1; i < graphs.length; i++) {
      updateGraph(graphs[i], ordMaps[i]);
    }
    if (initializedNodes != null && maxOrd > 0) {
      for (int node = initializedNodes.nextClearBit(0, maxOrd);
          node != NO_MORE_DOCS;
          node =
              (node + 1 < maxOrd)
                  ? initializedNodes.nextClearBit(node + 1, maxOrd)
                  : NO_MORE_DOCS) {
        addGraphNode(node);
      }
    }
    pruneLevel0(maxOrd);
    return getCompletedGraph();
  }

  /** Merge one source graph: join-set full insert, remaining nodes with eps + cross-query. */
  private void updateGraph(HnswGraph gS, int[] ordMapS) throws IOException {
    int size = gS.size();
    if (size == 0) {
      return;
    }
    IntHashSet j = UpdateGraphsUtils.computeJoinSet(gS);
    int[] nodes = j.toArray();
    Arrays.sort(nodes);
    for (int node : nodes) {
      int neu = ordMapS[node];
      if (neu >= 0) {
        addGraphNode(neu);
      }
    }
    for (int u = 0; u < size; u++) {
      if (j.contains(u)) {
        continue;
      }
      int neu = ordMapS[u];
      if (neu < 0) {
        continue;
      }
      IntHashSet eps = new IntHashSet();
      gS.seek(0, u);
      for (int v = gS.nextNeighbor(); v != NO_MORE_DOCS; v = gS.nextNeighbor()) {
        if (v < u || j.contains(v)) {
          int newv = ordMapS[v];
          if (newv < 0 || hnsw.nodeExistAtLevel(0, newv) == false) {
            continue;
          }
          eps.add(newv);
          hnsw.seek(0, newv);
          int friendOrd;
          while ((friendOrd = hnsw.nextNeighbor()) != NO_MORE_DOCS) {
            eps.add(friendOrd);
          }
        }
      }
      addGraphNode(neu, eps);
    }
  }

  @Override
  protected NeighborArray augmentLevel0Candidates(int node, NeighborArray scratch)
      throws IOException {
    if (graphs == null || graphs.length <= 1) {
      return scratch;
    }
    IntHashSet seen = new IntHashSet();
    for (int i = 0; i < scratch.size(); i++) {
      seen.add(scratch.nodes()[i]);
    }
    OrdScoreList extra = new OrdScoreList(16);
    for (int g = 1; g < graphs.length; g++) {
      HnswGraph src = graphs[g];
      if (src == null || src.size() == 0 || src.entryNode() < 0) {
        continue;
      }
      searchSourceGraph(node, src, ordMaps[g], seen, extra);
    }
    if (extra.size == 0) {
      return scratch;
    }
    int total = scratch.size() + extra.size;
    int[] nodes = new int[total];
    float[] scores = new float[total];
    for (int i = 0; i < scratch.size(); i++) {
      nodes[i] = scratch.nodes()[i];
      scores[i] = scratch.getScores(i);
    }
    System.arraycopy(extra.ords, 0, nodes, scratch.size(), extra.size);
    System.arraycopy(extra.scores, 0, scores, scratch.size(), extra.size);
    sortByScoreAscending(nodes, scores, total);
    NeighborArray union = new NeighborArray(total, false);
    for (int i = 0; i < total; i++) {
      union.addInOrder(nodes[i], scores[i]);
    }
    return union;
  }

  private void searchSourceGraph(
      int queryNewOrd, HnswGraph src, int[] ordMap, IntHashSet seen, OrdScoreList extra)
      throws IOException {
    int k = Math.max(1, beamCandidates.k());
    ensureSourceSearch(src.maxNodeId() + 1, k);
    sourceCollector.clear();
    RemappingScorer remap = new RemappingScorer(scorer, ordMap);
    int[] eps = sourceEntryPoints(queryNewOrd, src, ordMap);
    if (eps.length == 0) {
      return;
    }
    Bits live = new MappedLiveBits(ordMap);
    sourceSearcher.searchLevel(sourceCollector, remap, 0, eps, src, live);
    int hits = sourceCollector.size();
    for (int i = 0; i < hits; i++) {
      float score = sourceCollector.minimumScore();
      int oldOrd = sourceCollector.popNode();
      if (oldOrd < 0 || oldOrd >= ordMap.length) {
        continue;
      }
      int neu = ordMap[oldOrd];
      if (neu < 0 || neu == queryNewOrd || seen.contains(neu)) {
        continue;
      }
      seen.add(neu);
      extras.add(queryNewOrd, neu);
      extras.add(neu, queryNewOrd);
      // Only attach now if the target already exists in gL. Future nodes stay in extras
      // and are wired during the search-free prune after every live node has been inserted.
      if (hnsw.nodeExistAtLevel(0, neu)) {
        extra.add(neu, score);
      }
    }
  }

  private int[] sourceEntryPoints(int queryNewOrd, HnswGraph src, int[] ordMap) throws IOException {
    IntHashSet eps = new IntHashSet();
    int entry = src.entryNode();
    if (entry >= 0) {
      eps.add(entry);
    }
    int g = homeGraph[queryNewOrd];
    if (g >= 0 && graphs[g] == src) {
      int old = homeOldOrd[queryNewOrd];
      if (old >= 0 && src instanceof OnHeapHnswGraph onHeap && onHeap.nodeExistAtLevel(0, old)) {
        NeighborArray nbrs = onHeap.getNeighbors(0, old);
        for (int i = 0; i < nbrs.size(); i++) {
          eps.add(nbrs.nodes()[i]);
        }
      }
    }
    int[] arr = eps.toArray();
    // drop deleted source ords so search does not score holes
    int w = 0;
    for (int old : arr) {
      if (old >= 0 && old < ordMap.length && ordMap[old] >= 0) {
        arr[w++] = old;
      }
    }
    return w == arr.length ? arr : Arrays.copyOf(arr, w);
  }

  private void ensureSourceSearch(int graphSize, int k) {
    if (sourceSearcher == null) {
      // Snapshot neighbors instead of HnswGraph.seek: source graphs are shared across
      // concurrent workers and seek() mutates iterator state on the graph object.
      sourceSearcher =
          new SnapshotOnHeapSearcher(
              new NeighborQueue(k, true), new FixedBitSet(Math.max(1, graphSize)));
      sourceCollector = new GraphBuilderKnnCollector(k);
    }
  }

  /**
   * Copies neighbor ids so concurrent readers do not share {@link OnHeapHnswGraph} iterator state.
   */
  static final class SnapshotOnHeapSearcher extends HnswGraphSearcher {
    private int[] nodeBuffer;
    private int upto;
    private int size;

    SnapshotOnHeapSearcher(NeighborQueue candidates, BitSet visited) {
      super(candidates, visited);
    }

    @Override
    void graphSeek(HnswGraph graph, int level, int targetNode) {
      NeighborArray nbrs = ((OnHeapHnswGraph) graph).getNeighbors(level, targetNode);
      int n = nbrs.size();
      if (nodeBuffer == null || nodeBuffer.length < n) {
        nodeBuffer = new int[Math.max(n, 8)];
      }
      System.arraycopy(nbrs.nodes(), 0, nodeBuffer, 0, n);
      size = n;
      upto = -1;
    }

    @Override
    int graphNextNeighbor(HnswGraph graph) {
      if (++upto < size) {
        return nodeBuffer[upto];
      }
      return NO_MORE_DOCS;
    }
  }

  /**
   * Search-free level-0 prune: neighbor list = RobustPrune(incoming ∪ forward ∪ original ∪
   * cross-query extras, maxConn=2M).
   */
  void pruneLevel0(int maxOrd) throws IOException {
    if (maxOrd <= 0) {
      return;
    }
    int maxConn = M * 2;
    for (int v = 0; v < maxOrd; v++) {
      if (needsPrune(v)) {
        pruneNode(v, maxConn);
      }
    }
  }

  boolean needsPrune(int v) {
    if (hnsw.nodeExistAtLevel(0, v) == false) {
      return false;
    }
    boolean fromBase = homeGraph != null && v < homeGraph.length && homeGraph[v] == 0;
    return fromBase == false || incomingEdges.size(v) > 0 || extras.size(v) > 0;
  }

  void pruneNode(int v, int maxConn) throws IOException {
    NeighborArray cur = hnsw.getNeighbors(0, v);
    IntHashSet seen = new IntHashSet();
    int cap = cur.size() + incomingEdges.size(v) + extras.size(v) + maxConn + 8;
    OrdScoreList cands = new OrdScoreList(cap);
    appendUnique(cur.nodes(), cur.size(), v, seen, cands);
    appendUnique(incomingEdges.get(v), incomingEdges.size(v), v, seen, cands);
    appendUnique(extras.get(v), extras.size(v), v, seen, cands);
    appendOriginalSourceNeighbors(v, seen, cands);
    if (cands.size == 0) {
      return;
    }
    scorer.setScoringOrdinal(v);
    for (int i = 0; i < cands.size; i++) {
      cands.scores[i] = scorer.score(cands.ords[i]);
    }
    sortByScoreAscending(cands.ords, cands.scores, cands.size);
    NeighborArray candidates = new NeighborArray(cands.size, false);
    for (int i = 0; i < cands.size; i++) {
      candidates.addInOrder(cands.ords[i], cands.scores[i]);
    }
    cur.clear();
    selectAndLinkDiverse(v, cur, candidates, maxConn, scorer, false);
  }

  private void appendOriginalSourceNeighbors(int v, IntHashSet seen, OrdScoreList cands)
      throws IOException {
    if (homeGraph == null || v >= homeGraph.length) {
      return;
    }
    int g = homeGraph[v];
    if (g < 0) {
      return;
    }
    int old = homeOldOrd[v];
    HnswGraph src = graphs[g];
    if (src instanceof OnHeapHnswGraph onHeap && onHeap.nodeExistAtLevel(0, old)) {
      NeighborArray nbrs = onHeap.getNeighbors(0, old);
      for (int i = 0; i < nbrs.size(); i++) {
        int nbr = nbrs.nodes()[i];
        if (nbr < 0 || nbr >= ordMaps[g].length) {
          continue;
        }
        int neu = ordMaps[g][nbr];
        if (neu < 0 || neu == v || seen.contains(neu)) {
          continue;
        }
        seen.add(neu);
        cands.add(neu, 0f);
      }
    } else {
      src.seek(0, old);
      for (int nbr = src.nextNeighbor(); nbr != NO_MORE_DOCS; nbr = src.nextNeighbor()) {
        if (nbr < 0 || nbr >= ordMaps[g].length) {
          continue;
        }
        int neu = ordMaps[g][nbr];
        if (neu < 0 || neu == v || seen.contains(neu)) {
          continue;
        }
        seen.add(neu);
        cands.add(neu, 0f);
      }
    }
  }

  private static void appendUnique(
      int[] src, int srcSize, int self, IntHashSet seen, OrdScoreList cands) {
    int limit = Math.min(srcSize, src.length);
    for (int i = 0; i < limit; i++) {
      int x = src[i];
      if (x < 0 || x == self || seen.contains(x)) {
        continue;
      }
      seen.add(x);
      cands.add(x, 0f);
    }
  }

  private static void sortByScoreAscending(int[] nodes, float[] scores, int n) {
    // insertion sort; n is beam-sized or a few dozen neighbors
    for (int i = 1; i < n; i++) {
      int node = nodes[i];
      float score = scores[i];
      int j = i - 1;
      while (j >= 0 && scores[j] > score) {
        nodes[j + 1] = nodes[j];
        scores[j + 1] = scores[j];
        j--;
      }
      nodes[j + 1] = node;
      scores[j + 1] = score;
    }
  }

  /**
   * Scores source-graph (old) ordinals by remapping onto the merged-ordinal scorer. The delegate
   * must already be aimed at the query's merged ordinal.
   */
  static final class RemappingScorer implements UpdateableRandomVectorScorer {
    private final UpdateableRandomVectorScorer delegate;
    private final int[] ordMap;

    RemappingScorer(UpdateableRandomVectorScorer delegate, int[] ordMap) {
      this.delegate = delegate;
      this.ordMap = ordMap;
    }

    @Override
    public void setScoringOrdinal(int node) throws IOException {
      if (node >= 0 && node < ordMap.length && ordMap[node] >= 0) {
        delegate.setScoringOrdinal(ordMap[node]);
      }
    }

    @Override
    public float score(int node) throws IOException {
      if (node < 0 || node >= ordMap.length) {
        return Float.NEGATIVE_INFINITY;
      }
      int neu = ordMap[node];
      if (neu < 0) {
        return Float.NEGATIVE_INFINITY;
      }
      return delegate.score(neu);
    }

    @Override
    public float bulkScore(int[] nodes, float[] scores, int numNodes) throws IOException {
      float max = Float.NEGATIVE_INFINITY;
      for (int i = 0; i < numNodes; i++) {
        scores[i] = score(nodes[i]);
        max = Math.max(max, scores[i]);
      }
      return max;
    }

    @Override
    public int maxOrd() {
      return delegate.maxOrd();
    }
  }

  static final class OrdScoreList {
    int[] ords;
    float[] scores;
    int size;

    OrdScoreList(int cap) {
      this.ords = new int[Math.max(1, cap)];
      this.scores = new float[Math.max(1, cap)];
    }

    void add(int ord, float score) {
      if (size == ords.length) {
        ords = Arrays.copyOf(ords, ords.length * 2);
        scores = Arrays.copyOf(scores, scores.length * 2);
      }
      ords[size] = ord;
      scores[size] = score;
      size++;
    }
  }

  /** Accept source ordinals whose remapped merged ordinal is live (not {@code -1}). */
  static final class MappedLiveBits implements Bits {
    private final int[] ordMap;

    MappedLiveBits(int[] ordMap) {
      this.ordMap = ordMap;
    }

    @Override
    public boolean get(int index) {
      return index >= 0 && index < ordMap.length && ordMap[index] >= 0;
    }

    @Override
    public int length() {
      return ordMap.length;
    }
  }
}
