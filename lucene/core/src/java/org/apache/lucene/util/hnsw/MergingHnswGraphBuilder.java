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

/**
 * A graph builder that is used during segments' merging.
 *
 * <p>This builder uses a smart algorithm to merge multiple graphs into a single graph. The
 * algorithm is based on the idea that if we know where we want to insert a node, we have a good
 * idea of where we want to insert its neighbors.
 *
 * <p>The algorithm is based on the following steps:
 *
 * <ul>
 *   <li>Get all graphs that don't have deletions and sort them by size desc.
 *   <li>Copy the largest graph to the new graph (gL).
 *   <li>For each remaining small graph (gS):
 *       <ul>
 *         <li>Find the nodes that best cover gS: join set `j`. These nodes will be inserted into gL
 *             as usual: by searching gL to find the best candidates `w` to which connect the nodes.
 *         <li>For each remaining node in gS:
 *             <ul>
 *               <li>We provide eps to search in gL. We form `eps` by the union of the node's
 *                   neighbors in gS and the node's neighbors' neighbors in gL. We also limit
 *                   beamWidth (efConstruction to M*3)
 *             </ul>
 *       </ul>
 * </ul>
 *
 * <p>We expect the size of join set `j` to be small, around 1/5 to 1/2 of the size of gS. For the
 * rest of the nodes in gS, we expect savings by performing lighter searches in gL.
 *
 * @lucene.experimental
 */
public final class MergingHnswGraphBuilder extends HnswGraphBuilder {
  /**
   * Repair after a lazy (forward-only) merge. {@code LIGHT} only symmetrizes existing forward edges
   * where the reverse side has room. {@code THOROUGH} also runs a diversity-aware local search on
   * underfull level-0 nodes. Override with {@code -Dhnsw.lazy.repair=thorough}.
   */
  public enum LazyRepairMode {
    LIGHT,
    THOROUGH
  }

  public static volatile LazyRepairMode lazyRepairMode = parseRepairMode();

  private final HnswGraph[] graphs;
  private final int[][] ordMaps;
  private final BitSet initializedNodes;

  private static LazyRepairMode parseRepairMode() {
    String p = System.getProperty("hnsw.lazy.repair", "light");
    if (p != null && p.equalsIgnoreCase("thorough")) {
      return LazyRepairMode.THOROUGH;
    }
    return LazyRepairMode.LIGHT;
  }

  private MergingHnswGraphBuilder(
      RandomVectorScorerSupplier scorerSupplier,
      int beamWidth,
      long seed,
      OnHeapHnswGraph initializedGraph,
      HnswGraph[] graphs,
      int[][] ordMaps,
      BitSet initializedNodes)
      throws IOException {
    super(scorerSupplier, beamWidth, seed, initializedGraph);
    this.graphs = graphs;
    this.ordMaps = ordMaps;
    this.initializedNodes = initializedNodes;
  }

  /**
   * Create a new HnswGraphBuilder that is initialized with the provided HnswGraph.
   *
   * @param scorerSupplier the scorer to use for vectors
   * @param beamWidth the number of nodes to explore in the search
   * @param seed the seed for the random number generator
   * @param graphs the graphs to merge
   * @param ordMaps the ordinal maps for the graphs
   * @param totalNumberOfVectors the total number of vectors in the new graph, this should include
   *     all vectors expected to be added to the graph in the future
   * @param initializedNodes the nodes will be initialized through the merging, if null, all nodes
   *     should be already initialized after {@link #updateGraph(HnswGraph, int[])} being called
   * @return a new HnswGraphBuilder that is initialized with the provided HnswGraph
   * @throws IOException when reading the graph fails
   */
  public static MergingHnswGraphBuilder fromGraphs(
      RandomVectorScorerSupplier scorerSupplier,
      int beamWidth,
      long seed,
      HnswGraph[] graphs,
      int[][] ordMaps,
      int totalNumberOfVectors,
      BitSet initializedNodes)
      throws IOException {
    if (graphs == null || graphs.length == 0 || graphs[0] == null || graphs[0].size() == 0) {
      throw new IllegalArgumentException("graphs[0] must be a non-empty base graph");
    }
    OnHeapHnswGraph graph =
        InitializedHnswGraphBuilder.initGraph(
            graphs[0], ordMaps[0], totalNumberOfVectors, beamWidth, scorerSupplier);
    return new MergingHnswGraphBuilder(
        scorerSupplier, beamWidth, seed, graph, graphs, ordMaps, initializedNodes);
  }

  @Override
  public OnHeapHnswGraph build(int maxOrd) throws IOException {
    if (frozen) {
      throw new IllegalStateException("This HnswGraphBuilder is frozen and cannot be updated");
    }
    if (infoStream.isEnabled(HNSW_COMPONENT)) {
      String graphSizes = "";
      for (HnswGraph g : graphs) {
        graphSizes += g.size() + " ";
      }
      infoStream.message(
          HNSW_COMPONENT,
          "build graph from merging "
              + graphs.length
              + " graphs of "
              + maxOrd
              + " vectors, graph sizes:"
              + graphSizes);
    }
    lazyBackwardConnect = true;
    for (int i = 1; i < graphs.length; i++) {
      if (graphs[i] != null && graphs[i].size() > 0) {
        updateGraph(graphs[i], ordMaps[i]);
      }
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
    lazyBackwardConnect = false;
    repairBackwardLinks(maxOrd);

    return getCompletedGraph();
  }

  /** Merge the smaller graph into the current larger graph. */
  private void updateGraph(HnswGraph gS, int[] ordMapS) throws IOException {
    int size = gS.size();
    if (size == 0) {
      return;
    }
    IntHashSet j = UpdateGraphsUtils.computeJoinSet(gS);

    // add nodes in the join set directly to the graph
    // sort for stability
    int[] nodes = j.toArray();
    Arrays.sort(nodes);
    for (int node : nodes) {
      int mapped = mapOrd(ordMapS, node);
      if (mapped >= 0) {
        addGraphNode(mapped);
      }
    }

    // for each node outside of j set:
    // form the entry points set for the node
    // by joining the node's neighbours in gS with
    // the node's neighbours' neighbours in gL
    for (int u = 0; u < size; u++) {
      if (j.contains(u)) {
        continue;
      }
      int mapped = mapOrd(ordMapS, u);
      if (mapped < 0) {
        continue;
      }
      IntHashSet eps = new IntHashSet();
      gS.seek(0, u);
      for (int v = gS.nextNeighbor(); v != NO_MORE_DOCS; v = gS.nextNeighbor()) {
        // if u's neighbour v is in the join set, or already added to gL (v < u),
        // then we add v's neighbours from gL to the candidate list
        if (v < u || j.contains(v)) {
          int newv = mapOrd(ordMapS, v);
          if (newv < 0) {
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
      addGraphNode(mapped, eps);
    }
  }

  private void repairBackwardLinks(int maxOrd) throws IOException {
    if (maxOrd <= 0) {
      return;
    }
    if (lazyRepairMode == LazyRepairMode.THOROUGH) {
      thoroughRepair(maxOrd);
    }
    lightRepair(maxOrd);
  }

  /** Symmetrize level-0 forward edges where the reverse side has room. No diversity check. */
  private void lightRepair(int maxOrd) throws IOException {
    int maxConn0 = M * 2;
    for (int u = 0; u < maxOrd; u++) {
      NeighborArray nu = hnsw.getNeighbors(0, u);
      int[] uNodes = nu.nodes();
      for (int i = 0; i < nu.size(); i++) {
        int v = uNodes[i];
        if (v < 0 || v >= maxOrd || v == u) {
          continue;
        }
        NeighborArray nv = hnsw.getNeighbors(0, v);
        if (nv.size() >= maxConn0 || containsNode(nv, u)) {
          continue;
        }
        nv.addOutOfOrder(u, nu.getScores(i));
      }
    }
  }

  /**
   * For level-0 nodes with unused neighbour slots, search from the current neighbourhood and add
   * diversity-aware links (including reverse, since {@code lazyBackwardConnect} is off).
   */
  private void thoroughRepair(int maxOrd) throws IOException {
    int maxConn0 = M * 2;
    for (int node = 0; node < maxOrd; node++) {
      NeighborArray neighbors = hnsw.getNeighbors(0, node);
      if (neighbors.size() >= maxConn0) {
        continue;
      }
      scorer.setScoringOrdinal(node);
      int[] eps;
      if (neighbors.size() > 0) {
        eps = Arrays.copyOf(neighbors.nodes(), neighbors.size());
      } else if (hnsw.entryNode() >= 0) {
        eps = new int[] {hnsw.entryNode()};
      } else {
        continue;
      }
      beamCandidates.clear();
      graphSearcher.searchLevel(beamCandidates, scorer, 0, eps, hnsw, null);
      NeighborArray scratch = new NeighborArray(Math.max(beamCandidates.k(), M + 1), false);
      popToScratch(beamCandidates, scratch);
      addDiverseNeighbors(0, node, scratch, scorer, true);
    }
  }

  private static boolean containsNode(NeighborArray arr, int node) {
    int[] nodes = arr.nodes();
    for (int i = 0; i < arr.size(); i++) {
      if (nodes[i] == node) {
        return true;
      }
    }
    return false;
  }

  private static int mapOrd(int[] ordMap, int oldOrd) {
    if (oldOrd < 0 || oldOrd >= ordMap.length) {
      return -1;
    }
    return ordMap[oldOrd];
  }
}
