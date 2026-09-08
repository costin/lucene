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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.NamedThreadFactory;

public class TestCombinedHnswGraphBuilder extends LuceneTestCase {
  private static final VectorSimilarityFunction SIM = VectorSimilarityFunction.EUCLIDEAN;
  private static final int M = 8;
  private static final int BEAM = 16;

  public void testSingleSegmentCopyIsSearchable() throws IOException {
    int n = atLeast(40);
    float[][] vectors = HnswGraphTestCase.createRandomFloatVectors(n, 8, random());
    OnHeapHnswGraph src = buildGraph(vectors);
    int[] map = identityMap(n);
    CombinedHnswGraphBuilder combined =
        CombinedHnswGraphBuilder.fromGraphs(
            scorer(vectors),
            BEAM,
            random().nextLong(),
            new HnswGraph[] {src},
            new int[][] {map},
            n,
            null);
    OnHeapHnswGraph merged = combined.build(n);
    assertEquals(n, merged.size());
    assertTrue(merged.entryNode() >= 0);
    assertDegreeAtMost(merged, M * 2);
    assertRecallAtLeast(merged, vectors, 0.5);
  }

  public void testTwoSegmentMergeAndDeletedOrdinals() throws IOException {
    int n1 = 30;
    int n2 = 20;
    float[][] all = HnswGraphTestCase.createRandomFloatVectors(n1 + n2, 8, random());
    float[][] a = slice(all, 0, n1);
    float[][] b = slice(all, n1, n2);
    OnHeapHnswGraph g1 = buildGraph(a);
    OnHeapHnswGraph g2 = buildGraph(b);
    int[] map1 = identityMap(n1);
    int[] map2 = new int[n2];
    // mark one deleted ordinal as -1 and compact the rest
    int deleted = 3;
    int next = n1;
    float[][] live = new float[n1 + n2 - 1][];
    System.arraycopy(all, 0, live, 0, n1);
    int livePos = n1;
    for (int i = 0; i < n2; i++) {
      if (i == deleted) {
        map2[i] = -1;
      } else {
        map2[i] = next++;
        live[livePos++] = b[i];
      }
    }
    int mergedSize = n1 + n2 - 1;
    CombinedHnswGraphBuilder combined =
        CombinedHnswGraphBuilder.fromGraphs(
            scorer(live),
            BEAM,
            random().nextLong(),
            new HnswGraph[] {g1, g2},
            new int[][] {map1, map2},
            mergedSize,
            null);
    OnHeapHnswGraph merged = combined.build(mergedSize);
    assertEquals(mergedSize, merged.size());
    assertDegreeAtMost(merged, M * 2);
    assertRecallAtLeast(merged, live, 0.4);
  }

  public void testJoinSetNodesInserted() throws IOException {
    int n1 = 40;
    int n2 = 40;
    float[][] all = HnswGraphTestCase.createRandomFloatVectors(n1 + n2, 8, random());
    OnHeapHnswGraph g1 = buildGraph(slice(all, 0, n1));
    OnHeapHnswGraph g2 = buildGraph(slice(all, n1, n2));
    int[] map1 = identityMap(n1);
    int[] map2 = new int[n2];
    for (int i = 0; i < n2; i++) {
      map2[i] = n1 + i;
    }
    CombinedHnswGraphBuilder combined =
        CombinedHnswGraphBuilder.fromGraphs(
            scorer(all),
            BEAM,
            random().nextLong(),
            new HnswGraph[] {g1, g2},
            new int[][] {map1, map2},
            n1 + n2,
            null);
    OnHeapHnswGraph merged = combined.build(n1 + n2);
    assertEquals(n1 + n2, merged.size());
    for (int i = 0; i < n1 + n2; i++) {
      assertTrue("missing node " + i, merged.nodeExistAtLevel(0, i));
    }
  }

  public void testUninitializedNodesFromDeletedSegment() throws IOException {
    int n1 = 25;
    int n2 = 15;
    float[][] all = HnswGraphTestCase.createRandomFloatVectors(n1 + n2, 8, random());
    OnHeapHnswGraph g1 = buildGraph(slice(all, 0, n1));
    // only the first graph is a source; second segment's live vectors are uninitialized
    int[] map1 = identityMap(n1);
    FixedBitSet initialized = new FixedBitSet(n1 + n2);
    for (int i = 0; i < n1; i++) {
      initialized.set(i);
    }
    CombinedHnswGraphBuilder combined =
        CombinedHnswGraphBuilder.fromGraphs(
            scorer(all),
            BEAM,
            random().nextLong(),
            new HnswGraph[] {g1},
            new int[][] {map1},
            n1 + n2,
            initialized);
    OnHeapHnswGraph merged = combined.build(n1 + n2);
    assertEquals(n1 + n2, merged.size());
    assertRecallAtLeast(merged, all, 0.4);
  }

  public void testEmptyAndSingleNode() throws IOException {
    float[][] one = HnswGraphTestCase.createRandomFloatVectors(1, 4, random());
    OnHeapHnswGraph src = buildGraph(one);
    CombinedHnswGraphBuilder combined =
        CombinedHnswGraphBuilder.fromGraphs(
            scorer(one), BEAM, 1L, new HnswGraph[] {src}, new int[][] {new int[] {0}}, 1, null);
    OnHeapHnswGraph merged = combined.build(1);
    assertEquals(1, merged.size());
    assertEquals(0, merged.entryNode());
  }

  public void testConcurrentProducesSearchableGraph() throws Exception {
    int n1 = 60;
    int n2 = 40;
    float[][] all = HnswGraphTestCase.createRandomFloatVectors(n1 + n2, 8, random());
    OnHeapHnswGraph g1 = buildGraph(slice(all, 0, n1));
    OnHeapHnswGraph g2 = buildGraph(slice(all, n1, n2));
    int[] map1 = identityMap(n1);
    int[] map2 = new int[n2];
    for (int i = 0; i < n2; i++) {
      map2[i] = n1 + i;
    }
    OnHeapHnswGraph init =
        InitializedHnswGraphBuilder.initGraph(g1, map1, n1 + n2, BEAM, scorer(all));
    FixedBitSet initialized = new FixedBitSet(n1 + n2);
    for (int i = 0; i < n1; i++) {
      initialized.set(i);
    }
    ExecutorService exec =
        Executors.newFixedThreadPool(4, new NamedThreadFactory("test-combined-merge"));
    try {
      CombinedConcurrentHnswGraphBuilder builder =
          new CombinedConcurrentHnswGraphBuilder(
              new TaskExecutor(exec),
              4,
              scorer(all),
              BEAM,
              init,
              initialized,
              new HnswGraph[] {g1, g2},
              new int[][] {map1, map2});
      OnHeapHnswGraph merged = builder.build(n1 + n2);
      assertEquals(n1 + n2, merged.size());
      assertTrue(builder.workerThreadIds().size() >= 1);
      assertEquals(0L, builder.writeLockCount());
      assertDegreeAtMost(merged, M * 2);
      assertRecallAtLeast(merged, all, 0.4);
    } finally {
      exec.shutdownNow();
    }
  }

  public void testIncomingRecordedWithoutNeighborWriteLock() throws IOException {
    int n1 = 20;
    int n2 = 20;
    float[][] all = HnswGraphTestCase.createRandomFloatVectors(n1 + n2, 8, random());
    OnHeapHnswGraph g1 = buildGraph(slice(all, 0, n1));
    OnHeapHnswGraph g2 = buildGraph(slice(all, n1, n2));
    int[] map1 = identityMap(n1);
    int[] map2 = new int[n2];
    for (int i = 0; i < n2; i++) {
      map2[i] = n1 + i;
    }
    CombinedHnswGraphBuilder combined =
        CombinedHnswGraphBuilder.fromGraphs(
            scorer(all),
            BEAM,
            random().nextLong(),
            new HnswGraph[] {g1, g2},
            new int[][] {map1, map2},
            n1 + n2,
            null);
    combined.build(n1 + n2);
    assertTrue(
        "incoming edges should be recorded during the forward pass",
        combined.incomingEdges().addCount() > 0);
  }

  private static OnHeapHnswGraph buildGraph(float[][] vectors) throws IOException {
    HnswGraphBuilder.randSeed = 42;
    return HnswGraphBuilder.create(scorer(vectors), M, BEAM, 42L, vectors.length)
        .build(vectors.length);
  }

  private static RandomVectorScorerSupplier scorer(float[][] vectors) throws IOException {
    return DefaultFlatVectorScorer.INSTANCE.getRandomVectorScorerSupplier(
        SIM, MockVectorValues.fromValues(vectors));
  }

  private static int[] identityMap(int n) {
    int[] map = new int[n];
    for (int i = 0; i < n; i++) {
      map[i] = i;
    }
    return map;
  }

  private static float[][] slice(float[][] src, int from, int len) {
    float[][] out = new float[len][];
    System.arraycopy(src, from, out, 0, len);
    return out;
  }

  private static void assertDegreeAtMost(OnHeapHnswGraph graph, int max) throws IOException {
    for (int i = 0; i < graph.size(); i++) {
      if (graph.nodeExistAtLevel(0, i) == false) {
        continue;
      }
      graph.seek(0, i);
      int deg = 0;
      for (int n = graph.nextNeighbor(); n != NO_MORE_DOCS; n = graph.nextNeighbor()) {
        deg++;
      }
      assertTrue("degree " + deg + " > " + max + " at " + i, deg <= max);
    }
  }

  private static void assertRecallAtLeast(OnHeapHnswGraph graph, float[][] vectors, double min)
      throws IOException {
    int k = Math.min(5, vectors.length);
    int queries = Math.min(10, vectors.length);
    int hits = 0;
    int total = queries * k;
    for (int q = 0; q < queries; q++) {
      RandomVectorScorer scorer =
          DefaultFlatVectorScorer.INSTANCE.getRandomVectorScorer(
              SIM, MockVectorValues.fromValues(vectors), vectors[q]);
      KnnCollector nn = HnswGraphSearcher.search(scorer, k, graph, null, Integer.MAX_VALUE);
      TopDocs td = nn.topDocs();
      NeighborQueue expected = new NeighborQueue(k, false);
      for (int i = 0; i < vectors.length; i++) {
        expected.add(i, SIM.compare(vectors[q], vectors[i]));
        if (expected.size() > k) {
          expected.pop();
        }
      }
      int[] exact = expected.nodes();
      java.util.HashSet<Integer> want = new java.util.HashSet<>();
      for (int e : exact) {
        want.add(e);
      }
      for (int i = 0; i < td.scoreDocs.length; i++) {
        if (want.contains(td.scoreDocs[i].doc)) {
          hits++;
        }
      }
    }
    double recall = hits / (double) total;
    assertTrue("recall=" + recall, recall >= min);
  }
}
