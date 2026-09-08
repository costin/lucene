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

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.FixedBitSet;

/** Tests FGIM Phase-1 cross-query merge (search already-inserted source graphs). */
public class TestFgimHnswMerge extends LuceneTestCase {

  private static final int DIM = 8;
  private static final int M = 8;
  private static final int BEAM = 20;
  private static final VectorSimilarityFunction SIM = VectorSimilarityFunction.EUCLIDEAN;

  public void testTwoGraphMergeIsSearchable() throws IOException {
    MergeFixture f = fixture(new int[] {40, 30}, 0);
    OnHeapHnswGraph merged = f.merge();
    assertEquals(70, merged.size());
    assertTrue(recallAt10(merged, f.merged) > 0.5);
  }

  public void testThreeGraphCrossQueryPath() throws IOException {
    // graphs[2] must cross-query graphs[0] and graphs[1] (already inserted).
    MergeFixture f = fixture(new int[] {25, 20, 20}, 0);
    OnHeapHnswGraph merged = f.merge();
    assertEquals(65, merged.size());
    assertTrue(recallAt10(merged, f.merged) > 0.5);
  }

  public void testDeletedOrdinalsAreSkipped() throws IOException {
    MergeFixture f = fixture(new int[] {30, 30}, 6);
    OnHeapHnswGraph merged = f.merge();
    assertEquals(f.merged.size(), merged.size());
    assertTrue(recallAt10(merged, f.merged) > 0.5);
  }

  public void testEmptyBaseRejected() {
    expectThrows(
        IllegalArgumentException.class,
        () ->
            MergingHnswGraphBuilder.fromGraphs(
                supplier(randomVectors(4, 6)),
                BEAM,
                42L,
                new HnswGraph[] {HnswGraph.EMPTY},
                new int[][] {new int[0]},
                0,
                null));
  }

  public void testSingleSourceGraph() throws IOException {
    List<float[]> vecs = randomVectors(40, 1);
    OnHeapHnswGraph g = build(vecs);
    OnHeapHnswGraph merged =
        MergingHnswGraphBuilder.fromGraphs(
                supplier(vecs),
                BEAM,
                42L,
                new HnswGraph[] {g},
                new int[][] {identity(40)},
                40,
                null)
            .build(40);
    assertEquals(40, merged.size());
  }

  public void testUninitializedNodesFromExcludedSegment() throws IOException {
    List<float[]> all = randomVectors(36, 3);
    List<float[]> base = all.subList(0, 20);
    List<float[]> live = new ArrayList<>(base);
    for (int i = 20; i < 36; i++) {
      if (i % 2 == 0) {
        live.add(all.get(i));
      }
    }
    FixedBitSet initialized = new FixedBitSet(live.size());
    for (int i = 0; i < 20; i++) {
      initialized.set(i);
    }
    OnHeapHnswGraph merged =
        MergingHnswGraphBuilder.fromGraphs(
                supplier(live),
                BEAM,
                42L,
                new HnswGraph[] {build(base)},
                new int[][] {identity(20)},
                live.size(),
                initialized)
            .build(live.size());
    assertEquals(live.size(), merged.size());
    assertTrue(recallAt10(merged, live) > 0.5);
  }

  private static MergeFixture fixture(int[] sizes, int deletedInLast) throws IOException {
    List<float[]> merged = new ArrayList<>();
    HnswGraph[] graphs = new HnswGraph[sizes.length];
    int[][] maps = new int[sizes.length][];
    int next = 0;
    int deletedLeft = deletedInLast;
    for (int s = 0; s < sizes.length; s++) {
      List<float[]> seg = randomVectors(sizes[s], 10 + s);
      graphs[s] = build(seg);
      maps[s] = new int[sizes[s]];
      boolean deleteHere = s == sizes.length - 1 && deletedInLast > 0;
      for (int i = 0; i < sizes[s]; i++) {
        if (deleteHere && deletedLeft > 0 && i % 3 == 0) {
          maps[s][i] = -1;
          deletedLeft--;
        } else {
          maps[s][i] = next++;
          merged.add(seg.get(i));
        }
      }
    }
    return new MergeFixture(graphs, maps, merged);
  }

  private static OnHeapHnswGraph build(List<float[]> vecs) throws IOException {
    HnswGraphBuilder.randSeed = 42L;
    return HnswGraphBuilder.create(supplier(vecs), M, BEAM, 42L, vecs.size()).build(vecs.size());
  }

  private static RandomVectorScorerSupplier supplier(List<float[]> vecs) throws IOException {
    return DefaultFlatVectorScorer.INSTANCE.getRandomVectorScorerSupplier(
        SIM, FloatVectorValues.fromFloats(vecs, DIM));
  }

  private static List<float[]> randomVectors(int n, long seed) {
    Random r = new Random(seed);
    List<float[]> out = new ArrayList<>(n);
    for (int i = 0; i < n; i++) {
      float[] v = new float[DIM];
      for (int d = 0; d < DIM; d++) {
        v[d] = r.nextFloat();
      }
      out.add(v);
    }
    return out;
  }

  private static int[] identity(int n) {
    int[] m = new int[n];
    for (int i = 0; i < n; i++) {
      m[i] = i;
    }
    return m;
  }

  private static double recallAt10(OnHeapHnswGraph graph, List<float[]> vecs) throws IOException {
    Random r = new Random(123);
    int k = Math.min(10, vecs.size());
    int overlap = 0;
    FloatVectorValues values = FloatVectorValues.fromFloats(vecs, DIM);
    for (int q = 0; q < 20; q++) {
      float[] query = new float[DIM];
      for (int d = 0; d < DIM; d++) {
        query[d] = r.nextFloat();
      }
      KnnCollector nn =
          HnswGraphSearcher.search(
              DefaultFlatVectorScorer.INSTANCE.getRandomVectorScorer(SIM, values, query),
              k,
              graph,
              null,
              Integer.MAX_VALUE);
      TopDocs td = nn.topDocs();
      NeighborQueue exact = new NeighborQueue(k, false);
      for (int i = 0; i < vecs.size(); i++) {
        exact.add(i, SIM.compare(query, vecs.get(i)));
        if (exact.size() > k) {
          exact.pop();
        }
      }
      java.util.HashSet<Integer> want = new java.util.HashSet<>();
      for (int n : exact.nodes()) {
        want.add(n);
      }
      for (var sd : td.scoreDocs) {
        if (want.contains(sd.doc)) {
          overlap++;
        }
      }
    }
    return overlap / (20.0 * k);
  }

  private static final class MergeFixture {
    final HnswGraph[] graphs;
    final int[][] maps;
    final List<float[]> merged;

    MergeFixture(HnswGraph[] graphs, int[][] maps, List<float[]> merged) {
      this.graphs = graphs;
      this.maps = maps;
      this.merged = merged;
    }

    OnHeapHnswGraph merge() throws IOException {
      return MergingHnswGraphBuilder.fromGraphs(
              supplier(merged), BEAM, 42L, graphs, maps, merged.size(), null)
          .build(merged.size());
    }
  }
}
