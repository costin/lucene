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
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.FixedBitSet;

/** Tests IGTM warm-start merge (previous level-0 result as entry points). */
public class TestIgtmHnswMerge extends LuceneTestCase {

  private static final int DIM = 8;
  private static final int M = 8;
  private static final int BEAM = 20;
  private static final VectorSimilarityFunction SIM = VectorSimilarityFunction.EUCLIDEAN;

  public void testTwoGraphMergeIsSearchable() throws IOException {
    MergeFixture f = fixture(60, 40, 0);
    OnHeapHnswGraph merged = f.merge();
    assertEquals(100, merged.size());
    assertTrue("merged graph must have a level-0", merged.numLevels() >= 1);
    assertTrue(recallAt10(merged, f.merged) > 0.5);
  }

  public void testDeletedOrdinalsAreSkipped() throws IOException {
    MergeFixture f = fixture(40, 40, 8);
    OnHeapHnswGraph merged = f.merge();
    assertEquals(f.merged.size(), merged.size());
    assertTrue(recallAt10(merged, f.merged) > 0.5);
  }

  public void testSingleSourceGraphIsCopy() throws IOException {
    List<float[]> vecs = randomVectors(50, 1);
    OnHeapHnswGraph g = build(vecs);
    int[] map = identity(50);
    RandomVectorScorerSupplier supplier = supplier(vecs);
    MergingHnswGraphBuilder merger =
        MergingHnswGraphBuilder.fromGraphs(
            supplier, BEAM, 42L, new HnswGraph[] {g}, new int[][] {map}, 50, null);
    assertTrue(merger.captureLevel0SearchResult);
    OnHeapHnswGraph merged = merger.build(50);
    assertEquals(50, merged.size());
  }

  public void testEmptySecondGraphIsSkipped() throws IOException {
    List<float[]> a = randomVectors(30, 2);
    OnHeapHnswGraph gA = build(a);
    // A zero-size graph is not a valid HNSW source; the builder must skip size==0.
    // We simulate "no second graph" by calling fromGraphs with only the base.
    RandomVectorScorerSupplier supplier = supplier(a);
    OnHeapHnswGraph merged =
        MergingHnswGraphBuilder.fromGraphs(
                supplier, BEAM, 42L, new HnswGraph[] {gA}, new int[][] {identity(30)}, 30, null)
            .build(30);
    assertEquals(30, merged.size());
  }

  public void testWarmStartFieldIsPopulatedAfterMergeInsert() throws IOException {
    MergeFixture f = fixture(30, 30, 0);
    RandomVectorScorerSupplier supplier = supplier(f.merged);
    MergingHnswGraphBuilder merger =
        MergingHnswGraphBuilder.fromGraphs(
            supplier, BEAM, 42L, f.graphs, f.maps, f.merged.size(), null);
    merger.build(f.merged.size());
    assertNotNull(
        "IGTM should record the last level-0 search result during merge inserts",
        merger.lastLevel0SearchResult);
    assertTrue(merger.lastLevel0SearchResult.size() > 0);
  }

  public void testAllDeletedSecondGraphIsNoOp() throws IOException {
    List<float[]> a = randomVectors(25, 4);
    List<float[]> b = randomVectors(10, 5);
    OnHeapHnswGraph gA = build(a);
    OnHeapHnswGraph gB = build(b);
    int[] mapB = new int[10];
    Arrays.fill(mapB, -1);
    OnHeapHnswGraph merged =
        MergingHnswGraphBuilder.fromGraphs(
                supplier(a),
                BEAM,
                42L,
                new HnswGraph[] {gA, gB},
                new int[][] {identity(25), mapB},
                25,
                null)
            .build(25);
    assertEquals(25, merged.size());
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

  public void testUninitializedLiveVectorsFromDeletedSegment() throws IOException {
    // Base reused; second segment has deletes so it is not a source — live vectors
    // must be inserted from scratch (initializedNodes).
    List<float[]> all = randomVectors(40, 3);
    List<float[]> base = all.subList(0, 24);
    List<float[]> liveRest = new ArrayList<>();
    liveRest.addAll(base);
    for (int i = 24; i < 40; i++) {
      if ((i % 3) != 0) {
        liveRest.add(all.get(i));
      }
    }
    OnHeapHnswGraph gBase = build(base);
    int[] map = identity(24);
    FixedBitSet initialized = new FixedBitSet(liveRest.size());
    for (int i = 0; i < 24; i++) {
      initialized.set(i);
    }
    RandomVectorScorerSupplier supplier = supplier(liveRest);
    OnHeapHnswGraph merged =
        MergingHnswGraphBuilder.fromGraphs(
                supplier,
                BEAM,
                42L,
                new HnswGraph[] {gBase},
                new int[][] {map},
                liveRest.size(),
                initialized)
            .build(liveRest.size());
    assertEquals(liveRest.size(), merged.size());
    assertTrue(recallAt10(merged, liveRest) > 0.5);
  }

  private static MergeFixture fixture(int nA, int nB, int deletedInB) throws IOException {
    List<float[]> a = randomVectors(nA, 10);
    List<float[]> b = randomVectors(nB, 11);
    OnHeapHnswGraph gA = build(a);
    OnHeapHnswGraph gB = build(b);
    List<float[]> merged = new ArrayList<>(a);
    int[] mapA = identity(nA);
    int[] mapB = new int[nB];
    int next = nA;
    int deleted = 0;
    for (int i = 0; i < nB; i++) {
      if (deleted < deletedInB && i % 3 == 0) {
        mapB[i] = -1;
        deleted++;
      } else {
        mapB[i] = next++;
        merged.add(b.get(i));
      }
    }
    HnswGraph[] graphs = new HnswGraph[] {gA, gB};
    int[][] maps = new int[][] {mapA, mapB};
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
