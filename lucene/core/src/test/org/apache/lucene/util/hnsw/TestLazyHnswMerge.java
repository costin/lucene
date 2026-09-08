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

/** Tests lazy backward-connect merge and both repair modes. */
public class TestLazyHnswMerge extends LuceneTestCase {

  private static final int DIM = 8;
  private static final int M = 8;
  private static final int BEAM = 20;
  private static final VectorSimilarityFunction SIM = VectorSimilarityFunction.EUCLIDEAN;

  public void testLazyFlagSkipsReverseLinks() throws IOException {
    List<float[]> vecs = randomVectors(40, 1);
    HnswGraphBuilder builder = HnswGraphBuilder.create(supplier(vecs), M, BEAM, 42L, vecs.size());
    builder.lazyBackwardConnect = true;
    OnHeapHnswGraph g = builder.build(vecs.size());
    int reverseMissing = countMissingReverse(g);
    assertTrue(
        "forward-only construction should leave missing reverse edges, saw " + reverseMissing,
        reverseMissing > 0);
  }

  public void testDefaultBuildIsMostlySymmetric() throws IOException {
    List<float[]> vecs = randomVectors(40, 2);
    OnHeapHnswGraph g = build(vecs);
    // Diversity pruning can leave a few one-way edges, but far fewer than lazy-without-repair.
    assertTrue(countMissingReverse(g) < 40);
  }

  public void testLightRepairMergeIsSearchable() throws IOException {
    MergingHnswGraphBuilder.lazyRepairMode = MergingHnswGraphBuilder.LazyRepairMode.LIGHT;
    MergeFixture f = fixture(40, 30, 0);
    OnHeapHnswGraph merged = f.merge();
    assertEquals(70, merged.size());
    assertTrue(recallAt10(merged, f.merged) > 0.5);
  }

  public void testThoroughRepairMergeIsSearchable() throws IOException {
    MergingHnswGraphBuilder.lazyRepairMode = MergingHnswGraphBuilder.LazyRepairMode.THOROUGH;
    try {
      MergeFixture f = fixture(40, 30, 0);
      OnHeapHnswGraph merged = f.merge();
      assertEquals(70, merged.size());
      assertTrue(recallAt10(merged, f.merged) > 0.5);
    } finally {
      MergingHnswGraphBuilder.lazyRepairMode = MergingHnswGraphBuilder.LazyRepairMode.LIGHT;
    }
  }

  public void testDeletedOrdinalsAreSkipped() throws IOException {
    MergingHnswGraphBuilder.lazyRepairMode = MergingHnswGraphBuilder.LazyRepairMode.LIGHT;
    MergeFixture f = fixture(30, 30, 6);
    OnHeapHnswGraph merged = f.merge();
    assertEquals(f.merged.size(), merged.size());
    assertTrue(recallAt10(merged, f.merged) > 0.5);
  }

  public void testSingleSourceAndUninitializedNodes() throws IOException {
    List<float[]> all = randomVectors(36, 4);
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

  private static int countMissingReverse(OnHeapHnswGraph g) {
    int missing = 0;
    for (int u = 0; u < g.size(); u++) {
      NeighborArray nu = g.getNeighbors(0, u);
      for (int i = 0; i < nu.size(); i++) {
        int v = nu.nodes()[i];
        NeighborArray nv = g.getNeighbors(0, v);
        boolean found = false;
        for (int j = 0; j < nv.size(); j++) {
          if (nv.nodes()[j] == u) {
            found = true;
            break;
          }
        }
        if (!found) {
          missing++;
        }
      }
    }
    return missing;
  }

  private static MergeFixture fixture(int nA, int nB, int deletedInB) throws IOException {
    List<float[]> a = randomVectors(nA, 10);
    List<float[]> b = randomVectors(nB, 11);
    List<float[]> merged = new ArrayList<>(a);
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
    return new MergeFixture(
        new HnswGraph[] {build(a), build(b)}, new int[][] {identity(nA), mapB}, merged);
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
