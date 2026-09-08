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
import java.util.List;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.Bits;

public class TestRemappingVectorScorer extends LuceneTestCase {

  public void testScoreTranslatesThroughMap() throws IOException {
    List<float[]> merged =
        List.of(new float[] {0f, 0f}, new float[] {3f, 0f}, new float[] {0f, 4f});
    UpdateableRandomVectorScorer delegate =
        DefaultFlatVectorScorer.INSTANCE
            .getRandomVectorScorerSupplier(
                VectorSimilarityFunction.EUCLIDEAN, FloatVectorValues.fromFloats(merged, 2))
            .scorer();
    delegate.setScoringOrdinal(0); // query is {0,0}
    int[] map = new int[] {2, -1, 1}; // source 0->merged 2, 1 deleted, 2->merged 1
    RemappingVectorScorer remap = new RemappingVectorScorer(delegate, map);

    float expected2 = VectorSimilarityFunction.EUCLIDEAN.compare(merged.get(0), merged.get(2));
    float expected1 = VectorSimilarityFunction.EUCLIDEAN.compare(merged.get(0), merged.get(1));
    assertEquals(expected2, remap.score(0), 0f);
    assertEquals(Float.NEGATIVE_INFINITY, remap.score(1), 0f);
    assertEquals(expected1, remap.score(2), 0f);
    assertEquals(Float.NEGATIVE_INFINITY, remap.score(-1), 0f);
    assertEquals(Float.NEGATIVE_INFINITY, remap.score(99), 0f);
    assertEquals(3, remap.maxOrd());
  }

  public void testBulkScoreAndAcceptOrds() throws IOException {
    List<float[]> merged = List.of(new float[] {1f}, new float[] {2f}, new float[] {3f});
    UpdateableRandomVectorScorer delegate =
        DefaultFlatVectorScorer.INSTANCE
            .getRandomVectorScorerSupplier(
                VectorSimilarityFunction.EUCLIDEAN, FloatVectorValues.fromFloats(merged, 1))
            .scorer();
    delegate.setScoringOrdinal(0);
    int[] map = new int[] {0, -1, 2};
    RemappingVectorScorer remap = new RemappingVectorScorer(delegate, map);
    float[] scores = new float[3];
    float max = remap.bulkScore(new int[] {0, 1, 2}, scores, 3);
    assertEquals(scores[0], remap.score(0), 0f);
    assertEquals(Float.NEGATIVE_INFINITY, scores[1], 0f);
    assertEquals(scores[2], remap.score(2), 0f);
    assertEquals(Math.max(scores[0], scores[2]), max, 0f);

    Bits accept = remap.getAcceptOrds(null);
    assertTrue(accept.get(0));
    assertFalse(accept.get(1));
    assertTrue(accept.get(2));
    assertEquals(3, accept.length());
  }
}
