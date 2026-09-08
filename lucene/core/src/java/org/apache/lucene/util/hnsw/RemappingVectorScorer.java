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
import org.apache.lucene.util.Bits;

/**
 * Scores source-graph ordinals by translating them through an ordinal map into the merged vector
 * space. Deleted source ordinals ({@code ordMap[node] == -1}) score as {@link
 * Float#NEGATIVE_INFINITY} so they are never selected as neighbours.
 *
 * <p>Used by FGIM Phase-1 cross-query: search a source HNSW graph whose ordinals are local, while
 * the query vector lives in the merged graph.
 *
 * @lucene.experimental
 */
public final class RemappingVectorScorer implements RandomVectorScorer {
  private final RandomVectorScorer delegate;
  private final int[] ordMap;

  public RemappingVectorScorer(RandomVectorScorer delegate, int[] ordMap) {
    this.delegate = delegate;
    this.ordMap = ordMap;
  }

  @Override
  public float score(int node) throws IOException {
    int mapped = mappedOrd(node);
    if (mapped < 0) {
      return Float.NEGATIVE_INFINITY;
    }
    return delegate.score(mapped);
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
    return ordMap.length;
  }

  @Override
  public int ordToDoc(int ord) {
    return ord;
  }

  @Override
  public Bits getAcceptOrds(Bits acceptDocs) {
    return new Bits() {
      @Override
      public boolean get(int index) {
        if (mappedOrd(index) < 0) {
          return false;
        }
        return acceptDocs == null || acceptDocs.get(index);
      }

      @Override
      public int length() {
        return ordMap.length;
      }
    };
  }

  private int mappedOrd(int node) {
    if (node < 0 || node >= ordMap.length) {
      return -1;
    }
    return ordMap[node];
  }
}
