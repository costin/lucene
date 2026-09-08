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

import java.util.concurrent.locks.Lock;
import org.apache.lucene.util.BitSet;

/**
 * Copies a neighbor list under {@link HnswLock#read} so concurrent modification of the on-heap
 * graph does not tear a search. Used by {@link HnswConcurrentMergeBuilder} and the combined merger.
 */
final class LockedHnswGraphSearcher extends HnswGraphSearcher {
  private final HnswLock hnswLock;
  private int[] nodeBuffer;
  private int upto;
  private int size;

  LockedHnswGraphSearcher(NeighborQueue candidates, HnswLock hnswLock, BitSet visited) {
    super(candidates, visited);
    this.hnswLock = hnswLock;
  }

  @Override
  void graphSeek(HnswGraph graph, int level, int targetNode) {
    Lock lock = hnswLock.read(level, targetNode);
    try {
      NeighborArray neighborArray = ((OnHeapHnswGraph) graph).getNeighbors(level, targetNode);
      if (nodeBuffer == null || nodeBuffer.length < neighborArray.size()) {
        nodeBuffer = new int[neighborArray.size()];
      }
      size = neighborArray.size();
      System.arraycopy(neighborArray.nodes(), 0, nodeBuffer, 0, size);
    } finally {
      lock.unlock();
    }
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
