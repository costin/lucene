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

import java.util.concurrent.atomic.AtomicLong;
import org.apache.lucene.util.ArrayUtil;

/**
 * Per-node incoming-edge lists for the HNSW-Merger backward-connect pass.
 *
 * <p>Appends are synchronized on a stripe (not {@link HnswLock#write}, which serializes
 * reverse-link updates on the neighbor arrays themselves). After the forward insert pass the lists
 * are read-only.
 *
 * @lucene.internal
 */
final class IncomingEdges {
  private static final int STRIPES = 256;
  private static final int[] EMPTY = new int[0];

  private final int[][] lists;
  private final int[] sizes;
  private final Object[] stripeLocks;
  private final AtomicLong addWaitNanos = new AtomicLong();
  private final AtomicLong addCount = new AtomicLong();

  IncomingEdges(int numNodes) {
    if (numNodes < 0) {
      throw new IllegalArgumentException("numNodes must be >= 0");
    }
    this.lists = new int[numNodes][];
    this.sizes = new int[numNodes];
    this.stripeLocks = new Object[STRIPES];
    for (int i = 0; i < STRIPES; i++) {
      stripeLocks[i] = new Object();
    }
  }

  /** Record that {@code source} selected {@code target} as a forward neighbor. */
  void add(int target, int source) {
    if (target < 0 || target >= lists.length || target == source) {
      return;
    }
    long t0 = System.nanoTime();
    synchronized (stripeLocks[target & (STRIPES - 1)]) {
      addWaitNanos.addAndGet(System.nanoTime() - t0);
      addCount.incrementAndGet();
      int[] a = lists[target];
      int n = sizes[target];
      if (a == null) {
        a = new int[4];
        lists[target] = a;
      } else if (n == a.length) {
        a = ArrayUtil.grow(a, n + 1);
        lists[target] = a;
      }
      a[n] = source;
      sizes[target] = n + 1;
    }
  }

  int[] get(int target) {
    if (target < 0 || target >= lists.length) {
      return EMPTY;
    }
    int[] a = lists[target];
    return a == null ? EMPTY : a;
  }

  int size(int target) {
    if (target < 0 || target >= sizes.length) {
      return 0;
    }
    return sizes[target];
  }

  int numNodes() {
    return lists.length;
  }

  long addWaitNanos() {
    return addWaitNanos.get();
  }

  long addCount() {
    return addCount.get();
  }
}
