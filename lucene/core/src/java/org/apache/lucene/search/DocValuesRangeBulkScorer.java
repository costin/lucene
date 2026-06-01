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
package org.apache.lucene.search;

import java.io.IOException;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.search.SkipBlockRangeIterator.Match;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.MathUtil;

/** Bulk scorer for singleton numeric doc-values range queries backed by a skip index. */
final class DocValuesRangeBulkScorer extends BulkScorer {

  private final SkipBlockRangeIterator blockIterator;
  private final NumericDocValues values;
  private final long minValue;
  private final long maxValue;
  private final SimpleScorable scorer = new SimpleScorable();
  private final FixedBitSet windowMatches = new FixedBitSet(DenseConjunctionBulkScorer.WINDOW_SIZE);

  DocValuesRangeBulkScorer(
      float score,
      SkipBlockRangeIterator blockIterator,
      NumericDocValues values,
      long minValue,
      long maxValue) {
    this.blockIterator = blockIterator;
    this.values = values;
    this.minValue = minValue;
    this.maxValue = maxValue;
    scorer.setScore(score);
  }

  @Override
  public int score(LeafCollector collector, Bits acceptDocs, int min, int max) throws IOException {
    collector.setScorer(scorer);
    DocIdSetIterator competitiveIterator = collector.competitiveIterator();
    if (competitiveIterator != null) {
      return scoreCompetitiveIterator(collector, acceptDocs, competitiveIterator, min, max);
    }

    if (blockIterator.docID() < min) {
      blockIterator.advance(min);
    }
    for (int doc = blockIterator.docID(); doc < max; doc = blockIterator.docID()) {
      Match match = blockIterator.getMatch();
      int blockEnd =
          Math.min(
              max, match == Match.YES ? blockIterator.docIDRunEnd() : blockIterator.blockEnd());
      switch (match) {
        case YES:
          if (acceptDocs == null) {
            collector.collectRange(doc, blockEnd);
          } else {
            collectAcceptedRange(collector, acceptDocs, doc, blockEnd);
          }
          break;
        case YES_IF_PRESENT:
        case MAYBE:
          collectIntoBitSet(collector, acceptDocs, match, doc, blockEnd);
          break;
      }
      blockIterator.advance(blockEnd);
    }
    return blockIterator.docID();
  }

  @Override
  public long cost() {
    return values.cost();
  }

  private int scoreCompetitiveIterator(
      LeafCollector collector,
      Bits acceptDocs,
      DocIdSetIterator competitiveIterator,
      int min,
      int max)
      throws IOException {
    if (competitiveIterator.docID() > min) {
      min = Math.min(competitiveIterator.docID(), max);
    }
    if (blockIterator.docID() < min) {
      blockIterator.advance(min);
    }
    for (int doc = blockIterator.docID(); doc < max; ) {
      assert competitiveIterator.docID() <= doc;
      if (competitiveIterator.docID() < doc) {
        int competitiveNext = competitiveIterator.advance(doc);
        if (competitiveNext != doc) {
          doc = blockIterator.advance(competitiveNext);
          continue;
        }
      }

      Match match = blockIterator.getMatch();
      int blockEnd =
          Math.min(
              max, match == Match.YES ? blockIterator.docIDRunEnd() : blockIterator.blockEnd());
      if ((acceptDocs == null || acceptDocs.get(doc)) && matches(match, doc)) {
        collector.collect(doc);
      }

      doc++;
      if (doc >= blockEnd) {
        doc = blockIterator.advance(blockEnd);
      }
    }
    return blockIterator.docID();
  }

  private boolean matches(Match match, int doc) throws IOException {
    switch (match) {
      case YES:
        return true;
      case YES_IF_PRESENT:
        return values.advanceExact(doc);
      case MAYBE:
        if (values.advanceExact(doc)) {
          long value = values.longValue();
          return value >= minValue && value <= maxValue;
        }
        return false;
      default:
        throw new AssertionError(match);
    }
  }

  private void collectIntoBitSet(
      LeafCollector collector, Bits acceptDocs, Match match, int start, int end)
      throws IOException {
    for (int windowBase = start; windowBase < end; ) {
      int windowMax =
          MathUtil.unsignedMin(end, windowBase + DenseConjunctionBulkScorer.WINDOW_SIZE);
      assert windowMatches.scanIsEmpty();
      if (match == Match.YES_IF_PRESENT) {
        for (int doc = windowBase; doc < windowMax; doc++) {
          if (values.advanceExact(doc)) {
            windowMatches.set(doc - windowBase);
          }
        }
      } else {
        values.rangeIntoBitSet(
            windowBase, windowMax, minValue, maxValue, windowMatches, windowBase);
      }
      if (acceptDocs != null) {
        acceptDocs.applyMask(windowMatches, windowBase);
      }
      collector.collect(new BitSetDocIdStream(windowMatches, windowBase));
      windowMatches.clear();
      windowBase = windowMax;
    }
  }

  private static void collectAcceptedRange(
      LeafCollector collector, Bits acceptDocs, int start, int end) throws IOException {
    int rangeStart = -1;
    for (int doc = start; doc < end; doc++) {
      if (acceptDocs.get(doc)) {
        if (rangeStart < 0) {
          rangeStart = doc;
        }
      } else if (rangeStart >= 0) {
        collector.collectRange(rangeStart, doc);
        rangeStart = -1;
      }
    }
    if (rangeStart >= 0) {
      collector.collectRange(rangeStart, end);
    }
  }
}
