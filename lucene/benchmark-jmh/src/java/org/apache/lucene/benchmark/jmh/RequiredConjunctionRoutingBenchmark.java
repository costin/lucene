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
package org.apache.lucene.benchmark.jmh;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field.Store;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause.Occur;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BulkScorer;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.DocIdStream;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.LeafCollector;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.Scorable;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.Weight;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.MMapDirectory;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

/**
 * Measures routing of non-dense required conjunctions through {@code DenseConjunctionBulkScorer}
 * (lowered density threshold) and {@code ConstantScoreBulkScorer} instead of {@code
 * DefaultBulkScorer}.
 *
 * <p>Query shape: {@code FILTER(term @ selectivity) AND FILTER(dv_range @ dvRangeSelectivity)}.
 * The term controls the lead cost (kept below maxDoc/32 for non-dense). The dv_range produces a
 * {@code TwoPhaseIterator} via {@code DocValuesRangeIterator}. Lower {@code dvRangeSelectivity}
 * means more NO/MAYBE skip blocks where bulk {@code intoBitSet} wins over per-doc evaluation.
 *
 * <p>A {@code TermQuery} is used instead of {@code MatchAllDocsQuery} because {@code
 * MatchAllDocsQuery} gets rewritten away by {@code IndexSearcher}.
 */
@State(Scope.Thread)
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 3, time = 3)
@Measurement(iterations = 5, time = 5)
@Fork(
    value = 1,
    warmups = 1,
    jvmArgsAppend = {"-Xmx2g", "-Xms2g", "-XX:+AlwaysPreTouch"})
public class RequiredConjunctionRoutingBenchmark {

  private static final int DOC_COUNT = 1_000_000;
  private static final String LEAD_FIELD = "lead";
  private static final String VALUE_FIELD = "value";
  private static final String YES = "yes";

  private Directory dir;
  private IndexReader reader;
  private Path path;
  private IndexSearcher searcher;
  private Query query;
  private LeafReaderContext context;
  private int expectedHitCount;

  @Param({"0.005", "0.01", "0.02"})
  double selectivity;

  @Param({"0.2", "0.5", "0.9"})
  double dvRangeSelectivity;

  @Setup(Level.Trial)
  public void setup() throws Exception {
    path = Files.createTempDirectory("requiredConjunctionRoutingBench");
    dir = MMapDirectory.open(path);

    Random random = new Random(42);

    // The lead term controls conjunction selectivity. The dv_range selectivity controls how many
    // docs pass the TwoPhaseIterator — lower dvRangeSelectivity means more NO/MAYBE skip blocks,
    // which is where DenseConjunctionBulkScorer's per-clause bulk intoBitSet wins.
    long[] values = new long[DOC_COUNT];
    for (int i = 0; i < DOC_COUNT; i++) {
      values[i] = random.nextLong();
    }
    long[] sorted = values.clone();
    Arrays.sort(sorted);
    int rangeCount = Math.max(1, (int) (DOC_COUNT * dvRangeSelectivity));
    long lowerBound = sorted[(DOC_COUNT - rangeCount) / 2];
    long upperBound = sorted[(DOC_COUNT + rangeCount) / 2 - 1];

    Random sparseRandom = new Random(43);

    IndexWriter w = new IndexWriter(dir, new IndexWriterConfig());
    for (int i = 0; i < DOC_COUNT; i++) {
      Document doc = new Document();
      if (sparseRandom.nextDouble() < selectivity) {
        doc.add(new StringField(LEAD_FIELD, YES, Store.NO));
      }
      doc.add(new NumericDocValuesField(VALUE_FIELD, values[i]));
      w.addDocument(doc);
    }
    w.forceMerge(1);
    reader = DirectoryReader.open(w);
    w.close();

    searcher = new IndexSearcher(reader);
    searcher.setQueryCache(null);

    context = reader.leaves().get(0);

    Query leadTermQuery = new TermQuery(new Term(LEAD_FIELD, YES));
    Query dvRangeQuery =
        NumericDocValuesField.newSlowRangeQuery(VALUE_FIELD, lowerBound, upperBound);

    query =
        new BooleanQuery.Builder()
            .add(leadTermQuery, Occur.FILTER)
            .add(dvRangeQuery, Occur.FILTER)
            .build();

    Weight weight =
        searcher.createWeight(searcher.rewrite(query), ScoreMode.COMPLETE_NO_SCORES, 1f);

    ScorerSupplier scorerSupplier = weight.scorerSupplier(context);
    if (scorerSupplier != null) {
      BulkScorer bulkScorer = scorerSupplier.bulkScorer();
      System.out.println(
          "[RequiredConjunctionRoutingBenchmark] selectivity="
              + selectivity
              + " bulkScorer="
              + bulkScorer.getClass().getSimpleName()
              + " leadCost="
              + scorerSupplier.cost());
    }

    expectedHitCount = searcher.count(query);

    checkHitCount(countViaBulkScorer());
    checkHitCount(countViaPerDoc());
  }

  @Benchmark
  public int countBulkScorer() throws IOException {
    return countViaBulkScorer();
  }

  @Benchmark
  public int countPerDoc() throws IOException {
    return countViaPerDoc();
  }

  private int countViaBulkScorer() throws IOException {
    Weight w =
        searcher.createWeight(searcher.rewrite(query), ScoreMode.COMPLETE_NO_SCORES, 1f);
    BulkScorer bulkScorer = w.bulkScorer(context);
    if (bulkScorer == null) {
      return 0;
    }
    CountingLeafCollector collector = new CountingLeafCollector();
    bulkScorer.score(collector, context.reader().getLiveDocs(), 0, context.reader().maxDoc());
    return collector.count;
  }

  private int countViaPerDoc() throws IOException {
    Weight w =
        searcher.createWeight(searcher.rewrite(query), ScoreMode.COMPLETE_NO_SCORES, 1f);
    ScorerSupplier scorerSupplier = w.scorerSupplier(context);
    if (scorerSupplier == null) {
      return 0;
    }
    Scorer scorer = scorerSupplier.get(Long.MAX_VALUE);
    DocIdSetIterator iterator = scorer.iterator();
    int maxDoc = context.reader().maxDoc();
    CountingLeafCollector collector = new CountingLeafCollector();
    for (int doc = iterator.nextDoc(); doc < maxDoc; doc = iterator.nextDoc()) {
      collector.collect(doc);
    }
    return collector.count;
  }

  private void checkHitCount(int hitCount) {
    if (hitCount != expectedHitCount) {
      throw new AssertionError("hitCount=" + hitCount + " expected=" + expectedHitCount);
    }
  }

  private static class CountingLeafCollector implements LeafCollector {
    int count;

    @Override
    public void setScorer(Scorable scorer) {}

    @Override
    public void collect(int doc) {
      ++count;
    }

    @Override
    public void collect(DocIdStream stream) throws IOException {
      for (int streamCount = stream.count(); streamCount != 0; streamCount = stream.count()) {
        count += streamCount;
      }
    }
  }

  @TearDown(Level.Trial)
  public void tearDown() throws IOException {
    if (reader != null) {
      reader.close();
    }
    if (dir != null) {
      dir.close();
    }
    if (path != null && Files.exists(path)) {
      try (Stream<Path> walk = Files.walk(path)) {
        walk.sorted(Comparator.reverseOrder())
            .forEach(
                p -> {
                  try {
                    Files.delete(p);
                  } catch (IOException _) {
                  }
                });
      }
    }
  }
}
