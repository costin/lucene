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
import java.util.Comparator;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.ConstantScoreScorerSupplier;
import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.ScorerSupplier;
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
 * Benchmarks {@link FieldExistsQuery} when sparse indexed doc values route through constant-score
 * dense bulk scoring.
 */
@State(Scope.Thread)
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 3, time = 3)
@Measurement(iterations = 5, time = 5)
@Fork(value = 1, warmups = 1)
public class FieldExistsQueryBenchmark {

  private static final String FIELD = "dv";

  private Directory dir;
  private IndexReader reader;
  private Path path;
  private Query query;

  @State(Scope.Benchmark)
  public static class Params {
    @Param({"100000", "1000000"})
    public int docCount;

    @Param({"2", "4"})
    public int valueFrequency;
  }

  @Setup(Level.Trial)
  public void setup(Params params) throws Exception {
    path = Files.createTempDirectory("fieldExistsQueryBench");
    dir = MMapDirectory.open(path);

    IndexWriterConfig iwc = new IndexWriterConfig();
    try (IndexWriter w = new IndexWriter(dir, iwc)) {
      for (int i = 0; i < params.docCount; i++) {
        Document doc = new Document();
        if (i % params.valueFrequency == 0) {
          doc.add(NumericDocValuesField.indexedField(FIELD, i));
        }
        w.addDocument(doc);
      }
      w.forceMerge(1);
      reader = DirectoryReader.open(w);
    }

    query = new FieldExistsQuery(FIELD);
    verifyBenchmarkPath();
  }

  private void verifyBenchmarkPath() throws IOException {
    IndexSearcher searcher = new IndexSearcher(reader);
    LeafReaderContext leaf = reader.leaves().get(0);
    if (leaf.reader().getDocValuesSkipper(FIELD) == null) {
      throw new IllegalStateException("benchmark requires a doc-values skipper");
    }
    Query rewritten = query.rewrite(searcher);
    if (rewritten instanceof MatchAllDocsQuery) {
      throw new IllegalStateException("benchmark query must not rewrite to MatchAllDocsQuery");
    }

    Weight weight = query.createWeight(searcher, ScoreMode.COMPLETE_NO_SCORES, 1f);
    ScorerSupplier scorerSupplier = weight.scorerSupplier(leaf);
    if (scorerSupplier instanceof ConstantScoreScorerSupplier == false) {
      throw new IllegalStateException(
          "expected ConstantScoreScorerSupplier but got " + scorerSupplier.getClass().getName());
    }
    String bulkScorerClass = scorerSupplier.bulkScorer().getClass().getName();
    if (bulkScorerClass.endsWith(".DenseConjunctionBulkScorer") == false) {
      throw new IllegalStateException(
          "expected DenseConjunctionBulkScorer but got " + bulkScorerClass);
    }
  }

  @TearDown(Level.Trial)
  public void tearDown() throws Exception {
    reader.close();
    if (dir != null) {
      dir.close();
      dir = null;
    }
    if (Files.exists(path)) {
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

  @Benchmark
  public int searchFieldExists(Params params) throws IOException {
    IndexSearcher searcher = new IndexSearcher(reader);
    return searcher.count(query);
  }
}
