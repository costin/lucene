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
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause.Occur;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
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

/** Benchmarks top-score disjunctions with and without FILTER clauses. */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 3, time = 3)
@Measurement(iterations = 5, time = 5)
@Fork(value = 1, warmups = 1)
public class MaxScoreBulkScorerBenchmark {

  private static final String TEXT_FIELD = "text";
  private static final String FILTER_FIELD = "filter";
  private static final String FILTER_VALUE = "yes";
  private static final int MAX_CLAUSE_COUNT = 45;

  @Param({"1000000"})
  public int docCount;

  @Param({"12", "45"})
  public int clauseCount;

  @Param({"none", "0.01", "0.03", "0.03125", "0.05", "0.1", "0.2", "0.5", "0.8", "0.95"})
  public String filterSelectivity;

  private Directory dir;
  private IndexReader reader;
  private IndexSearcher searcher;
  private Path path;
  private Query query;

  @Setup(Level.Trial)
  public void setup() throws Exception {
    path = Files.createTempDirectory("maxScoreBulkScorerBenchmark");
    dir = MMapDirectory.open(path);

    try (IndexWriter writer = new IndexWriter(dir, new IndexWriterConfig())) {
      for (int docID = 0; docID < docCount; docID++) {
        Document doc = new Document();
        for (int slot = 0; slot < 3; slot++) {
          int term = Math.floorMod(docID * (slot * 17 + 3) + slot, MAX_CLAUSE_COUNT);
          doc.add(new StringField(TEXT_FIELD, "t" + term, Field.Store.NO));
        }
        if (matchesFilter(docID)) {
          doc.add(new StringField(FILTER_FIELD, FILTER_VALUE, Field.Store.NO));
        }
        writer.addDocument(doc);
      }
      writer.forceMerge(1);
      reader = DirectoryReader.open(writer);
    }

    searcher = new IndexSearcher(reader);
    query = buildQuery();
  }

  private Query buildQuery() {
    BooleanQuery.Builder builder = new BooleanQuery.Builder();
    for (int i = 0; i < clauseCount; i++) {
      Query termQuery = new TermQuery(new Term(TEXT_FIELD, "t" + i));
      builder.add(new BoostQuery(termQuery, 1f + (i % 5) * 0.2f), Occur.SHOULD);
    }
    // Keep this at exactly 1: filtered optional clauses only route to MaxScoreBulkScorer when
    // BooleanScorerSupplier.filteredOptionalBulkScorer() sees minShouldMatch == 1.
    builder.setMinimumNumberShouldMatch(1);
    if (filterSelectivity.equals("none") == false) {
      builder.add(new TermQuery(new Term(FILTER_FIELD, FILTER_VALUE)), Occur.FILTER);
    }
    return builder.build();
  }

  private boolean matchesFilter(int docID) {
    return switch (filterSelectivity) {
      case "none" -> false;
      case "0.01" -> docID % 100 == 0;
      case "0.03" -> docID % 100 < 3;
      case "0.03125" -> docID % 32 == 0;
      case "0.05" -> docID % 20 == 0;
      case "0.1" -> docID % 10 == 0;
      case "0.2" -> docID % 5 == 0;
      case "0.5" -> docID % 2 == 0;
      case "0.8" -> docID % 5 != 0;
      case "0.95" -> docID % 20 != 0;
      default -> throw new IllegalArgumentException("Unknown selectivity: " + filterSelectivity);
    };
  }

  @TearDown(Level.Trial)
  public void tearDown() throws Exception {
    reader.close();
    dir.close();
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
  public int searchTopHits() throws IOException {
    TopDocs topDocs = searcher.search(query, 10);
    return topDocs.scoreDocs.length;
  }
}
