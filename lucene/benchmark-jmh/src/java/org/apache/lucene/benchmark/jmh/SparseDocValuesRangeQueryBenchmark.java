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
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.TotalHitCountCollectorManager;
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
import org.openjdk.jmh.annotations.Threads;
import org.openjdk.jmh.annotations.Warmup;

/** Benchmarks sorted and unsorted skip-index doc-values range queries. */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@State(Scope.Benchmark)
@Threads(1)
@Warmup(iterations = 3, time = 3)
@Measurement(iterations = 5, time = 5)
@Fork(value = 1, warmups = 1)
public class SparseDocValuesRangeQueryBenchmark {

  private static final String FIELD = "number";

  @Param({"sorted", "unsorted"})
  public String indexSort;

  @Param({"0.01", "0.1", "0.5", "1.0"})
  public double selectivity;

  @Param({"1000000", "10000000"})
  public int numDocs;

  @Param({"true", "false"})
  public boolean dense;

  private Directory dir;
  private IndexReader reader;
  private IndexSearcher searcher;
  private Path path;
  private Query query;

  @Setup(Level.Trial)
  public void setup() throws Exception {
    path = Files.createTempDirectory("sparseDocValuesRangeBenchmark");
    dir = MMapDirectory.open(path);

    IndexWriterConfig iwc = new IndexWriterConfig();
    if (indexSort.equals("sorted")) {
      iwc.setIndexSort(new Sort(new SortField(FIELD, SortField.Type.LONG, false, Long.MAX_VALUE)));
    }

    IndexWriter writer = new IndexWriter(dir, iwc);
    for (int docID = 0; docID < numDocs; docID++) {
      Document doc = new Document();
      if (dense || docID % 5 != 0) {
        doc.add(NumericDocValuesField.indexedField(FIELD, docID));
      }
      writer.addDocument(doc);
    }
    writer.forceMerge(1);
    reader = DirectoryReader.open(writer);
    writer.close();
    searcher = new IndexSearcher(reader);

    long rangeSize = Math.max(1, Math.round(numDocs * selectivity));
    long lowerValue = (numDocs - rangeSize) / 2;
    long upperValue = lowerValue + rangeSize - 1;
    query = SortedNumericDocValuesField.newSlowRangeQuery(FIELD, lowerValue, upperValue);
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
  public int searchRange() throws IOException {
    return searcher.search(query, new TotalHitCountCollectorManager(searcher.getSlices()));
  }
}
