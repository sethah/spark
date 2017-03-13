/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.sql.streaming

import java.util.concurrent.CountDownLatch

import org.apache.commons.lang3.RandomStringUtils
import org.scalactic.TolerantNumerics
import org.scalatest.concurrent.Eventually._
import org.scalatest.BeforeAndAfter
import org.scalatest.concurrent.PatienceConfiguration.Timeout
import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{AttributeReference, GenericInternalRow, UnsafeProjection, UnsafeRow}
import org.apache.spark.sql.catalyst.expressions.codegen.GenerateUnsafeProjection
import org.apache.spark.sql.catalyst.plans.logical.{LogicalPlan, Project, SpecialSum}
import org.apache.spark.sql.execution.{SparkPlan, SparkStrategy}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.types._
import org.apache.spark.{SparkEnv, SparkException}
import org.apache.spark.sql.execution.streaming._
import org.apache.spark.sql.expressions._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.streaming.util.BlockingSource
import org.apache.spark.util.ManualClock
import org.apache.spark.sql.execution.debug._
import org.apache.spark.storage.{BlockId, StateStoreBlockId, StorageLevel}
import org.apache.spark.unsafe.types.UTF8String

import scala.collection.mutable


class StreamingQuerySuite extends StreamTest with BeforeAndAfter with Logging {

  import AwaitTerminationTester._
  import testImplicits._

  // To make === between double tolerate inexact values
  implicit val doubleEquality = TolerantNumerics.tolerantDoubleEquality(0.01)

  after {
    sqlContext.streams.active.foreach(_.stop())
  }

  test("name unique in active queries") {
    withTempDir { dir =>
      def startQuery(name: Option[String]): StreamingQuery = {
        val writer = MemoryStream[Int].toDS.writeStream
        name.foreach(writer.queryName)
        writer
          .foreach(new TestForeachWriter)
          .start()
      }

      // No name by default, multiple active queries can have no name
      val q1 = startQuery(name = None)
      assert(q1.name === null)
      val q2 = startQuery(name = None)
      assert(q2.name === null)

      // Can be set by user
      val q3 = startQuery(name = Some("q3"))
      assert(q3.name === "q3")

      // Multiple active queries cannot have same name
      val e = intercept[IllegalArgumentException] {
        startQuery(name = Some("q3"))
      }

      q1.stop()
      q2.stop()
      q3.stop()
    }
  }

  test("blocks") {
    val bm = SparkEnv.get.blockManager
    val keyValues = Seq(("a", 2), ("b", 4), ("d", 7))
    val blockId = StateStoreBlockId(0, 0)
    bm.putSingle(blockId, keyValues, StorageLevel.MEMORY_AND_DISK)
    val fetched = bm.get[Any](blockId)
    fetched.foreach { res => res.data.foreach(println)}
  }

  test("udaf2") {
    //    val df = Seq(0, 1, 2).toDF("x")
    //    val aggdf = df.agg((new ModelAgg)(col("x")))//.show()
    //    println(aggdf.logicalPlan.treeString)
    val inputData = MemoryStream[(String, Int, Int)]
    val query = inputData.toDS
      .withColumn("lit", lit("model"))
      .groupBy("lit")
      .agg(sum("_2"), max("_3"))
      .writeStream
      .outputMode("complete")
      .format("console")
      .start()
    inputData.addData(Seq(("a", 1, 4), ("b", 2, 3), ("a", 3, 1)))
    Thread.sleep(1000)
    inputData.addData(Seq(("b", 1, 9), ("c", 2, 4), ("b", 5, -1)))
    query.awaitTermination(5000)
  }

  test("udaf") {
//    val df = Seq(0, 1, 2).toDF("x")
//    val aggdf = df.agg((new ModelAgg)(col("x")))//.show()
//    println(aggdf.logicalPlan.treeString)
    val inputData = MemoryStream[(String, Int)]
    val query = inputData.toDS
      .groupBy("_1")
      .agg((new MySimpleAgg())(col("_2")))
      .writeStream
      .outputMode("complete")
      .format("console")
      .start()
    inputData.addData(Seq(("a", 1), ("b", 2), ("a", 3)))
    Thread.sleep(1000)
    inputData.addData(Seq(("b", 1), ("c", 2), ("b", 5)))
    query.awaitTermination(5000)
  }
  // TODO: just do a thing without initialization where you
  // save the model and can figure out where it got saved using the
  // unsafe projection method. THEN figure out how to actually recover
  // the state and use it at each iteration.

  test("unsafe rows") {
    val keyRow = new GenericInternalRow(Array(UTF8String.fromString("a").asInstanceOf[Any],
      2.asInstanceOf[Any]))
//    val row = new UnsafeRow(2)
//    val data = new Array[Byte](1024)
//    row.pointTo(data, 32)
//    row.setLong(0, 103079215105L)
//    row.setLong(1, 2)
    val row = InternalRow(UTF8String.fromString("a"), 2)
    val converter = UnsafeProjection.create(Array[DataType](StringType, IntegerType))
    val unsafeRow = converter.apply(row)
    println(unsafeRow)
    val keyExpr = StructType(Array(StructField("int", IntegerType)))
    val output = Seq(
        AttributeReference("string", StringType)(),
        AttributeReference("int", IntegerType)()).map(_.toAttribute)
    val proj = GenerateUnsafeProjection.generate(output.tail, output)
    println(proj(unsafeRow))
    // [0,1800000001,2,62,0]
//    println(row)
//    println(data.take(16).mkString(","))
  }

  test("read another partition") {
//    val batch1 = Seq("a", "b", "c", "a", "a", "c", "c", "b")
//    val batch2 = Seq("a", "b", "c", "a", "a", "c", "c", "b")
    val batch1 = Seq("a", "c", "a", "a", "a", "a")
    val batch2 = Seq("b", "b", "b", "b")
    val batch3 = Seq("a", "a", "a", "a")
    val inputData = MemoryStream[String]
    val query = inputData.toDS
      .groupBy("value")
      .count
      .writeStream
      .outputMode("complete")
      .format("console")
      .start()
    inputData.addData(batch1)
    Thread.sleep(1000)
    inputData.addData(batch2)
    query.awaitTermination(5000)
  }

  test("sgd") {
    val sgdAgg = new SGDAgg(1, None)
    val rng = new scala.util.Random(42)
    val data = (0 until 100).map { j =>
      val label = rng.nextInt(10).toDouble
      val features = Array(label / 2.0 + rng.nextDouble() * 0.1)
      (label, features)
    }
    val (batch1, batch2) = data.splitAt(50)
//    val df = data.toDF("y", "x")
//    val aggdf = df.agg(sgdAgg(col("y"), col("x")))
//    println(aggdf.first().getAs[mutable.WrappedArray[Double]](0).mkString(","))
    val inputData = MemoryStream[(Double, Array[Double])]
    val query = inputData.toDS
      .withColumn("model", lit("model"))
      .groupBy("model")
      .agg(sgdAgg(col("_1"), col("_2")))
      .writeStream
      .outputMode("complete")
      .format("console")
      .start()
    inputData.addData(batch1)
    Thread.sleep(1000)
    inputData.addData(batch2)
//    Thread.sleep(1000)
//    inputData.addData(batch1)
//    Thread.sleep(1000)
//    inputData.addData(batch2)
    query.awaitTermination(5000)
  }

  test("mytest") {
    val mystrategy = new SparkStrategy {
      override def apply(plan: LogicalPlan): Seq[SparkPlan] = plan match {
        case SpecialSum(child) =>
          SpecialSumExec(planLater(child)) :: Nil
        case _ => Nil
      }
    }
    spark.experimental.extraStrategies = mystrategy :: Nil
//    val dataset = spark.sparkContext.parallelize(Seq("a", "b", "c", "a", "a", "c", "c", "b"), 4)
//      .map(Tuple1.apply).toDF("x")
//    val dataset = spark.sparkContext.parallelize(0 to 4)
//      .map(Tuple1.apply).toDF("x")
    val dataset = Seq(Array(1.0, 2.0), Array(2.0, 3.0), Array(2.0, 3.0)).toDF("x")
//    val dataset = spark.createDataFrame(rows)
//    val dataset = Seq((1, 2), (3, 4), (5, 6)).toDF("x", "y")
//    val dataset = Seq(0,1,2).toDF("x")
//    val df = dataset.groupBy("x").count()
//    dataset.show()
//    val grow = new GenericInternalRow(Array(1.0)map(_.asInstanceOf[Any]))
//    val proj = GenerateUnsafeProjection.generate()
    dataset.specialsum(col("x")).show()
//    val df = dataset.agg((new ModelAgg())(col("x")))
//    val df = dataset.agg(sum(col("x")), sum(col("y")))
//    println(df.debugCodegen())
//    df.show()
//    val myExecution = new IncrementalModelExecution(spark, df.logicalPlan)
//    myExecution.sparkPlan

//    val inputData = MemoryStream[Int]
//    val query = inputData.toDS
//      .groupBy("value")
//        .agg((new ModelAgg())(col("value")))
//      .writeStream
//      .outputMode("complete")
//      .format("console")
//      .start()
//    inputData.addData(Seq(1, 2, 3))
//    Thread.sleep(1000)
//    inputData.addData(Seq(1, 2, 4))
//    query.awaitTermination(5000)
  }

  test(
    "id unique in active queries + persists across restarts, runId unique across start/restarts") {
    val inputData = MemoryStream[Int]
    withTempDir { dir =>
      var cpDir: String = null

      def startQuery(restart: Boolean): StreamingQuery = {
        if (cpDir == null || !restart) cpDir = s"$dir/${RandomStringUtils.randomAlphabetic(10)}"
        MemoryStream[Int].toDS().groupBy().count()
          .writeStream
          .format("memory")
          .outputMode("complete")
          .queryName(s"name${RandomStringUtils.randomAlphabetic(10)}")
          .option("checkpointLocation", cpDir)
          .start()
      }

      // id and runId unique for new queries
      val q1 = startQuery(restart = false)
      val q2 = startQuery(restart = false)
      assert(q1.id !== q2.id)
      assert(q1.runId !== q2.runId)
      q1.stop()
      q2.stop()

      // id persists across restarts, runId unique across restarts
      val q3 = startQuery(restart = false)
      q3.stop()

      val q4 = startQuery(restart = true)
      q4.stop()
      assert(q3.id === q3.id)
      assert(q3.runId !== q4.runId)

      // Only one query with same id can be active
      val q5 = startQuery(restart = false)
      val e = intercept[IllegalStateException] {
        startQuery(restart = true)
      }
    }
  }

  testQuietly("isActive, exception, and awaitTermination") {
    val inputData = MemoryStream[Int]
    val mapped = inputData.toDS().map { 6 / _}

    testStream(mapped)(
      AssertOnQuery(_.isActive === true),
      AssertOnQuery(_.exception.isEmpty),
      AddData(inputData, 1, 2),
      CheckAnswer(6, 3),
      TestAwaitTermination(ExpectBlocked),
      TestAwaitTermination(ExpectBlocked, timeoutMs = 2000),
      TestAwaitTermination(ExpectNotBlocked, timeoutMs = 10, expectedReturnValue = false),
      StopStream,
      AssertOnQuery(_.isActive === false),
      AssertOnQuery(_.exception.isEmpty),
      TestAwaitTermination(ExpectNotBlocked),
      TestAwaitTermination(ExpectNotBlocked, timeoutMs = 2000, expectedReturnValue = true),
      TestAwaitTermination(ExpectNotBlocked, timeoutMs = 10, expectedReturnValue = true),
      StartStream(),
      AssertOnQuery(_.isActive === true),
      AddData(inputData, 0),
      ExpectFailure[SparkException](),
      AssertOnQuery(_.isActive === false),
      TestAwaitTermination(ExpectException[SparkException]),
      TestAwaitTermination(ExpectException[SparkException], timeoutMs = 2000),
      TestAwaitTermination(ExpectException[SparkException], timeoutMs = 10),
      AssertOnQuery(q => {
        q.exception.get.startOffset ===
          q.committedOffsets.toOffsetSeq(Seq(inputData), OffsetSeqMetadata()).toString &&
          q.exception.get.endOffset ===
            q.availableOffsets.toOffsetSeq(Seq(inputData), OffsetSeqMetadata()).toString
      }, "incorrect start offset or end offset on exception")
    )
  }

  testQuietly("status, lastProgress, and recentProgress") {
    import StreamingQuerySuite._
    clock = new StreamManualClock

    /** Custom MemoryStream that waits for manual clock to reach a time */
    val inputData = new MemoryStream[Int](0, sqlContext) {
      // Wait for manual clock to be 100 first time there is data
      override def getOffset: Option[Offset] = {
        val offset = super.getOffset
        if (offset.nonEmpty) {
          clock.waitTillTime(300)
        }
        offset
      }

      // Wait for manual clock to be 300 first time there is data
      override def getBatch(start: Option[Offset], end: Offset): DataFrame = {
        clock.waitTillTime(600)
        super.getBatch(start, end)
      }
    }

    // This is to make sure thatquery waits for manual clock to be 600 first time there is data
    val mapped = inputData.toDS().as[Long].map { x =>
      clock.waitTillTime(1100)
      10 / x
    }.agg(count("*")).as[Long]

    case class AssertStreamExecThreadToWaitForClock()
      extends AssertOnQuery(q => {
        eventually(Timeout(streamingTimeout)) {
          if (q.exception.isEmpty) {
            assert(clock.asInstanceOf[StreamManualClock].isStreamWaitingAt(clock.getTimeMillis))
          }
        }
        if (q.exception.isDefined) {
          throw q.exception.get
        }
        true
      }, "")

    var lastProgressBeforeStop: StreamingQueryProgress = null

    testStream(mapped, OutputMode.Complete)(
      StartStream(ProcessingTime(100), triggerClock = clock),
      AssertStreamExecThreadToWaitForClock(),
      AssertOnQuery(_.status.isDataAvailable === false),
      AssertOnQuery(_.status.isTriggerActive === false),
      AssertOnQuery(_.status.message === "Waiting for next trigger"),
      AssertOnQuery(_.recentProgress.count(_.numInputRows > 0) === 0),

      // Test status and progress while offset is being fetched
      AddData(inputData, 1, 2),
      AdvanceManualClock(100), // time = 100 to start new trigger, will block on getOffset
      AssertStreamExecThreadToWaitForClock(),
      AssertOnQuery(_.status.isDataAvailable === false),
      AssertOnQuery(_.status.isTriggerActive === true),
      AssertOnQuery(_.status.message.startsWith("Getting offsets from")),
      AssertOnQuery(_.recentProgress.count(_.numInputRows > 0) === 0),

      // Test status and progress while batch is being fetched
      AdvanceManualClock(200), // time = 300 to unblock getOffset, will block on getBatch
      AssertStreamExecThreadToWaitForClock(),
      AssertOnQuery(_.status.isDataAvailable === true),
      AssertOnQuery(_.status.isTriggerActive === true),
      AssertOnQuery(_.status.message === "Processing new data"),
      AssertOnQuery(_.recentProgress.count(_.numInputRows > 0) === 0),

      // Test status and progress while batch is being processed
      AdvanceManualClock(300), // time = 600 to unblock getBatch, will block in Spark job
      AssertOnQuery(_.status.isDataAvailable === true),
      AssertOnQuery(_.status.isTriggerActive === true),
      AssertOnQuery(_.status.message === "Processing new data"),
      AssertOnQuery(_.recentProgress.count(_.numInputRows > 0) === 0),

      // Test status and progress while batch processing has completed
      AdvanceManualClock(500), // time = 1100 to unblock job
      AssertOnQuery { _ => clock.getTimeMillis() === 1100 },
      CheckAnswer(2),
      AssertOnQuery(_.status.isDataAvailable === true),
      AssertOnQuery(_.status.isTriggerActive === false),
      AssertOnQuery(_.status.message === "Waiting for next trigger"),
      AssertOnQuery { query =>
        assert(query.lastProgress != null)
        assert(query.recentProgress.exists(_.numInputRows > 0))
        assert(query.recentProgress.last.eq(query.lastProgress))

        val progress = query.lastProgress
        assert(progress.id === query.id)
        assert(progress.name === query.name)
        assert(progress.batchId === 0)
        assert(progress.timestamp === "1970-01-01T00:00:00.100Z") // 100 ms in UTC
        assert(progress.numInputRows === 2)
        assert(progress.processedRowsPerSecond === 2.0)

        assert(progress.durationMs.get("getOffset") === 200)
        assert(progress.durationMs.get("getBatch") === 300)
        assert(progress.durationMs.get("queryPlanning") === 0)
        assert(progress.durationMs.get("walCommit") === 0)
        assert(progress.durationMs.get("triggerExecution") === 1000)

        assert(progress.sources.length === 1)
        assert(progress.sources(0).description contains "MemoryStream")
        assert(progress.sources(0).startOffset === null)
        assert(progress.sources(0).endOffset !== null)
        assert(progress.sources(0).processedRowsPerSecond === 2.0)

        assert(progress.stateOperators.length === 1)
        assert(progress.stateOperators(0).numRowsUpdated === 1)
        assert(progress.stateOperators(0).numRowsTotal === 1)

        assert(progress.sink.description contains "MemorySink")
        true
      },

      AddData(inputData, 1, 2),
      AdvanceManualClock(100), // allow another trigger
      CheckAnswer(4),
      AssertOnQuery(_.status.isDataAvailable === true),
      AssertOnQuery(_.status.isTriggerActive === false),
      AssertOnQuery(_.status.message === "Waiting for next trigger"),
      AssertOnQuery { query =>
        assert(query.recentProgress.last.eq(query.lastProgress))
        assert(query.lastProgress.batchId === 1)
        assert(query.lastProgress.sources(0).inputRowsPerSecond === 1.818)
        true
      },

      // Test status and progress after data is not available for a trigger
      AdvanceManualClock(100), // allow another trigger
      AssertStreamExecThreadToWaitForClock(),
      AssertOnQuery(_.status.isDataAvailable === false),
      AssertOnQuery(_.status.isTriggerActive === false),
      AssertOnQuery(_.status.message === "Waiting for next trigger"),

      // Test status and progress after query stopped
      AssertOnQuery { query =>
        lastProgressBeforeStop = query.lastProgress
        true
      },
      StopStream,
      AssertOnQuery(_.lastProgress.json === lastProgressBeforeStop.json),
      AssertOnQuery(_.status.isDataAvailable === false),
      AssertOnQuery(_.status.isTriggerActive === false),
      AssertOnQuery(_.status.message === "Stopped"),

      // Test status and progress after query terminated with error
      StartStream(ProcessingTime(100), triggerClock = clock),
      AddData(inputData, 0),
      AdvanceManualClock(100),
      ExpectFailure[SparkException](),
      AssertOnQuery(_.status.isDataAvailable === false),
      AssertOnQuery(_.status.isTriggerActive === false),
      AssertOnQuery(_.status.message.startsWith("Terminated with exception"))
    )
  }

  test("lastProgress should be null when recentProgress is empty") {
    BlockingSource.latch = new CountDownLatch(1)
    withTempDir { tempDir =>
      val sq = spark.readStream
        .format("org.apache.spark.sql.streaming.util.BlockingSource")
        .load()
        .writeStream
        .format("org.apache.spark.sql.streaming.util.BlockingSource")
        .option("checkpointLocation", tempDir.toString)
        .start()
      // Creating source is blocked so recentProgress is empty and lastProgress should be null
      assert(sq.lastProgress === null)
      // Release the latch and stop the query
      BlockingSource.latch.countDown()
      sq.stop()
    }
  }

  test("codahale metrics") {
    val inputData = MemoryStream[Int]

    /** Whether metrics of a query is registered for reporting */
    def isMetricsRegistered(query: StreamingQuery): Boolean = {
      val sourceName = s"spark.streaming.${query.id}"
      val sources = spark.sparkContext.env.metricsSystem.getSourcesByName(sourceName)
      require(sources.size <= 1)
      sources.nonEmpty
    }
    // Disabled by default
    assert(spark.conf.get("spark.sql.streaming.metricsEnabled").toBoolean === false)

    withSQLConf("spark.sql.streaming.metricsEnabled" -> "false") {
      testStream(inputData.toDF)(
        AssertOnQuery { q => !isMetricsRegistered(q) },
        StopStream,
        AssertOnQuery { q => !isMetricsRegistered(q) }
      )
    }

    // Registered when enabled
    withSQLConf("spark.sql.streaming.metricsEnabled" -> "true") {
      testStream(inputData.toDF)(
        AssertOnQuery { q => isMetricsRegistered(q) },
        StopStream,
        AssertOnQuery { q => !isMetricsRegistered(q) }
      )
    }
  }

  test("input row calculation with mixed batch and streaming sources") {
    val streamingTriggerDF = spark.createDataset(1 to 10).toDF
    val streamingInputDF = createSingleTriggerStreamingDF(streamingTriggerDF).toDF("value")
    val staticInputDF = spark.createDataFrame(Seq(1 -> "1", 2 -> "2")).toDF("value", "anotherValue")

    // Trigger input has 10 rows, static input has 2 rows,
    // therefore after the first trigger, the calculated input rows should be 10
    val progress = getFirstProgress(streamingInputDF.join(staticInputDF, "value"))
    assert(progress.numInputRows === 10)
    assert(progress.sources.size === 1)
    assert(progress.sources(0).numInputRows === 10)
  }

  test("input row calculation with trigger input DF having multiple leaves") {
    val streamingTriggerDF =
      spark.createDataset(1 to 5).toDF.union(spark.createDataset(6 to 10).toDF)
    require(streamingTriggerDF.logicalPlan.collectLeaves().size > 1)
    val streamingInputDF = createSingleTriggerStreamingDF(streamingTriggerDF)

    // After the first trigger, the calculated input rows should be 10
    val progress = getFirstProgress(streamingInputDF)
    assert(progress.numInputRows === 10)
    assert(progress.sources.size === 1)
    assert(progress.sources(0).numInputRows === 10)
  }

  testQuietly("StreamExecution metadata garbage collection") {
    val inputData = MemoryStream[Int]
    val mapped = inputData.toDS().map(6 / _)
    withSQLConf(SQLConf.MIN_BATCHES_TO_RETAIN.key -> "1") {
      // Run 3 batches, and then assert that only 2 metadata files is are at the end
      // since the first should have been purged.
      testStream(mapped)(
        AddData(inputData, 1, 2),
        CheckAnswer(6, 3),
        AddData(inputData, 1, 2),
        CheckAnswer(6, 3, 6, 3),
        AddData(inputData, 4, 6),
        CheckAnswer(6, 3, 6, 3, 1, 1),

        AssertOnQuery("metadata log should contain only two files") { q =>
          val metadataLogDir = new java.io.File(q.offsetLog.metadataPath.toString)
          val logFileNames = metadataLogDir.listFiles().toSeq.map(_.getName())
          val toTest = logFileNames.filter(!_.endsWith(".crc")).sorted // Workaround for SPARK-17475
          assert(toTest.size == 2 && toTest.head == "1")
          true
        }
      )
    }

    val inputData2 = MemoryStream[Int]
    withSQLConf(SQLConf.MIN_BATCHES_TO_RETAIN.key -> "2") {
      // Run 5 batches, and then assert that 3 metadata files is are at the end
      // since the two should have been purged.
      testStream(inputData2.toDS())(
        AddData(inputData2, 1, 2),
        CheckAnswer(1, 2),
        AddData(inputData2, 1, 2),
        CheckAnswer(1, 2, 1, 2),
        AddData(inputData2, 3, 4),
        CheckAnswer(1, 2, 1, 2, 3, 4),
        AddData(inputData2, 5, 6),
        CheckAnswer(1, 2, 1, 2, 3, 4, 5, 6),
        AddData(inputData2, 7, 8),
        CheckAnswer(1, 2, 1, 2, 3, 4, 5, 6, 7, 8),

        AssertOnQuery("metadata log should contain three files") { q =>
          val metadataLogDir = new java.io.File(q.offsetLog.metadataPath.toString)
          val logFileNames = metadataLogDir.listFiles().toSeq.map(_.getName())
          val toTest = logFileNames.filter(!_.endsWith(".crc")).sorted // Workaround for SPARK-17475
          assert(toTest.size == 3 && toTest.head == "2")
          true
        }
      )
    }
  }

  test("StreamingQuery should be Serializable but cannot be used in executors") {
    def startQuery(ds: Dataset[Int], queryName: String): StreamingQuery = {
      ds.writeStream
        .queryName(queryName)
        .format("memory")
        .start()
    }

    val input = MemoryStream[Int]
    val q1 = startQuery(input.toDS, "stream_serializable_test_1")
    val q2 = startQuery(input.toDS.map { i =>
      // Emulate that `StreamingQuery` get captured with normal usage unintentionally.
      // It should not fail the query.
      q1
      i
    }, "stream_serializable_test_2")
    val q3 = startQuery(input.toDS.map { i =>
      // Emulate that `StreamingQuery` is used in executors. We should fail the query with a clear
      // error message.
      q1.explain()
      i
    }, "stream_serializable_test_3")
    try {
      input.addData(1)

      // q2 should not fail since it doesn't use `q1` in the closure
      q2.processAllAvailable()

      // The user calls `StreamingQuery` in the closure and it should fail
      val e = intercept[StreamingQueryException] {
        q3.processAllAvailable()
      }
      assert(e.getCause.isInstanceOf[SparkException])
      assert(e.getCause.getCause.isInstanceOf[IllegalStateException])
      assert(e.getMessage.contains("StreamingQuery cannot be used in executors"))
    } finally {
      q1.stop()
      q2.stop()
      q3.stop()
    }
  }

  /** Create a streaming DF that only execute one batch in which it returns the given static DF */
  private def createSingleTriggerStreamingDF(triggerDF: DataFrame): DataFrame = {
    require(!triggerDF.isStreaming)
    // A streaming Source that generate only on trigger and returns the given Dataframe as batch
    val source = new Source() {
      override def schema: StructType = triggerDF.schema
      override def getOffset: Option[Offset] = Some(LongOffset(0))
      override def getBatch(start: Option[Offset], end: Offset): DataFrame = triggerDF
      override def stop(): Unit = {}
    }
    StreamingExecutionRelation(source)
  }

  /** Returns the query progress at the end of the first trigger of streaming DF */
  private def getFirstProgress(streamingDF: DataFrame): StreamingQueryProgress = {
    try {
      val q = streamingDF.writeStream.format("memory").queryName("test").start()
      q.processAllAvailable()
      q.recentProgress.head
    } finally {
      spark.streams.active.map(_.stop())
    }
  }

  /**
   * A [[StreamAction]] to test the behavior of `StreamingQuery.awaitTermination()`.
   *
   * @param expectedBehavior  Expected behavior (not blocked, blocked, or exception thrown)
   * @param timeoutMs         Timeout in milliseconds
   *                          When timeoutMs <= 0, awaitTermination() is tested (i.e. w/o timeout)
   *                          When timeoutMs > 0, awaitTermination(timeoutMs) is tested
   * @param expectedReturnValue Expected return value when awaitTermination(timeoutMs) is used
   */
  case class TestAwaitTermination(
      expectedBehavior: ExpectedBehavior,
      timeoutMs: Int = -1,
      expectedReturnValue: Boolean = false
    ) extends AssertOnQuery(
      TestAwaitTermination.assertOnQueryCondition(expectedBehavior, timeoutMs, expectedReturnValue),
      "Error testing awaitTermination behavior"
    ) {
    override def toString(): String = {
      s"TestAwaitTermination($expectedBehavior, timeoutMs = $timeoutMs, " +
        s"expectedReturnValue = $expectedReturnValue)"
    }
  }

  object TestAwaitTermination {

    /**
     * Tests the behavior of `StreamingQuery.awaitTermination`.
     *
     * @param expectedBehavior  Expected behavior (not blocked, blocked, or exception thrown)
     * @param timeoutMs         Timeout in milliseconds
     *                          When timeoutMs <= 0, awaitTermination() is tested (i.e. w/o timeout)
     *                          When timeoutMs > 0, awaitTermination(timeoutMs) is tested
     * @param expectedReturnValue Expected return value when awaitTermination(timeoutMs) is used
     */
    def assertOnQueryCondition(
        expectedBehavior: ExpectedBehavior,
        timeoutMs: Int,
        expectedReturnValue: Boolean
      )(q: StreamExecution): Boolean = {

      def awaitTermFunc(): Unit = {
        if (timeoutMs <= 0) {
          q.awaitTermination()
        } else {
          val returnedValue = q.awaitTermination(timeoutMs)
          assert(returnedValue === expectedReturnValue, "Returned value does not match expected")
        }
      }
      AwaitTerminationTester.test(expectedBehavior, awaitTermFunc)
      true // If the control reached here, then everything worked as expected
    }
  }
}

object StreamingQuerySuite {
  // Singleton reference to clock that does not get serialized in task closures
  var clock: ManualClock = null
}


