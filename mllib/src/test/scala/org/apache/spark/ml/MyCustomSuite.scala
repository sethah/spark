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
package org.apache.spark.ml


import org.apache.spark.ml.linalg.{Vectors, BLAS, Vector}
import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.sql.execution.streaming.Sink
import org.apache.spark.sql.sources.StreamSinkProvider
import org.apache.spark.sql.streaming.OutputMode
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.expressions.{GenericMutableRow, SpecificMutableRow}
import org.apache.spark.sql.types._
import org.apache.spark.sql.expressions.{MutableAggregationBuffer,
UserDefinedAggregateFunction}
import scala.collection.mutable.WrappedArray
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions._

class MyCustomSuite extends SparkFunSuite with MLlibTestSparkContext {

//  test("foreach") {
//    val checkpointDir = "/Users/sethhendrickson/StreamingSandbox/checkpoint"
//    val dataDir = "/Users/sethhendrickson/StreamingSandbox/data2"
//    val static = spark.read.format("csv").option("inferSchema", "true").csv(dataDir)
//    val schema = static.schema
//    val df = spark
//      .readStream
//      .format("csv")
//      .schema(schema)
//      .option("inferSchema", "true")
//      .csv(dataDir)
//    df.createOrReplaceTempView("df")
//    val inputCols = Array.tabulate(10) { i => s"_c$i"}
//    val vecAssembler = new VectorAssembler().setInputCols(inputCols).setOutputCol("vec")
//    val assembled = vecAssembler.transform(df).select("_c0", "vec")
//    val arrUDF = udf((vec: Vector) => vec.toArray)
//    val weights = Array.fill(10)(math.random)
//    val q0 = assembled.select(col("_c0"), arrUDF(col("vec")).as("arr"))
//      .agg(new VectorSum(weights)(col("_c0"), col("arr")))
//    val q1 = df.agg(sum("_c0"), sum("_c1"))
////    val q3 = spark.sql("SELECT * FROM ")
////    val q2 = spark.sql("SELECT _c0, SUM(_c1) FROM df GROUP BY _c0")
//    val query = q0.writeStream.outputMode("complete").foreach(new MyForeachWriter()).start()
//    query.awaitTermination()
//  }
  class MLSink extends Sink {
    val lr = new LinearRegression()
    def addBatch(batchId: Long, df: DataFrame): Unit = {
      val model = lr.fit(df)
      println(model.coefficients)
    }
  }

  class MySinkProvider extends StreamSinkProvider {
    def createSink(
        sqlContext: SQLContext,
        parameters: Map[String, String],
        partitionColumns: Seq[String],
        outputMode: OutputMode): Sink = {
      new MLSink
    }
  }

  test("sink pipeline") {
    val checkpointDir = "/Users/sethhendrickson/StreamingSandbox/checkpoint"
    val dataDir = "/Users/sethhendrickson/StreamingSandbox/data2"
    val dataTmpDir = "/Users/sethhendrickson/StreamingSandbox/data1"
    val static = spark.read.format("csv").option("inferSchema", "true").csv(dataTmpDir)
    val schema = static.schema
    val df = spark
      .readStream
      .format("csv")
      .schema(schema)
      .option("inferSchema", "true")
      .csv(dataDir)
    val inputCols = Array.tabulate(10) { i => s"_c${i + 1}"}
    val vecAssembler = new VectorAssembler().setInputCols(inputCols).setOutputCol("features")
    val assembled = vecAssembler.transform(df).select("_c0", "features").toDF("label", "features")
    val query = assembled.writeStream.outputMode("append")
      .option("checkpointLocation", checkpointDir).format(new MySinkProvider()).start()
    query.awaitTermination()
  }

  test("query order") {
    val checkpointDir = "/Users/sethhendrickson/StreamingSandbox/checkpoint"
    val dataDir = "/Users/sethhendrickson/StreamingSandbox/data2"
    val static = spark.read.format("csv").option("inferSchema", "true").csv(dataDir)
    val schema = static.schema
    val df = spark
      .readStream
      .format("com.sethah.mysource")
      .schema(schema)
      .option("inferSchema", "true")
      .csv(dataDir)
    df.createOrReplaceTempView("df")
    val inputCols = Array.tabulate(10) { i => s"_c$i"}
    val vecAssembler = new VectorAssembler().setInputCols(inputCols).setOutputCol("vec")
    val assembled = vecAssembler.transform(df).select("_c0", "vec")
    val arrUDF = udf((vec: Vector) => vec.toArray)
    val weights = Array.fill(10)(math.random)
    val q0 = assembled.select(col("_c0"), arrUDF(col("vec")).as("arr"))
      .agg(new VectorSum(weights)(col("_c0"), col("arr")))
    val q1 = df.agg(sum("_c0"), sum("_c1"))
    val q2 = df.agg(sum("_c2"))
//    val q3 = spark.sql("SELECT * FROM ")
//    val q2 = spark.sql("SELECT _c0, SUM(_c1) FROM df GROUP BY _c0")
    val query = q1.writeStream.outputMode("complete").foreach(new MyForeachWriter()).start()
    val query2 = q2.writeStream.outputMode("complete").foreach(new MyForeachWriter()).start()
    query.awaitTermination()
  }

//  test("pipeline") {
//    val checkpointDir = "/Users/sethhendrickson/StreamingSandbox/checkpoint"
//    val dataDir = "/Users/sethhendrickson/StreamingSandbox/data2"
//    val static = spark.read.format("csv").option("inferSchema", "true").csv(dataDir)
//    val schema = static.schema
//    val df = spark
//      .readStream
//      .format("csv")
//      .schema(schema)
//      .option("inferSchema", "true")
//      .csv(dataDir)
//    df.createOrReplaceTempView("df")
//    val q1 = spark.sql("SELECT MIN(_c0) FROM df")
//    val query = q1.writeStream.outputMode("complete").foreach(new MyForeachWriterMin()).start()
//    val myUDF = udf((x: Int) => x / MyCustomTransformer.min.toDouble)
//    val q2 = df.select(col("_c0"), myUDF(col("_c0")).as("scaled"))
//    val query2 = q2.writeStream.outputMode("append").foreach(new MyForeachWriter()).start()
//    query.awaitTermination()
//    query2.awaitTermination()
//  }


}


class VectorSum (weights: Array[Double]) extends UserDefinedAggregateFunction {
  def inputSchema: StructType = StructType(
    StructField("label", DoubleType) ::
    StructField("features", ArrayType(DoubleType)) :: Nil
  )
  def bufferSchema: StructType = StructType(
    StructField("count", LongType) ::
    StructField("gradient", ArrayType(DoubleType)) :: Nil
  )
  // return type
  def dataType: DataType = ArrayType(DoubleType)
  def deterministic: Boolean = true
  val n = weights.length

  def initialize(buffer: MutableAggregationBuffer): Unit = {
    buffer(0) = 0L
    buffer(1) = Array.fill(n)(0.0)
  }

  def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
    buffer(0) = buffer.getAs[Long](0) + 1L
    val cumGradient = buffer.getAs[WrappedArray[Double]](1).toArray
    val label = input.getAs[Double](0)
    val features = input.getAs[WrappedArray[Double]](1)
    val diff = BLAS.dot(Vectors.dense(features.toArray), Vectors.dense(weights)) - label
    BLAS.axpy(diff, Vectors.dense(features.toArray), Vectors.dense(cumGradient))
    buffer.update(1, cumGradient)
//    println(s"updating! ${cumGradient.mkString}")
  }

  def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
    val buff1 = buffer1.getAs[WrappedArray[Double]](1)
    val buff2 = buffer2.getAs[WrappedArray[Double]](1)
//    println(buff1.mkString, "**", buff2.mkString)
    for ((x, i) <- buff2.zipWithIndex) {
      buff1(i) += x
    }
    buffer1.update(1, buff1)
    buffer1.update(0, buffer1.getAs[Long](0) + buffer2.getAs[Long](0))
  }

  def evaluate(buffer: Row): Any = {
    val count = buffer.getAs[Long](0)
    println(s"Count: $count")
    val cumGradient = Vectors.dense(buffer.getAs[Seq[Double]](1).toArray)
    val stepSize = 0.1
    BLAS.axpy(-stepSize / count.toDouble, cumGradient, Vectors.dense(weights))
    weights
  }
}


object MyCustomTransformer {
  var min = -1
}

private[ml] class MyForeachWriter extends ForeachWriter[Row] {
  def open(partitionId: Long, version: Long): Boolean = {
    true
  }
  def process(value: Row, partitionId: Long, version: Long): Unit = if (value != null) {
    println(s"$value, ($partitionId, $version, ${Thread.currentThread().getId()})")
  }
  def close(errorOrNull: Throwable): Unit = if (errorOrNull != null) println(errorOrNull)
}

private[ml] class MyForeachWriterMin extends ForeachWriter[Row] {
  def open(partitionId: Long, version: Long): Boolean = {
    true
  }
  def process(value: Row, partitionId: Long, version: Long): Unit = if (value != null) {
    println(s"$value, ($partitionId, $version, ${Thread.currentThread().getId()})")
    value match {
      case Row(x: Int) => {
        MyCustomTransformer.min = x
        println(s"TransformerMin: ${MyCustomTransformer.min}")
      }
    }
  }
  def close(errorOrNull: Throwable): Unit = if (errorOrNull != null) println(errorOrNull)
}
