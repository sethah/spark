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

import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.classification.{NaiveBayes, NaiveBayesSuite, LogisticRegressionSuite}

import org.apache.spark.ml.classification.{NaiveBayesSuite, LogisticRegressionSuite}
import org.apache.spark.ml.linalg.{Vectors, BLAS, Vector}
import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.streaming.{StreamingStringIndexer, StreamingNaiveBayes, StreamingPipeline}
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.sql.expressions.{MutableAggregationBuffer,
UserDefinedAggregateFunction}
import scala.collection.mutable.WrappedArray
import org.apache.spark.ml.feature._
import org.apache.spark.ml.feature.{LabeledPoint, VectorAssembler}
import org.apache.spark.sql.functions._

class MyCustomSuite extends SparkFunSuite with MLlibTestSparkContext {

  @transient var dataset: DataFrame = _
  private val eps: Double = 1e-5

  override def beforeAll(): Unit = {
    super.beforeAll()

    val pi = Array(0.5, 0.1, 0.4).map(math.log)
    val theta = Array(
      Array(0.70, 0.10, 0.10, 0.10), // label 0
      Array(0.10, 0.70, 0.10, 0.10), // label 1
      Array(0.10, 0.10, 0.70, 0.10) // label 2
    ).map(_.map(math.log))
    val stringMap = Map(0.0 -> "class0", 1.0 -> "class1", 2.0 -> "class2")
    val df1 = spark.createDataFrame(NaiveBayesSuite.generateNaiveBayesInput(pi, theta, 100, 42))
    val stringUDF = udf((cls: Double) => stringMap.getOrElse(cls, "unknown"))
    val df = df1.withColumn("stringLabel", stringUDF(df1("label")))

    dataset = df.cache()
  }

  /**
   * Enable the ignored test to export the dataset into CSV format,
   * so we can validate the training accuracy compared with R's glmnet package.
   */
  ignore("export test data into CSV format") {
    val rdd = dataset.rdd.map {
      case Row(label: Double, features: Vector, stringLabel: String) =>
        label + "," + features.toArray.mkString(",") + "," + stringLabel
    }.repartition(10)
    rdd.saveAsTextFile("/Users/sethhendrickson/StreamingSandbox/nb_dataset")
//    rdd.saveAsTextFile("target/tmp/MultinomialLogisticRegressionSuite/multinomialDataset")
  }

//  test("temp table") {
//    val df = spark.readStream
//      .format("socket")
//      .options(Map("host" -> "localhost", "port" -> "9999"))
//      .load()
//    df.createOrReplaceTempView("data")
//    val agg = df.groupBy("value").count.select(col("count"), col("value").as("avalue"))
//    val schema = StructType(Seq(StructField("count", IntegerType),
//      StructField("avalue", StringType)))
//    val q = agg.writeStream
//      .format(new TempTableSinkProvider("model", spark, schema))
//      .outputMode("complete")
//      .option("checkpointLocation", "/Users/sethhendrickson/tmp/checkpoint")
//      .start()
//    val predicted = spark.sql("SELECT * FROM data JOIN model ON data.value=model.avalue")
//    val q2 = predicted
//      .writeStream
//      .format("console")
//      .outputMode("append")
//      .start()
//    q.awaitTermination()
//  }

  ignore("streaming pipeline") {
    val dataDir = "/Users/sethhendrickson/StreamingSandbox/data2"
    val dataTmpDir = "/Users/sethhendrickson/StreamingSandbox/data1"
    val schema = StructType(Seq(
      StructField("label", DoubleType),
      StructField("feature1", DoubleType),
      StructField("feature2", DoubleType),
      StructField("feature3", DoubleType),
      StructField("feature4", DoubleType),
      StructField("stringLabel", StringType)
    ))
    val df = spark
      .readStream
      .format("csv")
      .schema(schema)
      .csv(dataDir)
    val inputCols = Array.tabulate(4) { i => s"feature${i + 1}"}
    val vecAssembler = new VectorAssembler()
      .setInputCols(inputCols)
      .setOutputCol("features")
    val assembled = vecAssembler
      .transform(df)
      .select("stringLabel", "features")
    val indexer = new StreamingStringIndexer()
      .setInputCol("stringLabel")
      .setOutputCol("indexedLabel")
    val snb = new StreamingNaiveBayes()
      .setFeaturesCol("features")
      .setLabelCol("indexedLabel")
    val pipeline = new StreamingPipeline()
      .setStages(Array(indexer, snb))
      .setCheckpointLocation("/Users/sethhendrickson/StreamingSandbox/checkpoint")
    val query = pipeline.fitStreaming(assembled)
    val pipelineModel = pipeline.getModel
      .setCheckpointLocation("/Users/sethhendrickson/StreamingSandbox/checkpointPredict")
    query.awaitTermination()
  }

  test("20news") {
//    import org.apache.spark.ml.feature._
//    import org.apache.spark.ml.classification.NaiveBayes
    val dataDir = "/Users/sethhendrickson/StreamingSandbox/data2"
    val dataTmpDir = "/Users/sethhendrickson/StreamingSandbox/data1"
    val checkpoint = "/Users/sethhendrickson/StreamingSandbox/checkpoint"
    val path = "/Users/sethhendrickson/StreamingSandbox/20newssample"
//    val ds = spark.read.parquet(path)
//    ds.cache()
    val getClassFunc = (path: String) => {
      val splits = path.split("/")
      splits(splits.length - 2)
    }
    val schema = StructType(Seq(
      StructField("path", StringType),
      StructField("text", StringType)
    ))
    val df = spark
      .readStream
      .schema(schema)
      .parquet(dataDir)
//    val query = df.writeStream.format("console").start()
//    query.awaitTermination()
    val getClassUDF = udf(getClassFunc)
    val withLabel = df.withColumn("stringLabel", getClassUDF(col("path")))
//    val allLabels = withLabel.select("stringLabel").as[String].rdd.distinct().collect()
    val allLabels = Array("alt.atheism", "rec.autos", "sci.med", "rec.sport.hockey")
    val tokenizer = new RegexTokenizer().setInputCol("text").setOutputCol("tokenized")
    val tokenized = tokenizer.transform(withLabel)
    val indexerModel = new StringIndexerModel(allLabels)
      .setInputCol("stringLabel")
      .setOutputCol("label")
//    val indexed = indexerModel.transform(tokenized)
    val hashingTF = new HashingTF().setInputCol("tokenized").setOutputCol("features")
//    val hashed = hashingTF.transform(indexed)
    val snb = new StreamingNaiveBayes()
      .setFeaturesCol("features")
      .setLabelCol("label")
    val pipeline = new StreamingPipeline()
      .setStages(Array(tokenizer, indexerModel, hashingTF, snb))
      .setCheckpointLocation(checkpoint)
    val query = pipeline.fitStreaming(withLabel)
//    val pipelineModel = pipeline.getModel
//      .setCheckpointLocation("/Users/sethhendrickson/StreamingSandbox/checkpointPredict")
    query.awaitTermination()
//    val nb = new NaiveBayes()
//    val nbModel = nb.fit(hashed)
//    val predictions = nbModel.transform(hashed)
  }

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
//  class MLSink extends Sink {
//    val lr = new LinearRegression()
//    def addBatch(batchId: Long, df: DataFrame): Unit = {
//      val model = lr.fit(df)
//      println(model.coefficients)
//    }
//  }
//
//  class MySinkProvider extends StreamSinkProvider {
//    def createSink(
//        sqlContext: SQLContext,
//        parameters: Map[String, String],
//        partitionColumns: Seq[String],
//        outputMode: OutputMode): Sink = {
//      new MLSink
//    }
//  }
//
//  test("sink pipeline") {
//    val checkpointDir = "/Users/sethhendrickson/StreamingSandbox/checkpoint"
//    val dataDir = "/Users/sethhendrickson/StreamingSandbox/data2"
//    val dataTmpDir = "/Users/sethhendrickson/StreamingSandbox/data1"
//    val static = spark.read.format("csv").option("inferSchema", "true").csv(dataTmpDir)
//    val schema = static.schema
//    val df = spark
//      .readStream
//      .format("csv")
//      .schema(schema)
//      .option("inferSchema", "true")
//      .csv(dataDir)
//    val inputCols = Array.tabulate(10) { i => s"_c${i + 1}"}
//    val vecAssembler = new VectorAssembler().setInputCols(inputCols).setOutputCol("features")
//    val assembled = vecAssembler.transform(df).select("_c0", "features").toDF("label", "features")
//    val query = assembled.writeStream.outputMode("append")
//      .option("checkpointLocation", checkpointDir).format(new MySinkProvider()).start()
//    query.awaitTermination()
//  }
//
//  test("query order") {
//    val checkpointDir = "/Users/sethhendrickson/StreamingSandbox/checkpoint"
//    val dataDir = "/Users/sethhendrickson/StreamingSandbox/data2"
//    val static = spark.read.format("csv").option("inferSchema", "true").csv(dataDir)
//    val schema = static.schema
//    val df = spark
//      .readStream
//      .format("com.sethah.mysource")
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
//    val q2 = df.agg(sum("_c2"))
////    val q3 = spark.sql("SELECT * FROM ")
////    val q2 = spark.sql("SELECT _c0, SUM(_c1) FROM df GROUP BY _c0")
//    val query = q1.writeStream.outputMode("complete").foreach(new MyForeachWriter()).start()
//    val query2 = q2.writeStream.outputMode("complete").foreach(new MyForeachWriter()).start()
//    query.awaitTermination()
//  }

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
