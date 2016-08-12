///*
// * Licensed to the Apache Software Foundation (ASF) under one or more
// * contributor license agreements.  See the NOTICE file distributed with
// * this work for additional information regarding copyright ownership.
// * The ASF licenses this file to You under the Apache License, Version 2.0
// * (the "License"); you may not use this file except in compliance with
// * the License.  You may obtain a copy of the License at
// *
// *    http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS,
// * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// * See the License for the specific language governing permissions and
// * limitations under the License.
// */
//package org.apache.spark.sql
//
//import org.apache.spark.ml.linalg.Vector
//import org.apache.spark.SparkFunSuite
//import org.apache.spark.sql.catalyst.expressions.{GenericMutableRow, SpecificMutableRow}
//import org.apache.spark.sql.test.SharedSQLContext
//import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}
//import org.apache.spark.sql.expressions.{MutableAggregationBuffer,
//UserDefinedAggregateFunction}
//import org.apache.spark.sql.types.{StructType, ArrayType, DoubleType}
//import scala.collection.mutable.WrappedArray
//import org.apache.spark.ml.feature.VectorAssembler
//
//class MyCustomSuite  extends SparkFunSuite with SharedSQLContext {
//
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
//    val inputCols = Array.tabulate(10) { i => s"_c$i"}
//    val vecAssembler = new VectorAssembler().setInputCols(inputCols).setOutputCol("vec")
//    val assembled = vecAssembler.transform(df).select("_c0", "vec")
//    val arrUDF = udf((vec: Vector) => vec.toArray)
//    val arrayAssembled = assembled.select(arrUDF(col("vec")).as("arr"))
//      .agg(new VectorSum(10)(col("arr")))
//    arrayAssembled.printSchema()
//    val q = arrayAssembled.writeStream.outputMode("complete").foreach(new MyForeachWriter()).start()
//    q.awaitTermination()
//  }
//
//
//}
//
//
//class VectorSum (n: Int) extends UserDefinedAggregateFunction {
//  def inputSchema = new StructType().add("v", ArrayType(DoubleType))
//  def bufferSchema = new StructType().add("buff", ArrayType(DoubleType))
//  def dataType = ArrayType(DoubleType)
//  def deterministic = true
//
//  def initialize(buffer: MutableAggregationBuffer) = {
//    buffer.update(0, Array.fill(n)(0.0))
//  }
//
//  def update(buffer: MutableAggregationBuffer, input: Row) = {
//    if (!input.isNullAt(0)) {
//      val buff = buffer.getAs[WrappedArray[Double]](0)
//      val v = input.getAs[WrappedArray[Double]](0)
//      for (i <- v.indices) {
//        buff(i) += v(i)
//      }
//      buffer.update(0, buff)
//    }
//  }
//
//  def merge(buffer1: MutableAggregationBuffer, buffer2: Row) = {
//    val buff1 = buffer1.getAs[WrappedArray[Double]](0)
//    val buff2 = buffer2.getAs[WrappedArray[Double]](0)
//    for ((x, i) <- buff2.zipWithIndex) {
//      buff1(i) += x
//    }
//    buffer1.update(0, buff1)
//  }
//
//  def evaluate(buffer: Row) = buffer.getAs[Seq[Double]](0).toArray
//}
//
//
//class MyForeachWriter extends _root_.org.apache.spark.sql.ForeachWriter[_root_.org.apache.spark.sql.Row] {
//  def open(partitionId: Long, version: Long) = {
//    println("***", partitionId, version)
//    true
//  }
//  //  def process(value: Int) = println(value)
//  def process(value: _root_.org.apache.spark.sql.Row, partitionId: Long, version: Long) = if (value != null) {
//    value match {
//      case _root_.org.apache.spark.sql.Row(arr: WrappedArray[Double]) =>
//        println(s"${arr.mkString}, ($partitionId, $version)")
//      case x => println(x)
//    }
//  }
//  def close(errorOrNull: Throwable): Unit = if (errorOrNull != null) println(errorOrNull)
//}
