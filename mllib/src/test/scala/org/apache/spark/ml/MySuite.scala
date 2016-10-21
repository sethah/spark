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
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.sql.expressions.{MutableAggregationBuffer,
UserDefinedAggregateFunction}
import scala.collection.mutable.WrappedArray
import org.apache.spark.ml.feature._
import org.apache.spark.ml.feature.{LabeledPoint, VectorAssembler}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.execution.debug._

class MySuite extends SparkFunSuite with MLlibTestSparkContext {
  test("mytest") {
    val ctx = spark
    import ctx.sqlContext.implicits._
    val rng = new scala.util.Random(42)
    val data = sc.parallelize(Array.fill(10) {
      LabeledPoint(rng.nextDouble(), Vectors.dense(Array.fill(5)(rng.nextDouble())))
    })
    val df = data.toDF()
    val df2 = df.agg(avg("label").as("asdf"))
    println(df2.logicalPlan.toString)
    println(df2.queryExecution.sparkPlan.toString)
//    df2.cache()
//    println(df2.debugCodegen())
//    df.count()
//    df.agg(sum("label")).show()
//    val rdd = df.rdd
//    println(rdd.count)
//    println(df.queryExecution.debug.codegen())
  }

  test("parquet") {
    val df = spark.read.parquet("/Users/sethhendrickson/Development/datasets/multinomialDataset/")
    println(df.debugCodegen())
//    val rdd = df.rdd
//    println(rdd.take(10).map(_.getClass().getName()))
  }
}
