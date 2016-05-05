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

package org.apache.spark.examples.ml

// scalastyle:off println

// $example on$
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
// $example off$
import org.apache.spark.sql.SparkSession
// $example on$
import org.apache.spark.sql.types.{DataTypes, StructField, StructType}
// $example off$


/**
 * An example demonstrating a k-means clustering.
 * Run with
 * {{{
 * bin/run-example ml.KMeansExample
 * }}}
 */
object KMeansExample {

  def main(args: Array[String]): Unit = {
    // Creates a Spark context and a SQL context
    val spark = SparkSession.builder.appName(s"${this.getClass.getSimpleName}").getOrCreate()

    // $example on$
    // Crates a DataFrame
    val vecAssembler = new VectorAssembler()
      .setInputCols(Array("x", "y", "z"))
      .setOutputCol("features")

    val schema = StructType(Array(
      StructField("x", DataTypes.DoubleType),
      StructField("y", DataTypes.DoubleType),
      StructField("z", DataTypes.DoubleType)))

    val dataset = vecAssembler.transform(
      spark.read
      .format("csv")
      .option("sep", " ")
      .schema(schema)
      .load("data/mllib/kmeans_data.txt"))

    // Trains a k-means model
    val kmeans = new KMeans()
      .setK(2)
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
    val model = kmeans.fit(dataset)

    // Shows the result
    println("Within Set Sum of Squared Errors = " + model.computeCost(dataset))
    println("Final Centers:")
    model.clusterCenters.foreach(println)
    // $example off$

    spark.stop()
  }
}
// scalastyle:on println
