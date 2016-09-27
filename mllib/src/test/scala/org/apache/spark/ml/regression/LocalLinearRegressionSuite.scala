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

package org.apache.spark.ml.regression

import org.apache.spark.ml.optim.WeightedLeastSquares

import scala.util.Random

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.feature.{StandardScaler, Instance, LabeledPoint}
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.ml.param.ParamsSuite
import org.apache.spark.ml.util.{DefaultReadWriteTest, MLTestingUtils}
import org.apache.spark.ml.util.TestingUtils._
import org.apache.spark.mllib.util.{LinearDataGenerator, MLlibTestSparkContext}
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions._

class LocalLinearRegressionSuite
  extends SparkFunSuite with MLlibTestSparkContext with DefaultReadWriteTest {

  private val seed: Int = 42
  @transient var datasetWithDenseFeature: DataFrame = _
  @transient var datasetWithDenseFeatureWithoutIntercept: DataFrame = _
  @transient var datasetWithSparseFeature: DataFrame = _
  @transient var datasetWithWeight: DataFrame = _
  @transient var datasetWithWeightConstantLabel: DataFrame = _
  @transient var datasetWithWeightZeroLabel: DataFrame = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    val rng = scala.util.Random
    rng.setSeed(42L)
    val numFeatures = 10
    datasetWithDenseFeature = spark.createDataFrame(
      sc.parallelize(LinearDataGenerator.generateLinearInput(
        intercept = 6.3, weights = Array.fill(numFeatures)(rng.nextDouble), xMean = Array.fill(numFeatures)(rng.nextDouble),
        xVariance = Array.fill(numFeatures)(rng.nextDouble), nPoints = 10000, seed, eps = 0.1), 2).map(_.asML))
    /*
       datasetWithDenseFeatureWithoutIntercept is not needed for correctness testing
       but is useful for illustrating training model without intercept
     */
    datasetWithDenseFeatureWithoutIntercept = spark.createDataFrame(
      sc.parallelize(LinearDataGenerator.generateLinearInput(
        intercept = 0.0, weights = Array(4.7, 7.2), xMean = Array(0.9, -1.3),
        xVariance = Array(0.7, 1.2), nPoints = 10000, seed, eps = 0.1), 2).map(_.asML))

    val r = new Random(seed)
    // When feature size is larger than 4096, normal optimizer is choosed
    // as the solver of linear regression in the case of "auto" mode.
    val featureSize = 4100
    datasetWithSparseFeature = spark.createDataFrame(
      sc.parallelize(LinearDataGenerator.generateLinearInput(
        intercept = 0.0, weights = Seq.fill(featureSize)(r.nextDouble()).toArray,
        xMean = Seq.fill(featureSize)(r.nextDouble()).toArray,
        xVariance = Seq.fill(featureSize)(r.nextDouble()).toArray, nPoints = 200,
        seed, eps = 0.1, sparsity = 0.7), 2).map(_.asML))

    /*
       R code:

       A <- matrix(c(0, 1, 2, 3, 5, 7, 11, 13), 4, 2)
       b <- c(17, 19, 23, 29)
       w <- c(1, 2, 3, 4)
       df <- as.data.frame(cbind(A, b))
     */
    datasetWithWeight = spark.createDataFrame(
      sc.parallelize(Seq(
        Instance(17.0, 1.0, Vectors.dense(0.0, 5.0).toSparse),
        Instance(19.0, 2.0, Vectors.dense(1.0, 7.0)),
        Instance(23.0, 3.0, Vectors.dense(2.0, 11.0)),
        Instance(29.0, 4.0, Vectors.dense(3.0, 13.0))
      ), 2))

    /*
       R code:

       A <- matrix(c(0, 1, 2, 3, 5, 7, 11, 13), 4, 2)
       b.const <- c(17, 17, 17, 17)
       w <- c(1, 2, 3, 4)
       df.const.label <- as.data.frame(cbind(A, b.const))
     */
    datasetWithWeightConstantLabel = spark.createDataFrame(
      sc.parallelize(Seq(
        Instance(17.0, 1.0, Vectors.dense(0.0, 5.0).toSparse),
        Instance(17.0, 2.0, Vectors.dense(1.0, 7.0)),
        Instance(17.0, 3.0, Vectors.dense(2.0, 11.0)),
        Instance(17.0, 4.0, Vectors.dense(3.0, 13.0))
      ), 2))
    datasetWithWeightZeroLabel = spark.createDataFrame(
      sc.parallelize(Seq(
        Instance(0.0, 1.0, Vectors.dense(0.0, 5.0).toSparse),
        Instance(0.0, 2.0, Vectors.dense(1.0, 7.0)),
        Instance(0.0, 3.0, Vectors.dense(2.0, 11.0)),
        Instance(0.0, 4.0, Vectors.dense(3.0, 13.0))
      ), 2))
  }
  test("export test data into CSV format") {
    datasetWithDenseFeature.rdd.map { case Row(label: Double, features: Vector) =>
      label + "," + features.toArray.mkString(",")
    }.repartition(1).saveAsTextFile("target/tmp/LinearRegressionSuite/llr")
  }

  test("weighted least squares") {

    val sqlContext = datasetWithDenseFeature.sqlContext
    import sqlContext.implicits._
    val ss = new StandardScaler().setWithStd(true).setInputCol("features")
      .setOutputCol("stdFeatures")
    val ssModel = ss.fit(datasetWithDenseFeature)
    val df = ssModel.transform(datasetWithDenseFeature).select(col("label"),
      col("stdFeatures").as("features"))
    val rdd = datasetWithDenseFeature.as[LabeledPoint].rdd.map { lp =>
      Instance(lp.label, 1.0, lp.features)
    }
    val solver = "quasi-newton"
//    val solver = "cholesky"
    val regParam = 0.1
    val fitIntercept = true
    val standardize = false
    val elasticNetParam = 0.5
    val wls = new WeightedLeastSquares(fitIntercept, regParam, standardize, true, elasticNetParam, solver)
    val wlsModel = wls.fit(rdd)
    val lr = new LinearRegression().setSolver("L-BFGS").setRegParam(regParam)
      .setStandardization(standardize)
      .setFitIntercept(fitIntercept)
      .setElasticNetParam(elasticNetParam)
    val lrModel = lr.fit(datasetWithDenseFeature)
    println(wlsModel.coefficients, wlsModel.intercept)
    println(lrModel.coefficients, lrModel.intercept)
  }

  test("local linear regression") {
    val sqlContext = datasetWithDenseFeature.sqlContext
    import sqlContext.implicits._
    val ss = new StandardScaler().setWithStd(true).setInputCol("features")
      .setOutputCol("stdFeatures")
    val ssModel = ss.fit(datasetWithDenseFeature)
//    ss.setInputCol("label").setOutputCol("stdLabel")
//    val ssModel2 = ss.fit(datasetWithDenseFeature)
    val df = ssModel.transform(datasetWithDenseFeature).select(col("label"),
      col("stdFeatures").as("features"))
//    val df2 = ssModel2.transform(df).select(col("stdLabel").as("label"), col("features"))
    val rdd = datasetWithDenseFeature.as[LabeledPoint].rdd.map { lp =>
//      val f = lp.features.toArray ++ Array(1.0)
//      Instance(lp.label, 1.0, Vectors.dense(f))
      Instance(lp.label, 1.0, lp.features)
    }
//    println(rdd.first().features)
    val llr = new LocalLinearRegression()
    val coef = llr.fit(rdd)
    println(coef.count(_ > 0.0))
    println(Vectors.dense(coef))
    val lr = new LinearRegression().setFitIntercept(true).setSolver("LBFGS")//.setStandardization(false)
//    lr.setElasticNetParam(1.0).setRegParam(0.1)
    val lrModel = lr.fit(rdd.toDF())
    println(lrModel.coefficients.toArray.count(_ > 0.0))
    println(lrModel.coefficients, lrModel.intercept)
  }
}

