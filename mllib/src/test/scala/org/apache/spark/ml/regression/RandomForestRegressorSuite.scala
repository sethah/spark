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

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.feature.{Instance, LabeledPoint}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.tree.{ContinuousSplit, InternalNode, LeafNode, Node}
import org.apache.spark.ml.tree.impl.TreeTests
import org.apache.spark.ml.util.DefaultReadWriteTest
import org.apache.spark.ml.util.MLTestingUtils
import org.apache.spark.ml.util.TestingUtils._
import org.apache.spark.mllib.regression.{LabeledPoint => OldLabeledPoint}
import org.apache.spark.mllib.tree.{EnsembleTestHelper, RandomForest => OldRandomForest}
import org.apache.spark.mllib.tree.configuration.{Algo => OldAlgo}
import org.apache.spark.mllib.util.{LinearDataGenerator, MLlibTestSparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._


/**
 * Test suite for [[RandomForestRegressor]].
 */
class RandomForestRegressorSuite extends SparkFunSuite with MLlibTestSparkContext
  with DefaultReadWriteTest{

  import RandomForestRegressorSuite.compareAPIs
  import testImplicits._

  private var orderedLabeledPoints50_1000: RDD[LabeledPoint] = _
  private var datasetWithStrongNoise: DataFrame = _

  private val seed = 42

  override def beforeAll() {
    super.beforeAll()
    orderedLabeledPoints50_1000 =
      sc.parallelize(EnsembleTestHelper.generateOrderedLabeledPoints(numFeatures = 50, 1000)
        .map(_.asML))

    datasetWithStrongNoise = sc.parallelize(LinearDataGenerator.generateLinearInput(
      intercept = 6.3, weights = Array(4.7, 7.2), xMean = Array(0.9, -1.3),
      xVariance = Array(0.7, 1.2), nPoints = 1000, seed, eps = 0.5), 2).map(_.asML).toDF()
  }

  /////////////////////////////////////////////////////////////////////////////
  // Tests calling train()
  /////////////////////////////////////////////////////////////////////////////

  def regressionTestWithContinuousFeatures(rf: RandomForestRegressor) {
    val categoricalFeaturesInfo = Map.empty[Int, Int]
    val newRF = rf
      .setImpurity("variance")
      .setMaxDepth(2)
      .setMaxBins(10)
      .setNumTrees(1)
      .setFeatureSubsetStrategy("auto")
      .setSeed(123)
    compareAPIs(orderedLabeledPoints50_1000, newRF, categoricalFeaturesInfo)
  }

  test("Regression with continuous features:" +
    " comparing DecisionTree vs. RandomForest(numTrees = 1)") {
    val rf = new RandomForestRegressor()
    regressionTestWithContinuousFeatures(rf)
  }

  test("Regression with continuous features and node Id cache :" +
    " comparing DecisionTree vs. RandomForest(numTrees = 1)") {
    val rf = new RandomForestRegressor()
      .setCacheNodeIds(true)
    regressionTestWithContinuousFeatures(rf)
  }

  test("Feature importance with toy data") {
    val rf = new RandomForestRegressor()
      .setImpurity("variance")
      .setMaxDepth(3)
      .setNumTrees(3)
      .setFeatureSubsetStrategy("all")
      .setSubsamplingRate(1.0)
      .setSeed(123)

    // In this data, feature 1 is very important.
    val data: RDD[LabeledPoint] = TreeTests.featureImportanceData(sc)
    val categoricalFeatures = Map.empty[Int, Int]
    val df: DataFrame = TreeTests.setMetadata(data.map(_.toInstance), categoricalFeatures, 0)

    val model = rf.fit(df)

    val importances = model.featureImportances
    val mostImportantFeature = importances.argmax
    assert(mostImportantFeature === 1)
    assert(importances.toArray.sum === 1.0)
    assert(importances.toArray.forall(_ >= 0.0))
  }

  test("should support all NumericType labels and not support other types") {
    val rf = new RandomForestRegressor().setMaxDepth(1)
    MLTestingUtils.checkNumericTypes[RandomForestRegressionModel, RandomForestRegressor](
      rf, spark, isClassification = false) { (expected, actual) =>
        TreeTests.checkEqual(expected, actual)
      }
  }

//  test("training with weighted data") {
//    val rf = new RandomForestRegressor().setNumTrees(2).setMaxDepth(3)
//    TreeTests.testWeightedPredictions[RandomForestRegressionModel, RandomForestRegressor](
//      rf, spark, isClassification = true)
//  }

  test("random forest with weighted samples") {
    val sqlContext = spark.sqlContext
    import sqlContext.implicits._
    val numClasses = 0
    val data = EnsembleTestHelper.generateNonlinearRegressionData(1, 1000)
    val df = data.toSeq.map(_.asML).toDF()
    def featureImportanceEquals(m1: RandomForestRegressionModel,
                                m2: RandomForestRegressionModel): Unit = {
      println(m1.featureImportances, m2.featureImportances)
      assert(m1.featureImportances ~== m2.featureImportances relTol 0.05)
    }

    val testParams = Seq(
      // (numTrees, minWeightFractionPerNode, minInstancesPerNode)
      (20, 0.0, 10)
//      (10, 0.0, 10),
//      (10, 0.05, 1)
    )

    for ((numTrees, minWeightFractionPerNode, minInstancesPerNode) <- testParams) {
      val estimator = new RandomForestRegressor()
        .setMinWeightFractionPerNode(minWeightFractionPerNode)
        .setMinInstancesPerNode(minInstancesPerNode)
        .setNumTrees(numTrees)
        .setMaxDepth(10)
        .setSeed(42)

      // compare predictions on data instead of actual models because the randomness introduced in
      // bootstrapping can cause models built on the same data to predict differently in some
      // cases, especially near boundaries where the label's behavior changes abruptly
//      MLTestingUtils.testArbitrarilyScaledWeights[RandomForestRegressionModel,
//        RandomForestRegressor](df.as[LabeledPoint], estimator,
//        MLTestingUtils.modelPredictionEquals(df, MLTestingUtils.relativeTolerance(0.1), 0.98))
      datasetWithStrongNoise.show()
      MLTestingUtils.testOutliersWithSmallWeights[RandomForestRegressionModel,
        RandomForestRegressor](datasetWithStrongNoise.as[LabeledPoint], estimator,
        numClasses,
        MLTestingUtils.modelPredictionEquals(datasetWithStrongNoise, MLTestingUtils.relativeTolerance(0.2), 0.9),
        outlierRatio = 1)
//      MLTestingUtils.testOversamplingVsWeighting[RandomForestRegressionModel,
//        RandomForestRegressor](datasetWithStrongNoise.as[LabeledPoint], estimator,
//        featureImportanceEquals, 42L)
//        MLTestingUtils.modelPredictionEquals(df, MLTestingUtils.relativeTolerance(0.1), 0.9), 42L)
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // Tests of model save/load
  /////////////////////////////////////////////////////////////////////////////

  test("read/write") {
    def checkModelData(
        model: RandomForestRegressionModel,
        model2: RandomForestRegressionModel): Unit = {
      TreeTests.checkEqual(model, model2)
      assert(model.numFeatures === model2.numFeatures)
    }

    val rf = new RandomForestRegressor().setNumTrees(2)
    val rdd = TreeTests.getTreeReadWriteData(sc)

    val allParamSettings = TreeTests.allParamSettings ++ Map("impurity" -> "variance")

    val continuousData: DataFrame =
      TreeTests.setMetadata(rdd.map(_.toInstance), Map.empty[Int, Int], numClasses = 0)
    testEstimatorAndModelReadWrite(rf, continuousData, allParamSettings, checkModelData)
  }
}

private object RandomForestRegressorSuite extends SparkFunSuite {

  /**
   * Train 2 models on the given dataset, one using the old API and one using the new API.
   * Convert the old model to the new format, compare them, and fail if they are not exactly equal.
   */
  def compareAPIs(
      data: RDD[LabeledPoint],
      rf: RandomForestRegressor,
      categoricalFeatures: Map[Int, Int]): Unit = {
    val numFeatures = data.first().features.size
    val oldStrategy =
      rf.getOldStrategy(categoricalFeatures, numClasses = 0, OldAlgo.Regression, rf.getOldImpurity)
    val oldModel = OldRandomForest.trainRegressor(data.map(OldLabeledPoint.fromML), oldStrategy,
      rf.getNumTrees, rf.getFeatureSubsetStrategy, rf.getSeed.toInt)
    val newData: DataFrame = TreeTests.setMetadata(data.map(_.toInstance),
      categoricalFeatures, numClasses = 0)
    val newModel = rf.fit(newData)
    // Use parent from newTree since this is not checked anyways.
    val oldModelAsNew = RandomForestRegressionModel.fromOld(
      oldModel, newModel.parent.asInstanceOf[RandomForestRegressor], categoricalFeatures)
    TreeTests.checkEqual(oldModelAsNew, newModel)
    assert(newModel.numFeatures === numFeatures)
  }
}
