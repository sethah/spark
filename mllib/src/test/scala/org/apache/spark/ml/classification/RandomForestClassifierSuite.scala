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

package org.apache.spark.ml.classification

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.feature.{Instance, LabeledPoint}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamsSuite
import org.apache.spark.ml.util.TestingUtils._
import org.apache.spark.ml.tree.impl.TreeTests
import org.apache.spark.ml.tree.LeafNode
import org.apache.spark.ml.util.{DefaultReadWriteTest, MLTestingUtils}
import org.apache.spark.mllib.regression.{LabeledPoint => OldLabeledPoint}
import org.apache.spark.mllib.tree.{EnsembleTestHelper, RandomForest => OldRandomForest}
import org.apache.spark.mllib.tree.configuration.{Algo => OldAlgo}
import org.apache.spark.mllib.util.MLlibTestSparkContext
//import org.apache.spark.mllib.util.TestingUtils._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions._

/**
 * Test suite for [[RandomForestClassifier]].
 */
class RandomForestClassifierSuite
  extends SparkFunSuite with MLlibTestSparkContext with DefaultReadWriteTest {

  import RandomForestClassifierSuite.compareAPIs
  import testImplicits._

  private var orderedLabeledPoints50_1000: RDD[LabeledPoint] = _
  private var orderedLabeledPoints5_20: RDD[LabeledPoint] = _
  private var smallMultinomialDataset: Dataset[_] = _

  override def beforeAll() {
    super.beforeAll()
    orderedLabeledPoints50_1000 =
      sc.parallelize(EnsembleTestHelper.generateOrderedLabeledPoints(numFeatures = 50, 1000))
        .map(_.asML)
    orderedLabeledPoints5_20 =
      sc.parallelize(EnsembleTestHelper.generateOrderedLabeledPoints(numFeatures = 5, 20))
        .map(_.asML)
    smallMultinomialDataset = {
      val nPoints = 100
      val coefficients = Array(
        -0.57997, 0.912083, -0.371077,
        -0.16624, -0.84355, -0.048509)

      val xMean = Array(5.843, 3.057)
      val xVariance = Array(0.6856, 0.1899)

      val testData = LogisticRegressionSuite.generateMultinomialLogisticInput(
        coefficients, xMean, xVariance, addIntercept = true, nPoints, 42)

      val df = sc.parallelize(testData, 4).toDF()
      df.cache()
      df
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // Tests calling train()
  /////////////////////////////////////////////////////////////////////////////

  def binaryClassificationTestWithContinuousFeatures(rf: RandomForestClassifier) {
    val categoricalFeatures = Map.empty[Int, Int]
    val numClasses = 2
    val newRF = rf
      .setImpurity("Gini")
      .setMaxDepth(2)
      .setNumTrees(1)
      .setFeatureSubsetStrategy("auto")
      .setSeed(123)
    compareAPIs(orderedLabeledPoints50_1000, newRF, categoricalFeatures, numClasses)
  }

  test("params") {
    ParamsSuite.checkParams(new RandomForestClassifier)
    val model = new RandomForestClassificationModel("rfc",
      Array(new DecisionTreeClassificationModel("dtc", new LeafNode(0.0, 0.0, null), 1, 2)), 2, 2)
    ParamsSuite.checkParams(model)
  }

  test("Binary classification with continuous features:" +
    " comparing DecisionTree vs. RandomForest(numTrees = 1)") {
    val rf = new RandomForestClassifier()
    binaryClassificationTestWithContinuousFeatures(rf)
  }

  test("Binary classification with continuous features and node Id cache:" +
    " comparing DecisionTree vs. RandomForest(numTrees = 1)") {
    val rf = new RandomForestClassifier()
      .setCacheNodeIds(true)
    binaryClassificationTestWithContinuousFeatures(rf)
  }

  test("alternating categorical and continuous features with multiclass labels to test indexing") {
    val arr = Array(
      LabeledPoint(0.0, Vectors.dense(1.0, 0.0, 0.0, 3.0, 1.0)),
      LabeledPoint(1.0, Vectors.dense(0.0, 1.0, 1.0, 1.0, 2.0)),
      LabeledPoint(0.0, Vectors.dense(2.0, 0.0, 0.0, 6.0, 3.0)),
      LabeledPoint(2.0, Vectors.dense(0.0, 2.0, 1.0, 3.0, 2.0))
    )
    val rdd = sc.parallelize(arr)
    val categoricalFeatures = Map(0 -> 3, 2 -> 2, 4 -> 4)
    val numClasses = 3

    val rf = new RandomForestClassifier()
      .setImpurity("Gini")
      .setMaxDepth(5)
      .setNumTrees(2)
      .setFeatureSubsetStrategy("sqrt")
      .setSeed(12345)
    compareAPIs(rdd, rf, categoricalFeatures, numClasses)
  }

  test("subsampling rate in RandomForest") {
    val rdd = orderedLabeledPoints5_20
    val categoricalFeatures = Map.empty[Int, Int]
    val numClasses = 2

    val rf1 = new RandomForestClassifier()
      .setImpurity("Gini")
      .setMaxDepth(2)
      .setCacheNodeIds(true)
      .setNumTrees(3)
      .setFeatureSubsetStrategy("auto")
      .setSeed(123)
    compareAPIs(rdd, rf1, categoricalFeatures, numClasses)

    val rf2 = rf1.setSubsamplingRate(0.5)
    compareAPIs(rdd, rf2, categoricalFeatures, numClasses)
  }

  test("predictRaw and predictProbability") {
    val rdd = orderedLabeledPoints5_20.map(_.toInstance)
    val rf = new RandomForestClassifier()
      .setImpurity("Gini")
      .setMaxDepth(3)
      .setNumTrees(3)
      .setSeed(123)
    val categoricalFeatures = Map.empty[Int, Int]
    val numClasses = 2

    val df: DataFrame = TreeTests.setMetadata(rdd, categoricalFeatures, numClasses)
    val model = rf.fit(df)

    // copied model must have the same parent.
    MLTestingUtils.checkCopy(model)

    val predictions = model.transform(df)
      .select(rf.getPredictionCol, rf.getRawPredictionCol, rf.getProbabilityCol)
      .collect()

    predictions.foreach { case Row(pred: Double, rawPred: Vector, probPred: Vector) =>
      assert(pred === rawPred.argmax,
        s"Expected prediction $pred but calculated ${rawPred.argmax} from rawPrediction.")
      val sum = rawPred.toArray.sum
      assert(Vectors.dense(rawPred.toArray.map(_ / sum)) === probPred,
        "probability prediction mismatch")
      assert(probPred.toArray.sum ~== 1.0 relTol 1E-5)
    }
  }

  test("Fitting without numClasses in metadata") {
    val df: DataFrame = spark.createDataFrame(TreeTests.featureImportanceData(sc))
    val rf = new RandomForestClassifier().setMaxDepth(1).setNumTrees(1)
    rf.fit(df)
  }

  /////////////////////////////////////////////////////////////////////////////
  // Tests of feature importance
  /////////////////////////////////////////////////////////////////////////////
  test("Feature importance with toy data") {
    val numClasses = 2
    val rf = new RandomForestClassifier()
      .setImpurity("Gini")
      .setMaxDepth(3)
      .setNumTrees(3)
      .setFeatureSubsetStrategy("all")
      .setSubsamplingRate(1.0)
      .setSeed(123)

    // In this data, feature 1 is very important.
    val data: RDD[Instance] = TreeTests.featureImportanceData(sc).map(_.toInstance)
    val categoricalFeatures = Map.empty[Int, Int]
    val df: DataFrame = TreeTests.setMetadata(data, categoricalFeatures, numClasses)

    val importances = rf.fit(df).featureImportances
    val mostImportantFeature = importances.argmax
    assert(mostImportantFeature === 1)
    assert(importances.toArray.sum === 1.0)
    assert(importances.toArray.forall(_ >= 0.0))
  }

  test("should support all NumericType labels and not support other types") {
    val rf = new RandomForestClassifier().setMaxDepth(1)
    MLTestingUtils.checkNumericTypes[RandomForestClassificationModel, RandomForestClassifier](
      rf, spark) { (expected, actual) =>
        TreeTests.checkEqual(expected, actual)
      }
  }

  test("random forest with weighted samples") {
    val sqlContext = spark.sqlContext
    import sqlContext.implicits._
    val numClasses = 2
    val data = EnsembleTestHelper.generateOrderedLabeledPoints(50, 1000, noise = 0.4)
    val df = smallMultinomialDataset.toDF()
//    val df = data.toSeq.map(_.asML).toDF()
//    val df = orderedLabeledPoints50_1000.toDF()

    //    def relativeTolerance(x: Double, y: Double, tol: Double): Boolean = {
    //      val diff = math.abs(x - y)
    //      if (math.abs(x) < Double.MinPositiveValue || math.abs(y) < Double.MinPositiveValue) {
    //        throw new IllegalArgumentException("x or y is close to zero")
    //      } else {
    //        diff < tol * math.min(math.abs(x), math.abs(y))
    //      }
    //    }
    //    def modelPredEquals(
    //        df: DataFrame,
    //        predTol: Double,
    //        fractionInTol: Double)(
    //        m1: RandomForestRegressionModel,
    //        m2: RandomForestRegressionModel): Unit = {
    //      val pred1 = m1.transform(df).select("label", "prediction", "features")
    //      val pred2 = m2.transform(df).select("label", "prediction", "features")
    //      val numExamples = df.count
    //      val inTol = pred1.collect().zip(pred2.collect()).map { case (p1, p2) =>
    //        val x = p1.getDouble(1)
    //        val y = p2.getDouble(1)
    //        val diff = math.abs(x - y)
    //        diff < predTol * math.min(math.abs(x), math.abs(y))
    //      }
    //      assert(inTol.count(b => b) / numExamples.toDouble > fractionInTol)
    //    }
    val testParams = Seq(
      // (numTrees, minWeightFractionPerNode, minInstancesPerNode)
//      (5, 0.0, 10),
      (100, 0.0, 10)
//      (10, 0.05, 1)
    )

    def featureImportanceEquals(m1: RandomForestClassificationModel,
                                m2: RandomForestClassificationModel): Unit = {
      assert(m1.featureImportances ~== m2.featureImportances absTol 0.01)
    }

    for ((numTrees, minWeightFractionPerNode, minInstancesPerNode) <- testParams) {
      val estimator = new RandomForestClassifier()
        .setMinWeightFractionPerNode(minWeightFractionPerNode)
        .setMinInstancesPerNode(minInstancesPerNode)
        .setNumTrees(numTrees)
        .setMaxDepth(10)
        .setSeed(42)

      // compare predictions on data instead of actual models because the randomness introduced in
      // bootstrapping can cause models built on the same data to predict differently in some
      // cases, especially near boundaries where the label's behavior changes abruptly
      val compareFunc = (x: Double, y: Double) => x == y
//      MLTestingUtils.testArbitrarilyScaledWeights[RandomForestClassificationModel,
//        RandomForestClassifier](df.as[LabeledPoint], estimator,
//        MLTestingUtils.modelPredictionEquals(df, compareFunc, 0.98))
//      MLTestingUtils.testOutliersWithSmallWeights[RandomForestClassificationModel,
//        RandomForestClassifier](df.as[LabeledPoint], estimator,
//        numClasses, MLTestingUtils.modelPredictionEquals(df, compareFunc, 0.9), outlierRatio = 1)
//      val (overSampledData, weightedData) =
//        MLTestingUtils.genEquivalentOversampledAndWeightedInstances(df.as[LabeledPoint], 42)
//      val model1 = estimator.fit(overSampledData)
//      val model2 = estimator.setWeightCol("weight").fit(weightedData)
//      println(model1.featureImportances)
//      println(model2.featureImportances)
      MLTestingUtils.testOversamplingVsWeighting[RandomForestClassificationModel,
        RandomForestClassifier](df.as[LabeledPoint], estimator,
        featureImportanceEquals, 42L)
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // Tests of model save/load
  /////////////////////////////////////////////////////////////////////////////

  test("read/write") {
    def checkModelData(
        model: RandomForestClassificationModel,
        model2: RandomForestClassificationModel): Unit = {
      TreeTests.checkEqual(model, model2)
      assert(model.numFeatures === model2.numFeatures)
      assert(model.numClasses === model2.numClasses)
    }

    val rf = new RandomForestClassifier().setNumTrees(2)
    val rdd = TreeTests.getTreeReadWriteData(sc).map(_.toInstance)

    val allParamSettings = TreeTests.allParamSettings ++ Map("impurity" -> "entropy")

    val continuousData: DataFrame =
      TreeTests.setMetadata(rdd, Map.empty[Int, Int], numClasses = 2)
    testEstimatorAndModelReadWrite(rf, continuousData, allParamSettings, checkModelData)
  }
}

private object RandomForestClassifierSuite extends SparkFunSuite {

  /**
   * Train 2 models on the given dataset, one using the old API and one using the new API.
   * Convert the old model to the new format, compare them, and fail if they are not exactly equal.
   */
  def compareAPIs(
      data: RDD[LabeledPoint],
      rf: RandomForestClassifier,
      categoricalFeatures: Map[Int, Int],
      numClasses: Int): Unit = {
    val numFeatures = data.first().features.size
    val oldStrategy =
      rf.getOldStrategy(categoricalFeatures, numClasses, OldAlgo.Classification, rf.getOldImpurity)
    val oldModel = OldRandomForest.trainClassifier(
      data.map(OldLabeledPoint.fromML), oldStrategy, rf.getNumTrees, rf.getFeatureSubsetStrategy,
      rf.getSeed.toInt)
    val newData: DataFrame =
      TreeTests.setMetadata(data.map(_.toInstance), categoricalFeatures, numClasses)
    val newModel = rf.fit(newData)
    // Use parent from newTree since this is not checked anyways.
    val oldModelAsNew = RandomForestClassificationModel.fromOld(
      oldModel, newModel.parent.asInstanceOf[RandomForestClassifier], categoricalFeatures,
      numClasses)
    TreeTests.checkEqual(oldModelAsNew, newModel)
    assert(newModel.hasParent)
    assert(!newModel.trees.head.asInstanceOf[DecisionTreeClassificationModel].hasParent)
    assert(newModel.numClasses === numClasses)
    assert(newModel.numFeatures === numFeatures)
  }
}
