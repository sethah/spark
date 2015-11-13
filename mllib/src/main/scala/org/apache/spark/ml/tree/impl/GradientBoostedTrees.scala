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

package org.apache.spark.ml.tree.impl

import org.apache.spark.Logging
import org.apache.spark.annotation.Since
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.impl.PeriodicRDDCheckpointer
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.ml.regression.{DecisionTreeRegressionModel, DecisionTreeRegressor}
<<<<<<< Updated upstream
import org.apache.spark.ml.tree.configuration.Algo._
import org.apache.spark.ml.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.impl.TimeTracker
import org.apache.spark.ml.tree.impurity.Variance
import org.apache.spark.ml.tree.loss.Loss
=======
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.impl.TimeTracker
import org.apache.spark.mllib.tree.impurity.Variance
import org.apache.spark.mllib.tree.loss.Loss
>>>>>>> Stashed changes
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.ml.tree._

private[ml] object GradientBoostedTrees extends Logging {

  /**
   * Method to train a gradient boosting model
   * @param input Training dataset: RDD of [[org.apache.spark.mllib.regression.LabeledPoint]].
   * @return a gradient boosted trees model that can be used for prediction
   */
  @Since("1.7.0")
  def run(input: RDD[LabeledPoint],
           boostingStrategy: BoostingStrategy): (Array[DecisionTreeRegressionModel], Array[Double]) = {
    val algo = boostingStrategy.treeStrategy.algo
    algo match {
      case Regression =>
        GradientBoostedTrees.boost(input, input, boostingStrategy, validate = false)
      case Classification =>
        // Map labels to -1, +1 so binary classification can be treated as regression.
        val remappedInput = input.map(x => new LabeledPoint((x.label * 2) - 1, x.features))
        GradientBoostedTrees.boost(remappedInput, remappedInput, boostingStrategy, validate = false)
      case _ =>
        throw new IllegalArgumentException(s"$algo is not supported by the gradient boosting.")
    }
  }

  /**
   * Java-friendly API for [[org.apache.spark.mllib.tree.GradientBoostedTrees!#run]].
   */
<<<<<<< Updated upstream
  // TODO
=======
>>>>>>> Stashed changes
//  @Since("1.7.0")
//  def run(input: JavaRDD[LabeledPoint]): (Array[DecisionTreeRegressionModel], Array[Double]) = {
//    run(input.rdd)
//  }

  /**
   * Method to validate a gradient boosting model
   * @param input Training dataset: RDD of [[org.apache.spark.mllib.regression.LabeledPoint]].
   * @param validationInput Validation dataset.
   *                        This dataset should be different from the training dataset,
   *                        but it should follow the same distribution.
   *                        E.g., these two datasets could be created from an original dataset
   *                        by using [[org.apache.spark.rdd.RDD.randomSplit()]]
   * @return a gradient boosted trees model that can be used for prediction
   */
  @Since("1.7.0")
  def runWithValidation(
      input: RDD[LabeledPoint],
      validationInput: RDD[LabeledPoint],
      boostingStrategy: BoostingStrategy): (Array[DecisionTreeRegressionModel], Array[Double]) = {
    val algo = boostingStrategy.treeStrategy.algo
    algo match {
      case Regression =>
        GradientBoostedTrees.boost(input, validationInput, boostingStrategy, validate = true)
      case Classification =>
        // Map labels to -1, +1 so binary classification can be treated as regression.
        val remappedInput = input.map(
          x => new LabeledPoint((x.label * 2) - 1, x.features))
        val remappedValidationInput = validationInput.map(
          x => new LabeledPoint((x.label * 2) - 1, x.features))
        GradientBoostedTrees.boost(remappedInput, remappedValidationInput, boostingStrategy,
          validate = true)
      case _ =>
        throw new IllegalArgumentException(s"$algo is not supported by the gradient boosting.")
    }
  }

  /**
   * Java-friendly API for [[org.apache.spark.mllib.tree.GradientBoostedTrees!#runWithValidation]].
   */
<<<<<<< Updated upstream
  // TODO
=======
>>>>>>> Stashed changes
//  @Since("1.7.0")
//  def runWithValidation(
//                         input: JavaRDD[LabeledPoint],
//                         validationInput: JavaRDD[LabeledPoint]): GradientBoostedTreesModel = {
//    runWithValidation(input.rdd, validationInput.rdd)
//  }

  /**
   * Compute the initial predictions and errors for a dataset for the first
   * iteration of gradient boosting.
   * @param data: training data.
   * @param initTreeWeight: learning rate assigned to the first tree.
   * @param initTree: first DecisionTreeModel.
   * @param loss: evaluation metric.
   * @return a RDD with each element being a zip of the prediction and error
   *         corresponding to every sample.
   */
  @Since("1.7.0")
  def computeInitialPredictionAndError(
      data: RDD[LabeledPoint],
      initTreeWeight: Double,
      initTree: DecisionTreeRegressionModel,
      loss: Loss): RDD[(Double, Double)] = {
    data.map { lp =>
      val leafNode = initTree.rootNode.predictImpl(lp.features)
      val pred = initTreeWeight * leafNode.prediction
      val error = loss.computeError(pred, lp.label)
      (pred, error)
    }
  }

  /**
   * Update a zipped predictionError RDD
   * (as obtained with computeInitialPredictionAndError)
   * @param data: training data.
   * @param predictionAndError: predictionError RDD
   * @param treeWeight: Learning rate.
   * @param tree: Tree using which the prediction and error should be updated.
   * @param loss: evaluation metric.
   * @return a RDD with each element being a zip of the prediction and error
   *         corresponding to each sample.
   */
  @Since("1.7.0")
  def updatePredictionError(
      data: RDD[LabeledPoint],
      predictionAndError: RDD[(Double, Double)],
      treeWeight: Double,
      tree: DecisionTreeRegressionModel,
      loss: Loss): RDD[(Double, Double)] = {

    val newPredError = data.zip(predictionAndError).mapPartitions { iter =>
      iter.map { case (lp, (pred, error)) =>
        val leafNode = tree.rootNode.predictImpl(lp.features)
        val newPred = pred + leafNode.prediction * treeWeight
        val newError = loss.computeError(newPred, lp.label)
        (newPred, newError)
      }
    }
    newPredError
  }

  /**
   * Internal method for performing regression using trees as base learners.
   * @param input training dataset
   * @param validationInput validation dataset, ignored if validate is set to false.
   * @param boostingStrategy boosting parameters
   * @param validate whether or not to use the validation dataset.
   * @return a gradient boosted trees model that can be used for prediction
   */
  private def boost(
      input: RDD[LabeledPoint],
      validationInput: RDD[LabeledPoint],
      boostingStrategy: BoostingStrategy,
      validate: Boolean): (Array[DecisionTreeRegressionModel], Array[Double]) = {
    val timer = new TimeTracker()
    timer.start("total")
    timer.start("init")
zzzzzz
    boostingStrategy.assertValid()

    // Initialize gradient boosting parameters
    val numIterations = boostingStrategy.numIterations
    val baseLearners = new Array[DecisionTreeRegressionModel](numIterations)
    val baseLearnerWeights = new Array[Double](numIterations)
    val loss = boostingStrategy.loss
    val learningRate = boostingStrategy.learningRate
    // Prepare strategy for individual trees, which use regression with variance impurity.
    val treeStrategy = boostingStrategy.treeStrategy.copy
    val validationTol = boostingStrategy.validationTol
    treeStrategy.algo = Regression
    treeStrategy.impurity = Variance
    treeStrategy.assertValid()

    // Cache input
    val persistedInput = if (input.getStorageLevel == StorageLevel.NONE) {
      input.persist(StorageLevel.MEMORY_AND_DISK)
      true
    } else {
      false
    }

    // Prepare periodic checkpointers
    val predErrorCheckpointer = new PeriodicRDDCheckpointer[(Double, Double)](
      treeStrategy.getCheckpointInterval, input.sparkContext)
    val validatePredErrorCheckpointer = new PeriodicRDDCheckpointer[(Double, Double)](
      treeStrategy.getCheckpointInterval, input.sparkContext)

    timer.stop("init")

    logDebug("##########")
    logDebug("Building tree 0")
    logDebug("##########")

    // Initialize tree
    timer.start("building tree 0")
    val firstTree = new DecisionTreeRegressor()
    val firstTreeModel = firstTree.trainOld(input, treeStrategy)
    val firstTreeWeight = 1.0
    baseLearners(0) = firstTreeModel
    baseLearnerWeights(0) = firstTreeWeight

    var predError: RDD[(Double, Double)] =
      computeInitialPredictionAndError(input, firstTreeWeight, firstTreeModel, loss)
    predErrorCheckpointer.update(predError)
    logDebug("error of gbt = " + predError.values.mean())

    // Note: A model of type regression is used since we require raw prediction
    timer.stop("building tree 0")

    var validatePredError: RDD[(Double, Double)] =
      computeInitialPredictionAndError(validationInput, firstTreeWeight, firstTreeModel, loss)
    if (validate) validatePredErrorCheckpointer.update(validatePredError)
    var bestValidateError = if (validate) validatePredError.values.mean() else 0.0
    var bestM = 1

    var m = 1
    var doneLearning = false
    while (m < numIterations && !doneLearning) {
      // Update data with pseudo-residuals
      val data = predError.zip(input).map { case ((pred, _), point) =>
        LabeledPoint(-loss.gradient(pred, point.label), point.features)
      }

      timer.start(s"building tree $m")
      logDebug("###################################################")
      logDebug("Gradient boosting tree iteration " + m)
      logDebug("###################################################")
      val dt = new DecisionTreeRegressor()
      val model = dt.trainOld(data, treeStrategy)
      val refinedModel = loss.refineTree(model, input, predError)
      timer.stop(s"building tree $m")
      // Update partial model
      baseLearners(m) = refinedModel
      // Note: The setting of baseLearnerWeights is incorrect for losses other than SquaredError.
      //       Technically, the weight should be optimized for the particular loss.
      //       However, the behavior should be reasonable, though not optimal.
//      baseLearnerWeights(m) = learningRate
      baseLearnerWeights(m) = 1.0

      predError = updatePredictionError(
        input, predError, baseLearnerWeights(m), baseLearners(m), loss)
      predErrorCheckpointer.update(predError)
      logDebug("error of gbt = " + predError.values.mean())

      if (validate) {
        // Stop training early if
        // 1. Reduction in error is less than the validationTol or
        // 2. If the error increases, that is if the model is overfit.
        // We want the model returned corresponding to the best validation error.

        validatePredError = updatePredictionError(
          validationInput, validatePredError, baseLearnerWeights(m), baseLearners(m), loss)
        validatePredErrorCheckpointer.update(validatePredError)
        val currentValidateError = validatePredError.values.mean()
        if (bestValidateError - currentValidateError < validationTol * Math.max(
          currentValidateError, 0.01)) {
          doneLearning = true
        } else if (currentValidateError < bestValidateError) {
          bestValidateError = currentValidateError
          bestM = m + 1
        }
      }
      m += 1
    }

    timer.stop("total")

    logInfo("Internal timing for DecisionTree:")
    logInfo(s"$timer")

    predErrorCheckpointer.deleteAllCheckpoints()
    validatePredErrorCheckpointer.deleteAllCheckpoints()
    if (persistedInput) input.unpersist()

    if (validate) {
      (baseLearners.slice(0, bestM), baseLearnerWeights.slice(0, bestM))
    } else {
      (baseLearners, baseLearnerWeights)
    }
  }
}
