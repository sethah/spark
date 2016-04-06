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

package org.apache.spark.ml.tree.configuration

import scala.beans.BeanProperty

import org.apache.spark.mllib.tree.configuration.{BoostingStrategy => OldBoostingStrategy}
import org.apache.spark.mllib.tree.loss.Loss


/**
 * Configuration options for [[org.apache.spark.ml.tree.impl.GradientBoostedTrees]].
 *
 * @param treeStrategy Parameters for the tree algorithm. We support regression and binary
 *                     classification for boosting. Impurity setting will be ignored.
 * @param loss Loss function used for minimization during gradient boosting.
 * @param numIterations Number of iterations of boosting.  In other words, the number of
 *                      weak hypotheses used in the final model.
 * @param learningRate Learning rate for shrinking the contribution of each estimator. The
 *                     learning rate should be between in the interval (0, 1]
 * @param validationTol validationTol is a condition which decides iteration termination when
 *                      runWithValidation is used.
 *                      The end of iteration is decided based on below logic:
 *                      If the current loss on the validation set is > 0.01, the diff
 *                      of validation error is compared to relative tolerance which is
 *                      validationTol * (current loss on the validation set).
 *                      If the current loss on the validation set is <= 0.01, the diff
 *                      of validation error is compared to absolute tolerance which is
 *                      validationTol * 0.01.
 *                      Ignored when
 *                      [[org.apache.spark.ml.tree.impl.GradientBoostedTrees.run()]] is used.
 */
private[tree] case class BoostingStrategy (
    // Required boosting parameters
    @BeanProperty var treeStrategy: Strategy,
    @BeanProperty var loss: Loss,
    // Optional boosting parameters
    @BeanProperty var numIterations: Int = 100,
    @BeanProperty var learningRate: Double = 0.1,
    @BeanProperty var validationTol: Double = 0.001) extends Serializable {

  /**
   * Check validity of parameters.
   * Throws exception if invalid.
   */
  def assertValid(): Unit = {
    treeStrategy.algo match {
      case Algo.Classification =>
        require(treeStrategy.numClasses == 2,
          "Only binary classification is supported for boosting.")
      case Algo.Regression =>
        // nothing
      case _ =>
        throw new IllegalArgumentException(
          s"BoostingStrategy given invalid algo parameter: ${treeStrategy.algo}." +
            s"  Valid settings are: Classification, Regression.")
    }
    require(learningRate > 0 && learningRate <= 1,
      "Learning rate should be in range (0, 1]. Provided learning rate is " + s"$learningRate.")
  }

  /**
   * Convert a Boosting Strategy instance to the old API.
   */
  def toOld: OldBoostingStrategy = {
    new OldBoostingStrategy(treeStrategy.toOld, loss, numIterations, learningRate,
      validationTol)
  }
}
