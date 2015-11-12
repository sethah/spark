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

package org.apache.spark.ml.tree.loss

import org.apache.spark.annotation.{DeveloperApi, Since}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.ml.tree.TreeEnsembleModel
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.tree._
import org.apache.spark.ml.regression.DecisionTreeRegressionModel


/**
 * :: DeveloperApi ::
 * Trait for adding "pluggable" loss functions for the gradient boosting algorithm.
 */
@Since("1.2.0")
@DeveloperApi
trait Loss extends Serializable {

  /**
   * Method to calculate the gradients for the gradient boosting calculation.
   * @param prediction Predicted feature
   * @param label true label.
   * @return Loss gradient.
   */
  @Since("1.2.0")
  def gradient(prediction: Double, label: Double): Double

//  /**
//   * Method to calculate error of the base learner for the gradient boosting calculation.
//   * Note: This method is not used by the gradient boosting algorithm but is useful for debugging
//   * purposes.
//   * @param model Model of the weak learner.
//   * @param data Training dataset: RDD of [[org.apache.spark.mllib.regression.LabeledPoint]].
//   * @return Measure of model error on data
//   */
//  @Since("1.2.0")
//  def computeError(model: TreeEnsembleModel, data: RDD[LabeledPoint]): Double = {
//    data.map(point => computeError(model.predict(point.features), point.label)).mean()
//  }

  /**
   * Method to calculate loss when the predictions are already known.
   * Note: This method is used in the method evaluateEachIteration to avoid recomputing the
   * predicted values from previously fit trees.
   * @param prediction Predicted label.
   * @param label True label.
   * @return Measure of model error on datapoint.
   */
  // TODO: changed scope of this method
  private[spark] def computeError(prediction: Double, label: Double): Double

  private[ml] def refinePredictions(predsAndLabels: RDD[(Int, (Double, Double))]): Map[Int, Double]

  private[ml] def refineTree(
      tree: DecisionTreeRegressionModel,
      input: RDD[LabeledPoint],
      predError: RDD[(Double, Double)]): DecisionTreeRegressionModel = {

    val predsAndLabels = input.zip(predError).map { case (lp, (pred, _)) =>
      val leafNode = tree.rootNode.predictImpl(lp.features)
      (leafNode.id, (pred, lp.label))
    }

    val refinedPreds = refinePredictions(predsAndLabels)
    def changeTerminalNodePredictions(topNode: Node, refinedPreds: Map[Int, Double]): Node = {
      topNode match {
        case node: LeafNode =>
          new LeafNode(node.id, refinedPreds(node.id), node.impurity, node.impurityStats)
        case node: InternalNode =>
          val leftChild = changeTerminalNodePredictions(node.leftChild, refinedPreds)
          val rightChild = changeTerminalNodePredictions(node.rightChild, refinedPreds)
          new InternalNode(node.id, node.prediction, node.impurity, node.gain,
            leftChild, rightChild, node.split, node.impurityStats)
      }
    }
    val newTree = changeTerminalNodePredictions(tree.rootNode, refinedPreds)
    new DecisionTreeRegressionModel(tree.uid, newTree, tree.numFeatures)
  }
}
