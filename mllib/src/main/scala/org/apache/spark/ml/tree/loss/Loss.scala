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

import org.apache.spark.annotation.DeveloperApi


/**
 * :: DeveloperApi ::
 * Trait for adding "pluggable" loss functions for the gradient boosting algorithm.
 */
@DeveloperApi
private[spark] trait Loss extends Serializable {

  /**
   * Method to calculate the gradients for the gradient boosting calculation.
   * @param prediction Predicted feature
   * @param label true label.
   * @return Loss gradient.
   */
  def gradient(prediction: Double, label: Double): Double

  /**
   * Method to calculate loss when the predictions are already known.
   * Note: This method is used in the method evaluateEachIteration to avoid recomputing the
   * predicted values from previously fit trees.
   * @param prediction Predicted label.
   * @param label True label.
   * @return Measure of model error on datapoint.
   */
  private[spark] def computeError(prediction: Double, label: Double): Double
}
