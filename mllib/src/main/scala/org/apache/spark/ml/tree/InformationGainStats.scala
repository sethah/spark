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

package org.apache.spark.ml.tree

import org.apache.spark.annotation.{DeveloperApi, Since}
import org.apache.spark.ml.tree.impurity.ImpurityCalculator

/**
 * :: DeveloperApi ::
 * Impurity statistics for each split
 * @param gain information gain value
 * @param impurity current node impurity
 * @param impurityCalculator impurity statistics for current node
 * @param leftImpurityCalculator impurity statistics for left child node
 * @param rightImpurityCalculator impurity statistics for right child node
 * @param valid whether the current split satisfies minimum info gain or
 *              minimum number of instances per node
 */
@DeveloperApi
private[spark] class ImpurityStats(
    val gain: Double,
    val impurity: Double,
    val impurityCalculator: ImpurityCalculator,
    val leftImpurityCalculator: ImpurityCalculator,
    val rightImpurityCalculator: ImpurityCalculator,
    val valid: Boolean = true) extends Serializable {

  override def toString: String = {
    s"gain = $gain, impurity = $impurity, left impurity = $leftImpurity, " +
      s"right impurity = $rightImpurity"
  }

  def leftImpurity: Double = if (leftImpurityCalculator != null) {
    leftImpurityCalculator.calculate()
  } else {
    -1.0
  }

  def rightImpurity: Double = if (rightImpurityCalculator != null) {
    rightImpurityCalculator.calculate()
  } else {
    -1.0
  }
}

private[spark] object ImpurityStats {

  /**
   * Return an [[ImpurityStats]] object to
   * denote that current split doesn't satisfies minimum info gain or
   * minimum number of instances per node.
   */
  def getInvalidImpurityStats(impurityCalculator: ImpurityCalculator): ImpurityStats = {
    new ImpurityStats(Double.MinValue, impurityCalculator.calculate(),
      impurityCalculator, null, null, false)
  }

  /**
   * Return an [[ImpurityStats]] object
   * that only 'impurity' and 'impurityCalculator' are defined.
   */
  def getEmptyImpurityStats(impurityCalculator: ImpurityCalculator): ImpurityStats = {
    new ImpurityStats(Double.NaN, impurityCalculator.calculate(), impurityCalculator, null, null)
  }
}
