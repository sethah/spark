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
package org.apache.spark.ml.optim.optimizers

import org.apache.spark.ml.optim.linesearch.{StrongWolfe, BacktrackingLineSearch}
import org.apache.spark.ml.optim.DifferentiableFunction
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable

class GradientDescent[T](override val uid: String)
  (implicit space: NormedInnerProductSpace[T, Double])
  extends FirstOrderOptimizer[T] {

  def this()(implicit space: NormedInnerProductSpace[T, Double]) = {
    this(Identifiable.randomUID("gd"))
  }

  /**
   * Maximum number of iterations for gradient descent.
   */
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  override def copy(extra: ParamMap): GradientDescent[T] = {
    new GradientDescent[T](uid)
  }

  type History = T

  def initialHistory(lossFunction: DifferentiableFunction[T], initialParams: T): History = {
    // TODO
    initialParams
  }

  def converged(state: State): Boolean = {
    state.iter > 10
  }

  def updateHistory(
      position: T,
      gradient: T,
      value: Double,
      state: State): T = {
    // TODO: do we need to keep a history for gradient descent?
    gradient
  }

  def takeStep(position: T, stepDirection: T, stepSize: Double, history: History): T = {
    space.combine(Seq((position, 1.0), (stepDirection, stepSize)))
  }

  /**
   * Constant step size for now.
   */
  def chooseStepSize(lossFunction: DifferentiableFunction[T],
                     direction: T, state: State): Double = {
    val dirNorm = space.dot(direction, direction)
//    val lineSearch = new BacktrackingLineSearch(dirNorm)
    val lineSearch = new StrongWolfe()
    val lineSearchFunction = new DifferentiableFunction[Double] {
      def apply(x: Double): Double = {
        compute(x)._1
      }

      def gradientAt(x: Double): Double = compute(x)._2

      def compute(x: Double): (Double, Double) = {
        val nextPoint = space.combine(Seq((state.params, 1.0), (direction, x)))
        val (f, grad) = lossFunction.compute(nextPoint)
        (f, space.dot(grad, direction))
      }
    }
    lineSearch.optimize(lineSearchFunction, 1.0)
//    stepSize
  }

  /**
   * The step direction for gradient descent is simply the gradient.
   */
  def chooseDescentDirection(state: State): T = {
    space.combine(Seq((state.gradient, -1.0)))
  }

}
