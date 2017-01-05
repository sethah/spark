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

import org.apache.spark.ml.optim.DifferentiableFunction
import org.apache.spark.ml.param.Params
import org.apache.spark.ml.param.shared.HasMaxIter

trait FirstOrderOptimizerParams extends Params with HasMaxIter
//
//trait FirstOrderOptimizer[T] extends IterativeOptimizer[T, DifferentiableFunction[T]]
//  with FirstOrderOptimizerParams {
//
//
//  /** An abstract type alias for the history required to be tracked in the subclass optimizers. */
//  // TODO: would be nice to eliminate ambiguous things like this
//  type History
//
//  override type State = FirstOrderOptimizerState[T, History]
//
//  /*
//  Should we make a one-size-fits-all State for first order optimizers? The reason to do this is
//  that we can implement some methods here that construct and return that state. If we make the state
//  generic like a trait, then how can we construct and return it here? The reason not to is that
//  maybe some things don't apply to all subclasses - like gd doesn't need a history...
//   */
//
//  //  def stoppingCriteria: StoppingCriteria[State]
////  def stoppingCriteria: (State => Boolean)
//
//  def initialState(
//      lossFunction: DifferentiableFunction[T],
//      initialParams: T): State = {
//    val (firstLoss, firstGradient) = lossFunction.compute(initialParams)
//    FirstOrderOptimizerState(initialParams, 0, firstLoss, firstGradient,
//      initialHistory(lossFunction, initialParams))
//  }
//
//  /**
//   * Step into the next state from the previous state. This code defines the general template for
//   * first order optimizers: choose a direction and a step size, update parameters, then compute
//   * the loss and gradient of the loss function evaluated at the new parameters.
//   *
//   * @param lossFunction The differentiable loss function.
//   * @param state The current optimization state.
//   * @return The next optimization state.
//   */
//  def iterateOnce(lossFunction: DifferentiableFunction[T])(state: State): State = {
//    val direction = chooseDescentDirection(state)
//    val stepSize = chooseStepSize(lossFunction, direction, state)
//    val nextPosition = takeStep(state.params, direction, stepSize, state.history)
//    val (nextLoss, nextGradient) = lossFunction.compute(nextPosition)
//    val nextHistory = updateHistory(nextPosition, nextGradient, nextLoss, state)
//    FirstOrderOptimizerState(nextPosition, state.iter + 1, nextLoss, nextGradient, nextHistory)
//  }
//
//  def chooseDescentDirection(state: State): T
//
//  def chooseStepSize(lossFunction: DifferentiableFunction[T], direction: T, state: State): Double
//
//  /**
//   * Return the next set of parameters from the current parameters.
//   */
//  def takeStep(position: T, stepDirection: T, stepSize: Double, history: History): T
//
//  def initialHistory(lossFunction: DifferentiableFunction[T], initialParams: T): History
//
//  /**
//   * Update the history for the optimizer. For LBFGS, this might be the information required to
//   * compute the approximate inverse Hessian matrix.
//   */
//  def updateHistory(position: T, gradient: T, value: Double, state: State): History
//
//}
//
//
/**
 * Data structure holding pertinent information about the optimizer state.
 *
 * @tparam T The type of parameters being optimized.
 */
trait OptimizerState[+T] {
  def params: T
}
trait IterativeOptimizerState[+T] extends OptimizerState[T] {
  def iter: Int
  def loss: Double
}

case class BreezeWrapperState[T](
    params: T,
    iter: Int,
    loss: Double) extends IterativeOptimizerState[T]

case class FirstOrderOptimizerState[+T, +History](
    params: T,
    iter: Int,
    loss: Double,
    gradient: T,
    history: History) extends IterativeOptimizerState[T]
