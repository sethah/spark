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

import breeze.linalg.{DenseVector => BDV}
import breeze.optimize.{OWLQN => BreezeOWLQN}

import org.apache.spark.annotation.Since
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.optim.{DifferentiableFunction, HasL1Reg}
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.param.shared.{HasMaxIter, HasTol}
import org.apache.spark.ml.util.Identifiable

trait OWLQNParams extends Params with HasMaxIter with HasTol with HasL1Reg

@Since("2.2.0")
class OWLQN @Since("2.2.0") (@Since("2.2.0") override val uid: String)
  extends IterativeMinimizer[DenseVector, DifferentiableFunction[DenseVector],
    BreezeWrapperState[DenseVector]] with OWLQNParams with Logging {
  // TODO: We can make it inherit from first order minimizer in the future, right?

  @Since("2.2.0")
  def this() = this(Identifiable.randomUID("owlqn"))

  private type State = BreezeWrapperState[DenseVector]

  /**
   * Sets the L1 regularization function, mapping feature index to regularization.
   *
   * @group setParam
   */
  @Since("2.2.0")
  def setL1RegFunc(value: Int => Double): this.type = set(l1RegFunc, value)

  /**
   * Sets the maximum number of iterations.
   *
   * @group setParam
   */
  @Since("2.2.0")
  def setMaxIter(value: Int): this.type = set(maxIter, value)
  setDefault(maxIter -> 100)

  /**
   * Sets the convergence tolerance for this minimizer.
   *
   * @group setParam
   */
  @Since("2.2.0")
  def setTol(value: Double): this.type = set(tol, value)
  setDefault(tol -> 1e-6)

  private def initialState(
      lossFunction: DifferentiableFunction[DenseVector],
      initialParams: DenseVector): State = {
    val (firstLoss, _) = lossFunction.compute(initialParams)
    BreezeWrapperState(initialParams, 0, firstLoss)
  }

  @Since("2.2.0")
  override def iterations(
      lossFunction: DifferentiableFunction[DenseVector],
      initialParameters: DenseVector): Iterator[State] = {
    val firstState = initialState(lossFunction, initialParameters)
    val breezeLoss = DifferentiableFunction.toBreeze(lossFunction,
      (x: DenseVector) => new BDV[Double](x.values),
      (x: BDV[Double]) => new DenseVector(x.data))
    val breezeOptimizer = new BreezeOWLQN[Int, BDV[Double]](getMaxIter, 10, getL1RegFunc, getTol)
    val breezeIterations = breezeOptimizer.iterations(breezeLoss,
      firstState.params.asBreeze.toDenseVector)
    breezeIterations.map { breezeState =>
      BreezeWrapperState(new DenseVector(breezeState.x.data), breezeState.iter + 1,
        breezeState.adjustedValue)
    }
  }

  @Since("2.2.0")
  override def copy(extra: ParamMap): OWLQN = defaultCopy(extra)
}

