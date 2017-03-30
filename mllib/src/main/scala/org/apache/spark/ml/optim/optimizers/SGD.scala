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
import breeze.optimize.{LBFGS => BreezeLBFGS}
import breeze.optimize.StochasticGradientDescent.SimpleSGD
import org.apache.spark.annotation.Since
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.ml.optim.DifferentiableFunction
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.param.shared.{HasMaxIter, HasTol}
import org.apache.spark.ml.util.Identifiable

trait SGDParams extends Params with HasMaxIter with HasTol

@Since("2.2.0")
class SGD @Since("2.2.0") (@Since("2.2.0") override val uid: String)
  extends IterativeMinimizer[Vector, DifferentiableFunction[Vector],
    BreezeWrapperState[Vector]] with SGDParams with Logging {

  @Since("2.2.0")
  def this() = this(Identifiable.randomUID("sgd"))

  /** Type alias for convenience */
  private type State = BreezeWrapperState[Vector]

  /**
   * Sets the maximum number of iterations.
   * Default is 100.
   *
   * @group setParam
   */
  @Since("2.2.0")
  def setMaxIter(value: Int): this.type = set(maxIter, value)
  setDefault(maxIter -> 100)

  /**
   * Sets the convergence tolerance for this minimizer.
   * Default is 1e-6.
   *
   * @group setParam
   */
  @Since("2.2.0")
  def setTol(value: Double): this.type = set(tol, value)
  setDefault(tol -> 1e-6)

  private def initialState(
                            lossFunction: DifferentiableFunction[Vector],
                            initialParams: Vector): State = {
    val firstLoss = lossFunction.apply(initialParams)
    BreezeWrapperState(initialParams, 0, firstLoss, Vectors.zeros(initialParams.size))
  }

  @Since("2.2.0")
  override def iterations(
                           lossFunction: DifferentiableFunction[Vector],
                           initialParameters: Vector): Iterator[State] = {
    val start = initialState(lossFunction, initialParameters)
    val breezeLoss = DifferentiableFunction.toBreeze(lossFunction,
      (x: Vector) => new BDV[Double](x.toArray),
      (x: BDV[Double]) => new DenseVector(x.data))
    val breezeOptimizer = new SimpleSGD[BDV[Double]](eta = 4, maxIter = getMaxIter)
    val breezeIterations = breezeOptimizer.iterations(breezeLoss,
      start.params.asBreeze.toDenseVector)
    breezeIterations.map { breezeState =>
      BreezeWrapperState(new DenseVector(breezeState.x.data), breezeState.iter + 1,
        breezeState.adjustedValue, Vectors.fromBreeze(breezeState.adjustedGradient))
    }
  }

  @Since("2.2.0")
  override def copy(extra: ParamMap): SGD = defaultCopy(extra)
}

