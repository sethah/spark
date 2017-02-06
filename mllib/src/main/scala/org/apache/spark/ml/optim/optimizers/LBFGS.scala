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

import org.apache.spark.annotation.Since
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.ml.optim.DifferentiableFunction
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.param.shared.{HasMaxIter, HasTol}
import org.apache.spark.ml.util.Identifiable

trait LBFGSParams extends Params with HasMaxIter with HasTol

@Since("2.2.0")
class LBFGS @Since("2.2.0") (@Since("2.2.0") override val uid: String)
  extends IterativeMinimizer[Vector, DifferentiableFunction[Vector],
    BreezeWrapperState[Vector]] with LBFGSParams with Logging {

  @Since("2.2.0")
  def this() = this(Identifiable.randomUID("lbfgs"))

  private type State = BreezeWrapperState[Vector]

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
      lossFunction: DifferentiableFunction[Vector],
      initialParams: Vector): State = {
    val (firstLoss, _) = lossFunction.compute(initialParams)
    BreezeWrapperState(initialParams, 0, firstLoss)
  }

  @Since("2.2.0")
  override def iterations(
      lossFunction: DifferentiableFunction[Vector],
      initialParameters: Vector): Iterator[State] = {
    val start = initialState(lossFunction, initialParameters)
    val breezeLoss = DifferentiableFunction.toBreeze(lossFunction,
      (x: Vector) => new BDV[Double](x.toArray),
      (x: BDV[Double]) => new DenseVector(x.data))
    val breezeOptimizer = new BreezeLBFGS[BDV[Double]](getMaxIter, 10, getTol)
    val breezeIterations = breezeOptimizer.iterations(breezeLoss,
      start.params.asBreeze.toDenseVector)
    breezeIterations.map { breezeState =>
      BreezeWrapperState(new DenseVector(breezeState.x.data), breezeState.iter + 1,
        breezeState.adjustedValue)
    }
  }

  @Since("2.2.0")
  override def copy(extra: ParamMap): LBFGS = defaultCopy(extra)
}
