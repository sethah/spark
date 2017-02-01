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
import breeze.optimize.{CachedDiffFunction, DiffFunction, LBFGSB => BreezeLBFGSB}
import org.apache.spark.SparkException
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.ml.optim.DifferentiableFunction
import org.apache.spark.ml.param.{Params, ParamMap}
import org.apache.spark.ml.param.shared.{HasMaxIter, HasTol}
import org.apache.spark.ml.util.Identifiable

class LBFGSB(override val uid: String, lowerBounds: Vector, upperBounds: Vector)
  extends IterativeOptimizer[DenseVector,
    DifferentiableFunction[DenseVector],
    BreezeWrapperState[DenseVector]]
    with LBFGSParams with Logging {

//  def this() = this(Identifiable.randomUID("lbfgs"))

  private type State = BreezeWrapperState[DenseVector]

  def setMaxIter(value: Int): this.type = set(maxIter, value)
  setDefault(maxIter -> 100)

  def setTol(value: Double): this.type = set(tol, value)
  setDefault(tol -> 1e-6)

  override def copy(extra: ParamMap): LBFGS = {
    new LBFGS(uid)
  }

  def initialState(
                    lossFunction: DifferentiableFunction[DenseVector],
                    initialParams: DenseVector): State = {
    val (firstLoss, firstGradient) = lossFunction.compute(initialParams)
    BreezeWrapperState(initialParams, 0, firstLoss)
  }

  override def iterations(lossFunction: DifferentiableFunction[DenseVector],
                          initialParameters: DenseVector): Iterator[State] = {
    val start = initialState(lossFunction, initialParameters)
    val breezeLoss = new DiffFunction[BDV[Double]] {
      override def valueAt(x: BDV[Double]): Double = {
        lossFunction.apply(new DenseVector(x.data))
      }
      override def gradientAt(x: BDV[Double]): BDV[Double] = {
        lossFunction.gradientAt(new DenseVector(x.data)).asBreeze.toDenseVector
      }
      override def calculate(x: BDV[Double]): (Double, BDV[Double]) = {
        val (f, grad) = lossFunction.compute(new DenseVector(x.data))
        (f, grad.asBreeze.toDenseVector)
      }
    }
    val breezeOptimizer = new BreezeLBFGSB(lowerBounds.asBreeze.toDenseVector,
      upperBounds.asBreeze.toDenseVector, getMaxIter, 10, getTol)
    val bIter = breezeOptimizer.iterations(breezeLoss, start.params.asBreeze.toDenseVector)
    bIter.map { bstate =>
      BreezeWrapperState(new DenseVector(bstate.x.data), bstate.iter + 1, bstate.adjustedValue)
    }
  }
}
