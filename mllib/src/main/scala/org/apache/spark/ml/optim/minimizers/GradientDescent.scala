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
package org.apache.spark.ml.optim.minimizers

import breeze.linalg.{DenseVector => BDV}
import breeze.optimize.{AdaDeltaGradientDescent, CachedDiffFunction, DiffFunction, StochasticGradientDescent}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.{BLAS, DenseVector, Vector, Vectors}
import org.apache.spark.ml.optim.DiffFun
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.param.shared.{HasMaxIter, HasTol}
import org.apache.spark.ml.util.Identifiable


trait GradientDescentParams extends Params with HasMaxIter with HasTol

class GradientDescent(override val uid: String) extends IterativeMinimizer[Vector,
  DiffFun[Vector], BreezeWrapperState[Vector]] with LBFGSParams
  with Logging {

  def this() = this(Identifiable.randomUID("lbfgs"))

  private type State = BreezeWrapperState[Vector]

  def setMaxIter(value: Int): this.type = set(maxIter, value)
  setDefault(maxIter -> 100)

  def setTol(value: Double): this.type = set(tol, value)
  setDefault(tol -> 1e-6)

  override def copy(extra: ParamMap): LBFGS = {
    new LBFGS(uid)
  }

  def initialState(lossFunction: DiffFun[Vector], initialParams: Vector): State = {
    val (firstLoss, _) = lossFunction.compute(initialParams)
    BreezeWrapperState(Vectors.dense(Array.fill(initialParams.size)(Double.MaxValue)),
      initialParams, 0, firstLoss)
  }

  override def iterations(lossFunction: DiffFun[Vector],
                          initialParameters: Vector): Iterator[State] = {
    val start = initialState(lossFunction, initialParameters)
    val breezeLoss = new DiffFunction[BDV[Double]] {
      override def valueAt(x: BDV[Double]): Double = {
        lossFunction.apply(new DenseVector(x.data))
      }
      override def gradientAt(x: BDV[Double]): BDV[Double] = {
        lossFunction.grad(new DenseVector(x.data)).asBreeze.toDenseVector
      }
      override def calculate(x: BDV[Double]): (Double, BDV[Double]) = {
        val (f, grad) = lossFunction.compute(new DenseVector(x.data))
        (f, grad.asBreeze.toDenseVector)
      }
    }
    val breezeOptimizer = new AdaDeltaGradientDescent[BDV[Double]](0.9, maxIter = getMaxIter)
    val bIter = breezeOptimizer.iterations(breezeLoss, start.params.asBreeze.toDenseVector)
    val bIterable = bIter.toIterable
    val reason = breezeOptimizer.convergenceCheck.apply(bIterable.last,
      bIterable.last.convergenceInfo)
//    println(s"Convergence reason: ${reason}")
    bIterable.map { bstate =>
      BreezeWrapperState(Vectors.zeros(initialParameters.size),
        new DenseVector(bstate.x.data), bstate.iter + 1, bstate.adjustedValue)
    }.toIterator
  }
}

