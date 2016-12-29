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
import breeze.optimize.{CachedDiffFunction, DiffFunction, OWLQN => BreezeOWLQN}
import org.apache.spark.SparkException
import org.apache.spark.internal.Logging
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.optim.DifferentiableFunction
import org.apache.spark.ml.optim.DifferentiableFunction
import org.apache.spark.ml.param.shared.{HasTol, HasMaxIter}
import org.apache.spark.ml.param.{Param, Params, ParamMap}
import org.apache.spark.ml.util.Identifiable

import scala.collection.mutable

trait HasL1Reg extends Params {

  /**
   * Param for maximum number of iterations (&gt;= 0).
   *
   * @group param
   */
  final val l1RegFunc: Param[Int => Double] = new Param(this, "l1RegFunc",
    "function for applying L1 regularization to parameters.")

  /** @group getParam */
  final def getL1RegFunc: Int => Double = $(l1RegFunc)

}

trait OWLQNParams extends Params with HasMaxIter with HasTol with HasL1Reg


class OWLQN(override val uid: String)
  extends IterativeOptimizer[DenseVector, DifferentiableFunction[DenseVector],
    BreezeWrapperState[DenseVector]] with OWLQNParams with Logging {

  private type State = BreezeWrapperState[DenseVector]
  private val lossHistoryLength = 5

  def this() = this(Identifiable.randomUID("owlqn"))

  override def copy(extra: ParamMap): OWLQN = {
    new OWLQN()
  }

  def setL1RegFunc(value: Int => Double): this.type = set(l1RegFunc, value)

  def setMaxIter(value: Int): this.type = set(maxIter, value)
  setDefault(maxIter -> 100)

  def setTol(value: Double): this.type = set(tol, value)
  setDefault(tol -> 1e-6)

  def initialState(
                    lossFunction: DifferentiableFunction[DenseVector],
                    initialParams: DenseVector): State = {
    val (firstLoss, firstGradient) = lossFunction.compute(initialParams)
    BreezeWrapperState(initialParams, 0, firstLoss)
  }


  def converged(state: State): Boolean = {
    val maxIterReached = state.iter > getMaxIter
    maxIterReached
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
    val breezeOptimizer = new BreezeOWLQN[Int, BDV[Double]](getMaxIter, 10, getL1RegFunc, getTol)
    val bIter = breezeOptimizer.iterations(breezeLoss, start.params.asBreeze.toDenseVector)
    bIter.map { bstate =>
      BreezeWrapperState(new DenseVector(bstate.x.data), bstate.iter + 1, bstate.adjustedValue)
    }
  }

}

