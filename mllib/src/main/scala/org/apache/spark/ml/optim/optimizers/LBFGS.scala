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
import breeze.optimize.{CachedDiffFunction, DiffFunction, LBFGS => BreezeLBFGS}
import org.apache.spark.SparkException
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.optim.DifferentiableFunction
import org.apache.spark.ml.param.{Params, ParamMap}
import org.apache.spark.ml.param.shared.{HasMaxIter, HasTol}
import org.apache.spark.ml.util.Identifiable

import scala.collection.mutable

trait LBFGSParams extends Params with HasMaxIter with HasTol

class LBFGS(override val uid: String)
  extends IterativeOptimizer[DenseVector,
    DifferentiableFunction[DenseVector],
    BreezeWrapperState[DenseVector]]
    with LBFGSParams with Logging {

  def this() = this(Identifiable.randomUID("lbfgs"))

  private type State = BreezeWrapperState[DenseVector]
  private val lossHistoryLength = 5

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
    val breezeOptimizer = new BreezeLBFGS[BDV[Double]](getMaxIter, 10, getTol)
    val bIter = breezeOptimizer.iterations(breezeLoss, start.params.asBreeze.toDenseVector)
    bIter.map { bstate =>
      BreezeWrapperState(new DenseVector(bstate.x.data), bstate.iter + 1, bstate.adjustedValue)
    }
  }
}

//class LBFGS[T](
//    m: Int,
//    override val stoppingCriteria: (FirstOrderOptimizerState[T, (IndexedSeq[T], IndexedSeq[T])]) => Boolean)
//    (implicit space: NormedInnerProductSpace[T, Double],
//     bspace: breeze.math.MutableInnerProductModule[T, Double])
//  extends FirstOrderOptimizer[T] {
//
//  type History = (IndexedSeq[T], IndexedSeq[T])
//
//  override val uid: String = "asdf"
//
//  override def takeStep(position: T, stepDirection: T, stepSize: Double): T = {
//    space.combine(Seq((position, 1.0), (stepDirection, stepSize)))
//  }
//
//  override def chooseStepSize(lossFunction: DifferentiableFunction[T],
//                              direction: T, state: State): Double = {
//    val x = state.params
//    val grad = state.gradient
//    val breezeFunc = DifferentiableFunction.toBreeze(lossFunction)
//
//    val ff = LineSearch.functionFromSearchDirection(breezeFunc, x, direction)
//    // TODO: Need good default values here.
//    val search = new StrongWolfeLineSearch(maxZoomIter = 10, maxLineSearchIter = 10)
//    val alpha = search.minimize(ff, if (state.iter == 0.0) 1.0 / space.norm(direction) else 1.0)
//
//    if(alpha * space.norm(grad) < 1E-10) throw new Exception("Step size underflow")
//    alpha
//  }
//
//  override def chooseDescentDirection(state: State): T = {
//    val bHess = new ApproximateInverseHessian(m, state.history._1, state.history._2)
//    bHess.*(state.gradient)
//  }
//
//  override def initialHistory(lossFunction: DifferentiableFunction[T],
//                              initialParams: T): History = {
//    (IndexedSeq.empty[T], IndexedSeq.empty[T])
//  }
//
//  override def updateHistory(position: T, gradient: T, value: Double, state: State): History = {
//    val (oldDeltaPositions, oldDeltaGrads) = state.history
//    val newDeltaGrad = space.combine(Seq((gradient, 1.0), (state.gradient, -1.0)))
//    val newDeltaPosition = space.combine(Seq((position, 1.0), (state.params, -1.0)))
//    ((oldDeltaPositions :+ newDeltaPosition).take(m), (oldDeltaGrads :+ newDeltaGrad).take(m))
//  }
//
//  override def copy(extra: ParamMap): LBFGS[T] = {
//    new LBFGS[T](m, stoppingCriteria)
//  }
//}
