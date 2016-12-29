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
package org.apache.spark.ml.optim

import breeze.optimize.LBFGS.ApproximateInverseHessian
import breeze.optimize.{LBFGS => BreezeLBFGS, CachedDiffFunction, LineSearch, StrongWolfeLineSearch}
import org.apache.spark.SparkException
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.internal.Logging

import scala.collection.mutable

class LBFGS[T](initialCoefficients: T)
              (implicit bspace: breeze.math.MutableInnerProductModule[T, Double],
               bCopy: breeze.linalg.support.CanCopy[T])
  extends Optimizer[T, DifferentiableFunction[T]] with Logging {

  override val uid = "lbfgs"

  override def copy(extra: ParamMap): LBFGS[T] = {
    new LBFGS[T](initialCoefficients)
  }

  def optimize(lossFunction: DifferentiableFunction[T], initialParameters: T): T = {
    val breezeLoss = DifferentiableFunction.toBreeze(lossFunction)
    val breezeOptimizer = new BreezeLBFGS[T](10, 10, 1e-6)

    val states = breezeOptimizer.iterations(new CachedDiffFunction(breezeLoss),
      initialCoefficients)
    val arrayBuilder = mutable.ArrayBuilder.make[Double]
    var state: breezeOptimizer.State = null
    while (states.hasNext) {
      state = states.next()
      arrayBuilder += state.adjustedValue
    }
    if (state == null) {
      val msg = s"${breezeOptimizer.getClass.getName} failed."
      logError(msg)
      throw new SparkException(msg)
    }
    state.x
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
