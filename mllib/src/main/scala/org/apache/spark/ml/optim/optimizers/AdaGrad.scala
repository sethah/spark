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

import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.optim.DifferentiableFunction
import org.apache.spark.ml.param.ParamMap

class AdaGrad(override val uid: String, stepSize: Double)
             (implicit space: NormedInnerProductSpace[DenseVector, Double])
  extends FirstOrderOptimizer[DenseVector] {

  override def copy(extra: ParamMap): AdaGrad = {
    new AdaGrad(uid, stepSize)
  }

  type History = (DenseVector, Long)

  def initialHistory(lossFunction: DifferentiableFunction[DenseVector],
                     initialParams: DenseVector): History = {
    // TODO
    (new DenseVector(Array.fill(initialParams.size)(0.0)), 0L)
  }

  def converged(state: State): Boolean = {
    state.iter > 1000
  }

  def updateHistory(
                     position: DenseVector,
                     gradient: DenseVector,
                     value: Double,
                     state: State): History = {
    // TODO
    val nextHist = Array.tabulate(position.size) { i =>
      state.history._1(i) + gradient(i) * gradient(i)
    }
    val tmp = new DenseVector(nextHist)
//    println(tmp)
    (tmp, state.history._2 + 1L)
  }

  def takeStep(position: DenseVector, stepDirection: DenseVector,
               stepSize: Double, history: History): DenseVector = {
    val stepped = new DenseVector(new Array[Double](position.size))
    val gamma = 0.1
    (0 until stepDirection.size).foreach { i =>
      stepped.toArray(i) = 1.0 / math.sqrt(gamma * history._1(i) / history._2 +
        stepDirection(i) + 1e-8) * stepDirection(i)
    }
//    println("stepped", stepped)

    val next = space.combine(Seq((position, 1.0), (stepped, stepSize)))
    println(next)
    next
  }

  def chooseStepSize(lossFunction: DifferentiableFunction[DenseVector], direction: DenseVector,
                     state: State): Double = {
    stepSize
  }

  def chooseDescentDirection(state: State): DenseVector = {
    space.combine(Seq((state.gradient, -1.0)))
  }

}
