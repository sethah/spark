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
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable

class VLBFGS[T](override val uid: String, m: Int)
               (implicit space: NormedInnerProductSpace[T, Double])
  extends IterativeOptimizer[T, DifferentiableFunction[T], IterativeOptimizerState[T]] {

  type State

  def this() = this(Identifiable.randomUID("vlbfgs"))

  def copy(extra: ParamMap): VLBFGS[T] = {
    new VLBFGS[T](uid, m)
  }

  def iterations(lossFunction: DifferentiableFunction[T], initialParams: T): Iterator[State] = {
    // choose descent direction
    // line search for step size
    // take step
    // get new loss and gradient
    // update history
    // return next state
  }


}

object VLBFGS {
  case class History[T](
      m: Int,
      k: Int,
      innerProducts: Array[Array[Double]],
      posDeltas: Array[T],
      gradDeltas: Array[T])(implicit space: NormedInnerProductSpace[T, Double]) {
    private val numBasisVectors = 2 * m + 1

    def getBasisVector(idx: Int, gradient: T): T = {
      if (idx < m) posDeltas(idx)
      else if (idx < 2 * m) gradDeltas(idx - m)
      else if (idx == 2 * m) gradient
      else throw new IndexOutOfBoundsException(s"Basis vector index was invalid: $idx")
    }

    private def shiftArr(elem: T, arr: Array[T]): Array[T] = {
      // TODO: using a seq or list or whatever is probably fine here since m is small
      val newArr = new Array[T](arr.length)
      for (i <- arr.length - 2 to 0 by (-1)) {
        newArr(i + 1) = arr(i)
      }
      newArr(0) = elem
      newArr
    }

    private def shiftMatrix(matrix: Array[Array[Double]]): Array[Array[Double]] = {
      val newMatrix = Array.ofDim[Double](numBasisVectors, numBasisVectors)
      for (i <- numBasisVectors - 2 to 0 by (-1); j <- numBasisVectors - 2 to 0 by (-1)) {
        newMatrix(i + 1)(j + 1) = matrix(i)(j)
      }
      newMatrix
    }

    // TODO: we don't need to use mutable state, refactor to be more functional
    def update(gradient: T, posDelta: T, gradDelta: T): History[T] = {
      // add in the new history, shift out the old, and compute new dot products
      val newPosDeltas = shiftArr(posDelta, posDeltas)
      val newGradDeltas = shiftArr(gradDelta, gradDeltas)
      val shiftedInnerProducts = shiftMatrix(innerProducts)
      val indices = (0 to 2 * m).flatMap(i => List((0, i), (m.min(i), m.max(i)), (i, 2 * m)))
      indices.par.foreach {
        case (i, j) =>
          val dotProd = space.dot(getBasisVector(i, gradient), getBasisVector(j, gradient))
          shiftedInnerProducts(i)(j) = dotProd
          shiftedInnerProducts(j)(i) = dotProd
      }
      History(m, k + 1, shiftedInnerProducts, newPosDeltas, newGradDeltas)
    }

    /**
     * Compute sum_j=1^2m+1^ delta_j * b_j
     * @param gradient
     * @return
     */
    def computeDirection(gradient: T): T = {
      val deltas = basisCoefficients
      println(deltas.mkString(","))
      val zipped = (0 until numBasisVectors).map { i => (getBasisVector(i, gradient), deltas(i)) }
      space.combine(zipped)
//      var dir = deltas(numBasisVectors - 1) *: gradient
//      var i = 0
//      while (i < k) {
//        dir += deltas(i) *: posDeltas(i)
//        dir += deltas(i + m) *: gradDeltas(i)
//        i += 1
//      }
//      dir
    }

    /**
     * This should not be called until k > 1
     * @return
     */
    def basisCoefficients: Array[Double] = {
      val deltas = Array.tabulate(numBasisVectors) { i =>
        if (i == numBasisVectors - 1) -1.0 else 0.0
      }
      if (k == 0) {
        deltas
      } else {
        val alphas = new Array[Double](m)
        for (i <- 0 until k) {
          val rho = innerProducts(i)(i + m)
          val num = (0 until numBasisVectors).foldLeft(0.0) { (acc, j) =>
            acc + innerProducts(i)(j) * deltas(j)
          }
          println("rho", rho, num)
          alphas(i) = num / rho
          deltas(m + i) = deltas(m + i) - alphas(i)
        }

        val diag = innerProducts(0)(m) / innerProducts(m)(m)
        (0 until numBasisVectors).foreach { j =>
          deltas(j) = deltas(j) * diag
        }

        for(i <- (k - 1) to 0 by (-1)) {
          // beta = yi * p / (si * yi)
          val betaDenom = innerProducts(i)(m + i)
          val betaNum = (0 until numBasisVectors).foldLeft(0.0) { (acc, j) =>
            acc + innerProducts(m + i)(j) * deltas(j)
          }
          println("beta", betaDenom, betaNum)
          val beta = betaNum / betaDenom
          deltas(i) = deltas(i) + (alphas(i) - beta)
        }
        deltas
      }
    }
  }

  case class VLBFGSState[T](iter: Int, loss: Double, params: T, gradient: T, history: History[T])
    extends IterativeOptimizerState[T] {

  }
}
