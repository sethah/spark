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
import org.apache.spark.ml.optim.linesearch.StrongWolfe
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasMaxIter, HasTol}
import org.apache.spark.ml.util.Identifiable

import scala.reflect.ClassTag

/**
 * Vector free L-BFGS algorithm for non-linear optimization, described here:
 *   Chen, et al. "Larg-scale L-BFGS using MapReduce."
 *
 * The algorithm is generic in the type of parameters, enabling the use of distributed storage.
 * This makes it capable of scaling up to billions of parameters.
 */
class VLBFGS[T: ClassTag](override val uid: String, m: Int)
                         (implicit space: NormedInnerProductSpace[T, Double])
  extends IterativeOptimizer[T, DifferentiableFunction[T], IterativeOptimizerState[T]]
    with HasMaxIter with HasTol {

  type State = VLBFGS.VLBFGSState[T]

  def this(m: Int)(implicit space: NormedInnerProductSpace[T, Double]) = {
    this(Identifiable.randomUID("vlbfgs"), m)
  }

  def copy(extra: ParamMap): VLBFGS[T] = {
    defaultCopy(extra)
  }

  def setMaxIter(value: Int): this.type = set(maxIter, value)

  /**
   * Optimization iterations for VL-BFGS algorithm. This should be basically identical to
   * L-BFGS update. The difference comes as we compute the new search direction using the
   * history.
   *
   * @param lossFunction Differentiable loss function for computing gradients.
   * @param initialParams Initial parameters.
   * @return Iterator[State]
   */
  def iterations(lossFunction: DifferentiableFunction[T], initialParams: T): Iterator[State] = {
    val initialHistory = new VLBFGS.History[T](m, Array.ofDim[Double](2 * m + 1, 2 * m + 1),
      new Array[T](m), new Array[T](m))
    space.prepare(initialParams)
    val (initialLoss, initialGradient) = lossFunction.compute(initialParams)
    val initialState = VLBFGS.VLBFGSState(0, initialLoss, initialParams, initialGradient,
      initialHistory)
    Iterator.iterate(initialState) { state =>
      println(s"Iteration ${state.iter}: ${state.params}  ")
      val direction = state.history.computeDirection(state.gradient)
      // TODO: persist direction
      val ls = new StrongWolfe()
      val lineSearchFunction = new DifferentiableFunction[Double] {
        def apply(x: Double): Double = {
          compute(x)._1
        }

        def gradientAt(x: Double): Double = compute(x)._2

        def compute(x: Double): (Double, Double) = {
          val next = space.combine(Seq((state.params, 1.0), (direction, x)))
          space.prepare(next)
          val (f, grad) = lossFunction.compute(next)
          space.clean(next)
          (f, space.dot(grad, direction))
        }
      }
      val dirNorm = space.norm(direction)
      val initialAlpha = if (state.iter == 0) 1.0 / dirNorm else 1.0
      val alpha = ls.optimize(lineSearchFunction, initialAlpha)
      val nextPosition = space.combine(Seq((state.params, 1.0), (direction, alpha)))
//      println("next pos", nextPosition)
      space.prepare(nextPosition)
      // TODO: persist nextPosition
      // TODO: now unpersist direction
      val (nextLoss, nextGradient) = lossFunction.compute(nextPosition)
      val posDelta = space.combine(Seq((nextPosition, 1.0), (state.params, -1.0)))
      val gradDelta = space.combine(Seq((nextGradient, 1.0), (state.gradient, -1.0)))
      // TODO: unpersist state.params
      space.clean(state.params)
      // TODO: persist posDelta and gradDelta
      // TODO: unpersist the shifted out history in `update`
      val newHistory = state.history.update(nextGradient, posDelta, gradDelta)
      VLBFGS.VLBFGSState(state.iter + 1, nextLoss, nextPosition, nextGradient, newHistory)
    }.takeWhile { state =>
      // TODO: convergence checks
      state.iter <= getMaxIter
    }
  }
}

object VLBFGS {

  /**
   * The history required by the VL-BFGS algorithm. This is what differentiates it from L-BFGS.
   * @param m Length of the gradient and position delta histories.
   * @param innerProducts The matrix of basis vector inner products.
   * @param posDeltas History of position deltas between iterations.
   * @param gradDeltas History of gradient deltas between iterations.
   * @tparam T The type the parameter vector.
   */
  case class History[T: ClassTag](
      m: Int,
      innerProducts: Array[Array[Double]],
      posDeltas: Array[T],
      gradDeltas: Array[T])(implicit space: NormedInnerProductSpace[T, Double])
  extends Serializable {

    private val numBasisVectors = 2 * m + 1
    private val k = posDeltas.takeWhile(_ != null).length

    /**
     * Get the appropriate basis vector from the raw index in range [0, 2m]
     *
     * @param idx Basis vector index.
     * @param pds Position delta array.
     * @param gds Gradient delta array.
     * @param gradient The gradient for this history.
     * @return Basis vector b_idx.
     */
    private def getBasisVector(idx: Int, pds: Array[T], gds: Array[T], gradient: T): T = {
      if (idx < m) pds(idx)
      else if (idx < 2 * m) gds(idx - m)
      else if (idx == 2 * m) gradient
      else throw new IndexOutOfBoundsException(s"Basis vector index was invalid: $idx")
    }

    /**
     * Create a copy of the array with each element shifted one to the right. Add a new
     * element at position zero in the new array.
     *   [a, b, null, null] -> [elem, a, b, null]
     *
     * @param elem The head element of the new array
     * @param arr The unshifted array.
     * @return The shifted array with elem.
     */
    private def shiftArr(elem: T, arr: Array[T]): Array[T] = {
      // TODO: here we would want to unpersist the ones that get shifted out
      // TODO: and make sure that the new one is persisted
      val newArr = new Array[T](arr.length)
      for (i <- arr.length - 2 to 0 by (-1)) {
        newArr(i + 1) = arr(i)
      }
      newArr(0) = elem
      newArr
    }

    /**
     * Shift all elements in the original matrix by one row and column, i.e.
     *   (i, j) -> (i + 1, j + 1)
     * The elements in the last row and column are shifted out of the matrix.
     *
     * @param matrix Unshifted matrix.
     * @return Shifted matrix.
     */
    private def shiftMatrix(matrix: Array[Array[Double]]): Array[Array[Double]] = {
      val newMatrix = Array.ofDim[Double](numBasisVectors, numBasisVectors)
      for (i <- numBasisVectors - 2 to 0 by (-1); j <- numBasisVectors - 2 to 0 by (-1)) {
        newMatrix(i + 1)(j + 1) = matrix(i)(j)
      }
      newMatrix
    }

    /**
     * Produce the next history object from the current one, adding in the new position and
     * gradient delta vectors, and updating the inner product matrix.
     *
     * @param gradient The gradient basis vector.
     * @param posDelta The new position delta (x_{i + 1} - x_i)
     * @param gradDelta The new gradient delta (g_{i + 1} - g+i)
     * @return The next History.
     */
    def update(gradient: T, posDelta: T, gradDelta: T): History[T] = {
      // add in the new history, shift out the old, and compute new dot products
      val newPosDeltas = shiftArr(posDelta, posDeltas)
      val newGradDeltas = shiftArr(gradDelta, gradDeltas)
      val shiftedInnerProducts = shiftMatrix(innerProducts)
      // generate a list of inner product indices, and cast as set to avoid duplicate work
      val indices = (0 to 2 * m).flatMap { i =>
        // Ensure that the tuples are ordered uniformly
        List((0, i), (math.min(m, i), math.max(m, i)), (i, 2 * m))
      }.toSet
      indices.par.foreach { case (i, j) =>
        val v1 = getBasisVector(i, newPosDeltas, newGradDeltas, gradient)
        val v2 = getBasisVector(j, newPosDeltas, newGradDeltas, gradient)
        // TODO: we can surely rid ourselves of null checks
        if (v1 != null && v2 != null) {
          val dotProd = space.dot(v1, v2)
          shiftedInnerProducts(i)(j) = dotProd
          shiftedInnerProducts(j)(i) = dotProd
        }
      }
      History(m, shiftedInnerProducts, newPosDeltas, newGradDeltas)
    }

    /**
     * Compute the direction from the basis vectors and their coefficients.
     *   sum_j=1^2m+1^ delta_j * b_j
     * @param gradient The 2m + 1 basis vector.
     * @return Next search direction.
     */
    def computeDirection(gradient: T): T = {
      val deltas = basisCoefficients
      val zipped = (0 until numBasisVectors).map { i =>
        (getBasisVector(i, posDeltas, gradDeltas, gradient), deltas(i))
      }.filter(_._1 != null)
      space.combine(zipped)
    }

    /**
     * Lazily compute the basis coefficients using the algorithm 3 from:
     *   Chen, et al. Large-scale L-BFGS using MapReduce.
     *
     * This is really just the typical L-BFGS recursive update formulated in terms of the
     * decomposition of the search direction using {s_0, ..., s_m-1, y_0, ..., y_m-1, g}
     * as a set of basis vectors. It involves only scalar operations since we have already
     * computed all of the (2 * m + 1)^2^ inner products that are required by the computation.
     */
    lazy val basisCoefficients: Array[Double] = {
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
          val beta = betaNum / betaDenom
          deltas(i) = deltas(i) + (alphas(i) - beta)
        }
        deltas
      }
    }
  }

  case class VLBFGSState[T](iter: Int, loss: Double, params: T, gradient: T, history: History[T])
    extends IterativeOptimizerState[T]
}
