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

import org.apache.spark.annotation.Since
import org.apache.spark.ml.optim.DifferentiableFunction
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.optim.Implicits.HasSubProblems
import org.apache.spark.ml.optim.loss.EMSOLossFunction
import org.apache.spark.ml.param.shared.{HasMaxIter, HasTol}
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.rdd.RDD

import scala.collection.mutable
import scala.reflect.ClassTag

trait EMSOParams extends Params with HasMaxIter with HasTol

/**
 * The partition minimizer needs to minimize a cost function over each partition's data. The cost
 * function for that minimization problem needs to be an `EMSOLoss` which is just a wrapper on the
 * original cost function (e.g. logistic cost) that also regularizes from the current solution.
 *
 * @param uid
 * @param partitionMinimizer
 * @param gamma
 * @tparam F
 *           TODO: use a function State => PartitionMinimizer instead of a single partition min
 */
class EMSO[F <: DifferentiableFunction[Vector]](
    override val uid: String,
    val partitionMinimizer: IterativeMinimizer[Vector,
      EMSOLossFunction[DifferentiableFunction[Vector]], IterativeMinimizerState[Vector]],
    val gamma: Int => Double)(implicit subProblems: HasSubProblems[RDD, F])
  extends IterativeMinimizer[Vector, F, IterativeMinimizerState[Vector]] with EMSOParams {

  type State = EMSO.EMSOState

  def this(
      partitionMinimizer: IterativeMinimizer[Vector,
        EMSOLossFunction[DifferentiableFunction[Vector]], IterativeMinimizerState[Vector]],
      gamma: Int => Double)(implicit sp: HasSubProblems[RDD, F]) = {
    this(Identifiable.randomUID("emso"), partitionMinimizer, gamma)(sp)
  }

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

  def initialState(initialParameters: Vector): State = {
    val size = initialParameters.size
    EMSO.EMSOState(0, 0.0, initialParameters, new DenseVector(Array.fill(size)(1.0)))
  }

  override def iterations(lossFunction: F, initialParameters: Vector): Iterator[State] = {
    val numFeatures = initialParameters.size
    var continue = true
    Iterator.iterate(initialState(initialParameters)) { state =>
      val oldParams = state.params
      println("params", oldParams)
      println("grad", state.gradient)
      println("loss", state.loss)
      val solutions = subProblems.nextSubproblems(lossFunction).mapPartitionsWithIndex((i, p) => {
        p.map { subProb =>
          val emsoSubProb = new EMSOLossFunction(subProb, oldParams, gamma(state.iter))

          val optIterations = partitionMinimizer.iterations(emsoSubProb, initialParameters)

          var lastIter: IterativeMinimizerState[Vector] = null
          val arrayBuilder = mutable.ArrayBuilder.make[Double]
          while (optIterations.hasNext) {
            lastIter = optIterations.next()
            arrayBuilder += lastIter.loss
          }
//          println(s"Converged in ${lastIter.iter} iterations")
          val tmp = i
          lastIter
        }
      })
      // (count, loss, params, gradient)
      val initialValues = (0L, 0.0, Vectors.sparse(numFeatures, Array(), Array()),
        Vectors.sparse(numFeatures, Array(), Array()))
      val _next = solutions.treeAggregate(initialValues)(
        seqOp = (acc, state) => {
          val paramsRes = acc._3.toDense
          BLAS.axpy(1.0, state.params, paramsRes)
          val gradRes = acc._4.toDense
          BLAS.axpy(1.0, state.gradient, gradRes)
          (acc._1 + 1L, acc._2 + state.loss, paramsRes, gradRes)
        },
        combOp = (acc1, acc2) => {
          val gradRes = acc1._4.toDense
          BLAS.axpy(1.0, acc2._4, gradRes)
          val paramsRes = acc1._3.toDense
          BLAS.axpy(1.0, acc2._3, paramsRes)
          (acc1._1 + acc2._1, acc1._2 + acc2._2, paramsRes, gradRes)
        })

      // TODO: weighted average?
      val _nextModel = _next._3
      BLAS.scal(1.0 / _next._1, _nextModel)
      val _nextGradient = _next._4
      BLAS.scal(1.0 / _next._1, _nextGradient)

      EMSO.EMSOState(state.iter + 1, _next._2 / _next._1, _nextModel, _nextGradient)
    }.takeWhile { state =>
      val norm = Vectors.norm(state.gradient, 2)
      // this hack is required because we need the final iteration, but takewhile only gives you
      // the iteration before the conditions are violated
      val tmp = continue
      if (!(state.iter < getMaxIter && norm > getTol)) continue = false
      tmp
    }
  }

  override def copy(extra: ParamMap): EMSO[F] = defaultCopy(extra)

}

object EMSO {
  case class EMSOState(iter: Int, loss: Double, params: Vector, gradient: Vector)
    extends IterativeMinimizerState[Vector] with DifferentiableMinimizerState[Vector]
}
