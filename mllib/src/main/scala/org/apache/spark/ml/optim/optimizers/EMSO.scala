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
 */
class EMSO[F <: DifferentiableFunction[Vector]](
    override val uid: String,
    val partitionMinimizer: IterativeMinimizer[Vector,
      EMSOLossFunction[DifferentiableFunction[Vector]], IterativeMinimizerState[Vector]],
    val gamma: Double)(implicit subProblems: HasSubProblems[RDD, F])
  extends IterativeMinimizer[Vector, F, IterativeMinimizerState[Vector]] with EMSOParams {

  type State = EMSO.EMSOState

  def this(
      partitionMinimizer: IterativeMinimizer[Vector,
        EMSOLossFunction[DifferentiableFunction[Vector]], IterativeMinimizerState[Vector]],
      gamma: Double)(implicit sp: HasSubProblems[RDD, F]) = {
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


  def initialState(initialParameters: Vector): State = {
    EMSO.EMSOState(0, 0.0, initialParameters)
  }

  override def iterations(lossFunction: F, initialParameters: Vector): Iterator[State] = {
    val numFeatures = initialParameters.size
    Iterator.iterate(initialState(initialParameters)) { state =>
      val oldParams = state.params
      println(state.loss)
      val solutions = subProblems.nextSubproblems(lossFunction).map { subProb =>
        val emsoSubProb = new EMSOLossFunction(subProb, oldParams, gamma)
        // idea: make this an iterative minimizer and then get the iterations. Take all until last,
        // but save the first one because that should be the loss/gradient

        val optIterations = partitionMinimizer.iterations(emsoSubProb, initialParameters)

        var lastIter: IterativeMinimizerState[Vector] = null
        val arrayBuilder = mutable.ArrayBuilder.make[Double]
        while (optIterations.hasNext) {
          lastIter = optIterations.next()
          arrayBuilder += lastIter.loss
        }
//        partitionMinimizer.minimize(emsoSubProb, initialParameters)
        lastIter
      }
      // (count, loss, params)
      val initialValues = (0L, 0.0, Vectors.sparse(numFeatures, Array(), Array()))
      val _next = solutions.treeAggregate(initialValues)(
        seqOp = (acc, state) => {
          val paramsRes = acc._3.toDense
          BLAS.axpy(1.0, state.params, paramsRes)
//          val gradRes = acc._3
//          BLAS.axpy(1.0, state.gradient, paramsRes)
          (acc._1 + 1L, acc._2 + state.loss, paramsRes)
        },
        combOp = (acc1, acc2) => {
//          val gradRes = acc1._3.toDense
//          BLAS.axpy(1.0, acc1._3, gradRes)
          val paramsRes = acc1._3.toDense
          BLAS.axpy(1.0, acc1._3, paramsRes)
          (acc1._1 + acc2._1, acc1._2 + acc2._2, paramsRes)
        })

      val _nextModel = _next._3
      BLAS.scal(1.0 / _next._1, _nextModel)

      // TODO: loss
      EMSO.EMSOState(state.iter + 1, _next._2, _nextModel)
    }.takeWhile { state =>
      state.iter < getMaxIter //&& BLAS.dot(state.)
    }
  }

  override def copy(extra: ParamMap): EMSO[F] = defaultCopy(extra)

}

object EMSO {
  case class EMSOState(iter: Int, loss: Double, params: Vector)
    extends IterativeMinimizerState[Vector]
}
