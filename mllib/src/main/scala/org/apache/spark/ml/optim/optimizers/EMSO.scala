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
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.optim.Implicits.HasSubProblems
import org.apache.spark.ml.optim.loss.EMSOLossFunction
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

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
    val partitionMinimizer: Minimizer[Vector, EMSOLossFunction[DifferentiableFunction[Vector]]],
    val gamma: Double)(implicit subProblems: HasSubProblems[RDD, F])
  extends IterativeMinimizer[Vector, F, IterativeMinimizerState[Vector]] {

  type State = EMSO.EMSOState

  def this(
      partitionMinimizer: Minimizer[Vector, EMSOLossFunction[DifferentiableFunction[Vector]]],
      gamma: Double)(implicit sp: HasSubProblems[RDD, F]) = {
    this(Identifiable.randomUID("emso"), partitionMinimizer, gamma)(sp)
  }


  def initialState(initialParameters: Vector): State = {
    EMSO.EMSOState(0, 0.0, initialParameters)
  }

  override def iterations(lossFunction: F, initialParameters: Vector): Iterator[State] = {
    val numFeatures = initialParameters.size
    Iterator.iterate(initialState(initialParameters)) { state =>
      val oldParams = state.params
      val solutions = subProblems.nextSubproblems(lossFunction).map { subProb =>
        val emsoSubProb = new EMSOLossFunction(subProb, oldParams, gamma)
        partitionMinimizer.minimize(emsoSubProb, initialParameters)
      }
      val _next = solutions.treeAggregate((0L, Vectors.sparse(numFeatures, Array(), Array())))(
        seqOp = (acc, v) => {
          val res = acc._2.toDense
          BLAS.axpy(1.0, v, res)
          (acc._1 + 1L, res)
        },
        combOp = (acc1, acc2) => {
          val res = acc1._2.toDense
          BLAS.axpy(1.0, acc1._2, res)
          (acc1._1 + acc2._1, res)
        })

      val _nextModel = _next._2
      BLAS.scal(1.0 / _next._1, _nextModel)

      // TODO: loss
      EMSO.EMSOState(state.iter + 1, 0.0, _nextModel)
    }.take(10)
  }

  override def copy(extra: ParamMap): EMSO[F] = defaultCopy(extra)

}

object EMSO {
  case class EMSOState(iter: Int, loss: Double, params: Vector)
    extends IterativeMinimizerState[Vector]
}
