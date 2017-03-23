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
package org.apache.spark.ml.optim.loss

import scala.reflect.ClassTag
import breeze.linalg.{DenseVector => BDV}
import breeze.optimize.DiffFunction
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.{BLAS, Vector, Vectors}
import org.apache.spark.ml.optim.DifferentiableFunction
import org.apache.spark.ml.optim.Implicits.Aggregable
import org.apache.spark.ml.optim.aggregator.DifferentiableLossAggregator
import org.apache.spark.rdd.RDD

import scala.language.higherKinds

/**
 * This class computes the gradient and loss of a differentiable loss function by mapping a
 * [[DifferentiableLossAggregator]] over an [[RDD]] of [[Instance]]s. The loss function is the
 * sum of the loss computed on a single instance across all points in the RDD. Therefore, the actual
 * analytical form of the loss function is specified by the aggregator, which computes each points
 * contribution to the overall loss.
 *
 * A differentiable regularization component can also be added by providing a
 * [[DifferentiableRegularization]] loss function.
 *
 * @param instances
 * @param getAggregator A function which gets a new loss aggregator in every tree aggregate step.
 * @param regularization An option representing the regularization loss function to apply to the
 *                       coefficients.
 * @param aggregationDepth The aggregation depth of the tree aggregation step.
 * @tparam Agg Specialization of [[DifferentiableLossAggregator]], representing the concrete type
 *             of the aggregator.
 */
class LossFunction[M[_], Agg <: DifferentiableLossAggregator[Instance, Agg]: ClassTag](
   val instances: M[Instance],
   val getAggregator: (Vector => Agg),
   val regularization: Option[DifferentiableRegularization[Array[Double]]],
   val aggregationDepth: Int = 2)(implicit canAgg: Aggregable[M])
  extends DifferentiableFunction[Vector] {

  override def doCompute(coefficients: Vector): (Double, Vector) = {
    val thisAgg = getAggregator(coefficients)
    val seqOp = (agg: Agg, x: Instance) => agg.add(x)
    val combOp = (agg1: Agg, agg2: Agg) => agg1.merge(agg2)
    val newAgg = canAgg.aggregate(instances, thisAgg)(seqOp, combOp)
    val gradient = newAgg.gradient
    val regLoss = regularization.map { regFun =>
      val (regLoss, regGradient) = regFun.compute(coefficients.toArray)
      BLAS.axpy(1.0, Vectors.dense(regGradient), gradient)
      regLoss
    }.getOrElse(0.0)
    // TODO: this would be a problem
//    bcCoefficients.destroy(blocking = false)
    (newAgg.loss + regLoss, gradient)
  }
}
