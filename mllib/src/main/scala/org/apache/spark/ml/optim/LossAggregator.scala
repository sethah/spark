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

import breeze.optimize.DiffFunction
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg._
import org.apache.spark.rdd.RDD
import breeze.linalg.{DenseVector => BDV}
import org.apache.spark.broadcast.Broadcast

import scala.reflect.ClassTag

trait LossAggregator[Datum, Coeff, Agg <: LossAggregator[Datum, Coeff, Agg]] {
  protected var weightSum: Double
  protected var lossSum: Double

  def add(instance: Datum): this.type
  def merge(other: Agg): this.type = {

    if (other.weightSum != 0) {
      weightSum += other.weightSum
      lossSum += other.lossSum
    }
    this
  }
  def loss: Double = {
    require(weightSum > 0.0, s"The effective number of instances should be " +
      s"greater than 0.0, but $weightSum.")
    lossSum / weightSum
  }
}

trait DifferentiableLossAggregator[Datum, Coeff,
Agg <: DifferentiableLossAggregator[Datum, Coeff, Agg]]
  extends LossAggregator[Datum, Coeff, Agg] {
  protected val dim: Int
  protected lazy val gradientSumArray: Array[Double] = Array.ofDim[Double](dim)

//  def addInPlace(instance: Datum): this.type
  override def merge(other: Agg): this.type = {
    require(dim == other.dim, s"Dimensions mismatch when merging with another " +
      s"LeastSquaresAggregator. Expecting $dim but got ${other.dim}.")

    if (other.weightSum != 0) {
      weightSum += other.weightSum
      lossSum += other.lossSum

      var i = 0
      val localThisGradientSumArray = this.gradientSumArray
      val localOtherGradientSumArray = other.gradientSumArray
      while (i < dim) {
        localThisGradientSumArray(i) += localOtherGradientSumArray(i)
        i += 1
      }
    }
    this
  }

  def gradient: Coeff = {
    require(weightSum > 0.0, s"The effective number of instances should be " +
      s"greater than 0.0, but $weightSum.")
    val result = Vectors.dense(gradientSumArray.clone())
    BLAS.scal(1.0 / weightSum, result)
    result
  }

  def create(coeff: Broadcast[Coeff]): Agg
}

//class MyCostFunction[Agg: ClassTag](
//   val instances: RDD[Instance],
//   val agg: Agg,
//   val regFun: (gradient, coeff) => Double)(implicit canAdd: CanAdd[Agg, Instance],
//                       canMerge: CanMerge[Agg]) extends DiffFunction[BDV[Double]] {
//
//  override def calculate(coefficients: BDV[Double]): (Double, BDV[Double]) = {
//    val bcCoefficients = instances.context.broadcast(Vectors.dense(coefficients.data))
//    val thisAgg = agg.create(bcCoefficients)
//    val seqOp = (agg: Agg, x: Instance) => canAdd.add(agg, x)
//    val combOp = (agg1: Agg, agg2: Agg) => canMerge.merge(agg2)
////    val seqOp = (agg: Agg, x: Instance) => agg.add(x)
////    val combOp = (agg1: Agg, agg2: Agg) => agg1.merge(agg2)
//    val newAgg = instances.treeAggregate(thisAgg)(seqOp, combOp)
//    val gradient = newAgg.gradient
//    val regLoss = regFun(gradient, new DenseVector(coefficients.data))
//    (newAgg.loss + regLoss, gradient.asBreeze.toDenseVector)
//
//  }
//
//}

class CostFunction[Agg <: DifferentiableLossAggregator[Instance, Vector, Agg]: ClassTag](
     val instances: RDD[Instance],
     val agg: Agg,
     val regFun: (Vector, Vector) => Double)
  extends DiffFunction[BDV[Double]] {

  override def calculate(coefficients: BDV[Double]): (Double, BDV[Double]) = {
    val bcCoefficients = instances.context.broadcast(Vectors.dense(coefficients.data))
    val thisAgg = agg.create(bcCoefficients)
    val seqOp = (agg: Agg, x: Instance) => agg.add(x)
    val combOp = (agg1: Agg, agg2: Agg) => agg1.merge(agg2)
    val newAgg = instances.treeAggregate(thisAgg)(seqOp, combOp)
    val gradient = newAgg.gradient
    val regLoss = regFun(gradient, new DenseVector(coefficients.data))
    (newAgg.loss + regLoss, gradient.asBreeze.toDenseVector)
  }
}
