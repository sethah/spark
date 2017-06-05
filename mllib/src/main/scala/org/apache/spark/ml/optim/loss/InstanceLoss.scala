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

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.optim.DiffFun
import org.apache.spark.ml.linalg.{BLAS, Vector}
import org.apache.spark.ml.regression.GeneralizedLinearRegression.FamilyAndLink
import org.apache.spark.mllib.util.MLUtils

case class StdBinomialLoss(
    instance: Instance,
    fitIntercept: Boolean,
    featuresStd: Broadcast[Array[Double]]) extends DiffFun[Vector] {

  private val numFeaturesPlusIntercept = instance.features.size + (if (fitIntercept) 1 else 0)
  override def weight: Double = instance.weight

  override def doCompute(x: Vector): (Double, Vector) = {
    throw new NotImplementedError("not implemented!")
  }

  def doComputeInPlace(x: Vector, grad: Vector): Double = {
    val localFeaturesStd = featuresStd.value
    val localCoefficients = x.toArray
    val margin = - {
      var sum = 0.0
      instance.features.foreachActive { (index, value) =>
        if (localFeaturesStd(index) != 0.0 && value != 0.0) {
          sum += localCoefficients(index) * value / localFeaturesStd(index)
        }
      }
      if (fitIntercept) sum += localCoefficients(numFeaturesPlusIntercept - 1)
      sum
    }

    val multiplier = weight * (1.0 / (1.0 + math.exp(margin)) - instance.label)

    val localGradientArray = grad.toArray

    instance.features.foreachActive { (index, value) =>
      if (localFeaturesStd(index) != 0.0 && value != 0.0) {
        localGradientArray(index) += multiplier * value / localFeaturesStd(index)
      }
    }

    if (fitIntercept) {
      localGradientArray(numFeaturesPlusIntercept - 1) += multiplier
    }

    if (instance.label > 0) {
      // The following is equivalent to log(1 + exp(margin)) but more numerically stable.
      weight * MLUtils.log1pExp(margin)
    } else {
      weight * (MLUtils.log1pExp(margin) - margin)
    }
  }

}

case class StdSquaredLoss(
    instance: Instance,
    fitIntercept: Boolean,
    labelStd: Double,
    featuresStd: Broadcast[Array[Double]]) extends DiffFun[Vector] {

  override val weight: Double = instance.weight

  override def doCompute(x: Vector): (Double, Vector) = {
    throw new NotImplementedError("not implemented!")
  }

  def doComputeInPlace(x: Vector, grad: Vector): Double = {
    val localFeaturesStd = featuresStd.value
    val localCoefficients = x.toArray
    val pred = {
      var sum = 0.0
      instance.features.foreachActive { (index, value) =>
        if (localFeaturesStd(index) != 0.0 && value != 0.0) {
          sum += localCoefficients(index) * value
        }
      }
      if (fitIntercept) sum += localCoefficients(x.size - 1)
      sum
    }
    val err = pred - instance.label / labelStd
    if (err != 0) {
      val localGradientSumArray = grad.toArray
      instance.features.foreachActive { (index, value) =>
        val fStd = localFeaturesStd(index)
        if (fStd != 0.0 && value != 0.0) {
          localGradientSumArray(index) += instance.weight * err * value / fStd
        }
      }
    }
    0.5 * instance.weight * err * err
  }
}

case class StdGLMLoss(
                           instance: Instance,
                     familyAndLink: FamilyAndLink,
                           fitIntercept: Boolean,
                           featuresStd: Broadcast[Array[Double]]) extends DiffFun[Vector] {

  private val numFeaturesPlusIntercept = instance.features.size + (if (fitIntercept) 1 else 0)
  override def weight: Double = instance.weight

  override def doCompute(x: Vector): (Double, Vector) = {
    throw new NotImplementedError("not implemented!")
  }

  def doComputeInPlace(x: Vector, grad: Vector): Double = {
    val localFeaturesStd = featuresStd.value
    val localCoefficients = x.toArray
    val eta = {
      var sum = 0.0
      instance.features.foreachActive { (index, value) =>
        if (localFeaturesStd(index) != 0.0 && value != 0.0) {
          sum += localCoefficients(index) * value
        }
      }
      if (fitIntercept) sum += localCoefficients(numFeaturesPlusIntercept - 1)
      sum
    }
    val mu = familyAndLink.fitted(eta)
    val error = mu - instance.label
    val mult = error / (familyAndLink.link.deriv(mu) * familyAndLink.family.variance(mu))

    if (error != 0) {
      val localGradientSumArray = grad.toArray
      instance.features.foreachActive { (index, value) =>
        val fStd = localFeaturesStd(index)
        if (fStd != 0.0 && value != 0.0) {
          localGradientSumArray(index) += weight * mult * value / fStd
        }
      }
      if (fitIntercept) localGradientSumArray(instance.features.size) += weight * mult
    }
    val ll = -familyAndLink.family.logLikelihood(instance.label, mu, weight)
//    println(ll)
    ll
  }
}
