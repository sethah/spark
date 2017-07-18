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
package org.apache.spark.ml.optim.aggregator

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.feature.{Instance, OffsetInstance}
import org.apache.spark.ml.linalg.{BLAS, DenseVector, Vector, Vectors}
import org.apache.spark.ml.regression.GeneralizedLinearRegression.FamilyAndLink

private[ml] class GLRAggregator(
                                 bcFeaturesStd: Broadcast[Array[Double]],
                                 familyAndLink: FamilyAndLink,
                                 fitIntercept: Boolean)(bcCoefficients: Broadcast[Vector])
  extends DifferentiableLossAggregator[OffsetInstance, GLRAggregator] {

  protected override val dim: Int = bcCoefficients.value.size
  private val numFeatures = dim - (if (fitIntercept) 1 else 0)
  // make transient so we do not serialize between aggregation stages
  @transient private lazy val featuresStd = bcFeaturesStd.value
  @transient private lazy val effectiveCoefAndOffset = {
    val coefficientsArray = new Array[Double](numFeatures)
    val coefValues = bcCoefficients.value.toArray
    System.arraycopy(coefValues, 0, coefficientsArray, 0, numFeatures)
    var i = 0
    while (i < numFeatures) {
      if (featuresStd(i) != 0.0) {
        coefficientsArray(i) /= featuresStd(i)
      } else {
        coefficientsArray(i) = 0.0
      }
      i += 1
    }
    (Vectors.dense(coefficientsArray), coefValues(coefValues.length - 1))
  }
  // do not use tuple assignment above because it will circumvent the @transient tag
  @transient private lazy val effectiveCoefficientsVector = effectiveCoefAndOffset._1
  @transient private lazy val offset = effectiveCoefAndOffset._2

  /**
   * Add a new training instance to this GLRAggregator, and update the loss and gradient
   * of the objective function.
   *
   * @param instance The instance of data point to be added.
   * @return This LeastSquaresAggregator object.
   */
  def add(instance: OffsetInstance): GLRAggregator = {
    instance match { case OffsetInstance(label, weight, _, features) =>
      require(numFeatures == features.size, s"Dimensions mismatch when adding new sample." +
        s" Expecting $numFeatures but got ${features.size}.")
      require(weight >= 0.0, s"instance weight, $weight has to be >= 0.0")

      if (weight == 0.0) return this

      val eta = BLAS.dot(features, effectiveCoefficientsVector) + offset
      val mu = familyAndLink.fitted(eta)
      val error = mu - label
      val mult = error / (familyAndLink.link.deriv(mu) * familyAndLink.family.variance(mu))
//      println("asfds")
//      println(label)
//      println(mult)

      if (error != 0) {
        val localGradientSumArray = gradientSumArray
        val localFeaturesStd = featuresStd
        features.foreachActive { (index, value) =>
          val fStd = localFeaturesStd(index)
          if (fStd != 0.0 && value != 0.0) {
            localGradientSumArray(index) += weight * mult * value / fStd
          }
        }
        if (fitIntercept) localGradientSumArray(numFeatures) += weight * mult
        val ll = familyAndLink.family.loglikelihood(label, mu, weight)
//        println(s"ll $ll")
        lossSum -= ll
      }
      weightSum += weight
      this
    }
  }
}