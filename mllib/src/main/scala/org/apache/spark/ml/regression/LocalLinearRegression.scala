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
package org.apache.spark.ml.regression

import breeze.linalg.{DenseVector => BDV}
import breeze.optimize.{CachedDiffFunction, DiffFunction, LBFGS => BreezeLBFGS, OWLQN => BreezeOWLQN}
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.{DenseVector, BLAS, Vector, Vectors}
import org.apache.spark.ml.optim.WeightedLeastSquares.Aggregator
import org.apache.spark.rdd.RDD

import scala.collection.mutable

class LocalLinearRegression {

  def fit(instances: RDD[Instance]): Array[Double] = {
    val fitIntercept = true
    val summary = instances.treeAggregate(new Aggregator)(_.add(_), _.merge(_))
    summary.validate()
    val numFeatures = summary.abBar.size
    val numFeaturesPlusIntercept = numFeatures + (if (fitIntercept) 1 else 0)
    val numInstances = summary.count
    val k = if (fitIntercept) summary.k + 1 else summary.k
    val triK = summary.triK
    val wSum = summary.wSum
    val bBar = summary.bBar
    val bStd = summary.bStd
    val aBar = summary.aBar
    val aVar = summary.aVar
    val abBar = summary.abBar
    val aaBar = summary.aaBar
    val aaValues = aaBar.values
    println("bstd", bStd)

    val aBarStd = aBar.values.clone()
    aBarStd.indices.foreach { i =>
      aBarStd(i) /= math.sqrt(aVar(i))
    }
    val abStd = abBar.toArray.indices.map { i =>
      abBar(i) / (math.sqrt(aVar(i)) * bStd)
    }.toArray
    val aaBarStd = aaBar.values.clone()

    var j = 0
    var kk = 0
    while (j < numFeatures) {
      var i = 0
      while (i <= j) {
        aaBarStd(kk) /= math.sqrt(aVar(i)) * math.sqrt(aVar(j))
        kk += 1
        i += 1
      }
      j += 1
    }
    println(new DenseVector(aaBarStd))
    println(bBar)
    println(abBar)
    println(Vectors.dense(abStd))
    println(Vectors.dense(aBarStd))

    val aa = if (fitIntercept) {
      Array.concat(aaBarStd, aBarStd, Array(1.0))
    } else {
      aaBarStd
    }
    val ab = if (fitIntercept) {
      Array.concat(abStd, Array(bBar / bStd))
    } else {
      abStd
    }
    val initialCoefficientsWithIntercept = Vectors.zeros(numFeaturesPlusIntercept)
    if (fitIntercept) {
      initialCoefficientsWithIntercept.toArray(numFeaturesPlusIntercept - 1) = bBar
    }

    val costFun = new LocalLinearCostFun(summary.bBar / bStd, summary.bbBar / (bStd * bStd),
      new DenseVector(ab), new DenseVector(aa), new DenseVector(aBarStd), bStd, fitIntercept,
      numFeaturesPlusIntercept)
    val effectiveRegParam = 0.0
    val effectiveL1RegParam = 1.0 * effectiveRegParam
    val effectiveL2RegParam = (1.0 - 1.0) * effectiveRegParam
    val standardizationParam = false
    val optimizer = if (effectiveRegParam != 0.0) {
      def effectiveL1RegFun = (index: Int) => {
        val isIntercept = fitIntercept && index == numFeatures
        if (isIntercept) {
          0.0
        } else effectiveL1RegParam
      }
      new BreezeOWLQN[Int, BDV[Double]](10, 10, effectiveL1RegFun, 1e-6)
    } else {
      new BreezeLBFGS[BDV[Double]](10, 10, 1e-6)
    }
    val states = optimizer.iterations(new CachedDiffFunction(costFun),
      initialCoefficientsWithIntercept.asBreeze.toDenseVector)

    val arrayBuilder = mutable.ArrayBuilder.make[Double]
    var state: optimizer.State = null
    while (states.hasNext) {
      state = states.next()
      arrayBuilder += state.adjustedValue
    }
    val rawCoefficients = state.x.toArray.clone()
//    println(rawCoefficients.mkString(","))
    var i = 0
    val len = rawCoefficients.length - 1
    while (i < len) {
      rawCoefficients(i) *= { if (aVar(i) != 0.0) bStd / math.sqrt(aVar(i)) else 0.0 }
      i += 1
    }
    rawCoefficients
  }

}

private class LocalLinearCostFun(
    bBar: Double,
    bbBar: Double,
    ab: DenseVector,
    aa: DenseVector,
    abar: DenseVector,
    bstd: Double,
    fitIntercept: Boolean,
    numFeatures: Int) extends DiffFunction[BDV[Double]] {

  override def calculate(coefficients: BDV[Double]): (Double, BDV[Double]) = {
    val sparkCoefficients = Vectors.fromBreeze(coefficients).toDense
    val onlyCoef = new DenseVector(sparkCoefficients.toArray.init)
    val intercept = bBar - BLAS.dot(onlyCoef, abar)
    if (fitIntercept) {
      sparkCoefficients.toArray(numFeatures - 1) = intercept
    }
    // loss = Y^T W Y - 2 beta^T X^T W Y + beta^T X^T W X beta
//    println(bbBar, ab, aa, sparkCoefficients)
    val loss1 = bbBar
    val loss2 = 2.0 * BLAS.dot(ab, sparkCoefficients.copy)
//    println(loss1, loss2, coefficients)
    val xxb = Vectors.zeros(numFeatures).toDense
//    println("aabar", aa)
    BLAS.dspmv("U", numFeatures, 1.0, aa, sparkCoefficients, 1.0, xxb)
//    println("xxb", xxb)
    val loss3 = BLAS.dot(sparkCoefficients, xxb)
    val loss = 0.5 * (loss1 - loss2 + loss3)
    BLAS.axpy(-1.0, ab, xxb)
//    println("coef", loss, loss1, loss2, loss3, sparkCoefficients, xxb)
    (loss, xxb.asBreeze.toDenseVector)
  }
}
