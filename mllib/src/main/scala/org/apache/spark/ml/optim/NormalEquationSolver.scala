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
import breeze.linalg.{DenseVector => BDV}
import breeze.optimize.{CachedDiffFunction, DiffFunction, LBFGS => BreezeLBFGS, OWLQN => BreezeOWLQN}

import org.apache.spark.ml.linalg.{BLAS, Vectors, DenseVector}
import org.apache.spark.ml.param.shared.HasElasticNetParam
import org.apache.spark.mllib.linalg.CholeskyDecomposition

import scala.collection.mutable

abstract class NormalEquationSolver {

  def solve(bBar: Double, bbBar: Double, abBar: DenseVector,
            aaBar: DenseVector, aBar: DenseVector): (Double, DenseVector, Option[DenseVector])

}

class CholeskySolver(val fitIntercept: Boolean) extends NormalEquationSolver {

  def solve(bBar: Double, bbBar: Double, abBar: DenseVector,
            aaBar: DenseVector, aBar: DenseVector): (Double, DenseVector, Option[DenseVector]) = {
    val k = abBar.size
    val x = CholeskyDecomposition.solve(aaBar.values, abBar.values)

    val aaInv = CholeskyDecomposition.inverse(aaBar.values, k)

    val (coefficients, intercept) = if (fitIntercept) {
      (new DenseVector(x.slice(0, x.length - 1)), x.last)
    } else {
      (new DenseVector(x), 0.0)
    }
    (intercept, coefficients, Some(new DenseVector(aaInv)))

  }
}

class QuasiNewtonSolver(
    standardizeFeatures: Boolean,
    standardizeLabel: Boolean,
    regParam: Double,
    elasticNetParam: Double,
    val fitIntercept: Boolean) extends NormalEquationSolver {

  def solve(bBar: Double, bbBar: Double, abBar: DenseVector,
            aaBar: DenseVector, aBar: DenseVector): (Double, DenseVector, Option[DenseVector]) = {
    val numFeatures = aBar.size
    val numFeaturesPlusIntercept = abBar.size
    val initialCoefficientsWithIntercept = Vectors.zeros(numFeaturesPlusIntercept)
    if (fitIntercept) {
      initialCoefficientsWithIntercept.toArray(numFeaturesPlusIntercept - 1) = bBar
    }

    val costFun = new LocalLinearCostFun(bBar, bbBar,
      abBar, aaBar, aBar, fitIntercept, numFeaturesPlusIntercept)
    val effectiveRegParam = regParam
    val effectiveL1RegParam = elasticNetParam * effectiveRegParam
    val effectiveL2RegParam = (1.0 - elasticNetParam) * effectiveRegParam
    val standardizationParam = false
    // TODO: handle standardization
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
    val x = state.x.toArray.clone()
    val (coefficients, intercept) = if (fitIntercept) {
      (new DenseVector(x.slice(0, x.length - 1)), x.last)
    } else {
      (new DenseVector(x), 0.0)
    }
    (intercept, coefficients, None)
  }


  private class LocalLinearCostFun(
      bBar: Double,
      bbBar: Double,
      ab: DenseVector,
      aa: DenseVector,
      abar: DenseVector,
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

}
