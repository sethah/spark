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

private[ml] abstract class NormalEquationSolver {

  def solve(
      bBar: Double,
      bbBar: Double,
      abBar: DenseVector,
      aaBar: DenseVector,
      aBar: DenseVector,
      aVar: DenseVector): (Double, DenseVector, Option[DenseVector])

}

private[ml] class CholeskySolver(val fitIntercept: Boolean) extends NormalEquationSolver {

  def solve(
      bBar: Double,
      bbBar: Double,
      abBar: DenseVector,
      aaBar: DenseVector,
      aBar: DenseVector,
      aVar: DenseVector): (Double, DenseVector, Option[DenseVector]) = {
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

private[ml] class QuasiNewtonSolver(
    standardizeFeatures: Boolean,
    standardizeLabel: Boolean,
    regParam: Double,
    elasticNetParam: Double,
    val fitIntercept: Boolean) extends NormalEquationSolver {

  def solve(
      bBar: Double,
      bbBar: Double,
      abBar: DenseVector,
      aaBar: DenseVector,
      aBar: DenseVector,
      aVar: DenseVector): (Double, DenseVector, Option[DenseVector]) = {
    val numFeatures = aBar.size
    val numFeaturesPlusIntercept = abBar.size
    val initialCoefficientsWithIntercept = Vectors.zeros(numFeaturesPlusIntercept)
    if (fitIntercept) {
      // TODO: this correct?
      initialCoefficientsWithIntercept.toArray(numFeaturesPlusIntercept - 1) = bBar
    }

    val costFun = new NormalEquationCostFun(bBar, bbBar,
      abBar, aaBar, aBar, fitIntercept, numFeatures)
    val effectiveRegParam = regParam
    val effectiveL1RegParam = elasticNetParam * effectiveRegParam
    val optimizer = if (effectiveL1RegParam != 0.0) {
      // TODO: pass this in as argument
      def effectiveL1RegFun = (index: Int) => {
        val isIntercept = fitIntercept && index == numFeatures
        if (isIntercept) {
          0.0
        } else {
          if (standardizeFeatures) {
            effectiveL1RegParam
          } else {
            effectiveL1RegParam / math.sqrt(aVar(index))
          }
        }
      }
      println("OWLQN")
      new BreezeOWLQN[Int, BDV[Double]](100, 10, effectiveL1RegFun, 1e-6)
    } else {
      println("LBFGS")
      new BreezeLBFGS[BDV[Double]](100, 10, 1e-6)
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


  private class NormalEquationCostFun(
      bBar: Double,
      bbBar: Double,
      ab: DenseVector,
      aa: DenseVector,
      aBar: DenseVector,
      fitIntercept: Boolean,
      numFeatures: Int) extends DiffFunction[BDV[Double]] {

    private val numFeaturesPlusIntercept = if (fitIntercept) numFeatures + 1 else numFeatures

    override def calculate(coefficients: BDV[Double]): (Double, BDV[Double]) = {
      val coef = Vectors.fromBreeze(coefficients).toDense
      if (fitIntercept) {
        // TODO: check this
        val coefArray = coef.toArray
        val interceptIndex = numFeaturesPlusIntercept - 1
        val coefWithoutIntercept = coefArray.init
        coefArray(interceptIndex) = bBar - BLAS.dot(Vectors.dense(coefWithoutIntercept), aBar)
      }
      val xxb = Vectors.zeros(numFeaturesPlusIntercept).toDense
      BLAS.dspmv("U", numFeaturesPlusIntercept, 1.0, aa, coef, 1.0, xxb)
      // loss = Y^T W Y - 2 beta^T X^T W Y + beta^T X^T W X beta
      val loss = 0.5 * bbBar - BLAS.dot(ab, coef) + 0.5 * BLAS.dot(coef, xxb)
      // -gradient = X^T W X beta - X^T W Y
      BLAS.axpy(-1.0, ab, xxb)
      (loss, xxb.asBreeze.toDenseVector)
    }
  }

}
