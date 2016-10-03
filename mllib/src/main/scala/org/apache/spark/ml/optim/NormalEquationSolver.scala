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
import org.apache.spark.mllib.linalg.CholeskyDecomposition

import scala.collection.mutable

private[ml] class NormalEquationSolution(
    val fitIntercept: Boolean,
    private val _coefficients: DenseVector,
    val aaInv: Option[DenseVector],
    val objectiveHistory: Option[Array[Double]]) {

  lazy val (coefficients, intercept) = {
    val x = _coefficients.values
    if (fitIntercept) {
      (new DenseVector(x.slice(0, x.length - 1)), x.last)
    } else {
      (new DenseVector(x), 0.0)
    }
  }
}

/**
 * Interface for classes that solve the normal equations.
 */
private[ml] trait NormalEquationSolver {

  def solve(
      bBar: Double,
      bbBar: Double,
      abBar: DenseVector,
      aaBar: DenseVector,
      aBar: DenseVector): NormalEquationSolution
}

private[ml] object NormalEquationSolver {
  val Cholesky: String = "cholesky"
  val QuasiNewton: String = "quasi-newton"
  val Auto: String = "auto"
}

private[ml] class CholeskySolver(val fitIntercept: Boolean) extends NormalEquationSolver {

  def solve(
      bBar: Double,
      bbBar: Double,
      abBar: DenseVector,
      aaBar: DenseVector,
      aBar: DenseVector): NormalEquationSolution = {
    val k = abBar.size
    val x = CholeskyDecomposition.solve(aaBar.values, abBar.values)
    val aaInv = CholeskyDecomposition.inverse(aaBar.values, k)

    new NormalEquationSolution(fitIntercept, new DenseVector(x), Some(new DenseVector(aaInv)), None)
  }
}

private[ml] class QuasiNewtonSolver(
    standardizeFeatures: Boolean,
    standardizeLabel: Boolean,
    val fitIntercept: Boolean,
    maxIter: Int,
    tol: Double,
    l1RegFunc: Option[(Int) => Double]) extends NormalEquationSolver {

  def solve(
      bBar: Double,
      bbBar: Double,
      abBar: DenseVector,
      aaBar: DenseVector,
      aBar: DenseVector): NormalEquationSolution = {
    val numFeatures = aBar.size
    val numFeaturesPlusIntercept = if (fitIntercept) numFeatures + 1 else numFeatures
    val initialCoefficientsWithIntercept = Vectors.zeros(numFeaturesPlusIntercept)
    if (fitIntercept) {
      // TODO: this correct?
      initialCoefficientsWithIntercept.toArray(numFeaturesPlusIntercept - 1) = bBar
    }

    val costFun = new NormalEquationCostFun(bBar, bbBar, abBar, aaBar, aBar, fitIntercept,
      numFeatures)
    val optimizer = l1RegFunc.map { func =>
      new BreezeOWLQN[Int, BDV[Double]](maxIter, 10, func, tol)
    }.getOrElse(new BreezeLBFGS[BDV[Double]](maxIter, 10, tol))
    val states = optimizer.iterations(new CachedDiffFunction(costFun),
      initialCoefficientsWithIntercept.asBreeze.toDenseVector)

    val arrayBuilder = mutable.ArrayBuilder.make[Double]
    var state: optimizer.State = null
    while (states.hasNext) {
      state = states.next()
      arrayBuilder += state.adjustedValue
    }
    val x = state.x.toArray.clone()
    new NormalEquationSolution(fitIntercept, new DenseVector(x), None, Some(arrayBuilder.result()))
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
        // TODO: use while loop here?
        val coefArray = coef.toArray
        val interceptIndex = numFeaturesPlusIntercept - 1
        val coefWithoutIntercept = coefArray.init
        coefArray(interceptIndex) = bBar - BLAS.dot(Vectors.dense(coefWithoutIntercept), aBar)
      }
      val xxb = Vectors.zeros(numFeaturesPlusIntercept).toDense
      BLAS.dspmv(numFeaturesPlusIntercept, 1.0, aa, coef, xxb)
      // loss = 1/2 (Y^T W Y - 2 beta^T X^T W Y + beta^T X^T W X beta)
      val loss = 0.5 * bbBar - BLAS.dot(ab, coef) + 0.5 * BLAS.dot(coef, xxb)
      // -gradient = X^T W X beta - X^T W Y
      BLAS.axpy(-1.0, ab, xxb)
      (loss, xxb.asBreeze.toDenseVector)
    }
  }

}
