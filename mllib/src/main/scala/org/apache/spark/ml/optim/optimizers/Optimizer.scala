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

import breeze.linalg.{DenseVector => BDV}
import org.apache.spark.ml.linalg.{BLAS, DenseVector}
import org.apache.spark.ml.optim.DifferentiableFunction
import org.apache.spark.ml.param.Params

import scala.language.implicitConversions

/**
 * Base trait for implementing optimization algorithms in Spark ML.
 *
 * @tparam T The type of parameters to be optimized.
 * @tparam F The type of loss function.
 */
trait Optimizer[T, F <: (T => Any)] extends Params {

  def optimize(lossFunction: F, initialParameters: T): T

}

trait IterativeOptimizer[T, F <: (T => Any), +State <: IterativeOptimizerState[T]]
  extends Optimizer[T, F] {

  def iterations(lossFunction: F, initialParams: T): Iterator[State]

  def optimize(lossFunction: F, initialParameters: T): T = {
    val allIterations = iterations(lossFunction, initialParameters)
    if (allIterations.hasNext) {
      var lastIteration: State = allIterations.next()
      while (allIterations.hasNext) {
        lastIteration = allIterations.next()
      }
      lastIteration.params
    } else {
      initialParameters
    }
  }
}

trait NormedInnerProductSpace[T, F] extends Serializable {

  def combine(v: Seq[(T, F)]): T

  def scale(alpha: F, v: T): T = combine(Seq((v, alpha)))

  def times(alpha: F, v: T, beta: F, u: T): T = {
    combine(Seq((v, alpha), (u, beta)))
  }

  def dot(x: T, y: T): F

  def norm(x: T): F

  def zero: T

  def clean(x: T): Unit = {}

}

object OptimizerImplicits {
  implicit object DenseVectorSpace extends NormedInnerProductSpace[DenseVector, Double] {
    def combine(v: Seq[(DenseVector, Double)]): DenseVector = {
      require(v.nonEmpty)
      val hd = new DenseVector(v.head._1.values.clone())
      BLAS.scal(v.head._2, hd)
      v.tail.foreach { case (vec, d) =>
        BLAS.axpy(d, vec, hd)
      }
      hd
    }

    def dot(x: DenseVector, y: DenseVector): Double = {
      BLAS.dot(x, y)
    }

    def norm(x: DenseVector): Double = {
      math.sqrt(dot(x, x))
    }

    def zero: DenseVector = new DenseVector(Array.empty[Double])
  }

}
