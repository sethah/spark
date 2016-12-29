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
import breeze.linalg.{DenseVector => BDV, NumericOps}
import scala.language.implicitConversions
import org.apache.spark.ml.linalg.{BLAS, DenseVector}
import org.apache.spark.ml.param.Params

/**
 *
 * @tparam T The type of parameters to be optimized.
 * @tparam S The type of loss function.
 */
trait Optimizer[T, S <: (T => Any)] extends Params {

  def optimize(lossFunction: S, initialParameters: T): T

}

trait NormedInnerProductSpace[T, F] {

  def combine(v: Seq[(T, F)]): T

  def dot(x: T, y: T): F

  def norm(x: T): F

  def zero: T

}

object OptimizerImplicits {
  implicit object DenseVectorSpace extends NormedInnerProductSpace[DenseVector, Double] {
    def combine(v: Seq[(DenseVector, Double)]): DenseVector = {
      require(v.nonEmpty)
      val hd = new DenseVector(v.head._1.values.clone())
      BLAS.scal(v.head._2, hd)
      v.tail.foreach { case (v, d) =>
        BLAS.axpy(d, v, hd)
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
