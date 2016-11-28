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


import org.apache.spark.ml.linalg.{Vectors, DenseVector, Vector, BLAS}

object OptimizerImplicits {
  trait CanMath[T] {
    def addInPlace(x: T, y: T): Unit
    def scalarMultiply(x: T, alpha: Double): Unit
  }

  implicit object CanMathVector extends CanMath[Vector] {
    def addInPlace(x: Vector, y: Vector): Unit = {
      BLAS.axpy(1.0, x, y)
    }

    def scalarMultiply(x: Vector, alpha: Double): Unit = {
      BLAS.scal(alpha, x)
    }

  }

//  implicit class CanAddOps[T](value: T)(implicit ops: CanAdd[T]) {
//    def +(other: T): T = ops.+(value, other)
//    def -(other: T): T = ops.-(value, other)
//  }
}


trait Optimizer[T] {

  def optimize(lossFunction: LossFunction[T], initialParameters: T): T

}

class GradientDescent[T: OptimizerImplicits.CanMath] extends Optimizer[T] {

  def optimize(lossFunction: LossFunction[T], initialParameters: T): T = {
    val ops = implicitly[OptimizerImplicits.CanMath[T]]
    val stepSize = 0.1
    val theta = initialParameters
    var iter = 0
    while (iter < 10) {
      val (loss, gradient) = lossFunction.compute(theta)
      ops.scalarMultiply(gradient, stepSize)
      ops.addInPlace(gradient, theta)
      iter += 1
    }
    initialParameters
  }
}
