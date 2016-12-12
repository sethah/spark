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

import org.apache.spark.ml.linalg.{BLAS, Vector}

object OptimizerImplicits {
  trait CanMath[T] {
    def addInPlace(x: T, y: T): Unit
    def scalarMultiply(x: T, alpha: Double): Unit
  }

  object CanMath {
    def apply[T: CanMath]: CanMath[T] = implicitly
  }

  implicit object CanMathVector extends CanMath[Vector] {
    def addInPlace(x: Vector, y: Vector): Unit = {
      BLAS.axpy(1.0, y, x)
    }

//    def subtractInPlace(x: Vector, y: Vector): Unit = {
//      BLAS.axpy()
//    }

    def scalarMultiply(x: Vector, alpha: Double): Unit = {
      BLAS.scal(alpha, x)
    }
  }

  implicit class CanMathOps[T: CanMath](value: T) {
    def +(other: T): Unit = CanMath[T].addInPlace(value, other)
    def *(alpha: Double): Unit = CanMath[T].scalarMultiply(value, alpha)
  }
}
