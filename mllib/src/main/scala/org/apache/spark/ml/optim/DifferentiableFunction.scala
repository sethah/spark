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

import breeze.optimize.DiffFunction

/**
 *
 * @tparam T The type of the function's domain.
 */
trait DifferentiableFunction[T] extends (T => Double) {

  def apply(x: T): Double

  def gradientAt(x: T): T

  def compute(x: T): (Double, T)

}

object DifferentiableFunction {
  def toBreeze[T](f: DifferentiableFunction[T]): DiffFunction[T] = {
    new DiffFunction[T] {
      override def valueAt(x: T): Double = f(x)
      override def gradientAt(x: T): T = f.gradientAt(x)
      override def calculate(x: T): (Double, T) = f.compute(x)
    }
  }
  def fromBreeze[T](f: DiffFunction[T]): DifferentiableFunction[T] = {
    new DifferentiableFunction[T] {
      override def apply(x: T): Double = f.valueAt(x)
      override def gradientAt(x: T): T = f.gradientAt(x)
      override def compute(x: T): (Double, T) = f.calculate(x)
    }
  }
}
