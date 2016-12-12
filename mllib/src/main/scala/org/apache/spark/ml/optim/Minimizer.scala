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
import OptimizerImplicits._



trait Optimizer[T] {

  def optimize(lossFunction: LossFunction[T], initialParameters: T): T

}

class GradientDescent[T: OptimizerImplicits.CanMath] extends Optimizer[T] {

  def optimize(lossFunction: LossFunction[T], initialParameters: T): T = {
    println(initialParameters match {
      case _: Vector => "I'm a vector!"
      case _ => "NOT"
    })
    val stepSize = 0.01
    val theta = initialParameters
    var iter = 0
    while (iter < 100) {
      val (loss, gradient) = lossFunction.compute(theta)
      println(s"Loss: $loss, gradient: $gradient")
      gradient * stepSize
      theta + gradient
      iter += 1
    }
    initialParameters
  }
}
