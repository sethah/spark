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

import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.{Vectors, BLAS, Vector}
import org.apache.spark.sql.DataFrame

/*
  Notes:
  -We should consider the past design of mllib, breeze's design, scikit, scipy, etc...
  -MLlib design used three key abstractions: Updater, Optimizer, Gradient
    -optimize just takes data and an initial guess and returns the optimized params
    -updater returns new weights based on old weights, step size, reg, and gradient
    -gradient just computes the gradient of a loss function
  -In this current design, we bake optimize and updater in together.
  -any solution should easily extend to different types of regularization
  -breeze uses two key abstractions: a minimizer, and a diff function. We take a similar
    approach here...
  -for mlp, the label is not a double, but instead a vector. We need to accomodate this use
   case
    -right now, we don't impose restrictions on the label. We only say that you must implment
     a function to compute the gradient given the current weights. Why did the mllib version
     do this?
 */

trait LossFunction[T] {

//  def loss(theta: T): Double
//
//  def gradient(theta: T): T

  def compute(theta: T): (Double, T)

}

// TODO: abstract the type
class LeastSquaresCostFun(data: DataFrame, numExamples: Long) extends LossFunction[Vector] {
  import data.sparkSession.implicits._
  val rdd = data.as[Instance].rdd

  def compute(theta: Vector): (Double, Vector) = {
    val n = theta.size
    // aggregate over the data
    val (gradientSum, lossSum) = rdd.treeAggregate((Vectors.zeros(n), 0.0))(
      seqOp = (c, v) => (c, v) match { case ((grad, loss), instance) =>
        val diff = BLAS.dot(instance.features, theta) - instance.label
        val loss = diff * diff / 2.0
        val gradient = instance.features.copy
        BLAS.scal(diff, gradient)
        (gradient, loss)
      },
      combOp = (c1, c2) => (c1, c2) match { case ((grad1, loss1), (grad2, loss2)) =>
        BLAS.axpy(1.0, grad2, grad1)
        (grad1, loss1 + loss2)
      })
    val loss = lossSum / numExamples
    BLAS.scal(1.0 / numExamples, gradientSum)
    (loss, gradientSum)
  }
}