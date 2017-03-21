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
package org.apache.spark.ml.optim.loss

import org.apache.spark.ml.optim.DifferentiableFunction
import org.apache.spark.ml.linalg._

class EMSOLossFunction[F <: DifferentiableFunction[Vector]](subCost: F, prev: Vector, gamma: Double)
  extends DifferentiableFunction[Vector] with Serializable {

  override def doCompute(x: Vector): (Double, Vector) = {
    val (l, g) = subCost.compute(x)
    val grad = x.copy
    BLAS.axpy(-1.0, prev, grad)
    BLAS.scal(gamma, grad)
    BLAS.axpy(1.0, g, grad)
    val loss = l + 0.5 * gamma * Vectors.sqdist(x, prev)
    (loss, grad)
  }
}
