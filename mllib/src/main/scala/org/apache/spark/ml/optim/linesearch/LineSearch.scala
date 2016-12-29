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
package org.apache.spark.ml.optim.linesearch

import org.apache.spark.ml.optim.DifferentiableFunction


trait LineSearch {

  def optimize(f: DifferentiableFunction[Double], initialGuess: Double): Double

}

class BacktrackingLineSearch(dirNorm: Double) extends LineSearch {

  def optimize(f: DifferentiableFunction[Double], initialGuess: Double): Double = {
    // somewhat of a magic number
    val beta = 0.8
    val phiZero = f(0.0)
    var alpha = 1.0
    while (f(alpha) > (phiZero - dirNorm * alpha / 10.0) && alpha > 1e-10) {
      alpha = alpha * beta
    }
    alpha
  }
}
