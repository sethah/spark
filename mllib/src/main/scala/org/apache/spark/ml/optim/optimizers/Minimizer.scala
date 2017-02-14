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

import org.apache.spark.ml.param.Params

/**
 * Base trait for implementing optimization algorithms in Spark ML.
 *
 * @tparam T The type of parameters to be optimized.
 * @tparam F The type of loss function.
 */
trait Minimizer[T, F <: (T => Double)] extends Params {

  /**
   * Minimize a loss function over the parameter space.
   *
   * @param lossFunction Real-valued loss function to minimize.
   * @param initialParameters Initial point in the parameter space.
   */
  def minimize(lossFunction: F, initialParameters: T): T

}

/**
 * A minimizer that iteratively minimizes a set of parameters.
 *
 * @tparam State Type that holds information about the state of the minimization at each iteration.
 */
trait IterativeMinimizer[T, F <: (T => Double), +State <: IterativeMinimizerState[T]]
  extends Minimizer[T, F] {

  /**
   * Produces an iterator of states which hold information about the progress of the minimization.
   *
   * @param lossFunction Real-valued loss function to minimize.
   * @param initialParameters Initial point in the parameter space.
   */
  def iterations(lossFunction: F, initialParameters: T): Iterator[State]

  override def minimize(lossFunction: F, initialParameters: T): T = {
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

