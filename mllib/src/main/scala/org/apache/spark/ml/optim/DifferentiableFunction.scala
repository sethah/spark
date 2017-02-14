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
 * A [[Function1]] that can be differentiated with respect its input.
 *
 * @tparam T The type of the function's domain.
 */
trait DifferentiableFunction[T] extends (T => Double) { self =>

  /**
   * Evaluate the function at a point in the function domain.
   *
   * @param x Point in the parameter space where function is evaluated.
   * @return f(x)
   */
  override def apply(x: T): Double = {
    compute(x)._1
  }

  /**
   * The first derivative of the function evaluated at a point in the function domain.
   *
   * @param x Point in the parameter space where gradient is evaluated.
   * @return d/dx f(x)
   */
  def gradientAt(x: T): T = {
    compute(x)._2
  }

  /**
   * Compute the gradient and the function value at a point in the function domain.
   *
   * @param x Point in the parameter space where function and gradient are evaluated.
   * @return Tuple of (f(x), d/dx f(x))
   */
  def compute(x: T): (Double, T) = doCompute(x)

  protected def doCompute(x: T): (Double, T)

  /** Get a version of this [[DifferentiableFunction]] which caches the most recent computation. */
  def cached(): CachedDifferentiableFunction[T] = {
    new CachedDifferentiableFunction[T](this)
  }
}

object DifferentiableFunction {

  /**
   * Convert a [[DifferentiableFunction]] to a [[breeze.optimize.DiffFunction]]
   * @tparam BT Breeze parameter type.
   * @tparam T Spark parameter type.
   */
  def toBreeze[BT, T](
      diffFun: DifferentiableFunction[T],
      sparkToBreeze: T => BT,
      breezeToSpark: BT => T): DiffFunction[BT] = {
    new DiffFunction[BT] {
      override def valueAt(x: BT): Double = {
        diffFun.apply(breezeToSpark(x))
      }
      override def gradientAt(x: BT): BT = {
        sparkToBreeze(diffFun.gradientAt(breezeToSpark(x)))
      }
      override def calculate(x: BT): (Double, BT) = {
        val (f, grad) = diffFun.compute(breezeToSpark(x))
        (f, sparkToBreeze(grad))
      }
    }
  }
}

/**
 * A [[DifferentiableFunction]] that can caches the most recent result for potential re-use.
 * Minimization algorithms that use line-searches usually compute a zero value, which often has
 * already been computed in the process of finding the descent direction.
 *
 * @tparam T The type of the function's domain.
 */
class CachedDifferentiableFunction[T](private[ml] val diffFun: DifferentiableFunction[T])
  extends DifferentiableFunction[T] {

  private[this] var lastData: Option[(T, Double, T)] = None

  /**
   * Get the function value and gradient, potentially re-using the last result if the point of
   * evaluation has not changed.
   */
  override def compute(x: T): (Double, T) = {
    val (fx, gx) = lastData
      .filter(_._1 == x)
      .map { case (_, lastFx, lastGx) => (lastFx, lastGx) }
      .getOrElse(doCompute(x))
    lastData = Some(x, fx, gx)
    (fx, gx)
  }

  /**
   * Compute function value and gradient at a new point in the function domain.
   */
  protected override def doCompute(x: T): (Double, T) = {
    diffFun.compute(x)
  }

  override def cached(): CachedDifferentiableFunction[T] = this
}

