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
trait DifferentiableFunction[T] extends (T => Double) { self =>

  def apply(x: T): Double = {
    compute(x)._1
  }

  def gradientAt(x: T): T = {
    compute(x)._2
  }

  def compute(x: T): (Double, T)

  def cached(): CachedDifferentiableFunction[T] = new CachedDifferentiableFunction[T] {
    def doCompute(x: T): (Double, T) = self.compute(x)
  }

}

object DifferentiableFunction {

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

trait CachedDifferentiableFunction[T] extends DifferentiableFunction[T] {

  private var lastData: (T, Double, T) = null

  override def compute(x: T): (Double, T) = {
    var ld = lastData
    if (ld == null || x != ld._1) {
      val newData = doCompute(x: T)
      ld = (x, newData._1, newData._2)
      lastData = ld
    }
    val (_, v, g) = ld
    v -> g
  }

  def doCompute(x: T): (Double, T)

  override def cached(): CachedDifferentiableFunction[T] = this

}

