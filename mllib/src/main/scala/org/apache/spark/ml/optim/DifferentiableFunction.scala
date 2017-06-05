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
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.{BLAS, Vector, Vectors}
import org.apache.spark.ml.optim.aggregator.{DiffFunAggregator, DifferentiableLossAggregator}
import org.apache.spark.ml.optim.loss.{DifferentiableRegularization, EnumeratedRegularization}
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.optim.Implicits._

import scala.reflect.ClassTag

trait DiffFun[T] extends (T => Double) with Serializable {

  private var lastData: Option[(T, Double, T)] = None

  def loss(x: T): Double = compute(x)._1

  def grad(x: T): T = compute(x)._2

  final def compute(x: T): (Double, T) = {
    if (cache) {
      val (fx, gx) = lastData
        .filter(_._1 == x)
        .map { case (_, lastFx, lastGx) => (lastFx, lastGx) }
        .getOrElse(doCompute(x))
      lastData = Some((x, fx, gx))
      (fx, gx)
    } else {
      doCompute(x)
    }
  }

  final def computeInPlace(x: T, grad: T): Double = {
    if (cache) {
      val (gx, fx) = lastData
        .filter(_._1 == x)
        .map { case (_, lastFx, lastGx) => (lastGx, lastFx) }
        .getOrElse {
          val loss = doComputeInPlace(x, grad)
          (grad, loss)
        }
      lastData = Some((x, fx, gx))
      fx
    } else {
      doComputeInPlace(x, grad)
    }
  }

  /** Compute the gradient and the loss at a specific point in parameter space. */
  def doCompute(x: T): (Double, T)

  /** Same as `doCompute` but update the gradient in place. */
  def doComputeInPlace(x: T, grad: T): Double

  /** Importance weight of this loss function. */
  def weight: Double = 1.0

  override def apply(x: T): Double = loss(x)

  /**
   * Whether or not to cache the gradient and loss of the last call to `compute`. This is helpful
   * when doing things like line-searches which often evaluate the loss at the previously
   * computed point.
   */
  def cache: Boolean = false

}

object Implicits {

  trait Aggregable[M[_]] extends Serializable {

    def aggregate[A: ClassTag, B: ClassTag](ma: M[A], b: B)(add: (B, A) => B,
                                                            combine: (B, B) => B): B

  }

  implicit object RDDCanAggregate extends Aggregable[RDD] {
    override def aggregate[A: ClassTag, B: ClassTag](
        ma: RDD[A], b: B)(add: (B, A) => B, combine: (B, B) => B): B = {
      ma.treeAggregate(b)(add, combine)
    }
  }

  implicit object IterableCanAggregate extends Aggregable[Iterable] {
    override def aggregate[A: ClassTag, B: ClassTag](fa: Iterable[A], b: B)(
        add: (B, A) => B, combine: (B, B) => B): B = {
      fa.foldLeft(b)(add)
    }
  }

  implicit object IteratorCanAggregate extends Aggregable[Iterator] {
    override def aggregate[A: ClassTag, B: ClassTag](fa: Iterator[A], b: B)(
        add: (B, A) => B, combine: (B, B) => B): B = {
      fa.foldLeft(b)(add)
    }
  }
}

trait HasRegularization[T] extends Serializable {

  def regularizers: List[T => Double]

}

class SeparableDiffFun[M[_]: Aggregable](
    val losses: M[DiffFun[Vector]],
    val getAggregator: (Vector => DiffFunAggregator),
    override val regularizers: List[EnumeratedRegularization[Vector, _]],
    override val cache: Boolean = false)
  extends DiffFun[Vector] with HasRegularization[Vector] {

  self =>

  private[spark] var numCalls: Int = 0


  private type Agg = DiffFunAggregator

  override def doCompute(x: Vector): (Double, Vector) = {
    val grad = Vectors.zeros(x.size)
    val loss = computeInPlace(x, grad)
    (loss, grad)
  }

  override def doComputeInPlace(x: Vector, grad: Vector): Double = {
    numCalls += 1
    val seqOp = (agg: Agg, l: DiffFun[Vector]) => agg.add(l)
    val combOp = (agg1: Agg, agg2: Agg) => agg1.merge(agg2)
    val thisAgg = getAggregator(x)
    val newAgg = implicitly[Aggregable[M]].aggregate(losses, thisAgg)(seqOp, combOp)
    BLAS.axpy(1.0, newAgg.gradient, grad)

    val regLoss = regularizers.map {
      case diffReg: DiffFun[Vector] =>
        val loss = diffReg.computeInPlace(x, grad)
        loss
      case _ => 0.0
    }.sum
    newAgg.loss + regLoss
  }

  def addRegularization[T <: EnumeratedRegularization[Vector, T]](reg: T): SeparableDiffFun[M] = {
    val composed = if (!regularizers.exists(_.isInstanceOf[T])) {
      regularizers.map {
        case r: T => r.compose(reg)
        case other => other
      }
    } else {
      regularizers ++ List(reg)
    }
    new SeparableDiffFun[M](losses, getAggregator, composed, cache)
  }
}

/**
 * Loss function for "Efficient Minibatch Stochastic Optimization" by Li, et al. Takes an
 * existing loss function and adds a regularization term that penalizes deviations from the
 * previous parameters.
 *
 * @param subCost Original cost function.
 * @param prev Previous parameters.
 * @param gamma Regularization strength.
 */
class EMSOLossFunction[F <: DiffFun[Vector]](subCost: F, prev: Vector, gamma: Double)
  extends DiffFun[Vector] with Serializable {

  override def doCompute(x: Vector): (Double, Vector) = {
    val (l, g) = subCost.compute(x)
    val grad = x.copy
    BLAS.axpy(-1.0, prev, grad)
    BLAS.scal(gamma, grad)
    BLAS.axpy(1.0, g, grad)
    val loss = l + 0.5 * gamma * Vectors.sqdist(x, prev)
    (loss, grad)
  }

  override def doComputeInPlace(x: Vector, grad: Vector): Double = {
    throw new NotImplementedError()
  }
}

class ADMMLossFunction[F <: DiffFun[Vector]](subCost: F, z: Vector, u: Vector, rho: Double)
  extends DiffFun[Vector] with Serializable {

  override def doCompute(x: Vector): (Double, Vector) = {
    val (l, g) = subCost.compute(x)
    val grad = x.copy
    BLAS.axpy(-1.0, z, grad)
    BLAS.axpy(1.0, u, grad)
    val loss = l + 0.5 * rho * BLAS.dot(grad, grad)
    BLAS.axpy(rho, grad, g)
    (loss, g)
  }

  override def doComputeInPlace(x: Vector, grad: Vector): Double = {
    throw new NotImplementedError()
  }
}

