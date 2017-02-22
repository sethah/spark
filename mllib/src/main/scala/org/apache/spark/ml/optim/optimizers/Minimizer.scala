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

import org.apache.spark.ml.optim.DifferentiableFunction
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}

import scala.reflect.ClassTag

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

//trait DataSpecificMinimizer[Features, Label, T, F <: (T => Double)] extends Minimizer[T, F] {
//
//  def minimize(lossFunction: (Features, Label, T) => Double, initialParameters: T): T
//
//}

//trait CanAggregate[U, T, Collection] {
//  def aggregate(data: Collection)(zeroValue: U)(seqOp: (U, T) => U, combOp: (U, U) => U): U
//}
//
//class RDDCanAggregate[U, T] extends CanAggregate[U, T, RDD[T]] {
//  def aggregate(data: RDD[T])(zeroValue: U)(seqOp: (U, T) => U, combOp: (U, U) => U): U = {
//    data.treeAggregate(zeroValue)(seqOp, combOp)
//  }
//}
//
//class DataFrameCanAggregate[U, T: ClassTag] extends CanAggregate[U, T, DataFrame] {
//  def aggregate(data: DataFrame)(zeroValue: U)(seqOp: (U, T) => U, combOp: (U, U) => U): U = {
//    data.as[T].rdd.treeAggregate(zeroValue)(seqOp, combOp)
//  }
//}

//trait CollectionMinimizer[Datum, Params, Collection] extends Minimizer[Params, (Params => Double)] {
//  def datumCostFun: Datum => (Params, Double)
//  def minimize(init: Params): Params
//  def getCostFun(singleCost: Datum => (Params, Double)): (Params => Double)
//  def minimize(loss: Params => Double, init: Params): Params
//}
//
//class FullGradientMinimizer[Datum, Params,
//Collection: CanAggregate[DifferentiableLossFunctionAggregator[Params, Datum], Datum, Collection]]
//(data: Collection, aggregator: DifferentiableLossFunctionAggregator[Params, Datum],
// minimizer: IterativeMinimizer[Params, DifferentiableFunction[Params],
//   IterativeMinimizerState[Params]])
//  extends CollectionMinimizer[Datum, Params, Collection] {
//
//  val ev = implicitly[CanAggregate[Agg, Datum, Collection]]
//  type Agg = _ <: DifferentiableLossFunctionAggregator[Params, Datum]
//  val costFun = new DifferentiableFunction[Params] {
//    def doCompute(x: Params): (Params, Double) = {
//      val aggregator = {
//        val seqOp = (c: Agg, instance: Datum) => c.add(instance)
//        val combOp =
//          (c1: Agg, c2: Agg) => c1.merge(c2)
//        ev.aggregate(data)(aggregator)(seqOp, combOp)
//      }
//      (aggregator.gradient, aggregator.loss)
//    }
//  }
//
//  override def minimize(lossFunction: (Params => Double), initialParameters: Params): Params = {
//
//  }
//
//
//}

//class FullGradientMinimizer2[Datum, Params,
//  Collection: CanAggregate[DifferentiableLossFunctionAggregator[Params, Datum],
//    Datum, Collection]](data: Collection,
//                        aggregator: DifferentiableLossFunctionAggregator[Params, Datum])
//  extends IterativeMinimizer[Params, DifferentiableFunction[Params],
//  IterativeMinimizerState[Params]] {
//  /** Type alias for convenience */
//  private type State = BreezeWrapperState[Params]
//  type Agg = _ <: DifferentiableLossFunctionAggregator[Params, Datum]
//  val ev = implicitly[CanAggregate[Agg, Datum, Collection]]
//  val costFun = new DifferentiableFunction[Params] {
//    def doCompute(x: Params): (Params, Double) = {
//      val aggregator = {
//        val seqOp = (c: Agg, instance: Datum) => c.add(instance)
//        val combOp =
//          (c1: Agg, c2: Agg) => c1.merge(c2)
//        ev.aggregate(data)(aggregator)(seqOp, combOp)
//      }
//      (aggregator.gradient, aggregator.loss)
//    }
//  }
////  def minimize(lossFunctionAggregator: Agg, initialParameters: T): T = {
////    val ev = implicitly[CanAggregate[Agg, Datum, Collection]]
////    initialParameters
////  }
////  override def iterations(lossFunction: DifferentiableFunction[T],
////                          initialParameters: T): Iterator[State] = {
//
////}
//
//
//  override def copy(extra: ParamMap): this.type = {
//    defaultCopy(extra)
//  }
//}

//class FullGradientCostFun[T, C](data: C) extends DifferentiableFunction[T] {
//  def doCompute(x: T): (T, Double) = {
//
//  }
//}

/*
  val lr = new LogisticRegression()
    .setOptimizationStrategy("full") // or "sgd", "emso", "cd"
    .setOptimizer(new LFBGS())

  val minimizer = new FullGradientMinimizer(lbfgs, data)
  minimizer.minimize(logisticCostFun
 */


//trait LossFunctionAggregator[Datum] {
//  def add(instance: Datum): this.type
//  def merge(other: this.type): this.type
//  def loss: Double
//}
//
//trait DifferentiableLossFunctionAggregator[T, Datum] extends LossFunctionAggregator[Datum] {
//  def gradient: T
//}
/*
  val addGradient(instance: Instance, weights: Vector): (Double, Vector) = {
    blas.dot(features, weights) +
  val agg =
 */





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

