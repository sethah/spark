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
package org.apache.spark.ml.optim.minimizers

import org.apache.spark.annotation.Since
import org.apache.spark.ml.optim.{DiffFun, EMSOLossFunction, SeparableDiffFun}
import org.apache.spark.ml.optim.aggregator.{DiffFunAggregator, DifferentiableLossAggregator}
import org.apache.spark.ml.linalg.{BLAS, Vector, Vectors}
import org.apache.spark.ml.param.{DoubleParam, ParamMap, ParamValidators, Params}
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.optim.Implicits._
import org.apache.spark.ml.util.Identifiable

import scala.collection.mutable
import scala.reflect.ClassTag

trait EMSOParams extends Params {

  final val gamma: DoubleParam = new DoubleParam(this, "gamma", "gamma", ParamValidators.gtEq(0.0))

  def getGamma: Double = $(gamma)

}

class EMSOMinimizer(partitionMinimizer: IterativeMinimizer[Vector,
  DiffFun[Vector], IterativeMinimizerState[Vector]], @Since("2.3.0") override val uid: String)
  extends IterativeMinimizer[Vector, SeparableDiffFun[RDD], BreezeWrapperState[Vector]]
  with EMSOParams {

  @Since("2.3.0")
  def this(partitionMinimizer: IterativeMinimizer[Vector,
  DiffFun[Vector], IterativeMinimizerState[Vector]]) = this(partitionMinimizer,
    Identifiable.randomUID("emso"))

  type State = BreezeWrapperState[Vector]

  def setGamma(value: Double): this.type = set(gamma, value)
  setDefault(gamma -> 0.001)

  def initialState(lossFunction: SeparableDiffFun[RDD], initialParams: Vector): State = {
    val firstLoss = lossFunction.loss(initialParams)
    BreezeWrapperState(Vectors.dense(Array.fill(initialParams.size)(1.0)),
      initialParams, 0, firstLoss)
  }

  def iterations(lossFunction: SeparableDiffFun[RDD],
                 initialParameters: Vector): Iterator[State] = {
    val numFeatures = initialParameters.size
    Iterator.iterate(initialState(lossFunction, initialParameters)) { state =>
      val oldParams = state.params
      println(s"Old params: ${state.params}, old loss: ${state.loss}")
      val solutions = lossFunction.losses.mapPartitionsWithIndex { (i, problems) =>
        val probIter = problems.toIterable
        val numSamples = probIter.size
        val partitionLoss = new SeparableDiffFun(probIter, lossFunction.getAggregator,
          lossFunction.regularizers)
        val emsoLoss = new EMSOLossFunction(partitionLoss, oldParams, $(gamma))
        val (lastIter, lossHistory) = partitionMinimizer.takeLast(lossFunction, initialParameters)
        println(s"Partition $i: ${lastIter.params} - ${lossHistory.length}")
        BLAS.scal(numSamples, lastIter.params)
        Iterator.single((lastIter.loss * numSamples, lastIter.params, numSamples.toLong))
      }
      val seqOp = (acc: (Double, Vector, Long), x: (Double, Vector, Long)) => {
        val res: Vector = acc._2.toDense
        BLAS.axpy(1.0, x._2, res)
        (acc._1 + x._1, res, acc._3 + x._3)
      }
      val combOp = (acc1: (Double, Vector, Long), acc2: (Double, Vector, Long)) => {
        val res = acc2._2.toDense
        BLAS.axpy(1.0, acc1._2, res)
        (acc1._1 + acc2._1, res, acc1._3 + acc2._3)
      }
      val (loss, gradSum, count) = solutions.treeAggregate(
        (0.0, Vectors.sparse(numFeatures, Array.emptyIntArray, Array.emptyDoubleArray), 0L))(
        seqOp, combOp)
      BLAS.scal(1 / count.toDouble, gradSum)
      BreezeWrapperState(state.params, gradSum, state.iter + 1, loss / count.toDouble)
    }.takeWhile { state =>
      val dist = Vectors.sqdist(state.prev, state.params)
      dist > 1e-8
      state.iter < 100
    }
  }

  override def copy(extra: ParamMap): this.type = defaultCopy(extra)

}
