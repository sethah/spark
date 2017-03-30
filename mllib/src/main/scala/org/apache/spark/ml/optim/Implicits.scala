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
import org.apache.spark.ml.optim.loss.LossFunction
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.optim.aggregator.{DifferentiableLossAggregator, LeastSquaresAggregator}
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.VectorImplicits._

import scala.language.higherKinds
import scala.reflect.ClassTag

object Implicits {

  trait Aggregable[M[_]] extends Serializable {

    def aggregate[A: ClassTag, B: ClassTag](ma: M[A], b: B)
                                           (add: (B, A) => B, combine: (B, B) => B): B

  }

  implicit object RDDCanAggregate extends Aggregable[RDD] {
    override def aggregate[A: ClassTag, B: ClassTag](ma: RDD[A], b: B)
                                                    (add: (B, A) => B, combine: (B, B) => B): B = {
      ma.treeAggregate(b)(add, combine)
    }
  }

  implicit object IterableCanAggregate extends Aggregable[Iterable] {
    override def aggregate[A: ClassTag, B: ClassTag](fa: Iterable[A], b: B)(
      add: (B, A) => B, combine: (B, B) => B): B = {
      fa.foldLeft(b)(add)
    }
  }

  implicit def makeSubproblems[Agg <: DifferentiableLossAggregator[Instance, Agg]: ClassTag]
    : LossFunctionHasSubproblems[Agg] = {
    new LossFunctionHasSubproblems[Agg]
  }

  trait HasSubProblems[M[_], F <: DifferentiableFunction[Vector]] extends Serializable {
    def nextSubproblems(original: F): M[DifferentiableFunction[Vector]]
  }

  class LossFunctionHasSubproblems[Agg <: DifferentiableLossAggregator[Instance, Agg]: ClassTag]
    extends HasSubProblems[RDD, LossFunction[RDD, Agg]] {

    override def nextSubproblems(
       original: LossFunction[RDD, Agg]): RDD[DifferentiableFunction[Vector]] = {
      val reg = original.regularization
      val aggDepth = original.aggregationDepth
      val numPartitions = original.instances.getNumPartitions
      val nxt = original.instances.mapPartitionsWithIndex { (idx, it) =>
        val rng = new scala.util.Random()
        it.zipWithIndex.map { case (instance, i) =>
          (rng.nextInt(numPartitions), instance)
        }
      }
      original.instances.mapPartitions { it =>
        val rng = new scala.util.Random()
        // TODO: you could compute the full loss maybe, since we have to iterate over everything
        // anywya
        val filtered = it.filter { _ =>
          rng.nextDouble() < 0.2
        }
//        val iterable = it.toIterable
        val subProb = new LossFunction[Iterable, Agg](filtered.toIterable,
          original.aggProvider, reg, aggDepth)
        Iterator.single(subProb)
      }
//      nxt.groupByKey(numPartitions).map { case (part, it) =>
//        val subProb = new LossFunction[Iterable, Agg](it, original.aggProvider, reg, aggDepth)
//        subProb
//      }
    }
  }
}
