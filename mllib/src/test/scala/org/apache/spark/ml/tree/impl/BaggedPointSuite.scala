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

package org.apache.spark.ml.tree.impl

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.feature.{Instance, LabeledPoint}
import org.apache.spark.ml.linalg.{BLAS, Vectors}
import org.apache.spark.mllib.tree.EnsembleTestHelper
import org.apache.spark.mllib.tree.configuration.{Algo, Strategy}
import org.apache.spark.mllib.tree.impurity.Gini
import org.apache.spark.mllib.util.MLlibTestSparkContext

/**
 * Test suite for [[BaggedPoint]].
 */
class BaggedPointSuite extends SparkFunSuite with MLlibTestSparkContext  {

  test("BaggedPoint RDD: without subsampling") {
    val arr = EnsembleTestHelper.generateOrderedLabeledPoints(1, 1000)
    val rdd = sc.parallelize(arr)
    val baggedRDD = BaggedPoint.convertToBaggedRDD(rdd, 1.0, 1, false, seed = 42)
    baggedRDD.collect().foreach { baggedPoint =>
      assert(baggedPoint.subsampleCounts.size == 1 && baggedPoint.subsampleCounts(0) == 1)
    }
  }
  test("my with replacement") {
    val ctx = spark.sqlContext
    import ctx.implicits._
    val dataSeq = Array(
      Instance(0.0, 100, Vectors.dense(0.0)),
      Instance(0.0, 10, Vectors.dense(0.0)),
      Instance(0.0, 50, Vectors.dense(0.0)),
      Instance(0.0, 500, Vectors.dense(0.0))
    )
    val oversampled = dataSeq.flatMap { x =>
      Iterator.fill(x.weight.toInt)(Instance(x.label, 1.0, x.features))
    }
    val rng = new scala.util.Random(42)
//    val dataSeq = Array.tabulate[Instance](100) { i =>
//      Instance(i.toDouble, rng.nextDouble, Vectors.dense(0.0))
//    }
    val data = dataSeq.toSeq.toDF()
    val strategy = new Strategy(
      Algo.Classification,
      Gini,
      maxDepth = 2,
      numClasses = 0,
      maxBins = 5)
    val metadata = DecisionTreeMetadata.buildMetadata(data.as[Instance].rdd, strategy)
    val extractWeightFunc = (tp: Instance) => tp.weight *
      metadata.numExamples / metadata.weightedNumExamples
    val numTrees = 1000
    val baggedRDD =
      BaggedPoint.convertToBaggedRDD[Instance](data.as[Instance].rdd, 1.0, numTrees, true,
        extractWeightFunc)
    val cumWeightVec = Vectors.zeros(numTrees)
    val cumVec = Vectors.zeros(numTrees)
    var j = 0
    baggedRDD.collect().foreach {bp =>
//            println(bp.datum, bp.subsampleCounts.mkString(","))
      println(bp.subsampleCounts.sum / bp.subsampleCounts.length.toDouble)
      BLAS.axpy(1.0, Vectors.dense(bp.subsampleCounts.map(_ * dataSeq(j).weight)), cumWeightVec)
      BLAS.axpy(1.0, Vectors.dense(bp.subsampleCounts.map(_.toDouble)), cumVec)
      j += 1
    }
    println("c VEC!", cumVec)
    j = 0
    baggedRDD.collect().foreach { bp =>
      val fracs = bp.subsampleCounts.zip(cumVec.toArray).map { case (x, total) =>
        if (total == 0) 0.0 else x / total.toDouble
      }
      j += 1
      println("weight frac", fracs.sum / fracs.length)
    }
    val weightSum = dataSeq.map(_.weight).sum
    val normCumWeightVec = cumWeightVec.toArray.map(_ / weightSum)
//    println(normCumWeightVec.sum / normCumWeightVec.length)
    println(cumVec.toArray.sum / cumVec.size)
  }

  test("my without replacement") {
    val ctx = spark.sqlContext
    import ctx.implicits._
//    val dataSeq = Array(
//      Instance(0.0, 100, Vectors.dense(0.0)),
//      Instance(0.0, 10, Vectors.dense(0.0)),
//      Instance(0.0, 50, Vectors.dense(0.0)),
//      Instance(0.0, 500, Vectors.dense(0.0))
//    )
    val rng = new scala.util.Random(42)
    var wsum = 0.0
    val dataSeq = Array.tabulate[Instance](100) { i =>
      val _w = rng.nextDouble
      wsum += _w
      Instance(i.toDouble, _w, Vectors.dense(0.0))
    }
    val beta = 0.7
    val psum = dataSeq.map { i => math.min(i.weight * dataSeq.length * beta / wsum, 1.0)}.sum
    val data = dataSeq.toSeq.toDF()
    val strategy = new Strategy(
      Algo.Classification,
      Gini,
      maxDepth = 2,
      numClasses = 0,
      maxBins = 5)
    val metadata = DecisionTreeMetadata.buildMetadata(data.as[Instance].rdd, strategy)
    val extractWeightFunc = (tp: Instance) => tp.weight *
        metadata.numExamples / metadata.weightedNumExamples
    val numTrees = 1000
    val baggedRDD =
      BaggedPoint.convertToBaggedRDD[Instance](data.as[Instance].rdd, beta, numTrees, false,
        extractWeightFunc)
    val cumVec = Vectors.zeros(numTrees)
    var j = 0
    baggedRDD.collect().foreach {bp =>
//      println(bp.datum)
      println(bp.subsampleCounts.sum / bp.subsampleCounts.length.toDouble)
      BLAS.axpy(1.0, Vectors.dense(bp.subsampleCounts.map(_.toDouble)), cumVec)
        j += 1
    }
    println(cumVec)
    println(metadata.numExamples, metadata.weightedNumExamples, wsum, psum / dataSeq.length)
    val weightSum = dataSeq.map(_.weight).sum
    val normCumVec = cumVec.toArray.map(_ / dataSeq.length.toDouble)
    println(normCumVec.sum / normCumVec.length)
  }

  test("BaggedPoint RDD: with subsampling with replacement (fraction = 1.0)") {
    val numSubsamples = 100
    val (expectedMean, expectedStddev) = (1.0, 1.0)

    val seeds = Array(123, 5354, 230, 349867, 23987)
    val arr = EnsembleTestHelper.generateOrderedLabeledPoints(1, 1000)
    val rdd = sc.parallelize(arr)
    seeds.foreach { seed =>
      val baggedRDD = BaggedPoint.convertToBaggedRDD(rdd, 1.0, numSubsamples, true, seed = seed)
      val subsampleCounts: Array[Array[Double]] =
        baggedRDD.map(_.subsampleCounts.map(_.toDouble)).collect()
      EnsembleTestHelper.testRandomArrays(subsampleCounts, numSubsamples, expectedMean,
        expectedStddev, epsilon = 0.01)
    }
  }

  test("BaggedPoint RDD: with subsampling with replacement (fraction = 0.5)") {
    val numSubsamples = 100
    val subsample = 0.5
    val (expectedMean, expectedStddev) = (subsample, math.sqrt(subsample))

    val seeds = Array(123, 5354, 230, 349867, 23987)
    val arr = EnsembleTestHelper.generateOrderedLabeledPoints(1, 1000)
    val rdd = sc.parallelize(arr)
    seeds.foreach { seed =>
      val baggedRDD =
        BaggedPoint.convertToBaggedRDD(rdd, subsample, numSubsamples, true, seed = seed)
      val subsampleCounts: Array[Array[Double]] =
        baggedRDD.map(_.subsampleCounts.map(_.toDouble)).collect()
      EnsembleTestHelper.testRandomArrays(subsampleCounts, numSubsamples, expectedMean,
        expectedStddev, epsilon = 0.01)
    }
  }

  test("BaggedPoint RDD: with subsampling without replacement (fraction = 1.0)") {
    val numSubsamples = 100
    val (expectedMean, expectedStddev) = (1.0, 0)

    val seeds = Array(123, 5354, 230, 349867, 23987)
    val arr = EnsembleTestHelper.generateOrderedLabeledPoints(1, 1000)
    val rdd = sc.parallelize(arr)
    seeds.foreach { seed =>
      val baggedRDD = BaggedPoint.convertToBaggedRDD(rdd, 1.0, numSubsamples, false, seed = seed)
      val subsampleCounts: Array[Array[Double]] =
        baggedRDD.map(_.subsampleCounts.map(_.toDouble)).collect()
      EnsembleTestHelper.testRandomArrays(subsampleCounts, numSubsamples, expectedMean,
        expectedStddev, epsilon = 0.01)
    }
  }

  test("BaggedPoint RDD: with subsampling without replacement (fraction = 0.5)") {
    val numSubsamples = 100
    val subsample = 0.5
    val (expectedMean, expectedStddev) = (subsample, math.sqrt(subsample * (1 - subsample)))

    val seeds = Array(123, 5354, 230, 349867, 23987)
    val arr = EnsembleTestHelper.generateOrderedLabeledPoints(1, 1000)
    val rdd = sc.parallelize(arr)
    seeds.foreach { seed =>
      val baggedRDD = BaggedPoint.convertToBaggedRDD(rdd, subsample, numSubsamples, false,
        seed = seed)
      val subsampleCounts: Array[Array[Double]] =
        baggedRDD.map(_.subsampleCounts.map(_.toDouble)).collect()
      EnsembleTestHelper.testRandomArrays(subsampleCounts, numSubsamples, expectedMean,
        expectedStddev, epsilon = 0.01)
    }
  }
}
