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
package org.apache.spark.ml.streaming

import org.apache.spark.SparkException
import org.apache.spark.ml.classification.ProbabilisticClassificationModel
import org.apache.spark.ml.classification.{ProbabilisticClassificationModel, ProbabilisticClassifier}
import org.apache.spark.sql.sources.StreamSinkProvider
import org.apache.spark.sql.streaming._
import org.apache.spark.sql.streaming.OutputMode

import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.param._
import org.apache.spark.sql.types.DoubleType

trait StreamingNaiveBayesParams extends Params {
  /**
   * The smoothing parameter.
   * (default = 1.0).
   *
   * @group param
   */
  final val smoothing: DoubleParam = new DoubleParam(this, "smoothing", "The smoothing parameter.",
    ParamValidators.gtEq(0))

  /** @group getParam */
  final def getSmoothing: Double = getOrDefault(smoothing)
}

class StreamingNaiveBayesModel(
    val uid: String,
    override val numFeatures: Int,
    val numClasses: Int) extends ProbabilisticClassificationModel[Vector, StreamingNaiveBayesModel]
  with StreamingNaiveBayesParams with StreamingModel[Array[(Double, (Long, DenseVector))]] {

  def update(updates: Array[(Double, (Long, DenseVector))]): Unit = {
    merge(updates) // updates the sufficient stats
    updateModel // updates theta and pi
  }

  private def merge(update: Array[(Double, (Long, DenseVector))]): Unit = {
    // TODO: use default value
    update.foreach { case (label, (numDocs, termCounts)) =>
      countsByClass.get(label) match {
        case Some((n, c)) =>
          BLAS.axpy(1.0, termCounts, c)
          countsByClass(label) = (n + numDocs, c)
        case None =>
          // new label encountered
          throw new SparkException("nb encountered a class label outside its range")
      }
    }
  }

  private def updateModel: Unit = {
    // TODO: fix this to getSmoothing
    val lambda = 1.0
    var numDocuments = 0L
    countsByClass.foreach { case (_, (n, _)) =>
      numDocuments += n
    }
    val numFeatures = countsByClass.head match { case (_, (_, v)) => v.size }

    val labels = new Array[Double](numClasses)
    val pi = new Array[Double](numClasses)
    val theta = Array.fill(numClasses)(new Array[Double](numFeatures))

    val piLogDenom = math.log(numDocuments + numClasses * lambda)
    var i = 0
    countsByClass.toArray.sortBy(_._1).foreach { case (label, (n, sumTermFreqs)) =>
      labels(i) = label
      pi(i) = math.log(n + lambda) - piLogDenom
      val thetaLogDenom = math.log(sumTermFreqs.values.sum + numFeatures * lambda)
      var j = 0
      while (j < numFeatures) {
        theta(i)(j) = math.log(sumTermFreqs(j) + lambda) - thetaLogDenom
        j += 1
      }
      i += 1
    }
    _pi = Vectors.dense(pi)
    _theta = new DenseMatrix(numClasses, numFeatures, theta.flatten, true)
  }

  private val countsByClass = {
    val mp = new collection.mutable.HashMap[Double, (Long, DenseVector)]
    (0 until numClasses).foreach { c =>
      mp.put(c.toDouble, (0L, Vectors.zeros(numFeatures).toDense))
    }
    mp
  }


  private val dataSize = numClasses * numFeatures

  private var _theta: Matrix =
    new DenseMatrix(numClasses, numFeatures, Array.fill(dataSize)(0.0), true)
  def theta: Matrix = _theta

  private var _pi: Vector = Vectors.dense(Array.fill(numClasses)(0.0))
  def pi: Vector = _pi

//  override def numFeatures: Int = theta.numCols // this is not safe
//
//  override def numClasses: Int = pi.size // this is not safe

  private def multinomialCalculation(features: Vector) = {
    val prob = theta.multiply(features)
    BLAS.axpy(1.0, pi, prob)
    prob
  }

  override protected def predictRaw(features: Vector): Vector = {
    multinomialCalculation(features)
  }

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction match {
      case dv: DenseVector =>
        var i = 0
        val size = dv.size
        val maxLog = dv.values.max
        while (i < size) {
          dv.values(i) = math.exp(dv.values(i) - maxLog)
          i += 1
        }
        val probSum = dv.values.sum
        i = 0
        while (i < size) {
          dv.values(i) = dv.values(i) / probSum
          i += 1
        }
        dv
      case sv: SparseVector =>
        throw new RuntimeException("Unexpected error in NaiveBayesModel:" +
          " raw2probabilityInPlace encountered SparseVector")
    }
  }

  override def copy(extra: ParamMap): StreamingNaiveBayesModel = {
    // TODO: not correct
    copyValues(new StreamingNaiveBayesModel(uid, numFeatures, numClasses)
      .setParent(this.parent), extra)
  }

}

class StreamingNaiveBayes (override val uid: String, numClasses: Int, numFeatures: Int)
  extends ProbabilisticClassifier[Vector, StreamingNaiveBayes, StreamingNaiveBayesModel]
    with StreamingNaiveBayesParams with StreamingEstimator[Array[(Double, (Long, DenseVector))]]
    with Serializable {

  def this(numClasses: Int, numFeatures: Int) =
    this(Identifiable.randomUID("snb"), numClasses, numFeatures)

  /**
   * Set the smoothing parameter.
   * Default is 1.0.
   *
   * @group setParam
   */
  def setSmoothing(value: Double): this.type = set(smoothing, value)
  setDefault(smoothing -> 1.0)

  // this should copy all the params here to the model
  // what if someone updates these? How to make them take effect in the model?
  // params should not be used in the model?
  // probably should not be able to set params once model has started streaming
  // or we could bake them into the sufficient stats, but gets messy fast
  var model: StreamingNaiveBayesModel = new StreamingNaiveBayesModel(uid, numFeatures, numClasses)
  def getModel: StreamingNaiveBayesModel = model

  override protected def train(dataset: Dataset[_]): StreamingNaiveBayesModel = {
    // TODO: actually implement this method
    getModel
  }

  /**
   * Update the class counts with a new chunk of data.
   *
   * @param ds Dataframe to add
   */
  def update(batchId: Long, ds: Dataset[_]): Unit = {
    import ds.sparkSession.implicits._
    val data = ds.select(
      col($(labelCol)).cast(DoubleType), col($(featuresCol))).rdd.map {
      case Row(label: Double, features: Vector) => LabeledPoint(label, features)
    }
    val newCountsByClass = add(data)
    model.update(newCountsByClass)
  }

  /**
   * Get class counts for a new chunk of data.
   * The logic for aggregating a new batch of data.
   */
  private def add(data: RDD[LabeledPoint]): Array[(Double, (Long, DenseVector))] = {
    data.map(lp => (lp.label, lp.features)).combineByKey[(Long, DenseVector)](
      createCombiner = (v: Vector) => {
        (1L, v.copy.toDense)
      },
      mergeValue = (c: (Long, DenseVector), v: Vector) => {
        // TODO: deal with sparse
        BLAS.axpy(1.0, v.toDense, c._2)
        (c._1 + 1L, c._2)
      },
      mergeCombiners = (c1: (Long, DenseVector), c2: (Long, DenseVector)) => {
        BLAS.axpy(1.0, c2._2, c1._2)
        (c1._1 + c2._1, c1._2)
      }
    ).collect()
  }

  override def copy(extra: ParamMap): StreamingNaiveBayes = defaultCopy(extra)
}
