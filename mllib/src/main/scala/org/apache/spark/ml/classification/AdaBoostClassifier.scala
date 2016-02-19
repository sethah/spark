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

package org.apache.spark.ml.classification

import org.apache.spark.Logging
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{Identifiable, MetadataUtils}
import org.apache.spark.ml.{PredictorParams, PredictionModel, Predictor}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.ml.feature.Instance
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.ml.param._
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.param.shared._
import org.apache.spark.mllib.linalg.{BLAS, Vector, DenseVector}
import org.apache.spark.ml.classification.AdditiveClassifierParams._
import scala.language.existentials

private[classification] trait AdaBoostClassifierParams extends PredictorParams
  with HasMaxIter with HasTol with HasWeightCol with Logging {

}

final class AdaBoostClassifier (override val uid: String)
  extends AdditiveClassifier[Vector, AdaBoostClassifier, AdaBoostClassificationModel]
  with AdaBoostClassifierParams {

  def this() = this(Identifiable.randomUID("gbtc"))

  def setWeightCol(value: String): this.type = set(weightCol, value)
  setDefault(weightCol -> "")

  def setBaseEstimators(value: Array[BaseClassifierType[Vector]]): this.type = set(baseEstimators, value)
  setDefault(baseEstimators -> Array(new LogisticRegression))

  protected def makeLearner: BaseClassifierType[Vector] = {
    getBaseEstimators.head.copy(ParamMap.empty)
  }

  override protected def train(dataset: DataFrame): AdaBoostClassificationModel = {
    import dataset.sqlContext.implicits._
    val w = if ($(weightCol).isEmpty) lit(1.0) else col($(weightCol))
    val instances: RDD[Instance] = dataset.select(
      col($(labelCol)), w, col($(featuresCol))).map {
      case Row(label: Double, weight: Double, features: Vector) =>
        Instance(label, weight, features)
      }

    val numIterations = 10
    val models = new Array[BaseClassificationModelType[Vector]](numIterations)
    val alphas = new Array[Double](numIterations)
    var weightedInput = instances
    var m = 0
    while (m < numIterations) {
      println(m)
      val learner = makeLearner
      val dfWeighted = weightedInput.toDF()
      val model = learner.fit(dfWeighted)

      val weightedError = weightedInput.map { case Instance(label, weight, features) =>
        val predicted = model.predict(features)
        indicator(predicted, label, 0.0, weight)
      }.sum()
      val totalWeight = weightedInput.map(x => x.weight).sum()
      val err = weightedError / totalWeight
      val alpha = if (err < 0.00001) 0.0 else math.log((1 - err) / err)

      models(m) = model
      alphas(m) = alpha

      weightedInput = weightedInput.map { case Instance(label, weight, features) =>
        val predicted = model.predict(features)
        val e = indicator(predicted, label, 0.0, alpha)
        Instance(label, weight * math.exp(e), features)
      }
      val totalAfterReweighting = weightedInput.map(_.weight).sum()
      weightedInput = weightedInput.map( instance =>
        Instance(instance.label, instance.weight / totalAfterReweighting, instance.features)
      )

      m += 1

    }
    new AdaBoostClassificationModel(this.uid, models, alphas)
  }

  private[this] def indicator(predicted: Double, label: Double,
                              trueValue: Double = 1.0, falseValue: Double = 0.0): Double = {
    if (predicted == label) trueValue else falseValue
  }


  override def copy(extra: ParamMap): AdaBoostClassifier = defaultCopy(extra)

}

final class AdaBoostClassificationModel private[ml](
   override val uid: String,
   val _models: Array[BaseClassificationModelType[Vector]],
   val _weights: Array[Double],
   val numClasses: Int = 2
   ) extends AdditiveClassificationModel[Vector, AdaBoostClassificationModel] with Serializable {

  val models_ = new Array[BaseClassificationModelType[Vector]](1)
  override private[ml] def predictRaw(features: Vector): Vector = {
    val prediction = new Array[Double](numClasses)
    for (m <- 0 until _models.length) {
      val rawPrediction = _models(m).predictRaw(features).toArray
      val weight = _weights(m)
      for  (k <- 0 until numClasses) {
        prediction(k) += rawPrediction(k) * weight
      }
    }
    new DenseVector(prediction)
  }

  def predict_(features: Vector): Vector = {
    predictRaw(features)
  }

  override private[ml] def predict(features: Vector): Double = {
    val totalPrediction = _models.zip(_weights).foldLeft(0.0) { case (acc, (model, weight)) =>
      val predicted = 2 * model.predict(features) - 1
      acc + predicted * weight
    }
    if (totalPrediction >= 0.0) 1.0 else 0.0
  }

  /**
   * Estimate the probability of each class given the raw prediction,
   * doing the computation in-place.
   * These predictions are also called class conditional probabilities.
   *
   * This internal method is used to implement [[transform()]] and output [[probabilityCol]].
   *
   * @return Estimated class conditional probabilities (modified input vector)
   */
  protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    new DenseVector(Array(0))
  }

  override def copy(extra: ParamMap): AdaBoostClassificationModel = {
    copyValues(new AdaBoostClassificationModel(uid, _models, _weights),
      extra).setParent(parent)
  }
}
