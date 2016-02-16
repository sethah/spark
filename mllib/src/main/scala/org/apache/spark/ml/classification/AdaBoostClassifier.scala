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
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.ml.feature.Instance
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.ml.param._
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.param.shared._
import org.apache.spark.mllib.linalg.{BLAS, Vector}

private[classification] trait AdaBoostClassifierParams extends PredictorParams
  with HasMaxIter with HasTol with HasWeightCol with Logging {

}

final class AdaBoostClassifier (override val uid: String)
  extends Predictor[Vector, AdaBoostClassifier, AdaBoostClassificationModel]
  with AdaBoostClassifierParams{

  def this() = this(Identifiable.randomUID("gbtc"))

  def setWeightCol(value: String): this.type = set(weightCol, value)
  setDefault(weightCol -> "")

  val baseLearner = new LogisticRegression()

  override protected def train(dataset: DataFrame): AdaBoostClassificationModel = {
    val w = if ($(weightCol).isEmpty) lit(1.0) else col($(weightCol))
    val instances: RDD[Instance] = dataset.select(
      col($(labelCol)), w, col($(featuresCol))).map {
      case Row(label: Double, weight: Double, features: Vector) =>
        Instance(label, weight, features)
      }

    val numIterations = 10
    val K = 2
    val models = new Array[LogisticRegressionModel](numIterations)
    val alphas = new Array[Double](numIterations)
    var weightedInput = instances
    var m = 0
    while (m < numIterations) {
      val learner = baseLearner.copy(ParamMap.empty)
      val model = learner.train(weightedInput, false)
      models(m) = model
      val weightedError = weightedInput.map { case Instance(label, weight, features) =>
        val predicted = model.predict(features)
        if (predicted == label) 0.0 else weight
      }.sum()
      val totalWeight = weightedInput.map(x => x.weight).sum()
      val err = weightedError / totalWeight
      val alpha = math.log((1 - err) / err) + math.log(K - 1)
      alphas(m) = alpha
      weightedInput = weightedInput.map { case Instance(label, weight, features) =>
        val predicted = model.predict(features)
        val e = if (predicted == label) 0.0 else weight
        Instance(label, alpha * math.exp(e), features)
      }
      val totalAfterReweighting = weightedInput.map(x => x.weight).sum()
      weightedInput = weightedInput.map( x =>
        Instance(x.label, x.weight / totalAfterReweighting, x.features)
      )
      m += 1

    }
    new AdaBoostClassificationModel(this.uid, models, alphas)
  }

//  private[this] def boost(instances: RDD[Instance]): (LogisticRegressionModel, Double, RDD[Instances])

  override def copy(extra: ParamMap): AdaBoostClassifier = defaultCopy(extra)

}

final class AdaBoostClassificationModel private[ml](
   override val uid: String,
   val _models: Array[LogisticRegressionModel],
   val _weights: Array[Double]
   ) extends PredictionModel[Vector, AdaBoostClassificationModel] with Serializable {

  override private[ml] def predict(features: Vector): Double = {
    val totalPrediction = _models.zip(_weights).foldLeft(0.0) { case (acc, (model, weight)) =>
      val predicted = 2 * model.predict(features) - 1
      acc + predicted * weight
    }
    if (totalPrediction >= 0.0) 1.0 else 0.0
  }

  override def copy(extra: ParamMap): AdaBoostClassificationModel = {
    copyValues(new AdaBoostClassificationModel(uid, _models, _weights),
      extra).setParent(parent)
  }
}
