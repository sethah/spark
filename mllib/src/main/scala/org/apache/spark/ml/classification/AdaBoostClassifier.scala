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
import org.apache.spark.annotation.Since
import org.apache.spark.ml.classification.WeightBoostingClassifierParams.{WeightBoostingClassifierBaseType, WeightBoostingClassificationModelBaseType}
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
//import org.apache.spark.ml.classification.WeightBoostingClassifierParams
//import org.apache.spark.ml.classification.WeightBoostingClassifierParams
import scala.language.existentials
import org.apache.spark.ml.attribute._
import org.apache.spark.ml.feature.StringIndexer

private[classification] trait AdaBoostClassifierParams
  extends WeightBoostingClassifierParams[Vector] with HasTol with HasWeightCol with Logging {

}

final class AdaBoostClassifier (override val uid: String)
  extends WeightBoostingClassifier[Vector, AdaBoostClassifier, AdaBoostClassificationModel]
  with AdaBoostClassifierParams {

  def this() = this(Identifiable.randomUID("abc"))

  @Since("2.0.0")
  def setWeightCol(value: String): this.type = set(weightCol, value)
  setDefault(weightCol -> "")

  @Since("2.0.0")
  override def setMaxIter(value: Int): this.type = super.setMaxIter(value)

  @Since("2.0.0")
  def setBaseEstimators(value: Array[WeightBoostingClassifierBaseType[Vector]]): this.type = set(baseEstimators, value)
  setDefault(baseEstimators -> Array(new DecisionTreeClassifier().setWeightCol("weight").setMaxDepth(1).setMinInstancesPerNode(0)))

  protected def makeLearner: WeightBoostingClassifierBaseType[Vector] = {
    getBaseEstimators.head.copy(ParamMap.empty)
  }

  override protected def train(dataset: DataFrame): AdaBoostClassificationModel = {
    import dataset.sqlContext.implicits._
    val numExamples = dataset.count()
    val numClasses: Int = MetadataUtils.getNumClasses(dataset.schema($(labelCol))) match {
      case Some(n: Int) => n
      case None => throw new IllegalArgumentException("AdaBoostClassifier was given input" +
        s" with invalid label column ${$(labelCol)}, without the number of classes" +
        " specified. See StringIndexer.")
      // TODO: Automatically index labels: SPARK-7126
    }
    val w = if ($(weightCol).isEmpty) lit(1.0) else col($(weightCol))
    val instances: RDD[Instance] = dataset.select(
      col($(labelCol)), w, col($(featuresCol))).map {
      case Row(label: Double, weight: Double, features: Vector) =>
        Instance(label, weight, features)
      }

    val numIterations = getMaxIter
    val models = new Array[WeightBoostingClassificationModelBaseType[Vector]](numIterations)
    val alphas = new Array[Double](numIterations)
    var weightedInput = instances
    var m = 0
    while (m < numIterations) {
      val learner = makeLearner
      val dfWeighted = weightedInput.toDF()
      val labelMeta = NominalAttribute.defaultAttr.withName("label")
        .withNumValues(numClasses).toMetadata()
      val model = learner.fit(dfWeighted.select(dfWeighted("features"),
        dfWeighted("label").as("label", labelMeta), dfWeighted("weight")))

      val weightedError = weightedInput.map { case Instance(label, weight, features) =>
        val predicted = model.predict(features)
        indicator(predicted, label, 0.0, weight)
      }.sum()
      val totalWeight = weightedInput.map(_.weight).sum()
      val err = weightedError / totalWeight
      val alpha = if (err < 0.00001) 0.0 else math.log((1 - err) / err) + math.log(numClasses - 1)

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
    copyValues(new AdaBoostClassificationModel(uid, models, alphas, numClasses))
  }

  private[this] def indicator(predicted: Double, label: Double,
                              trueValue: Double = 1.0, falseValue: Double = 0.0): Double = {
    if (predicted == label) trueValue else falseValue
  }


  override def copy(extra: ParamMap): AdaBoostClassifier = defaultCopy(extra)

}

final class AdaBoostClassificationModel private[ml](
   override val uid: String,
   val _models: Array[WeightBoostingClassificationModelBaseType[Vector]],
   val _weights: Array[Double],
   val numClasses: Int
   ) extends AdditiveClassificationModel[Vector, AdaBoostClassificationModel] with Serializable {

  override private[ml] def predictRaw(features: Vector): Vector = {
    val weightSum = _weights.sum
    val prediction = new Array[Double](numClasses)
    for (m <- 0 until _models.length) {
      val rawPrediction = _models(m).predictRaw(features).toArray
      val rawSum = rawPrediction.sum
//      rawPrediction.foreach(x => printf(s"${x.toString},"))
//      println()
      val weight = _weights(m)
      for  (k <- 0 until numClasses) {
        prediction(k) += rawPrediction(k) / rawSum * weight / weightSum
      }
    }
//    println("-----")
    new DenseVector(prediction)
  }

  def predictRaw_(features: Vector): Vector = {
    predictRaw(features)
  }

  def predict_(features: Vector): Double = {
    predict(features)
  }

  private[ml] def predict2(features: Vector): Double = {
    val totalPrediction = _models.zip(_weights).foldLeft(0.0) { case (acc, (model, weight)) =>
      val predicted = 2 * model.predict(features) - 1
      acc + predicted * weight
    }
    if (totalPrediction >= 0.0) 1.0 else 0.0
  }

  override private[ml] def predict(features: Vector): Double = {
    val predictions = new Array[Double](numClasses)
    _models.zip(_weights).foreach { case (model, weight) =>
      predictions(model.predict(features).toInt) += weight
    }
    predictions.zipWithIndex.maxBy(x => x._1)._2
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
    copyValues(new AdaBoostClassificationModel(uid, _models, _weights, numClasses),
      extra).setParent(parent)
  }
}
