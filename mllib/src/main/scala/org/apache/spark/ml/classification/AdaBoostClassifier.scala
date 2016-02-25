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

import scala.language.existentials

import org.apache.spark.annotation.Since
import org.apache.spark.Logging
import org.apache.spark.ml.attribute._
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util.{Identifiable, MetadataUtils}
import org.apache.spark.mllib.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions._


private[classification] trait AdaBoostClassifierParams
  extends WeightBoostingParams[Vector] with HasWeightCol with Logging {

  final val algo: Param[String] = new Param[String](this, "algo", "")

  def getAlgo: String = $(algo)
}

/**
* :: Experimental ::
* AdaBoost Classification.
*/
final class AdaBoostClassifier (override val uid: String)
  extends WeightBoostingClassifier[Vector, AdaBoostClassifier, AdaBoostClassificationModel]
  with AdaBoostClassifierParams {

  def this() = this(Identifiable.randomUID("abc"))

  @Since("2.0.0")
  def setWeightCol(value: String): this.type = set(weightCol, value)
  setDefault(weightCol -> "")

  @Since("2.0.0")
  override def setStepSize(value: Double): this.type = super.setStepSize(value)

  override def setSeed(value: Long): this.type = super.setSeed(value)

  @Since("2.0.0")
  override def setMaxIter(value: Int): this.type = super.setMaxIter(value)
  setDefault(maxIter -> 10)

  @Since("2.0.0")
  def setBaseEstimators(value: Array[BaseEstimatorType[Vector]]): this.type =
    set(baseEstimators, value)
  setDefault(baseEstimators -> Array(
      new DecisionTreeClassifier().setWeightCol("weight").setMaxDepth(1).setMinInstancesPerNode(0)))

  @Since("2.0.0")
  def setAlgo(value: String): this.type = set(algo, value)

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
    val learningRate = getStepSize
    val models = new Array[BaseTransformerType[Vector]](numIterations)
    val estimatorWeights = new Array[Double](numIterations)
    var weightedInput = instances
    var m = 0
    var earlyStop = false
    while (m < numIterations && !earlyStop) {
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
      val estimatorError = weightedError / totalWeight

      if (estimatorError <= 1.0 - (1.0 / numClasses)) {
        throw new Exception("Error is worse than random guessing, model cannot be fit")
      }
      val estimatorWeight = if (estimatorError <= 0.0) {
        1.0
      } else {
        math.log((1 - estimatorError) / estimatorError) + math.log(numClasses - 1)
      }

      earlyStop = (estimatorError <= 0.0) || (m == numIterations - 1)
      models(m) = model
      estimatorWeights(m) = estimatorWeight

      if (!earlyStop) {
        weightedInput = weightedInput.map { case Instance(label, weight, features) =>
          val predicted = model.predict(features)
          val e = indicator(predicted, label, 0.0, estimatorWeight)
          Instance(label, weight * math.exp(e), features)
        }
        val totalAfterReweighting = weightedInput.map(_.weight).sum()
        weightedInput = weightedInput.map( instance =>
          Instance(instance.label, instance.weight / totalAfterReweighting, instance.features)
        )
      }
      m += 1
    }
    copyValues(new AdaBoostClassificationModel(uid, models, estimatorWeights, numClasses))
  }

  private[this] def indicator(predicted: Double, label: Double,
                              trueValue: Double = 1.0, falseValue: Double = 0.0): Double = {
    if (predicted == label) trueValue else falseValue
  }

  override def copy(extra: ParamMap): AdaBoostClassifier = defaultCopy(extra)
}

/**
* :: Experimental ::
* Model produced by [[AdaBoostClassifier]].
*/
final class AdaBoostClassificationModel private[ml](
   override val uid: String,
   val _models: Array[_ <: ProbabilisticClassificationModel[Vector, _]],
   val _modelWeights: Array[Double],
   val numClasses: Int
   ) extends WeightBoostingClassificationModel[Vector, AdaBoostClassificationModel]
  with AdaBoostClassifierParams with Serializable {

  @Since("1.4.0")
  def models: Array[BaseTransformerType[Vector]] =
    _models.asInstanceOf[Array[BaseTransformerType[Vector]]]

  @Since("1.4.0")
  def modelWeights: Array[Double] = _modelWeights

  override private[ml] def predictRaw(features: Vector): Vector = {
    val weightSum = _modelWeights.sum
    val prediction = new Array[Double](numClasses)
    for (m <- 0 until _models.length) {
      val rawPrediction = _models(m).predictProbability(features).toArray
      val weight = _modelWeights(m)
      for  (k <- 0 until numClasses) {
        prediction(k) += rawPrediction(k) * weight / weightSum
      }
    }
    new DenseVector(prediction)
  }

  def predictRaw_(features: Vector): Vector = {
    predictRaw(features)
  }

  def predict_(features: Vector): Double = {
    predict(features)
  }

  override private[ml] def predict(features: Vector): Double = {
    val predictions = Vectors.zeros(numClasses).asInstanceOf[DenseVector]
    _models.zip(_modelWeights).foreach { case (model, weight) =>
      predictions.values(model.predict(features).toInt) += weight
    }
    predictions.argmax
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
    rawPrediction
  }

  override def copy(extra: ParamMap): AdaBoostClassificationModel = {
    copyValues(new AdaBoostClassificationModel(uid, _models, _modelWeights, numClasses),
      extra).setParent(parent)
  }
}
