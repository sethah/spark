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
import org.apache.spark.ml.param.{Param, ParamMap, ParamValidators}
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util.{Identifiable, MetadataUtils}
import org.apache.spark.mllib.linalg.{BLAS, DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.functions._

/**
 * Params for AdaBoost Classifiers.
 */
private[classification] trait AdaBoostClassifierParams
  extends WeightBoostingParams[Vector] with Logging {

  /**
   * Set the AdaBoost algorithm.
   * Supported: "SAMME", "SAMME.R".
   * Default is "SAMME.R".
    *
    * @group param
   */
  final val algo: Param[String] = new Param[String](this, "algo", s"AdaBoost algorithm variant.  " +
    s"Supported options: ${AdaBoostClassifier.supportedAlgos.mkString(", ")}",
    ParamValidators.inArray[String](AdaBoostClassifier.supportedAlgos.toArray))

  def getAlgo: String = $(algo)
}

/**
* :: Experimental ::
* AdaBoost Classification.
*/
final class AdaBoostClassifier (override val uid: String)
  extends WeightBoostingClassifier[Vector, AdaBoostClassifier, AdaBoostClassificationModel]
  with AdaBoostClassifierParams {

  def this() = this(Identifiable.randomUID("adac"))

  @Since("2.0.0")
  override def setWeightCol(value: String): this.type = super.setWeightCol(value)

  @Since("2.0.0")
  override def setStepSize(value: Double): this.type = super.setStepSize(value)

  @Since("2.0.0")
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
  setDefault(algo -> "SAMME.R")

  override protected def train(dataset: DataFrame): AdaBoostClassificationModel = {
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
    val algorithm = getAlgo
    val models = new Array[BaseTransformerType[Vector]](numIterations)
    val estimatorWeights = new Array[Double](numIterations)
    var weightedInput = instances
    var m = 0
    var earlyStop = false
    while (m < numIterations && !earlyStop) {
      val (model, estimatorWeight, reweightFunction, stopBoosting) = algorithm match {
        case "SAMME" => boostDiscrete(dataset.sqlContext, weightedInput, numClasses, m)
        case "SAMME.R" => boostReal(dataset.sqlContext, weightedInput, numClasses, m)
      }

      earlyStop = (m == numIterations - 1) || stopBoosting
      if (estimatorWeight > 0) {
        models(m) = model
        estimatorWeights(m) = estimatorWeight
      }

      // skip reweighting if no more boosting iterations are left
      if (!earlyStop) {
        weightedInput = weightedInput.map(reweightFunction)
        val totalAfterReweighting = weightedInput.map(_.weight).sum()
        weightedInput = weightedInput.map( instance =>
          Instance(instance.label, instance.weight / totalAfterReweighting, instance.features)
        )
      }
      m += 1
    }
    copyValues(new AdaBoostClassificationModel(uid, models.takeWhile(_!=null),
      estimatorWeights.takeWhile(_!=null), numClasses))
  }

  private def shouldStopBoosting(estimatorError: Double, iter: Int, numClasses: Int): Boolean = {
    val isErrorWorseThanRandom = estimatorError >= 1.0 - (1.0 / numClasses)
    val isClassificationPerfect = estimatorError <= 0.0
    val isFirstIteration = iter == 0
    if (isErrorWorseThanRandom && isFirstIteration) {
      throw new RuntimeException("Error is worse than random guessing, model cannot be fit")
    }
    isErrorWorseThanRandom || isClassificationPerfect
  }

  private def getEstimatorError(transformer: BaseTransformerType[Vector],
      weightedInput: RDD[Instance]): Double = {
    val weightedError = weightedInput.map { case Instance(label, weight, features) =>
      val predicted = transformer.predict(features)
      AdaBoostClassifier.indicator(predicted, label, 0.0, weight)
    }.sum()
    val totalWeight = weightedInput.map(_.weight).sum()
    weightedError / totalWeight
  }

  /**
   * Perform a single boosting iteration for the SAMME discrete AdaBoost algorithm
   * for classification.
   *
   * @see [[http://en.wikipedia.org/wiki/AdaBoost AdaBoost]]
   *
   * The implementation is based upon:
   *   Zhu et al.,  "Multi-class AdaBoost."  January 12, 2006.
   *
   * @param sqlContext
   * @param weightedInput Input data to be boosted.
   * @param numClasses The number of classes for the target variable.
   * @param boostIter The current boosting iteration index.
   */
  private def boostDiscrete(sqlContext: SQLContext,
      weightedInput: RDD[Instance],
      numClasses: Int,
      boostIter: Int):
    (BaseTransformerType[Vector], Double, Instance => Instance, Boolean) = {
    val learner = makeLearner()
    val dfWeighted = sqlContext.createDataFrame(weightedInput)
    val labelMeta = NominalAttribute.defaultAttr.withName("label")
      .withNumValues(numClasses).toMetadata()
    val model = learner.fit(dfWeighted.select(dfWeighted("features"),
      dfWeighted("label").as("label", labelMeta), dfWeighted("weight")))

    val estimatorError = getEstimatorError(model, weightedInput)

    val stopBoosting = shouldStopBoosting(estimatorError, boostIter, numClasses)

    val estimatorWeight = if (estimatorError <= 0.0) {
      1.0
    } else {
      getStepSize * math.log((1 - estimatorError) / estimatorError) + math.log(numClasses - 1)
    }
    val reweightFunction: Instance => Instance = (instance: Instance) => {
      val e = AdaBoostClassifier.indicator(model.predict(instance.features),
        instance.label, 0.0, estimatorWeight)
      val newWeight = instance.weight * math.exp(e)
      Instance(instance.label, newWeight, instance.features)
    }
    (model, estimatorWeight, reweightFunction, stopBoosting)
  }

  /**
   * Perform a single boosting iteration for the SAMME.R real AdaBoost algorithm
   * for classification.
   *
   * @see [[http://en.wikipedia.org/wiki/AdaBoost AdaBoost]]
   *
   * The implementation is based upon:
   *   Zhu et al.,  "Multi-class AdaBoost."  January 12, 2006.
   *
   * @param sqlContext
   * @param weightedInput Input data to be boosted.
   * @param numClasses The number of classes for the target variable.
   * @param boostIter The current boosting iteration index.
   */
  private def boostReal(sqlContext: SQLContext,
      weightedInput: RDD[Instance],
      numClasses: Int,
      boostIter: Int):
    (BaseTransformerType[Vector], Double, (Instance => Instance), Boolean) = {
    val learner = makeLearner()
    val dfWeighted = sqlContext.createDataFrame(weightedInput)
    val labelMeta = NominalAttribute.defaultAttr.withName("label")
      .withNumValues(numClasses).toMetadata()
    val model = learner.fit(dfWeighted.select(dfWeighted("features"),
      dfWeighted("label").as("label", labelMeta), dfWeighted("weight")))

    val estimatorError = getEstimatorError(model, weightedInput)

    val stopBoosting = shouldStopBoosting(estimatorError, boostIter, numClasses)

    val reweightFunction: (Instance) => Instance = (instance: Instance) => {
      val p = model.predictProbability(instance.features)
      val coded = Array.tabulate(numClasses) { i =>
        if (i != instance.label) -1.0 / (numClasses - 1.0) else 1.0
      }
      val estimatorWeight = -1.0 * getStepSize * (numClasses / (numClasses - 1.0)) *
        BLAS.dot(new DenseVector(coded), p)
      val newWeight = math.exp(estimatorWeight)
      Instance(instance.label, newWeight, instance.features)
    }

    (model, 1.0, reweightFunction, stopBoosting)
  }

  override def copy(extra: ParamMap): AdaBoostClassifier = defaultCopy(extra)
}

@Since("2.0.0")
object AdaBoostClassifier {

  private def indicator(predicted: Double, label: Double,
      trueValue: Double = 1.0, falseValue: Double = 0.0): Double = {
    if (predicted == label) trueValue else falseValue
  }

  /** Set of Adaboost variants that AdaboostClassifier supports. */
  private[ml] val supportedAlgos = Set("SAMME", "SAMME.R")

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

  @Since("2.0.0")
  def models: Array[BaseTransformerType[Vector]] =
    _models.asInstanceOf[Array[BaseTransformerType[Vector]]]

  @Since("2.0.0")
  def modelWeights: Array[Double] = _modelWeights

  /**
   * SAMME probability function.
   *
   * This is algorithm 4, step 2, part c from
   *   Zhu et al.,  "Multi-class AdaBoost."  January 12, 2006.
   */
  private def sammeProba(transformer: BaseTransformerType[Vector], features: Vector): Vector = {
    val rawProba = transformer.predictProbability(features).toArray
    val logProba = rawProba.map(math.log(_))
    val sumLogProba = logProba.sum
    val proba = logProba.map { x =>
      (numClasses - 1.0) * (x - (1.0 / numClasses) * sumLogProba)
    }
    new DenseVector(proba)
  }

  def predictRaw_(features: Vector): Vector = {
    predictRaw(features)
  }

  def predict_(features: Vector): Double = {
    predict(features)
  }

  /**
   * Predict label for the given feature vector.
   */
  override private[ml] def predict(features: Vector): Double = {
    getAlgo match {
      case "SAMME" => discretePredict(features)
      case "SAMME.R" => realPredict(features)
      case other =>
        throw new RuntimeException(s"Cannot predict with unknown algorithm: $other")
    }
  }

  /**
   * Predict label for the "SAMME" AdaBoost algorithm.
   *
   * The prediction is a weighted sum of discrete predictions from each
   * model in the ensemble.
   */
  private def discretePredict(features: Vector): Double = {
    val predictions = Vectors.zeros(numClasses).asInstanceOf[DenseVector]
    _models.zip(_modelWeights).foreach { case (model, weight) =>
      predictions.values(model.predict(features).toInt) += weight
    }
    predictions.argmax
  }

  /**
   * Predict label for the "SAMME.R" AdaBoost algorithm.
   */
  private def realPredict(features: Vector): Double = {
    val sumProba = Vectors.zeros(numClasses)
    models.foreach { model =>
      BLAS.axpy(1.0, sammeProba(model, features), sumProba)
    }
    sumProba.argmax
  }

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction
  }

  override private[ml] def predictRaw(features: Vector): Vector = {
    getAlgo match {
      case "SAMME" => discretePredictRaw(features)
      case "SAMME.R" => realPredictRaw(features)
      case other =>
        throw new RuntimeException(s"Cannot predict with unknown algorithm: $other")
    }
  }

  /**
   * Raw prediction for "SAMME.R".
   *
   * This is the equation for recovering class conditional probabilities
   * in section 2.1 of
   *   Zhu et al.,  "Multi-class AdaBoost."  January 12, 2006.
   */
  private def realPredictRaw(features: Vector): Vector = {
    val proba = Vectors.zeros(numClasses).asInstanceOf[DenseVector]
    models.foreach { model =>
      BLAS.axpy(1.0 / _modelWeights.length.toDouble, sammeProba(model, features), proba)
    }
    var j = 0
    var probaSum = 0.0
    while (j < proba.size) {
      proba.values(j) = math.exp(1.0 / (numClasses - 1.0) * proba.values(j))
      probaSum += proba.values(j)
      j += 1
    }
    BLAS.scal(1.0 / probaSum, proba)
    proba
  }

  /**
   * Raw prediction for "SAMME".
   *
   * This is the equation for recovering class conditional probabilities
   * in section 2.1 of
   *   Zhu et al.,  "Multi-class AdaBoost."  January 12, 2006.
   */
  private def discretePredictRaw(features: Vector): Vector = {
    val weightSum = _modelWeights.sum
    val prediction = new Array[Double](numClasses)
    val proba = Vectors.zeros(numClasses).asInstanceOf[DenseVector]
    models.zip(modelWeights).foreach { case (model, weight) =>
      BLAS.axpy(weight / weightSum, model.predictProbability(features), proba)
    }
    var j = 0
    var probaSum = 0.0
    while (j < proba.size) {
      proba.values(j) = math.exp(1.0 / (numClasses - 1.0) * proba.values(j))
      probaSum += proba.values(j)
      j += 1
    }
    BLAS.scal(1.0 / probaSum, proba)
    proba
  }

  override def copy(extra: ParamMap): AdaBoostClassificationModel = {
    copyValues(new AdaBoostClassificationModel(uid, _models, _modelWeights, numClasses),
      extra).setParent(parent)
  }
}
