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


private[classification] trait AdaBoostClassifierParams
  extends WeightBoostingParams[Vector] with HasWeightCol with Logging {

  final val algo: Param[String] = new Param[String](this, "algo", "",
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

  def this() = this(Identifiable.randomUID("abc"))

  /**
  * Set the smoothing parameter.
  * Default is 1.0.
  * @group setParam
  */
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
  setDefault(algo -> "SAMME.R")

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
    val algorithm = getAlgo
    val models = new Array[BaseTransformerType[Vector]](numIterations)
    val estimatorWeights = new Array[Double](numIterations)
    var weightedInput = instances
    var m = 0
    var earlyStop = false
    while (m < numIterations && !earlyStop) {
      val (model, estimatorWeight, reweightFunction) = if (algorithm == "SAMME.R") {
        boostDiscrete(dataset.sqlContext, weightedInput, numClasses)
      } else {
        boostReal(dataset.sqlContext, weightedInput, numClasses)
      }

      earlyStop = (m == numIterations - 1) // || (estimatorError <= 0.0)
      models(m) = model
      estimatorWeights(m) = estimatorWeight

      if (!earlyStop) {
        weightedInput = weightedInput.map(reweightFunction)
        val totalAfterReweighting = weightedInput.map(_.weight).sum()
        weightedInput = weightedInput.map( instance =>
          Instance(instance.label, instance.weight / totalAfterReweighting, instance.features)
        )
      }
      m += 1
    }
    copyValues(new AdaBoostClassificationModel(uid, models.view(0, m).toArray,
      estimatorWeights.view(0, m).toArray, numClasses))
  }

  private def boostDiscrete(sqlContext: SQLContext, weightedInput: RDD[Instance], numClasses: Int):
    (BaseTransformerType[Vector], Double, Instance => Instance) = {
    val learner = makeLearner()
    val dfWeighted = sqlContext.createDataFrame(weightedInput)
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

    // TODO: should not throw an error here, just stop early (unless this is first iteration)
//    if (estimatorError >= 1.0 - (1.0 / numClasses)) {
//      throw new RuntimeException("Error is worse than random guessing, model cannot be fit")
//    }
    val estimatorWeight = if (estimatorError <= 0.0) {
      1.0
    } else {
      math.log((1 - estimatorError) / estimatorError) + math.log(numClasses - 1)
    }
    val reweightFunction: Instance => Instance = (instance: Instance) => {
      val e = indicator(model.predict(instance.features), instance.label, 0.0, estimatorWeight)
      val newWeight = instance.weight * math.exp(e)
      Instance(instance.label, newWeight, instance.features)
    }
    (model, estimatorWeight, reweightFunction)
  }

  private def boostReal(sqlContext: SQLContext, weightedInput: RDD[Instance], numClasses: Int):
    (BaseTransformerType[Vector], Double, (Instance => Instance)) = {
    val learner = makeLearner()
    val dfWeighted = sqlContext.createDataFrame(weightedInput)
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

    val reweightFunction: (Instance) => Instance = (instance: Instance) => {
      val p = model.predictProbability(instance.features)
      val coded = Array.tabulate(numClasses) { i =>
        if (i != instance.label) -1.0 / (numClasses - 1.0) else 1.0
      }
      val estimatorWeight = -(numClasses / (numClasses - 1.0)) * BLAS.dot(new DenseVector(coded), p)
      val newWeight = math.exp(estimatorWeight)
      Instance(instance.label, newWeight, instance.features)
    }

    (model, 1.0, reweightFunction)
  }

  private[this] def indicator(predicted: Double, label: Double,
                              trueValue: Double = 1.0, falseValue: Double = 0.0): Double = {
    if (predicted == label) trueValue else falseValue
  }

  override def copy(extra: ParamMap): AdaBoostClassifier = defaultCopy(extra)
}

@Since("2.0.0")
object AdaBoostClassifier {

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

  @Since("1.4.0")
  def models: Array[BaseTransformerType[Vector]] =
    _models.asInstanceOf[Array[BaseTransformerType[Vector]]]

  @Since("1.4.0")
  def modelWeights: Array[Double] = _modelWeights

  override private[ml] def predictRaw(features: Vector): Vector = {
    getAlgo match {
      case "SAMME" => discretePredictRaw(features)
      case "SAMME.R" => realPredictRaw(features)
      case _ =>
        throw new RuntimeException(s"Cannot predict with unknown algorithm: $getAlgo")
    }
  }

  private def realPredictRaw(features: Vector): Vector = {
    val sumProba = Vectors.zeros(numClasses)
    models.foreach { model =>
      BLAS.axpy(1.0, sammeProba(model, features), sumProba)
    }
    BLAS.scal(1.0 / _modelWeights.length.toDouble, sumProba)
    val proba = sumProba.asInstanceOf[DenseVector].values.map { x =>
      math.exp(1.0 / (numClasses - 1.0) * x)
    }
    val probaSum = proba.sum
    new DenseVector(proba.map(_/probaSum))
  }

  private def discretePredictRaw(features: Vector): Vector = {
    val weightSum = _modelWeights.sum
    val prediction = new Array[Double](numClasses)
    for (m <- 0 until _models.length) {
      val rawPrediction = _models(m).predictProbability(features).toArray
      val weight = _modelWeights(m)
      for  (k <- 0 until numClasses) {
        prediction(k) += rawPrediction(k) * weight / weightSum
      }
    }
    val proba = prediction.map { x => math.exp(1.0 / (numClasses - 1.0) * x)}
    val probaSum = proba.sum
    new DenseVector(proba.map(_/probaSum))
  }

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

  override private[ml] def predict(features: Vector): Double = {
    getAlgo match {
      case "SAMME" => discretePredict(features)
      case "SAMME.R" => realPredict(features)
      case _ =>
        throw new RuntimeException(s"Cannot predict with unknown algorithm: $getAlgo")
    }
  }

  private def discretePredict(features: Vector): Double = {
    val predictions = Vectors.zeros(numClasses).asInstanceOf[DenseVector]
    _models.zip(_modelWeights).foreach { case (model, weight) =>
      predictions.values(model.predict(features).toInt) += weight
    }
    predictions.argmax
  }

  private def realPredict(features: Vector): Double = {
    val sumProba = Vectors.zeros(numClasses)
    models.foreach { model =>
      BLAS.axpy(1.0, sammeProba(model, features), sumProba)
    }
    BLAS.scal(_modelWeights.length.toDouble, sumProba)
    sumProba.argmax
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
