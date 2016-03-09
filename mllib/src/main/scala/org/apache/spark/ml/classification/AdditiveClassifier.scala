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

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.param.shared._
import org.apache.spark.mllib.linalg.Vector

/**
 * :: DeveloperApi ::
 *
 * Single-label binary or multiclass classifier which is composed of one or more base classifiers.
 * Output classfications are generally an aggregation of the base classifications.
 *
 * @tparam FeaturesType  Type of input features.  E.g., [[Vector]]
 * @tparam E  Concrete Estimator type
 * @tparam M  Concrete Model type
 */
@DeveloperApi
abstract class AdditiveClassifier[
    FeaturesType,
    E <: AdditiveClassifier[FeaturesType, E, M],
    M <: AdditiveClassificationModel[FeaturesType, M]]
  extends ProbabilisticClassifier[FeaturesType, E, M] with AdditiveClassifierParams[FeaturesType] {

  /**
   * Get a boosting base learner for a single boosting iteration.
   */
  protected def makeLearner(): ProbabilisticClassifier[FeaturesType, _, _]

}

/**
 * :: DeveloperApi ::
 *
 * Model produced by a [[AdditiveClassifier]].
 *
 * @tparam FeaturesType  Type of input features.  E.g., [[Vector]]
 * @tparam M  Concrete Model type
 */
@DeveloperApi
abstract class AdditiveClassificationModel[
    FeaturesType,
    M <: AdditiveClassificationModel[FeaturesType, M]]
  extends ProbabilisticClassificationModel[FeaturesType, M]
    with AdditiveClassifierParams[FeaturesType] {

  def models: Array[_ <: ProbabilisticClassificationModel[FeaturesType, _]]

  def modelWeights: Array[Double]
}

/**
 * :: DeveloperApi ::
 *
 * An additive classifier which is fit by iteratively boosting [[ProbabilisticClassifier]]s
 * that can accept sample weights.
 *
 * @tparam FeaturesType  Type of input features.  E.g., [[Vector]]
 * @tparam E  Concrete Estimator type
 * @tparam M  Concrete Model type
 */
@DeveloperApi
abstract class WeightBoostingClassifier[
    FeaturesType,
    E <: AdditiveClassifier[FeaturesType, E, M],
    M <: AdditiveClassificationModel[FeaturesType, M]]
  extends AdditiveClassifier[FeaturesType, E, M] with WeightBoostingParams[FeaturesType] {

  /**
   * Get a boosting base learner for a single boosting iteration.
   *
   * Only a single candidate base learner is currently supported.
   */
  override protected def makeLearner(): BaseEstimatorType[FeaturesType] = {
    getBaseEstimators.head.copy(ParamMap.empty)
  }
}

/**
 * :: DeveloperApi ::
 *
 * Model produced by a [[WeightBoostingClassifier]].
 *
 * @tparam FeaturesType  Type of input features.  E.g., [[Vector]]
 * @tparam M  Concrete Model type
 */
@DeveloperApi
abstract class WeightBoostingClassificationModel[
    FeaturesType,
    M <: AdditiveClassificationModel[FeaturesType, M]]
  extends AdditiveClassificationModel[FeaturesType, M] with WeightBoostingParams[FeaturesType] {

}

/**
 * (private[classification])  Params for additive classification.
 */
private[classification] trait AdditiveClassifierParams[FeaturesType]
  extends ProbabilisticClassifierParams with HasMaxIter with HasStepSize
    with HasCheckpointInterval {

  /** @group setParam */
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  /** @group setParam */
  def setStepSize(value: Double): this.type = set(stepSize, value)
  setDefault(stepSize -> 1.0)

  /** @group setParam */
  def setCheckpointInterval(value: Int): this.type = set(checkpointInterval, value)
  setDefault(checkpointInterval -> 10)

}

/**
 * (private[classification])  Params for weighted boosted classification.
 */
private[classification] trait WeightBoostingParams[FeaturesType]
  extends AdditiveClassifierParams[FeaturesType] with HasWeightCol {

  /**
   * The candidate base estimators to be chosen from at each boosting iteration.
   * (default = Array(DecisionTreeClassifier).
   *
   * @group param
   */
  val baseEstimators: Param[Set[BaseEstimatorType[FeaturesType]]] =
    new Param(this, "baseEstimators", "The set of candidate base learners to be chosen from at " +
      "each boosting iteration.")

  /** @group getParam */
  def getBaseEstimators: Set[BaseEstimatorType[FeaturesType]] =
    $(baseEstimators)

  /** @group setParam */
  def setWeightCol(value: String): this.type = set(weightCol, value)
  setDefault(weightCol -> "")

  // scalastyle:off structural.type
  type BaseEstimatorType[F] = ProbabilisticClassifier[F, BE, BM] with HasWeightCol forSome {
    type BM <: ProbabilisticClassificationModel[F, BM]
    type BE <: ProbabilisticClassifier[F, BE, BM] with HasWeightCol
  }

  type BaseTransformerType[F] = ProbabilisticClassificationModel[F, BM] forSome {
    type BM <: ProbabilisticClassificationModel[F, BM]
  }
  // scalastyle:on structural.type
}
