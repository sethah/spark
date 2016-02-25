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

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.param.shared.{HasThresholds, HasProbabilityCol}
import org.apache.spark.ml.util.SchemaUtils
import org.apache.spark.mllib.linalg.{Vectors, VectorUDT, Vector}
import org.apache.spark.sql.DataFrame
import scala.language.existentials
import org.apache.spark.sql.functions._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.param.{Param, ParamMap, Params}
import org.apache.spark.sql.types.{DataType, StructType}



/**
  * :: DeveloperApi ::
  * TODO
  */
@DeveloperApi
abstract class AdditiveClassifier[
    FeaturesType,
    E <: AdditiveClassifier[FeaturesType, E, M],
    M <: AdditiveClassificationModel[FeaturesType, M]]
  extends ProbabilisticClassifier[FeaturesType, E, M] with AdditiveClassifierParams[FeaturesType] {



}

/**
  * :: DeveloperApi ::
  * TODO
  */
@DeveloperApi
abstract class WeightBoostingClassifier[
    FeaturesType,
    E <: AdditiveClassifier[FeaturesType, E, M],
    M <: AdditiveClassificationModel[FeaturesType, M]]
  extends AdditiveClassifier[FeaturesType, E, M] with AdditiveClassifierParams[FeaturesType] {


}

/**
  * :: DeveloperApi ::
  *
  * TODO
  */
@DeveloperApi
abstract class AdditiveClassificationModel[
    FeaturesType,
    M <: AdditiveClassificationModel[FeaturesType, M]]
  extends ProbabilisticClassificationModel[FeaturesType, M] with AdditiveClassifierParams[FeaturesType] {

}

/**
  * TODO
  */
private[classification] trait AdditiveClassifierParams[FeaturesType]
  extends ProbabilisticClassifierParams with HasMaxIter {

  import AdditiveClassifierParams._

  /** @group setParam */
  def setMaxIter(value: Int): this.type = set(maxIter, value)

}

object AdditiveClassifierParams {
  // scalastyle:off structural.type
  type AdditiveClassifierBaseType[FeaturesType] = ProbabilisticClassifier[FeaturesType, BE, BM]
      forSome {
        type BM <: ProbabilisticClassificationModel[FeaturesType, BM]
        type BE <: ProbabilisticClassifier[FeaturesType, BE, BM]
      }

  type AdditiveClassificationModelBaseType[F] = ProbabilisticClassificationModel[F, BM] forSome {
    type BM <: ProbabilisticClassificationModel[F, BM]
  }
  // scalastyle:on structural.type
}

private[classification] trait WeightBoostingClassifierParams[FeaturesType]
  extends ProbabilisticClassifierParams with HasMaxIter with HasStepSize {

  import org.apache.spark.ml.classification.WeightBoostingClassifierParams._

  val baseEstimators: Param[Array[WeightBoostingClassifierBaseType[FeaturesType]]] =
    new Param(this, "baseEstimators", "")

  def getBaseEstimators: Array[WeightBoostingClassifierBaseType[FeaturesType]] =
    $(baseEstimators)

  def setStepSize(value: Double): this.type = set(stepSize, value)
  setDefault(stepSize -> 1.0  )

}

object WeightBoostingClassifierParams {
  // scalastyle:off structural.type
  type WeightBoostingClassifierBaseType[F] = AdditiveClassifierParams.AdditiveClassifierBaseType[F]

  type WeightBoostingClassificationModelBaseType[F] =
    AdditiveClassifierParams.AdditiveClassificationModelBaseType[F]
  // scalastyle:on structural.type
}
