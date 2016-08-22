///*
// * Licensed to the Apache Software Foundation (ASF) under one or more
// * contributor license agreements.  See the NOTICE file distributed with
// * this work for additional information regarding copyright ownership.
// * The ASF licenses this file to You under the Apache License, Version 2.0
// * (the "License"); you may not use this file except in compliance with
// * the License.  You may obtain a copy of the License at
// *
// *    http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS,
// * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// * See the License for the specific language governing permissions and
// * limitations under the License.
// */
//
//package org.apache.spark.ml.streaming
//
//import org.apache.hadoop.fs.Path
//
//import org.apache.spark.annotation.{Experimental, Since}
//import org.apache.spark.ml.feature.{MaxAbsScalerParams, MaxAbsScalerModel}
//import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
//import org.apache.spark.ml.{Transformer, Estimator, Model}
//import org.apache.spark.ml.linalg.{Vector, Vectors, VectorUDT}
//import org.apache.spark.ml.param.{ParamMap, Params}
//import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
//import org.apache.spark.ml.util._
//import org.apache.spark.mllib.linalg.{Vector => OldVector, Vectors => OldVectors}
//import org.apache.spark.mllib.stat.Statistics
//import org.apache.spark.rdd.RDD
//import org.apache.spark.sql._
//import org.apache.spark.sql.functions._
//import org.apache.spark.sql.types.{StructField, StructType}
//
///**
// * :: Experimental ::
// * Rescale each feature individually to range [-1, 1] by dividing through the largest maximum
// * absolute value in each feature. It does not shift/center the data, and thus does not destroy
// * any sparsity.
// */
//@Experimental
//@Since("2.0.0")
//class MaxAbsScaler @Since("2.0.0") (@Since("2.0.0") override val uid: String)
//  extends Estimator[MaxAbsScalerModel] with MaxAbsScalerParams with DefaultParamsWritable
//  with StreamingEstimator {
//
//  var model: StreamingMaxAbsScalerModel = new StreamingMaxAbsScalerModel
//
//  def getModel: StreamingMaxAbsScalerModel = model
//
//  def update(batch: Dataset[_]): Unit = {
//
//  }
//
//  @Since("2.0.0")
//  def this() = this(Identifiable.randomUID("maxAbsScal"))
//
//  /** @group setParam */
//  @Since("2.0.0")
//  def setInputCol(value: String): this.type = set(inputCol, value)
//
//  /** @group setParam */
//  @Since("2.0.0")
//  def setOutputCol(value: String): this.type = set(outputCol, value)
//
//  @Since("2.0.0")
//  override def fit(dataset: Dataset[_]): MaxAbsScalerModel = {
//    transformSchema(dataset.schema, logging = true)
//    val input: RDD[OldVector] = dataset.select($(inputCol)).rdd.map {
//      case Row(v: Vector) => OldVectors.fromML(v)
//    }
//    val summary = Statistics.colStats(input)
//    val minVals = summary.min.toArray
//    val maxVals = summary.max.toArray
//    val n = minVals.length
//    val maxAbs = Array.tabulate(n) { i => math.max(math.abs(minVals(i)), math.abs(maxVals(i))) }
//
//    copyValues(new MaxAbsScalerModel(uid, Vectors.dense(maxAbs)).setParent(this))
//  }
//
//  @Since("2.0.0")
//  override def transformSchema(schema: StructType): StructType = {
//    validateAndTransformSchema(schema)
//  }
//
//  @Since("2.0.0")
//  override def copy(extra: ParamMap): MaxAbsScaler = defaultCopy(extra)
//}
//
//@Since("2.0.0")
//object MaxAbsScaler extends DefaultParamsReadable[MaxAbsScaler] {
//
//  @Since("2.0.0")
//  override def load(path: String): MaxAbsScaler = super.load(path)
//}
//
//class StreamingMaxAbsScalerModel extends StreamingTransformer {
//
//  val uid = "asdf"
//
//  def transform(df: Dataset[_]): DataFrame = df.toDF()
//
//  def transformSchema(schema: StructType): StructType = schema
//
//  def copy(extra: ParamMap): Transformer = this
//}
//
//
////class StreamingLinearRegression extends LinearRegression with StreamingEstimator {
////  var model = new StreamingLinearRegressionModel("asd")
////  def getModel: StreamingLinearRegressionModel = model
////  def update(batch: Dataset[_]): Unit = {
////
////  }
////}
////
////class StreamingLinearRegressionModel(override val uid: String)
////  extends LinearRegressionModel(uid, null, 0.0) with StreamingModel[(Vector, Double)] {
////
////  def update(updates: (Vector, Double)): Unit = {
////    _coefficients = updates._1
////    _intercept = updates._2
////  }
////
////}
//
