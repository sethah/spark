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
//package org.apache.spark.ml.streaming
//
//import org.apache.spark.ml.param.{ParamMap, Param}
//import org.apache.spark.ml.param.shared.HasInputCol
//import org.apache.spark.ml.linalg._
//import org.apache.spark.ml.feature.{MinMaxScalerModel, MinMaxScalerParams}
//import org.apache.spark.ml.util.Identifiable
//import org.apache.spark.sql.{DataFrame, Dataset}
//import org.apache.spark.sql.functions.{min => sqlMin, max => sqlMax, udf, col}
//import org.apache.spark.sql.types.StructType
//
//class StreamingMinMaxScaler(override val uid: String) extends StreamingEstimator
//    with MinMaxScalerParams {
//
//  def this() = this(Identifiable.randomUID("minMaxScal"))
//
//  var model: StreamingMinMaxScalerModel = new StreamingMinMaxScalerModel(uid)
//
//  def getModel: StreamingMinMaxScalerModel = model
//
//  def setInputCol(value: String): this.type = set(inputCol, value)
//  setDefault(inputCol -> "label")
//
//  def update(batch: Dataset[_]): Unit = {
//    val result = batch.agg(sqlMin("label"), sqlMax("label")).first()
//    println(result)
//    model.setMin(result.getDouble(0))
//    model.setMax(result.getDouble(1))
//  }
//  override def copy(extra: ParamMap): StreamingMinMaxScaler = {
//    val copied = new StreamingMinMaxScaler(uid)
//    copyValues(copied, extra)
//  }
//
//  override def transformSchema(schema: StructType): StructType = {
//    validateAndTransformSchema(schema)
//  }
//
//}
//
////class StreamingMinMaxModel(
////                          override val uid: String,
////                          override var originalMin: Vector,
////                          override var originalMax: Vector
////                          ) extends MinMaxScalerModel(uid, originalMin, originalMax) {
////
////}
//
//
//
//class StreamingMinMaxScalerModel(override val uid: String)
//  extends StreamingTransformer
//    with MinMaxScalerParams {
//
//  def update(updates: (Double, Double)): Unit = {
//    setMin(updates._1)
//    setMax(updates._2)
//  }
//
//  /** @group setParam */
//  def setMin(value: Double): this.type = set(min, value)
//  setDefault(min -> 0.0)
//
//  /** @group setParam */
//  def setMax(value: Double): this.type = set(max, value)
//  setDefault(max -> 1.0)
//
//  final def setInputCol(value: String): this.type = set(inputCol, value)
//  setDefault(inputCol -> "label")
//
//  def transform(dataset: Dataset[_]): DataFrame = {
//    val mn = $(min)
//    val mx = $(max)
//    println(mn, mx)
//    val predictUDF = udf((raw: Double) => (raw - mn) / (mx - mn) * (mx - mn) + mn)
//    dataset.withColumn("predicted", predictUDF(col(getInputCol)))
//  }
//
//  override def transformSchema(schema: StructType): StructType = {
//    validateAndTransformSchema(schema)
//  }
//
//  override def copy(extra: ParamMap): StreamingMinMaxScalerModel = {
//    val copied = new StreamingMinMaxScalerModel(uid)
//    copyValues(copied, extra)
//  }
//}
