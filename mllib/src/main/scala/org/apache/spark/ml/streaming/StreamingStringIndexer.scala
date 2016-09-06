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
package org.apache.spark.ml.streaming

import org.apache.spark.SparkException
import org.apache.spark.ml.attribute.NominalAttribute
import org.apache.spark.ml.{Model, Estimator}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.feature.{StringIndexerBase, StringIndexerModel, LabeledPoint}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.param._
import org.apache.spark.sql.types.{StringType, StructType, DoubleType}
import org.apache.spark.util.collection.OpenHashMap

class StreamingStringIndexer(
    override val uid: String) extends Estimator[StreamingStringIndexerModel]
  with StringIndexerBase with StreamingEstimator[Array[String]] {

  def this() = this(Identifiable.randomUID("streamingStrIdx"))

  var model: StreamingStringIndexerModel = new StreamingStringIndexerModel(uid)

  def getModel: StreamingStringIndexerModel = model

  def update(batchId: Long, batch: Dataset[_]): Unit = {
    val counts = batch.select(col($(inputCol)).cast(StringType))
      .rdd
      .map(_.getString(0))
      .countByValue()
    val labels = counts.toSeq.sortBy(-_._2).map(_._1).toArray
    model.update(labels)
  }

  def setHandleInvalid(value: String): this.type = set(handleInvalid, value)
  setDefault(handleInvalid, "error")

  /** @group setParam */
  // TODO: this uh, is a problem here how to copy params to the model
  def setInputCol(value: String): this.type = {
    model.setInputCol(value)
    set(inputCol, value)
  }

  /** @group setParam */
  def setOutputCol(value: String): this.type = {
    model.setOutputCol(value)
    set(outputCol, value)
  }

  override def fit(dataset: Dataset[_]): StreamingStringIndexerModel = {
    transformSchema(dataset.schema, logging = true)
    val counts = dataset.select(col($(inputCol)).cast(StringType))
      .rdd
      .map(_.getString(0))
      .countByValue()
    val labels = counts.toSeq.sortBy(-_._2).map(_._1).toArray
    copyValues(new StreamingStringIndexerModel(uid))
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): StreamingStringIndexer = defaultCopy(extra)
}

class StreamingStringIndexerModel(
    override val uid: String) extends Model[StreamingStringIndexerModel]
  with StreamingModel[Array[String]] with StringIndexerBase {

  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  private val labelToIndex: OpenHashMap[String, Double] = new OpenHashMap[String, Double](10)

  def labels: Array[String] = labelToIndex.iterator.map(_._1).toArray

  def update(updates: Array[String]): Unit = {
    var i = labelToIndex.size
    var j = 0
    while (j < updates.length) {
      val label = updates(j)
      if (!labelToIndex.contains(label)) {
        labelToIndex.update(label, i)
        i += 1
      }
      j += 1
    }
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    if (!dataset.schema.fieldNames.contains($(inputCol))) {
      logInfo(s"Input column ${$(inputCol)} does not exist during transformation. " +
        "Skip StringIndexerModel.")
      return dataset.toDF
    }
    transformSchema(dataset.schema, logging = true)

    val indexer = udf { label: String =>
      if (labelToIndex.contains(label)) {
        labelToIndex(label)
      } else {
        throw new SparkException(s"Unseen label: $label.")
      }
    }

    val metadata = NominalAttribute.defaultAttr
      .withName($(outputCol)).withValues(labels).toMetadata()
    // If we are skipping invalid records, filter them out.
    val filteredDataset = "" match {
      case "skip" =>
        val filterer = udf { label: String =>
          labelToIndex.contains(label)
        }
        dataset.where(filterer(dataset($(inputCol))))
      case _ => dataset
    }
    filteredDataset.select(col("*"),
      indexer(dataset($(inputCol)).cast(StringType)).as($(outputCol), metadata))
  }

  override def transformSchema(schema: StructType): StructType = {
    if (schema.fieldNames.contains($(inputCol))) {
      validateAndTransformSchema(schema)
    } else {
      // If the input column does not exist during transformation, we skip StringIndexerModel.
      schema
    }
  }

  override def copy(extra: ParamMap): StreamingStringIndexerModel = {
    val copied = new StreamingStringIndexerModel(uid)
    copyValues(copied, extra).setParent(parent)
  }
}
