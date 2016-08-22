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

import org.apache.spark.ml.param.{Param, Params}
import org.apache.spark.sql.execution.streaming.Sink
import org.apache.spark.sql.streaming.{StreamingQuery, OutputMode}
import org.apache.spark.sql.{DataFrame, SQLContext, Dataset}
import org.apache.spark.sql.sources.StreamSinkProvider
import org.apache.spark.sql.functions._

//class StreamingPipelineSinkProvider(pipeline: StreamingPipeline) extends StreamSinkProvider {
//  def createSink(
//      sqlContext: SQLContext,
//      parameters: Map[String, String],
//      partitionColumns: Seq[String],
//      outputMode: OutputMode): Sink = {
//    new StreamingPipelineSink(pipeline)
//  }
//}
//
//class StreamingPipelineSink(pipeline: StreamingPipeline) extends Sink {
//  def addBatch(batchId: Long, df: DataFrame): Unit = {
//    val pipeline.update(df)
//  }
//}
//
//
//class StreamingPipelineModelSinkProvider(pipelineModel: StreamingPipelineModel)
//    extends StreamSinkProvider {
//  def createSink(
//      sqlContext: SQLContext,
//      parameters: Map[String, String],
//      partitionColumns: Seq[String],
//      outputMode: OutputMode): Sink = {
//    new StreamingPipelineModelSink(pipelineModel)
//  }
//}
//
//class StreamingPipelineModelSink(pipelineModel: StreamingPipelineModel) extends Sink {
//  def addBatch(batchId: Long, df: DataFrame): Unit = {
//    val transformed = pipelineModel.transform(df)
//    transformed.show()
//  }
//}
//
//class StreamingPipeline {
//
//  var query: StreamingQuery = null
//  val stages = Array((new MyCustomAlgo).asInstanceOf[StreamingAlgo])
//  var model: StreamingPipelineModel = new StreamingPipelineModel(stages.map(_.model))
//  def update(df: DataFrame): Unit = {
//    // for each stage, fit it on the new data, which should update the model
//    // then transform it and repeat for the next stage
//    var curDataset = df
//    stages.zipWithIndex.foreach { case (stage, index) =>
//      stage.update(curDataset)
//      curDataset = model.stages(index).transform(curDataset)
//    }
//  }
//
//  def fitStreaming(dataset: Dataset[_]): StreamingPipelineModel = {
//    require(dataset.isStreaming, "need a streaming dataset")
//
//    query = dataset
//      .writeStream
//      .outputMode("append")
//      .option("checkpointLocation", "/Users/sethhendrickson/StreamingSandbox/checkpoint")
//      .format(new StreamingPipelineSinkProvider(this))
//      .start()
//    model
//  }
//}
//
//class StreamingPipelineModel(val stages: Array[StreamingAlgoModel]) {
//  var query: StreamingQuery = _
//
//  def transform(df: DataFrame): DataFrame = {
//    stages.foldLeft(df)((cur, transformer) => transformer.transform(cur))
//  }
//  def transformStreaming(dataset: Dataset[_]): StreamingQuery = {
//    require(dataset.isStreaming, "need a streaming dataset")
//
//    query = dataset
//      .writeStream
//      .outputMode("append")
//      .option("checkpointLocation", "/Users/sethhendrickson/StreamingSandbox/checkpoint2")
//      .format(new StreamingPipelineModelSinkProvider(this))
//      .start()
//    query
//  }
//}
//
//trait StreamingAlgo {
//  def model: StreamingAlgoModel
//  def update(df: DataFrame): Unit
//}
//trait StreamingAlgoModel {
//  def transform(df: DataFrame): DataFrame
//}
//
//class MyCustomAlgo extends StreamingAlgo {
//  var model: MyCustomAlgoModel = new MyCustomAlgoModel
//
//  def update(df: DataFrame): Unit = {
//    // "train" the model
//    val mean = df.agg(avg("label")).first().getDouble(0)
//    // "update" the model
//    model.multiplier = mean
//    println(model.multiplier)
//  }
//}
//
//class MyCustomAlgoModel extends StreamingAlgoModel {
//  var multiplier = 0.0
//
//  def transform(df: DataFrame): DataFrame = {
//    df.withColumn("prediction", col("label") * multiplier)
//  }
//}
