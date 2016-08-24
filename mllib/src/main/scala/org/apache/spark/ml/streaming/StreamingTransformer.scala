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

import org.apache.spark.ml.param.{ParamMap, Param, Params}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{PipelineStage, Estimator, Transformer}
import org.apache.spark.sql.sources.StreamSinkProvider
import org.apache.spark.sql.{SQLContext, DataFrame, Dataset}
import org.apache.spark.sql.execution.streaming.Sink
import org.apache.spark.sql.streaming.{OutputMode, StreamingQuery}

abstract class StreamingTransformer extends Transformer {


}

trait StreamingModel[S] extends Transformer {
  def update(updates: S): Unit
}

// TODO: inherit estimator?
trait StreamingEstimator[S] extends PipelineStage {

  def model: StreamingModel[S]

  def getModel: StreamingModel[S]

  def update(batch: Dataset[_]): Unit

}


class StreamingPipeline(override val uid: String) extends Params {

  def this() = this(Identifiable.randomUID("streamingPipeline"))

  var query: Option[StreamingQuery] = None

  val stages: Param[Array[PipelineStage]] = new Param(this, "stages",
    "stages of the streaming pipeline")

  def setStages(value: Array[_ <: PipelineStage]): this.type = {
    set(stages, value.asInstanceOf[Array[PipelineStage]])
  }

  def update(dataset: Dataset[_]): Dataset[_] = {
    var curDataset = dataset
    $(stages).foreach {
      case streamingEstimator: StreamingEstimator[_] =>
        streamingEstimator.update(curDataset)
        curDataset = streamingEstimator.model.transform(curDataset)
      case transformer: Transformer =>
        curDataset = transformer.transform(curDataset)
    }
    curDataset
  }

  def fitTransformStreaming(dataset: Dataset[_]): StreamingQuery = {
    require(dataset.isStreaming, "need a streaming dataset")

    query = Some(dataset
      .writeStream
      .outputMode("append")
      .option("checkpointLocation", "/Users/sethhendrickson/StreamingSandbox/checkpoint")
      .format(new StreamingPipelineSinkProvider(this))
      .start()
    )
    query.get
  }

  override def copy(extra: ParamMap): StreamingPipeline = {
    val map = extractParamMap(extra)
    val newStages = map(stages).map(_.copy(extra))
    new StreamingPipeline().setStages(newStages)
  }
}

class StreamingPipelineModel(val stages: Array[Transformer]) {

  var query: Option[StreamingQuery] = None

  def transform(df: DataFrame): DataFrame = {
    stages.foldLeft(df)((cur, transformer) => transformer.transform(cur))
  }

  def transformStreaming(dataset: Dataset[_]): this.type = {
    val q = dataset
      .writeStream
      .outputMode("append")
      .option("checkpointLocation", "/Users/sethhendrickson/StreamingSandbox/checkpoint")
      .format(new StreamingPipelineSinkModelProvider(this))
      .start()
    query = Some(q)
    this
  }
}

// if we have a streaming pipeline model then we'll have a transform method which starts the
// query, then calls transform batch in the sink.

class StreamingPipelineSinkProvider(pipeline: StreamingPipeline) extends StreamSinkProvider {
  def createSink(
      sqlContext: SQLContext,
      parameters: Map[String, String],
      partitionColumns: Seq[String],
      outputMode: OutputMode): Sink = {
    new StreamingPipelineSink(pipeline)
  }
}

class StreamingPipelineSink(pipeline: StreamingPipeline) extends Sink {
  def addBatch(batchId: Long, df: DataFrame): Unit = {
    df.show()
    val transformed = pipeline.update(df)
    transformed.show()
  }
}

class StreamingPipelineSinkModelProvider(pipelineModel: StreamingPipelineModel)
  extends StreamSinkProvider {
  def createSink(
      sqlContext: SQLContext,
      parameters: Map[String, String],
      partitionColumns: Seq[String],
      outputMode: OutputMode): Sink = {
    new StreamingPipelineModelSink(pipelineModel)
  }
}

class StreamingPipelineModelSink(pipelineModel: StreamingPipelineModel) extends Sink {
  def addBatch(batchId: Long, df: DataFrame): Unit = {
    val transformed = pipelineModel.transform(df)
    transformed.show()
  }
}
