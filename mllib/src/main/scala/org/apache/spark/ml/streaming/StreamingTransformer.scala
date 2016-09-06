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


import org.apache.hadoop.fs.Path
import org.apache.spark.ml.param.{ParamMap, Param, Params}
import org.apache.spark.ml.util.{MLWritable, Identifiable}
import org.apache.spark.ml.{PipelineStage, Estimator, Transformer}
import org.apache.spark.sql.sources.StreamSinkProvider
import org.apache.spark.sql._
import org.apache.spark.sql.execution.streaming.Sink
import org.apache.spark.sql.streaming.{OutputMode, StreamingQuery}
import org.apache.spark.sql.types.StructType

abstract class StreamingTransformer extends Transformer {


}

trait StreamingModel[S] extends Transformer {
  def update(updates: S): Unit
}

// TODO: inherit estimator?
trait StreamingEstimator[S] extends PipelineStage {

  def model: StreamingModel[S]

  def getModel: StreamingModel[S]

  def update(batchId: Long, batch: Dataset[_]): Unit

}


class StreamingPipeline(override val uid: String) extends Params {

  def this() = this(Identifiable.randomUID("streamingPipeline"))

  var query: Option[StreamingQuery] = None

  val stages: Param[Array[PipelineStage]] = new Param(this, "stages",
    "stages of the streaming pipeline")

  var model: StreamingPipelineModel = null

  def setStages(value: Array[_ <: PipelineStage]): this.type = {
    set(stages, value.asInstanceOf[Array[PipelineStage]])
  }

  val checkpointLocation: Param[String] = new Param[String](this, "checkpointLocation",
    "the checkpoint directory to use for this stream")

  def getCheckpointLocation: String = $(checkpointLocation)

  def setCheckpointLocation(value: String): this.type = set(checkpointLocation, value)

  // TODO: this should be synchronized so that we cannot grab the model while updating.
  def update(batchId: Long, dataset: Dataset[_]): Dataset[_] = {
    // TODO: we could return unit here, and not pipe df through to the end
    var curDataset = dataset
    $(stages).foreach {
      case streamingEstimator: StreamingEstimator[_] =>
        streamingEstimator.update(batchId, curDataset)
        curDataset = streamingEstimator.model.transform(curDataset)
      case transformer: Transformer =>
        curDataset = transformer.transform(curDataset)
    }
    curDataset
  }

  def fitStreaming(dataset: Dataset[_]): StreamingQuery = {
    require(dataset.isStreaming, "need a streaming dataset")

    // TODO: need to initialize the model here

    val transformerStages = $(stages).map {
      case streamingEstimator: StreamingEstimator[_] => streamingEstimator.getModel
      case transformer: Transformer => transformer
    }

    model = new StreamingPipelineModel(uid, transformerStages)

    query = Some(dataset
      .writeStream
      .outputMode("append")
      .option("checkpointLocation", getCheckpointLocation)
      .format(new StreamingPipelineSinkProvider(this))
      .start()
    )
    query.get
  }

  override def copy(extra: ParamMap): StreamingPipeline = {
    // TODO: copy checkpoint location
    val map = extractParamMap(extra)
    val newStages = map(stages).map(_.copy(extra))
    new StreamingPipeline().setStages(newStages)
  }
}

class StreamingPipelineModel(
    override val uid: String,
    val stages: Array[Transformer]) extends Params {

  val checkpointLocation: Param[String] = new Param[String](this, "checkpointLocation",
    "the checkpoint directory to use for this stream")

  def getCheckpointLocation: String = $(checkpointLocation)

  def setCheckpointLocation(value: String): this.type = set(checkpointLocation, value)

  var query: Option[StreamingQuery] = None

  def transform(df: DataFrame): DataFrame = {
    stages.foldLeft(df)((cur, transformer) => transformer.transform(cur))
  }

  def transformStreaming(dataset: Dataset[_]): StreamingQuery = {
    val q = dataset
      .writeStream
      .outputMode("append")
      .option("checkpointLocation", getCheckpointLocation)
      .format(new StreamingPipelineModelSinkProvider(this))
      .start()
    query = Some(q)
    q
  }

  def copy(extra: ParamMap): StreamingPipelineModel = {
    // TODO: shallow or deep?
    val map = extractParamMap(extra)
    new StreamingPipelineModel(uid, stages)
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
    val transformed = pipeline.update(batchId, df)
    transformed.show()
  }
}

class StreamingPipelineModelSinkProvider(pipelineModel: StreamingPipelineModel)
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

//class TempTableSinkProvider(tableName: String, spark: SparkSession, schema: StructType)
//  extends StreamSinkProvider {
//  def createSink(
//                  sqlContext: SQLContext,
//                  parameters: Map[String, String],
//                  partitionColumns: Seq[String],
//                  outputMode: OutputMode): Sink = {
//    new TempTableSink(tableName, schema, spark)
//  }
//}
//
//class TempTableSink(tableName: String, schema: StructType, spark: SparkSession) extends Sink {
//  val empty = spark.createDataFrame(spark.sparkContext.emptyRDD[Row], schema)
//  empty.createOrReplaceTempView(tableName)
//
//  def addBatch(batchId: Long, df: DataFrame): Unit = {
//    println("batch being added now!")
////    println("adding batch")
//    Thread.sleep(1000)
//    df.createOrReplaceTempView(tableName)
//  }
//}
