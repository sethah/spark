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

package org.apache.spark.sql.execution.streaming

import java.util.concurrent.atomic.AtomicInteger

import org.apache.spark.internal.Logging
import org.apache.spark.{SparkEnv, sql}
import org.apache.spark.sql.catalyst.expressions.{Alias, AttributeReference, CurrentBatchTimestamp, ExprId, Literal, UnsafeProjection}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.aggregate.AggregateExpression
import org.apache.spark.sql.catalyst.plans.logical._
import org.apache.spark.sql.catalyst.plans.physical.HashPartitioning
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.aggregate.{HashAggregateExec, ScalaModelUDAF, ScalaUDAF, SortAggregateExec}
import org.apache.spark.sql.execution.{QueryExecution, SparkPlan, SparkPlanner, UnaryExecNode}
import org.apache.spark.sql.expressions.{ModelAgg, MutableAggregationBuffer, SGDAgg}
import org.apache.spark.sql.streaming.OutputMode
import org.apache.spark.sql.types.StringType
import org.apache.spark.storage.StreamingModelBlockId

/**
 * A variant of [[QueryExecution]] that allows the execution of the given [[LogicalPlan]]
 * plan incrementally. Possibly preserving state in between each execution.
 */
class IncrementalExecution(
    sparkSession: SparkSession,
    logicalPlan: LogicalPlan,
    val outputMode: OutputMode,
    val checkpointLocation: String,
    val currentBatchId: Long,
    val currentEventTimeWatermark: Long)
  extends QueryExecution(sparkSession, logicalPlan) with Logging {

  // TODO: make this always part of planning.
  val streamingExtraStrategies =
    sparkSession.sessionState.planner.StatefulAggregationStrategy +:
    sparkSession.sessionState.planner.MapGroupsWithStateStrategy +:
    sparkSession.sessionState.planner.StreamingRelationStrategy +:
      sparkSession.sessionState.planner.MyStrategy +:
    sparkSession.sessionState.experimentalMethods.extraStrategies

  // Modified planner with stateful operations.
  override def planner: SparkPlanner =
    new SparkPlanner(
      sparkSession.sparkContext,
      sparkSession.sessionState.conf,
      streamingExtraStrategies)

  /**
   * See [SPARK-18339]
   * Walk the optimized logical plan and replace CurrentBatchTimestamp
   * with the desired literal
   */
  override lazy val optimizedPlan: LogicalPlan = {
    sparkSession.sessionState.optimizer.execute(withCachedData) transformAllExpressions {
      case ts @ CurrentBatchTimestamp(timestamp, _, _) =>
        logInfo(s"Current batch timestamp = $timestamp")
        ts.toLiteral
    }
  }
  val rng = new scala.util.Random(42)

  /**
   * Records the current id for a given stateful operator in the query plan as the `state`
   * preparation walks the query plan.
   */
  private val operatorId = new AtomicInteger(0)

  /** Locates save/restore pairs surrounding aggregation. */
  val state = new Rule[SparkPlan] {
    override def apply(plan: SparkPlan): SparkPlan = {
      println("PLAN to transform")
      println(plan)
      plan transform {
        case StatefulInit(HashAggregateExec(cd, ge, ae, aa, iibo, re, child)) =>
          val newAggExpressions = ae.map { expr => expr match {
            case AggregateExpression(ScalaModelUDAF(children, ModelAgg(None), mabo, iabo),
            mode, isDistinct, resultId) =>
              println("found the model agg pattern here!!!!")
              val blockId = Some(StreamingModelBlockId(0, currentBatchId - 1))
              AggregateExpression(ScalaModelUDAF(children, ModelAgg(blockId), mabo, iabo),
                mode, isDistinct, resultId)
            case other => other
          }
          }
          HashAggregateExec(cd, ge, newAggExpressions, aa, iibo, re, child)
        case StatefulInit(SortAggregateExec(cd, ge, ae, aa, iibo, re, child)) =>
          // here we look for the stateful initializer node and we replace it with an aggregate
          // that knows to get it's initial state from the block manager
          val newAggExpressions = ae.map { expr => expr match {
            case AggregateExpression(ScalaModelUDAF(children, SGDAgg(numFeatures, None), mabo, iabo),
            mode, isDistinct, resultId) =>
              println("found the SGD AGG pattern here!!!!")
              val blockId = Some(StreamingModelBlockId(0, currentBatchId - 1))
              AggregateExpression(ScalaModelUDAF(children, SGDAgg(numFeatures, blockId), mabo, iabo),
                mode, isDistinct, resultId)
            case other => other
          }
          }
          SortAggregateExec(cd, ge, newAggExpressions, aa, iibo, re, child)
        case StateStoreSaveExec(keys, None, None, None,
        UnaryExecNode(agg,
        StateStoreRestoreExec(keys2, None, child))) =>
          val stateId =
            OperatorStateId(checkpointLocation, operatorId.getAndIncrement(), currentBatchId)

          val newKeys = keys.map {
            case a @ AttributeReference(name, dtype, nullable, metadata) =>
              println("FOUND THE ATTRIBUTE REFRERENCE")
              AttributeReference("model", dtype, nullable,
                metadata)(a.exprId, a.qualifier, a.isGenerated)
            case other => other
          }
          println("AGGGGGG", agg)
          val newAgg = agg match {
            case HashAggregateExec(cd, ge, ae, aa, iibo, re, c) =>
              HashAggregateExec(cd, newKeys, ae, aa, iibo, re, c)
            case other => other
          }

          StateStoreSaveExec(
            keys,
            Some(stateId),
            Some(outputMode),
            Some(currentEventTimeWatermark),
            agg.withNewChildren(
              StateStoreRestoreExec(
                keys,
                Some(stateId),
                child) :: Nil))
        case HashAggregateExec(cd, ge, ae, aa, iibo, re, c) =>
          val newGroupings = ge.map {
            case al @ Alias(ch, name) =>
              Alias(Literal("model" + currentBatchId), name)(al.exprId, al.qualifier,
                al.explicitMetadata, al.isGenerated)
            case other => other
          }
          HashAggregateExec(cd, newGroupings, ae, aa, iibo, re, c)
        case ModelStateStoreRestoreExec(cd, ge, ae, aa, iibo, re, None, spart, c) =>
          val newGroupings = ge.map {
            case al @ Alias(l: Literal, name) =>
              Alias(Literal("model" + currentBatchId), name)(al.exprId, al.qualifier,
                al.explicitMetadata, al.isGenerated)
            case other => other
          }
          val stateId =
            OperatorStateId(checkpointLocation, operatorId.getAndIncrement(), currentBatchId)
          ModelStateStoreRestoreExec(cd, newGroupings, ae, aa, iibo, re, Some(stateId),
            0, c)
        case StateStoreSaveExec(keys, None, None, None, child) =>
          val stateId =
            OperatorStateId(checkpointLocation, operatorId.get(), currentBatchId)
          StateStoreSaveExec(keys, Some(stateId), Some(outputMode), Some(currentEventTimeWatermark),
            child)
        case StatefulAggExec(None, child) =>
          val stateId =
            OperatorStateId(checkpointLocation, operatorId.getAndIncrement(), currentBatchId)
          StatefulAggExec(Some(stateId), child)
        case StatefulAggSaveExec(keys, child, None) =>
          val stateId =
            OperatorStateId(checkpointLocation, operatorId.getAndIncrement(), currentBatchId)
          StatefulAggSaveExec(keys, child, Some(stateId))

        case MapGroupsWithStateExec(
        f, kDeser, vDeser, group, data, output, None, stateDeser, stateSer, child) =>
          val stateId =
            OperatorStateId(checkpointLocation, operatorId.getAndIncrement(), currentBatchId)
          MapGroupsWithStateExec(
            f, kDeser, vDeser, group, data, output, Some(stateId), stateDeser, stateSer, child)
      }
    }
  }

  override def preparations: Seq[Rule[SparkPlan]] = state +: super.preparations

  /** No need assert supported, as this check has already been done */
  override def assertSupported(): Unit = { }
}


class IncrementalModelExecution(
                            sparkSession: SparkSession,
                            logicalPlan: LogicalPlan)
  extends QueryExecution(sparkSession, logicalPlan) with Logging {

  // TODO: make this always part of planning.
  val streamingExtraStrategies: Seq[sql.Strategy] =
    sparkSession.sessionState.planner.MyStrategy +:
      sparkSession.sessionState.experimentalMethods.extraStrategies

//  val myrule = new Rule[SparkPlan] {

//    override def apply(plan: SparkPlan): SparkPlan = plan transform {
//      case ModelA
//    }
//  }

  override def planner: SparkPlanner =
    new SparkPlanner(
      sparkSession.sparkContext,
      sparkSession.sessionState.conf,
      streamingExtraStrategies)


}