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

package org.apache.spark.sql.execution.aggregate

import org.apache.spark.TaskContext
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.aggregate._
import org.apache.spark.sql.catalyst.plans.physical._
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.exchange.BroadcastExchangeExec
import org.apache.spark.sql.execution.streaming._
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types.{DataType, IntegerType, StringType}
import org.apache.spark.unsafe.types.UTF8String

import scala.util.Random

/**
 * Utility functions used by the query planner to convert our plan to new aggregation code path.
 */
object AggUtils {
  private def createAggregate(
      requiredChildDistributionExpressions: Option[Seq[Expression]] = None,
      groupingExpressions: Seq[NamedExpression] = Nil,
      aggregateExpressions: Seq[AggregateExpression] = Nil,
      aggregateAttributes: Seq[Attribute] = Nil,
      initialInputBufferOffset: Int = 0,
      resultExpressions: Seq[NamedExpression] = Nil,
      child: SparkPlan,
      getModel: Boolean = false): SparkPlan = {
    // TODO: notes to self
    /*
      We use this function to "plan" the aggregate, which just means create a physical ndoe that
      does some part of the aggregation. In our streaming case, we want to modify how the agg
      buffer is initialized so that it gets the previous model from the block manager or from
      the state store somehow. One way to do this is to create a rule in the incremental execution
      that finds our initial aggregation phase and passes in a block id to get from the block store.
      You have to leave that as a placeholder here, because we won't know the block id until
      the stream is running. Problem is that it might be a hash agg exec we are looking for or
      sort or something else. We would like a single place to modify this I think.


     */
    // NOTE: this checks to make sure that all the types in the aggregation are
    // mutable. If not, we use a sort agg.
    val useHash = HashAggregateExec.supportsAggregate(
      aggregateExpressions.flatMap(_.aggregateFunction.aggBufferAttributes))
    if (useHash) {
      HashAggregateExec(
        requiredChildDistributionExpressions = requiredChildDistributionExpressions,
        groupingExpressions = groupingExpressions,
        aggregateExpressions = aggregateExpressions,
        aggregateAttributes = aggregateAttributes,
        initialInputBufferOffset = initialInputBufferOffset,
        resultExpressions = resultExpressions,
        child = child)
    } else {
      val objectHashEnabled = child.sqlContext.conf.useObjectHashAggregation
      val useObjectHash = ObjectHashAggregateExec.supportsAggregate(aggregateExpressions)

      if (objectHashEnabled && useObjectHash) {
        ObjectHashAggregateExec(
          requiredChildDistributionExpressions = requiredChildDistributionExpressions,
          groupingExpressions = groupingExpressions,
          aggregateExpressions = aggregateExpressions,
          aggregateAttributes = aggregateAttributes,
          initialInputBufferOffset = initialInputBufferOffset,
          resultExpressions = resultExpressions,
          child = child)
      } else {
        SortAggregateExec(
          requiredChildDistributionExpressions = requiredChildDistributionExpressions,
          groupingExpressions = groupingExpressions,
          aggregateExpressions = aggregateExpressions,
          aggregateAttributes = aggregateAttributes,
          initialInputBufferOffset = initialInputBufferOffset,
          resultExpressions = resultExpressions,
          child = child)
      }
    }
  }

  def planAggregateWithoutDistinct(
      groupingExpressions: Seq[NamedExpression],
      aggregateExpressions: Seq[AggregateExpression],
      resultExpressions: Seq[NamedExpression],
      child: SparkPlan): Seq[SparkPlan] = {
    // Check if we can use HashAggregate.

    // 1. Create an Aggregate Operator for partial aggregations.

    val groupingAttributes = groupingExpressions.map(_.toAttribute)
    // partial aggs are just copies of the complete aggs with the mode changed to partial
    val partialAggregateExpressions = aggregateExpressions.map(_.copy(mode = Partial))
    println("partial aggs", partialAggregateExpressions.head.getClass().getName())
    val partialAggregateAttributes =
      partialAggregateExpressions.flatMap(_.aggregateFunction.aggBufferAttributes)
    val partialResultExpressions =
      groupingAttributes ++
        partialAggregateExpressions.flatMap(_.aggregateFunction.inputAggBufferAttributes)

    val partialAggregate = createAggregate(
        requiredChildDistributionExpressions = None,
        groupingExpressions = groupingExpressions,
        aggregateExpressions = partialAggregateExpressions,
        aggregateAttributes = partialAggregateAttributes,
        initialInputBufferOffset = 0,
        resultExpressions = partialResultExpressions,
        child = child)

    // 2. Create an Aggregate Operator for final aggregations.
    val finalAggregateExpressions = aggregateExpressions.map(_.copy(mode = Final))
    // The attributes of the final aggregation buffer, which is presented as input to the result
    // projection:
    val finalAggregateAttributes = finalAggregateExpressions.map(_.resultAttribute)
    println("plan aggregate")
    println(partialAggregate)
    partialAggregate.children.foreach { println }

    val finalAggregate = createAggregate(
        requiredChildDistributionExpressions = Some(groupingAttributes),
        groupingExpressions = groupingAttributes,
        aggregateExpressions = finalAggregateExpressions,
        aggregateAttributes = finalAggregateAttributes,
        initialInputBufferOffset = groupingExpressions.length,
        resultExpressions = resultExpressions,
        child = partialAggregate)

    println("plan aggregate")
    println(finalAggregate.getClass().getName())
    println(finalAggregate.treeString(true))
//    finalAggregate.children.foreach { println }

    finalAggregate :: Nil
  }

  def planAggregateWithOneDistinct(
      groupingExpressions: Seq[NamedExpression],
      functionsWithDistinct: Seq[AggregateExpression],
      functionsWithoutDistinct: Seq[AggregateExpression],
      resultExpressions: Seq[NamedExpression],
      child: SparkPlan): Seq[SparkPlan] = {

    // functionsWithDistinct is guaranteed to be non-empty. Even though it may contain more than one
    // DISTINCT aggregate function, all of those functions will have the same column expressions.
    // For example, it would be valid for functionsWithDistinct to be
    // [COUNT(DISTINCT foo), MAX(DISTINCT foo)], but [COUNT(DISTINCT bar), COUNT(DISTINCT foo)] is
    // disallowed because those two distinct aggregates have different column expressions.
    val distinctExpressions = functionsWithDistinct.head.aggregateFunction.children
    val namedDistinctExpressions = distinctExpressions.map {
      case ne: NamedExpression => ne
      case other => Alias(other, other.toString)()
    }
    val distinctAttributes = namedDistinctExpressions.map(_.toAttribute)
    val groupingAttributes = groupingExpressions.map(_.toAttribute)

    // 1. Create an Aggregate Operator for partial aggregations.
    val partialAggregate: SparkPlan = {
      val aggregateExpressions = functionsWithoutDistinct.map(_.copy(mode = Partial))
      val aggregateAttributes = aggregateExpressions.map(_.resultAttribute)
      // We will group by the original grouping expression, plus an additional expression for the
      // DISTINCT column. For example, for AVG(DISTINCT value) GROUP BY key, the grouping
      // expressions will be [key, value].
      createAggregate(
        groupingExpressions = groupingExpressions ++ namedDistinctExpressions,
        aggregateExpressions = aggregateExpressions,
        aggregateAttributes = aggregateAttributes,
        resultExpressions = groupingAttributes ++ distinctAttributes ++
          aggregateExpressions.flatMap(_.aggregateFunction.inputAggBufferAttributes),
        child = child)
    }

    // 2. Create an Aggregate Operator for partial merge aggregations.
    val partialMergeAggregate: SparkPlan = {
      val aggregateExpressions = functionsWithoutDistinct.map(_.copy(mode = PartialMerge))
      val aggregateAttributes = aggregateExpressions.map(_.resultAttribute)
      createAggregate(
        requiredChildDistributionExpressions =
          Some(groupingAttributes ++ distinctAttributes),
        groupingExpressions = groupingAttributes ++ distinctAttributes,
        aggregateExpressions = aggregateExpressions,
        aggregateAttributes = aggregateAttributes,
        initialInputBufferOffset = (groupingAttributes ++ distinctAttributes).length,
        resultExpressions = groupingAttributes ++ distinctAttributes ++
          aggregateExpressions.flatMap(_.aggregateFunction.inputAggBufferAttributes),
        child = partialAggregate)
    }

    // 3. Create an Aggregate operator for partial aggregation (for distinct)
    val distinctColumnAttributeLookup = distinctExpressions.zip(distinctAttributes).toMap
    val rewrittenDistinctFunctions = functionsWithDistinct.map {
      // Children of an AggregateFunction with DISTINCT keyword has already
      // been evaluated. At here, we need to replace original children
      // to AttributeReferences.
      case agg @ AggregateExpression(aggregateFunction, mode, true, _) =>
        aggregateFunction.transformDown(distinctColumnAttributeLookup)
          .asInstanceOf[AggregateFunction]
    }

    val partialDistinctAggregate: SparkPlan = {
      val mergeAggregateExpressions = functionsWithoutDistinct.map(_.copy(mode = PartialMerge))
      // The attributes of the final aggregation buffer, which is presented as input to the result
      // projection:
      val mergeAggregateAttributes = mergeAggregateExpressions.map(_.resultAttribute)
      val (distinctAggregateExpressions, distinctAggregateAttributes) =
        rewrittenDistinctFunctions.zipWithIndex.map { case (func, i) =>
          // We rewrite the aggregate function to a non-distinct aggregation because
          // its input will have distinct arguments.
          // We just keep the isDistinct setting to true, so when users look at the query plan,
          // they still can see distinct aggregations.
          val expr = AggregateExpression(func, Partial, isDistinct = true)
          // Use original AggregationFunction to lookup attributes, which is used to build
          // aggregateFunctionToAttribute
          val attr = functionsWithDistinct(i).resultAttribute
          (expr, attr)
      }.unzip

      val partialAggregateResult = groupingAttributes ++
          mergeAggregateExpressions.flatMap(_.aggregateFunction.inputAggBufferAttributes) ++
          distinctAggregateExpressions.flatMap(_.aggregateFunction.inputAggBufferAttributes)
      createAggregate(
        groupingExpressions = groupingAttributes,
        aggregateExpressions = mergeAggregateExpressions ++ distinctAggregateExpressions,
        aggregateAttributes = mergeAggregateAttributes ++ distinctAggregateAttributes,
        initialInputBufferOffset = (groupingAttributes ++ distinctAttributes).length,
        resultExpressions = partialAggregateResult,
        child = partialMergeAggregate)
    }

    // 4. Create an Aggregate Operator for the final aggregation.
    val finalAndCompleteAggregate: SparkPlan = {
      val finalAggregateExpressions = functionsWithoutDistinct.map(_.copy(mode = Final))
      // The attributes of the final aggregation buffer, which is presented as input to the result
      // projection:
      val finalAggregateAttributes = finalAggregateExpressions.map(_.resultAttribute)

      val (distinctAggregateExpressions, distinctAggregateAttributes) =
        rewrittenDistinctFunctions.zipWithIndex.map { case (func, i) =>
          // We rewrite the aggregate function to a non-distinct aggregation because
          // its input will have distinct arguments.
          // We just keep the isDistinct setting to true, so when users look at the query plan,
          // they still can see distinct aggregations.
          val expr = AggregateExpression(func, Final, isDistinct = true)
          // Use original AggregationFunction to lookup attributes, which is used to build
          // aggregateFunctionToAttribute
          val attr = functionsWithDistinct(i).resultAttribute
          (expr, attr)
      }.unzip

      createAggregate(
        requiredChildDistributionExpressions = Some(groupingAttributes),
        groupingExpressions = groupingAttributes,
        aggregateExpressions = finalAggregateExpressions ++ distinctAggregateExpressions,
        aggregateAttributes = finalAggregateAttributes ++ distinctAggregateAttributes,
        initialInputBufferOffset = groupingAttributes.length,
        resultExpressions = resultExpressions,
        child = partialDistinctAggregate)
    }

    finalAndCompleteAggregate :: Nil
  }

  /**
   * Plans a streaming aggregation using the following progression:
   *  - Partial Aggregation
   *  - Shuffle
   *  - Partial Merge (now there is at most 1 tuple per group)
   *  - StateStoreRestore (now there is 1 tuple from this batch + optionally one from the previous)
   *  - PartialMerge (now there is at most 1 tuple per group)
   *  - StateStoreSave (saves the tuple for the next batch)
   *  - Complete (output the current result of the aggregation)
   */
  def planStreamingAggregation(
      groupingExpressions: Seq[NamedExpression],
      functionsWithoutDistinct: Seq[AggregateExpression],
      resultExpressions: Seq[NamedExpression],
      child: SparkPlan): Seq[SparkPlan] = {
    println("planning!!!")
    println(child)

    val groupingAttributes = groupingExpressions.map(_.toAttribute)

    val partialAggregate: SparkPlan = {
      val aggregateExpressions = functionsWithoutDistinct.map(_.copy(mode = Partial))
      val aggregateAttributes = aggregateExpressions.map(_.resultAttribute)
      // We will group by the original grouping expression, plus an additional expression for the
      // DISTINCT column. For example, for AVG(DISTINCT value) GROUP BY key, the grouping
      // expressions will be [key, value].
      createAggregate(
        groupingExpressions = groupingExpressions,
        aggregateExpressions = aggregateExpressions,
        aggregateAttributes = aggregateAttributes,
        resultExpressions = groupingAttributes ++
            aggregateExpressions.flatMap(_.aggregateFunction.inputAggBufferAttributes),
        child = child)
    }

    val partialMerged1: SparkPlan = {
      val aggregateExpressions = functionsWithoutDistinct.map(_.copy(mode = PartialMerge))
      val aggregateAttributes = aggregateExpressions.map(_.resultAttribute)
      createAggregate(
        requiredChildDistributionExpressions =
            Some(groupingAttributes),
        groupingExpressions = groupingAttributes,
        aggregateExpressions = aggregateExpressions,
        aggregateAttributes = aggregateAttributes,
        initialInputBufferOffset = groupingAttributes.length,
        resultExpressions = groupingAttributes ++
            aggregateExpressions.flatMap(_.aggregateFunction.inputAggBufferAttributes),
        child = partialAggregate)
    }

    val restored = StateStoreRestoreExec(groupingAttributes, None, partialMerged1)

    val partialMerged2: SparkPlan = {
      val aggregateExpressions = functionsWithoutDistinct.map(_.copy(mode = PartialMerge))
      val aggregateAttributes = aggregateExpressions.map(_.resultAttribute)
      createAggregate(
        requiredChildDistributionExpressions =
            Some(groupingAttributes),
        groupingExpressions = groupingAttributes,
        aggregateExpressions = aggregateExpressions,
       aggregateAttributes = aggregateAttributes,
        initialInputBufferOffset = groupingAttributes.length,
        resultExpressions = groupingAttributes ++
            aggregateExpressions.flatMap(_.aggregateFunction.inputAggBufferAttributes),
        child = restored)
    }
    // Note: stateId and returnAllStates are filled in later with preparation rules
    // in IncrementalExecution.
    val saved =
      StateStoreSaveExec(
        groupingAttributes,
        stateId = None,
        outputMode = None,
        eventTimeWatermark = None,
        partialMerged2)


    val finalAndCompleteAggregate: SparkPlan = {
      val finalAggregateExpressions = functionsWithoutDistinct.map(_.copy(mode = Final))
      // The attributes of the final aggregation buffer, which is presented as input to the result
      // projection:
      val finalAggregateAttributes = finalAggregateExpressions.map(_.resultAttribute)

      createAggregate(
        requiredChildDistributionExpressions = Some(groupingAttributes),
        groupingExpressions = groupingAttributes,
        aggregateExpressions = finalAggregateExpressions,
        aggregateAttributes = finalAggregateAttributes,
        initialInputBufferOffset = groupingAttributes.length,
        resultExpressions = resultExpressions,
        child = saved)
    }

    finalAndCompleteAggregate :: Nil
  }

  def planModelStreamingAggregation(
                                  groupingExpressions: Seq[NamedExpression],
                                  functionsWithoutDistinct: Seq[AggregateExpression],
                                  resultExpressions: Seq[NamedExpression],
                                  child: SparkPlan): Seq[SparkPlan] = {

    println("PLAN STREAMING MODEL AGGREGATION")
    val groupingAttributes = groupingExpressions.map(_.toAttribute)
    // idea: make a phys plan inject keys which just maps the rows to keyed rows
    // then change the grouping expressions to be the key

    // idea: put the aggregate inside of a placeholder node, then pattern match on that as a
    // strategy: case DummyExec(HashAggregateExec(_)) => HashAggregateExec(batchId)
    // ya! try it out
    println("GROUPING", groupingExpressions.mkString(","))

//    val partialAggregate: SparkPlan = {
//      val aggregateExpressions = functionsWithoutDistinct.map(_.copy(mode = Partial))
//      val aggregateAttributes = aggregateExpressions.map(_.resultAttribute)
//      // We will group by the original grouping expression, plus an additional expression for the
//      // DISTINCT column. For example, for AVG(DISTINCT value) GROUP BY key, the grouping
//      // expressions will be [key, value].
//      createAggregate(
//        groupingExpressions = groupingExpressions,
//        aggregateExpressions = aggregateExpressions,
//        aggregateAttributes = aggregateAttributes,
//        resultExpressions = groupingAttributes ++
//          aggregateExpressions.flatMap(_.aggregateFunction.inputAggBufferAttributes),
//        child = child)
//    }
//    private def createAggregate(
//                                 requiredChildDistributionExpressions: Option[Seq[Expression]] = None,
//                                 groupingExpressions: Seq[NamedExpression] = Nil,
//                                 aggregateExpressions: Seq[AggregateExpression] = Nil,
//                                 aggregateAttributes: Seq[Attribute] = Nil,
//                                 initialInputBufferOffset: Int = 0,
//                                 resultExpressions: Seq[NamedExpression] = Nil,
//                                 child: SparkPlan,
//                                 getModel: Boolean = false): SparkPlan = {


    val partialAggregate: SparkPlan = {
      val aggregateExpressions = functionsWithoutDistinct.map(_.copy(mode = Partial))
      val aggregateAttributes = aggregateExpressions.map(_.resultAttribute)
      ModelStateStoreRestoreExec(None, groupingExpressions, aggregateExpressions,
        aggregateAttributes, 0, groupingAttributes ++
          aggregateExpressions.flatMap(_.aggregateFunction.inputAggBufferAttributes),
        None, -1, child = child)
    }

    println("OUTPUT", partialAggregate.output.mkString(","))


    val partialMerged1: SparkPlan = {
      val aggregateExpressions = functionsWithoutDistinct.map(_.copy(mode = PartialMerge))
      val aggregateAttributes = aggregateExpressions.map(_.resultAttribute)
      createAggregate(
        requiredChildDistributionExpressions =
          Some(groupingAttributes),
        groupingExpressions = groupingAttributes,
        aggregateExpressions = aggregateExpressions,
        aggregateAttributes = aggregateAttributes,
        initialInputBufferOffset = groupingAttributes.length,
        resultExpressions = groupingAttributes ++
          aggregateExpressions.flatMap(_.aggregateFunction.inputAggBufferAttributes),
        child = partialAggregate)
    }

//    val dist = partialMerged1.requiredChildDistribution
//    def createPartitioning(
//                                    requiredDistribution: Distribution,
//                                    numPartitions: Int): Partitioning = {
//      requiredDistribution match {
//        case AllTuples => SinglePartition
//        case ClusteredDistribution(clustering) => HashPartitioning(clustering, numPartitions)
//        case OrderedDistribution(ordering) => RangePartitioning(ordering, numPartitions)
//        case dist => sys.error(s"Do not know how to satisfy distribution $dist")
//      }
//    }
//    val maxChildrenNumPartitions =
//      partialMerged1.children.map(_.outputPartitioning.numPartitions).max
//    val numPartitions = {
//      // Let's see if we need to shuffle all child's outputs when we use
//      // maxChildrenNumPartitions.
//      val shufflesAllChildren = partialMerged1.children.zip(dist).forall {
//        case (child, distribution) =>
//          !child.outputPartitioning.guarantees(
//            createPartitioning(distribution, maxChildrenNumPartitions))
//      }
//      // If we need to shuffle all children, we use defaultNumPreShufflePartitions as the
//      // number of partitions. Otherwise, we use maxChildrenNumPartitions.
//      if (shufflesAllChildren) 5 else maxChildrenNumPartitions
//    }
//    var partitioning: Partitioning = null
//    partialMerged1.children.zip(dist).foreach {
//      case (child, distribution) if child.outputPartitioning.satisfies(distribution) =>
//      case (child, BroadcastDistribution(mode)) =>
//      case (child, distribution) =>
//        distribution match {
//          case ClusteredDistribution(clustering) =>
//            println("clustering", clustering.mkString(","))
//            partitioning = HashPartitioning(clustering, numPartitions)
//          case OrderedDistribution(ordering) =>
//            partitioning = RangePartitioning(ordering, numPartitions)
//          case AllTuples => SinglePartition
//          case other =>
//            println("Other partitioning", other)
//        }
//    }
//    println("PARTITIONING", partitioning)
//    val partitionExtractor = partitioning match {
//      case h: HashPartitioning =>
//        val projection =
//          UnsafeProjection.create(h.partitionIdExpression :: Nil, partialMerged1.output)
//        row: InternalRow => projection(row).getInt(0)
//      case _ => sys.error(s"Exchange not implemented for $partitioning")
//    }
//
//    val row = InternalRow(UTF8String.fromString("model" + 1))
//    val converter = UnsafeProjection.create(Array[DataType](StringType))
//    val unsafeRow = converter.apply(row)
//    println("the partition is!!!", partitionExtractor(unsafeRow))

//    val restored = StateStoreRestoreExec(groupingAttributes, None, partialMerged1)

    val partialMerged2: SparkPlan = {
      val aggregateExpressions = functionsWithoutDistinct.map(_.copy(mode = PartialMerge))
      val aggregateAttributes = aggregateExpressions.map(_.resultAttribute)
      createAggregate(
        requiredChildDistributionExpressions =
          Some(groupingAttributes),
        groupingExpressions = groupingAttributes,
        aggregateExpressions = aggregateExpressions,
        aggregateAttributes = aggregateAttributes,
        initialInputBufferOffset = groupingAttributes.length,
        resultExpressions = groupingAttributes ++
          aggregateExpressions.flatMap(_.aggregateFunction.inputAggBufferAttributes),
        child = partialMerged1)
    }
    // Note: stateId and returnAllStates are filled in later with preparation rules
    // in IncrementalExecution.
    val saved =
    StateStoreSaveExec(
      groupingAttributes,
      stateId = None,
      outputMode = None,
      eventTimeWatermark = None,
      partialMerged2)


    val finalAndCompleteAggregate: SparkPlan = {
      val finalAggregateExpressions = functionsWithoutDistinct.map(_.copy(mode = Final))
      // The attributes of the final aggregation buffer, which is presented as input to the result
      // projection:
      val finalAggregateAttributes = finalAggregateExpressions.map(_.resultAttribute)

      createAggregate(
        requiredChildDistributionExpressions = Some(groupingAttributes),
        groupingExpressions = groupingAttributes,
        aggregateExpressions = finalAggregateExpressions,
        aggregateAttributes = finalAggregateAttributes,
        initialInputBufferOffset = groupingAttributes.length,
        resultExpressions = resultExpressions,
        child = saved)
    }

    finalAndCompleteAggregate :: Nil

  }
}
