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

package org.apache.spark.sql.expressions

import org.apache.spark.SparkEnv
import org.apache.spark.annotation.InterfaceStability
import org.apache.spark.sql.{Column, Row}
import org.apache.spark.sql.catalyst.expressions.aggregate.{AggregateExpression, Complete}
import org.apache.spark.sql.execution.aggregate.{ScalaModelUDAF, ScalaUDAF}
import org.apache.spark.sql.types._
import org.apache.spark.storage.BlockId

import scala.collection.mutable

/**
 * The base class for implementing user-defined aggregate functions (UDAF).
 *
 * @since 1.5.0
 */
@InterfaceStability.Stable
abstract class UserDefinedAggregateFunction extends Serializable {

  /**
   * A `StructType` represents data types of input arguments of this aggregate function.
   * For example, if a [[UserDefinedAggregateFunction]] expects two input arguments
   * with type of `DoubleType` and `LongType`, the returned `StructType` will look like
   *
   * ```
   *   new StructType()
   *    .add("doubleInput", DoubleType)
   *    .add("longInput", LongType)
   * ```
   *
   * The name of a field of this `StructType` is only used to identify the corresponding
   * input argument. Users can choose names to identify the input arguments.
   *
   * @since 1.5.0
   */
  def inputSchema: StructType

  /**
   * A `StructType` represents data types of values in the aggregation buffer.
   * For example, if a [[UserDefinedAggregateFunction]]'s buffer has two values
   * (i.e. two intermediate values) with type of `DoubleType` and `LongType`,
   * the returned `StructType` will look like
   *
   * ```
   *   new StructType()
   *    .add("doubleInput", DoubleType)
   *    .add("longInput", LongType)
   * ```
   *
   * The name of a field of this `StructType` is only used to identify the corresponding
   * buffer value. Users can choose names to identify the input arguments.
   *
   * @since 1.5.0
   */
  def bufferSchema: StructType

  /**
   * The `DataType` of the returned value of this [[UserDefinedAggregateFunction]].
   *
   * @since 1.5.0
   */
  def dataType: DataType

  /**
   * Returns true iff this function is deterministic, i.e. given the same input,
   * always return the same output.
   *
   * @since 1.5.0
   */
  def deterministic: Boolean

  /**
   * Initializes the given aggregation buffer, i.e. the zero value of the aggregation buffer.
   *
   * The contract should be that applying the merge function on two initial buffers should just
   * return the initial buffer itself, i.e.
   * `merge(initialBuffer, initialBuffer)` should equal `initialBuffer`.
   *
   * @since 1.5.0
   */
  def initialize(buffer: MutableAggregationBuffer): Unit

  /**
   * Updates the given aggregation buffer `buffer` with new input data from `input`.
   *
   * This is called once per input row.
   *
   * @since 1.5.0
   */
  def update(buffer: MutableAggregationBuffer, input: Row): Unit

  /**
   * Merges two aggregation buffers and stores the updated buffer values back to `buffer1`.
   *
   * This is called when we merge two partially aggregated data together.
   *
   * @since 1.5.0
   */
  def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit

  /**
   * Calculates the final result of this [[UserDefinedAggregateFunction]] based on the given
   * aggregation buffer.
   *
   * @since 1.5.0
   */
  def evaluate(buffer: Row): Any

  /**
   * Creates a `Column` for this UDAF using given `Column`s as input arguments.
   *
   * @since 1.5.0
   */
  @scala.annotation.varargs
  def apply(exprs: Column*): Column = {
    val aggregateExpression =
      AggregateExpression(
        ScalaUDAF(exprs.map(_.expr), this),
        Complete,
        isDistinct = false)
    Column(aggregateExpression)
  }

  /**
   * Creates a `Column` for this UDAF using the distinct values of the given
   * `Column`s as input arguments.
   *
   * @since 1.5.0
   */
  @scala.annotation.varargs
  def distinct(exprs: Column*): Column = {
    val aggregateExpression =
      AggregateExpression(
        ScalaUDAF(exprs.map(_.expr), this),
        Complete,
        isDistinct = true)
    Column(aggregateExpression)
  }
}

abstract class ModelUserDefinedAggregateFunction extends UserDefinedAggregateFunction {

//  def initialize(buffer: MutableAggregationBuffer, state: Any): Unit


  override def apply(exprs: Column*): Column = {
    val aggregateExpression =
      AggregateExpression(
        ScalaModelUDAF(exprs.map(_.expr), this),
        Complete,
        isDistinct = false)
    Column(aggregateExpression)
  }
}

case class ModelAgg(initBlock: Option[BlockId]) extends ModelUserDefinedAggregateFunction {

  // Input Data Type Schema
  def inputSchema: StructType = StructType(Array(StructField("item", IntegerType)))

  // Intermediate Schema
  def bufferSchema: StructType = StructType(Array(
    StructField("sum", IntegerType)
  ))

  // Returned Data Type .
  def dataType: DataType = IntegerType

  // Self-explaining
  def deterministic: Boolean = true

  //  def initialize(buffer: MutableAggregationBuffer, state: Any): Unit = {
  //    buffer(0) = state.asInstanceOf[Int]
  //  }

  // This function is called whenever key changes
  def initialize(buffer: MutableAggregationBuffer): Unit = {
    val initialState = initBlock.flatMap { block =>
      SparkEnv.get.blockManager.getLocalValues(block).map(_.data.next().asInstanceOf[Int])
    }.getOrElse(0)
    println(s"INITIAL STATE IS: $initialState")
    buffer(0) = initialState
  }

  // Iterate over each entry of a group
  def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
    buffer(0) = buffer.getInt(0) + 3
  }

  // Merge two partial aggregates
  def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
    buffer1(0) = buffer1.getInt(0) + buffer2.getInt(0)
  }

  // Called after all the entries are exhausted.
  def evaluate(buffer: Row): Integer = {
    buffer.getInt(0)
  }

}

/*
Another option is to have this class take in weights, so at each iteration, we get the previous
weights from the state store or wherever, and then we compute the gradients like normal. When
we do the merge operation, we take this.weights +
 */
case class SGDAgg(numFeatures: Int, initBlock: Option[BlockId])
  extends ModelUserDefinedAggregateFunction {

  val stepSize = 0.01

  // Input Data Type Schema
  def inputSchema: StructType = StructType(Array(
    StructField("label", DoubleType),
    StructField("features", ArrayType(DoubleType))
  ))

  // Intermediate Schema
  def bufferSchema: StructType = StructType(Array(
    StructField("count", LongType),
    StructField("coefficients", ArrayType(DoubleType))
  ))

  // Returned Data Type .
  def dataType: DataType = ArrayType(DoubleType)

  // Self-explaining
  def deterministic: Boolean = true

//  def initialize(buffer: MutableAggregationBuffer, state: Any): Unit = {
//    val previousModel = state.asInstanceOf[(Long, Array[Double])]
//    buffer.update(0, previousModel._1)
//    buffer.update(1, previousModel._2)
//    println("INITIAL BUFFER", buffer.getAs[mutable.WrappedArray[Double]](1).mkString(","))
//  }

  // This function is called whenever key changes
  def initialize(buffer: MutableAggregationBuffer): Unit = {
    val initialState = initBlock.flatMap { block =>
      val localVals = SparkEnv.get.blockManager.getLocalValues(block)
      localVals.foreach(v => println("FOUND THE INITIAL BLOCK FOR SGD AGG"))
      localVals.map(_.data.next().asInstanceOf[(Long, Array[Double])])
    }.getOrElse((0L, new Array[Double](numFeatures)))
    println(s"INITIAL STATE WAS ${initialState._1}, ${initialState._2.mkString(",")}")
    buffer.update(0, initialState._1)
    buffer.update(1, initialState._2)
  }

  // Iterate over each entry of a group
  def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
    if (!input.isNullAt(0)) {
      val buff = buffer.getAs[mutable.WrappedArray[Double]](1)
      val features = input.getAs[mutable.WrappedArray[Double]](1)
      val label = input.getDouble(0)
      val error = label - dot(features.toArray, buff.toArray)
      val gradient = features.map(_ * error)
      //      println("old model", buff.mkString(","))
      //      println("data", label, features.mkString(","))
      buff.indices.foreach { i =>
        buff(i) += gradient(i) * stepSize
      }
      //      println("new model", buff.mkString(","))
      //      println("gradient", gradient.mkString(","))

      buffer.update(0, buffer.getLong(0) + 1)
      buffer.update(1, buff)
    }
  }

  def dot(x: Array[Double], y: Array[Double]): Double = {
    x.zip(y).foldLeft(0.0) { case (acc, tup) => acc + tup._1 * tup._2}
  }

  // Merge two partial aggregates
  def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
    // weighted average of the two
    val otherCount = buffer2.getLong(0)
    val otherFeatures = buffer2.getAs[mutable.WrappedArray[Double]](1)
    val thisCount = buffer1.getLong(0)
    val buff = buffer1.getAs[mutable.WrappedArray[Double]](1)
    buff.indices.foreach { i =>
      buff(i) = (buff(i) * thisCount + otherFeatures(i) * otherCount) / (thisCount + otherCount)
    }

    buffer1.update(1, buff)
    buffer1.update(0, buffer1.getLong(0) + otherCount)

  }

  // Called after all the entries are exhausted.
  def evaluate(buffer: Row): Array[Double] = {
    buffer.getAs[mutable.WrappedArray[Double]](1).toArray
  }

}

/**
 * A `Row` representing a mutable aggregation buffer.
 *
 * This is not meant to be extended outside of Spark.
 *
 * @since 1.5.0
 */
@InterfaceStability.Stable
abstract class MutableAggregationBuffer extends Row {

  /** Update the ith value of this buffer. */
  def update(i: Int, value: Any): Unit
}
