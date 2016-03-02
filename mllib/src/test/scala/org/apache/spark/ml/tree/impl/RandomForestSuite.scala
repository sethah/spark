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

package org.apache.spark.ml.tree.impl

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.impl.TreeTests
import org.apache.spark.ml.tree.{ContinuousSplit, DecisionTreeModel, LeafNode, Node}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.impurity.GiniCalculator
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.mllib.util.TestingUtils._
import org.apache.spark.util.collection.OpenHashMap
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.feature.{StringIndexer, IndexToString, VectorIndexer}
import org.apache.spark.sql.functions._

/**
 * Test suite for [[RandomForest]].
 */
class RandomForestSuite extends SparkFunSuite with MLlibTestSparkContext {

  import RandomForestSuite.mapToVec

  test("computeFeatureImportance, featureImportances") {
    /* Build tree for testing, with this structure:
          grandParent
      left2       parent
                left  right
     */
    val leftImp = new GiniCalculator(Array(3.0, 2.0, 1.0))
    val left = new LeafNode(0.0, leftImp.calculate(), leftImp)

    val rightImp = new GiniCalculator(Array(1.0, 2.0, 5.0))
    val right = new LeafNode(2.0, rightImp.calculate(), rightImp)

    val parent = TreeTests.buildParentNode(left, right, new ContinuousSplit(0, 0.5))
    val parentImp = parent.impurityStats

    val left2Imp = new GiniCalculator(Array(1.0, 6.0, 1.0))
    val left2 = new LeafNode(0.0, left2Imp.calculate(), left2Imp)

    val grandParent = TreeTests.buildParentNode(left2, parent, new ContinuousSplit(1, 1.0))
    val grandImp = grandParent.impurityStats

    // Test feature importance computed at different subtrees.
    def testNode(node: Node, expected: Map[Int, Double]): Unit = {
      val map = new OpenHashMap[Int, Double]()
      RandomForest.computeFeatureImportance(node, map)
      assert(mapToVec(map.toMap) ~== mapToVec(expected) relTol 0.01)
    }

    // Leaf node
    testNode(left, Map.empty[Int, Double])

    // Internal node with 2 leaf children
    val feature0importance = parentImp.calculate() * parentImp.count -
      (leftImp.calculate() * leftImp.count + rightImp.calculate() * rightImp.count)
    testNode(parent, Map(0 -> feature0importance))

    // Full tree
    val feature1importance = grandImp.calculate() * grandImp.count -
      (left2Imp.calculate() * left2Imp.count + parentImp.calculate() * parentImp.count)
    testNode(grandParent, Map(0 -> feature0importance, 1 -> feature1importance))

    // Forest consisting of (full tree) + (internal node with 2 leafs)
    val trees = Array(parent, grandParent).map { root =>
      new DecisionTreeClassificationModel(root, numFeatures = 2, numClasses = 3)
        .asInstanceOf[DecisionTreeModel]
    }
    val importances: Vector = RandomForest.featureImportances(trees, 2)
    val tree2norm = feature0importance + feature1importance
    val expected = Vectors.dense((1.0 + feature0importance / tree2norm) / 2.0,
      (feature1importance / tree2norm) / 2.0)
    assert(importances ~== expected relTol 0.01)
  }

  test("info gain") {
    val data = Array(LabeledPoint(1.0, Vectors.dense(1.0, 0.0)),
      LabeledPoint(0.0, Vectors.dense(0.0, 0.0)),
      LabeledPoint(0.0, Vectors.dense(0.0, 0.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 0.0)),
      LabeledPoint(0.0, Vectors.dense(0.0, 0.0)),
      LabeledPoint(0.0, Vectors.dense(0.0, 0.0)))
    val rdd = sc.parallelize(data)
    val df = sqlContext.createDataFrame(rdd).withColumn("weight", lit(1.0 / 100.0))
    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(df)
    val df2 = labelIndexer.transform(df)
    val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setWeightCol("weight").setMinInstancesPerNode(0).setMaxDepth(3)
    val model = dt.fit(df2)
    val preds = rdd.collect().map { lp => model.predict(lp.features)}
    preds.foreach(println)

    val df3 = sqlContext.createDataFrame(rdd).withColumn("weight", lit(1.0))
    val labelIndexer2 = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(df)
    val df4 = labelIndexer2.transform(df3)
    val dt2 = new DecisionTreeClassifier().setLabelCol("indexedLabel").setWeightCol("weight").setMinInstancesPerNode(0).setMaxDepth(3)
    val model2 = dt2.fit(df4)
    val preds2 = rdd.collect().map { lp => model2.predict(lp.features)}
    preds2.foreach(println)
  }

  test("normalizeMapValues") {
    val map = new OpenHashMap[Int, Double]()
    map(0) = 1.0
    map(2) = 2.0
    RandomForest.normalizeMapValues(map)
    val expected = Map(0 -> 1.0 / 3.0, 2 -> 2.0 / 3.0)
    assert(mapToVec(map.toMap) ~== mapToVec(expected) relTol 0.01)
  }

}

private object RandomForestSuite {

  def mapToVec(map: Map[Int, Double]): Vector = {
    val size = (map.keys.toSeq :+ 0).max + 1
    val (indices, values) = map.toSeq.sortBy(_._1).unzip
    Vectors.sparse(size, indices.toArray, values.toArray)
  }
}
