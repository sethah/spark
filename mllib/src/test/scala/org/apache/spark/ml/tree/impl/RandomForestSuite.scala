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
import org.apache.spark.ml.tree.{CategoricalSplit, ContinuousSplit}
import org.apache.spark.ml.tree.DecisionTreeModel
import org.apache.spark.ml.tree.{LeafNode, LearningNode, Node}
import org.apache.spark.ml.tree.impl.RandomForest.NodeIndexInfo
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.EnsembleTestHelper
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.configuration.{QuantileStrategy, Strategy}
import org.apache.spark.mllib.tree.impl.{BaggedPoint, DecisionTreeMetadata}
import org.apache.spark.mllib.tree.impurity.{Entropy, Gini, GiniCalculator}
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.mllib.util.TestingUtils._
import org.apache.spark.util.collection.OpenHashMap

import scala.collection.mutable


/**
 * Test suite for [[RandomForest]].
 */
class RandomForestSuite extends SparkFunSuite with MLlibTestSparkContext {

  import RandomForestSuite.mapToVec

    /////////////////////////////////////////////////////////////////////////////
    // Tests examining individual elements of training
    /////////////////////////////////////////////////////////////////////////////

    test("Binary classification with continuous features: split and bin calculation") {
      val arr = RandomForestSuite.generateOrderedLabeledPointsWithLabel1()
      assert(arr.length === 1000)
      val rdd = sc.parallelize(arr)
      val strategy = new Strategy(Classification, Gini, 3, 2, 100)
      val metadata = DecisionTreeMetadata.buildMetadata(rdd, strategy)
      assert(!metadata.isUnordered(featureIndex = 0))
      val splits = RandomForest.findSplits(rdd, metadata, seed = 0L)
      assert(splits.length === 2)
      assert(splits(0).length === 99)
    }

    test("Binary classification with binary (ordered) categorical features:" +
        " split and bin calculation") {
        val arr = RandomForestSuite.generateCategoricalDataPoints()
        assert(arr.length === 1000)
        val rdd = sc.parallelize(arr)
        val strategy = new Strategy(
          Classification,
          Gini,
          maxDepth = 2,
          numClasses = 2,
          maxBins = 100,
          categoricalFeaturesInfo = Map(0 -> 2, 1-> 2))

        val metadata = DecisionTreeMetadata.buildMetadata(rdd, strategy)
        val splits = RandomForest.findSplits(rdd, metadata, seed = 0L)
        assert(!metadata.isUnordered(featureIndex = 0))
        assert(!metadata.isUnordered(featureIndex = 1))
        assert(splits.length === 2)
        // no bins or splits pre-computed for ordered categorical features
        assert(splits(0).length === 0)
      }

    test("Binary classification with 3-ary (ordered) categorical features," +
      " with no samples for one category") {
      val arr = RandomForestSuite.generateCategoricalDataPoints()
      assert(arr.length === 1000)
      val rdd = sc.parallelize(arr)
      val strategy = new Strategy(
        Classification,
        Gini,
        maxDepth = 2,
        numClasses = 2,
        maxBins = 100,
        categoricalFeaturesInfo = Map(0 -> 3, 1 -> 3))

      val metadata = DecisionTreeMetadata.buildMetadata(rdd, strategy)
      assert(!metadata.isUnordered(featureIndex = 0))
      assert(!metadata.isUnordered(featureIndex = 1))
      val splits = RandomForest.findSplits(rdd, metadata, seed = 0L)
      assert(splits.length === 2)
      // no bins or splits pre-computed for ordered categorical features
      assert(splits(0).length === 0)
    }

  test("extract categories from a number for multiclass classification") {
    val l = RandomForest.extractMultiClassCategories(13, 10)
    assert(l.length === 3)
    assert(List(3.0, 2.0, 0.0).toSeq === l.toSeq)
  }

  test("find splits for a continuous feature") {
    // find splits for normal case
    {
      val fakeMetadata = new DecisionTreeMetadata(1, 0, 0, 0,
        Map(), Set(),
        Array(6), Gini, QuantileStrategy.Sort,
        0, 0, 0.0, 0, 0
      )
      val featureSamples = Array.fill(200000)(math.random)
      val splits = RandomForest.findSplitsForContinuousFeature(featureSamples, fakeMetadata, 0)
      assert(splits.length === 5)
      assert(fakeMetadata.numSplits(0) === 5)
      assert(fakeMetadata.numBins(0) === 6)
      // check returned splits are distinct
      assert(splits.distinct.length === splits.length)
    }

    // find splits should not return identical splits
    // when there are not enough split candidates, reduce the number of splits in metadata
    {
      val fakeMetadata = new DecisionTreeMetadata(1, 0, 0, 0,
        Map(), Set(),
        Array(5), Gini, QuantileStrategy.Sort,
        0, 0, 0.0, 0, 0
      )
      val featureSamples = Array(1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3).map(_.toDouble)
      val splits = RandomForest.findSplitsForContinuousFeature(featureSamples, fakeMetadata, 0)
      assert(splits.length === 3)
      // check returned splits are distinct
      assert(splits.distinct.length === splits.length)
    }

    // find splits when most samples close to the minimum
    {
      val fakeMetadata = new DecisionTreeMetadata(1, 0, 0, 0,
        Map(), Set(),
        Array(3), Gini, QuantileStrategy.Sort,
        0, 0, 0.0, 0, 0
      )
      val featureSamples = Array(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 5).map(_.toDouble)
      val splits = RandomForest.findSplitsForContinuousFeature(featureSamples, fakeMetadata, 0)
      assert(splits.length === 2)
      assert(splits(0) === 2.0)
      assert(splits(1) === 3.0)
    }

    // find splits when most samples close to the maximum
    {
      val fakeMetadata = new DecisionTreeMetadata(1, 0, 0, 0,
        Map(), Set(),
        Array(3), Gini, QuantileStrategy.Sort,
        0, 0, 0.0, 0, 0
      )
      val featureSamples = Array(0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2).map(_.toDouble)
      val splits = RandomForest.findSplitsForContinuousFeature(featureSamples, fakeMetadata, 0)
      assert(splits.length === 1)
      assert(splits(0) === 1.0)
    }
  }

    test("Multiclass classification with unordered categorical features:" +
        " split and bin calculations") {
      val arr = RandomForestSuite.generateCategoricalDataPoints()
      assert(arr.length === 1000)
      val rdd = sc.parallelize(arr)
      val strategy = new Strategy(
        Classification,
        Gini,
        maxDepth = 2,
        numClasses = 100,
        maxBins = 100,
        categoricalFeaturesInfo = Map(0 -> 3, 1-> 3))

      val metadata = DecisionTreeMetadata.buildMetadata(rdd, strategy)
      assert(metadata.isUnordered(featureIndex = 0))
      assert(metadata.isUnordered(featureIndex = 1))
      val splits = RandomForest.findSplits(rdd, metadata, seed = 0L)
      assert(splits.length === 2)
      assert(splits(0).length === 3)

      // Expecting 2^2 - 1 = 3 bins/splits
      assert(splits(0)(0).isInstanceOf[CategoricalSplit])
      assert(splits(0)(0).asInstanceOf[CategoricalSplit].featureIndex === 0)
      assert(splits(0)(0).asInstanceOf[CategoricalSplit].leftCategories.length === 1)
      assert(splits(0)(0).asInstanceOf[CategoricalSplit].leftCategories.contains(0.0))
      assert(splits(1)(0).isInstanceOf[CategoricalSplit])
      assert(splits(1)(0).asInstanceOf[CategoricalSplit].featureIndex === 1)
      assert(splits(1)(0).asInstanceOf[CategoricalSplit].leftCategories.length === 1)
      assert(splits(1)(0).asInstanceOf[CategoricalSplit].leftCategories.contains(0.0))

      assert(splits(0)(1).isInstanceOf[CategoricalSplit])
      assert(splits(0)(1).asInstanceOf[CategoricalSplit].featureIndex === 0)
      assert(splits(0)(1).asInstanceOf[CategoricalSplit].leftCategories.length === 1)
      assert(splits(0)(1).asInstanceOf[CategoricalSplit].leftCategories.contains(1.0))
      assert(splits(1)(1).isInstanceOf[CategoricalSplit])
      assert(splits(1)(1).asInstanceOf[CategoricalSplit].featureIndex === 1)
      assert(splits(1)(1).asInstanceOf[CategoricalSplit].leftCategories.length === 1)
      assert(splits(1)(1).asInstanceOf[CategoricalSplit].leftCategories.contains(1.0))

      assert(splits(0)(2).isInstanceOf[CategoricalSplit])
      assert(splits(0)(2).asInstanceOf[CategoricalSplit].featureIndex === 0)
      assert(splits(0)(2).asInstanceOf[CategoricalSplit].leftCategories.length === 2)
      assert(splits(0)(2).asInstanceOf[CategoricalSplit].leftCategories.contains(0.0))
      assert(splits(0)(2).asInstanceOf[CategoricalSplit].leftCategories.contains(1.0))
      assert(splits(1)(2).isInstanceOf[CategoricalSplit])
      assert(splits(1)(2).asInstanceOf[CategoricalSplit].featureIndex === 1)
      assert(splits(1)(2).asInstanceOf[CategoricalSplit].leftCategories.length === 2)
      assert(splits(1)(2).asInstanceOf[CategoricalSplit].leftCategories.contains(0.0))
      assert(splits(1)(2).asInstanceOf[CategoricalSplit].leftCategories.contains(1.0))
    }

    test("Multiclass classification with ordered categorical features: split and bin calculations") {
      val arr = RandomForestSuite.generateCategoricalDataPointsForMulticlassForOrderedFeatures()
      assert(arr.length === 3000)
      val rdd = sc.parallelize(arr)
      val strategy = new Strategy(
        Classification,
        Gini,
        maxDepth = 2,
        numClasses = 100,
        maxBins = 100,
        categoricalFeaturesInfo = Map(0 -> 10, 1-> 10))
      // 2^(10-1) - 1 > 100, so categorical features will be ordered

      val metadata = DecisionTreeMetadata.buildMetadata(rdd, strategy)
      assert(!metadata.isUnordered(featureIndex = 0))
      assert(!metadata.isUnordered(featureIndex = 1))
      val splits = RandomForest.findSplits(rdd, metadata, seed = 0L)
      assert(splits.length === 2)
      // no bins or splits pre-computed for ordered categorical features
      assert(splits(0).length === 0)
    }

    test("Avoid aggregation on the last level") {
      val arr = Array(
        LabeledPoint(0.0, Vectors.dense(1.0, 0.0, 0.0)),
        LabeledPoint(1.0, Vectors.dense(0.0, 1.0, 1.0)),
        LabeledPoint(0.0, Vectors.dense(2.0, 0.0, 0.0)),
        LabeledPoint(1.0, Vectors.dense(0.0, 2.0, 1.0)))
      val input = sc.parallelize(arr)

      val strategy = new Strategy(algo = Classification, impurity = Gini, maxDepth = 1,
        numClasses = 2, categoricalFeaturesInfo = Map(0 -> 3))
      val metadata = DecisionTreeMetadata.buildMetadata(input, strategy)
      val splits = RandomForest.findSplits(input, metadata, seed = 0L)

      val treeInput = TreePoint.convertToTreeRDD(input, splits, metadata)
      val baggedInput = BaggedPoint.convertToBaggedRDD(treeInput, 1.0, 1, false)

      val topNode = LearningNode.emptyNode(nodeIndex = 1)
      assert(topNode.stats === null)
      assert(topNode.id === 1)
      assert(topNode.isLeaf === false)

      val nodesForGroup = Map((0, Array(topNode)))
      val treeToNodeToIndexInfo = Map((0, Map(
        (topNode.id, new RandomForest.NodeIndexInfo(0, None))
      )))
      val nodeQueue = new mutable.Queue[(Int, LearningNode)]()
      RandomForest.findBestSplits(baggedInput, metadata, Array(topNode),
        nodesForGroup, treeToNodeToIndexInfo, splits, nodeQueue)

      // don't enqueue leaf nodes into node queue
      assert(nodeQueue.isEmpty)

      // set impurity and predict for topNode
      // TODO: should we convert these to permanent nodes and then test?
      assert(topNode.stats !== null)

      // set impurity and predict for child nodes
      assert(topNode.leftChild.get.stats.impurityCalculator.predict === 0.0)
      assert(topNode.rightChild.get.stats.impurityCalculator.predict === 1.0)
      assert(topNode.leftChild.get.stats.impurity === 0.0)
      assert(topNode.rightChild.get.stats.impurity === 0.0)
    }

      test("Avoid aggregation if impurity is 0.0") {
        val arr = Array(
          LabeledPoint(0.0, Vectors.dense(1.0, 0.0, 0.0)),
          LabeledPoint(1.0, Vectors.dense(0.0, 1.0, 1.0)),
          LabeledPoint(0.0, Vectors.dense(2.0, 0.0, 0.0)),
          LabeledPoint(1.0, Vectors.dense(0.0, 2.0, 1.0)))
        val input = sc.parallelize(arr)

        val strategy = new Strategy(algo = Classification, impurity = Gini, maxDepth = 5,
          numClasses = 2, categoricalFeaturesInfo = Map(0 -> 3))
        val metadata = DecisionTreeMetadata.buildMetadata(input, strategy)
        val splits = RandomForest.findSplits(input, metadata, seed = 0L)

        val treeInput = TreePoint.convertToTreeRDD(input, splits, metadata)
        val baggedInput = BaggedPoint.convertToBaggedRDD(treeInput, 1.0, 1, false)

        val topNode = LearningNode.emptyNode(nodeIndex = 1)
        assert(topNode.stats === null)
        assert(topNode.isLeaf === false)

        val nodesForGroup = Map((0, Array(topNode)))
        val treeToNodeToIndexInfo = Map((0, Map(
          (topNode.id, new RandomForest.NodeIndexInfo(0, None))
        )))
        val nodeQueue = new mutable.Queue[(Int, LearningNode)]()
        RandomForest.findBestSplits(baggedInput, metadata, Array(topNode),
          nodesForGroup, treeToNodeToIndexInfo, splits, nodeQueue)

        // don't enqueue a node into node queue if its impurity is 0.0
        assert(nodeQueue.isEmpty)

        // set impurity and predict for topNode
        assert(topNode.stats !== null)

        // set impurity and predict for child nodes
        assert(topNode.leftChild.get.stats.impurityCalculator.predict === 0.0)
        assert(topNode.rightChild.get.stats.impurityCalculator.predict === 1.0)
        assert(topNode.leftChild.get.stats.impurity === 0.0)
        assert(topNode.rightChild.get.stats.impurity === 0.0)
      }

      test("Second level node building with vs. without groups") {
        val arr = RandomForestSuite.generateOrderedLabeledPoints()
        assert(arr.length === 1000)
        val rdd = sc.parallelize(arr)
        val strategy = new Strategy(Classification, Entropy, 3, 2, 100)
        val metadata = DecisionTreeMetadata.buildMetadata(rdd, strategy)
        val splits = RandomForest.findSplits(rdd, metadata, seed = 0L)
        assert(splits.length === 2)
        assert(splits(0).length === 99)

        val treeInput = TreePoint.convertToTreeRDD(rdd, splits, metadata)
        val baggedInput = BaggedPoint.convertToBaggedRDD(treeInput, 1.0, 1, false)

        // Train a 1-node model
        val strategyOneNode = new Strategy(Classification, Entropy, maxDepth = 1,
          numClasses = 2, maxBins = 100)
        val topNode = LearningNode.emptyNode(nodeIndex = 1)
        val topNodes = Array(topNode)
        val nodesForGroup1 = Map((0, topNodes))
        val treeToNodeToIndexInfo1 = Map((0, Map((topNodes(0).id, new NodeIndexInfo(0, None)))))
        val nodeQueue1 = new mutable.Queue[(Int, LearningNode)]()
        RandomForest.findBestSplits(baggedInput, metadata, topNodes, nodesForGroup1,
          treeToNodeToIndexInfo1, splits, nodeQueue1)
        val rootNode1 = RandomForestSuite.deepCopyLearningTree(topNode)
        val rootNode2 = RandomForestSuite.deepCopyLearningTree(topNode)
        assert(rootNode1.leftChild.nonEmpty)
        assert(rootNode1.rightChild.nonEmpty)

        // Single group second level tree construction.
        val nodesForGroup = Map((0, Array(rootNode1.leftChild.get, rootNode1.rightChild.get)))
        val treeToNodeToIndexInfo = Map((0, Map(
          (rootNode1.leftChild.get.id, new RandomForest.NodeIndexInfo(0, None)),
          (rootNode1.rightChild.get.id, new RandomForest.NodeIndexInfo(1, None)))))
        val nodeQueue = new mutable.Queue[(Int, LearningNode)]()
        RandomForest.findBestSplits(baggedInput, metadata, Array(rootNode1),
          nodesForGroup, treeToNodeToIndexInfo, splits, nodeQueue)
        val children1 = new Array[LearningNode](2)
        children1(0) = rootNode1.leftChild.get
        children1(1) = rootNode1.rightChild.get

        // Train one second-level node at a time.
        val nodesForGroupA = Map((0, Array(rootNode2.leftChild.get)))
        val treeToNodeToIndexInfoA = Map((0, Map(
          (rootNode2.leftChild.get.id, new RandomForest.NodeIndexInfo(0, None)))))
        nodeQueue.clear()
        RandomForest.findBestSplits(baggedInput, metadata, Array(rootNode2),
          nodesForGroupA, treeToNodeToIndexInfoA, splits, nodeQueue)
        val nodesForGroupB = Map((0, Array(rootNode2.rightChild.get)))
        val treeToNodeToIndexInfoB = Map((0, Map(
          (rootNode2.rightChild.get.id, new RandomForest.NodeIndexInfo(0, None)))))
        nodeQueue.clear()
        RandomForest.findBestSplits(baggedInput, metadata, Array(rootNode2),
          nodesForGroupB, treeToNodeToIndexInfoB, splits, nodeQueue)
        val children2 = new Array[LearningNode](2)
        children2(0) = rootNode2.leftChild.get
        children2(1) = rootNode2.rightChild.get

        // Verify whether the splits obtained using single group and multiple group level
        // construction strategies are the same.
        for (i <- 0 until 2) {
          assert(children1(i).stats.gain > 0)
          assert(children2(i).stats.gain > 0)
          assert(children1(i).split === children2(i).split)
          val stats1 = children1(i).stats
          val stats2 = children2(i).stats
          assert(stats1.gain === stats2.gain)
          assert(stats1.impurity === stats2.impurity)
          assert(stats1.leftImpurity === stats2.leftImpurity)
          assert(stats1.rightImpurity === stats2.rightImpurity)
          assert(children1(i).stats.impurityCalculator.predict === children2(i).stats.impurityCalculator.predict)
        }
      }

  /////////////////////////////////////////////////////////////////////////////
  // Tests specific to random forest
  /////////////////////////////////////////////////////////////////////////////

    def binaryClassificationTestWithContinuousFeaturesAndSubsampledFeatures(strategy: Strategy) {
      val numFeatures = 50
      val arr = EnsembleTestHelper.generateOrderedLabeledPoints(numFeatures, 1000)
      val rdd = sc.parallelize(arr)

      // Select feature subset for top nodes.  Return true if OK.
      def checkFeatureSubsetStrategy(
          numTrees: Int,
          featureSubsetStrategy: String,
          numFeaturesPerNode: Int): Unit = {
        val seeds = Array(123, 5354, 230, 349867, 23987)
        val maxMemoryUsage: Long = 128 * 1024L * 1024L
        val metadata =
          DecisionTreeMetadata.buildMetadata(rdd, strategy, numTrees, featureSubsetStrategy)
        seeds.foreach { seed =>
          val failString = s"Failed on test with:" +
            s"numTrees=$numTrees, featureSubsetStrategy=$featureSubsetStrategy," +
            s" numFeaturesPerNode=$numFeaturesPerNode, seed=$seed"
          val nodeQueue = new mutable.Queue[(Int, LearningNode)]()
          val topNodes: Array[LearningNode] = new Array[LearningNode](numTrees)
          Range(0, numTrees).foreach { treeIndex =>
            topNodes(treeIndex) = LearningNode.emptyNode(nodeIndex = 1)
            nodeQueue.enqueue((treeIndex, topNodes(treeIndex)))
          }
          val rng = new scala.util.Random(seed = seed)
          val (nodesForGroup: Map[Int, Array[LearningNode]],
              treeToNodeToIndexInfo: Map[Int, Map[Int, RandomForest.NodeIndexInfo]]) =
            RandomForest.selectNodesToSplit(nodeQueue, maxMemoryUsage, metadata, rng)

          assert(nodesForGroup.size === numTrees, failString)
          assert(nodesForGroup.values.forall(_.size == 1), failString) // 1 node per tree

          if (numFeaturesPerNode == numFeatures) {
            // featureSubset values should all be None
            assert(treeToNodeToIndexInfo.values.forall(_.values.forall(_.featureSubset.isEmpty)),
              failString)
          } else {
            // Check number of features.
            assert(treeToNodeToIndexInfo.values.forall(_.values.forall(
              _.featureSubset.get.size === numFeaturesPerNode)), failString)
          }
        }
      }

      checkFeatureSubsetStrategy(numTrees = 1, "auto", numFeatures)
      checkFeatureSubsetStrategy(numTrees = 1, "all", numFeatures)
      checkFeatureSubsetStrategy(numTrees = 1, "sqrt", math.sqrt(numFeatures).ceil.toInt)
      checkFeatureSubsetStrategy(numTrees = 1, "log2",
        (math.log(numFeatures) / math.log(2)).ceil.toInt)
      checkFeatureSubsetStrategy(numTrees = 1, "onethird", (numFeatures / 3.0).ceil.toInt)

      checkFeatureSubsetStrategy(numTrees = 2, "all", numFeatures)
      checkFeatureSubsetStrategy(numTrees = 2, "auto", math.sqrt(numFeatures).ceil.toInt)
      checkFeatureSubsetStrategy(numTrees = 2, "sqrt", math.sqrt(numFeatures).ceil.toInt)
      checkFeatureSubsetStrategy(numTrees = 2, "log2",
        (math.log(numFeatures) / math.log(2)).ceil.toInt)
      checkFeatureSubsetStrategy(numTrees = 2, "onethird", (numFeatures / 3.0).ceil.toInt)
    }

    test("Binary classification with continuous features: subsampling features") {
      val categoricalFeaturesInfo = Map.empty[Int, Int]
      val strategy = new Strategy(algo = Classification, impurity = Gini, maxDepth = 2,
        numClasses = 2, categoricalFeaturesInfo = categoricalFeaturesInfo)
      binaryClassificationTestWithContinuousFeaturesAndSubsampledFeatures(strategy)
    }

    test("Binary classification with continuous features and node Id cache: subsampling features") {
      val categoricalFeaturesInfo = Map.empty[Int, Int]
      val strategy = new Strategy(algo = Classification, impurity = Gini, maxDepth = 2,
        numClasses = 2, categoricalFeaturesInfo = categoricalFeaturesInfo,
        useNodeIdCache = true)
      binaryClassificationTestWithContinuousFeaturesAndSubsampledFeatures(strategy)
    }

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

  def generateOrderedLabeledPointsWithLabel1(): Array[LabeledPoint] = {
    val arr = new Array[LabeledPoint](1000)
    for (i <- 0 until 1000) {
      val lp = new LabeledPoint(1.0, Vectors.dense(i.toDouble, 999.0 - i))
      arr(i) = lp
    }
    arr
  }

  def generateCategoricalDataPoints(): Array[LabeledPoint] = {
    val arr = new Array[LabeledPoint](1000)
    for (i <- 0 until 1000) {
      if (i < 600) {
        arr(i) = new LabeledPoint(1.0, Vectors.dense(0.0, 1.0))
      } else {
        arr(i) = new LabeledPoint(0.0, Vectors.dense(1.0, 0.0))
      }
    }
    arr
  }

  def generateCategoricalDataPointsForMulticlassForOrderedFeatures():
  Array[LabeledPoint] = {
    val arr = new Array[LabeledPoint](3000)
    for (i <- 0 until 3000) {
      if (i < 1000) {
        arr(i) = new LabeledPoint(2.0, Vectors.dense(2.0, 2.0))
      } else if (i < 2000) {
        arr(i) = new LabeledPoint(1.0, Vectors.dense(1.0, 2.0))
      } else {
        arr(i) = new LabeledPoint(1.0, Vectors.dense(2.0, 2.0))
      }
    }
    arr
  }

  def generateOrderedLabeledPoints(): Array[LabeledPoint] = {
    val arr = new Array[LabeledPoint](1000)
    for (i <- 0 until 1000) {
      val label = if (i < 100) {
        0.0
      } else if (i < 500) {
        1.0
      } else if (i < 900) {
        0.0
      } else {
        1.0
      }
      arr(i) = new LabeledPoint(label, Vectors.dense(i.toDouble, 1000.0 - i))
    }
    arr
  }

  /**
   * Returns a deep copy of the subtree rooted at this learning node.
   */
  def deepCopyLearningTree(node: LearningNode): LearningNode = {
    if (node.leftChild.isEmpty) {
      assert(node.rightChild.isEmpty)
      new LearningNode(node.id, None, None, None, true, node.stats)
    } else {
      assert(node.rightChild.nonEmpty)
      new LearningNode(node.id, Some(deepCopyLearningTree(node.leftChild.get)),
        Some(deepCopyLearningTree(node.rightChild.get)), node.split, false, node.stats)
    }
  }
}
