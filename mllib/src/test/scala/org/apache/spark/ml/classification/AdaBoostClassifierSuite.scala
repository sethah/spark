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

package org.apache.spark.ml.classification

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.attribute.NominalAttribute
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLlibTestSparkContext

class AdaBoostClassifierSuite extends SparkFunSuite with MLlibTestSparkContext {
  test ("decision tree base estimator") {
    val numClasses = 2
    val numIterations = 1
    val data = AdaBoostClassifierSuite.generateOrderedLabeledPoints(3, 10)
    val df = sqlContext.createDataFrame(data)
    val ada = new AdaBoostClassifier().setMaxIter(numIterations)
        .setBaseEstimators(Array(new LogisticRegression()))
    val labelMeta = NominalAttribute.defaultAttr.withName("label")
      .withNumValues(numClasses).toMetadata()
    val model = ada.fit(df.select(df("features"), df("label").as("label", labelMeta)))
    model.transform(df).show()
    model.models(0).transform(df).show()
    model.modelWeights.foreach(println)
    assert(model.models.length === numIterations)
    AdaBoostClassifierSuite.validateClassifier(model, data, 1.0)
  }

  test ("logistic regression base estimator") {

  }

  test ("naive bayes base estimator") {

  }
}

object AdaBoostClassifierSuite {
  def generateOrderedLabeledPoints(numFeatures: Int, numInstances: Int): Array[LabeledPoint] = {
    val arr = new Array[LabeledPoint](numInstances)
    for (i <- 0 until numInstances) {
      val label = if (i < numInstances / 10) {
        0.0
      } else if (i < numInstances / 2) {
        1.0
      } else if (i < 0.9 * numInstances){
        0.0
      } else {
        1.0
      }
      val features = Array.fill[Double](numFeatures)(i.toDouble)
      arr(i) = new LabeledPoint(label, Vectors.dense(features))
    }
    arr
  }

  def generateLinearlySeparableLabeledPoints(numFeatures: Int,
      numInstances: Int): Array[LabeledPoint] = {
    val arr = new Array[LabeledPoint](numInstances)
    for (i <- 0 until numInstances) {
      val label = if (i < numInstances / 10) {
        0.0
      } else if (i < numInstances / 2) {
        1.0
      } else {
        1.0
      }
      val features = Array.fill[Double](numFeatures)(i.toDouble)
      arr(i) = new LabeledPoint(label, Vectors.dense(features))
    }
    arr
  }

  def validateClassifier(
      model: AdaBoostClassificationModel,
      input: Seq[LabeledPoint],
      requiredAccuracy: Double) {
    val predictions = input.map(x => model.predict(x.features))
    val numOffPredictions = predictions.zip(input).count { case (prediction, expected) =>
      prediction != expected.label
    }
    val accuracy = (input.length - numOffPredictions).toDouble / input.length
    assert(accuracy >= requiredAccuracy,
      s"validateClassifier calculated accuracy $accuracy but required $requiredAccuracy.")
  }
}
