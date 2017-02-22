///*
// * Licensed to the Apache Software Foundation (ASF) under one or more
// * contributor license agreements.  See the NOTICE file distributed with
// * this work for additional information regarding copyright ownership.
// * The ASF licenses this file to You under the Apache License, Version 2.0
// * (the "License"); you may not use this file except in compliance with
// * the License.  You may obtain a copy of the License at
// *
// *    http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS,
// * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// * See the License for the specific language governing permissions and
// * limitations under the License.
// */
//package org.apache.spark.ml.classification
//
//import breeze.optimize.DiffFunction
//import org.apache.spark.ml.linalg.BLAS
//import org.apache.log4j.{Level, Logger}
//import org.apache.spark.broadcast.Broadcast
//import breeze.linalg.{DenseVector => BDV}
//import org.apache.spark.ml.feature.Instance
//import org.apache.spark.ml.linalg._
//import org.apache.spark.ml.optim.{EMSOCostFunction, WeightedLeastSquares}
//import org.apache.spark.ml.optim.optimizers.{IterativeMinimizerState, LBFGS}
//import org.apache.spark.ml.regression.LeastSquaresAggregator
//import org.apache.spark.rdd.RDD
//import org.apache.spark.sql.{Dataset, SparkSession}
//
//import scala.collection.mutable
//
//object EMSOOptimization {
//  def run(rdd: RDD[Instance]): Unit = {
//    val numFeatures = rdd.first().features.size
//    val initialWeights = Vectors.zeros(numFeatures)
//    println(initialWeights)
//    var workingWeights = initialWeights
//    for (iter <- 0 until 20) {
//      val bw = rdd.context.broadcast(workingWeights)
//      val partitionWeights = rdd.mapPartitions { iterator =>
//        val w = bw.value.toArray
//        val oldW = bw.value
//        val gamma = 0.1
//        val optimizer = new LBFGS()
//        val costFun = new EMSOCostFunction(iterator.toIterable, oldW, gamma)
//        val optIterations = optimizer.iterations(costFun, Vectors.dense(w))
//
//        var lastIter: IterativeMinimizerState[Vector] = null
//        val arrayBuilder = mutable.ArrayBuilder.make[Double]
//        while (optIterations.hasNext) {
//          lastIter = optIterations.next()
//          arrayBuilder += lastIter.loss
//        }
//
////        val eta = 0.05
////        iterator.foreach { instance =>
////          val features = instance.features.toArray :+ 1.0
////          val yhat = BLAS.dot(Vectors.dense(features), Vectors.dense(w))
////          val err = -(instance.label - yhat)
////          var i = 0
////          while (i < numFeatures) {
////            w(i) -= eta * (err * features(i) + gamma * (w(i) - oldW(i)))
////            i += 1
////          }
////          w(numFeatures) -= eta * (err + gamma * (w(numFeatures) - oldW(numFeatures)))
////        }
//        Iterator.single(lastIter.params)
//      }.collect()
//      val coeff = Vectors.zeros(numFeatures)
//      partitionWeights.foreach { wts =>
////        println(iter, wts)
//        BLAS.axpy(1.0, wts, coeff)
//      }
//      BLAS.scal(1 / rdd.getNumPartitions.toDouble, coeff)
//      workingWeights = coeff
//      println(workingWeights)
//    }
//  }
//
//// class LocalLeastSquaresCostFun(
////                                   instances: Iterable[Instance],
////                                   labelStd: Double,
////                                   labelMean: Double,
////                                   fitIntercept: Boolean,
////                                   standardization: Boolean,
////                                   bcFeaturesStd: Broadcast[Array[Double]],
////                                   bcFeaturesMean: Broadcast[Array[Double]],
////                                   effectiveL2regParam: Double,
////                                   aggregationDepth: Int) extends DiffFunction[BDV[Double]] {
////
////  override def calculate(coefficients: BDV[Double]): (Double, BDV[Double]) = {
////    val coeffs = Vectors.fromBreeze(coefficients)
////
////    val agg = new LeastSquaresAggregator(bc)
////    val leastSquaresAggregator = {
////      val seqOp = (c: LeastSquaresAggregator, instance: Instance) => c.add(instance)
////      val combOp = (c1: LeastSquaresAggregator, c2: LeastSquaresAggregator) => c1.merge(c2)
////
////      instances.treeAggregate(
////        new LeastSquaresAggregator(bcCoeffs, labelStd, labelMean, fitIntercept, bcFeaturesStd,
////          bcFeaturesMean))(seqOp, combOp, aggregationDepth)
////    }
////
////    val totalGradientArray = leastSquaresAggregator.gradient.toArray
////    bcCoeffs.destroy(blocking = false)
////
////    val regVal = if (effectiveL2regParam == 0.0) {
////      0.0
////    } else {
////      var sum = 0.0
////      coeffs.foreachActive { (index, value) =>
////        // The following code will compute the loss of the regularization; also
////        // the gradient of the regularization, and add back to totalGradientArray.
////        sum += {
////          if (standardization) {
////            totalGradientArray(index) += effectiveL2regParam * value
////            value * value
////          } else {
////            if (localFeaturesStd(index) != 0.0) {
////              // If `standardization` is false, we still standardize the data
////              // to improve the rate of convergence; as a result, we have to
////              // perform this reverse standardization by penalizing each component
////              // differently to get effectively the same objective function when
////              // the training dataset is not standardized.
////              val temp = value / (localFeaturesStd(index) * localFeaturesStd(index))
////              totalGradientArray(index) += effectiveL2regParam * temp
////              value * temp
////            } else {
////              0.0
////            }
////          }
////        }
////      }
////      0.5 * effectiveL2regParam * sum
////    }
////
////    (leastSquaresAggregator.loss + regVal, new BDV(totalGradientArray))
////  }
//
//
//   class LeastSquaresAggregator(
//                                 coefficients: Vector,
//                                 labelStd: Double,
//                                 labelMean: Double,
//                                 fitIntercept: Boolean,
//                                 bcFeaturesStd: Broadcast[Array[Double]],
//                                 bcFeaturesMean: Broadcast[Array[Double]]) extends Serializable {
//
//     private var totalCnt: Long = 0L
//     private var weightSum: Double = 0.0
//     private var lossSum = 0.0
//
//     private val dim = coefficients.size
//     // make transient so we do not serialize between aggregation stages
//     @transient private lazy val featuresStd = bcFeaturesStd.value
//     @transient private lazy val effectiveCoefAndOffset = {
//       val coefficientsArray = coefficients.toArray.clone()
//       val featuresMean = bcFeaturesMean.value
//       var sum = 0.0
//       var i = 0
//       val len = coefficientsArray.length
//       while (i < len) {
//         if (featuresStd(i) != 0.0) {
//           coefficientsArray(i) /=  featuresStd(i)
//           sum += coefficientsArray(i) * featuresMean(i)
//         } else {
//           coefficientsArray(i) = 0.0
//         }
//         i += 1
//       }
//       val offset = if (fitIntercept) labelMean / labelStd - sum else 0.0
//       (Vectors.dense(coefficientsArray), offset)
//     }
//     // do not use tuple assignment above because it will circumvent the @transient tag
//     @transient private lazy val effectiveCoefficientsVector = effectiveCoefAndOffset._1
//     @transient private lazy val offset = effectiveCoefAndOffset._2
//
//     private val gradientSumArray = Array.ofDim[Double](dim)
//
//     /**
//      * Add a new training instance to this LeastSquaresAggregator, and update the loss and gradient
//      * of the objective function.
//      *
//      * @param instance The instance of data point to be added.
//      * @return This LeastSquaresAggregator object.
//      */
//     def add(instance: Instance): this.type = {
//       instance match { case Instance(label, weight, features) =>
//         require(dim == features.size, s"Dimensions mismatch when adding new sample." +
//           s" Expecting $dim but got ${features.size}.")
//         require(weight >= 0.0, s"instance weight, $weight has to be >= 0.0")
//
//         if (weight == 0.0) return this
//
//         val diff = BLAS.dot(features, effectiveCoefficientsVector) - label / labelStd + offset
//
//         if (diff != 0) {
//           val localGradientSumArray = gradientSumArray
//           val localFeaturesStd = featuresStd
//           features.foreachActive { (index, value) =>
//             if (localFeaturesStd(index) != 0.0 && value != 0.0) {
//               localGradientSumArray(index) += weight * diff * value / localFeaturesStd(index)
//             }
//           }
//           lossSum += weight * diff * diff / 2.0
//         }
//
//         totalCnt += 1
//         weightSum += weight
//         this
//       }
//     }
//
//     /**
//      * Merge another LeastSquaresAggregator, and update the loss and gradient
//      * of the objective function.
//      * (Note that it's in place merging; as a result, `this` object will be modified.)
//      *
//      * @param other The other LeastSquaresAggregator to be merged.
//      * @return This LeastSquaresAggregator object.
//      */
//     def merge(other: LeastSquaresAggregator): this.type = {
//       require(dim == other.dim, s"Dimensions mismatch when merging with another " +
//         s"LeastSquaresAggregator. Expecting $dim but got ${other.dim}.")
//
//       if (other.weightSum != 0) {
//         totalCnt += other.totalCnt
//         weightSum += other.weightSum
//         lossSum += other.lossSum
//
//         var i = 0
//         val localThisGradientSumArray = this.gradientSumArray
//         val localOtherGradientSumArray = other.gradientSumArray
//         while (i < dim) {
//           localThisGradientSumArray(i) += localOtherGradientSumArray(i)
//           i += 1
//         }
//       }
//       this
//     }
//
//     def count: Long = totalCnt
//
//     def loss: Double = {
//       require(weightSum > 0.0, s"The effective number of instances should be " +
//         s"greater than 0.0, but $weightSum.")
//       lossSum / weightSum
//     }
//
//     def gradient: Vector = {
//       require(weightSum > 0.0, s"The effective number of instances should be " +
//         s"greater than 0.0, but $weightSum.")
//       val result = Vectors.dense(gradientSumArray.clone())
//       BLAS.scal(1.0 / weightSum, result)
//       result
//     }
//   }
//}
