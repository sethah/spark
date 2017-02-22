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
//package org.apache.spark.ml.optim
//
//import org.apache.spark.ml.feature.Instance
//import org.apache.spark.ml.linalg._
//import org.apache.spark.ml.optim.optimizers.{IterativeMinimizer, IterativeMinimizerState, Minimizer}
//import org.apache.spark.ml.param.ParamMap
//import org.apache.spark.rdd.RDD
//
//import scala.collection.mutable
//
//class EMSOOptimizer(val uid: String, instances: RDD[Instance],
//                    localMinimizer: Minimizer[Vector, DifferentiableFunction[Vector]])
//  extends Minimizer[Vector, DifferentiableFunction[Vector]] {
//  // the optimizer we pass to spark ml is supposed to take the differentiable function we give it
//  // which is one that makes passes over the data. Here we want the differentiable function to
//  // be one that just computes locally
//  // what I want to do: lr.setOptimizer(___).fit(df)
//  // the above uses the internal cost function defined by spark, which is strictly an all-reduce
//  // style full gradient update. You cannot change it. So, we would need to do something like
//  /*
//    - paradigm 1 -  an optimizer takes a function and an initial argument and gives back
//      optimized parameters. Very general
//    - for fleixibility it would be cool if I could just pass in a different optimizer and have it
//      automatically do EMSO, or SGD, or whatever. In optimizer iterations we would just repeatedly
//      get back the partition updates, and merge them together. We could also get back the loss
//      Then what is the cost function? It is a function which maps partitions, returns the loss,
//      and the parameters
//   */
//  override def minimize(lossFunction: DifferentiableFunction[Vector],
//                        initialParams: Vector): Vector = {
//
//    val numFeatures = instances.first().features.size
//    val initialWeights = Vectors.zeros(numFeatures)
//    println(initialWeights)
//    var workingWeights = initialWeights
//    for (iter <- 0 until 20) {
//      val bw = instances.context.broadcast(workingWeights)
//      val partitionWeights = instances.mapPartitions { iterator =>
//        val w = bw.value.toArray
//        val oldW = bw.value
//        val gamma = 0.1
//        val costFun = new EMSOCostFunction(iterator.toIterable, oldW, gamma)
//        Iterator.single(localMinimizer.minimize(costFun, Vectors.dense(w)))
//      }.collect()
//      val coeff = Vectors.zeros(numFeatures)
//      partitionWeights.foreach { wts =>
//        BLAS.axpy(1.0, wts, coeff)
//      }
//      BLAS.scal(1 / instances.getNumPartitions.toDouble, coeff)
//      workingWeights = coeff
//      println(workingWeights)
//    }
//    workingWeights
//  }
//
//  override def copy(extra: ParamMap): this.type = defaultCopy(extra)
//}
//
////class EMSOCostFun(instances: RDD[Instance],
////                  localMinimizer: IterativeMinimizer[Vector, DifferentiableFunction[Vector],
////                    IterativeMinimizerState[Vector]],
////                  localCostFun: DifferentiableFunction[Vector])
////  extends (Vector => Double) {
//
////  def getLossAndNewWeights(currentWeights: Vector): (Double, Vector) = {
////    val bw = instances.context.broadcast(currentWeights)
////    val numFeatures = currentWeights.size
////    val partitionWeights = instances.mapPartitions { iterator =>
////      val w = bw.value.toArray
////      val oldW = bw.value
////      val gamma = 0.1
////      val costFun = new EMSOCostFunction(iterator.toIterable, oldW, gamma)
////      val optIterations = localMinimizer.iterations(costFun, oldW)
////
////      var lastIter: IterativeMinimizerState[Vector] = null
////      val arrayBuilder = mutable.ArrayBuilder.make[Double]
////      while (optIterations.hasNext) {
////        lastIter = optIterations.next()
////        arrayBuilder += lastIter.loss
////      }
////      Iterator.single((lastIter.params, arrayBuilder.result().head))
////    }.collect()
////    val coeff = Vectors.zeros(numFeatures)
////    partitionWeights.foreach { wts =>
////      BLAS.axpy(1.0, wts, coeff)
////    }
////    BLAS.scal(1 / instances.getNumPartitions.toDouble, coeff)
////  }
//
////}
//
//class EMSOCostFunction(instances: Iterable[Instance],
//                       oldWeights: Vector,
//                       gamma: Double)
//  extends DifferentiableFunction[Vector] {
//
//  override def doCompute(x: Vector): (Double, Vector) = {
//    val numFeatures = x.size
//    val gradient = new LeastSquaresGradient
//
//    val seqOp = (c: (Vector, Double), v: (Double, Vector)) =>
//      (c, v) match {
//        case ((grad, loss), (label, features)) =>
//          val denseGrad = grad.toDense
//          val l = gradient.compute(features, label, x, denseGrad)
//          (denseGrad, loss + l)
//      }
//
//    val combOp = (c1: (Vector, Double), c2: (Vector, Double)) =>
//      (c1, c2) match { case ((grad1, loss1), (grad2, loss2)) =>
//        val denseGrad1 = grad1.toDense
//        val denseGrad2 = grad2.toDense
//        BLAS.axpy(1.0, denseGrad2, denseGrad1)
//        (denseGrad1, loss1 + loss2)
//      }
//
//    val zeroSparseVector = Vectors.sparse(numFeatures, Seq())
//    val (gradientSum, lossSum, count) = instances.foldLeft((zeroSparseVector, 0.0, 0)) {
//      case ((grad, loss, c), instance) =>
//        val (g, l) = seqOp((grad, loss), (instance.label, instance.features))
//        (g, l, c + 1)
//    }
//    var j = 0
//    val gradientArray = gradientSum.toArray
//    while (j < numFeatures) {
//      gradientArray(j) += gamma * (x(j) - oldWeights(j))
//      j += 1
//    }
//    BLAS.scal(1.0 / count, gradientSum)
////    println("grad", gradientSum, count)
//    (lossSum / count.toDouble, gradientSum)
//
//  }
//}
//class LeastSquaresGradient {
////  def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
////    val diff = BLAS.dot(data, weights) - label
////    val loss = diff * diff / 2.0
////    val gradient = data.copy
////    BLAS.scal(diff, gradient)
////    (gradient, loss)
////  }
//
//  def compute(
//        data: Vector,
//        label: Double,
//        weights: Vector,
//        cumGradient: Vector): Double = {
//    val diff = BLAS.dot(data, weights) - label
//    BLAS.axpy(diff, data, cumGradient)
//    diff * diff / 2.0
//  }
//}
