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
//package org.apache.spark.ml.optim.optimizers
//
//import org.apache.spark.ml.linalg.{BLAS, DenseVector}
//import org.apache.spark.ml.optim.DifferentiableFunction
//import org.apache.spark.ml.param.ParamMap
//
//class AdaGrad(override val uid: String, stepSize: Double)
//             (implicit space: NormedInnerProductSpace[DenseVector, Double])
//  extends FirstOrderOptimizer[DenseVector] {
//
//  private val rho = 0.9
//
//  override def copy(extra: ParamMap): AdaGrad = {
//    new AdaGrad(uid, stepSize)
//  }
//
//  type History = AdaGrad.AdaHistory
//
//  def initialHistory(lossFunction: DifferentiableFunction[DenseVector],
//                     initialParams: DenseVector): History = {
//    // TODO
//    val size = initialParams.size
//    AdaGrad.AdaHistory(new DenseVector(new Array[Double](size)), 0L,
//      new DenseVector(new Array[Double](size)), 0L, rho)
////    (new DenseVector(Array.fill(initialParams.size)(0.0)), 0L)
//  }
//
//  def converged(state: State): Boolean = {
//    state.iter > 100
//  }
//
//  def updateHistory(
//                     position: DenseVector,
//                     gradient: DenseVector,
//                     value: Double,
//                     state: State): History = {
//    // TODO
////    val nextHist = Array.tabulate(position.size) { i =>
////      state.history._1(i) + gradient(i) * gradient(i)
////    }
////    val tmp = new DenseVector(nextHist)
//////    println(tmp)
////    (tmp, state.history._2 + 1L)
//    val tmp1 = state.history.updateGradient(gradient)
//    val update = tmp1.computeUpdate(gradient)
//    tmp1.updateUpdate(update)
//  }
//
//  def takeStep(position: DenseVector, stepDirection: DenseVector,
//               stepSize: Double, history: History): DenseVector = {
//    println(history.gg, history.xx)
//    // compute Eg
//    val newHistory = history.updateGradient(stepDirection)
//    // compute RMSg
//    // compute update = -RMSx / RMSg * g
//    val update = newHistory.computeUpdate(stepDirection)
//    // return position + update
////    val stepped = new DenseVector(new Array[Double](position.size))
//
//    val next = space.combine(Seq((position, 1.0), (update, 1.0)))
//    println(next)
//    next
//  }
//
//  def chooseStepSize(lossFunction: DifferentiableFunction[DenseVector], direction: DenseVector,
//                     state: State): Double = {
//    stepSize
//  }
//
//  def chooseDescentDirection(state: State): DenseVector = {
//    space.combine(Seq((state.gradient, -1.0)))
//  }
//
//}
//
//object AdaGrad {
//  case class AdaHistory(gg: DenseVector, ng: Long, xx: DenseVector, nx: Long, rho: Double) {
//    val epsilon = 1e-2
//    def updateGradient(g: DenseVector): AdaHistory = {
//      val tmp = new DenseVector(g.values.map((1.0 - rho) * math.pow(_, 2)))
//      if (ng != 0) BLAS.axpy(rho / ng, gg, tmp)
//      AdaHistory(tmp, ng + 1, xx, nx, rho)
//    }
//    def updateUpdate(delta: DenseVector): AdaHistory = {
//      val tmp = new DenseVector(delta.values.map((1.0 - rho) * math.pow(_, 2)))
//      if (nx != 0) BLAS.axpy(rho / nx, xx, tmp)
//      AdaHistory(gg, ng, tmp, nx + 1, rho)
//    }
//
//    private def rms(x: DenseVector, n: Long): DenseVector = {
//      new DenseVector(x.values.map(y => math.sqrt(y / n + epsilon)))
//    }
//
//    def rmsGradient: DenseVector = rms(gg, ng)
//
//    def rmsUpdate: DenseVector = rms(xx, nx)
//
//    def computeUpdate(g: DenseVector): DenseVector = {
//      require(ng == nx + 1, "gradient should have one more update than delta x")
//      if (nx == 0) {
//        new DenseVector(g.values.map(1.0 * _))
//      } else {
//        val ggValues = gg.values
//        val xxValues = xx.values
//        val gValues = g.values
//        var i = 0
//        val updateValues = new Array[Double](gg.size)
//        while (i < gg.size) {
//          updateValues(i) = math.sqrt((xxValues(i) / nx + epsilon) / (ggValues(i) / ng + epsilon)) *
//            gValues(i)
//          i += 1
//        }
//        new DenseVector(updateValues)
//      }
//    }
//  }
//}
