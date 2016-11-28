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

package org.apache.spark.ml.optim

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.{BLAS, Vectors, Vector}
import org.apache.spark.ml.util.TestingUtils._
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.optim.OptimizerImplicits._

class MinimizerSuite extends SparkFunSuite with MLlibTestSparkContext {

  test("mytest") {
    val ctx = spark.sqlContext
    import ctx.implicits._
    val df = Seq(Instance(1.0, 1.0, Vectors.dense(1.0, 2.0))).toDF()
//    implicit val imp = CanMathVector
    val optimizer = new GradientDescent[Vector]
    val initialCoef = Vectors.dense(0.2, 0.3)
    val loss = new LeastSquaresCostFun(df, df.count())
    val optimizedCoef = optimizer.optimize(loss, initialCoef)
    println(optimizedCoef)
  }

}
