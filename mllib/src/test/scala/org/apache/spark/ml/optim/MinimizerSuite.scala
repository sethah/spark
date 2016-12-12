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
  import OptimizerImplicits._

  test("mytest") {
    val ctx = spark.sqlContext
    import ctx.implicits._

    /*
    -0.3595531615405413,
 0.9766390364837128,
 0.402341641177549,
 -0.813146282044454,
 -0.8877857476301128
 */
    val x = Array(-0.3595531615405413, 0.9766390364837128, 0.402341641177549,
      -0.813146282044454, -0.8877857476301128)
    val y = Array(-31.745993948575236,
    -28.79914206045906,
    -2.184502944965036,
    28.96679711751562,
    -30.87911690476842)
    val df = x.zip(y).map { case (_x, _y) =>
      Instance(_y, 1.0, Vectors.dense(1.0, _x))
    }.toList.toDF()
//    val df = Seq(
//      Instance(24.0, 1.0, Vectors.dense(-0.3595531615405413)),
//      Instance(21.6, 1.0, Vectors.dense(0.9766390364837128)),
//      Instance(1, 1, Vectors.dense(0.402341641177549))
//    ).toDF()
//    implicit val imp = CanMathVector
//    assert(2.0 === 3.0)
//    implicit val ops = new CanMathOps[Vector]
    val optimizer = new GradientDescent[Vector]()
    val initialCoef = Vectors.dense(0, 0.0)
    val loss = new LeastSquaresCostFun(df, df.count())
    val optimizedCoef = optimizer.optimize(loss, initialCoef)
    println(optimizedCoef)
  }

}
