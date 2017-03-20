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

class DifferentiableFunctionSuite extends SparkFunSuite {

  test("cached differentiable function") {
    var counter = 0
    val testDiffFun = new DifferentiableFunction[Double] {
      override def doCompute(x: Double) = {
        counter += 1
        (2.0 * x, 2.0)
      }
    }
    val cachedTestFun = testDiffFun.cached()
    val testValues = Seq(0, 0, 1, 2, 2, 2, 2, 1, 3)
    testValues.foreach { x =>
      testDiffFun.compute(x)
    }
    assert(counter === 9)
    counter = 0
    testValues.foreach { x =>
      cachedTestFun.compute(x)
    }
    assert(counter === 5)
  }
}
