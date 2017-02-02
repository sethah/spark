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

import org.apache.spark.ml.param.{Param, Params}

trait HasL1Reg extends Params {

 /**
  * Param for maximum number of iterations (&gt;= 0).
  *
  * @group param
  */
 final val l1RegFunc: Param[Int => Double] = new Param(this, "l1RegFunc",
  "function for applying L1 regularization to parameters.")

 /** @group getParam */
 final def getL1RegFunc: Int => Double = $(l1RegFunc)

}
