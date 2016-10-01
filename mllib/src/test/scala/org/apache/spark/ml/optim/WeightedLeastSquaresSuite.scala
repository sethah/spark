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
import org.apache.spark.ml.feature.{LabeledPoint, Instance}
import org.apache.spark.ml.linalg.{BLAS, Vectors}
import org.apache.spark.ml.util.TestingUtils._
import org.apache.spark.mllib.util.{LinearDataGenerator, MLlibTestSparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

class WeightedLeastSquaresSuite extends SparkFunSuite with MLlibTestSparkContext {
  import testImplicits._

  private var instances: RDD[Instance] = _
  private var instancesConstLabel: RDD[Instance] = _
  private var datasetWithDenseFeature: RDD[Instance] = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    val rng = scala.util.Random
    rng.setSeed(42L)
    /*
       R code:

       A <- matrix(c(0, 1, 2, 3, 5, 7, 11, 13), 4, 2)
       b <- c(17, 19, 23, 29)
       w <- c(1, 2, 3, 4)
     */
    instances = sc.parallelize(Seq(
      Instance(17.0, 1.0, Vectors.dense(0.0, 5.0).toSparse),
      Instance(19.0, 2.0, Vectors.dense(1.0, 7.0)),
      Instance(23.0, 3.0, Vectors.dense(2.0, 11.0)),
      Instance(29.0, 4.0, Vectors.dense(3.0, 13.0))
    ), 2)

    /*
       R code:

       A <- matrix(c(0, 1, 2, 3, 5, 7, 11, 13), 4, 2)
       b.const <- c(17, 17, 17, 17)
       w <- c(1, 2, 3, 4)
     */
    instancesConstLabel = sc.parallelize(Seq(
      Instance(17.0, 1.0, Vectors.dense(0.0, 5.0).toSparse),
      Instance(17.0, 2.0, Vectors.dense(1.0, 7.0)),
      Instance(17.0, 3.0, Vectors.dense(2.0, 11.0)),
      Instance(17.0, 4.0, Vectors.dense(3.0, 13.0))
    ), 2)
  }

  test("two collinear features ") {
    val singularInstances = sc.parallelize(Seq(
      Instance(1.0, 1.0, Vectors.dense(1.0, 2.0)),
      Instance(2.0, 1.0, Vectors.dense(2.0, 4.0)),
      Instance(3.0, 1.0, Vectors.dense(3.0, 6.0)),
      Instance(4.0, 1.0, Vectors.dense(4.0, 8.0))
    ), 2)

    // cholesky solver does not handle singular input
    intercept[IllegalArgumentException] {
      new WeightedLeastSquares(
        false, regParam = 0.0, elasticNetParam = 0.0, standardizeFeatures = false,
        standardizeLabel = false, solver = NormalEquationSolver.Cholesky).fit(singularInstances)
    }

    // Should not throw an exception since regularization is applied
    new WeightedLeastSquares(
      false, regParam = 1.0, elasticNetParam = 0.0, standardizeFeatures = false,
      standardizeLabel = false, solver = NormalEquationSolver.Cholesky).fit(singularInstances)

    // quasi-newton solvers should handle singular input and make correct predictions
    // TODO: should we test with regularization? I don't think so
    for (fitIntercept <- Seq(false, true)) {
      for (standardization <- Seq(false, true)) {
        val singularModel = new WeightedLeastSquares(fitIntercept, regParam = 0.0,
          elasticNetParam = 0.0, standardizeFeatures = standardization,
          standardizeLabel = standardization,
          NormalEquationSolver.QuasiNewton).fit(singularInstances)
        singularInstances.collect().foreach { case Instance(l, w, f) =>
          val pred = BLAS.dot(singularModel.coefficients, f) + singularModel.intercept
          assert(pred ~== l absTol 1e-6)
        }
      }
    }


  }

  test("WLS against lm") {
    /*
       R code:

       df <- as.data.frame(cbind(A, b))
       for (formula in c(b ~ . -1, b ~ .)) {
         model <- lm(formula, data=df, weights=w)
         print(as.vector(coef(model)))
       }

       [1] -3.727121  3.009983
       [1] 18.08  6.08 -0.60
     */

    val expected = Seq(
      Vectors.dense(0.0, -3.727121, 3.009983),
      Vectors.dense(18.08, 6.08, -0.60))

    var idx = 0
    for (fitIntercept <- Seq(false, true)) {
      for (standardization <- Seq(false, true)) {
        val wls = new WeightedLeastSquares(fitIntercept, regParam = 0.0, elasticNetParam = 0.0,
          standardizeFeatures = standardization, standardizeLabel = standardization).fit(instances)
        val actual = Vectors.dense(wls.intercept, wls.coefficients(0), wls.coefficients(1))
        assert(actual ~== expected(idx) absTol 1e-4)
      }
      idx += 1
    }
  }

  test("WLS against lm when label is constant and no regularization") {
    /*
       R code:

       df.const.label <- as.data.frame(cbind(A, b.const))
       for (formula in c(b.const ~ . -1, b.const ~ .)) {
         model <- lm(formula, data=df.const.label, weights=w)
         print(as.vector(coef(model)))
       }

      [1] -9.221298  3.394343
      [1] 17  0  0
    */

    val expected = Seq(
      Vectors.dense(0.0, -9.221298, 3.394343),
      Vectors.dense(17.0, 0.0, 0.0))

    var idx = 0
    for (fitIntercept <- Seq(false, true)) {
      for (standardization <- Seq(false, true)) {
        val wls = new WeightedLeastSquares(fitIntercept, regParam = 0.0, elasticNetParam = 0.0,
          standardizeFeatures = standardization,
          standardizeLabel = standardization).fit(instancesConstLabel)
        val actual = Vectors.dense(wls.intercept, wls.coefficients(0), wls.coefficients(1))
        assert(actual ~== expected(idx) absTol 1e-4)
      }
      idx += 1
    }
  }

  test("WLS with regularization when label is constant") {
    // if regParam is non-zero and standardization is true, the problem is ill-defined and
    // an exception is thrown.
    // TODO: What about when regularization is zero?
    for (solver <- Seq("auto", "cholesky", "quasi-newton")) {
      val wls = new WeightedLeastSquares(
        fitIntercept = false, regParam = 0.1, elasticNetParam = 0.0, standardizeFeatures = true,
        standardizeLabel = true, solver = solver)
      intercept[IllegalArgumentException]{
        wls.fit(instancesConstLabel)
      }
    }
  }

  test("WLS against glmnet with L1") {
    /*

       for (intercept in c(FALSE, TRUE)) {
         for (lambda in c(0.1, 0.5, 1.0)) {
           for (standardize in c(FALSE, TRUE)) {
             for (alpha in c(0.1, 0.5, 1.0)) {
               model <- glmnet(A, b, weights=w, intercept=intercept, lambda=lambda,
                             standardize=standardize, alpha=alpha, thresh=1E-14)
               print(as.vector(coef(model)))
             }
           }
         }
       }
      [1]  0.000000 -3.292821  2.921188
      [1]  0.000000 -3.230854  2.908484
      [1]  0.000000 -3.145586  2.891014
      [1]  0.000000 -2.919246  2.841724
      [1]  0.000000 -2.938323  2.846369
      [1]  0.000000 -2.965397  2.852838
      [1]  0.000000 -2.137858  2.684464
      [1]  0.000000 -1.680094  2.590844
      [1]  0.0000000 -0.8194631  2.4151405
      [1]  0.0000000 -0.9608375  2.4301013
      [1]  0.0000000 -0.6187922  2.3634907
      [1] 0.000000 0.000000 2.240811
      [1]  0.000000 -1.346573  2.521293
      [1]  0.0000000 -0.3680456  2.3212362
      [1] 0.000000 0.000000 2.244406
      [1] 0.000000 0.000000 2.219816
      [1] 0.000000 0.000000 2.223694
      [1] 0.00000 0.00000 2.22861
      [1] 13.5631592  3.2811513  0.3725517
      [1] 13.6953934  3.3336271  0.3497454
      [1] 13.9600276  3.4600170  0.2999941
      [1] 14.2389889  3.6589920  0.2349065
      [1] 15.2374080  4.2119643  0.0325638
      [1] 15.4  4.3  0.0
      [1] 10.442365  1.246065  1.063991
      [1] 8.9580718 0.1938471 1.4090610
      [1] 8.865672 0.000000 1.455224
      [1] 13.0430927  2.4927151  0.5741805
      [1] 13.814429  2.722027  0.455915
      [1] 16.2  3.9  0.0
      [1] 9.8904768 0.7574694 1.2110177
      [1] 9.072226 0.000000 1.435363
      [1] 9.512438 0.000000 1.393035
      [1] 13.3677796  2.1721216  0.6046132
      [1] 14.2554457  2.2285185  0.5084151
      [1] 17.2  3.4  0.0
      */

    val expected = Seq(
      Vectors.dense(0, -3.2928206726474, 2.92118822588649),
      Vectors.dense(0, -3.23085414359003, 2.90848366035008),
      Vectors.dense(0, -3.14558628299477, 2.89101408157209),
      Vectors.dense(0, -2.91924558816421, 2.84172398097327),
      Vectors.dense(0, -2.93832343383477, 2.84636891947663),
      Vectors.dense(0, -2.96539689593024, 2.85283836322185),
      Vectors.dense(0, -2.13785756976542, 2.68446351346705),
      Vectors.dense(0, -1.68009377560774, 2.59084422793154),
      Vectors.dense(0, -0.819463123385533, 2.41514053108346),
      Vectors.dense(0, -0.960837488151064, 2.43010130999756),
      Vectors.dense(0, -0.618792151647599, 2.36349074148962),
      Vectors.dense(0, 0, 2.24081114726441),
      Vectors.dense(0, -1.34657309253953, 2.52129296638512),
      Vectors.dense(0, -0.368045602821844, 2.32123616258871),
      Vectors.dense(0, 0, 2.24440619621343),
      Vectors.dense(0, 0, 2.21981559944924),
      Vectors.dense(0, 0, 2.22369447413621),
      Vectors.dense(0, 0, 2.22861024633605),
      Vectors.dense(13.5631591827557, 3.28115132060568, 0.372551747695477),
      Vectors.dense(13.6953934007661, 3.3336271417751, 0.349745414969587),
      Vectors.dense(13.960027608754, 3.46001702257532, 0.29999407173994),
      Vectors.dense(14.2389889013085, 3.65899196445023, 0.234906458633754),
      Vectors.dense(15.2374079667397, 4.21196428071551, 0.0325637953681963),
      Vectors.dense(15.4, 4.3, 0),
      Vectors.dense(10.4423647474653, 1.24606545153166, 1.06399080283378),
      Vectors.dense(8.95807177856822, 0.193847088148233, 1.4090609658784),
      Vectors.dense(8.86567164179104, 0, 1.45522388059702),
      Vectors.dense(13.0430927453034, 2.49271514356687, 0.574180477650271),
      Vectors.dense(13.8144287399675, 2.72202744354555, 0.455915035859752),
      Vectors.dense(16.2, 3.9, 0),
      Vectors.dense(9.89047681835741, 0.757469417613661, 1.21101772561685),
      Vectors.dense(9.07222551185964, 0, 1.43536293155196),
      Vectors.dense(9.51243781094527, 0, 1.39303482587065),
      Vectors.dense(13.3677796362763, 2.17212164262107, 0.604613180623227),
      Vectors.dense(14.2554457236073, 2.22851848830683, 0.508415124978748),
      Vectors.dense(17.2, 3.4, 0)
      )

    var idx = 0
    for (fitIntercept <- Seq(false, true);
         regParam <- Seq(0.1, 0.5, 1.0);
         standardizeFeatures <- Seq(false, true);
         elasticNetParam <- Seq(0.1, 0.5, 1.0)) {
      val wls = new WeightedLeastSquares(
        fitIntercept, regParam, elasticNetParam = elasticNetParam, standardizeFeatures,
          standardizeLabel = true)
        .fit(instances)
      val actual = Vectors.dense(wls.intercept, wls.coefficients(0), wls.coefficients(1))
      println(actual, regParam)
      assert(actual ~== expected(idx) absTol 1e-4)
      idx += 1
    }
  }


test("WLS against glmnet") {
    /*
       R code:

       library(glmnet)

       for (intercept in c(FALSE, TRUE)) {
         for (lambda in c(0.0, 0.1, 1.0)) {
           for (standardize in c(FALSE, TRUE)) {
             model <- glmnet(A, b, weights=w, intercept=intercept, lambda=lambda,
                             standardize=standardize, alpha=0, thresh=1E-14)
             print(as.vector(coef(model)))
           }
         }
       }

       [1]  0.000000 -3.727117  3.009982
       [1]  0.000000 -3.727117  3.009982
       [1]  0.000000 -3.307532  2.924206
       [1]  0.000000 -2.914790  2.840627
       [1]  0.000000 -1.526575  2.558158
       [1] 0.00000000 0.06984238 2.20488344
       [1] 18.0799727  6.0799832 -0.5999941
       [1] 18.0799727  6.0799832 -0.5999941
       [1] 13.5356178  3.2714044  0.3770744
       [1] 14.064629  3.565802  0.269593
       [1] 10.1238013  0.9708569  1.1475466
       [1] 13.1860638  2.1761382  0.6213134
     */

    val expected = Seq(
      Vectors.dense(0.0, -3.727117, 3.009982),
      Vectors.dense(0.0, -3.727117, 3.009982),
      Vectors.dense(0.0, -3.307532, 2.924206),
      Vectors.dense(0.0, -2.914790, 2.840627),
      Vectors.dense(0.0, -1.526575, 2.558158),
      Vectors.dense(0.0, 0.06984238, 2.20488344),
      Vectors.dense(18.0799727, 6.0799832, -0.5999941),
      Vectors.dense(18.0799727, 6.0799832, -0.5999941),
      Vectors.dense(13.5356178, 3.2714044, 0.3770744),
      Vectors.dense(14.064629, 3.565802, 0.269593),
      Vectors.dense(10.1238013, 0.9708569, 1.1475466),
      Vectors.dense(13.1860638, 2.1761382, 0.6213134))

    var idx = 0
    for (fitIntercept <- Seq(false, true);
         regParam <- Seq(0.0, 0.1, 1.0);
         standardizeFeatures <- Seq(false, true)) {
      for (solver <- Seq("cholesky", "quasi-newton")) {
        val wls = new WeightedLeastSquares(
          fitIntercept, regParam, elasticNetParam = 0.0, standardizeFeatures,
          standardizeLabel = true)
          .fit(instances)
        val actual = Vectors.dense(wls.intercept, wls.coefficients(0), wls.coefficients(1))
        assert(actual ~== expected(idx) absTol 1e-4)
      }
      idx += 1
    }
  }
}
