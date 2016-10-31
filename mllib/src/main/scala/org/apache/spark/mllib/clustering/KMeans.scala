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

package org.apache.spark.mllib.clustering

import org.apache.spark.ml.tree.impl.TimeTracker

import scala.collection.mutable

import org.apache.spark.annotation.Since
import org.apache.spark.internal.Logging
import org.apache.spark.ml.clustering.{KMeans => NewKMeans}
import org.apache.spark.ml.util.Instrumentation
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.BLAS.{axpy, gemm, scal}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.Utils
import org.apache.spark.util.random.XORShiftRandom

/**
 * K-means clustering with a k-means++ like initialization mode
 * (the k-means|| algorithm by Bahmani et al).
 *
 * This is an iterative algorithm that will make multiple passes over the data, so any RDDs given
 * to it should be cached by the user.
 */
@Since("0.8.0")
class KMeans private (
    private var k: Int,
    private var maxIterations: Int,
    private var initializationMode: String,
    private var initializationSteps: Int,
    private var epsilon: Double,
    private var seed: Long,
    private var blockSize: Int) extends Serializable with Logging {

  /**
   * Constructs a KMeans instance with default parameters: {k: 2, maxIterations: 20,
   * initializationMode: "k-means||", initializationSteps: 5, epsilon: 1e-4, seed: random,
   * blockSize: 4096}.
   */
  @Since("0.8.0")
  def this() = this(2, 20, KMeans.K_MEANS_PARALLEL, 5, 1e-4, Utils.random.nextLong(), 4096)

  /**
   * Number of clusters to create (k).
   */
  @Since("1.4.0")
  def getK: Int = k

  /**
   * Set the number of clusters to create (k). Default: 2.
   */
  @Since("0.8.0")
  def setK(k: Int): this.type = {
    require(k > 0,
      s"Number of clusters must be positive but got ${k}")
    this.k = k
    this
  }

  /**
   * Maximum number of iterations allowed.
   */
  @Since("1.4.0")
  def getMaxIterations: Int = maxIterations

  /**
   * Set maximum number of iterations allowed. Default: 20.
   */
  @Since("0.8.0")
  def setMaxIterations(maxIterations: Int): this.type = {
    require(maxIterations >= 0,
      s"Maximum of iterations must be nonnegative but got ${maxIterations}")
    this.maxIterations = maxIterations
    this
  }

  /**
   * The initialization algorithm. This can be either "random" or "k-means||".
   */
  @Since("1.4.0")
  def getInitializationMode: String = initializationMode

  /**
   * Set the initialization algorithm. This can be either "random" to choose random points as
   * initial cluster centers, or "k-means||" to use a parallel variant of k-means++
   * (Bahmani et al., Scalable K-Means++, VLDB 2012). Default: k-means||.
   */
  @Since("0.8.0")
  def setInitializationMode(initializationMode: String): this.type = {
    KMeans.validateInitMode(initializationMode)
    this.initializationMode = initializationMode
    this
  }

  /**
   * This function has no effect since Spark 2.0.0.
   */
  @Since("1.4.0")
  @deprecated("This has no effect and always returns 1", "2.1.0")
  def getRuns: Int = {
    logWarning("Getting number of runs has no effect since Spark 2.0.0.")
    1
  }

  /**
   * This function has no effect since Spark 2.0.0.
   */
  @Since("0.8.0")
  @deprecated("This has no effect", "2.1.0")
  def setRuns(runs: Int): this.type = {
    logWarning("Setting number of runs has no effect since Spark 2.0.0.")
    this
  }

  /**
   * Number of steps for the k-means|| initialization mode
   */
  @Since("1.4.0")
  def getInitializationSteps: Int = initializationSteps

  /**
   * Set the number of steps for the k-means|| initialization mode. This is an advanced
   * setting -- the default of 5 is almost always enough. Default: 5.
   */
  @Since("0.8.0")
  def setInitializationSteps(initializationSteps: Int): this.type = {
    require(initializationSteps > 0,
      s"Number of initialization steps must be positive but got ${initializationSteps}")
    this.initializationSteps = initializationSteps
    this
  }

  /**
   * The distance threshold within which we've consider centers to have converged.
   */
  @Since("1.4.0")
  def getEpsilon: Double = epsilon

  /**
   * Set the distance threshold within which we've consider centers to have converged.
   * If all centers move less than this Euclidean distance, we stop iterating one run.
   */
  @Since("0.8.0")
  def setEpsilon(epsilon: Double): this.type = {
    require(epsilon >= 0,
      s"Distance threshold must be nonnegative but got ${epsilon}")
    this.epsilon = epsilon
    this
  }

  /**
   * The random seed for cluster initialization.
   */
  @Since("1.4.0")
  def getSeed: Long = seed

  /**
   * Set the random seed for cluster initialization.
   */
  @Since("1.4.0")
  def setSeed(seed: Long): this.type = {
    this.seed = seed
    this
  }

  // Initial cluster centers can be provided as a KMeansModel object rather than using the
  // random or k-means|| initializationMode
  private var initialModel: Option[KMeansModel] = None

  /**
   * Set the initial starting point, bypassing the random initialization or k-means||
   * The condition model.k == this.k must be met, failure results
   * in an IllegalArgumentException.
   */
  @Since("1.4.0")
  def setInitialModel(model: KMeansModel): this.type = {
    require(model.k == k, "mismatched cluster count")
    initialModel = Some(model)
    this
  }

  private[spark] def getBlockSize: Int = blockSize

  private[spark] def setBlockSize(blockSize: Int): this.type = {
    this.blockSize = blockSize
    this
  }

  /**
   * Train a K-means model on the given set of points; `data` should be cached for high
   * performance, because this is an iterative algorithm.
   */
  @Since("0.8.0")
  def run(data: RDD[Vector]): KMeansModel = {
    run(data, None)
  }

  private[spark] def run(
      data: RDD[Vector],
      instr: Option[Instrumentation[NewKMeans]]): KMeansModel = {

    if (data.getStorageLevel == StorageLevel.NONE) {
      logWarning("The input data is not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.")
    }

    val zippedData = data.map { x => new VectorWithNorm(x) }

    val initStartTime = System.nanoTime()

    val centers = initialModel match {
      case Some(kMeansCenters) =>
        kMeansCenters.clusterCenters.map(new VectorWithNorm(_))
      case None =>
        if (initializationMode == KMeans.RANDOM) {
          initRandom(zippedData)
        } else {
          initKMeansParallel(zippedData)
        }
    }

    val initTimeInSeconds = (System.nanoTime() - initStartTime) / 1e9
    logInfo(s"Initialization with $initializationMode took " + "%.3f".format(initTimeInSeconds) +
      " seconds.")

    val samplePoint = data.first()
    val dim = samplePoint.size
    val isSparse = samplePoint.isInstanceOf[SparseVector]
    if (isSparse) {
      logWarning("KMeans will be less efficient if the input data is Sparse Vector.")
    }

    // Store data as block and cache it.
    val blockData = zippedData.mapPartitions { iter =>
      iter.grouped(blockSize).map { points =>
        val realSize = points.size
        val pointNormArray = new Array[Double](realSize)
        var numRows = 0

        val pointMatrix = if (isSparse) {
          val colPtrs = new Array[Int](realSize + 1)
          val rowIndices = mutable.ArrayBuilder.make[Int]
          val values = mutable.ArrayBuilder.make[Double]
          var nnz = 0

          points.foreach { point =>
            val sv = point.vector.asInstanceOf[SparseVector]
            sv.foreachActive { (index, value) =>
              rowIndices += index
              values += value
              nnz += 1
            }

            pointNormArray(numRows) = point.norm
            numRows += 1
            colPtrs(numRows) = nnz
          }
          new SparseMatrix(numRows, dim, colPtrs, rowIndices.result(), values.result(), true)
        } else {
          val pointArray = new Array[Double](realSize * dim)
          points.foreach { point =>
            System.arraycopy(point.vector.toArray, 0, pointArray, numRows * dim, dim)
            pointNormArray(numRows) = point.norm
            numRows += 1
          }
          new DenseMatrix(numRows, dim, pointArray, true)
        }

        (pointMatrix, pointNormArray)
      }
    }
    blockData.persist()
    blockData.count()

    val model = runAlgorithm(blockData, centers, instr)

    blockData.unpersist(blocking = false)
    // Warn at the end of the run as well, for increased visibility.
    if (blockData.getStorageLevel == StorageLevel.NONE) {
      logWarning("The input data was not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.")
    }
    model
  }

  /**
   * Implementation of K-Means algorithm.
   */
  private def runAlgorithm(
      data: RDD[(Matrix, Array[Double])],
      centers: Array[VectorWithNorm],
      instr: Option[Instrumentation[NewKMeans]]): KMeansModel = {

    val sc = data.sparkContext
    val dim = centers(0).vector.size
    var done = false
    var costs = 0.0
    var iteration = 0

    instr.foreach(_.logNumFeatures(dim))

    val iterationStartTime = System.nanoTime()

    // Execute Lloyd's algorithm until converged or reached the max number of iterations.
    while (iteration < maxIterations && !done) {
      val tIter0 = System.nanoTime()
      type WeightedPoint = (Vector, Long)
      def mergeContribs(x: WeightedPoint, y: WeightedPoint): WeightedPoint = {
        axpy(1.0, x._1, y._1)
        (y._1, x._2 + y._2)
      }

      val costAccums = sc.doubleAccumulator

      // Construct center array and broadcast it.
      val centersTime0 = System.nanoTime()
      val centerArray = new Array[Double](k * dim)
      val centerNormArray = new Array[Double](k)
      var i = 0
      centers.foreach { center =>
        System.arraycopy(center.vector.toArray, 0, centerArray, i * dim, dim)
        centerNormArray(i) = center.norm
        i += 1
      }
      val centersTime1 = System.nanoTime()
      println(s"centersTime: ${(centersTime1 - centersTime0) / 1e9}")
      val bcCenterArray = sc.broadcast(centerArray)
      val bcCenterNormArray = sc.broadcast(centerNormArray)

      // Find the sum and count of points mapping to each center.
      val flatMapTime0 = System.nanoTime()
      val totalContribs = data.flatMap { case (pointMatrix, pointNormArray) =>
        val timingMap = new mutable.HashMap[String, Long]()
        val t0 = System.nanoTime()
        val thisCenterArray = bcCenterArray.value
        val thisCenterNormArray = bcCenterNormArray.value

        val k = thisCenterNormArray.length
        val numRows = pointMatrix.numRows

        val sums = Array.fill(k)(Vectors.zeros(dim))
        val counts = Array.fill(k)(0L)

        // Construct centers matrix.
        val tCenters0 = System.nanoTime()
        val centerMatrix = new DenseMatrix(dim, k, thisCenterArray)
        val centersWithNorm = Array.tabulate(k) { idx =>
          new VectorWithNorm(Vectors.dense(thisCenterArray.slice(idx * dim, (idx + 1) * dim)),
            thisCenterNormArray(idx))
        }
        val tCenters1 = System.nanoTime()
        timingMap += ("Centers time" -> (tCenters1 - tCenters0))

        // Compute dot product matrix between data and center.
        val tGEMM0 = System.nanoTime()
        val dotProductMatrix = new DenseMatrix(numRows, k, Array.fill(numRows * k)(0.0))
        gemm(1.0, pointMatrix, centerMatrix, 0.0, dotProductMatrix)
        val tGEMM1 = System.nanoTime()
        timingMap += "GEMM time" -> (tGEMM1 - tGEMM0)

        // iterate over blockSize * k and if they pass the numerical precision test, the
        // distance is simply norm1 + norm2 - 2.0 * dotProd
        // if they do not pass the test, call vectors.sqdist
        val findClosestTime0 = System.nanoTime()
        val closest = new Array[Int](numRows)
        val minCosts = Array.fill(numRows)(Double.PositiveInfinity)
        val dotProdValues = dotProductMatrix.values
        i = 0
        while (i < k) {
          val centerNorm = thisCenterNormArray(i)
          val center = centersWithNorm(i).vector
          var j = 0
          while (j < numRows) {
            val pointNorm = pointNormArray(j)
            val normDiff = centerNorm - pointNorm
            val sumSquaredNorm = pointNorm * pointNorm + centerNorm * centerNorm
            val precisionBound1 = 2.0 * MLUtils.EPSILON * sumSquaredNorm /
              (normDiff * normDiff + MLUtils.EPSILON)
            val dist = if (precisionBound1 < 1e-6) {
              sumSquaredNorm - 2.0 * dotProdValues(i * numRows + j)
            } else {
              val point = pointMatrix match {
                case dm: DenseMatrix =>
                  Vectors.dense(dm.values.slice(j * dim, j * dim + dim))
                case sm: SparseMatrix =>
                  throw new Exception("no sparse")
              }
              Vectors.sqdist(center, point)
            }
            if (dist < minCosts(j)) {
              minCosts(j) = dist
              closest(j) = i
            }
            j += 1
          }
          i += 1
        }
        val findClosestTime1 = System.nanoTime()
        timingMap += "Find closest" -> (findClosestTime1 - findClosestTime0)

        val addContribsTime0 = System.nanoTime()
        // add points contributions to the appropriate centers
        pointMatrix.foreachActive { (rowIndex, colIndex, value) =>
          val closestCenter = closest(rowIndex)
          sums(closestCenter).toArray(colIndex) += value
        }
        closest.foreach { i => counts(i) += 1}
        val addContribsTime1 = System.nanoTime()
        timingMap += "Add contribs" -> (addContribsTime1 - addContribsTime0)
        val sumTimes = timingMap.values.sum
        val normedMap = timingMap.map { case (k, v) =>
          k -> v / sumTimes.toDouble
        }
        println(normedMap)
        println(timingMap.map { case (k, v ) => k -> v / 1e9})

//        val contribs = for (j <- 0 until k) yield {
//          (j, (sums(j), counts(j)))
//        }
//        contribs.iterator
        val t1 = System.nanoTime()
        println(s"Flatmap inner time: ${(t1 - t0) / 1e9}")
        counts.indices.filter(counts(_) > 0).map(j => (j, (sums(j), counts(j)))).iterator
      }.reduceByKey(mergeContribs).collectAsMap()
      val flatMapTime1 = System.nanoTime()
      println(s"Flat map time: ${(flatMapTime1 - flatMapTime0) / 1e9}")

      bcCenterArray.destroy(blocking = false)
      bcCenterNormArray.destroy(blocking = false)

      // Update the cluster centers and costs
      done = true
      totalContribs.foreach { case (j, (sum, count)) =>
        scal(1.0 / count, sum)
        val newCenter = new VectorWithNorm(sum)
        if (done && KMeans.fastSquaredDistance(newCenter, centers(j)) > epsilon * epsilon) {
          done = false
        }
        centers(j) = newCenter
      }

      // Update the cluster centers and costs
//      var changed = false
//      var j = 0
//      val trueK = totalContribs.size
//      while (j < trueK) {
//        val (sum, count) = totalContribs(j)
//        if (count != 0) {
//          scal(1.0 / count, sum)
//          val newCenter = new VectorWithNorm(sum)
//          if (KMeans.fastSquaredDistance(newCenter, centers(j)) > epsilon * epsilon) {
//            changed = true
//          }
//          centers(j) = newCenter
//        }
//        j += 1
//      }

      costs = costAccums.value
//      if (!done) {
//        logInfo(s"Run finished in ${iteration + 1} iterations")
//        done = true
//      }
      iteration += 1
      val tIter1 = System.nanoTime()
      println("---------------------")
      println(s"Iteration time:  ${(tIter1 - tIter0) / 1e9}")
      println("---------------------")
    }

    val iterationTimeInSeconds = (System.nanoTime() - iterationStartTime) / 1e9
    logInfo(s"Iterations took " + "%.3f".format(iterationTimeInSeconds) + " seconds.")

    if (iteration == maxIterations) {
      logInfo(s"KMeans reached the max number of iterations: $maxIterations.")
    } else {
      logInfo(s"KMeans converged in $iteration iterations.")
    }

    logInfo(s"The cost is $costs.")

    new KMeansModel(centers.map(_.vector))
  }

  /**
   * Initialize cluster centers at random.
   */
  private def initRandom(data: RDD[VectorWithNorm]): Array[VectorWithNorm] = {
    // Sample all the cluster centers in one pass to avoid repeated scans
    data.takeSample(true, k, new XORShiftRandom(this.seed).nextLong()).toSeq.toArray
  }

  /**
   * Initialize cluster centers using the k-means|| algorithm by Bahmani et al.
   * (Bahmani et al., Scalable K-Means++, VLDB 2012). This is a variant of k-means++ that tries
   * to find with dissimilar cluster centers by starting with a random center and then doing
   * passes where more centers are chosen with probability proportional to their squared distance
   * to the current cluster set. It results in a provable approximation to an optimal clustering.
   *
   * The original paper can be found at http://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf.
   */
  private def initKMeansParallel(data: RDD[VectorWithNorm]): Array[VectorWithNorm] = {
    // Initialize empty centers and point costs.
    val centers = mutable.ArrayBuffer.empty[VectorWithNorm]
    var costs = data.map(_ => Double.PositiveInfinity)

    // Initialize first center to a random point.
    val samples = data.takeSample(true, 1, new XORShiftRandom(this.seed).nextLong())
    // Could be empty if data is empty, fail with a better message early.
    require(samples.nonEmpty, s"No samples available from $data")
    var newCenters = Seq(samples.head.toDense)
    centers ++= newCenters

    // On each step, sample 2 * k points on average with probability proportional
    // to their squared distance from the centers. Note that only distances between points
    // and new centers are computed in each iteration.
    var step = 0
    while (step < initializationSteps) {
      val bcNewCenters = data.context.broadcast(newCenters)
      val preCosts = costs
      costs = data.zip(preCosts).map { case (point, cost) =>
        math.min(KMeans.pointCost(bcNewCenters.value, point), cost)
        }.persist(StorageLevel.MEMORY_AND_DISK)
      val sumCosts = costs.sum()

      bcNewCenters.unpersist(blocking = false)
      preCosts.unpersist(blocking = false)

      val chosen = data.zip(costs).mapPartitionsWithIndex { (index, pointsWithCosts) =>
        val rand = new XORShiftRandom(seed ^ (step << 16) ^ index)
        pointsWithCosts.filter { case (_, c) =>
          rand.nextDouble() < 2.0 * c * k / sumCosts
        }.map(_._1)
      }.collect()
      newCenters = chosen.map(_.toDense)
      centers ++= newCenters
      step += 1
    }

    costs.unpersist(blocking = false)

    // Finally, we might have a set of more than k candidate centers; weigh each
    // candidate by the number of points in the dataset mapping to it and run a local k-means++
    // on the weighted centers to pick just k of them
    val bcCenters = data.context.broadcast(centers)
    val countMap = data.map { p =>
      KMeans.findClosest(bcCenters.value, p)._1
    }.countByValue()

    bcCenters.destroy(blocking = false)

    val myWeights = centers.indices.map(countMap.getOrElse(_, 0L).toDouble).toArray
    LocalKMeans.kMeansPlusPlus(0, centers.toArray, myWeights, k, 30)
  }
}


/**
 * Top-level methods for calling K-means clustering.
 */
@Since("0.8.0")
object KMeans {

  // Initialization mode names
  @Since("0.8.0")
  val RANDOM = "random"
  @Since("0.8.0")
  val K_MEANS_PARALLEL = "k-means||"

  /**
   * Trains a k-means model using the given set of parameters.
   *
   * @param data Training points as an `RDD` of `Vector` types.
   * @param k Number of clusters to create.
   * @param maxIterations Maximum number of iterations allowed.
   * @param runs This param has no effect since Spark 2.0.0.
   * @param initializationMode The initialization algorithm. This can either be "random" or
   *                           "k-means||". (default: "k-means||")
   * @param seed Random seed for cluster initialization. Default is to generate seed based
   *             on system time.
   */
  @Since("1.3.0")
  def train(
      data: RDD[Vector],
      k: Int,
      maxIterations: Int,
      runs: Int,
      initializationMode: String,
      seed: Long): KMeansModel = {
    new KMeans().setK(k)
      .setMaxIterations(maxIterations)
      .setInitializationMode(initializationMode)
      .setSeed(seed)
      .run(data)
  }

  /**
   * Trains a k-means model using the given set of parameters.
   *
   * @param data Training points as an `RDD` of `Vector` types.
   * @param k Number of clusters to create.
   * @param maxIterations Maximum number of iterations allowed.
   * @param runs This param has no effect since Spark 2.0.0.
   * @param initializationMode The initialization algorithm. This can either be "random" or
   *                           "k-means||". (default: "k-means||")
   */
  @Since("0.8.0")
  def train(
      data: RDD[Vector],
      k: Int,
      maxIterations: Int,
      runs: Int,
      initializationMode: String): KMeansModel = {
    new KMeans().setK(k)
      .setMaxIterations(maxIterations)
      .setInitializationMode(initializationMode)
      .run(data)
  }

  /**
   * Trains a k-means model using specified parameters and the default values for unspecified.
   */
  @Since("0.8.0")
  def train(
      data: RDD[Vector],
      k: Int,
      maxIterations: Int): KMeansModel = {
    train(data, k, maxIterations, 1, K_MEANS_PARALLEL)
  }

  /**
   * Trains a k-means model using specified parameters and the default values for unspecified.
   */
  @Since("0.8.0")
  def train(
      data: RDD[Vector],
      k: Int,
      maxIterations: Int,
      runs: Int): KMeansModel = {
    train(data, k, maxIterations, runs, K_MEANS_PARALLEL)
  }

  /**
   * Returns the index of the closest center to the given point, as well as the squared distance.
   */
  private[mllib] def findClosest(
      centers: TraversableOnce[VectorWithNorm],
      point: VectorWithNorm): (Int, Double) = {
    var bestDistance = Double.PositiveInfinity
    var bestIndex = 0
    var i = 0
    centers.foreach { center =>
      // Since `\|a - b\| \geq |\|a\| - \|b\||`, we can use this lower bound to avoid unnecessary
      // distance computation.
      var lowerBoundOfSqDist = center.norm - point.norm
      lowerBoundOfSqDist = lowerBoundOfSqDist * lowerBoundOfSqDist
      if (lowerBoundOfSqDist < bestDistance) {
        val distance: Double = fastSquaredDistance(center, point)
        if (distance < bestDistance) {
          bestDistance = distance
          bestIndex = i
        }
      }
      i += 1
    }
    (bestIndex, bestDistance)
  }

  /**
   * Returns the K-means cost of a given point against the given cluster centers.
   */
  private[mllib] def pointCost(
      centers: TraversableOnce[VectorWithNorm],
      point: VectorWithNorm): Double =
    findClosest(centers, point)._2

  /**
   * Returns the squared Euclidean distance between two vectors computed by
   * [[org.apache.spark.mllib.util.MLUtils#fastSquaredDistance]].
   */
  private[clustering] def fastSquaredDistance(
      v1: VectorWithNorm,
      v2: VectorWithNorm): Double = {
    MLUtils.fastSquaredDistance(v1.vector, v1.norm, v2.vector, v2.norm)
  }

  private[spark] def validateInitMode(initMode: String): Boolean = {
    initMode match {
      case KMeans.RANDOM => true
      case KMeans.K_MEANS_PARALLEL => true
      case _ => false
    }
  }
}

/**
 * A vector with its norm for fast distance computation.
 *
 * @see [[org.apache.spark.mllib.clustering.KMeans#fastSquaredDistance]]
 */
private[clustering]
class VectorWithNorm(val vector: Vector, val norm: Double) extends Serializable {

  def this(vector: Vector) = this(vector, Vectors.norm(vector, 2.0))

  def this(array: Array[Double]) = this(Vectors.dense(array))

  /** Converts the vector to a dense vector. */
  def toDense: VectorWithNorm = new VectorWithNorm(Vectors.dense(vector.toArray), norm)
}
