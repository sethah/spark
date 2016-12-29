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
package org.apache.spark.ml.optim.linesearch

import org.apache.spark.ml.optim.DifferentiableFunction

class StrongWolfe(
    c1: Double = 1e-4,
    c2: Double = 0.9,
    maxLineSearchIter: Int = 10,
    maxZoomIter: Int = 10) extends LineSearch {
  import StrongWolfe._

  require(c2 >= c1 && c2 <= 1.0)
  require(c1 >= 0.0 && c1 <= 1.0)

  /** Evaluates the line search function at a point. */
  private def phi(alpha: Double, f: DifferentiableFunction[Double]): Bracket = {
    val (phiVal, phiPrime) = f.compute(alpha)
    Bracket(alpha, phiVal, phiPrime)
  }

  /**
   * General idea is to start with an interval of the domain of the line search function, and
   * keep shifting and expanding it until we find an interval that is guaranteed to contain points
   * which meet the strong Wolfe conditions. Once we have such an interval, we "zoom" in on that
   * interval until we find a point that is acceptable.
   *
   * @param f The 1D line search function which is differentiable w.r.t alpha.
   * @param initialGuess The initial guess at alpha.
   * @return A step size which meets the strong Wolfe conditions.
   */
  def optimize(f: DifferentiableFunction[Double], initialGuess: Double): Double = {
    var currentAlpha = initialGuess
    val phiZero = phi(0.0, f)
    var left = phiZero
    val _zoom = zoom(phiZero, f, 0) _

    // TODO: rewrite using takeWhile or something similar
    for (i <- 0 until maxLineSearchIter) {
      val right = phi(currentAlpha, f)
      if (right.phi.isInfinite() || right.phi.isNaN()) {
        currentAlpha /= 2.0
      } else {
        // check if any of the three conditions is met, otherwise increase alpha and repeat
        if ((right.phi > phiZero.phi + c1 * currentAlpha * phiZero.phiPrime) ||
          (right.phi >= left.phi && i > 0)) {
          return _zoom(left, right)
        }

        if (curvatureCondition(phiZero, right)) {
          return right.alpha
        }

        if (right.phiPrime >= 0.0) {
          return _zoom(right, left)
        }

        left = right
        currentAlpha *= 1.5
      }
    }
    throw new Exception("Line search failed")
  }

  private def interpolate(left: Bracket, right: Bracket): Double = {
    // compute the alpha that minimizes a cubic approximation of the line search function between
    // left and right brackets according to Nocedal and Wright, p. 59
    val d1 = left.phiPrime + right.phiPrime - 3.0 *
      (left.phi - right.phi) / (left.alpha - right.alpha)
    val d2 = math.sqrt(d1 * d1 - left.phiPrime * right.phiPrime)
    val nextAlpha = right.alpha - (right.alpha - left.alpha) * (right.phiPrime + d2 - d1) /
      (right.phiPrime - left.phiPrime + 2 * d2)

    // A safeguard, described by Nocedal and Wright p. 58, which ensures we make
    // progress on each iteration.
    val intervalLength = right.alpha - left.alpha
    val leftBound = left.alpha + 0.1 * intervalLength
    val rightBound = left.alpha + 0.9 * intervalLength
    if (nextAlpha < leftBound) {
      leftBound
    } else if (nextAlpha > rightBound) {
      rightBound
    } else {
      nextAlpha
    }
  }


  /**
   * Given an interval [left, right] which is guaranteed to contain alpha values that meet the
   * strong Wolfe conditions, this function narrows the interval until it finds a point that
   * satisfies the strong Wolfe conditions.
   *
   * @param left Value of line search function at left alpha value.
   * @param right Value of line search function at right alpha value.
   * @param phiZero Value of line search function at alpha = 0.
   * @param iter Zoom iteration number.
   * @param f The line search function.
   * @return The selected step size alpha which meets the strong Wolfe conditions.
   */
  private def zoom(phiZero: Bracket, f: DifferentiableFunction[Double], iter: Int)
                  (left: Bracket, right: Bracket): Double = {
    // interpolate requires left < right
    val nextAlpha = if (left.phi > right.phi) interpolate(right, left) else interpolate(left, right)
    val nextPhi = phi(nextAlpha, f)
    if (iter > maxZoomIter) {
      throw new Exception("line search failed")
    } else if (!decreaseCondition(phiZero, nextPhi) || nextPhi.phi >= left.phi) {
      // the decrease condition is not satisfied, zoom on the interval [alpha_low, nextAlpha]
      zoom(phiZero, f, iter + 1)(left, nextPhi)
    } else {
      if (!curvatureCondition(phiZero, nextPhi)) {
        if (nextPhi.phiPrime * (right.alpha - left.alpha) >= 0.0) {
          zoom(phiZero, f, iter + 1)(nextPhi, left)
        } else {
          zoom(phiZero, f, iter + 1)(nextPhi, right)
        }
      } else {
        // the next alpha meets both the decrease and curvature conditions, so return it.
        nextAlpha
      }
    }
  }

  def decreaseCondition(phiZero: Bracket, currentPhi: Bracket): Boolean = {
    currentPhi.phi <= phiZero.phi + c1 * currentPhi.alpha * phiZero.phiPrime
  }

  def curvatureCondition(phiZero: Bracket, currentPhi: Bracket): Boolean = {
    math.abs(currentPhi.phiPrime) <= c2 * math.abs(phiZero.phiPrime)
  }

}

object StrongWolfe {

  /**
   * A simple class representing a point value of the line search function. Holds information
   * needed for line search functions: alpha, phi(alpha), phi'(alpha).
   *
   * phi(alpha) = f(x_c + alpha * d)
   *
   * @param alpha The step size.
   * @param phi The value of the line search function at alpha.
   * @param phiPrime The derivative of the line search function w.r.t. alpha evaluated at alpha.
   */
  case class Bracket(alpha: Double, phi: Double, phiPrime: Double)
}
