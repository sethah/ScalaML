package NeuralNet

import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM}
import breeze.stats.distributions.Gaussian
import breeze.numerics._

class Network(val sizes: Array[Int]) {

  val numLayers = sizes.length

  private val sampler = Gaussian(0, 1)

  val biases = sizes.tail.map { layerSize =>
    val samples = sampler.sample(layerSize)
    BDV(samples.toArray)
  }

  val weights = sizes.dropRight(1).zip(sizes.drop(1)).map { case (n, m) =>
    val samples = sampler.sample(m * n)
    new BDM(m, n, samples.toArray)
  }

  def feedForward(a: BDV[Double]) = {
    biases.zip(weights).foldLeft(a){ case (acc, (b, w)) =>
      w * acc + b
    }
  }

  def sgd(X: BDM[Double], epochs: Int, miniBatchSize: Int,
          eta: Double, X_test: Option[BDM[Double]] = None): Unit = {
    val N = X.rows
    var j = 0
    while (j < epochs) {

      j += 1
    }
  }

//  def updateMiniBatch(miniBatch)

  def backprop(x: BDV[Double],
               y: BDV[Double]): (Array[BDV[Double]], Array[BDM[Double]]) = {
    val (zs, activations) = biases.zip(weights).scanLeft {
      (BDV.zeros[Double](x.length), x)}{
      case ((zacc, aacc), (b, w)) =>
        val z = w * aacc + b
        val activation = sigmoid(z)
        (z, activation)
    }.unzip

    var delta = Network.costDerivative(activations.last, y) :* Network.sigmoidPrime(zs.last)
    val nablaB = new Array[BDV[Double]](biases.length)
    val nablaW = new Array[BDM[Double]](weights.length)
    nablaB(numLayers - 1) = delta
    nablaW(numLayers - 1) = delta * activations(numLayers - 2).t
    for (j <- 2 to numLayers) {
      val layer = numLayers - j
      val z = zs(layer)
      val sp = Network.sigmoidPrime(z)
      delta = weights(layer + 1).t * delta :* sp
      nablaB(numLayers - j) = delta
      nablaW(numLayers - j) = delta * activations(layer - 1).t
    }

    (nablaB, nablaW)
  }
}

object Network {
  def sigmoidPrime(z: BDV[Double]): BDV[Double] = {
    val sigma = sigmoid(z)
    z.mapValues(v => sigmoid(v) * (1.0 - sigmoid(v)))
  }

  def costDerivative(activations: BDV[Double],
                     y: BDV[Double]): BDV[Double] = {
    activations - y
  }
}
