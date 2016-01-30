package NeuralNet

import breeze.linalg.{DenseVector, DenseMatrix}
import breeze.stats.distributions.Gaussian
import breeze.numerics._

class Network(val sizes: Array[Int]) {

  val numLayers = sizes.length

  private val sampler = Gaussian(0, 1)

  val biases = sizes.tail.map { layerSize =>
    val samples = sampler.sample(layerSize)
    DenseVector(samples.toArray)
  }

  val weights = sizes.dropRight(1).zip(sizes.drop(1)).map { case (n, m) =>
    val samples = sampler.sample(m * n)
    new DenseMatrix(m, n, samples.toArray)
  }

  def feedForward(a: DenseVector[Double]) = {
    biases.zip(weights).foldLeft(a){ case (acc, (b, w)) =>
      w * acc + b
    }
  }

  def sgd(X: DenseMatrix[Double], epochs: Int, miniBatchSize: Int,
          eta: Double, X_test: Option[DenseMatrix] = None): Unit = {

  }
}
