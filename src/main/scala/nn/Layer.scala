package nn

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, * => bCast}

class Layer(var weights: BDM[Double], var bias: BDV[Double], val activation: ActivationFunction) {

  val numInputs = weights.rows
  val numOutputs = weights.cols

  require(weights.rows == bias.length,
    s"Weight and bias dimensions do not match (${weights.rows} != ${bias.length})")

  def linearPredictor(a: BDV[Double]): BDV[Double] = {
    weights * a + bias
  }

  def linearPredictor(a: BDM[Double]): BDM[Double] = {
    val weighted = a * weights.t
//    println(s"${weighted.rows}, ${weighted.cols}, ${bias.length}")
    weighted(bCast, ::) + bias
  }

  def feedForward(a: BDV[Double]): BDV[Double] = {
    activation(linearPredictor(a))
  }

  def feedForward(a: BDM[Double]): BDM[Double] = {
    activation(linearPredictor(a))
  }

}

object Layer {
  def apply(numInputs: Int, numOutputs: Int, activation: ActivationFunction) = {
    new Layer(randomMatrix(numOutputs, numInputs), randomVector(numOutputs), activation)
  }

  def randomMatrix(m: Int, n: Int): BDM[Double] = {
    val sampler = breeze.stats.distributions.Gaussian(0, 1)
    new BDM(m, n, sampler.sample(m * n).toArray)
  }

  def randomVector(n: Int): BDV[Double] = {
    val sampler = breeze.stats.distributions.Gaussian(0, 1)
    new BDV(sampler.sample(n).toArray)
  }
}
