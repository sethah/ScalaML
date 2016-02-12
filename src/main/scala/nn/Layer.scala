package nn

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, *}

class Layer(var weights: BDM[Double],
            var bias: BDV[Double],
            val activationFunction: ActivationFunction) {

  val numInputs = weights.rows
  val numOutputs = weights.cols

  require(weights.rows == bias.length,
    s"Weight and bias dimensions do not match (${weights.rows} != ${bias.length})")

  def linearPredictor(a: BDV[Double]): BDV[Double] = {
    weights * a + bias
  }

  def linearPredictor(a: BDM[Double]): BDM[Double] = {
    val weighted = a * weights.t
    weighted(*, ::) + bias
  }

  def prevDelta(nextDelta: BDM[Double], activationDeriv: BDM[Double]): BDM[Double] = {
    (nextDelta * weights) :* activationDeriv
  }

  def feedForward(a: BDV[Double]): BDV[Double] = {
    activationFunction(linearPredictor(a))
  }

  def feedForward(a: BDM[Double]): BDM[Double] = {
    activationFunction(linearPredictor(a))
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
