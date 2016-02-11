package nn

import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM}
import breeze.numerics._

trait ActivationFunction {

  def derivative(z: BDV[Double]): BDV[Double]

  def derivative(z: BDM[Double]): BDM[Double]

  def apply(z: BDV[Double]): BDV[Double]

  def apply(z: BDM[Double]): BDM[Double]

}

object SigmoidFunction extends ActivationFunction {

  def apply(z: BDV[Double]): BDV[Double] = sigmoid(z)

  def apply(z: BDM[Double]): BDM[Double] = sigmoid(z)

  def derivative(z: BDV[Double]): BDV[Double] = {
    z.mapValues(v => sigmoid(v) * (1.0 - sigmoid(v)))
  }

  def derivative(z: BDM[Double]): BDM[Double] = {
    z.mapValues(v => sigmoid(v) * (1.0 - sigmoid(v)))
  }
}