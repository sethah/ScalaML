package nn

import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM}

trait CostFunction {

  def apply(predicted: BDV[Double], label: BDV[Double]): BDV[Double]

  def derivative(predicted: BDV[Double], label: BDV[Double]): BDV[Double]

  def derivative(predicted: BDM[Double], label: BDM[Double]): BDM[Double]
}

object Variance extends CostFunction {

  def apply(predicted: BDV[Double], label: BDV[Double]): BDV[Double] = {
    (predicted - label).mapValues(x => x * x)
  }

  def derivative(predicted: BDV[Double], label: BDV[Double]): BDV[Double] = {
    predicted - label
  }

  def derivative(predicted: BDM[Double], label: BDM[Double]): BDM[Double] = {
    require((predicted.rows == label.rows) && (predicted.cols == label.cols),
      s"Predicted: (${predicted.rows}, ${predicted.cols}) != Label: (${label.rows}, ${label.cols})")
    predicted - label
  }
}
