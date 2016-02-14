package nn

import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM}
import breeze.numerics._

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

object CrossEntropy extends CostFunction {

  def apply(predicted: BDV[Double], label: BDV[Double]): BDV[Double] = {
    label :* log(predicted) + (label * -1.0 + 1.0) :* log(predicted * -1.0 + 1.0)
  }

  def derivative(predicted: BDV[Double], label: BDV[Double]): BDV[Double] = {
    label :/ predicted - (label * -1.0 + 1.0) :/ (predicted * -1.0 + 1.0)
  }

  def derivative(predicted: BDM[Double], label: BDM[Double]): BDM[Double] = {
    val out = ((label * -1.0 + 1.0) :/ (predicted * -1.0 + 1.0)) - (label :/ predicted)
    out.mapValues(v => if (v.isNaN) 0.0 else v)
  }
}
