package nn

import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM}
import scala.util.Random

case class LabeledPoint(label: Double, features: BDV[Double])

class DataSet(val inputs: BDM[Double], val outputs: BDM[Double]) {

//  def this(features: Array[BDV[Double]], labels: Array[Int]) = {
//    this(features.zip(labels).map { case (f, l) => LabeledPoint(l, f) })
//  }

  val numFeatures = inputs.cols
  val numExamples = inputs.rows

  def miniBatches(size: Int): Iterator[DataSet] = {
    Stream.continually(Random.shuffle(0 to numExamples - 1)).flatten.grouped(size).map {
      rows =>
        new DataSet(inputs(rows, ::).copy.toDenseMatrix, outputs(rows, ::).copy.toDenseMatrix)
    }
  }
}
