package nn

import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM}
import breeze.numerics._

object TestUtil {
  def assertVectorEquals(a: BDV[Double], b: BDV[Double], tol: Double = 1e-5): Unit = {
    assert(a.length == b.length)
    a.valuesIterator.zip(b.valuesIterator).foreach { case (x, y) =>
      assert(abs(x - y) < tol)
    }
  }

  def assertMatrixEquals(a: BDM[Double], b: BDM[Double], tol: Double = 1e-5): Unit = {
    a.valuesIterator.zip(b.valuesIterator).foreach { case (x, y) =>
      assert(abs(x - y) < tol)
    }
    assert(a.rows == b.rows)
    assert(a.cols == b.cols)
  }
}