package nn

import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM}
import breeze.numerics._
import org.scalatest.FunSuite
import TestUtil.{assertMatrixEquals, assertVectorEquals}

class CostFunctionSuite extends FunSuite {
  test("cross entropy cost function") {
    val z = new BDM(3, 2, Array(-2.5607008 , 100., 2.78764494,
      1.70754762, 0.58181829, -2.20046714)).t
    val a = sigmoid(z)
    val label = new BDM(3, 2, Array(0.0, 1.0, 0.0, 1.0, 0.0, 0.0)).t
    val deriv = CrossEntropy.derivative(a, label)
    val expectedDeriv = new BDM(3, 2, Array(1.07725058, 0.0, 17.24272215, -1.18130989,
      2.78928892, 1.11075141)).t
    assertMatrixEquals(deriv, expectedDeriv)
    // outputDelta = a - y
    assertMatrixEquals(deriv :* (sigmoid(z) :* (sigmoid(z) * -1.0 + 1.0)), a - label)
  }
}
