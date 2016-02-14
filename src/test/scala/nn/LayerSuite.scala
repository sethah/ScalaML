package nn

import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM}
import breeze.numerics._
import org.scalatest.FunSuite
import TestUtil.{assertMatrixEquals, assertVectorEquals}

class LayerSuite extends FunSuite {

  def toyNetwork: NeuralNet = {
    val sizes = Array(3, 2, 3)
    val w1 = Array(1.0, -4.0, 7.0, -8.0, -7.0, -7.0)
    val w2 = Array(7.0, -5.0, 3.0, -8.0, 1.0, 0.0)
    val b1 = Array(2.0, 6.0)
    val b2 = Array(-10.0, -2.0, 2.0)
    val weights1 = new BDM(sizes(0), sizes(1), w1).t
    val bias1 = BDV(b1)
    val weights2 = new BDM(sizes(1), sizes(2), w2).t
    val bias2 = BDV(b2)
    val layer1 = new Layer(weights1, bias1, SigmoidFunction)
    val layer2 = new Layer(weights2, bias2, SigmoidFunction)
    (Array(weights1, weights2), (Array(bias1, bias2)))
    new NeuralNet(List(layer1, layer2), Variance)
  }

  def toyData: DataSet = {
    val N = 3
    val sizes = Array(3, 2, 3)
    val in = Array(3.0, 8.0, -7.0, -6.0, -7.0, -9.0, -10.0, 8.0, 8.0)
    val input = new BDM(sizes(0), N, in).t
    val out = Array(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    val output = new BDM(sizes.last, N, out).t
    new DataSet(input, output)
  }

  test("single layer forward prop") {
    val dataSet = toyData
    val costFunction = Variance
    val activationFunction = SigmoidFunction
    val nn = toyNetwork
    val sizes = nn.topology
    val N = dataSet.numExamples

    val expectedZ1 = new BDM(sizes(1), N, Array(-76.0, -25.0, -39.0, 166.0, 16.0, -26.0)).t
    val expectedZ2 = new BDM(sizes(2), N, Array(-10.0, -2.0, 2.0, -15.0, -10.0, 2.0, -3.0, 1.0, 3.0)).t
    val z1 = nn.layers(0).linearPredictor(dataSet.inputs)
    val a1 = nn.layers(0).feedForward(dataSet.inputs)
    val z2 = nn.layers(1).linearPredictor(a1)
    val a2 = nn.layers(1).feedForward(a1)
    assertMatrixEquals(expectedZ1, z1)
    assertMatrixEquals(activationFunction(expectedZ1), a1)
    assertMatrixEquals(expectedZ2, z2)
    assertMatrixEquals(activationFunction(expectedZ2), a2)
  }

  test("network forwardprop") {
    val dataSet = toyData
    val costFunction = Variance
    val activationFunction = SigmoidFunction
    val nn = toyNetwork
    val sizes = nn.topology
    val N = dataSet.numExamples

    val outputActivations = nn.feedForward(dataSet.inputs)
    val expectedZOutput = new BDM(sizes(2), N,
      Array(-10.0, -2.0, 2.0, -15.0, -10.0, 2.0, -3.0, 1.0, 3.0)).t
    assertMatrixEquals(outputActivations, sigmoid(expectedZOutput))
  }

  test("backprop") {
    val dataSet = toyData
    val costFunction = Variance
    val activationFunction = SigmoidFunction
    val nn = toyNetwork
    val sizes = nn.topology
    val N = dataSet.numExamples

    val (nablaB, nablaW) = nn.backprop(dataSet.inputs, dataSet.outputs)
    val expectedNablaWArray = Array(
      Array(-4.99723340e-07, 3.99778672e-07, 3.99778672e-07,
        5.51335456e-11, -5.85354746e-11, -3.77249389e-11),
      Array(2.14253916e-03,  -3.05902029e-07, 1.43734834e-01,
        2.06178109e-09, -2.14254274e-03, 9.24780432e-02))
    val expectedNablaBArray = Array(
      Array(4.99723340e-08, -7.31693432e-12),
      Array(0.00209684, 0.15625039, 0.18281354))
    val expectedNablaW = expectedNablaWArray.zipWithIndex.map { case (flatValues, i) =>
      new BDM(nablaW(i).cols, nablaW(i).rows, flatValues).t
    }
    val expectedNablaB = expectedNablaBArray.zipWithIndex.map { case (flatValues, i) =>
      BDV(flatValues)
    }
    nablaW.zip(expectedNablaW).foreach { case (actual, expected) =>
      assertMatrixEquals(actual, expected)
    }
    nablaB.zip(expectedNablaB).foreach { case (actual, expected) =>
      assertVectorEquals(actual, expected)
    }
  }
}
