package nn

import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM}
import breeze.numerics._
import org.scalatest.FunSuite

class LayerSuite extends FunSuite {

  def assertVectorEquals(a: BDV[Double], b: BDV[Double]): Unit = {
    assert(a.length == b.length)
    a.valuesIterator.zip(b.valuesIterator).foreach { case (x, y) =>
      assert(x == y)
    }
  }

  def assertMatrixEquals(a: BDM[Double], b: BDM[Double], tol: Double = 0.0001): Unit = {
    a.valuesIterator.zip(b.valuesIterator).foreach { case (x, y) =>
      assert(abs(x - y) < tol)
    }
    assert(a.rows == b.rows)
    assert(a.cols == b.cols)
  }

  def toyNetwork: (Array[BDM[Double]], Array[BDV[Double]]) = {
    val sizes = Array(3, 2, 3)
    val w1 = Array(1.0, -4.0, 7.0, -8.0, -7.0, -7.0)
    val w2 = Array(7.0, -5.0, 3.0, -8.0, 1.0, 0.0)
    val b1 = Array(2.0, 6.0)
    val b2 = Array(-10.0, -2.0, 2.0)
    val weights1 = new BDM(sizes(0), sizes(1), w1).t
    val bias1 = BDV(b1)
    val weights2 = new BDM(sizes(1), sizes(2), w2).t
    val bias2 = BDV(b2)
    (Array(weights1, weights2), (Array(bias1, bias2)))
  }

  def toyData: (BDM[Double], BDM[Double]) = {
    val N = 3
    val sizes = Array(3, 2, 3)
    val in = Array(3.0, 8.0, -7.0, -6.0, -7.0, -9.0, -10.0, 8.0, 8.0)
    val input = new BDM(sizes(0), N, in).t
    val out = Array(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    val output = new BDM(sizes.last, N, out).t
    (input, output)
  }

  test("single layer forward prop") {
    val N = 3
    val sizes = Array(3, 2, 3)
    val (inputs, outputs) = toyData
    val (weights, biases) = toyNetwork
    val costFunction = Variance
    val activationFunction = SigmoidFunction

    val layer1 = new Layer(weights(0), biases(0), SigmoidFunction)
    val layer2 = new Layer(weights(1), biases(1), SigmoidFunction)
    val expectedZ1 = new BDM(sizes(1), N, Array(-76.0, -25.0, -39.0, 166.0, 16.0, -26.0)).t
    val expectedZ2 = new BDM(sizes(2), N, Array(-10.0000000001, -2.00000000011, 2.0, -15.0, -10.0, 2.0, -3.00000078777, 0.999999662354, 2.99999988746)).t
    val z1 = layer1.linearPredictor(inputs)
    val a1 = layer1.feedForward(inputs)
    val z2 = layer2.linearPredictor(a1)
    val a2 = layer2.feedForward(a1)
    assertMatrixEquals(expectedZ1, z1)
    assertMatrixEquals(activationFunction(expectedZ1), a1)
    assertMatrixEquals(expectedZ2, z2)
    assertMatrixEquals(activationFunction(expectedZ2), a2)
//    val layers = List(layer1, layer2)
//    val outputActivations = layers.foldLeft(input){ case (acc, layer) => layer.feedForward(acc)}
//    println(outputActivations)
//    val costFunction = Variance
//    println(costFunction.derivative(outputActivations, output))
  }

  test("feedforward network") {
    val N = 3
    val sizes = Array(3, 2, 3)
    val (inputs, outputs) = toyData
    val (weights, biases) = toyNetwork
    val costFunction = Variance
    val activationFunction = SigmoidFunction

    val layer1 = new Layer(weights(0), biases(0), SigmoidFunction)
    val layer2 = new Layer(weights(1), biases(1), SigmoidFunction)
    val nn = new NeuralNet(sizes, costFunction, activationFunction)
    nn.layers = List(layer1, layer2)
    val outputActivations = nn.feedForward(inputs)
    println(outputActivations)
    println(nn.evaluate(new DataSet(inputs, outputs)))
  }

  test("backprop") {
    val N = 3
    val sizes = Array(3, 2, 3)
    val (inputs, outputs) = toyData
    val (weights, biases) = toyNetwork
    val costFunction = Variance
    val activationFunction = SigmoidFunction

    val layer1 = new Layer(weights(0), biases(0), SigmoidFunction)
    val layer2 = new Layer(weights(1), biases(1), SigmoidFunction)

    val nn = new NeuralNet(sizes, costFunction, activationFunction)
    nn.layers = List(layer1, layer2)
    val (nablaB, nablaW) = nn.backprop(inputs, outputs)
//    nablaW.foreach(println)
  }
}
