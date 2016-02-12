package nn

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, argmax, sum, *}

/**
 * Feedforward neural network.
 */
class NeuralNet(val sizes: Array[Int],
                val costFunction: CostFunction,
                activationFunction: ActivationFunction) {

  val layers = sizes.sliding(2).map { case Array(in, out) =>
    Layer(in, out, activationFunction)
  }.toList

  val numLayers = layers.length

  def train(dataSet: DataSet,
            miniBatchSize: Int,
            epochs: Int,
            learningRate: Double,
            evalInterval: Int = 10,
            validationData: Option[DataSet] = None) = {

    val shouldValidate = validationData.isDefined
    dataSet.miniBatches(miniBatchSize).take(epochs).zipWithIndex.foreach { case (batch, iter) =>
      val (deltaB, deltaW) = backprop(batch.inputs, batch.outputs)

      // update weights
      layers.zipWithIndex.foreach { case (layer, i) =>
        layer.weights = layer.weights - deltaW(i).mapValues(_ * (learningRate / batch.numExamples))
        layer.bias = layer.bias - deltaB(i).mapValues(_ * (learningRate / batch.numExamples))
      }

      println(s"Epoch: $iter")
      if (shouldValidate) {
        val (numCorrect, total) = evaluate(validationData.get)
        println(s"$numCorrect / $total")
      }
    }
  }

  def backprop(x: BDM[Double], y: BDM[Double]): (Array[BDV[Double]], Array[BDM[Double]]) = {

    val (linearPredictors, activations) = layers.scanLeft((BDM.zeros[Double](x.rows, x.cols), x)) {
      case ((z, a), layer) => (layer.linearPredictor(a), layer.feedForward(a))
    }.unzip

    val outputDelta = costFunction.derivative(activations.last, y) :*
      activationFunction.derivative(linearPredictors.last)

    val deltas = (0 until (numLayers - 1)).scanRight(outputDelta) { case (i, delta) =>
      val z = linearPredictors(i + 1)
      layers(i + 1).prevDelta(delta, layers(i).activationFunction.derivative(z))
    }

    val (nablaB, nablaW) = deltas.zipWithIndex.map { case (delta, i) =>
      (sum(delta.t(*, ::)), delta.t * activations(i))
    }.unzip

    (nablaB.toArray, nablaW.toArray)
  }

  def feedForward(input: BDV[Double]): BDV[Double] = {
    layers.foldLeft(input){ case (acc, layer) => layer.feedForward(acc)}
  }

  def feedForward(input: BDM[Double]): BDM[Double] = {
    layers.foldLeft(input){ case (acc, layer) => layer.feedForward(acc)}
  }

  def evaluate(data: DataSet): (Int, Int) = {
    val outputActivations = feedForward(data.inputs)
    val N = outputActivations.cols
    val predicted = outputActivations(*, ::).map(v => argmax(BDV(v.toArray)))
    val actual = data.outputs(*, ::).map(v => argmax(BDV(v.toArray)))
    val numCorrect = sum((predicted - actual).mapValues(v => if (v == 0) 1 else 0))
    (numCorrect, data.numExamples)
  }
}
