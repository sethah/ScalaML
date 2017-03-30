package nn

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, argmax, sum, *}

/**
 * Feedforward neural network.
 */
class NeuralNet(val layers: List[Layer], val costFunction: CostFunction) {

  def this(sizes: Array[Int], costFunction: CostFunction,
           activationFunction: ActivationFunction) = {
    this(NeuralNet.initLayers(sizes, activationFunction), costFunction)
  }

  val numLayers = layers.length

  val topology = layers.head.weights.cols +: layers.map(layer => layer.bias.length)

  def train(dataSet: DataSet,
            miniBatchSize: Int,
            epochs: Int,
            learningRate: Double,
            evalInterval: Int = 10,
            validationData: Option[DataSet] = None) = {

    val shouldValidate = validationData.isDefined
    val costs = new Array[Double](epochs)
    val accuracy = new Array[Double](epochs)
    dataSet.miniBatches(miniBatchSize).take(epochs).zipWithIndex.foreach { case (batch, iter) =>
      val (deltaB, deltaW, numCorrect, cost) = backprop(batch.inputs, batch.outputs)
      costs(iter) = cost
      accuracy(iter) = batch.numExamples - numCorrect

      // update weights
      layers.zipWithIndex.foreach { case (layer, i) =>
        layer.weights = layer.weights - deltaW(i).mapValues(_ * (learningRate / batch.numExamples))
        layer.bias = layer.bias - deltaB(i).mapValues(_ * (learningRate / batch.numExamples))
      }

      if (iter % evalInterval == 0 || iter == epochs - 1) {
        println(s"Epoch: $iter")
        if (shouldValidate) {
          val (numCorrect, total) = evaluate(validationData.get)
          println(s"$numCorrect / $total")
        }
      }
    }
    costs.grouped(10).map(_.sum / (10 * 10)).toArray.view(0, 10).foreach(println)
    println("***")
    costs.grouped(10).map(_.sum / (10 * 10)).toArray.view(epochs / 10 - 10, epochs / 10 - 1).foreach(println)
  }

  def backprop(x: BDM[Double], y: BDM[Double]): (Array[BDV[Double]], Array[BDM[Double]], Double, Double) = {

    val (linearPredictors, activations) = layers.scanLeft((BDM.zeros[Double](x.rows, x.cols), x)) {
      case ((z, a), layer) => (layer.linearPredictor(a), layer.feedForward(a))
    }.unzip

    val outputDelta = costFunction.derivative(activations.last, y) :*
      layers.last.activationFunction.derivative(linearPredictors.last)

    val cost = sum(costFunction(activations.last, y))
    val numCorrect = evaluate(activations.last, y)

    val deltas = (0 until (numLayers - 1)).scanRight(outputDelta) { case (i, delta) =>
      val z = linearPredictors(i + 1)
      layers(i + 1).prevDelta(delta, layers(i).activationFunction.derivative(z))
    }

    val (nablaB, nablaW) = deltas.zipWithIndex.map { case (delta, i) =>
      (sum(delta.t(*, ::)), delta.t * activations(i))
    }.unzip

    (nablaB.toArray, nablaW.toArray, numCorrect, cost)
  }

  def feedForward(input: BDV[Double]): BDV[Double] = {
    layers.foldLeft(input){ case (acc, layer) => layer.feedForward(acc)}
  }

  def feedForward(input: BDM[Double]): BDM[Double] = {
    layers.foldLeft(input){ case (acc, layer) => layer.feedForward(acc)}
  }

  def evaluate(data: DataSet): (Int, Int) = {
    val outputActivations = feedForward(data.inputs)
    val numCorrect = evaluate(outputActivations, data.outputs)
    (numCorrect, data.numExamples)
  }

  def evaluate(activations: BDM[Double], y: BDM[Double]): Int = {
    val predicted = activations(*, ::).map(v => argmax(BDV(v.toArray)))
    val actual = y(*, ::).map(v => argmax(BDV(v.toArray)))
    val numCorrect = sum((predicted - actual).mapValues(v => if (v == 0) 1 else 0))
    numCorrect
  }
}

object NeuralNet {

  def initLayers(sizes: Array[Int], activationFunction: ActivationFunction): List[Layer] = {
    sizes.sliding(2).map { case Array(in, out) =>
      Layer(in, out, activationFunction)
    }.toList
  }
}
