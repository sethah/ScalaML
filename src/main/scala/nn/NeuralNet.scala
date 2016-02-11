package nn

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, argmax, sum, *, norm}
import scala.util.Random

class NeuralNet(val sizes: Array[Int],
                val costFunction: CostFunction,
                activationFunction: ActivationFunction) {

  var layers = sizes.sliding(2).map { case Array(in, out) =>
    Layer(in, out, activationFunction)
  }.toList

  val numLayers = layers.length

  def train(dataSet: DataSet,
            miniBatchSize: Int,
            epochs: Int,
            learningRate: Double,
            evalInterval: Int = 10,
            validationData: Option[DataSet] = None) = {
    val shouldValidate = validationData match {
      case Some(data) => true
      case _ => false
    }

    val N = dataSet.numExamples.toDouble
    var x = BDM.zeros[Double](miniBatchSize, dataSet.numFeatures)
    var j = 0
    while (j < epochs){
      val miniBatches = getMiniBatches(dataSet, miniBatchSize)
      miniBatches.zipWithIndex.foreach { case (batch, iter) =>
        val (deltaB, deltaW) = backprop(batch.inputs, batch.outputs)
        // update weights
        layers.zipWithIndex.foreach { case (layer, i) =>
          layer.weights = layer.weights - deltaW(i).mapValues(v => v * (learningRate / batch.numExamples))
          layer.bias = layer.bias - deltaB(i).mapValues(v => v * (learningRate / batch.numExamples))
        }
      }
      println(s"Epoch: $j")
      if (shouldValidate) {
        val (numCorrect, total) = evaluate(validationData.get)
        println(s"$numCorrect / $total")
      }
      j += 1
    }
  }

  def getMiniBatches(data: DataSet, size: Int): Array[DataSet] = {
    val shuffled = Random.shuffle((0 until data.numExamples).toList)
    shuffled.grouped(size).toSeq.map { rows =>
      new DataSet(data.inputs(rows, ::).copy.toDenseMatrix, data.outputs(rows, ::).copy.toDenseMatrix)
    }.toArray
  }

  def backprop(x: BDM[Double],
    y: BDM[Double]): (Array[BDV[Double]], Array[BDM[Double]]) = {
    val (linearPredictors, activations) = layers.scanLeft((BDM.zeros[Double](x.rows, x.cols), x)) {
      case ((z, a), layer) => (layer.linearPredictor(a), layer.feedForward(a))
    }.unzip

    val outputDelta = costFunction.derivative(activations.last, y) :*
      activationFunction.derivative(linearPredictors.last)
    val deltas = (0 until (numLayers - 1)).scanRight(outputDelta) { case (i, delta) =>
      val z = linearPredictors(i + 1)
      val weights = layers(i + 1).weights
      val newDeltas = (delta * weights) :* layers(i).activation.derivative(z)
      newDeltas
    }
    val (nablaB, nablaW) = deltas.zipWithIndex.map { case (delta, i) =>
      val b = sum(delta.t(*, ::))
      (b, delta.t * activations(i))
    }.unzip
    (nablaB.toArray, nablaW.toArray)
  }

  def backprop(x: BDV[Double],
               y: BDV[Double]): (Array[BDV[Double]], Array[BDM[Double]]) = {
    val (linearPredictors, activations) = layers.scanLeft((BDV.zeros[Double](x.length), x)) {
      case ((z, a), layer) => (layer.linearPredictor(a), layer.feedForward(a))
    }.unzip

    val outputDelta = costFunction.derivative(activations.last, y) :*
      activationFunction.derivative(linearPredictors.last)
    val deltas = (0 until (numLayers - 1)).scanRight(outputDelta) { case (i, delta) =>
      val z = linearPredictors(i + 1)
      val weights = layers(i + 1).weights
      val newDelta = weights.t * delta :* layers(i + 1).activation.derivative(z)
      newDelta
    }
    val (nablaB, nablaW) = deltas.zipWithIndex.map { case (delta, i) =>
      (delta, delta * activations(i).t)
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

object NeuralNet {
  def labelAsVector(label: Double): BDV[Double] = {
    val vec = BDV.zeros[Double](10)
    vec(label.toInt) = 1.0
    vec
  }

  def labelAsInt(label: BDV[Double]): Int = argmax(label)
}