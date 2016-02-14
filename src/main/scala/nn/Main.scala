package nn

import data.MnistDataset
import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM, argmax, *}

object NeuralNetApp {
  def main(args: Array[String]) = {
    val numImages = 1000
    val numTestImages = 1000
    val path = "/Users/sethhendrickson/mnist"
    val dataset = new MnistDataset(path, "train")
    val testDataset = new MnistDataset(path, "t10k")
    val images = dataset.imagesAsMatrix(numImages)
    val labels = dataset.labelsAsMatrix(numImages)

    val testImages = testDataset.imagesAsMatrix(numTestImages)
    val testLabels = testDataset.labelsAsMatrix(numTestImages)
    labels.valuesIterator.take(20).foreach(println)


//    val nn = new NeuralNet(Array(784, 30, 10), CrossEntropy, SigmoidFunction)
//    val dataSet = new DataSet(images, labels)
//    val testDataSet = new DataSet(testImages, testLabels)
//    val epochs = 300
//    val miniBatchSize = 10
//    val eta = 3.0
//    nn.train(dataSet, miniBatchSize, epochs, eta, 1, Some(testDataSet))
  }
}
