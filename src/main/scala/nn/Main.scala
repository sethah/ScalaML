package nn

import data.MnistDataset
import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM, argmax, *}

object NeuralNetApp {
  def main(args: Array[String]) = {
    val numImages = 10000
    val numTestImages = 5000
    val path = "/Users/sethhendrickson/mnist"
    val dataset = new MnistDataset(path, "train")
    val testDataset = new MnistDataset(path, "t10k")
    val imageArray = dataset.imagesAsArray(numImages)
    val images = new BDM(784, numImages, imageArray).t
//    val lbls = dataset.labelsAsInts.take(10).toList
    val labels = new BDM(10, numImages, dataset.labelsAsArray(numImages)).t
//    imageArray.view(0, 783).foreach(println)
//    println(lbls)
//    labels.valuesIterator.take(50).grouped(10).foreach(println)

    val testImages = new BDM(784, numTestImages, testDataset.imagesAsArray(numTestImages)).t
    val testLabels = new BDM(10, numTestImages, testDataset.labelsAsArray(numTestImages)).t

    val nn = new NeuralNet(Array(784, 30, 10), Variance, SigmoidFunction)
    val dataSet = new DataSet(images, labels)
    val testDataSet = new DataSet(testImages, testLabels)
    val epochs = 5000
    val miniBatchSize = 10
    val eta = 3.0
    nn.train(dataSet, miniBatchSize, epochs, eta, 1000, Some(testDataSet))
  }
}
