package nn

import data.MnistDataset
import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM, argmax, *}

object NeuralNetApp {
  def main(args: Array[String]) = {
    val dataset = new MnistDataset("/Users/sethhendrickson/mnist", "train")
    val images = dataset.imagesAsVectors.take(1000)
    val imageData = new BDM(images.head.length, 1000, images.flatMap(_.toArray).toArray).t
    val labels = dataset.labelsAsVectors.take(1000)
    val labelData = new BDM(labels.head.length, 1000, labels.flatMap(_.toArray).toArray).t
    val testDataset = new MnistDataset("/Users/sethhendrickson/mnist", "t10k")
    val testImages = testDataset.imagesAsVectors.take(1000)
    val testImageData = new BDM(testImages.head.length, 1000, testImages.flatMap(_.toArray).toArray).t
    val testLabels = testDataset.labelsAsVectors.take(1000)
    val testLabelData = new BDM(testLabels.head.length, 1000, testLabels.flatMap(_.toArray).toArray).t
//    val data = images.zip(labels).map { case (i, l) =>
//      LabeledPoint(l, i)
//    }
    val nn = new NeuralNet(Array(784, 30, 10), Variance, SigmoidFunction)
    val dataSet = new DataSet(imageData, labelData)
//    val validationDataSet = new DataSet(testData.toArray)
    val epochs = 100
    val miniBatchSize = 100
    val eta = 3.0
    nn.train(new DataSet(imageData, labelData), miniBatchSize, epochs, eta, 1, Some(new DataSet(testImageData, testLabelData)))
//    val predicted = nn.feedForward(testImageData)
//    println(predicted(*, ::).map(v => argmax(BDV(v.toArray))))
//    println(testLabelData(*, ::).map(v => argmax(BDV(v.toArray))))
//    val weights1 = new BDM(2, 3, Array(2.0, -1.0, 6.0, -9.0, 4.0, 5.0))
//    val bias1 = BDV(Array(3.0, 1.0))
//    val layer1 = new Layer(weights1, bias1, SigmoidFunction)
//    val input = new BDM(2, 3, Array(2.0, -2.0, 3.0, 5.0, -4.0, 2.0))
//    println(layer1.feedForward(input))
  }
}
