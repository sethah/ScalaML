package data

import java.io.{ File, FileInputStream, FileOutputStream, DataInputStream }
import java.net.URL
import java.nio.file.{ Files, Paths }
import java.nio.channels.Channels
import java.util.zip.GZIPInputStream
import breeze.linalg._

class MnistFileReader(location: String, fileName: String) {

  private[this] val path = Paths.get(location, fileName)

  if (!Files.exists(path))
    download

  protected[this] val stream = new DataInputStream(new GZIPInputStream(new FileInputStream(path.toString)))

  private def download: Unit = {
    val rbc = Channels.newChannel(new URL(s"http://yann.lecun.com/exdb/mnist/$fileName").openStream())
    val fos = new FileOutputStream(s"$location/$fileName")
    fos.getChannel.transferFrom(rbc, 0, Long.MaxValue)
  }

}

class MnistLabelReader(location: String, fileName: String) extends MnistFileReader(location, fileName) {

  assert(stream.readInt() == 2049, "Wrong MNIST label stream magic")

  val count = stream.readInt()

//  val labelsAsInts = readLabels(0)
//  val labelsAsVectors = labelsAsInts.map { label =>
//    DenseVector.tabulate[Double](10) { i => if (i == label) 1.0 else 0.0 }
//  }
  def labelsAsArray(n: Int) = {
    val m = new Array[Double](n * 10)
    var i = 0
    while (i < n) {
      val label = stream.readByte()
      m(i * 10 + label) = 1.0
      i += 1
    }
    m
  }

  private[this] def readLabels(ind: Int): Stream[Int] =
    if (ind >= count)
      Stream.empty
    else
      Stream.cons(stream.readByte(), readLabels(ind + 1))

}

class MnistImageReader(location: String, fileName: String) extends MnistFileReader(location, fileName) {

  assert(stream.readInt() == 2051, "Wrong MNIST image stream magic")

  val count = stream.readInt()
  val width = stream.readInt()
  val height = stream.readInt()

//  val imagesAsMatrices = readImages(0)
//  val imagesAsVectors = imagesAsMatrices map { image =>
//    DenseVector.tabulate(width * height) { i => image(i / width, i % height) / 256.0 }
//  }
  def imagesAsArray(n: Int) = readAllImages(n)

  private[this] def readImages(ind: Int): Stream[DenseMatrix[Int]] =
    if (ind >= count)
      Stream.empty
    else
      Stream.cons(readImage(), readImages(ind + 1))

  private[this] def readAllImages(n: Int): Array[Double] = {
    val m = new Array[Double](height * width * n)

    for (i <- 0 until height * width * n) {
      m(i) = stream.readUnsignedByte().toDouble / 255.0
    }
    m
  }

  private[this] def readImage(): DenseMatrix[Int] = {
    val m = DenseMatrix.zeros[Int](height, width)

    for (y <- 0 until height; x <- 0 until width)
      m(y, x) = stream.readUnsignedByte()

    m
  }

}

/**
  * http://yann.lecun.com/exdb/mnist/
  */
class MnistDataset(location: String, dataset: String) {

  lazy val imageReader = new MnistImageReader(location, s"$dataset-images-idx3-ubyte.gz")
  lazy val labelReader = new MnistLabelReader(location, s"$dataset-labels-idx1-ubyte.gz")

  def imageWidth = imageReader.width
  def imageHeight = imageReader.height

//  def imagesAsMatrices = imageReader.imagesAsMatrices
//  def imagesAsVectors = imageReader.imagesAsVectors
  def imagesAsArray(n: Int) = imageReader.imagesAsArray(n)
//  def imagesAsMatrix(n: Int): DenseMatrix[Double] = {
//    val images = imagesAsVectors.take(n)
//    new DenseMatrix(images.head.length, n, images.flatMap(_.toArray).toArray).t
//  }


//  def labelsAsInts = labelReader.labelsAsInts
//  def labelsAsVectors = labelReader.labelsAsVectors
//  def labelsAsMatrix(n: Int): DenseMatrix[Double] = {
//    val labels = labelsAsVectors.take(n)
//    new DenseMatrix(labels.head.length, n, labels.flatMap(_.toArray).toArray).t
//  }
  def labelsAsArray(n: Int) = labelReader.labelsAsArray(n)

//  def examples = imagesAsVectors zip labelsAsVectors

}

object Mnist {

  val location = Option(System.getenv("MNIST_PATH")).getOrElse(List(System.getenv("HOME"), ".cache", "mnist").mkString(File.separator))
  val locationFile = new File(location)

  if (!locationFile.exists)
    locationFile.mkdirs

  val trainDataset = new MnistDataset(location, "train")
  val testDataset = new MnistDataset(location, "t10k")

}