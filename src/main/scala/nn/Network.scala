//package nn
//
//import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, argmax}
//import breeze.numerics._
//import breeze.stats.distributions.Gaussian
//import scala.util.Random
//
//case class LabeledPoint(label: BDV[Double], features: BDV[Double])
//
//class Network(val sizes: Array[Int]) {
//
//  val numLayers = sizes.length
//
//  private val sampler = Gaussian(0, 1)
//  private val rng = new scala.util.Random()
//
//  val biases = sizes.tail.map { layerSize =>
//    val samples = sampler.sample(layerSize)
//    BDV(samples.toArray)
//  }
//
//  val weights = sizes.dropRight(1).zip(sizes.drop(1)).map { case (n, m) =>
//    val samples = sampler.sample(m * n)
//    new BDM(m, n, samples.toArray)
//  }
//
//  def feedForward(a: BDV[Double]): BDV[Double] = {
//    biases.zip(weights).foldLeft(a){ case (acc, (b, w)) =>
//      sigmoid(w * acc + b)
//    }
//  }
//
//  def sgd(data: Array[LabeledPoint], epochs: Int, miniBatchSize: Int,
//          eta: Double, validationData: Option[Array[LabeledPoint]] = None): Unit = {
////    val N = data.size
////    val samplePoints = (1 to (N / miniBatchSize - 1)).scanLeft(0.0){
////      (acc, i) => acc + miniBatchSize.toDouble / N
////    } :+ 1.0
////    val sampleRanges = samplePoints.sliding(2).toArray
//    val shouldValidate = validationData match {
//      case Some(data) => true
//      case _ => false
//    }
//
//    var j = 0
//    while (j < epochs) {
//      println(s"Training epoch $j")
//      val miniBatches = Random.shuffle(data.toSeq).grouped(miniBatchSize)
//      miniBatches.foreach { batch =>
//        updateMiniBatch(batch.toArray, eta)
//      }
//
//      if (shouldValidate) {
//        val (numCorrect, total) = evaluate(validationData.get)
//        println(s"$numCorrect / $total")
//      }
//      j += 1
//    }
//  }
//
//  def updateMiniBatch(miniBatch: Array[LabeledPoint], eta: Double) = {
//    val N = miniBatch.length.toDouble
//    val updateB = biases.map{ b => BDV.zeros[Double](b.length)}
//    val updateW = weights.map{ w => BDM.zeros[Double](w.rows, w.cols)}
//    miniBatch.foreach { case LabeledPoint(y, x) =>
//      val (b, w) = backprop(x, y)
//      updateB.indices.foreach { i =>
//        updateB(i) = updateB(i) + b(i)
//        updateW(i) = updateW(i) + w(i)
//      }
//    }
//    weights.indices.foreach { i =>
//      weights(i) = weights(i) - updateW(i).mapValues(v => v * (eta / N))
//    }
//    biases.indices.foreach { i =>
//      biases(i) = biases(i) - updateB(i).mapValues(v => v * (eta / N))
//    }
//  }
//
//  def backprop(x: BDV[Double],
//               y: BDV[Double]): (Array[BDV[Double]], Array[BDM[Double]]) = {
//    val (zs, activations) = biases.zip(weights).scanLeft {
//      (BDV.zeros[Double](x.length), x)}{
//      case ((zacc, aacc), (b, w)) =>
//        val z = w * aacc + b
//        val activation = sigmoid(z)
//        (z, activation)
//    }.unzip
//
//    var delta = Network.costDerivative(activations.last, y) :* Network.sigmoidPrime(zs.last)
//    val nablaB = new Array[BDV[Double]](biases.length)
//    val nablaW = new Array[BDM[Double]](weights.length)
//    nablaB(biases.length - 1) = delta
//    nablaW(weights.length - 1) = delta * activations(activations.length - 2).t
//    for (j <- 2 until numLayers) {
//      val z = zs(zs.length - j)
//      val sp = Network.sigmoidPrime(z)
//      delta = weights(weights.length - j + 1).t * delta :* sp
//      nablaB(nablaB.length - j) = delta
//      nablaW(nablaW.length - j) = delta * activations(activations.length - j - 1).t
//    }
//    (nablaB, nablaW)
//  }
//
//  def evaluate(data: Array[LabeledPoint]): (Int, Int) = {
//    val numCorrect = data.foldLeft(0) { case (acc, LabeledPoint(l, f)) =>
//      if (argmax(feedForward(f)) == argmax(l)) acc + 1 else acc
//    }
//    (numCorrect, data.length)
//  }
//}
//
//object Network {
//  def sigmoidPrime(z: BDV[Double]): BDV[Double] = {
//    z.mapValues(v => sigmoid(v) * (1.0 - sigmoid(v)))
//  }
//
//  def costDerivative(activations: BDV[Double],
//                     y: BDV[Double]): BDV[Double] = {
//    activations - y
//  }
//
//
//}
