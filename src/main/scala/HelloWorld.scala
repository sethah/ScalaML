package NeuralNet

object HelloWorld {
  def main(args: Array[String]) = {
    println("Hello World!")
    val nn = new Network(Array(2, 1, 3))
    println(nn.numLayers)
    nn.weights.foreach(x => println(x.toString + "***"))
  }
}
