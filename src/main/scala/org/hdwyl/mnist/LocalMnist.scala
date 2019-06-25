package org.hdwyl.mnist

/**
  * Created by wangyanl on 2019/6/22.
  */
object LocalMnist {

  def main(args: Array[String]) {
    lazy val imageReader = new MnistImageReader(getClass().getClassLoader().getResource("data/mnist/train-images.idx3-ubyte").getPath)
    lazy val labelReader = new MnistLabelReader(getClass().getClassLoader().getResource("data/mnist/train-labels.idx1-ubyte").getPath)

    val imageWidth = imageReader.width
    println("imageWidth = %d".format(imageWidth))
    val imageHeight = imageReader.height
    println("imageHeight = %d".format(imageHeight))

    val imagesAsMatrices = imageReader.imagesAsMatrices
    println(imagesAsMatrices.take(1).toVector)
    val imagesAsVectors = imageReader.imagesAsVectors
    println(imagesAsVectors.take(1).toVector)

    val labelsAsInts = labelReader.labelsAsInts
    println(labelsAsInts.take(1).toVector)
    val labelsAsVectors = labelReader.labelsAsVectors
    println(labelsAsVectors.take(1).toVector)
  }
}
