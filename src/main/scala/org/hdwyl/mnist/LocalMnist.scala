package org.hdwyl.mnist

import org.hdwyl.Utils

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
    println("Images count: %d".format(imagesAsMatrices.size))
    val labelsAsInts = labelReader.labelsAsInts
    println("Labels count: %d".format(labelsAsInts.size))

    for (i <- 0 until 10) {
      Utils.printMatrix(imagesAsMatrices.toList(i), imageWidth)
      println("label = %d".format(labelsAsInts.toList(i)))
    }

    val imagesAsVectors = imageReader.imagesAsVectors
    println(imagesAsVectors.size)

    val labelsAsVectors = labelReader.labelsAsVectors
    println(labelsAsVectors.take(1).toVector)
  }

}
