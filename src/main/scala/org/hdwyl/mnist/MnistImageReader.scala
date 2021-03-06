package org.hdwyl.mnist

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * Created by wangyanl on 2019/6/22.
  */
class MnistImageReader(path: String) extends MnistFileReader(path) {

  assert(stream.readInt() == 2051, "Wrong MNIST image stream magic")

  val count = stream.readInt()
  val height = stream.readInt()
  val width = stream.readInt()

  val imagesAsMatrices = readImages(0)
  val imagesAsVectors = imagesAsMatrices.map { image =>
    DenseVector.tabulate(width * height) { i => image(i % height, i / width) }
  }

  private[this] def readImages(ind: Int): Stream[DenseMatrix[Double]] =
    if (ind >= count) {
      Stream.empty
    } else {
      Stream.cons(readImage(), readImages(ind + 1))
    }

  private[this] def readImage(): DenseMatrix[Double] = {
    val m = DenseMatrix.zeros[Double](height, width)

    for (y <- 0 until height; x <- 0 until width)
      m(x, y) = stream.readUnsignedByte()
    return m
  }

}
