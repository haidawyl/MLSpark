package org.hdwyl.mnist

import breeze.linalg.DenseVector

/**
  * Created by wangyanl on 2019/6/22.
  */
class MnistLabelReader(path: String) extends MnistFileReader(path) {

  assert(stream.readInt() == 2049, "Wrong MNIST label stream magic")

  val count = stream.readInt()

  val labelsAsInts = readLabels(0)
  val labelsAsVectors = labelsAsInts.map { label =>
    DenseVector.tabulate[Double](10) { i => if (i == label) 1.0 else 0.0 }
  }

  private[this] def readLabels(ind: Int): Stream[Int] =
    if (ind >= count) {
      Stream.empty
    } else {
      Stream.cons(stream.readByte(), readLabels(ind + 1))
    }
}
