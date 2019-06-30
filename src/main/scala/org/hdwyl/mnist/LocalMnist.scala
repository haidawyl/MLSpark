package org.hdwyl.mnist

import org.hdwyl.Utils

/**
  * Created by wangyanl on 2019/6/22.
  */
object LocalMnist {

  def main(args: Array[String]) {
    lazy val trainImageReader = new MnistImageReader(getClass().getClassLoader().getResource("data/mnist/train-images.idx3-ubyte").getPath)
    lazy val trainLabelReader = new MnistLabelReader(getClass().getClassLoader().getResource("data/mnist/train-labels.idx1-ubyte").getPath)
    lazy val testImageReader = new MnistImageReader(getClass().getClassLoader().getResource("data/mnist/t10k-images.idx3-ubyte").getPath)
    lazy val testLabelReader = new MnistLabelReader(getClass().getClassLoader().getResource("data/mnist/t10k-labels.idx1-ubyte").getPath)

    val imageWidth = trainImageReader.width
    val imageHeight = trainImageReader.height
    println("imageWidth = %d".format(imageWidth))
    println("imageHeight = %d".format(imageHeight))

    val trainImagesAsMatrices = trainImageReader.imagesAsMatrices
    val trainLabelsAsInts = trainLabelReader.labelsAsInts
    println("train images count: %d".format(trainImagesAsMatrices.size))
    println("train labels count: %d".format(trainLabelsAsInts.size))

    val K = 20
    val trainData = trainImagesAsMatrices zip trainLabelsAsInts
    trainData.toList.take(K).map { case (image, label) =>
      Utils.printMatrix(image, imageWidth)
      println("label = %d".format(label))
    }

    val testImagesAsMatrices = testImageReader.imagesAsMatrices
    val testLabelsAsInts = testLabelReader.labelsAsInts
    println("test images count: %d".format(testImagesAsMatrices.size))
    println("test labels count: %d".format(testLabelsAsInts.size))

    val testData = testImagesAsMatrices zip testLabelsAsInts
    testData.toList.take(K).map { case (image, label) =>
      Utils.printMatrix(image, imageWidth)
      println("label = %d".format(label))
    }

    /*
    for (i <- 0 until 10) {
      Utils.printMatrix(trainImagesAsMatrices.toList(i), imageWidth)
      println("label = %d".format(trainLabelsAsInts.toList(i)))
    }
    */

    val trainImagesAsVectors = trainImageReader.imagesAsVectors
    val trainLabelsAsVectors = trainLabelReader.labelsAsVectors
    val trainDataVectors = trainImagesAsVectors zip trainLabelsAsVectors
    trainDataVectors.toList.take(K).map { case (image, label) =>
      Utils.printVector(image, imageWidth)
      println("label = %d".format(label.findAll(x => x == 1.0)(0)))
    }

    val testImagesAsVectors = testImageReader.imagesAsVectors
    val testLabelsAsVectors = testLabelReader.labelsAsVectors
    val testDataVectors = testImagesAsVectors zip testLabelsAsVectors
    testDataVectors.toList.take(K).map { case (image, label) =>
      Utils.printVector(image, imageWidth)
      println("label = %d".format(label.findAll(x => x == 1.0)(0)))
    }
  }

}
