package org.hdwyl.mnist

import org.apache.spark.{SparkConf, SparkContext}
import org.hdwyl.Utils

/**
  * Created by wangyanl on 2019/6/27.
  */
object SparkMnist {

  def main(args: Array[String]) {
    val conf = new SparkConf()
    val sc = new SparkContext(conf)

    val featureFile = "hdfs://PATH/mnist/train-images.idx3-ubyte"
    val labelFile = "hdfs://PATH/mnist/train-labels.idx1-ubyte"

    val imagesAsMatrices = MnistHdfsReader.loadFeature(sc, featureFile)
    println("Images count: %d".format(imagesAsMatrices.size))
    val labelsAsInts = MnistHdfsReader.loadLabels(sc, labelFile)
    println("Labels count: %d".format(labelsAsInts.size))

    val rowNum = MnistHdfsReader.rowNum
    val colNum = MnistHdfsReader.colNum
    for (i <- 0 until 10) {
      Utils.printMatrix(imagesAsMatrices.toList(i), colNum)
      println("label = %d".format(labelsAsInts.toList(i)))
    }

    sc.stop()
  }

}
