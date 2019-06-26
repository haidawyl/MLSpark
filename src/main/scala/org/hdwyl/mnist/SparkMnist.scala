package org.hdwyl.mnist

import java.nio.ByteBuffer

import breeze.linalg.DenseMatrix
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}
import org.hdwyl.Utils

/**
  * Created by wangyanl on 2019/6/22.
  */
object SparkMnist {

  def main(args: Array[String]) {
    val conf = new SparkConf()
    val sc = new SparkContext(conf)

    val featureFile = "hdfs://PATH/mnist/train-images.idx3-ubyte"
    val labelFile = "hdfs://PATH/mnist/train-labels.idx1-ubyte"

    val imagesAsMatrices = loadFeature(sc, featureFile)
    println("Images count: %d".format(imagesAsMatrices.size))
    val labelsAsInts = loadLabels(sc, labelFile)
    println("Labels count: %d".format(labelsAsInts.size))

    val width = 28
    val height = 28
    for (i <- 0 until 10) {
      Utils.printMatrix(imagesAsMatrices.toList(i), width)
      println("label = %d".format(labelsAsInts.toList(i)))
    }

    val data = imagesAsMatrices.zipWithIndex.map { case (image, index) =>
      val features = image.toArray
      val label = labelsAsInts.toList(index)
      LabeledPoint(label, Vectors.dense(features))
    }

    // 迭代次数，用于逻辑回归和SVM模型
    val numIterations = 10
    // 最大树深度，用于决策树模型
    val maxTreeDepth = 5

    // 训练逻辑回归模型
//    val lrModel = new LogisticRegressionWithLBFGS().setNumClasses(10).run(data)


    sc.stop()
  }

  def loadBinaryFile(sc: SparkContext, filePath: String): Array[Byte] = {
    // 读文件
    val files = sc.binaryFiles(filePath)
    // 返回字符数组
    val bytes = files.first()._2.toArray()
    bytes
  }

  def loadLabels(sc: SparkContext, labelFile: String): Array[Int] = {
    val labelBuffer = ByteBuffer.wrap(loadBinaryFile(sc, labelFile))
    // magic number
    val labelMagicNumber = labelBuffer.getInt()
    require(labelMagicNumber == 2049)
    // number of items
    val labelCount = labelBuffer.getInt()

    val result = new Array[Int](labelCount)
    for (i <- 0 until labelCount) {
      result(i) = labelBuffer.get()
    }
    result
  }

  def loadFeature(sc: SparkContext, featureFile: String): Array[DenseMatrix[Double]] = {
    val featureBuffer = ByteBuffer.wrap(loadBinaryFile(sc, featureFile))
    // magic number
    val featureMagicNumber = featureBuffer.getInt()
    require(featureMagicNumber == 2051)
    // number of images
    val featureCount = featureBuffer.getInt()
    // number of rows
    val rowNum = featureBuffer.getInt()
    // number of columns
    val colNum = featureBuffer.getInt()

    val result = new Array[DenseMatrix[Double]](featureCount)
    for (i <- 0 until featureCount) {
      val m = DenseMatrix.zeros[Double](rowNum, colNum)
      for (y <- 0 until rowNum; x <- 0 until colNum) {
        m(x, y) = featureBuffer.get()
      }
      result(i) = m
    }
    result
  }
}
