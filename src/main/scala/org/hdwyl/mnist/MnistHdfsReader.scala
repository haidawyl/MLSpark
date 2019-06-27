package org.hdwyl.mnist

import java.nio.ByteBuffer

import breeze.linalg.DenseMatrix
import org.apache.spark.SparkContext

/**
  * Created by wangyanl on 2019/6/22.
  */
object MnistHdfsReader {

  var rowNum : Int = 0
  var colNum : Int = 0

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
    rowNum = featureBuffer.getInt()
    // number of columns
    colNum = featureBuffer.getInt()

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
