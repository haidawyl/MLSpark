package org.hdwyl

import breeze.linalg.{DenseVector, DenseMatrix}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD

/**
  * Created by wangyanl on 2019/6/15.
  */
object Utils {
  def getMedian(data: RDD[Int]): Int = {
    val sortedData = data.sortBy(x => x).collect().toList
    val count = data.count().toInt
    val median: Int = if (count % 2 == 0) {
      val l = count / 2 - 1
      val r = l + 1
      (sortedData(l) + sortedData(r)) / 2
    } else {
      sortedData(count / 2 + 1)
    }

    return median
  }

  def printMatrix(matrix: DenseMatrix[Double], width: Int) = {
    val array = matrix.toArray
    for (i <- 0 until array.size) {
      if (i > 0 && (i + 1) % width == 0) {
        println(array(i).toInt.toString())
      } else {
        print(array(i).toInt.toString() + "\t")
      }
    }
  }

  def printVector(vector: DenseVector[Double], width: Int) = {
    val array = vector.toArray
    for (i <- 0 until array.size) {
      if (i > 0 && (i + 1) % width == 0) {
        println(array(i).toInt.toString())
      } else {
        print(array(i).toInt.toString() + "\t")
      }
    }
  }

  def printVector(vector: Vector, width: Int) = {
    val array = vector.toArray
    for (i <- 0 until array.size) {
      if (i > 0 && (i + 1) % width == 0) {
        println(array(i).toInt.toString())
      } else {
        print(array(i).toInt.toString() + "\t")
      }
    }
  }

}
