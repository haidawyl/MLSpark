package org.hdwyl.mnist

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by wangyanl on 2019/6/22.
  */
object SparkMnist {

  def main(args: Array[String]) {
    val conf = new SparkConf()
    val sc = new SparkContext(conf)



    sc.stop()
  }
}
