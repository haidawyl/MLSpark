package org.hdwyl

import java.util.Calendar

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by wangyanl on 2019/6/14.
  */
object MovieStats {

  def main(args: Array[String]) {
    val conf = new SparkConf()
    val sc = new SparkContext(conf)
    // 读取数据集
    val movieData = sc.textFile("hdfs://PATH/ml-100k/u.item")
    // 输出第1行数据
    println(movieData.first())
    // 输出前k行数据
    println(movieData.take(10))

    // 统计电影数量
    val numMovies = movieData.count()
    // Movies: 1682
    println("Movies: %d".format(numMovies))

    val movieFields = movieData.map(line => line.split("|"))
    // 提取出电影的年份信息
    val years = movieFields.map(e => e(2)).map(date => convertYear(date))
    // 过滤掉1900年(即未记录年份)的电影数据
    val filteredYears = years.filter { year => year != 1900 }
    // 计算得到电影的年龄
    val thisYear = Calendar.getInstance().get(Calendar.YEAR)
    val movieAges = filteredYears.map(year => thisYear - year).countByValue()
    movieAges.foreach(println)

    sc.stop()
  }

  def convertYear(x: String): Int = {
    try {
      return Integer.parseInt(x.substring(x.length() - 4))
    } catch {
      case ex: NumberFormatException => {
        return 1900 // 若数据缺失年份则将其年份设为1900
      }
    }
  }
}
