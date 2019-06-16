package org.hdwyl

import java.util.Calendar

import breeze.linalg._
import org.apache.log4j.Logger
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.matching.Regex

/**
  * Created by wangyanl on 2019/6/14.
  */
object MovieStats {

  @transient lazy val logger = Logger.getLogger(this.getClass)

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
    val years = movieFields.map(e => e(2)).map(x => convertYear(x))
    // 过滤掉1900年(即未记录年份)的电影数据
    val filteredYears = years.filter { year => year != 1900 }
    // 计算得到电影的年龄
    val thisYear = Calendar.getInstance().get(Calendar.YEAR)
    val movieAges = filteredYears.map(year => thisYear - year).countByValue()
    movieAges.foreach(println)

    val meanYear = filteredYears.sum() / filteredYears.count()
    // 非1900年的全部年份的中位数值
    val medianYear = Utils.getMedian(filteredYears)
    println(years.filter(year => year == 1900))
    val yearsPreProcessed = years.map(year => if (year == 1900) medianYear else year)
    println(yearsPreProcessed.filter(year => year == 1900))
    // Mean year of release: 1989
    println("Mean year of release: %d".format(meanYear))
    // Median year of release: 1995
    println("Median year of release: %d".format(medianYear))
    // Index of '1900' before assigning median: List(266)
    println("Index of '1900' before assigning median: %s".format(years.zipWithIndex.collect().find(e => e._1 == 1900).map(e => e._2).toList))
    // Index of '1900' after assigning median: List()
    println("Index of '1900' after assigning median: %s".format(yearsPreProcessed.zipWithIndex.collect().find(e => e._1 == 1900).map(e => e._2).toList))

    val rawTitles = movieFields.map(e => e(1))
    val movieTitles = rawTitles.map(title => extractTitle(title))
    // 下面用简单空白分词法将标题分词为词
    val titleTerms = movieTitles.map(title => title.split(" "))
    // 下面取回所有可能的词，以便构建一个词到序号的映射字典
    val pattern = new Regex("[,\\(\\):]")
    val allTermsWithIndex = titleTerms.flatMap(e => e).map(e => pattern replaceAllIn (e, "")).distinct.zipWithIndex.collectAsMap()
    // Total number of terms: 2457
    println("Total number of terms: %d".format(allTermsWithIndex.size))
    // Index of term 'Dead': 19
    println("Index of term 'Dead': %d".format(allTermsWithIndex.get("Dead").get))
    // Index of term 'Rooms': 4
    println("Index of term 'Rooms': %d".format(allTermsWithIndex.get("Rooms").get))

    val allTermsWithIndexBcast = sc.broadcast(allTermsWithIndex)
    val termVectors = titleTerms.map(terms => createVector(terms.toList, allTermsWithIndexBcast.value.toMap))
    titleTerms.take(10).map(term => term.mkString(" ")).foreach(println)
    termVectors.take(10).foreach(println)

    sc.stop()
  }

  def convertYear(x: String): Int = {
    try {
      return Integer.parseInt(x.substring(x.length() - 4))
    } catch {
      case ex: NumberFormatException => {
        return 1900 // 若数据缺失年份则将其年份设为1900
      }
      case ex: StringIndexOutOfBoundsException => {
        println(x)
        return 1900 // 若数据缺失年份则将其年份设为1900
      }
    }
  }

  def extractTitle(raw: String): String = {
    // 该表达式找寻括号之间的非单词（数字）
    val pattern = new Regex("\\(\\d+\\)")
    return (pattern replaceFirstIn(raw, "")).trim
  }

  // 该函数输入一个词列表，并用k之1编码类似的方式将其编码为一个稀疏向量
  def createVector(terms: List[String], termDict: Map[String, Long]): DenseMatrix[Int] = {
    import breeze.linalg.DenseMatrix._
    val numTerms = termDict.size
    val x = DenseMatrix.zeros[Int](1, numTerms)
    for (term <- terms) {
      if (termDict.contains(term)) {
        val idx = termDict.get(term).get.toInt
        x(::, idx) := 1
      }
    }
    return x
  }
}
