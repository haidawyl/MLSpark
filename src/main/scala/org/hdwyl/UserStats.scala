package org.hdwyl

import breeze.linalg.Axis._1
import org.apache.log4j.Logger
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._

import scala.util.Sorting

/**
  * Created by wangyanl on 2019/6/14.
  */
object UserStats {

  @transient lazy val logger = Logger.getLogger(this.getClass)

  def main(args: Array[String]) {
    val conf = new SparkConf()
    val sc = new SparkContext(conf)
    // 读取数据集
    val userData = sc.textFile("hdfs://PATH/ml-100k/u.user")
    // 输出第1行数据
    println(userData.first())
    // 输出前k行数据
    println(userData.take(10))

    // 用户ID（user ID）、年龄（age）、性别（gender）、职业（occupation）和邮编（ZIP code）
    val userFields = userData.map(line => line.split("|")).map(e => (e(0), e(1), e(2), e(3), e(4)))
    userFields.take(10).foreach(println)
    // 统计用户数量
    val numUsers = userFields.map{ case (userId, age, gender, occupation, zipCode) => userId}.count()
    // 统计性别数量
    val numGenders = userFields.map{ case (userId, age, gender, occupation, zipCode) => gender}.distinct().count()
    // 统计职业数量
    val numOccupations = userFields.map{ case (userId, age, gender, occupation, zipCode) => occupation}.distinct().count()
    // 统计邮编数量
    val numZipCodes = userFields.map{ case (userId, age, gender, occupation, zipCode) => zipCode}.distinct().count()

    // Users: 943, genders: 2, occupations: 21, ZIP codes: 795
    println("Users: %d, genders: %d, occupations: %d, ZIP codes: %d".format(numUsers, numGenders, numOccupations, numZipCodes))

    // sortBy(_._k): 按第k个元素进行排序, 只能进行数字排序
    val countByAge = userFields.map{case (userId, age, gender, occupation, zipCode) => (age, 1)}.reduceByKey(_ + _).collect().sortBy(_._2)
    countByAge.foreach(println)
    val countByOccupation = userFields.map{case (userId, age, gender, occupation, zipcode) => (occupation, 1)}.reduceByKey(_ + _).collect().sortBy(_._2)
    val countByOccupation2 = userFields.map{case (userId, age, gender, occupation, zipcode) => occupation}.countByValue()
    println("MapReduce approach:")
    countByOccupation.foreach(println)
    println("countByValue approach:")
    countByOccupation2.foreach(println)

    val allOccupations = userFields.map{ case (userId, age, gender, occupation, zipCode) => occupation}.distinct().collect()
    Sorting.quickSort(allOccupations)
    allOccupations.foreach(println)
    try {
      val allOccupationsWithIndex = allOccupations.zipWithIndex
      allOccupationsWithIndex.foreach(println)
    } catch {
      case ex: Exception => {
      }
    }

    sc.stop()
  }


}
