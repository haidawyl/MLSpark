package org.hdwyl

import java.util.Calendar

import org.apache.log4j.Logger
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by wangyanl on 2019/6/14.
  */
object RatingStats {

  @transient lazy val logger = Logger.getLogger(this.getClass)

  def main(args: Array[String]) {
    val conf = new SparkConf()
    val sc = new SparkContext(conf)
    // 读取数据集
    val ratingDataRaw = sc.textFile("hdfs://PATH/ml-100k/u.data")
    // 输出第1行数据
    // println(ratingDataRaw.first())
    // 统计数量
    val numRatings = ratingDataRaw.count()
    // Ratings: 100000
    println("Ratings: %d".format(numRatings))

    val ratingData = ratingDataRaw.map(line => line.split("\t"))
    // 评分集合
    val ratings = ratingData.map(e => e(2).toInt)
    // 计算最高评分
    val maxRating = ratings.reduce((x, y) => Math.max(x, y))
    val maxRating2 = ratings.max()
    // 计算最低评分
    val minRating = ratings.reduce((x, y) => Math.min(x, y))
    val minRating2 = ratings.min()
    // 计算评分的平均值
    val meanRating = ratings.reduce((x, y) => x + y) * 1.0 / numRatings
    val meanRating2 = ratings.sum() * 1.0 / numRatings
    // 计算评分的中位数
    val medianRating = Utils.getMedian(ratings)
    // 用户数量
    val numUsers = 943
    // 用户的平均评分
    val ratingsPerUser = numRatings * 1.0 / numUsers
    // 电影数量
    val numMovies = 1682
    // 电影的平均评分
    val ratingsPerMovie = numRatings * 1.0 / numMovies
    // Min rating: 1
    println("Min rating: %d".format(minRating))
    println("Min rating: %d".format(minRating2))
    // Max rating: 5
    println("Max rating: %d".format(maxRating))
    println("Max rating: %d".format(maxRating2))
    // Average rating: 3.53
    println("Average rating: %2.2f".format(meanRating))
    println("Average rating: %2.2f".format(meanRating2))
    // Median rating: 4
    println("Median rating: %d".format(medianRating))
    // Average # of ratings per user: 106.00
    println("Average # of ratings per user: %2.2f".format(ratingsPerUser))
    // Average # of ratings per movie: 59.00
    println("Average # of ratings per movie: %2.2f".format(ratingsPerMovie))

    println(ratings.stats())

    val countByRating = ratings.countByValue()
    countByRating.foreach(println)

    // 按照用户进行分组
    val userRatingsGrouped = ratingData.map(e => (e(0).toInt, e(2).toInt)).groupByKey()
    // 统计每一个用户的评级次数
    val userRatingsByUser = userRatingsGrouped.map(e => (e._1, e._2.size)).sortBy(_._2)
    userRatingsByUser.foreach(println)

    val userRatingsByUserLocal = userRatingsByUser.map(e => e._2).collect()
    println(userRatingsByUserLocal)

    val timestamps = ratingData.map(e => e(3))
    val hourOfDays = timestamps.map(e => extractDatetime(e).get(Calendar.HOUR_OF_DAY))
    println(hourOfDays)

    val timeOfDays = hourOfDays.map(e => assignTod(e))
    println(timeOfDays)

    sc.stop()
  }

  def extractDatetime(ts: String): Calendar = {
    val calendar = Calendar.getInstance()
    calendar.setTimeInMillis(ts.toLong)
    return calendar
  }

  def assignTod(hour: Int): String = {
    val timesOfDay = Map("morning" -> "7,8,9,10,11", "lunch" -> "12,13", "afternoon" -> "14,15,16,17", "evening" -> "18,19,20,21,22", "night" -> "23,0,1,2,3,4,5,6");
    return timesOfDay.filter(e => e._2.split(",").contains(hour.toString)).keys.toList(0)
  }

}
