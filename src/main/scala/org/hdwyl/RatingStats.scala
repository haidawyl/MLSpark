package org.hdwyl

import java.util.Calendar

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by wangyanl on 2019/6/14.
  */
object RatingStats {
  def main(args: Array[String]) {
    val conf = new SparkConf()
    val sc = new SparkContext(conf)
    // 读取数据集
    val ratingDataRaw = sc.textFile("hdfs://PATH/ml-100k/u.data")
    // 输出第1行数据
    println(ratingDataRaw.first())
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
    val medianRating = getMedian(ratings)
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

  def getMedian(data: RDD[Int]): Int = {
    // 将数据分为4组
    val number = data.map(n => (n / 4, n)).sortByKey()
    // 每个分组的数据量
    val pairCount = data.map(n => (n / 4, 1)).reduceByKey(_ + _).sortByKey()
    // 数据总量
    val count = data.count().toInt
    // 中值在整个数据区间的偏移量
    var mid = 0
    if (count % 2 != 0) {
      mid = count / 2 + 1
    } else {
      mid = count / 2
    }

    var temp1 = 0 // 中值所在的区间累加的个数
    var temp2 = 0 // 中值所在的区间前面所有的区间累加的个数
    var index = 0 // 中值的区间
    val tongNumber = pairCount.count().toInt

    var foundIt = false
    for (i <- 0 to tongNumber - 1 if !foundIt) {
      temp1 = temp1 + pairCount.collectAsMap()(i)
      temp2 = temp1 - pairCount.collectAsMap()(i)
      if (temp1 >= mid) {
        index = i
        foundIt = true
      }
    }
    // 中位数在桶中的偏移量
    val tongInnerOffset = mid - temp2
    // takeOrdered: 默认将key从小到大排序后, 获取rdd中的前n个元素
    val median = number.filter(_._1 == index).takeOrdered(tongInnerOffset)
    return median(tongInnerOffset - 1)._2
  }
}
