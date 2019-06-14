package org.hdwyl

import java.util.Calendar

import scala.io.Source

/**
  * Created by wangyanl on 2019/6/14.
  */
object LocalStats {
  def main(args: Array[String]): Unit = {
//    statUser()
//    statMovie()
    statRating()
  }

  def statUser(): Unit = {
    val file = Source.fromInputStream(getClass().getClassLoader().getResourceAsStream("data/ml-100k/u.user"), "UTF-8")
    val userData = file.getLines().toList
    val userFields = userData.map(line => line.split("\\|")).map(e => (e(0), e(1), e(2), e(3), e(4)))
    // 统计用户数量
    val numUsers = userFields.map { case (userId, age, gender, occupation, zipCode) => userId }.size
    // 统计性别数量
    val numGenders = userFields.map { case (userId, age, gender, occupation, zipCode) => gender }.distinct.size
    // 统计职业数量
    val numOccupations = userFields.map { case (userId, age, gender, occupation, zipCode) => occupation }.distinct.size
    // 统计邮编数量
    val numZipCodes = userFields.map { case (userId, age, gender, occupation, zipCode) => zipCode }.distinct.size

    // Users: 943, genders: 2, occupations: 21, ZIP codes: 795
    println("Users: %d, genders: %d, occupations: %d, ZIP codes: %d".format(numUsers, numGenders, numOccupations, numZipCodes))

    val countByAge = userFields.map { case (userId, age, gender, occupation, zipCode) => (age, 1) }.groupBy(_._1).map(e => (e._1, e._2.size)).toList.sortBy(_._2).reverse
    countByAge.foreach(println)
    val countByOccupation = userFields.map { case (userId, age, gender, occupation, zipcode) => (occupation, 1) }.groupBy(_._1).map(e => (e._1, e._2.size)).toList.sortBy(_._2).reverse
    countByOccupation.foreach(println)

    val allOccupations = userFields.map { case (userId, age, gender, occupation, zipCode) => occupation }.distinct.sortWith(_.compareTo(_) < 0)
    allOccupations.foreach(println)
    try {
      val allOccupationsWithIndex = allOccupations.zipWithIndex
      allOccupationsWithIndex.foreach(println)
    } catch {
      case ex: Exception => {
      }
    }
    file.close()
  }

  def statMovie(): Unit = {
    val file = Source.fromInputStream(getClass().getClassLoader().getResourceAsStream("data/ml-100k/u.item"), "UTF-8")
    val lines = file.getLines()
    try {
      for (line <- lines) {
        println(convertYear(line.split("\\|")(2)))
      }
    } catch {
      case ex: StringIndexOutOfBoundsException => {
      }
    }

    file.close()
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

  def statRating(): Unit = {
    val file = Source.fromInputStream(getClass().getClassLoader().getResourceAsStream("data/ml-100k/u.data"), "UTF-8")
    val ratingDataRaw = file.getLines().toList
    // 统计数量
    val numRatings = ratingDataRaw.size
    // Ratings: 100000
    println("Ratings: %d".format(numRatings))

    val ratingData = ratingDataRaw.map(line => line.split("\t"))
    // 评分集合
    val ratings = ratingData.map(e => e(2).toInt)
    // 计算最高评分
    val maxRating = ratings.reduce((x, y) => Math.max(x, y))
    val maxRating2 = ratings.max
    // 计算最低评分
    val minRating = ratings.reduce((x, y) => Math.min(x, y))
    val minRating2 = ratings.min
    // 计算评分的平均值
    val meanRating = ratings.reduce((x, y) => x + y) * 1.0 / numRatings
    val meanRating2 = ratings.sum * 1.0 / numRatings
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

    val countByRating = ratings.map(e => (e, 1)).groupBy(_._1).map(e => (e._1, e._2.size)).toList.sortBy(_._2).reverse
    // countByRating.foreach(println)

    // 按照用户进行分组
    val userRatingsGrouped = ratingData.map(e => (e(0).toInt, e(2).toInt)).groupBy(_._1)
    // 统计每一个用户的评级次数
    val userRatingsByUser = userRatingsGrouped.map(e => (e._1, e._2.size)).toList.sortBy(_._2).reverse
    // userRatingsByUser.foreach(println)

    val userRatingsByUserLocal = userRatingsByUser.map(e => e._2)
    // println(userRatingsByUserLocal)

    val timestamps = ratingData.map(e => e(3))
    val hourOfDays = timestamps.map(e => extractDatetime(e).get(Calendar.HOUR_OF_DAY))
    // println(hourOfDays)

    val timeOfDays = hourOfDays.map(e => assignTod(e))
    // println(timeOfDays)

    val timeOfDaysWithIndex = timeOfDays.zipWithIndex
    // println(timeOfDaysWithIndex)

    file.close()
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

  def getMedian(data: List[Int]): Int = {
    // 将数据分为4组
    val number = data.map(n => (n / 4, n)).sortBy(_._1)
    // 每个分组的数据量
    val pairCount = data.map(n => (n / 4, 1)).groupBy(_._1).map(e => (e._1, e._2.size)).toList.sortBy(_._1)
    // 数据总量
    val count = data.size
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
    val tongNumber = pairCount.size

    var foundIt = false
    for (i <- 0 to tongNumber - 1 if !foundIt) {
      temp1 = temp1 + pairCount(i)._2
      temp2 = temp1 - pairCount(i)._2
      if (temp1 >= mid) {
        index = i
        foundIt = true
      }
    }
    // 中位数在桶中的偏移量
    val tongInnerOffset = mid - temp2
    // 将key从小到大排序后, 获取前n个元素
    val median = number.filter(_._1 == index).sortBy(_._1).take(tongInnerOffset)
    return median(tongInnerOffset - 1)._2
  }
}
