package org.hdwyl

import java.util.Calendar

import breeze.linalg._
import org.apache.log4j.Logger

import scala.io.Source
import scala.util.matching.Regex

/**
  * Created by wangyanl on 2019/6/14.
  */
object LocalStats {

  @transient lazy val logger = Logger.getLogger(this.getClass)

  def main(args: Array[String]): Unit = {
    statUser()
    statMovie()
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
        logger.error(ex.getMessage, ex)
      }
    }
    file.close()
  }

  def convertYear(x: String): Int = {
    try {
      return Integer.parseInt(x.substring(x.length() - 4))
    } catch {
      case ex: NumberFormatException => {
        logger.error(ex.getMessage)
        logger.warn("x:" + x)
        return 1900 // 若数据缺失年份则将其年份设为1900
      }
      case ex: StringIndexOutOfBoundsException => {
        logger.error(ex.getMessage)
        logger.warn("x:" + x)
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
  def createVector(terms: List[String], termDict: Map[String, Int]): DenseMatrix[Int] = {
    import breeze.linalg.DenseMatrix._
    val numTerms = termDict.size
    val x = DenseMatrix.zeros[Int](1, numTerms)
    for (term <- terms) {
      if (termDict.contains(term)) {
        val idx = termDict.get(term).get
        x(::, idx) := 1
      }
    }
    return x
  }

  def statMovie(): Unit = {
    val file = Source.fromInputStream(getClass().getClassLoader().getResourceAsStream("data/ml-100k/u.item"), "UTF-8")
    val movieData = file.getLines().toList

    // 统计电影数量
    val numMovies = movieData.size
    // Movies: 1682
    println("Movies: %d".format(numMovies))

    val movieFields = movieData.map(line => line.split("\\|"))
    // 提取出电影的年份信息
    val years = movieFields.map(e => e(2)).map(x => convertYear(x))
    // 过滤掉1900年(即未记录年份)的电影数据
    val filteredYears = years.filter { year => year != 1900 }
    // 计算得到电影的年龄
    val thisYear = Calendar.getInstance().get(Calendar.YEAR)
    // countByVale : map(e => (e, 1)).groupBy(_._1).map(e => (e._1, e._2.size))
    val movieAges = filteredYears.map(year => thisYear - year).map(e => (e, 1)).groupBy(_._1).map(e => (e._1, e._2.size))
    // println(movieAges)

    val meanYear = filteredYears.sum / filteredYears.size
    // 非1900年的全部年份的中位数值
    val medianYear = getMedian(filteredYears)
    println(years.filter(year => year == 1900))
    val yearsPreProcessed = years.map(year => if (year == 1900) medianYear else year)
    println(yearsPreProcessed.filter(year => year == 1900))
    // Mean year of release: 1989
    println("Mean year of release: %d".format(meanYear))
    // Median year of release: 1995
    println("Median year of release: %d".format(medianYear))
    // Index of '1900' before assigning median: List(266)
    println("Index of '1900' before assigning median: %s".format(years.zipWithIndex.find(e => e._1 == 1900).map(e => e._2).toList))
    // Index of '1900' after assigning median: List()
    println("Index of '1900' after assigning median: %s".format(yearsPreProcessed.zipWithIndex.find(e => e._1 == 1900).map(e => e._2).toList))

    val rawTitles = movieFields.map(e => e(1))
    val movieTitles = rawTitles.map(title => extractTitle(title))
    // 下面用简单空白分词法将标题分词为词
    val titleTerms = movieTitles.map(title => title.split(" "))
    // 下面取回所有可能的词，以便构建一个词到序号的映射字典
    val pattern = new Regex("[,\\(\\):]")
    val allTermsWithIndex = titleTerms.flatMap(e => e).map(e => pattern replaceAllIn(e, "")).distinct.zipWithIndex.toMap

    // Total number of terms: 2457
    println("Total number of terms: %d".format(allTermsWithIndex.size))
    // Index of term 'Dead': 19
    println("Index of term 'Dead': %d".format(allTermsWithIndex.get("Dead").get))
    // Index of term 'Rooms': 4
    println("Index of term 'Rooms': %d".format(allTermsWithIndex.get("Rooms").get))

    val termVectors = titleTerms.map(terms => createVector(terms.toList, allTermsWithIndex))
    println(allTermsWithIndex)
    titleTerms.take(10).map(term => term.mkString(" ")).foreach(println)
    termVectors.take(10).foreach(println)

    val x = DenseVector.rand(10)
    val normX2 = norm(x)
    val normalizedX = x :/ normX2
    println("x:\n%s".format(x))
    println("2-Norm of x: %2.4f".format(normX2))
    println("Normalized x:\n%s".format(normalizedX))
    println("2-Norm of normalized: %2.4f".format(norm(normalizedX)))

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
    val sortedData = data.sortBy(x => x)
    val count = data.size
    val median: Int = if (count % 2 == 0) {
      val l = count / 2 - 1
      val r = l + 1
      (sortedData(l) + sortedData(r)) / 2
    } else {
      sortedData(count / 2 + 1)
    }

    return median
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
    // val numUsers = 943
    val numUsers = ratingData.map { case Array(user, movie, rating, ts) => user }.distinct.size
    // 用户的平均评分
    val ratingsPerUser = numRatings * 1.0 / numUsers
    // 电影数量
    // val numMovies = 1682
    val numMovies = ratingData.map(fields => fields(1)).distinct.size
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

    // 用户-电影评分矩阵
    val matrix = DenseMatrix.zeros[Double](numUsers, numMovies)
    val vectors = List.tabulate(numUsers)(n => DenseVector.zeros[Double](numMovies))
    var vectorsFromMatrix: List[DenseVector[Double]] = List()

    ratingData.map { case Array(user, movie, rating, ts) =>
      matrix.update(user.toInt - 1, movie.toInt - 1, rating.toDouble)
      vectors(user.toInt - 1).update(movie.toInt - 1, rating.toDouble)
    }

    for (row <- 0 until numUsers) {
      val vector = DenseVector.zeros[Double](numMovies)
      for (col <- 0 until numMovies) {
        vector.update(col, matrix.valueAt(row, col))
      }
      vectorsFromMatrix = vectorsFromMatrix :+ vector
    }
    /*
    for (i <- 0 until 10) {
      println(Utils.printVector(vectorsFromMatrix(i), numMovies))
      println(Utils.printVector(vectors(i), numMovies))
    }
    */

    for (i <- 0 until 10) {
      println("vector from matrix")
      vectorsFromMatrix(i).toArray.map(x => (x, 1)).groupBy(_._1).map(e => (e._1, e._2.size)).filter(e => e._1 != 0).toList.sortBy(_._2).reverse.foreach(println)
      println("vector")
      vectors(i).toArray.map(x => (x, 1)).groupBy(_._1).map(e => (e._1, e._2.size)).filter(e => e._1 != 0).toList.sortBy(_._2).reverse.foreach(println)
    }

    val usersMoviesRating = vectors.zipWithIndex.map { case (vector, userIndex) =>
      val ratings = vector.toArray.zipWithIndex.filter(_._1 != 0.0).map { case (rating, movieIndex) =>
        (movieIndex.toInt + 1, rating)
      }
      // (用户ID, 用户对电影的评分)
      (userIndex.toInt + 1, ratings)
    }
    var sameUserCount = 0
    for (id <- 1 to numUsers) {
      val userRatings1 = usersMoviesRating.slice(id - 1, id).map { case (userid, ratings) => ratings.mkString("") }.mkString("")
      val userRatings2 = ratingData.filter(fields => fields(0).toInt == id).sortBy(fields => fields(1).toInt).map(fields => (fields(1), fields(2).toDouble)).mkString("")
      if (userRatings1.equals(userRatings2)) {
        sameUserCount += 1
      }
    }
    println(s"sameUserCount = ${sameUserCount}")

    file.close()
  }
}
