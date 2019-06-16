package org.hdwyl

import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by wangyanl on 2019/6/16.
  */
object SparkALS {

  def main(args: Array[String]) {
    val conf = new SparkConf()
    val sc = new SparkContext(conf)
    // 读取数据集
    val rawData = sc.textFile("hdfs://PATH/ml-100k/u.data")
    println(rawData.first())

    // 显示数据集
    val rawRatings1 = rawData.map(_.split("\t").take(3))
    println(rawRatings1.first())
    val ratings1 = rawRatings1.map { case Array(user, movie, rating) => Rating(user.toInt, movie.toInt, rating.toDouble) }
    println(ratings1.first())
    // 创建MatrixFactorizationModel对象, 该对象将用户因子和物品因子分别保存在一个(id,factor)对类型的RDD中, 它们分别称作userFeatures和productFeatures.
    val model1 = ALS.train(ratings1, 50, 10, 0.01)
    // User's factor: 943
    println("User's factor: %d".format(model1.userFeatures.count()))
    // Movie's factor: 1682
    println("Movie's factor: %d".format(model1.productFeatures.count()))
    val predictedRating = model1.predict(789, 123)
    println("predictedRating: %1.2f".format(predictedRating))

    val movies = sc.textFile("hdfs://PATH/ml-100k/u.item")
    val titles = movies.map(line => line.split("\\|").take(2)).map(e => (e(0).toInt, e(1))).collectAsMap()

    val moviesForUsers = ratings1.keyBy(_.user)
    val userData = rawRatings1.map(e => e(0).toInt).distinct()
    val K = 10
    for (user <- userData) {
      val topKRecs = model1.recommendProducts(user, K)
      println("User: %d, recommend %d Movies: %s".format(user, K,topKRecs.map(rating => (titles(rating.product), rating.rating)).mkString(" / ")))
      val moviesForUser = moviesForUsers.lookup(user)
      val ratingsForUser = moviesForUser.sortBy(_.rating)
      val n = if (ratingsForUser.size > 10) 10 else ratingsForUser.size
      val topNMovies = ratingsForUser.take(n).map(rating => (titles(rating.product), rating.rating))
      println("User: %d, top %d Movies: %s".format(user, n, topNMovies.toArray.mkString(" / ")))
    }

    // 隐式数据集
    val rawRatings2 = rawData.map(_.split("\t").take(3)).map(e => (e(0), e(1), if (e(2).toInt < 3) 0 else 1))
    println(rawRatings2.first())
    val ratings2 = rawRatings2.map { case (user, movie, rating) => Rating(user.toInt, movie.toInt, rating.toDouble) }
    println(ratings2.first())
    // 创建MatrixFactorizationModel对象, 该对象将用户因子和物品因子分别保存在一个(id,factor)对类型的RDD中, 它们分别称作userFeatures和productFeatures.
    val model2 = ALS.trainImplicit(ratings2, 50, 10, 0.01, 0.01)
    // User's factor: ?
    println("User's factor: %d".format(model2.userFeatures.count()))
    // Movie's factor: ?
    println("Movie's factor: %d".format(model2.productFeatures.count()))

    sc.stop()
  }
}
