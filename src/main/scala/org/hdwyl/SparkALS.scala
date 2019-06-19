package org.hdwyl

import org.apache.spark.mllib.evaluation.{RankingMetrics, RegressionMetrics}
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.{SparkConf, SparkContext}
import org.jblas.DoubleMatrix

/**
  * Created by wangyanl on 2019/6/16.
  */
object SparkALS {

  def main(args: Array[String]) {
    val conf = new SparkConf()
    val sc = new SparkContext(conf)
    // 读取数据集
    // Data Structure(separated with a tab): userId movieId rating timestamp
    val rawData = sc.textFile("hdfs://PATH/ml-100k/u.data")
    // println(rawData.first())

    // 显示数据集
    // Data Structure: Array(userId, movieId, rating)
    val rawRatings1 = rawData.map(_.split("\t").take(3))
    // println(rawRatings1.first().mkString(" "))
    val ratings1 = rawRatings1.map { case Array(user, movie, rating) => Rating(user.toInt, movie.toInt, rating.toDouble) }
    // println(ratings1.first())
    // 创建MatrixFactorizationModel对象, 该对象将用户因子和物品因子分别保存在一个(id,factor)对类型的RDD中, 它们分别称作userFeatures和productFeatures.
    val model1 = ALS.train(ratings1, 50, 10, 0.01)
    // User's factor: 943
    // Data Structure of the model1.userFeatures: (userId, factor)
    println("User's factor: %d".format(model1.userFeatures.count()))
    // Movie's factor: 1682
    // Data Structure of the model1.productFeatures: (movieId, factor)
    println("Movie's factor: %d".format(model1.productFeatures.count()))
    //
    val userId = 789
    val movieId = 123
    val predictedRating = model1.predict(userId, movieId)
    println("predictedRating: %1.2f".format(predictedRating))

    val movies = sc.textFile("hdfs://PATH/ml-100k/u.item")
    val titles = movies.map(line => line.split("\\|").take(2)).map(e => (e(0).toInt, e(1))).collectAsMap()

    val moviesForUsers = ratings1.keyBy(_.user)
    val userData = rawRatings1.map(e => e(0).toInt).distinct().collect()
    val K = 10
    /*
    for (userId <- userData) {
      val topKRecs = model1.recommendProducts(userId, K)
      println("User: %d, recommend %d Movies: %s".format(userId, K,topKRecs.map(rating => (titles(rating.product), rating.rating)).mkString(" / ")))
      val moviesForUser = moviesForUsers.lookup(userId)
      val ratingsForUser = moviesForUser.sortBy(_.rating)
      val n = if (ratingsForUser.size > 10) 10 else ratingsForUser.size
      val topNMovies = ratingsForUser.take(n).map(rating => (titles(rating.product), rating.rating))
      println("User: %d, top %d Movies: %s".format(userId, n, topNMovies.toArray.mkString(" / ")))
    }
    */

    // 隐式数据集
    val rawRatings2 = rawData.map(_.split("\t").take(3)).map(e => (e(0), e(1), if (e(2).toInt < 3) 0 else 1))
    // println(rawRatings2.first())
    val ratings2 = rawRatings2.map { case (user, movie, rating) => Rating(user.toInt, movie.toInt, rating.toDouble) }
    // println(ratings2.first())
    // 创建MatrixFactorizationModel对象, 该对象将用户因子和物品因子分别保存在一个(id,factor)对类型的RDD中, 它们分别称作userFeatures和productFeatures.
    val model2 = ALS.trainImplicit(ratings2, 50, 10, 0.01, 0.01)
    // User's factor: ?
    println("User's factor: %d".format(model2.userFeatures.count()))
    // Movie's factor: ?
    println("Movie's factor: %d".format(model2.productFeatures.count()))

    val aMatrix = new DoubleMatrix(Array(1.0, 2.0, 3.0))
    println(aMatrix)

    /*
    val itemData = rawRatings1.map(e => e(1).toInt).distinct().collect()
    val K = 10
    for (itemId <- itemData) {
      val itemFactor = model1.productFeatures.lookup(itemId).head
      val itemVector = new DoubleMatrix(itemFactor)
      println(cosineSimilarity(itemVector, itemVector))
    }
    */

    val itemId = 567
    val itemFactor = model1.productFeatures.lookup(itemId).head
    val itemVector = new DoubleMatrix(itemFactor)
    println(cosineSimilarity(itemVector, itemVector))

    // 计算物品567与其它物品的余弦相似度
    val itemSims = model1.productFeatures.map { case (id, factor) =>
      val factorVector = new DoubleMatrix(factor)
      val sim = cosineSimilarity(factorVector, itemVector)
      (id, sim)
    }
    // 取出与物品567最相似的前K个物品
    // top函数能分布式地计算出“前K个”结果
    // collect函数将结果返回驱动程序然后再本地排序
    val sortedItemSims = itemSims.top(K)(Ordering.by[(Int, Double), Double] { case (id, similarity) => similarity })
    sortedItemSims.foreach(println)

    println("编号为%d的电影名称是《%s》".format(itemId, titles(itemId)))
    val sortedItemSims2 = itemSims.top(K + 1)(Ordering.by[(Int, Double), Double] { case (id, similarity) => similarity })
    println("和电影《%s》最相似的%d部电影是:".format(titles(itemId), K))
    sortedItemSims2.slice(1, 11).map { case (id, sim) => (titles(id), sim) }.foreach(println)

    val userFactor = model1.userFeatures.lookup(userId).head
    val userVector = new DoubleMatrix(userFactor)
    println(cosineSimilarity(userVector, userVector))

    // 计算用户123与其他用户的余弦相似度
    val userSims = model1.userFeatures.map { case (id, factor) =>
      val factorVector = new DoubleMatrix(factor)
      val sim = cosineSimilarity(factorVector, userVector)
      (id, sim)
    }
    // 取出与用户123最相似的前K个用户
    // top函数能分布式地计算出“前K个”结果
    // collect函数将结果返回驱动程序然后再本地排序
    val sortedUserSims = userSims.top(K)(Ordering.by[(Int, Double), Double] { case (id, similarity) => similarity })
    sortedUserSims.foreach(println)

    val sortedUserSims2 = userSims.top(K + 1)(Ordering.by[(Int, Double), Double] { case (id, similarity) => similarity })
    println("和用户%d最相似的%d个用户是:".format(userId, K))
    sortedUserSims2.slice(1, 11).foreach(println)

    val moviesForUser = moviesForUsers.lookup(userId)
    val actualRating = moviesForUser.take(1)(0)
    println("用户%d的第一个评级是:%s".format(userId, actualRating))
    // 求模型的预测评级
    val predictedRating1 = model1.predict(userId, actualRating.product)
    println("模型对用户%d的第一个预测评级是:%f".format(userId, predictedRating1))

    // 计算实际评级和预计评级的平方误差
    val squaredError = math.pow(predictedRating - actualRating.rating, 2.0)
    println("实际评级和预计评级的平方误差是:%f".format(squaredError))

    val usersProducts = ratings1.map { case Rating(user, product, rating) => (user, product) }
    val predictions = model1.predict(usersProducts).map {
      case Rating(user, product, rating) => ((user, product), rating)
    }
    val ratingsAndPredictions = ratings1.map {
      case Rating(user, product, rating) => ((user, product), rating)
    }.join(predictions)
    println("ratingsAndPredictions:")
    // Data Structure: ((user, product), (actual, predicted))
    ratingsAndPredictions.take(K).foreach(println)

    // 均方差（Mean Squared Error，MSE）
    val MSE = ratingsAndPredictions.map {
      case ((user, product), (actual, predicted)) => math.pow((actual - predicted), 2)
    }.reduce(_ + _) / ratingsAndPredictions.count
    println("Mean Squared Error = " + MSE)
    // 计算均方根误差（Root Mean Squared Error，RMSE）
    val RMSE = math.sqrt(MSE)
    println("Root Mean Squared Error = " + RMSE)

    // 用户实际评级过的电影ID列表
    val actualMovies = moviesForUser.map(_.product)
    println("actualMovies: %s".format(actualMovies.mkString(", ")))

    // 模型针对用户789推荐的电影
    val topKRecs = model1.recommendProducts(userId, K)
    // 计算得到的推荐电影列表
    val predictedMovies = topKRecs.map(_.product)
    println("predictedMovies: %s".format(predictedMovies.mkString(", ")))

    // 计算平均准确率
    val apk10 = avgPrecisionK(actualMovies, predictedMovies, 10)
    println("apk10 = %f".format(apk10))

    // 使用电影因子向量构建一个DoubleMatrix对象
    val itemFactors = model1.productFeatures.map { case (id, factor) => factor }.collect()
    val itemMatrix = new DoubleMatrix(itemFactors)
    println("itemMatrix: %d rows, %d columns".format(itemMatrix.rows, itemMatrix.columns))

    // 广播itemMatrix
    val imBroadcast = sc.broadcast(itemMatrix)

    val allRecs = model1.userFeatures.map { case (userId, factors) =>
      // 使用用户因子向量构建一个DoubleMatrix对象
      val userVector = new DoubleMatrix(factors)
      val scores = imBroadcast.value.mmul(userVector)
      // 针对评分添加索引并排序
      val sortedWithId = scores.data.zipWithIndex.sortBy(_._1)
      // 针对排序后的(评分,索引)中的索引值+1（索引值从0开始，但电影编号从1开始，所以需要+1），
      // 再转换为列表
      val recommendedIds = sortedWithId.map(_._2 + 1).toSeq
      (userId, recommendedIds)
    }

    val userMovies = ratings1.map { case Rating(user, product, rating) =>
      (user, product)
    }.groupBy(_._1)
    println("allRecs join userMovies:")
    allRecs.join(userMovies).take(K).foreach(println)
    val MAPK = allRecs.join(userMovies).map { case (userId, (predicted, actualWithIds)) =>
      val actual = actualWithIds.map(_._2).toSeq
      avgPrecisionK(actual, predicted, K)
    }.reduce(_ + _) / allRecs.count()
    println("Mean Average Precision at K = " + MAPK)

    val predictedAndTrue = ratingsAndPredictions.map { case ((user, product), (predicted, actual)) =>
      (predicted, actual)
    }
    // 使用(预测值,实际值)键值对创建RegressionMetrics
    val regressionMetrics = new RegressionMetrics(predictedAndTrue)
    println("Mean Squared Error = " + regressionMetrics.meanSquaredError)
    println("Root Mean Squared Error = " + regressionMetrics.rootMeanSquaredError)

    val predictedAndTrueForRanking = allRecs.join(userMovies).map {
      case (userId, (predicted, actualWithIds)) =>
        val actual = actualWithIds.map(_._2)
        (predicted.toArray, actual.toArray)
    }
    // 使用(预测的推荐物品ID数组,实际的物品ID数组)键值对创建RankingMetrics
    val rankingMetrics = new RankingMetrics(predictedAndTrueForRanking)
    println("Mean Average Precision = " + rankingMetrics.meanAveragePrecision)

    // 计算全局平均准确率(Mean Average Precision, MAP)
    val MAPK2000 = allRecs.join(userMovies).map {
      case (userId, (predicted, actualWithIds)) =>
        val actual = actualWithIds.map(_._2).toSeq
        avgPrecisionK(actual, predicted, 2000)
    }.reduce(_ + _) / allRecs.count()
    println("Mean Average Precision = " + MAPK2000)

    sc.stop()
  }

  /**
    * 计算向量的余弦相似度
    * 余弦相似度是两个向量在n维空间里两者夹角的度数。它是两个向量的点积与各向量范数（或长度）的乘积的商。
    * （余弦相似度用的范数为L2-范数，即L2-norm。）
    * 该相似度的取值在-1到1之间。1表示完全相似，0表示两者互不相关（即无相似性）。
    * 它还能捕捉负相关性，也就是说，当为-1时则不仅表示两者不相关，还表示它们完全不同。
    *
    * @param vec1
    * @param vec2
    * @return
    */
  def cosineSimilarity(vec1: DoubleMatrix, vec2: DoubleMatrix): Double = {
    vec1.dot(vec2) / (vec1.norm2() * vec2.norm2())
  }

  /**
    * 计算K值平均准确率（Average Precision at K metric，APK）
    * Seq是列表，适合存有序重复数据，进行快速插入/删除元素等场景
    *
    * @param actual    实际列表值
    * @param predicted 对应的预测列表值
    * @param k
    * @return
    */
  def avgPrecisionK(actual: Seq[Int], predicted: Seq[Int], k: Int): Double = {
    val predK = predicted.take(k)
    var score = 0.0
    var numHits = 0.0
    for ((p, i) <- predK.zipWithIndex) {
      if (actual.contains(p)) {
        numHits += 1.0
        score += numHits / (i.toDouble + 1.0)
      }
    }
    if (actual.isEmpty) {
      1.0
    } else {
      score / math.min(actual.size, k).toDouble
    }
  }
}
