package org.hdwyl

import org.apache.log4j.Logger
import org.apache.spark.mllib.evaluation.{RankingMetrics, RegressionMetrics}
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.{SparkConf, SparkContext}
import org.jblas.DoubleMatrix

/**
  * Created by wangyanl on 2019/6/16.
  */
object SparkALS {

  @transient lazy val logger = Logger.getLogger(this.getClass)

  def main(args: Array[String]) {
    val conf = new SparkConf()
    val sc = new SparkContext(conf)
    val userId = 789
    val movieId = 123
    val K = 10

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
    // rank：对应ALS模型中的因子个数，也就是在低阶近似矩阵中的隐含特征个数。因子个数一般越多越好。
    // 但它也会直接影响模型训练和保存时所需的内存开销，尤其是在用户和物品很多的时候。
    // 因此实践中该参数常作为训练效果与系统开销之间的调节参数。通常，其合理取值为10到200。
    // iterations：对应运行时的迭代次数。ALS能确保每次迭代都能降低评级矩阵的重建误差，但一般经少数次迭代后
    // ALS模型便已能收敛为一个比较合理的好模型。这样，大部分情况下都没必要迭代太多次（10次左右一般就挺好）。
    // lambda：该参数控制模型的正则化过程，从而控制模型的过拟合情况。其值越高，正则化越严厉。
    // 该参数的赋值与实际数据的大小、特征和稀疏程度有关。和其它的机器学习模型一样，正则参数应该通过用非样本的测试数据进行交叉验证来调整。
    // 方法返回一个MatrixFactorizationModel对象, 该对象将用户因子和物品因子分别保存在一个(id, factors)对类型的RDD中，
    // 它们分别称作userFeatures和productFeatures。
    val model1 = ALS.train(ratings1, 50, 10, 0.01)
    // User's factor: 943
    // Data Structure of the model1.userFeatures: RDD[(Int, Array[Double])]
    println("User's factor: %d".format(model1.userFeatures.count()))
    model1.userFeatures.map { case (id, factors) => (id, factors.mkString(", ")) }.take(K).foreach(println)
    // Movie's factor: 1682
    // Data Structure of the model1.productFeatures: RDD[(Int, Array[Double])]
    println("Movie's factor: %d".format(model1.productFeatures.count()))
    model1.productFeatures.map { case (id, factors) => (id, factors.mkString(", ")) }.take(K).foreach(println)

    // 隐式数据集
    val rawRatings2 = rawData.map(_.split("\t").take(3)).map(e => (e(0), e(1), if (e(2).toInt < 3) 0 else 1))
    // println("rawRatings2")
    // rawRatings2.take(K).foreach(println)
    val ratings2 = rawRatings2.map { case (user, movie, rating) => Rating(user.toInt, movie.toInt, rating.toDouble) }
    // println("ratings2:")
    // ratings2.take(K).foreach(println)
    val alphas = List(5.0, 4.0, 3.0, 2.0, 1.0)
    for (alpha <- alphas) {
      // alpha参数指定了信心权重所应达到的基准线。该值越高则所训练出的模型越认为用户与他没评级过的电影之间没有相关性。
      // 方法返回一个MatrixFactorizationModel对象, 该对象将用户因子和物品因子分别保存在一个(id, factors)对类型的RDD中，
      // 它们分别称作userFeatures和productFeatures。
      val model2 = ALS.trainImplicit(ratings2, 50, 10, 0.01, alpha)
      // User's factor: 943
      println("User's factor: %d".format(model2.userFeatures.count()))
      // Movie's factor: 1682
      println("Movie's factor: %d".format(model2.productFeatures.count()))
      val predictedRating2 = model2.predict(userId, movieId)
      println("alpha = %f".format(alpha))
      println("userId:%d, movieId:%d, predictedRating: %1.2f".format(userId, movieId, predictedRating2))
    }

    // 计算用户789对电影123的预期得分
    val predictedRating1 = model1.predict(userId, movieId)
    println("userId:%d, movieId:%d, predictedRating: %1.2f".format(userId, movieId, predictedRating1))

    /*
    val userData = rawRatings1.map(e => e(0).toInt).distinct().collect()
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

    // 计算给用户789推荐的前K部电影
    val topKRecs = model1.recommendProducts(userId, K)
    // 计算得到的推荐电影列表
    val predictedMovies = topKRecs.map(_.product)
    println("predictedMovies: %s".format(predictedMovies.mkString(", ")))

    val movies = sc.textFile("hdfs://PATH/ml-100k/u.item")
    val titles = movies.map(line => line.split("\\|").take(2)).map(e => (e(0).toInt, e(1))).collectAsMap()
    // println("Movie Titles:")
    // titles.take(K).foreach(println)
    println("编号为%d的电影名称是%s".format(userId, titles(userId)))

    // keyBy: 为各个元素按指定的函数生成key, 形成key-value的RDD.
    val moviesForUsers = ratings1.keyBy(_.user)
    // println("moviesForUsers:")
    // moviesForUsers.take(K).foreach(println)
    val moviesForUser = moviesForUsers.lookup(userId)
    println("用户%d评价了%d部电影".format(userId, moviesForUser.size))

    println("用户%d评价最高的%d部电影是:".format(userId, K))
    moviesForUser.sortBy(_.rating).reverse.take(K).map(rating => (titles(rating.product), rating.rating)).foreach(println)

    println("给用户%d推荐的前%d部电影是:".format(userId, K))
    topKRecs.map(rating => (titles(rating.product), rating.rating)).foreach(println)

    val aMatrix = new DoubleMatrix(Array(1.0, 2.0, 3.0))
    println("aMatrix:")
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
    println("%d's productFeatures:".format(itemId))
    model1.productFeatures.lookup(itemId).map(e => e.mkString(", ")).foreach(println)
    println(model1.productFeatures.lookup(itemId).head.mkString(", "))
    // Data Structure of the itemFactor: Array[Double]
    val itemFactor = model1.productFeatures.lookup(itemId).head
    val itemVector = new DoubleMatrix(itemFactor)
    println(cosineSimilarity(itemVector, itemVector))

    // 计算物品567与其它物品的余弦相似度
    val itemSims = model1.productFeatures.map { case (id, factors) =>
      // Data Structure of the factors: Array[Double]
      val factorVector = new DoubleMatrix(factors)
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

    println("%d's userFeatures:".format(userId))
    // Data Structure of the userFactor: Array[Double]
    val userFactor = model1.userFeatures.lookup(userId).head
    val userVector = new DoubleMatrix(userFactor)
    println(cosineSimilarity(userVector, userVector))

    // 计算用户789与其他用户的余弦相似度
    val userSims = model1.userFeatures.map { case (id, factors) =>
      // Data Structure of the factors: Array[Double]
      val factorVector = new DoubleMatrix(factors)
      val sim = cosineSimilarity(factorVector, userVector)
      (id, sim)
    }
    // 取出与用户789最相似的前K个用户
    // top函数能分布式地计算出“前K个”结果
    // collect函数将结果返回驱动程序然后再本地排序
    val sortedUserSims = userSims.top(K)(Ordering.by[(Int, Double), Double] { case (id, similarity) => similarity })
    sortedUserSims.foreach(println)

    val sortedUserSims2 = userSims.top(K + 1)(Ordering.by[(Int, Double), Double] { case (id, similarity) => similarity })
    println("和用户%d最相似的%d个用户是:".format(userId, K))
    sortedUserSims2.slice(1, 11).foreach(println)

    // 均方差（Mean Squared Error，MSE）直接衡量“用户-物品”评级矩阵的重建误差。它常用于显式评级的情形。
    // 它的定义为各平方误差的和与总数目的商。其中平方误差是指预测到的评级与真实评级的差值的平方。

    // 取出用户789的第一个电影评级
    // Rating(userId, itemId, rating)
    val actualRating = moviesForUser.take(1)(0)
    println("用户%d的第一个评级是:%s".format(userId, actualRating))
    // 求模型的预测评级
    val predictedRating3 = model1.predict(userId, actualRating.product)
    println("模型对用户%d的第一个预测评级是:%f".format(userId, predictedRating3))

    // 计算实际评级和预计评级的平方误差
    val squaredError = math.pow(predictedRating3 - actualRating.rating, 2.0)
    println("实际评级和预计评级的平方误差是:%f".format(squaredError))

    // 从ratingsRDD里提取用户和物品的ID
    val usersProducts = ratings1.map { case Rating(user, product, rating) => (user, product) }
    // 使用model.predict来对各个“用户-物品”对做预测，然后转换为以“用户和物品ID”对作为主键，对应的预计评级作为值的RDD。
    val predictions = model1.predict(usersProducts).map {
      case Rating(user, product, rating) => ((user, product), rating)
    }
    // 转换ratings为以“用户和物品ID”对作为主键，实际的评级作为值的RDD。
    // 将前面2个RDD进行连接，创建一个新的RDD，这个RDD的主键为“用户和物品ID”对，键值为相应的实际评级和预计评级。
    val ratingsAndPredictions = ratings1.map {
      case Rating(user, product, rating) => ((user, product), rating)
    }.join(predictions)
    println("ratingsAndPredictions:")
    // Data Structure: ((user, product), (actual, predicted))
    ratingsAndPredictions.take(K).foreach(println)

    // 求均方差（Mean Squared Error，MSE），先用reduce来对平方误差求和，然后再除以count函数所求得的总记录数
    val MSE = ratingsAndPredictions.map {
      case ((user, product), (actual, predicted)) => math.pow((actual - predicted), 2.0)
    }.reduce(_ + _) / ratingsAndPredictions.count
    println("Mean Squared Error = %f".format(MSE))
    // 计算均方根误差（Root Mean Squared Error，RMSE），即在MSE上取平方根
    val RMSE = math.sqrt(MSE)
    println("Root Mean Squared Error = %f".format(RMSE))

    // K值平均准确率（MAPK）的意思是整个数据集上的K值平均准确率（Average Precision at K metric，APK）的均值。
    // APK是信息检索中常用的一个指标。它用于衡量针对某个查询所返回的“前K个”文档的平均相关性。
    // 对于每次查询，我们会将结果中的前K个与实际相关的文档进行比较。
    // 用APK指标计算时，结果中文档的排名十分重要。如果结果中文档的实际相关性越高且排名也更靠前，那APK分值也就越高。

    // 当用APK来做评估推荐模型时，每一个用户相当于一个查询，而每一个“前K个”推荐物组成的集合则相当于一个查到的文档结果集合。
    // 用户对电影的实际评级便对应着文档的实际相关性。这样，APK所试图衡量的是模型对用户感兴趣和会去接触的物品的预测能力。

    // 用户789实际评级过的电影ID列表
    val actualMovies = moviesForUser.map(_.product)
    println("用户%d实际评级过的电影ID列表: %s".format(userId, actualMovies.mkString(", ")))

    // predictedMovies: 模型给用户789推荐的前K个电影ID列表

    // 计算平均准确率
    val apk10 = avgPrecisionK(actualMovies, predictedMovies, 10)
    println("apk10 = %f".format(apk10))

    // 全局MAPK的求解要计算对每一个用户的APK得分，再求其平均。这就要为每一个用户都生成相应的推荐列表。

    // 使用电影因子向量构建一个DoubleMatrix对象
    val itemFactors = model1.productFeatures.map { case (id, factors) => factors }.collect()
    // Data Structure of the itemFactors: Array[Double]
    val itemMatrix = new DoubleMatrix(itemFactors)
    // itemMatrix: 1682 rows, 50 columns
    println("itemMatrix: %d rows, %d columns".format(itemMatrix.rows, itemMatrix.columns))

    // 广播itemMatrix，以便每个工作节点都能访问到
    val imBroadcast = sc.broadcast(itemMatrix)

    // 计算每一个用户的推荐
    val allRecs = model1.userFeatures.map { case (userId, factors) =>
      // 使用用户因子向量构建一个DoubleMatrix对象
      val userVector = new DoubleMatrix(factors)
      // 对用户因子矩阵和电影因子矩阵做乘积，其结果为一个表示各个电影预计评级的向量（长度为1682，即电影的总数目）
      val scores = imBroadcast.value.mmul(userVector)
      // 针对评分添加索引并根据评分排序
      val sortedWithId = scores.data.zipWithIndex.sortBy(_._1)
      // 针对排序后的(评分,索引)对中的索引值+1（索引值从0开始，但电影编号从1开始，所以需要+1），再转换为列表
      val recommendedIds = sortedWithId.map(_._2 + 1).toSeq
      // 创建(用户ID, 推荐电影ID列表)，RDD[(Int, Seq[Int])]
      (userId, recommendedIds)
    }

    // 取出全部的(用户ID, 已评级电影ID列表)对，RDD[(Int, Seq[(Int, Int)])]
    val userMovies = ratings1.map { case Rating(user, product, rating) =>
      (user, product)
    }.groupBy(_._1)
    println("allRecs join userMovies:")
    allRecs.join(userMovies).take(K).foreach(println)
    val MAPK = allRecs.join(userMovies).map { case (userId, (predicted, actualWithIds)) =>
      val actual = actualWithIds.map(_._2).toSeq
      avgPrecisionK(actual, predicted, K)
    }.reduce(_ + _) / allRecs.count()
    println("Mean Average Precision at K = %f".format(MAPK))

    val predictedAndTrue = ratingsAndPredictions.map { case ((user, product), (actual, predicted)) =>
      (predicted, actual)
    }
    println("predictedAndTrue:")
    predictedAndTrue.take(K).foreach(println)

    // 使用(预测值,实际值)键值对创建RegressionMetrics
    val regressionMetrics = new RegressionMetrics(predictedAndTrue)
    println("Mean Squared Error = %f".format(regressionMetrics.meanSquaredError))
    println("Root Mean Squared Error = %f".format(regressionMetrics.rootMeanSquaredError))

    val predictedAndTrueForRanking = allRecs.join(userMovies).map {
      case (userId, (predicted, actualWithIds)) =>
        val actual = actualWithIds.map(_._2)
        (predicted.toArray, actual.toArray)
    }
    println("predictedAndTrueForRanking:")
    predictedAndTrueForRanking.map(e => (e._1.mkString(", "), e._2.mkString(", "))).take(K).foreach(println)

    // 使用(预测的推荐物品ID数组,实际的物品ID数组)键值对创建RankingMetrics
    val rankingMetrics = new RankingMetrics(predictedAndTrueForRanking)
    println("Mean Average Precision = %f".format(rankingMetrics.meanAveragePrecision))

    // 计算全局平均准确率(Mean Average Precision, MAP)
    val MAPK2000 = allRecs.join(userMovies).map {
      case (userId, (predicted, actualWithIds)) =>
        val actual = actualWithIds.map(_._2).toSeq
        avgPrecisionK(actual, predicted, 2000)
    }.reduce(_ + _) / allRecs.count()
    println("Mean Average Precision = %f".format(MAPK2000))

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
