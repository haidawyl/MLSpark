package org.hdwyl

import breeze.linalg.{DenseVector, sum}
import breeze.numerics.pow
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by wangyanl on 2019/7/7.
  */
object SparkKMeans {

  def main(args: Array[String]) {
    val conf = new SparkConf()
    val sc = new SparkContext(conf)

    // K-均值算法试图将一系列样本分割成K个不同的类簇（其中K是模型的输入参数）。
    // K-均值聚类的目的是最小化所有类簇中的方差之和，其形式化的目标函数称为类簇内的方差和
    // (within cluster sum of squared errors，WCSS)
    // 即计算每个类簇中样本与类中心的平方差，并在最后求和。
    // 标准的K-均值算法初始化K个类中心（为每个类簇中所有样本的平均向量），
    // 后面的过程不断重复迭代下面两个步骤：
    // (1) 将样本分到WCSS最小的类簇中。因为方差之和为欧拉距离的平方，
    // 所以最后等价于将每个样本分配到欧拉距离最近的类中心。
    // (2) 根据第一步类分配情况重新计算每个类簇的类中心。
    // K-均值迭代算法结束条件为达到最大的迭代次数或者收敛。
    // 收敛意味着第一步类分配之后没有改变，因此WCSS的值也没有改变。

    //  K-均值本质上是迭代过程；
    //  模型依赖初始化时类中心的选择（这里指随机选择类中心）；
    //  最后的类簇分配可以很好地分割数据，但是对于较难的数据分割也会不好。

    // K-均值和最小方差回归一样使用方差函数作为优化目标，
    // 因此容易受到离群值（outlier）和较大方差的特征影响。

    // 读取电影数据
    val movies = sc.textFile("hdfs://PATH/ml-100k/u.item")
    println(movies.first())

    // 读取电影题材数据
    val genres = sc.textFile("hdfs://PATH/ml-100k/u.genre")
    genres.take(5).foreach(println)

    // 提取题材的映射关系
    // 先要过滤掉空行，然后分割成<题材, 索引>键值对，最后转换为<索引, 题材>键值对
    val genreMap = genres.filter(!_.isEmpty).map(line => line.split("\\|")).
      map(fields => (fields(1), fields(0))).collectAsMap()
    println(genreMap)

    val titlesAndGenres = movies.map(_.split("\\|")).map { fields =>
      // 提取电影的题材数据
      val genres = fields.toSeq.slice(5, fields.size)
      val genresAssigned = genres.zipWithIndex.filter {
        case (g, idx) => g == "1" // 过滤出对应题材位置值为1的数据
      }.map {
        case (g, idx) => genreMap(idx.toString) // 将题材索引转换为对应的题材名称
      }
      (fields(0).toInt, (fields(1), genresAssigned)) // (电影id, (电影名称, 题材列表))
    }
    println(titlesAndGenres.first())

    // 读取用户评分数据
    val rawData = sc.textFile("hdfs://PATH/ml-100k/u.data")
    // 提取前3列(用户id, 电影id, 评分值)
    val rawRatings = rawData.map(_.split("\t").take(3))
    // 转换为Rating对象
    val ratings = rawRatings.map { case Array(user, movie, rating) =>
      Rating(user.toInt, movie.toInt, rating.toDouble)
    }
    // 缓存
    ratings.cache()
    // 训练最小二乘法（Alternating Least Squares，ALS）推荐模型
    val alsModel = ALS.train(ratings, 50, 10, 0.1)

    // ALS模型训练所得到的电影相关因素的向量
    val movieFeatures = alsModel.productFeatures.map { case (id, factors) => (id, Vectors.dense(factors)) }
    val movieVectors = movieFeatures.map(_._2)
    println(movieVectors.first())
    // ALS模型训练所得到的用户相关因素的向量
    val userFeatures = alsModel.userFeatures.map { case (id, factors) => (id, Vectors.dense(factors)) }
    val userVectors = userFeatures.map(_._2)
    println(userVectors.first())

    // 在训练聚类模型之前，有必要观察一下输入数据的相关因素特征向量的分布，
    // 这可以告诉我们是否需要对训练数据进行归一化。
    // 使用MLlib中的RowMatrix进行各种统计
    val movieMatrix = new RowMatrix(movieVectors)
    val movieMatrixSummary = movieMatrix.computeColumnSummaryStatistics()
    val userMatrix = new RowMatrix(userVectors)
    val userMatrixSummary = userMatrix.computeColumnSummaryStatistics()
    println("Movie factors mean: " + movieMatrixSummary.mean) // 均值
    println("Movie factors variance: " + movieMatrixSummary.variance) // 方差
    println("User factors mean: " + userMatrixSummary.mean)
    println("User factors variance: " + userMatrixSummary.variance)

    // K-均值通常不能收敛到全局最优解，所以实际应用中需要多次训练并选择最优的模型。
    // K
    val numClusters = 5
    // 最大迭代次数
    // 使用较小的迭代次数进行多次训练时，通常得到的训练误差和已经收敛的模型结果类似。
    // 因此，多次训练可以有效找到可能最优的模型。
    val numIteration = 10
    // 训练次数
    val numRuns = 3
    // 在电影相关因素的特征向量上训练K-均值模型
    val movieClusterModel = KMeans.train(movieVectors, numClusters, numIteration)
    val movieClusterModelConverged = KMeans.train(movieVectors, numClusters, 100)

    // 在用户相关因素的特征向量上训练K-均值模型
    val userClusterModel = KMeans.train(userVectors, numClusters, numIteration)
    val userClusterModelConverged = KMeans.train(userVectors, numClusters, 100)

    // 使用训练的K-均值模型进行预测
    val movie1 = movieVectors.first()
    val movieCluster = movieClusterModel.predict(movie1)
    println(movieCluster)
    val movieClusterConverged = movieClusterModelConverged.predict(movie1)
    println(movieClusterConverged)

    val predictions = movieClusterModel.predict(movieVectors)
    println(predictions.take(10).mkString(", "))
    val predictionsConverged = movieClusterModelConverged.predict(movieVectors)
    println(predictionsConverged.take(10).mkString(", "))

    // K-均值最小化的目标函数是样本到其类中心的欧拉距离之和

    // 对每个电影计算其特征向量与所属类簇中心向量的距离
    // titlesAndGenres: (电影id, (电影名称, 题材列表))
    // movieFeatures: (电影id, 相关因素)
    val titlesWithFactors = titlesAndGenres.join(movieFeatures)
    val moviesAssigned = titlesWithFactors.map { case (id, ((title, genres), vector)) =>
      // 使用聚类模型对电影特征向量进行预测
      val prediction = movieClusterModel.predict(vector)
      // 获取预测值的类簇中心向量
      val clusterCentre = movieClusterModel.clusterCenters(prediction)
      // 计算电影特征向量与其类簇中心向量的欧拉距离
      val distance = computeDistance(DenseVector(clusterCentre.toArray), DenseVector(vector.toArray))
      // 电影ID、标题、题材、类别索引，以及电影的特征向量和类中心的距离
      (id, title, genres.mkString(" "), prediction, distance)
    }
    // 根据预测值进行分组
    val clusterMoviesAssignments = moviesAssigned.groupBy { case (id, title, genres, cluster, distance) =>
      cluster
    }.collectAsMap()

    // 枚举每个类簇并输出距离类中心最近的前20部电影
    for ((k, v) <- clusterMoviesAssignments.toSeq.sortBy(_._1)) {
      println(s"Cluster $k:")
      val m = v.toSeq.sortBy(_._5)
      println(m.take(20).map { case (_, title, genres, _, d) =>
        (title, genres, d)
      }.mkString("\n"))
      println("=====\n")
    }

    // 读取用户数据
    val users = sc.textFile("hdfs://PATH/ml-100k/u.user").
      map(_.split("\\|")).map { fields =>
      (fields(0).toInt, (fields(1).toInt, fields(2), fields(3), fields(4)))
    }

    // 用户数量
    val numUsers = rawRatings.map { case Array(user, movie, rating) => user }.distinct().count().toInt
    // 电影数量
    val numMovies = rawRatings.map { fields => fields(1) }.distinct().count().toInt
    // 用户-电影评分矩阵
    val vectors = List.tabulate(numUsers)(n => DenseVector.zeros[Double](numMovies))
    rawRatings.map { case Array(user, movie, rating) =>
      vectors(user.toInt - 1).update(movie.toInt - 1, rating.toDouble)
    }
    val usersMoviesRating = sc.parallelize(vectors).zipWithIndex.map { case (vector, userIndex) =>
      val ratings = vector.toArray.zipWithIndex.filter(_._1 != 0.0).map { case (rating, movieIndex) =>
        (movieIndex.toInt + 1, rating)
      }
      // (用户ID, 用户对电影的评分)
      (userIndex.toInt + 1, ratings)
    }

    val usersWithFactors = users.join(usersMoviesRating).join(userFeatures)
    val usersAssigned = usersWithFactors.map { case (id, (((age, gender, occupation, zipcode), ratings), vector)) =>
      // 使用聚类模型对用户特征向量进行预测
      val prediction = userClusterModel.predict(vector)
      // 获取预测值的类簇中心向量
      val clusterCentre = userClusterModel.clusterCenters(prediction)
      // 计算用户特征向量与其类簇中心向量的欧拉距离
      val distance = computeDistance(DenseVector(clusterCentre.toArray), DenseVector(vector.toArray))
      // 用户ID、类别索引，以及用户的特征向量和类中心的距离
      (id, age, gender, occupation, zipcode, ratings, prediction, distance)
    }
    // 根据预测值进行分组
    val clusterUsersAssignments = usersAssigned.groupBy { case (id, age, gender, occupation, zipcode, ratings, cluster, distance) =>
      cluster
    }.collectAsMap()
    // 枚举每个类簇并输出距离类中心最近的前20部电影
    for ((k, v) <- clusterUsersAssignments.toSeq.sortBy(_._1)) {
      println(s"Cluster $k:")
      val u = v.toSeq.sortBy(_._7)
      println(u.take(20).map { case (_, age, gender, occupation, zipcode, ratings, _, d) =>
        (age, gender, occupation, zipcode, ratings.mkString(", "), d)
      }.mkString("\n"))
      println("=====\n")
    }

    // 聚类的评估通常分为两部分：内部评估和外部评估。
    // 内部评估表示评估过程使用训练模型时使用的训练数据，外部评估则使用训练数据之外的数据。

    // 通用的内部评价指标包括WCSS（类簇内的方差和, within cluster sum of squared errors）、
    // Davies-Bouldin指数、Dunn指数和轮廓系数（silhouette coefficient）。
    // 所有这些度量指标都是使类簇内部的样本距离尽可能接近，不同类簇的样本相对较远。

    // MLlib提供的函数computeCost可以方便地计算出给定输入数据RDD[Vector]的WCSS。
    // 计算电影训练数据的性能
    val movieCost = movieClusterModel.computeCost(movieVectors)
    // 计算用户训练数据的性能
    val userCost = userClusterModel.computeCost(userVectors)
    // WCSS for movies: 2586.0777166339426
    println("WCSS for movies: " + movieCost)
    // WCSS for users: 1403.4137493396831
    println("WCSS for users: " + userCost)

    // K-均值模型只有一个可以调的参数，就是K，即类中心数目。

    // 将电影数据集分割为训练集和测试集
    val trainTestSplitMovies = movieVectors.randomSplit(Array(0.6, 0.4), 123)
    val trainMovies = trainTestSplitMovies(0)
    val testMovies = trainTestSplitMovies(1)
    // 计算不同K值训练的K-均值聚类模型的WCSS
    val costsMovies = Seq(2, 3, 4, 5, 10, 20).map { k =>
      (k, KMeans.train(trainMovies, k, 100).computeCost(testMovies))
    }
    println("Movie clustering cross-validation:")
    costsMovies.foreach { case (k, cost) => println(f"WCSS for K=$k id ${cost}%2.2f") }

    // 从结果可以看出，随着类中心数目增加，WCSS值会出现下降，然后又开始增大。
    // 另外一个现象，K-均值在交叉验证的情况，WCSS随着K的增大持续减小，但是达到某个值后，
    // 下降的速率突然会变得很平缓。这时的K通常为最优的K值（这称为拐点）。

    // 将用户数据集分割为训练集和测试集
    val trainTestSplitUsers = userVectors.randomSplit(Array(0.6, 0.4), 123)
    val trainUsers = trainTestSplitUsers(0)
    val testUsers = trainTestSplitUsers(1)
    // 计算不同K值训练的K-均值聚类模型的WCSS
    val costsUsers = Seq(2, 3, 4, 5, 10, 20).map { k =>
      (k, KMeans.train(trainUsers, k, 100).computeCost(testUsers))
    }
    println("User clustering cross-validation:")
    costsUsers.foreach { case (k, cost) => println(f"WCSS for K=$k id ${cost}%2.2f") }

    sc.stop()
  }

  /**
    * 计算两个向量的欧拉距离
    *
    * @param v1
    * @param v2
    * @return
    */
  def computeDistance(v1: DenseVector[Double], v2: DenseVector[Double]) = sum(pow(v1 - v2, 2))
}
