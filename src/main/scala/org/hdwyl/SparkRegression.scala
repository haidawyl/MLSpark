package org.hdwyl

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.{L1Updater, SquaredL2Updater, Updater}
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.{Map, immutable}

/**
  * 线性回归在应用L2正则化时通常称为岭回归（ridge regression），
  * 应用L1正则化时称为LASSO（Least Absolute Shrinkage and Selection Operator）。
  * 决策树在用于回归时使用的不纯度度量方法是方差。
  *
  * Created by wangyanl on 2019/6/30.
  */
object SparkRegression {

  def main(args: Array[String]) {
    val conf = new SparkConf()
    val sc = new SparkContext(conf)

    // 读取数据集
    val rawData = sc.textFile("hdfs://PATH/BikeSharing/hour_noheader.csv")
    val numData = rawData.count
    // numData = 17379
    println(s"numData = ${numData}")
    val records = rawData.map(line => line.split(","))
    val first = records.first()
    println(first.mkString(", "))

    // 缓存数据
    records.cache()

    println("Mapping of first categorical features column: " + getMapping(records, 2))

    val mappings = new Array[Map[String, Long]](8)
    for (i <- 2 until 10) {
      mappings(i - 2) = getMapping(records, i)
    }
    val catLen = mappings.map(m => m.size).sum
    val numLen = first.slice(11, 15).size
    val totalLen = numLen + catLen

    // Feature vector length for categorical features: 57
    println(s"Feature vector length for categorical features: ${catLen}")
    // Feature vector length for numerical features: 4
    println(s"Feature vector length for numerical features: ${numLen}")
    // Total feature vector length: 61
    println(s"Total feature vector length: ${totalLen}")

    // 提取每条数据记录的特征向量和标签
    val data = records.map(r => LabeledPoint(extractLabel(r), Vectors.dense(extractFeatures(mappings, r, catLen))))
    data.cache()
    val firstPoint = data.first()
    println("Raw data: %s".format(first.slice(2, first.size - 3).mkString(",")))
    println("Label: %f".format(firstPoint.label))
    println("Linear Model feature vector:\n %s".format(firstPoint.features.toString))
    println("Linear Model feature vector length: %d".format(firstPoint.features.size))

    val dataDt = records.map(r => LabeledPoint(extractLabel(r), Vectors.dense(extractFeaturesDt(r))))
    dataDt.cache()
    val firstPointDt = dataDt.first()
    println("Decision Tree feature vector:\n %s".format(firstPointDt.features.toString))
    println("Decision Tree feature vector length: %d".format(firstPointDt.features.size))

    val lr = new LinearRegressionWithSGD()
    lr.optimizer.setNumIterations(10).setStepSize(0.1)
    lr.setIntercept(false)
    val lrModel = lr.run(data)
    val trueVsPredicted = data.map(p => (p.label, lrModel.predict(p.features)))
    println("Linear Model predictions: %s".format(trueVsPredicted.take(5).toString))

    // categoricalFeaturesInfo为一个字典参数，这个字典参数将类型特征的索引映射到特征中类型的数目。
    // 如果某个特征值不在这个字典中，则将其映射设置为空。
    val categoricalFeaturesInfo = immutable.Map[Int, Int]()
    val impurity = "variance"
    val maxDepth = 5
    val maxBins = 16
    val dtModel = DecisionTree.trainRegressor(dataDt, categoricalFeaturesInfo, impurity, maxDepth, maxBins)
    val predictions = dtModel.predict(dataDt.map(p => p.features))
    val actual = dataDt.map(p => p.label)
    val trueVsPredictedDt = actual.zip(predictions)
    println("Decision Tree predictions: %s".format(trueVsPredictedDt.take(5).toString))
    println("Decision Tree depth: %d".format(dtModel.depth))
    println("Decision Tree number of nodes: %d".format(dtModel.numNodes))

    // 用于评估回归模型的方法包括：
    // 均方误差（MSE，Mean Squared Error）、
    // 均方根误差（RMSE，Root Mean Squared Error）、
    // 平均绝对误差（MAE，Mean Absolute Error）、
    // R-平方系数（R-squared coefficient）等。

    val mse = trueVsPredicted.map { case (t, p) => squaredError(t, p) }.mean()
    val mae = trueVsPredicted.map { case (t, p) => absError(t, p) }.mean()
    val rmsle = math.sqrt(trueVsPredicted.map { case (t, p) => squaredLogError(t, p) }.mean())
    println(s"Linear Model - Mean Squared Error: ${mse}%2.4f")
    println(s"Linear Model - Mean Absolute Error: ${mae}%2.4f")
    println(s"Linear Model - Root Mean Squared Log Error: ${rmsle}%2.4f")

    val mseDt = trueVsPredictedDt.map { case (t, p) => squaredError(t, p) }.mean()
    val maeDt = trueVsPredictedDt.map { case (t, p) => absError(t, p) }.mean()
    val rmsleDt = math.sqrt(trueVsPredictedDt.map { case (t, p) => squaredLogError(t, p) }.mean())
    println(s"Decision Tree - Mean Squared Error: ${mseDt}%2.4f")
    println(s"Decision Tree - Mean Absolute Error: ${maeDt}%2.4f")
    println(s"Decision Tree - Root Mean Squared Log Error: ${rmsleDt}%2.4f")

    // 对目标变量进行对数变换
    val dataLog = data.map(lp => LabeledPoint(math.log(lp.label), lp.features))
    val lrLog = new LinearRegressionWithSGD()
    lrLog.optimizer.setNumIterations(10).setStepSize(0.1)
    lrLog.setIntercept(false)
    val lrModelLog = lrLog.run(data)
    val trueVsPredictedLog = dataLog.map(p => (math.exp(p.label), math.exp(lrModelLog.predict(p.features))))
    val mseLog = trueVsPredictedLog.map { case (t, p) => squaredError(t, p) }.mean()
    val maeLog = trueVsPredictedLog.map { case (t, p) => absError(t, p) }.mean()
    val rmsleLog = math.sqrt(trueVsPredictedLog.map { case (t, p) => squaredLogError(t, p) }.mean())
    println("对目标变量进行对数变换后训练线性回归模型计算得到的MSE、MAE和RMSLE")
    println(s"Mean Squared Error: ${mseLog}%2.4f")
    println(s"Mean Absolute Error: ${maeLog}%2.4f")
    println(s"Root Mean Squared Log Error: ${rmsleLog}%2.4f")
    println("Non log-transformed predictions:\n" + trueVsPredicted.take(3).toString)
    println("Log-transformed predictions:\n" + trueVsPredictedLog.take(3).toString)

    val dataDtLog = dataDt.map(lp => LabeledPoint(math.log(lp.label), lp.features))
    val dtModelLog = DecisionTree.trainRegressor(dataDtLog, categoricalFeaturesInfo, impurity, maxDepth, maxBins)
    val predictionsLog = dtModelLog.predict(dataDtLog.map(p => p.features))
    val actualLog = dataDtLog.map(p => p.label)
    val trueVsPredictedDtLog = actualLog.zip(predictionsLog).map { case (t, p) => (math.exp(t), math.exp(p)) }
    val mseLogDt = trueVsPredictedDtLog.map { case (t, p) => squaredError(t, p) }.mean()
    val maeLogDt = trueVsPredictedDtLog.map { case (t, p) => absError(t, p) }.mean()
    val rmsleLogDt = math.sqrt(trueVsPredictedDtLog.map { case (t, p) => squaredLogError(t, p) }.mean())
    println("对目标变量进行对数变换后训练决策树模型计算得到的MSE、MAE和RMSLE")
    println(s"Mean Squared Error: ${mseLogDt}%2.4f")
    println(s"Mean Absolute Error: ${maeLogDt}%2.4f")
    println(s"Root Mean Squared Log Error: ${rmsleLogDt}%2.4f")
    println("Non log-transformed predictions:\n" + trueVsPredictedDt.take(3).toString)
    println("Log-transformed predictions:\n" + trueVsPredictedDtLog.take(3).toString)

    // 对目标变量进行取平方根变换
    val dataSqrt = data.map(lp => LabeledPoint(math.sqrt(lp.label), lp.features))
    val lrSqrt = new LinearRegressionWithSGD()
    lrSqrt.optimizer.setNumIterations(10).setStepSize(0.1)
    lrSqrt.setIntercept(false)
    val lrModelSqrt = lrSqrt.run(dataSqrt)
    val trueVsPredictedSqrt = dataSqrt.map(p => (math.pow(p.label, 2), math.pow(lrModelSqrt.predict(p.features), 2)))
    val mseSqrt = trueVsPredictedSqrt.map { case (t, p) => squaredError(t, p) }.mean()
    val maeSqrt = trueVsPredictedSqrt.map { case (t, p) => absError(t, p) }.mean()
    val rmsleSqrt = math.sqrt(trueVsPredictedSqrt.map { case (t, p) => squaredLogError(t, p) }.mean())
    println("对目标变量进行取平方根变换后训练线性回归模型计算得到的MSE、MAE和RMSLE")
    println(s"Mean Squared Error: ${mseSqrt}%2.4f")
    println(s"Mean Absolute Error: ${maeSqrt}%2.4f")
    println(s"Root Mean Squared Log Error: ${rmsleSqrt}%2.4f")
    println("Non sqrt-transformed predictions:\n" + trueVsPredicted.take(3).toString)
    println("Sqrt-transformed predictions:\n" + trueVsPredictedSqrt.take(3).toString)

    val dataDtSqrt = dataDt.map(lp => LabeledPoint(math.sqrt(lp.label), lp.features))
    val dtModelSqrt = DecisionTree.trainRegressor(dataDtSqrt, categoricalFeaturesInfo, impurity, maxDepth, maxBins)
    val predictionsSqrt = dtModelSqrt.predict(dataDtSqrt.map(p => p.features))
    val actualSqrt = dataDtSqrt.map(p => p.label)
    val trueVsPredictedDtSqrt = actualSqrt.zip(predictionsSqrt).map { case (t, p) => (math.pow(t, 2), math.pow(p, 2)) }
    val mseSqrtDt = trueVsPredictedDtSqrt.map { case (t, p) => squaredError(t, p) }.mean()
    val maeSqrtDt = trueVsPredictedDtSqrt.map { case (t, p) => absError(t, p) }.mean()
    val rmsleSqrtDt = math.sqrt(trueVsPredictedDtSqrt.map { case (t, p) => squaredLogError(t, p) }.mean())
    println("对目标变量进行取平方根变换后训练决策树模型计算得到的MSE、MAE和RMSLE")
    println(s"Mean Squared Error: ${mseSqrtDt}%2.4f")
    println(s"Mean Absolute Error: ${maeSqrtDt}%2.4f")
    println(s"Root Mean Squared Log Error: ${rmsleSqrtDt}%2.4f")
    println("Non sqrt-transformed predictions:\n" + trueVsPredictedDt.take(3).toString)
    println("Sqrt-transformed predictions:\n" + trueVsPredictedDtSqrt.take(3).toString)

    // 创建训练集和测试集
    val trainTestData = data.randomSplit(Array(0.8, 0.2), seed = 42)
    val trainData = trainTestData(0)
    val testData = trainTestData(1)
    println("Training data size: %d".format(trainData.count()))
    println("Test data size: %d".format(testData.count()))
    println("Total data size: %d".format(numData))
    println("Train + Test size : %d".format(trainData.count() + testData.count()))

    val trainTestDataDt = dataDt.randomSplit(Array(0.8, 0.2), seed = 42)
    val trainDataDt = trainTestDataDt(0)
    val testDataDt = trainTestDataDt(1)

    // 迭代
    // 通常在使用SGD训练模型的过程中，随着迭代次数增加可以实现更好的性能，
    // 但是性能在迭代次数达到一定数目时会增长得越来越慢。
    val iterParams = Seq(1, 5, 10, 20, 50, 100)
    val iterResults = iterParams.map { param =>
      evaluate(trainData, testData, param, 0.01, 0.0, new SquaredL2Updater, false)
    }
    println(iterParams)
    println(iterResults)

    // 步长
    // SGD模型在步长较大的时候容易收敛到最差的局部最优解，原因是算法收敛太快而不能得到最优解。
    // 小步长与相对较小的迭代次数对应的训练模型性能一般较差，
    // 而较小的步长与较大的迭代次数通常可以收敛得到较好的解。
    // 通常来讲，步长和迭代次数的设定需要权衡。较小的步长意味着收敛速度慢，
    // 需要较大的迭代次数。但是较大的迭代次数更加耗时，特别是在大数据集上。
    val stepParams = Seq(0.01, 0.025, 0.05, 0.1, 1.0)
    val stepResults = stepParams.map { param =>
      evaluate(trainData, testData, 10, param, 0.0, new SquaredL2Updater, false)
    }
    println(stepParams)
    println(stepResults)

    // L2正则化参数
    // 正则化是添加一个关于模型权重向量的函数作为损失项，来惩罚模型的复杂度。
    // 其中L2正则化是对权重向量进行L2-norm惩罚，而L1正则化是对权重向量进行L1-norm惩罚。
    // 随着正则化的提高，训练集的预测性能会下降，因为模型不能很好拟合数据。
    // 但是，设置合适的正则化参数，能够在测试集上达到最好的性能，最终得到一个泛化能力最优的模型。
    val l2RegParams = Seq(0.0, 0.01, 0.1, 1.0, 5.0, 10.0, 20.0)
    val l2RegParamResults = l2RegParams.map { param =>
      evaluate(trainData, testData, 10, 0.1, param, new SquaredL2Updater, false)
    }
    println(l2RegParams)
    println(l2RegParamResults)

    // L1正则化参数
    val l1RegParams = Seq(0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    val l1RegParamResults = l1RegParams.map { param =>
      evaluate(trainData, testData, 10, 0.1, param, new L1Updater, false)
    }
    println(l1RegParams)
    println(l1RegParamResults)

    // 使用L1正则化可以得到稀疏的权重向量。
    val lrL1_1 = new LinearRegressionWithSGD()
    lrL1_1.optimizer.setNumIterations(10).setStepSize(0.1).setRegParam(1.0).setUpdater(new L1Updater)
    lrL1_1.setIntercept(false)
    val lrModelL1_1 = lrL1_1.run(trainData)

    val lrL1_10 = new LinearRegressionWithSGD()
    lrL1_10.optimizer.setNumIterations(10).setStepSize(0.1).setRegParam(10.0).setUpdater(new L1Updater)
    lrL1_10.setIntercept(false)
    val lrModelL1_10 = lrL1_10.run(trainData)

    val lrL1_100 = new LinearRegressionWithSGD()
    lrL1_100.optimizer.setNumIterations(10).setStepSize(0.1).setRegParam(100.0).setUpdater(new L1Updater)
    lrL1_100.setIntercept(false)
    val lrModelL1_100 = lrL1_100.run(trainData)

    // 随着L1的正则化参数越来越大，模型的权重向量中0的数目也越来越大。
    println("L1 (1.0) number of zero weights: " + lrModelL1_1.weights.toArray.map(w => if (w == 0) 1 else 0).sum)
    println("L1 (10.0) number of zero weights: " + lrModelL1_10.weights.toArray.map(w => if (w == 0) 1 else 0).sum)
    println("L1 (100.0) number of zero weights: " + lrModelL1_100.weights.toArray.map(w => if (w == 0) 1 else 0).sum)

    // 截距
    // 截距是添加到权重向量的常数项，可以有效地影响目标变量的中值。
    // 如果数据已经被归一化，截距则没有必要。但是理论上截距的使用并不会带来坏处。
    val interceptParams = Seq(false, true)
    val interceptResults = interceptParams.map { param =>
      evaluate(trainData, testData, 10, 0.1, 1.0, new SquaredL2Updater, param)
    }
    println(interceptParams)
    println(interceptResults)

    // 决策树提供了两个主要的参数：最大树深度和最大划分数。
    // 最大树深度
    val maxDepthParams = Seq(1, 2, 3, 4, 5, 10, 20)
    val maxDepthResults = maxDepthParams.map { param =>
      evaluateDt(trainDataDt, testDataDt, param, 32)
    }
    println(maxDepthParams)
    println(maxDepthResults)

    // 最大划分数
    val maxBinsParams = Seq(2, 4, 8, 16, 32, 64, 100)
    val maxBinsResults = maxBinsParams.map { param =>
      evaluateDt(trainDataDt, testDataDt, 5, param)
    }
    print(maxBinsParams)
    print(maxBinsResults)


    sc.stop()
  }

  /**
    * 将类型特征表示成二维形式，同时将特征值映射到二元向量中非0的位置
    *
    * @param rdd
    * @param idx
    * @return
    */
  def getMapping(rdd: RDD[Array[String]], idx: Int): Map[String, Long] = {
    // 将第idx列的特征值去重，然后对每个值使用zipWithIndex函数映射到一个唯一的索引，
    // 组成一个RDD的键-值映射，键是变量，值是索引。该索引便是特征在二元向量中对应的非0位置。
    return rdd.map(fields => fields(idx)).distinct().zipWithIndex().collectAsMap()
  }

  /**
    * 提取特征
    *
    * @param mappings
    * @param record
    * @param catLen
    * @return
    */
  def extractFeatures(mappings: Array[Map[String, Long]], record: Array[String], catLen: Int): Array[Double] = {
    // 创建长度为catLen的数组，默认值全部为0.0
    val catArr = new Array[Double](catLen)
    // 各个特征的二元编码的累计长度，确保非0特征在整个特征向量中位于正确的位置
    var step = 0
    // 类型特征
    for (i <- 2 until 10) {
      // 特征值在二元编码中的映射，即(特征值, 索引)
      val mapping = mappings(i)
      // 特征值在二元编码中的索引
      val idx = mapping.get(record(i)).get.toInt
      catArr(idx + step) = 1.0
      step = step + mapping.size
    }
    val numArr = new Array[Double](4)
    // 数值特征
    for (i <- 10 until 14) {
      numArr(i - 10) = record(i).toDouble
    }
    return catArr ++ numArr
  }

  /**
    * 提取特征，将各个特征的二元编码拼接在一起
    *
    * @param mappings
    * @param record
    * @param catLen
    * @return
    */
  def extractFeatures(mappings: Array[Map[String, Long]], record: Array[String]): Array[Double] = {
    // 创建空数组
    var catArr = new Array[Double](0)
    // 类型特征
    for (i <- 2 until 10) {
      // 特征值在二元编码中的映射，即(特征值, 索引)
      val mapping = mappings(i)
      // 特征值在二元编码中的索引
      val idx = mapping.get(record(i)).get.toInt
      // 创建数组，大小为特征的二元编码长度
      val fieldArr = new Array[Double](mapping.size)
      fieldArr(idx) = 1.0
      catArr = catArr ++ fieldArr
    }
    val numArr = new Array[Double](4)
    // 数值特征
    for (i <- 10 until 14) {
      numArr(i - 10) = record(i).toDouble
    }
    return catArr ++ numArr
  }

  /**
    * 提取标签
    *
    * @param record
    * @return
    */
  def extractLabel(record: Array[String]): Double = {
    return record(record.size - 1).toDouble
  }

  /**
    * 提取特征，用于决策树模型
    *
    * @param record
    * @return
    */
  def extractFeaturesDt(record: Array[String]): Array[Double] = {
    return record.slice(2, 14).map(x => x.toDouble)
  }

  /**
    * 计算平方误差，即样本预测值和实际值的差的平方
    *
    * @param actual
    * @param predicted
    * @return
    */
  def squaredError(actual: Double, predicted: Double): Double = {
    return math.pow((actual - predicted), 2)
  }

  /**
    * 计算绝对误差，即样本预测值和实际值的差的绝对值
    *
    * @param actual
    * @param predicted
    * @return
    */
  def absError(actual: Double, predicted: Double): Double = {
    return math.abs(actual - predicted)
  }

  /**
    * 计算均方对数误差，即样本预测值和实际值进行对数变换后的MSE（均方误差）
    * 这个度量方法适用于目标变量值域很大，并且没有必要对预测值和目标值的误差进行惩罚的情况。
    * 另外，它也适用于计算误差的百分率而不是误差的绝对值。
    *
    * @param actual
    * @param predicted
    * @return
    */
  def squaredLogError(actual: Double, predicted: Double): Double = {
    return math.pow((math.log(predicted + 1) - math.log(actual + 1)), 2)
  }

  /**
    *
    * @param train
    * @param test
    * @param iterations
    * @param step
    * @param regParam
    * @param updater
    * @param intercept
    * @return
    */
  def evaluate(train: RDD[LabeledPoint], test: RDD[LabeledPoint], iterations: Int, step: Double, regParam: Double, updater: Updater, intercept: Boolean): Double = {
    val lr = new LinearRegressionWithSGD()
    lr.optimizer.setNumIterations(iterations).setStepSize(step).setRegParam(regParam).setUpdater(updater)
    lr.setIntercept(intercept)
    val lrModel = lr.run(train)
    val tp = test.map(p => (p.label, lrModel.predict(p.features)))
    val rmsle = math.sqrt(tp.map { case (t, p) => squaredLogError(t, p) }.mean())
    return rmsle
  }

  /**
    *
    * @param train
    * @param test
    * @param maxDepth
    * @param maxBins
    * @return
    */
  def evaluateDt(train: RDD[LabeledPoint], test: RDD[LabeledPoint], maxDepth: Int, maxBins: Int): Double = {
    val categoricalFeaturesInfo = immutable.Map[Int, Int]()
    val modelDt = DecisionTree.trainRegressor(train, categoricalFeaturesInfo, "variance", maxDepth, maxBins)
    val predicted = modelDt.predict(test.map(p => p.features))
    val actual = test.map(p => p.label)
    val tp = actual.zip(predicted)
    val rmsle = math.sqrt(tp.map { case (t, p) => squaredLogError(t, p) }.mean())
    return rmsle
  }
}
