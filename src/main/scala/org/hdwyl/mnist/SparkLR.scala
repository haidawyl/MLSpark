package org.hdwyl.mnist

import org.apache.spark.mllib.classification.{ClassificationModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.optimization.{L1Updater, SimpleUpdater, SquaredL2Updater, Updater}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by wangyanl on 2019/6/27.
  */
object SparkLR {

  def main(args: Array[String]) {
    val conf = new SparkConf()
    val sc = new SparkContext(conf)

    // 训练数据集
    val trainFeatureFile = "hdfs://PATH/mnist/train-images.idx3-ubyte"
    val trainLabelFile = "hdfs://PATH/mnist/train-labels.idx1-ubyte"

    val trainImagesAsMatrices = MnistHdfsReader.loadFeature(sc, trainFeatureFile)
    val trainLabelsAsInts = MnistHdfsReader.loadLabels(sc, trainLabelFile)

    /*
    val trainData = sc.parallelize(trainImagesAsMatrices).zipWithIndex.map { case (image, index) =>
      val features = image.toArray
      val label = trainLabelsAsInts.toList(index.toInt)
      LabeledPoint(label, Vectors.dense(features))
    }
    */
    val trainData = sc.parallelize(trainImagesAsMatrices zip trainLabelsAsInts).map { case (image, label) =>
      LabeledPoint(label, Vectors.dense(image.toArray))
    }
    trainData.cache()

    // 测试数据集
    val testFeatureFile = "hdfs://PATH/mnist/t10k-images.idx3-ubyte"
    val testLabelFile = "hdfs://PATH/mnist/t10k-labels.idx1-ubyte"

    val testImagesAsMatrices = MnistHdfsReader.loadFeature(sc, testFeatureFile)
    val testLabelsAsInts = MnistHdfsReader.loadLabels(sc, testLabelFile)

    /*
    val testData = sc.parallelize(testImagesAsMatrices).zipWithIndex.map { case (image, index) =>
      val features = image.toArray
      val label = testLabelsAsInts.toList(index.toInt)
      LabeledPoint(label, Vectors.dense(features))
    }
    */
    val testData = sc.parallelize(testImagesAsMatrices zip testLabelsAsInts).map { case (image, label) =>
      LabeledPoint(label, Vectors.dense(image.toArray))
    }
    testData.cache()

    // 训练逻辑回归模型
    val lrModel = new LogisticRegressionWithLBFGS().setNumClasses(10).run(trainData)

    // 使用模型对单个数据进行预测
    val dataPoint = trainData.first()
    val prediction = lrModel.predict(dataPoint.features)
    println("prediction = %f".format(prediction))
    // 数据的真实标签
    val trueLabel = dataPoint.label
    println("trueLabel = %f".format(trueLabel))

    val K = 10
    // 使用模型对整体数据进行预测
    val predictions = lrModel.predict(trainData.map(lp => lp.features))
    println("predictions:")
    predictions.take(K).foreach(println)
    // 对应数据的真实标签
    println("trueLabels:")
    trainData.map(lp => lp.label).take(K).foreach(println)

    // 计算模型的正确率
    val lrTotalCorrect = trainData.map { point =>
      if (lrModel.predict(point.features) == point.label) 1 else 0
    }.sum
    val lrAccuracy = lrTotalCorrect / trainData.count
    println(f"lrAccuracy = ${lrAccuracy * 100.0}%2.4f%%")

    // 评估模型
    val metrics = Seq(lrModel).map { model =>
      val scoreAndLabels = trainData.map { point =>
        (model.predict(point.features), point.label)
      }
      val metrics = new MulticlassMetrics(scoreAndLabels)
      (model.getClass.getSimpleName, metrics.accuracy, metrics.weightedPrecision, metrics.weightedRecall, metrics.weightedFMeasure)
    }

    metrics.foreach { case (m, accuracy, weightedPrecision, weightedRecall, weightedFMeasure) =>
      println(f"$m, Accuracy: ${accuracy * 100.0}%2.4f%%, WeightedPrecision: ${weightedPrecision * 100.0}%2.4f%%, " +
        f"WeightedRecall: ${weightedRecall * 100.0}%2.4f%%, WeightedFMeasure: ${weightedFMeasure * 100.0}%2.4f%%")
    }

    // 将特征向量用RowMatrix类表示成MLlib中的分布矩阵。
    // RowMatrix是一个由向量组成的RDD，其中每个向量是分布矩阵的一行。
    val vectors = trainData.map(lp => lp.features)
    val matrix = new RowMatrix(vectors)
    // computeColumnSummaryStatistics方法计算特征矩阵每列的不同统计数据，
    // 包括均值和方差，所有统计值按每列一项的方式存储在一个Vector中
    val matrixSummary = matrix.computeColumnSummaryStatistics()
    // 输出矩阵每列的均值
    println("mean = " + matrixSummary.mean)
    // 输出矩阵每列的最小值
    println("min = " + matrixSummary.min)
    // 输出矩阵每列的最大值
    println("max = " + matrixSummary.max)
    // 输出矩阵每列的方差
    println("variance = " + matrixSummary.variance)
    // 输出矩阵每列中非0项的数目
    println("numNonzeros = " + matrixSummary.numNonzeros)

    // 使用Spark的StandardScaler中的方法对每个特征进行标准化，使得每个特征是0均值和单位标准差。
    // withMean: 表示是否从数据中减去均值
    // withStd: 表示是否应用标准差缩放
    val scaler = new StandardScaler(withMean = true, withStd = true).fit(vectors)
    // 输入向量传到转换函数，并且返回归一化的向量。
    val scaledTrainData = trainData.map(lp => LabeledPoint(lp.label, scaler.transform(lp.features)))
    val scaledTestData = testData.map(lp => LabeledPoint(lp.label, scaler.transform(lp.features)))
    println("标准化之前的特征:")
    println(trainData.first.features)
    println("标准化之后的特征:")
    println(scaledTrainData.first.features)

    // 使用标准化之后的数据重新训练模型
    val lrModelScaled = new LogisticRegressionWithLBFGS().setNumClasses(10).run(scaledTrainData)
    val lrTotalCorrectScaled = scaledTrainData.map { point =>
      if (lrModelScaled.predict(point.features) == point.label) 1 else 0
    }.sum
    val lrAccuracyScaled = lrTotalCorrectScaled / scaledTrainData.count
    println(f"lrAccuracyScaled = ${lrAccuracyScaled}%f")
    val lrPredictionsVsTrue = scaledTrainData.map { point =>
      (lrModelScaled.predict(point.features), point.label)
    }
    val scaledMetrics = new MulticlassMetrics(lrPredictionsVsTrue)
    println(f"Accuracy: ${scaledMetrics.accuracy * 100.0}%2.4f%%, WeightedPrecision: ${scaledMetrics.weightedPrecision * 100.0}%2.4f%%, " +
      f"WeightedRecall: ${scaledMetrics.weightedRecall * 100.0}%2.4f%%, WeightedFMeasure: ${scaledMetrics.weightedFMeasure * 100.0}%2.4f%%")

    // 迭代次数调优
    val iterResults = Seq(1, 5, 10, 20, 30, 40, 50).map { param =>
      val model = trainWithParams(scaledTrainData, 0.0, param, new SimpleUpdater)
      createMetrics(s"$param iterations", scaledTrainData, model)
    }
    iterResults.foreach { case (param, accuracy) => println(f"$param, Accuracy = ${accuracy * 100}%2.2f%%") }

    // 迭代次数
    val numIterations = 10

    // L1正则化调优
    val l1RegResults = Seq(0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
      val model = trainWithParams(scaledTrainData, param, numIterations, new L1Updater)
      createMetrics(s"$param L1 regularization parameter", scaledTrainData, model)
    }
    l1RegResults.foreach { case (param, accuracy) => println(f"$param, Accuracy = ${accuracy * 100}%2.2f%%") }

    // L2正则化调优
    val l2RegResults = Seq(0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
      val model = trainWithParams(scaledTrainData, param, numIterations, new SquaredL2Updater)
      createMetrics(s"$param L2 regularization parameter", scaledTrainData, model)
    }
    l2RegResults.foreach { case (param, accuracy) => println(f"$param, Accuracy = ${accuracy * 100}%2.2f%%") }

    // 交叉验证
    val iterResultsCrossValidation = Seq(1, 5, 10, 20, 30, 40, 50).map { param =>
      // 使用训练集训练逻辑回归模型
      val model = trainWithParams(scaledTrainData, 0.0, param, new SquaredL2Updater)
      // 在测试集上计算模型相关的Accuracy
      createMetrics(s"$param iterations", scaledTestData, model)
    }
    iterResultsCrossValidation.foreach { case (param, accuracy) =>
      println(f"$param, Accuracy = ${accuracy * 100}%2.6f%%")
    }

    sc.stop()
  }

  /**
    * 根据参数进行逻辑回归模型训练
    *
    * @param input
    * @param regParam
    * @param numIterations
    * @param updater
    * @return
    */
  def trainWithParams(input: RDD[LabeledPoint], regParam: Double, numIterations: Int, updater: Updater) = {
    val lr = new LogisticRegressionWithLBFGS
    lr.setNumClasses(10)
    lr.optimizer.setNumIterations(numIterations).setUpdater(updater).setRegParam(regParam)
    lr.run(input)
  }

  /**
    * 根据输入数据和模型计算模型的正确率
    *
    * @param label
    * @param data
    * @param model
    * @return
    */
  def createMetrics(label: String, data: RDD[LabeledPoint], model: ClassificationModel) = {
    val scoreAndLabels = data.map { point =>
      (model.predict(point.features), point.label)
    }
    val metrics = new MulticlassMetrics(scoreAndLabels)
    (label, metrics.accuracy)
  }

}
