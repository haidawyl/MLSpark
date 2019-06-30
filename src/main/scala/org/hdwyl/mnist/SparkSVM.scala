package org.hdwyl.mnist

import org.apache.spark.mllib.classification.{ClassificationModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.optimization.Updater
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by wangyanl on 2019/6/27.
  */
object SparkSVM {

  def main(args: Array[String]) {
    // TODO 不支持多分类

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

    // 迭代次数，用于逻辑回归和SVM模型
    val numIterations = 10

    // 训练SVM模型
    val svmModel = SVMWithSGD.train(trainData, numIterations)

    // 使用模型对单个数据进行预测
    val dataPoint = trainData.first()
    val prediction = svmModel.predict(dataPoint.features)
    println("prediction = %f".format(prediction))
    // 数据的真实标签
    val trueLabel = dataPoint.label
    println("trueLabel = %f".format(trueLabel))

    val K = 10
    // 使用模型对整体数据进行预测
    val predictions = svmModel.predict(trainData.map(lp => lp.features))
    println("predictions:")
    predictions.take(K).foreach(println)
    // 对应数据的真实标签
    println("trueLabels:")
    trainData.map(lp => lp.label).take(K).foreach(println)

    // 计算模型的正确率
    val svmTotalCorrect = trainData.map { point =>
      if (svmModel.predict(point.features) == point.label) 1 else 0
    }.sum
    val svmAccuracy = svmTotalCorrect / trainData.count
    println(f"svmAccuracy = ${svmAccuracy}%f")

    // 评估模型
    val metrics = Seq(svmModel).map { model =>
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
    println(matrixSummary.mean)
    // 输出矩阵每列的最小值
    println(matrixSummary.min)
    // 输出矩阵每列的最大值
    println(matrixSummary.max)
    // 输出矩阵每列的方差
    println(matrixSummary.variance)
    // 输出矩阵每列中非0项的数目
    println(matrixSummary.numNonzeros)

    // 使用Spark的StandardScaler中的方法对每个特征进行标准化，使得每个特征是0均值和单位标准差。
    // withMean: 表示是否从数据中减去均值
    // withStd: 表示是否应用标准差缩放
    val scaler = new StandardScaler(withMean = true, withStd = true).fit(vectors)
    // 输入向量传到转换函数，并且返回归一化的向量。
    val scaledData = trainData.map(lp => LabeledPoint(lp.label, scaler.transform(lp.features)))
    println("标准化之前的特征:")
    println(trainData.first.features)
    println("标准化之后的特征:")
    println(scaledData.first.features)

    // 使用标准化之前的第一个特征减去该列的均值，然后除以该列的标准差（方差的平方根），结果等于标准化之后的第一个特征值
    // TODO 替换为实际值
    println((0.789131 - 0.41225805299526636) / math.sqrt(0.1097424416755897))

    // 使用标准化之后的数据重新训练模型
    val svmModelScaled = SVMWithSGD.train(scaledData, numIterations)
    val svmTotalCorrectScaled = scaledData.map { point =>
      if (svmModelScaled.predict(point.features) == point.label) 1 else 0
    }.sum
    val svmAccuracyScaled = svmTotalCorrectScaled / scaledData.count
    println(f"svmAccuracyScaled = ${svmAccuracyScaled}%f")
    val svmPredictionsVsTrue = scaledData.map { point =>
      (svmModelScaled.predict(point.features), point.label)
    }
    val scaledMetrics = new MulticlassMetrics(svmPredictionsVsTrue)
    println(f"Accuracy: ${scaledMetrics.accuracy * 100.0}%2.4f%%, WeightedPrecision: ${scaledMetrics.weightedPrecision * 100.0}%2.4f%%, " +
      f"WeightedRecall: ${scaledMetrics.weightedRecall * 100.0}%2.4f%%, WeightedFMeasure: ${scaledMetrics.weightedFMeasure * 100.0}%2.4f%%")

    sc.stop()
  }

  /**
    * 根据参数进行SVM模型训练
    *
    * @param input
    * @param regParam
    * @param numIterations
    * @param updater
    * @param stepSize
    * @return
    */
  def trainWithParams(input: RDD[LabeledPoint], regParam: Double, numIterations: Int, updater: Updater, stepSize: Double) = {
    val svm = new SVMWithSGD()
    svm.optimizer.setNumIterations(numIterations).setUpdater(updater).setRegParam(regParam).setStepSize(stepSize)
    svm.run(input)
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
