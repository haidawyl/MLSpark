package org.hdwyl.mnist

import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.{Entropy, Gini, Impurity}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by wangyanl on 2019/6/27.
  */
object SparkDT {

  def main(args: Array[String]) {
    val conf = new SparkConf()
    val sc = new SparkContext(conf)

    // 训练数据集
    val trainFeatureFile = "hdfs://PATH/mnist/train-images.idx3-ubyte"
    val trainLabelFile = "hdfs://PATH/mnist/train-labels.idx1-ubyte"

    val trainImagesAsMatrices = MnistHdfsReader.loadFeature(sc, trainFeatureFile)
    val trainLabelsAsInts = MnistHdfsReader.loadLabels(sc, trainLabelFile)

    val trainData = sc.parallelize(trainImagesAsMatrices).zipWithIndex.map { case (image, index) =>
      val features = image.toArray
      val label = trainLabelsAsInts.toList(index.toInt)
      LabeledPoint(label, Vectors.dense(features))
    }

    // 测试数据集
    val testFeatureFile = "hdfs://PATH/mnist/t10k-images.idx3-ubyte"
    val testLabelFile = "hdfs://PATH/mnist/t10k-labels.idx1-ubyte"

    val testImagesAsMatrices = MnistHdfsReader.loadFeature(sc, testFeatureFile)
    val testLabelsAsInts = MnistHdfsReader.loadLabels(sc, testLabelFile)

    val testData = sc.parallelize(testImagesAsMatrices).zipWithIndex.map { case (image, index) =>
      val features = image.toArray
      val label = testLabelsAsInts.toList(index.toInt)
      LabeledPoint(label, Vectors.dense(features))
    }

    // 最大树深度，用于决策树模型
    val maxTreeDepth = 5

    // 训练决策树模型
    val dtModel = DecisionTree.train(trainData, Algo.Classification, Entropy, maxTreeDepth)

    // 使用模型对单个数据进行预测
    val dataPoint = trainData.first()
    val prediction = dtModel.predict(dataPoint.features)
    println("prediction = %f".format(prediction))
    // 数据的真实标签
    val trueLabel = dataPoint.label
    println("trueLabel = %f".format(trueLabel))

    val K = 10
    // 使用模型对整体数据进行预测
    val predictions = dtModel.predict(trainData.map(lp => lp.features))
    println("predictions:")
    predictions.take(K).foreach(println)
    // 对应数据的真实标签
    println("trueLabels:")
    trainData.map(lp => lp.label).take(K).foreach(println)

    // 计算模型的正确率，决策树模型的预测阈值需要明确给出
    val dtTotalCorrect = trainData.map { point =>
      val score = dtModel.predict(point.features)
      val predicted = if (score > 0.5) 1 else 0
      if (predicted == point.label) 1 else 0
    }.sum
    val dtAccuracy = dtTotalCorrect / trainData.count
    println(f"dtAccuracy = ${dtAccuracy}%f")

    // 评估模型
    val metrics = Seq(dtModel).map { model =>
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

    // 决策树模型不需要进行数据标准化

    // 决策树模型有两种不纯度度量方式：Gini或者Entropy。

    // 使用Entropy不纯度进行最大深度调优
    println("train DecisionTree Model with Entropy")
    val dtResultsEntropy = Seq(1, 2, 3, 4, 5, 10, 20).map { param =>
      val model = trainWithParams(trainData, param, Entropy)
      val scoreAndLabels = trainData.map { point =>
        val score = model.predict(point.features)
        (if (score > 0.5) 1.0 else 0.0, point.label)
      }
      val metrics = new MulticlassMetrics(scoreAndLabels)
      (s"$param tree depth", metrics.accuracy)
    }
    dtResultsEntropy.foreach { case (param, accuracy) =>
      println(f"$param, Accuracy = ${accuracy * 100}%2.2f%%")
    }

    // 使用Gini不纯度进行最大深度调优
    println("train DecisionTree Model with Gini")
    val dtResultsGini = Seq(1, 2, 3, 4, 5, 10, 20).map { param =>
      val model = trainWithParams(trainData, param, Gini)
      val scoreAndLabels = trainData.map { point =>
        val score = model.predict(point.features)
        (if (score > 0.5) 1.0 else 0.0, point.label)
      }
      val metrics = new MulticlassMetrics(scoreAndLabels)
      (s"$param tree depth", metrics.accuracy)
    }
    dtResultsGini.foreach { case (param, accuracy) =>
      println(f"$param, Accuracy = ${accuracy * 100}%2.2f%%")
    }

    // 交叉验证
    println("train DecisionTree Model with Entropy")
    val dtResultsEntropyCrossValidation = Seq(1, 2, 3, 4, 5, 10, 20).map { param =>
      val model = trainWithParams(trainData, param, Entropy)
      val scoreAndLabels = testData.map { point =>
        val score = model.predict(point.features)
        (if (score > 0.5) 1.0 else 0.0, point.label)
      }
      val metrics = new MulticlassMetrics(scoreAndLabels)
      (s"$param tree depth", metrics.accuracy)
    }
    dtResultsEntropyCrossValidation.foreach { case (param, accuracy) =>
      println(f"$param, Accuracy = ${accuracy * 100}%2.2f%%")
    }

    sc.stop()
  }

  /**
    * 根据参数进行决策树模型训练
    *
    * @param input
    * @param maxDepth
    * @param impurity
    * @return
    */
  def trainWithParams(input: RDD[LabeledPoint], maxDepth: Int, impurity: Impurity) = {
    DecisionTree.train(input, Algo.Classification, impurity, maxDepth)
  }

}
