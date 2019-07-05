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

    // 最大树深度，用于决策树模型
    val maxDepth = 30
    val maxBins = 20
    // 
    val numClasses = 10
    
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    val categoricalFeaturesInfo = Map[Int, Int]()

    // 训练决策树模型
    // val dtModel = DecisionTree.train(trainData, Algo.Classification, Entropy, maxDepth)
    val dtModel = DecisionTree.trainClassifier(trainData, numClasses, categoricalFeaturesInfo, "entropy", maxDepth, maxBins)

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
      if (dtModel.predict(point.features) == point.label) 1 else 0
    }.sum
    val dtAccuracy = dtTotalCorrect / trainData.count
    println(f"dtAccuracy = ${dtAccuracy * 100.0}%2.4f%%")

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
    
    // DecisionTree currently only supports maxDepth <= 30
    val treeDepthRange = new Range(5, 31, 5)

    // 使用Entropy不纯度进行最大深度调优
    println("train DecisionTree Model with Entropy")
    val dtResultsEntropy = treeDepthRange.map { param =>
      val model = trainWithParams(trainData, param, maxBins, "entropy")
      val scoreAndLabels = trainData.map { point =>
        (model.predict(point.features), point.label)
      }
      val metrics = new MulticlassMetrics(scoreAndLabels)
      (s"$param tree depth", metrics.accuracy)
    }
    dtResultsEntropy.foreach { case (param, accuracy) =>
      println(f"$param, Accuracy = ${accuracy * 100}%2.2f%%")
    }

    // 使用Gini不纯度进行最大深度调优
    println("train DecisionTree Model with Gini")
    val dtResultsGini = treeDepthRange.map { param =>
      val model = trainWithParams(trainData, param, maxBins, "gini")
      val scoreAndLabels = trainData.map { point =>
        (model.predict(point.features), point.label)
      }
      val metrics = new MulticlassMetrics(scoreAndLabels)
      (s"$param tree depth", metrics.accuracy)
    }
    dtResultsGini.foreach { case (param, accuracy) =>
      println(f"$param, Accuracy = ${accuracy * 100}%2.2f%%")
    }

    // 交叉验证
    println("train DecisionTree Model with Entropy")
    val dtResultsDepthCV = treeDepthRange.map { param =>
      val model = trainWithParams(trainData, param, maxBins, "entropy")
      val scoreAndLabels = testData.map { point =>
        (model.predict(point.features), point.label)
      }
      val metrics = new MulticlassMetrics(scoreAndLabels)
      (s"$param tree depth", metrics.accuracy)
    }
    dtResultsDepthCV.foreach { case (param, accuracy) =>
      println(f"$param, Accuracy = ${accuracy * 100}%2.2f%%")
    }
    
    val treeBinsRange = new Range(16, 65, 4)
    val dtResultsBinsCV = treeBinsRange.map { param =>
      val model = trainWithParams(trainData, maxDepth, param, "entropy")
      val scoreAndLabels = testData.map { point =>
        (model.predict(point.features), point.label)
      }
      val metrics = new MulticlassMetrics(scoreAndLabels)
      (s"$param tree bins", metrics.accuracy)
    }
    dtResultsBinsCV.foreach { case (param, accuracy) =>
      println(f"$param, Accuracy = ${accuracy * 100}%2.2f%%")
    }
    


    sc.stop()
  }

  /**
    * 根据参数进行决策树模型训练
    *
    * @param input
    * @param maxDepth
    * @param maxBins
    * @param impurity
    * @return
    */
  def trainWithParams(input: RDD[LabeledPoint], maxDepth: Int, maxBins: Int, impurity: String) = {
    // DecisionTree.train(input, Algo.Classification, impurity, maxDepth)
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    val categoricalFeaturesInfo = Map[Int, Int]()
    DecisionTree.trainClassifier(input, 10, categoricalFeaturesInfo, impurity, maxDepth, maxBins)
  }

}
