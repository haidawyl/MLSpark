package org.hdwyl.mnist

import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by wangyanl on 2019/6/27.
  */
object SparkNB {

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
      val features = image.toArray.map(d => if (d < 0) 0.0 else d)
      val label = trainLabelsAsInts.toList(index.toInt)
      LabeledPoint(label, Vectors.dense(features))
    }
    */
    val trainData = sc.parallelize(trainImagesAsMatrices zip trainLabelsAsInts).map { case (image, label) =>
      val features = image.toArray.map(d => if (d < 0) 0.0 else d)
      LabeledPoint(label, Vectors.dense(features))
    }
    trainData.cache()

    // 测试数据集
    val testFeatureFile = "hdfs://PATH/mnist/t10k-images.idx3-ubyte"
    val testLabelFile = "hdfs://PATH/mnist/t10k-labels.idx1-ubyte"

    val testImagesAsMatrices = MnistHdfsReader.loadFeature(sc, testFeatureFile)
    val testLabelsAsInts = MnistHdfsReader.loadLabels(sc, testLabelFile)

    /*
    val testData = sc.parallelize(testImagesAsMatrices).zipWithIndex.map { case (image, index) =>
      val features = image.toArray.map(d => if (d < 0) 0.0 else d)
      val label = testLabelsAsInts.toList(index.toInt)
      LabeledPoint(label, Vectors.dense(features))
    }
    */
    val testData = sc.parallelize(testImagesAsMatrices zip testLabelsAsInts).map { case (image, label) =>
      val features = image.toArray.map(d => if (d < 0) 0.0 else d)
      LabeledPoint(label, Vectors.dense(features))
    }
    testData.cache()

    // 训练朴素贝叶斯模型
    val nbModel = NaiveBayes.train(trainData)

    // 使用模型对单个数据进行预测
    val dataPoint = trainData.first()
    val prediction = nbModel.predict(dataPoint.features)
    println("prediction = %f".format(prediction))
    // 数据的真实标签
    val trueLabel = dataPoint.label
    println("trueLabel = %f".format(trueLabel))

    val K = 10
    // 使用模型对整体数据进行预测
    val predictions = nbModel.predict(trainData.map(lp => lp.features))
    println("predictions:")
    predictions.take(K).foreach(println)
    // 对应数据的真实标签
    println("trueLabels:")
    trainData.map(lp => lp.label).take(K).foreach(println)

    // 计算模型的正确率
    val nbTotalCorrect = trainData.map { point =>
      if (nbModel.predict(point.features) == point.label) 1 else 0
    }.sum
    val nbAccuracy = nbTotalCorrect / trainData.count
    println(f"nbAccuracy = ${nbAccuracy * 100.0}%2.4f%%")

    // 评估模型
    val metrics = Seq(nbModel).map { model =>
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

    // 朴素贝叶斯模型不需要进行数据标准化

    val modelType = "multinomial"

    // lamda参数在朴素贝叶斯模型中可以控制相加式平滑（additive smoothing），
    // 解决数据中某个类别和某个特征值的组合没有同时出现的问题。
    val nbResults = Seq(0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
      val model = trainWithParams(trainData, param, modelType)
      val scoreAndLabels = trainData.map { point =>
        (model.predict(point.features), point.label)
      }
      val metrics = new MulticlassMetrics(scoreAndLabels)
      (s"$param lambda", metrics.accuracy)
    }
    nbResults.foreach { case (param, accuracy) =>
      println(f"$param, Accuracy = ${accuracy * 100}%2.2f%%")
    }

    // Bernoulli naive Bayes requires 0 or 1 feature values
    /*
    val nbResultsModelType = Seq("multinomial", "bernoulli").map { param =>
      val model = trainWithParams(trainData, 0.001, param)
      val scoreAndLabels = trainData.map { point =>
        (model.predict(point.features), point.label)
      }
      val metrics = new MulticlassMetrics(scoreAndLabels)
      (s"$param model type", metrics.accuracy)
    }
    nbResultsModelType.foreach { case (param, accuracy) =>
      println(f"$param, Accuracy = ${accuracy * 100}%2.2f%%")
    }
    */

    // 交叉验证
    println("交叉验证")
    val nbResultsCrossValidation = Seq(0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
      val model = trainWithParams(trainData, param, modelType)
      val scoreAndLabels = testData.map { point =>
        (model.predict(point.features), point.label)
      }
      val metrics = new MulticlassMetrics(scoreAndLabels)
      (s"$param lambda", metrics.accuracy)
    }
    nbResultsCrossValidation.foreach { case (param, accuracy) =>
      println(f"$param, Accuracy = ${accuracy * 100}%2.2f%%")
    }

    sc.stop()
  }

  /**
    * 根据参数进行朴素贝叶斯模型训练
    *
    * @param input
    * @param lambda
    * @param modelType
    * @return
    */
  def trainWithParams(input: RDD[LabeledPoint], lambda: Double, modelType: String) = {
    val nb = new NaiveBayes()
    // bernoulli or multinomial
    nb.setModelType(modelType)
    nb.setLambda(lambda)
    nb.run(input)
  }

}
