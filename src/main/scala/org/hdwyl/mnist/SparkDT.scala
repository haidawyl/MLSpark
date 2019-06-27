package org.hdwyl.mnist

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by wangyanl on 2019/6/27.
  */
object SparkDT {

  def main(args: Array[String]) {
    val conf = new SparkConf()
    val sc = new SparkContext(conf)

    val featureFile = "hdfs://PATH/mnist/train-images.idx3-ubyte"
    val labelFile = "hdfs://PATH/mnist/train-labels.idx1-ubyte"

    val imagesAsMatrices = MnistHdfsReader.loadFeature(sc, featureFile)
    val labelsAsInts = MnistHdfsReader.loadLabels(sc, labelFile)

    val data = sc.parallelize(imagesAsMatrices).zipWithIndex.map { case (image, index) =>
      val features = image.toArray
      val label = labelsAsInts.toList(index.toInt)
      LabeledPoint(label, Vectors.dense(features))
    }

    // 最大树深度，用于决策树模型
    val maxTreeDepth = 5

    // 训练决策树模型
    val dtModel = DecisionTree.train(data, Algo.Classification, Entropy, maxTreeDepth)

    // 使用模型对单个数据进行预测
    val dataPoint = data.first()
    val prediction = dtModel.predict(dataPoint.features)
    println("prediction = %f".format(prediction))
    // 数据的真实标签
    val trueLabel = dataPoint.label
    println("trueLabel = %f".format(trueLabel))

    val K = 10
    // 使用模型对整体数据进行预测
    val predictions = dtModel.predict(data.map(lp => lp.features))
    println("predictions:")
    predictions.take(K).foreach(println)
    // 对应数据的真实标签
    println("trueLabels:")
    data.map(lp => lp.label).take(K).foreach(println)

    sc.stop()
  }

}
