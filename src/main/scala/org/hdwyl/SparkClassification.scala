package org.hdwyl

import org.apache.spark.mllib.classification.{ClassificationModel, LogisticRegressionWithLBFGS, NaiveBayes, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.optimization.{L1Updater, SimpleUpdater, SquaredL2Updater, Updater}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.{Entropy, Gini, Impurity}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by wangyanl on 2019/6/22.
  */
object SparkClassification {

  def main(args: Array[String]) {
    val conf = new SparkConf()
    val sc = new SparkContext(conf)

    val rawData = sc.textFile("hdfs://PATH/stumbleupon/train_noheader.tsv")
    val records = rawData.map(line => line.split("\t"))
    println(records.first().mkString(" | "))

    val data = records.map { r =>
      // 去掉"
      val trimmed = r.map(_.replaceAll("\"", ""))
      // 标记变量
      val label = trimmed(r.size - 1).toInt
      // 特征矩阵，处理缺失数据(?变为0.0)
      val features = trimmed.slice(4, r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble)
      LabeledPoint(label, Vectors.dense(features))
    }
    // 缓存数据
    data.cache()
    val numData = data.count()
    // numData = 7395
    println("numData = %d".format(numData))

    val nbData = records.map { r =>
      // 去掉"
      val trimmed = r.map(_.replaceAll("\"", ""))
      // 标记变量
      val label = trimmed(r.size - 1).toInt
      // 特征矩阵，处理缺失数据(?变为0.0，负数变为0.0)
      val features = trimmed.slice(4, r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble)
        .map(d => if (d < 0) 0.0 else d)
      LabeledPoint(label, Vectors.dense(features))
    }
    // 缓存数据
    nbData.cache()

    // 迭代次数，用户逻辑回归和SVM模型
    val numIterations = 10
    // 最大树深度，用于决策树模型
    val maxTreeDepth = 5

    // 训练逻辑回归模型
    // val lrModel = LogisticRegressionWithSGD.train(data, numIterations)
    val lrModel = new LogisticRegressionWithLBFGS().setNumClasses(2).run(data)

    // 训练SVM模型
    val svmModel = SVMWithSGD.train(data, numIterations)

    // 训练朴素贝叶斯模型
    val nbModel = NaiveBayes.train(nbData)

    // 训练决策树模型
    val dtModel = DecisionTree.train(data, Algo.Classification, Entropy, maxTreeDepth)

    val dataPoint = data.first()
    val prediction = lrModel.predict(dataPoint.features)
    println("prediction = %f".format(prediction))
    val trueLabel = dataPoint.label
    println("trueLabel = %f".format(trueLabel))

    val K = 10
    val predictions = lrModel.predict(data.map(lp => lp.features))
    println("predictions:")
    predictions.take(K).foreach(println)
    println("trueLabels:")
    data.map(lp => lp.label).take(K).foreach(println)

    // 计算逻辑回归模型的正确率
    val lrTotalCorrect = data.map { point =>
      if (lrModel.predict(point.features) == point.label) 1 else 0
    }.sum()
    val lrAccuracy = lrTotalCorrect / numData
    println("lrAccuracy = %f".format(lrAccuracy))

    // 计算SVM模型的正确率
    val svmTotalCorrect = data.map { point =>
      if (svmModel.predict(point.features) == point.label) 1 else 0
    }.sum()
    val svmAccuracy = svmTotalCorrect / numData
    println("svmAccuracy = %f".format(svmAccuracy))

    // 计算朴素贝叶斯模型的正确率
    val nbTotalCorrect = data.map { point =>
      if (nbModel.predict(point.features) == point.label) 1 else 0
    }.sum()
    val nbAccuracy = nbTotalCorrect / numData
    println("nbAccuracy = %f".format(nbAccuracy))

    // 计算决策树模型的正确率
    val dtTotalCorrect = nbData.map { point =>
      val score = dtModel.predict(point.features)
      val predicted = if (score > 0.5) 1 else 0
      if (predicted == point.label) 1 else 0
    }.sum()
    val dtAccuracy = dtTotalCorrect / numData
    println("dtAccuracy = %f".format(dtAccuracy))

    // 在信息检索中，准确率通常用于评价结果的质量，而召回率用来评价结果的完整性。
    // 在二分类问题中，准确率定义为真阳性的数目除以真阳性和假阳性的总数，其中真阳性是指被正确预测的类别为1的样本，
    // 假阳性是错误预测为类别1的样本。如果每个被分类器预测为类别1的样本确实属于类别1，那准确率达到100%。
    // 召回率定义为真阳性的数目除以真阳性和假阴性的和，其中假阴性是类别为1却被预测为0的样本。
    // 如果任何一个类型为1的样本没有被错误预测为类别0（即没有假阴性），那召回率达到100%。
    // 通常，准确率和召回率是负相关的，高准确率常常对应低召回率，反之亦然。

    // 真阳性率（TPR）是真阳性的样本数除以真阳性和假阴性的样本数之和。
    // 换句话说，TPR是真阳性数目占所有正样本的比例。这和召回率类似，通常也称为敏感度。
    // 假阳性率（FPR）是假阳性的样本数除以假阳性和真阴性的样本数之和。
    // 换句话说，FPR是假阳性样本数占所有负样本总数的比例。

    // 计算逻辑回归模型和SVM模型的PR和ROC
    val metrics = Seq(lrModel, svmModel).map { model =>
      val scoreAndLabels = data.map { point =>
        (model.predict(point.features), point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (model.getClass.getSimpleName, metrics.areaUnderPR, metrics.areaUnderROC)
    }

    // 计算朴素贝叶斯模型的PR和ROC
    val nbMetrics = Seq(nbModel).map { model =>
      val scoreAndLabels = nbData.map { point =>
        val score = model.predict(point.features)
        (if (score > 0.5) 1.0 else 0.0, point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (model.getClass.getSimpleName, metrics.areaUnderPR, metrics.areaUnderROC)
    }

    // 计算决策树模型的PR和ROC
    val dtMetrics = Seq(dtModel).map { model =>
      val scoreAndLabels = data.map { point =>
        val score = dtModel.predict(point.features)
        (if (score > 0.5) 1.0 else 0.0, point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (model.getClass.getSimpleName, metrics.areaUnderPR, metrics.areaUnderROC)
    }

    val allMetrics = metrics ++ nbMetrics ++ dtMetrics
    allMetrics.foreach { case (m, pr, roc) =>
      println(f"$m, Area under PR: ${pr * 100.0}%2.4f%%, Area under ROC: ${roc * 100.0}%2.4f%%")
    }

    val vectors = data.map(lp => lp.features)
    val matrix = new RowMatrix(vectors)
    val matrixSummary = matrix.computeColumnSummaryStatistics()
    println(matrixSummary.mean)
    println(matrixSummary.min)
    println(matrixSummary.max)
    println(matrixSummary.variance)
    println(matrixSummary.numNonzeros)

    val scaler = new StandardScaler(withMean = true, withStd = true).fit(vectors)
    val scaledData = data.map(lp => LabeledPoint(lp.label, scaler.transform(lp.features)))
    println("标准化之前的特征:")
    println(data.first.features)
    println("标准化之后的特征:")
    println(scaledData.first.features)

    println((0.789131 - 0.41225805299526636) / math.sqrt(0.1097424416755897))

    // val lrModelScaled = LogisticRegressionWithSGD.train(scaledData, numIterations)
    val lrModelScaled = new LogisticRegressionWithLBFGS().setNumClasses(2).run(scaledData)
    val lrTotalCorrectScaled = scaledData.map { point =>
      if (lrModelScaled.predict(point.features) == point.label) 1 else 0
    }.sum()
    val lrAccuracyScaled = lrTotalCorrectScaled / numData
    val lrPredictionsVsTrue = scaledData.map { point =>
      (lrModelScaled.predict(point.features), point.label)
    }
    val lrMetricsScaled = new BinaryClassificationMetrics(lrPredictionsVsTrue)
    val lrPr = lrMetricsScaled.areaUnderPR()
    val lrRoc = lrMetricsScaled.areaUnderROC()
    // LogisticRegressionModel
    // Accuracy: 62.0419%
    // Area under PR: 72.7254%
    // Area under ROC: 61.9663%
    println(f"${lrModelScaled.getClass.getSimpleName}\n" +
      f"Accuracy: ${lrAccuracyScaled * 100}%2.4f%%\n" +
      f"Area under PR: ${lrPr * 100.0}%2.4f%%\n" +
      f"Area under ROC: ${lrRoc * 100.0}%2.4f%%")

    val categories = records.map(r => r(3)).distinct().collect().zipWithIndex.toMap
    val numCategories = categories.size
    println("categories:")
    println(categories)
    println(s"numCategories = ${numCategories}")

    val dataCategories = records.map { r =>
      val trimmed = r.map(_.replaceAll("\"", ""))
      val label = trimmed(r.size - 1).toInt
      val categoryIdx = categories(r(3))
      val categoryFeatures = Array.ofDim[Double](numCategories)
      categoryFeatures(categoryIdx) = 1.0
      val otherFeatures = trimmed.slice(4, r.size - 1).map(d =>
        if (d == "?") 0.0 else d.toDouble
      )
      val features = categoryFeatures ++ otherFeatures
      LabeledPoint(label, Vectors.dense(features))
    }
    println(dataCategories.first())

    val scalerCats = new StandardScaler(withMean = true, withStd = true).fit(
      dataCategories.map(lp => lp.features)
    )
    val scaledDataCats = dataCategories.map(lp =>
      LabeledPoint(lp.label, scalerCats.transform(lp.features))
    )
    println("标准化之前的特征:")
    println(dataCategories.first().features)
    println("标准化之后的特征:")
    println(scaledDataCats.first().features)

    // val lrModelScaledCats = LogisticRegressionWithSGD.train(scaledDataCats,numIterations)
    val lrModelScaledCats = new LogisticRegressionWithLBFGS().setNumClasses(2).run(scaledDataCats)
    val lrTotalCorrectScaledCats = scaledDataCats.map { point =>
      if (lrModelScaledCats.predict(point.features) == point.label) 1 else 0
    }.sum()
    val lrAccuracyScaledCats = lrTotalCorrectScaledCats / numData
    val lrPredictionsVsTrueCats = scaledDataCats.map { point =>
      (lrModelScaledCats.predict(point.features), point.label)
    }
    val lrMetricsScaledCats = new BinaryClassificationMetrics(lrPredictionsVsTrueCats)
    val lrPrCats = lrMetricsScaledCats.areaUnderPR()
    val lrRocCats = lrMetricsScaledCats.areaUnderROC()

    // LogisticRegressionModel
    // Accuracy: 66.5720%
    // Area under PR: 75.7964%
    // Area under ROC: 66.5483%
    println(f"${lrModelScaledCats.getClass.getSimpleName}\n" +
      f"Accuracy: ${lrAccuracyScaledCats * 100}%2.4f%%\n" +
      f"Area under PR: ${lrPrCats * 100.0}%2.4f%%\n" +
      f"Area under ROC: ${lrRocCats * 100.0}%2.4f%%")

    val dataNB = records.map { r =>
      val trimmed = r.map(_.replaceAll("\"", ""))
      val label = trimmed(r.size - 1).toInt
      val categoryIdx = categories(r(3))
      val categoryFeatures = Array.ofDim[Double](numCategories)
      categoryFeatures(categoryIdx) = 1.0
      LabeledPoint(label, Vectors.dense(categoryFeatures))
    }

    val nbModelCats = NaiveBayes.train(dataNB)
    val nbTotalCorrectCats = dataNB.map { point =>
      if (nbModelCats.predict(point.features) == point.label) 1 else 0
    }.sum
    val nbAccuracyCats = nbTotalCorrectCats / numData
    val nbPredictionsVsTrueCats = dataNB.map { point =>
      (nbModelCats.predict(point.features), point.label)
    }
    val nbMetricsCats = new BinaryClassificationMetrics(nbPredictionsVsTrueCats)
    val nbPrCats = nbMetricsCats.areaUnderPR
    val nbRocCats = nbMetricsCats.areaUnderROC
    // NavieBayesModel
    // Accuracy: 60.9601%
    // Area under PR: 74.0522%
    // Area under ROC: 60.5138%
    println(f"${nbMetricsCats.getClass.getSimpleName}\n" +
      f"Accuracy: ${nbAccuracyCats * 100}%2.4f%%\n" +
      f"Area under PR: ${nbPrCats * 100.0}%2.4f%%\n" +
      f"Area under ROC: ${nbRocCats * 100.0}%2.4f%%")

    scaledDataCats.cache

    val iterResults = Seq(1, 5, 10, 50).map { param =>
      val model = trainWithParams(scaledDataCats, 0.0, param, new SimpleUpdater, 1.0)
      createMetrics(s"$param iterations", scaledDataCats, model)
    }
    iterResults.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }

    val stepResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
      val model = trainWithParams(scaledDataCats, 0.0, numIterations, new SimpleUpdater, param)
      createMetrics(s"$param step size", scaledDataCats, model)
    }
    stepResults.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }

    val l1RegResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
      val model = trainWithParams(scaledDataCats, param, numIterations, new L1Updater, 1.0)
      createMetrics(s"$param L1 regularization parameter", scaledDataCats, model)
    }
    l1RegResults.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }

    val l2RegResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
      val model = trainWithParams(scaledDataCats, param, numIterations, new SquaredL2Updater, 1.0)
      createMetrics(s"$param L2 regularization parameter", scaledDataCats, model)
    }
    l2RegResults.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }

    val dtResultsEntropy = Seq(1, 2, 3, 4, 5, 10, 20).map { param =>
      val model = trainDTWithParams(data, param, Entropy)
      val scoreAndLabels = data.map { point =>
        val score = model.predict(point.features)
        (if (score > 0.5) 1.0 else 0.0, point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (s"$param tree depth", metrics.areaUnderROC)
    }
    dtResultsEntropy.foreach { case (param, auc) =>
      println(f"$param, AUC = ${auc * 100}%2.2f%%")
    }

    val dtResultsGini = Seq(1, 2, 3, 4, 5, 10, 20).map { param =>
      val model = trainDTWithParams(data, param, Gini)
      val scoreAndLabels = data.map { point =>
        val score = model.predict(point.features)
        (if (score > 0.5) 1.0 else 0.0, point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (s"$param tree depth", metrics.areaUnderROC)
    }
    dtResultsGini.foreach { case (param, auc) =>
      println(f"$param, AUC = ${auc * 100}%2.2f%%")
    }

    val nbResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
      val model = trainNBWithParams(dataNB, param)
      val scoreAndLabels = dataNB.map { point =>
        (model.predict(point.features), point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (s"$param lambda", metrics.areaUnderROC())
    }
    nbResults.foreach { case (param, auc) =>
      println(f"$param, AUC = ${auc * 100}%2.2f%%")
    }

    // 将数据集分成60%的训练集和40%的测试集（使用一个固定的随机种子123来保证每次实验能得到相同的结果）
    val trainTestSplit = scaledDataCats.randomSplit(Array(0.6, 0.4), 123)
    val train = trainTestSplit(0)
    val test = trainTestSplit(1)

    val regResultsTest = Seq(0.0, 0.001, 0.0025, 0.005, 0.01).map { param =>
      val model = trainWithParams(train, param, numIterations, new SquaredL2Updater, 1.0)
      createMetrics(s"$param L2 regularization parameter", test, model)
    }
    regResultsTest.foreach { case (param, auc) =>
      println(f"$param, AUC = ${auc * 100}%2.6f%%")
    }

    sc.stop()
  }

  def trainWithParams(input: RDD[LabeledPoint], regParam: Double, numIterations: Int, updater: Updater, stepSize: Double) = {
    // val lr = new LogisticRegressionWithSGD
    // lr.optimizer.setNumIterations(numIterations).setUpdater(updater).setRegParam(regParam).setStepSize(stepSize)
    val lr = new LogisticRegressionWithLBFGS
    lr.optimizer.setNumIterations(numIterations).setUpdater(updater).setRegParam(regParam)
    lr.run(input)
  }

  def createMetrics(label: String, data: RDD[LabeledPoint], model: ClassificationModel) = {
    val scoreAndLabels = data.map { point =>
      (model.predict(point.features), point.label)
    }
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    (label, metrics.areaUnderROC)
  }

  def trainDTWithParams(input: RDD[LabeledPoint], maxDepth: Int, impurity: Impurity) = {
    DecisionTree.train(input, Algo.Classification, impurity, maxDepth)
  }

  def trainNBWithParams(input: RDD[LabeledPoint], lambda: Double) = {
    val nb = new NaiveBayes()
    nb.setLambda(lambda)
    nb.run(input)
  }
}
