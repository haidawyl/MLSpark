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
      // 提取目标变量，即最后一列
      val label = trimmed(r.size - 1).toInt
      // 提取特征向量，同时处理缺失数据(?变为0.0)
      val features = trimmed.slice(4, r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble)
      // 分类模型通过LabeledPoint对象操作，其中封装了目标变量（标签）和特征向量
      LabeledPoint(label, Vectors.dense(features))
    }
    // 缓存数据
    data.cache()
    val numData = data.count()
    // numData = 7395
    println("numData = %d".format(numData))

    // 朴素贝叶斯模型的训练数据，特征值必须非负
    val nbData = records.map { r =>
      // 去掉"
      val trimmed = r.map(_.replaceAll("\"", ""))
      // 提取目标变量，即最后一列
      val label = trimmed(r.size - 1).toInt
      // 提取特征向量，同时处理缺失数据(?变为0.0，负数变为0.0)
      val features = trimmed.slice(4, r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble)
        .map(d => if (d < 0) 0.0 else d)
      // 分类模型通过LabeledPoint对象操作，其中封装了目标变量（标签）和特征向量
      LabeledPoint(label, Vectors.dense(features))
    }
    // 缓存数据
    nbData.cache()

    // 迭代次数，用于逻辑回归和SVM模型
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

    // 使用模型对单个数据进行预测
    val dataPoint = data.first()
    val prediction = lrModel.predict(dataPoint.features)
    println("prediction = %f".format(prediction))
    // 数据的真实标签
    val trueLabel = dataPoint.label
    println("trueLabel = %f".format(trueLabel))

    val K = 10
    // 使用模型对整体数据进行预测
    val predictions = lrModel.predict(data.map(lp => lp.features))
    println("predictions:")
    predictions.take(K).foreach(println)
    // 对应数据的真实标签
    println("trueLabels:")
    data.map(lp => lp.label).take(K).foreach(println)

    // 正确率等于训练样本中被正确分类的样本数目除以总样本数
    // 错误率等于训练样本中被错误分类的样本数目除以总样本数
    // 通过对输入特征进行预测并将预测值与实际标签进行比较，计算出模型在训练数据上的正确率
    // 将对正确分类的样本数目求和并除以样本总数，得到平均分类正确率

    // 计算逻辑回归模型的正确率
    val lrTotalCorrect = data.map { point =>
      if (lrModel.predict(point.features) == point.label) 1 else 0
    }.sum
    val lrAccuracy = lrTotalCorrect / numData
    println("lrAccuracy = %f".format(lrAccuracy))

    // 计算SVM模型的正确率
    val svmTotalCorrect = data.map { point =>
      if (svmModel.predict(point.features) == point.label) 1 else 0
    }.sum
    val svmAccuracy = svmTotalCorrect / numData
    println("svmAccuracy = %f".format(svmAccuracy))

    // 计算朴素贝叶斯模型的正确率
    val nbTotalCorrect = data.map { point =>
      if (nbModel.predict(point.features) == point.label) 1 else 0
    }.sum()
    val nbAccuracy = nbTotalCorrect / numData
    println("nbAccuracy = %f".format(nbAccuracy))

    // 计算决策树模型的正确率，决策树模型的预测阈值需要明确给出
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
        val score = model.predict(point.features)
        (if (score > 0.5) 1.0 else 0.0, point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (model.getClass.getSimpleName, metrics.areaUnderPR, metrics.areaUnderROC)
    }

    val allMetrics = metrics ++ nbMetrics ++ dtMetrics
    allMetrics.foreach { case (m, pr, roc) =>
      println(f"$m, Area under PR: ${pr * 100.0}%2.4f%%, Area under ROC: ${roc * 100.0}%2.4f%%")
    }

    // 将特征向量用RowMatrix类表示成MLlib中的分布矩阵。
    // RowMatrix是一个由向量组成的RDD，其中每个向量是分布矩阵的一行。
    val vectors = data.map(lp => lp.features)
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

    // 数据在原始形式下并不符合标准的高斯分布。为使数据更符合模型的假设，可以对每个特征进行标准化，
    // 使得每个特征是0均值和单位标准差。具体做法是对每个特征值减去列的均值，然后除以列的标准差以进行缩放。
    // 实际上，我们可以对数据集中每个特征向量，与均值向量按项依次做减法，然后依次按项除以特征的标准差向量。
    // 标准差向量可以由方差向量的每项求平方根得到。
    // 实际上，使用Spark的StandardScaler中的方法可以方便地完成这些操作。
    // withMean: 表示是否从数据中减去均值，设置为false将不会改变特征矩阵的稀疏性
    // withStd: 表示是否应用标准差缩放
    val scaler = new StandardScaler(withMean = true, withStd = true).fit(vectors)
    // 输入向量传到转换函数，并且返回归一化的向量。
    val scaledData = data.map(lp => LabeledPoint(lp.label, scaler.transform(lp.features)))
    println("标准化之前的特征:")
    println(data.first.features)
    println("标准化之后的特征:")
    println(scaledData.first.features)

    // 使用标准化之前的第一个特征减去该列的均值，然后除以该列的标准差（方差的平方根），结果等于标准化之后的第一个特征值
    println((0.789131 - 0.41225805299526636) / math.sqrt(0.1097424416755897))

    // 使用标准化之后的数据重新训练逻辑回归模型
    // val lrModelScaled = LogisticRegressionWithSGD.train(scaledData, numIterations)
    val lrModelScaled = new LogisticRegressionWithLBFGS().setNumClasses(2).run(scaledData)
    val lrTotalCorrectScaled = scaledData.map { point =>
      if (lrModelScaled.predict(point.features) == point.label) 1 else 0
    }.sum
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

    // 使用标准化之后的数据重新训练SVM模型
    val svmModelScaled = SVMWithSGD.train(scaledData, numIterations)
    val svmTotalCorrectScaled = scaledData.map { point =>
      if (svmModelScaled.predict(point.features) == point.label) 1 else 0
    }.sum
    val svmAccuracyScaled = svmTotalCorrectScaled / numData
    val svmPredictionsVsTrue = scaledData.map { point =>
      (svmModelScaled.predict(point.features), point.label)
    }
    val svmMetricsScaled = new BinaryClassificationMetrics(svmPredictionsVsTrue)
    val svmPr = svmMetricsScaled.areaUnderPR()
    val svmRoc = svmMetricsScaled.areaUnderROC()
    // SVMModel
    // TODO 替换为实际值
    // Accuracy: 62.0419%
    // Area under PR: 72.7254%
    // Area under ROC: 61.9663%
    println(f"${svmModelScaled.getClass.getSimpleName}\n" +
      f"Accuracy: ${svmAccuracyScaled * 100}%2.4f%%\n" +
      f"Area under PR: ${svmPr * 100.0}%2.4f%%\n" +
      f"Area under ROC: ${svmRoc * 100.0}%2.4f%%")

    // 对类别特征做1-of-k编码
    val categories = records.map(r => r(3)).distinct().collect().zipWithIndex.toMap
    // 类别数量
    val numCategories = categories.size
    println("categories:")
    println(categories)
    println(s"numCategories = ${numCategories}")

    val dataWithCategories = records.map { r =>
      // 去掉"
      val trimmed = r.map(_.replaceAll("\"", ""))
      // 提取目标变量，即最后一列
      val label = trimmed(r.size - 1).toInt
      // 提取类别的索引值
      val categoryIdx = categories(r(3))
      // 创建维度为类别数量的特征向量，默认值都是0.0
      val categoryFeatures = Array.ofDim[Double](numCategories)
      // 类别特征向量中指定索引的值变为1.0
      categoryFeatures(categoryIdx) = 1.0
      // 提取其它特征向量，同时处理缺失数据(?变为0.0)
      val otherFeatures = trimmed.slice(4, r.size - 1).map(d =>
        if (d == "?") 0.0 else d.toDouble
      )
      // 全部特征向量
      val features = categoryFeatures ++ otherFeatures
      // 分类模型通过LabeledPoint对象操作，其中封装了目标变量（标签）和特征向量
      LabeledPoint(label, Vectors.dense(features))
    }
    println(dataWithCategories.first())

    // 对原始数据进行标准化转换
    val scalerCats = new StandardScaler(withMean = true, withStd = true).fit(
      dataWithCategories.map(lp => lp.features)
    )
    val scaledDataCats = dataWithCategories.map(lp =>
      LabeledPoint(lp.label, scalerCats.transform(lp.features))
    )
    scaledDataCats.cache

    println("标准化之前的特征:")
    println(dataWithCategories.first().features)
    println("标准化之后的特征:")
    println(scaledDataCats.first().features)

    // 使用扩展后的特征矩阵训练逻辑回归模型
    // val lrModelScaledCats = LogisticRegressionWithSGD.train(scaledDataCats,numIterations)
    val lrModelScaledCats = new LogisticRegressionWithLBFGS().setNumClasses(2).run(scaledDataCats)
    val lrTotalCorrectScaledCats = scaledDataCats.map { point =>
      if (lrModelScaledCats.predict(point.features) == point.label) 1 else 0
    }.sum
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

    // 使用扩展后的特征矩阵训练SVM模型
    val svmModelScaledCats = SVMWithSGD.train(scaledDataCats, numIterations)
    val svmTotalCorrectScaledCats = scaledDataCats.map { point =>
      if (svmModelScaledCats.predict(point.features) == point.label) 1 else 0
    }.sum
    val svmAccuracyScaledCats = svmTotalCorrectScaledCats / numData
    val svmPredictionsVsTrueCats = scaledDataCats.map { point =>
      (svmModelScaledCats.predict(point.features), point.label)
    }
    val svmMetricsScaledCats = new BinaryClassificationMetrics(svmPredictionsVsTrueCats)
    val svmPrCats = svmMetricsScaledCats.areaUnderPR()
    val svmRocCats = svmMetricsScaledCats.areaUnderROC()
    println(f"${svmModelScaledCats.getClass.getSimpleName}\n" +
      f"Accuracy: ${svmAccuracyScaledCats * 100}%2.4f%%\n" +
      f"Area under PR: ${svmPrCats * 100.0}%2.4f%%\n" +
      f"Area under ROC: ${svmRocCats * 100.0}%2.4f%%")

    val nbDataWithCategories = records.map { r =>
      // 去掉"
      val trimmed = r.map(_.replaceAll("\"", ""))
      // 提取目标变量，即最后一列
      val label = trimmed(r.size - 1).toInt
      // 提取类别的索引值
      val categoryIdx = categories(r(3))
      // 创建维度为类别数量的特征向量，默认值都是0.0
      val categoryFeatures = Array.ofDim[Double](numCategories)
      // 类别特征向量中指定索引的值变为1.0
      categoryFeatures(categoryIdx) = 1.0
      // 提取其它特征向量，同时处理缺失数据(?变为0.0，负数变为0.0)
      val otherFeatures = trimmed.slice(4, r.size - 1).map(d =>
        if (d == "?") 0.0 else d.toDouble).map(d => if (d < 0) 0.0 else d)

      // 全部特征向量
      val features = categoryFeatures ++ otherFeatures
      // 分类模型通过LabeledPoint对象操作，其中封装了目标变量（标签）和特征向量
      LabeledPoint(label, Vectors.dense(features))
    }
    nbDataWithCategories.cache

    // 使用扩展后的特征矩阵训练朴素贝叶斯模型
    val nbModelCats = NaiveBayes.train(nbDataWithCategories)
    val nbTotalCorrectCats = nbDataWithCategories.map { point =>
      if (nbModelCats.predict(point.features) == point.label) 1 else 0
    }.sum
    val nbAccuracyCats = nbTotalCorrectCats / numData
    val nbPredictionsVsTrueCats = nbDataWithCategories.map { point =>
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

    // 使用扩展后的特征矩阵训练决策树模型
    val dtModelCats = DecisionTree.train(dataWithCategories, Algo.Classification, Entropy, maxTreeDepth)
    val dtTotalCorrectCats = dataWithCategories.map { point =>
      val score = dtModelCats.predict(point.features)
      val predicted = if (score > 0.5) 1 else 0
      if (predicted == point.label) 1 else 0
    }.sum
    val dtAccuracyCats = dtTotalCorrectCats / numData
    val dtPredictionsVsTrueCats = dataWithCategories.map { point =>
      (dtModelCats.predict(point.features), point.label)
    }
    val dtMetricsCats = new BinaryClassificationMetrics(dtPredictionsVsTrueCats)
    val dtPrCats = dtMetricsCats.areaUnderPR()
    val dtRocCats = dtMetricsCats.areaUnderROC()
    println(f"${dtMetricsCats.getClass.getSimpleName}\n" +
      f"Accuracy: ${dtAccuracyCats * 100}%2.4f%%\n" +
      f"Area under PR: ${dtPrCats * 100.0}%2.4f%%\n" +
      f"Area under ROC: ${dtRocCats * 100.0}%2.4f%%")

    // 使用仅包含类型特征的数据训练朴素贝叶斯模型
    val nbDataOnlyCategories = records.map { r =>
      // 去掉"
      val trimmed = r.map(_.replaceAll("\"", ""))
      // 提取目标变量，即最后一列
      val label = trimmed(r.size - 1).toInt
      // 提取类别的索引值
      val categoryIdx = categories(r(3))
      // 创建维度为类别数量的特征向量，默认值都是0.0
      val categoryFeatures = Array.ofDim[Double](numCategories)
      // 类别特征向量中指定索引的值变为1.0
      categoryFeatures(categoryIdx) = 1.0
      // 分类模型通过LabeledPoint对象操作，其中封装了目标变量（标签）和特征向量
      LabeledPoint(label, Vectors.dense(categoryFeatures))
    }
    val nbModelOnlyCats = NaiveBayes.train(nbDataOnlyCategories)
    val nbTotalCorrectOnlyCats = nbDataOnlyCategories.map { point =>
      if (nbModelOnlyCats.predict(point.features) == point.label) 1 else 0
    }.sum
    val nbAccuracyOnlyCats = nbTotalCorrectOnlyCats / numData
    val nbPredictionsVsTrueOnlyCats = nbDataOnlyCategories.map { point =>
      (nbModelOnlyCats.predict(point.features), point.label)
    }
    val nbMetricsOnlyCats = new BinaryClassificationMetrics(nbPredictionsVsTrueOnlyCats)
    val nbPrOnlyCats = nbMetricsOnlyCats.areaUnderPR()
    val nbRocOnlyCats = nbMetricsOnlyCats.areaUnderROC()
    println(f"${nbModelOnlyCats.getClass.getSimpleName}\n" +
      f"Accuracy: ${nbAccuracyOnlyCats * 100}%2.4f%%\n" +
      f"Area under PR: ${nbPrOnlyCats * 100.0}%2.4f%%\n" +
      f"Area under ROC: ${nbRocOnlyCats * 100.0}%2.4f%%")

    // 模型参数调优
    // 迭代次数调优
    val iterResults = Seq(1, 5, 10, 50).map { param =>
      val model = trainWithParams(scaledDataCats, 0.0, param, new SimpleUpdater, 1.0)
      createMetrics(s"$param iterations", scaledDataCats, model)
    }
    iterResults.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }

    // 步长调优（SGD算法支持，LBFGS算法不支持）
    // 在SGD算法中，训练每个样本并更新模型的权重向量时，步长用来控制算法在最陡的梯度方向
    // 上应该前进多远。较大的步长收敛较快，但是步长太大可能导致收敛到局部最优解。
    val stepResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
      val model = trainWithParams(scaledDataCats, 0.0, numIterations, new SimpleUpdater, param)
      createMetrics(s"$param step size", scaledDataCats, model)
    }
    stepResults.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }

    // 正则化调优
    // 正则化通过限制模型的复杂度避免模型在训练数据中过拟合。
    // 正则化的具体做法是在损失函数中添加一项关于模型权重向量的函数，从而会使损失增加。
    // 正则化在现实中几乎是必须的，当特征维度高于训练样本时（此时变量相关需要学习的权重数量
    // 也非常大）尤其重要。相反，虽然正则化可以得到一个简单模型，但正则化太高可能导致模型
    // 欠拟合，从而使模型性能变得很糟糕。
    // MLlib中可用的正则化形式有如下几个：
    // SimpleUpdater：相当于没有正则化，是逻辑回归的默认配置。
    // SquaredL2Updater：这个正则项基于权重向量的L2正则化，是SVM模型的默认值。
    // L1Updater：这个正则项基于权重向量的L1正则化，会导致得到一个稀疏的权重向量（不重要的权重的值接近0）。

    // L1正则化调优
    val l1RegResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
      val model = trainWithParams(scaledDataCats, param, numIterations, new L1Updater, 1.0)
      createMetrics(s"$param L1 regularization parameter", scaledDataCats, model)
    }
    l1RegResults.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }

    // L2正则化调优
    val l2RegResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
      val model = trainWithParams(scaledDataCats, param, numIterations, new SquaredL2Updater, 1.0)
      createMetrics(s"$param L2 regularization parameter", scaledDataCats, model)
    }
    l2RegResults.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }

    // 决策树模型有两种不纯度度量方式：Gini或者Entropy。
    // 决策树模型通常不需要特征的标准化和归一化，也不要求将类型特征进行二元编码。
    // 使用Entropy不纯度进行最大深度调优
    println("train DecisionTree Model with Entropy")
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

    // 使用Gini不纯度进行最大深度调优
    println("train DecisionTree Model with Gini")
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

    // lamda参数在朴素贝叶斯模型中可以控制相加式平滑（additive smoothing），
    // 解决数据中某个类别和某个特征值的组合没有同时出现的问题。
    val nbResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
      val model = trainNBWithParams(nbDataWithCategories, param)
      val scoreAndLabels = nbDataWithCategories.map { point =>
        (model.predict(point.features), point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (s"$param lambda", metrics.areaUnderROC())
    }
    nbResults.foreach { case (param, auc) =>
      println(f"$param, AUC = ${auc * 100}%2.2f%%")
    }

    // 交叉验证
    // 将数据集分成60%的训练集和40%的测试集（使用一个固定的随机种子123来保证每次实验能得到相同的结果）
    val trainTestSplit = scaledDataCats.randomSplit(Array(0.6, 0.4), 123)
    val train = trainTestSplit(0)
    val test = trainTestSplit(1)

    // 使用训练集训练模型并在训练集上评估模型
    // 训练集和测试集相同时，通常在正则化参数比较小的情况下可以得到最高的性能。
    val regResultsTrain = Seq(0.0, 0.001, 0.0025, 0.005, 0.01).map { param =>
      // 使用训练集训练逻辑回归模型
      val model = trainWithParams(train, param, numIterations, new SquaredL2Updater, 1.0)
      // 在训练集上计算模型相关的AUC
      createMetrics(s"$param L2 regularization parameter", train, model)
    }
    regResultsTrain.foreach { case (param, auc) =>
      println(f"$param, AUC = ${auc * 100}%2.6f%%")
    }

    // 使用训练集训练模型并在测试集上评估模型
    // 当训练集和测试集不同时，通常较高正则化可以得到较高的测试性能。
    val regResultsTest = Seq(0.0, 0.001, 0.0025, 0.005, 0.01).map { param =>
      // 使用训练集训练逻辑回归模型
      val model = trainWithParams(train, param, numIterations, new SquaredL2Updater, 1.0)
      // 在测试集上计算模型相关的AUC
      createMetrics(s"$param L2 regularization parameter", test, model)
    }
    regResultsTest.foreach { case (param, auc) =>
      println(f"$param, AUC = ${auc * 100}%2.6f%%")
    }

    // 在交叉验证中，一般选择测试集中性能表现最好的参数设置。
    // 然后用这些参数在所有的数据集上重新训练，最后用于新数据集的预测。

    sc.stop()
  }

  /**
    * 根据参数进行逻辑回归模型训练
    *
    * @param input
    * @param regParam
    * @param numIterations
    * @param updater
    * @param stepSize
    * @return
    */
  def trainWithParams(input: RDD[LabeledPoint], regParam: Double, numIterations: Int, updater: Updater, stepSize: Double) = {
    // val lr = new LogisticRegressionWithSGD
    // lr.optimizer.setNumIterations(numIterations).setUpdater(updater).setRegParam(regParam).setStepSize(stepSize)
    val lr = new LogisticRegressionWithLBFGS
    lr.optimizer.setNumIterations(numIterations).setUpdater(updater).setRegParam(regParam)
    lr.run(input)
  }

  /**
    * 根据输入数据和模型计算相关的AUC
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
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    (label, metrics.areaUnderROC)
  }

  /**
    * 根据参数进行决策树模型训练
    *
    * @param input
    * @param maxDepth
    * @param impurity
    * @return
    */
  def trainDTWithParams(input: RDD[LabeledPoint], maxDepth: Int, impurity: Impurity) = {
    DecisionTree.train(input, Algo.Classification, impurity, maxDepth)
  }

  /**
    * 根据参数进行朴素贝叶斯模型训练
    *
    * @param input
    * @param lambda
    * @return
    */
  def trainNBWithParams(input: RDD[LabeledPoint], lambda: Double) = {
    val nb = new NaiveBayes()
    nb.setLambda(lambda)
    nb.run(input)
  }
}
