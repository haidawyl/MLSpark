#!/usr/bin/python
#  -*- coding:utf-8 -*-

import numpy as np
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from pyspark.mllib.tree import DecisionTree


def get_mapping(rdd, idx):
    '''
    将类型特征表示成二维形式，同时将特征值映射到二元向量中非0的位置
    :param rdd:
    :param idx:
    :return:
    '''
    # 将第idx列的特征值去重，然后对每个值使用zipWithIndex函数映射到一个唯一的索引，
    # 组成一个RDD的键-值映射，键是变量，值是索引。该索引便是特征在二元向量中对应的非0位置。
    return rdd.map(lambda fields: fields[idx]).distinct().zipWithIndex().collectAsMap()


def extract_features(record):
    '''
    提取特征
    :param record:
    :return:
    '''
    # 创建长度为cat_len的全0向量
    cat_vec = np.zeros(cat_len)
    i = 0
    # 各个特征的二元编码的累计长度，确保非0特征在整个特征向量中位于正确的位置
    step = 0
    # mappings: 字典数组
    # 类型特征
    for field in record[2:10]:
        # 字典类型，特征值在二元编码中的映射，即{特征值: 索引}
        m = mappings[i]
        # 特征值在二元编码中的索引
        idx = m[field]
        cat_vec[idx + step] = 1
        i = i + 1
        step = step + len(m)
    # 数值特征
    num_vec = np.array([float(field) for field in record[10:14]])
    return np.concatenate((cat_vec, num_vec))


def extract_features2(record):
    '''
    提取特征，将各个特征的二元编码拼接在一起
    :param record:
    :return:
    '''
    # 创建空数组
    cat_vec = np.array([])
    i = 0
    # mappings: 字典数组
    # 类型特征
    for field in record[2:10]:
        # 字典类型，特征值在二元编码中的映射，即{特征值: 索引}
        m = mappings[i]
        # 特征值在二元编码中的索引
        idx = m[field]
        # 创建数组，大小为特征的二元编码长度
        field_vec = np.zeros(len(m))
        field_vec[idx] = 1
        cat_vec = np.concatenate((cat_vec, field_vec))
        i = i + 1
    # 数值特征
    num_vec = np.array([float(field) for field in record[10:14]])
    return np.concatenate((cat_vec, num_vec))


def extract_label(record):
    '''
    提取标签
    :param record:
    :return:
    '''
    return float(record[-1])


def squared_error(actual, predicted):
    '''
    计算平方误差，即预测值和实际值的差的平方
    :param actual: 实际值
    :param predicted: 预测值
    :return: 平方误差
    '''
    return (predicted - actual) ** 2


def abs_error(actual, predicted):
    '''
    计算绝对误差，即预测值和实际值的差的绝对值
    :param actual: 实际值
    :param predicted: 预测值
    :return: 平均绝对误差
    '''
    return np.abs(predicted - actual)


def squared_log_error(actual, predicted):
    '''
    计算均方对数误差，即样本预测值和实际值进行对数变换后的MSE（均方误差）
    这个度量方法适用于目标变量值域很大，并且没有必要对预测值和目标值的误差进行惩罚的情况。
    另外，它也适用于计算误差的百分率而不是误差的绝对值。
    :param actual: 实际值
    :param predicted: 预测值
    :return: 平均绝对误差
    '''
    return (np.log(predicted + 1) - np.log(actual + 1)) ** 2


def extract_features_dt(record):
    '''
    提取特征，用于决策树模型
    :param record:
    :return:
    '''
    return np.array(map(float, record[2:14]))


def evaluate(train, test, iterations, step, regParam, regType, intercept):
    '''
    :param train
    :param test
    :param iterations
    :param step
    :param regParam
    :param regType
    :param intercept
    :return:
    '''
    model = LinearRegressionWithSGD.train(train, iterations, step, regParam=regParam, regType=regType,
                                          intercept=intercept)
    tp = test.map(lambda p: (p.label, model.predict(p.features)))
    rmsle = np.sqrt(tp.map(lambda (t, p): squared_log_error(t, p)).mean())
    return rmsle


def evaluate_dt(train, test, maxDepth, maxBins):
    '''
    :param train:
    :param test:
    :param maxDepth:
    :param maxBins:
    :return:
    '''
    model = DecisionTree.trainRegressor(train, {}, impurity='variance', maxDepth=maxDepth, maxBins=maxBins)
    predicted = model.predict(test.map(lambda p: p.features))
    actual = test.map(lambda p: p.label)
    tp = actual.zip(predicted)
    rmsle = np.sqrt(tp.map(lambda (t, p): squared_log_error(t, p)).mean())
    return rmsle


if __name__ == "__main__":
    # 线性回归在应用L2正则化时通常称为岭回归（ridge regression），
    # 应用L1正则化时称为LASSO（Least Absolute Shrinkage and Selection Operator）。
    # 决策树在用于回归时使用的不纯度度量方法是方差。
    sc = SparkContext()
    # 读取数据集
    raw_data = sc.textFile("hdfs://PATH/BikeSharing/hour_noheader.csv")
    num_data = raw_data.count()
    # num_data = 17379
    print("num_data = %d" % num_data)
    records = raw_data.map(lambda x: x.split(","))
    first = records.first()
    print(first)

    # 缓存数据
    records.cache()

    print("Mapping of first categorical features column: %s" % get_mapping(records, 2))

    # mappings: [{}, {}, {}]
    mappings = [get_mapping(records, i) for i in range(2, 10)]
    cat_len = sum(map(len, mappings))
    num_len = len(first[11:15])
    total_len = num_len + cat_len

    # Feature vector length for categorical features: 57
    print("Feature vector length for categorical features: %d" % cat_len)
    # Feature vector length for numerical features: 4
    print("Feature vector length for numerical features: %d" % num_len)
    # Total feature vector length: 61
    print("Total feature vector length: %d" % total_len)

    # 提取每条数据记录的特征向量和标签
    data = records.map(lambda r: LabeledPoint(extract_label(r), extract_features(r)))
    data.cache()
    first_point = data.first()
    print("Raw data:" + str(first[2:-3]))
    print("Label:" + str(first_point.label))
    print("Linear Model feature vector:\n" + str(first_point.features))
    print("Linear Model feature vector length: " + str(len(first_point.features)))

    data_dt = records.map(lambda r: LabeledPoint(extract_label(r), extract_features_dt(r)))
    data_dt.cache()
    first_point_dt = data_dt.first()
    print("Decision Tree feature vector:\n" + str(first_point_dt.features))
    print("Decision Tree feature vector length: " + str(len(first_point_dt.features)))

    help(LinearRegressionWithSGD.train)
    help(DecisionTree.trainRegressor)

    linear_model = LinearRegressionWithSGD.train(data, iterations=10, step=0.1, intercept=False)
    true_vs_predicted = data.map(lambda p: (p.label, linear_model.predict(p.features)))
    print("Linear Model predictions: " + str(true_vs_predicted.take(5)))

    # categorical_features_info为一个字典参数，这个字典参数将类型特征的索引映射到特征中类型的数目。
    # 如果某个特征值不在这个字典中，则将其映射设置为空。
    categorical_features_info = {}
    dt_model = DecisionTree.trainRegressor(data_dt, categorical_features_info)
    predicted = dt_model.predict(data_dt.map(lambda p: p.features))
    actual = data_dt.map(lambda p: p.label)
    true_vs_predicted_dt = actual.zip(predicted)
    print("Decision Tree predictions: " + str(true_vs_predicted_dt.take(5)))
    print("Decision Tree depth: " + str(dt_model.depth()))
    print("Decision Tree number of nodes: " + str(dt_model.numNodes()))

    # 用于评估回归模型的方法包括：
    # 均方误差（MSE，Mean Squared Error）、
    # 均方根误差（RMSE，Root Mean Squared Error）、
    # 平均绝对误差（MAE，Mean Absolute Error）、
    # R-平方系数（R-squared coefficient）等。

    mse = true_vs_predicted.map(lambda (t, p): squared_error(t, p)).mean()
    mae = true_vs_predicted.map(lambda (t, p): abs_error(t, p)).mean()
    rmsle = np.sqrt(true_vs_predicted.map(lambda (t, p): squared_log_error(t, p)).mean())
    print("Linear Model - Mean Squared Error: %2.4f" % mse)
    print("Linear Model - Mean Absolute Error: %2.4f" % mae)
    print("Linear Model - Root Mean Squared Log Error: %2.4f" % rmsle)

    mse_dt = true_vs_predicted_dt.map(lambda (t, p): squared_error(t, p)).mean()
    mae_dt = true_vs_predicted_dt.map(lambda (t, p): abs_error(t, p)).mean()
    rmsle_dt = np.sqrt(true_vs_predicted_dt.map(lambda (t, p): squared_log_error(t, p)).mean())
    print("Decision Tree - Mean Squared Error: %2.4f" % mse_dt)
    print("Decision Tree - Mean Absolute Error: %2.4f" % mae_dt)
    print("Decision Tree - Root Mean Squared Log Error: %2.4f" % rmsle_dt)

    # 线性模型的假设为正态分布

    # 对目标变量进行对数变换
    data_log = data.map(lambda lp: LabeledPoint(np.log(lp.label), lp.features))
    model_log = LinearRegressionWithSGD.train(data_log, iterations=10, step=0.1)
    true_vs_predicted_log = data_log.map(lambda p: (np.exp(p.label), np.exp(model_log.predict(p.features))))
    mse_log = true_vs_predicted_log.map(lambda (t, p): squared_error(t, p)).mean()
    mae_log = true_vs_predicted_log.map(lambda (t, p): abs_error(t, p)).mean()
    rmsle_log = np.sqrt(true_vs_predicted_log.map(lambda (t, p): squared_log_error(t, p)).mean())
    print("对目标变量进行对数变换后训练线性回归模型计算得到的MSE、MAE和RMSLE")
    print("Mean Squared Error: %2.4f" % mse_log)
    print("Mean Absolute Error: %2.4f" % mae_log)
    print("Root Mean Squared Log Error: %2.4f" % rmsle_log)
    print("Non log-transformed predictions:\n" + str(true_vs_predicted.take(3)))
    print("Log-transformed predictions:\n" + str(true_vs_predicted_log.take(3)))

    data_dt_log = data_dt.map(lambda lp: LabeledPoint(np.log(lp.label), lp.features))
    categorical_features_info = {}
    dt_model_log = DecisionTree.trainRegressor(data_dt_log, categorical_features_info)
    predicted_log = dt_model_log.predict(data_dt_log.map(lambda p: p.features))
    actual_log = data_dt_log.map(lambda p: p.label)
    true_vs_predicted_dt_log = actual_log.zip(predicted_log).map(lambda (t, p): (np.exp(t), np.exp(p)))
    mse_log_dt = true_vs_predicted_dt_log.map(lambda (t, p): squared_error(t, p)).mean()
    mae_log_dt = true_vs_predicted_dt_log.map(lambda (t, p): abs_error(t, p)).mean()
    rmsle_log_dt = np.sqrt(true_vs_predicted_dt_log.map(lambda (t, p): squared_log_error(t, p)).mean())
    print("对目标变量进行对数变换后训练决策树模型计算得到的MSE、MAE和RMSLE")
    print("Mean Squared Error: %2.4f" % mse_log_dt)
    print("Mean Absolute Error: %2.4f" % mae_log_dt)
    print("Root Mean Squared Log Error: %2.4f" % rmsle_log_dt)
    print("Non log-transformed predictions:\n" + str(true_vs_predicted_dt.take(3)))
    print("Log-transformed predictions:\n" + str(true_vs_predicted_dt_log.take(3)))

    # 对目标变量进行取平方根变换
    data_sqrt = data.map(lambda lp: LabeledPoint(np.sqrt(lp.label), lp.features))
    model_sqrt = LinearRegressionWithSGD.train(data_sqrt, iterations=10, step=0.1)
    true_vs_predicted_sqrt = data_sqrt.map(lambda p: (p.label ** 2, model_sqrt.predict(p.features) ** 2))
    mse_sqrt = true_vs_predicted_sqrt.map(lambda (t, p): squared_error(t, p)).mean()
    mae_sqrt = true_vs_predicted_sqrt.map(lambda (t, p): abs_error(t, p)).mean()
    rmsle_sqrt = np.sqrt(true_vs_predicted_sqrt.map(lambda (t, p): squared_log_error(t, p)).mean())
    print("对目标变量进行取平方根变换后训练线性回归模型计算得到的MSE、MAE和RMSLE")
    print("Mean Squared Error: %2.4f" % mse_sqrt)
    print("Mean Absolute Error: %2.4f" % mae_sqrt)
    print("Root Mean Squared Log Error: %2.4f" % rmsle_sqrt)
    print("Non sqrt-transformed predictions:\n" + str(true_vs_predicted.take(3)))
    print("Sqrt-transformed predictions:\n" + str(true_vs_predicted_sqrt.take(3)))

    data_dt_sqrt = data_dt.map(lambda lp: LabeledPoint(np.sqrt(lp.label), lp.features))
    categorical_features_info = {}
    dt_model_sqrt = DecisionTree.trainRegressor(data_dt_sqrt, categorical_features_info)
    predicted_sqrt = dt_model_sqrt.predict(data_dt_sqrt.map(lambda p: p.features))
    actual_sqrt = data_dt_sqrt.map(lambda p: p.label)
    true_vs_predicted_dt_sqrt = actual_sqrt.zip(predicted_sqrt).map(lambda (t, p): (t ** 2, p ** 2))
    mse_sqrt_dt = true_vs_predicted_dt_sqrt.map(lambda (t, p): squared_error(t, p)).mean()
    mae_sqrt_dt = true_vs_predicted_dt_sqrt.map(lambda (t, p): abs_error(t, p)).mean()
    rmsle_sqrt_dt = np.sqrt(true_vs_predicted_dt_sqrt.map(lambda (t, p): squared_log_error(t, p)).mean())
    print("对目标变量进行取平方根变换后训练决策树模型计算得到的MSE、MAE和RMSLE")
    print("Mean Squared Error: %2.4f" % mse_sqrt_dt)
    print("Mean Absolute Error: %2.4f" % mae_sqrt_dt)
    print("Root Mean Squared Log Error: %2.4f" % rmsle_sqrt_dt)
    print("Non sqrt-transformed predictions:\n" + str(true_vs_predicted_dt.take(3)))
    print("Sqrt-transformed predictions:\n" + str(true_vs_predicted_dt_sqrt.take(3)))

    # 线性模型在经过对数处理后的数据得到较好的性能是意料之中的。因为本质上我们的目的是最小化均方差，
    # 一旦把目标值转换为对数值，便可以有效最小化损失函数，即最小化RMSLE。

    # 创建训练集和测试集
    data_with_idx = data.zipWithIndex().map(lambda (k, v): (v, k))
    # 随机采样提取测试集
    test = data_with_idx.sample(False, 0.2, 42)
    # 剩余数据组成训练集
    train = data_with_idx.subtractByKey(test)

    train_data = train.map(lambda (idx, p): p)
    test_data = test.map(lambda (idx, p): p)
    train_size = train_data.count()
    test_size = test_data.count()

    print("Training data size: %d" % train_size)
    print("Test data size: %d" % test_size)
    print("Total data size: %d " % num_data)
    print("Train + Test size : %d" % (train_size + test_size))

    data_with_idx_dt = data_dt.zipWithIndex().map(lambda (k, v): (v, k))
    # 随机采样提取测试集
    test_dt = data_with_idx_dt.sample(False, 0.2, 42)
    # 剩余数据组成训练集
    train_dt = data_with_idx_dt.subtractByKey(test)
    train_data_dt = train_dt.map(lambda (idx, p): p)
    test_data_dt = test_dt.map(lambda (idx, p): p)

    # 迭代
    # 通常在使用SGD训练模型的过程中，随着迭代次数增加可以实现更好的性能，
    # 但是性能在迭代次数达到一定数目时会增长得越来越慢。
    params = [1, 5, 10, 20, 50, 100]
    metrics = [evaluate(train_data, test_data, param, 0.01, 0.0, 'l2', False) for param in params]
    print(params)
    print(metrics)

    # 步长
    # SGD模型在步长较大的时候容易收敛到最差的局部最优解，原因是算法收敛太快而不能得到最优解。
    # 小步长与相对较小的迭代次数对应的训练模型性能一般较差，
    # 而较小的步长与较大的迭代次数通常可以收敛得到较好的解。
    # 通常来讲，步长和迭代次数的设定需要权衡。较小的步长意味着收敛速度慢，
    # 需要较大的迭代次数。但是较大的迭代次数更加耗时，特别是在大数据集上。
    params = [0.01, 0.025, 0.05, 0.1, 1.0]
    metrics = [evaluate(train_data, test_data, 10, param, 0.0, 'l2', False) for param in params]
    print(params)
    print(metrics)

    # L2正则化参数
    # 正则化是添加一个关于模型权重向量的函数作为损失项，来惩罚模型的复杂度。
    # 其中L2正则化是对权重向量进行L2-norm惩罚，而L1正则化是对权重向量进行L1-norm惩罚。
    # 随着正则化的提高，训练集的预测性能会下降，因为模型不能很好拟合数据。
    # 但是，设置合适的正则化参数，能够在测试集上达到最好的性能，最终得到一个泛化能力最优的模型。
    params = [0.0, 0.01, 0.1, 1.0, 5.0, 10.0, 20.0]
    metrics = [evaluate(train_data, test_data, 10, 0.1, param, 'l2', False) for param in params]
    print(params)
    print(metrics)

    # L1正则化参数
    params = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    metrics = [evaluate(train_data, test_data, 10, 0.1, param, 'l1', False) for param in params]
    print(params)
    print(metrics)

    # 使用L1正则化可以得到稀疏的权重向量。
    model_l1 = LinearRegressionWithSGD.train(train_data, 10, 0.1, regParam=1.0, regType='l1', intercept=False)
    model_l1_10 = LinearRegressionWithSGD.train(train_data, 10, 0.1, regParam=10.0, regType='l1', intercept=False)
    model_l1_100 = LinearRegressionWithSGD.train(train_data, 10, 0.1, regParam=100.0, regType='l1', intercept=False)
    # 随着L1的正则化参数越来越大，模型的权重向量中0的数目也越来越大。
    print("L1 (1.0) number of zero weights: " + str(sum(model_l1.weights.array == 0)))
    print("L1 (10.0) number of zero weights: " + str(sum(model_l1_10.weights.array == 0)))
    print("L1 (100.0) number of zero weights: " + str(sum(model_l1_100.weights.array == 0)))

    # 截距
    # 截距是添加到权重向量的常数项，可以有效地影响目标变量的中值。
    # 如果数据已经被归一化，截距则没有必要。但是理论上截距的使用并不会带来坏处。
    params = [False, True]
    metrics = [evaluate(train_data, test_data, 10, 0.1, 1.0, 'l2', param) for param in params]
    print(params)
    print(metrics)

    # 决策树提供了两个主要的参数：最大树深度和最大划分数。
    # 最大树深度
    params = [1, 2, 3, 4, 5, 10, 20]
    metrics = [evaluate_dt(train_data_dt, test_data_dt, param, 32) for param in params]
    print(params)
    print(metrics)

    # 最大划分数
    params = [2, 4, 8, 16, 32, 64, 100]
    metrics = [evaluate_dt(train_data_dt, test_data_dt, 5, param) for param in params]
    print(params)
    print(metrics)
