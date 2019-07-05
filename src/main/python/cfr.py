#!/usr/bin/python
#  -*- coding:utf-8 -*-

import numpy as np

np.seterr(divide='ignore', invalid='ignore')


def load_local_rating():
    my_mat = np.mat(np.zeros((943, 1682)))
    fr = open("../resources/data/ml-100k/u.data")
    for line in fr.readlines():
        cols = line.strip().split('\t')
        my_mat[int(cols[0]) - 1, int(cols[1]) - 1] = float(cols[2])
    return my_mat


def load_local_movie_titles():
    fr = open("../resources/data/ml-100k/u.item")
    movies = {}
    for line in fr.readlines():
        cols = line.strip().split('|')
        movies[int(cols[0])] = cols[1]
    return movies


def load_hdfs_rating():
    global sc
    # 读取数据集
    rating_data = sc.textFile("hdfs://PATH/ml-100k/u.data")
    rating_fields = rating_data.map(lambda line: line.split("\t"))
    # 用户数量
    num_users = rating_fields.map(lambda fields: int(fields[0])).distinct().count()
    # 电影数量
    num_movies = rating_fields.map(lambda fields: int(fields[1])).distinct().count()
    my_mat = np.mat(np.zeros((num_users, num_movies)))
    for fields in rating_fields.collect():
        my_mat[int(fields[0]) - 1, int(fields[1]) - 1] = float(fields[2])

    return my_mat


def load_hdfs_movie_titles():
    global sc
    # 读取数据集
    movie_data = sc.textFile("hdfs://PATH/ml-100k/u.item")
    movie_fields = movie_data.map(lambda line: line.split("|"))
    titles = movie_fields.map(lambda fields: (int(fields[0]), fields[1])).collectAsMap()
    return titles


def euclid_sim(inA, inB):
    '''
    使用欧式距离计算相似度(1/(1+欧式距离))
    :param inA: 列向量A
    :param inB: 列向量B
    :return: 相似度
    '''
    # np.linalg.norm(): 计算范数的方法, 默认是2范数
    # 向量A减去向量B, 再求向量差的2范数, 就得到了向量A和向量B的欧式距离.
    return 1.0 / (1.0 + np.linalg.norm(inA - inB))


def pears_sim(inA, inB):
    '''
    使用皮尔逊相关系数计算相似度(0.5+0.5*np.corrcoef())
    :param inA: 列向量A
    :param inB: 列向量B
    :return: 相似度
    '''
    if len(inA) < 3:  # 小于3个点则完全相关, 相似度为1.0
        return 1.0
    # np.corrcoef(): 计算皮尔逊相关系数, rowvar等于0, 说明传入的数据每一行代表一个样本; 不等于0, 说明传入的数据每一列代表一个样本.
    # 通过0.5+0.5*corrcoef()这个计算公式将取值范围归一化为0到1之间.
    return 0.5 + 0.5 * np.corrcoef(inA, inB, rowvar=0)[0][1]


def cos_sim(inA, inB):
    '''
    计算余弦相似度
    :param inA: 列向量A
    :param inB: 列向量B
    :return: 余弦相似度
    '''
    # 计算两个向量的内积, 即点乘加和
    num = float(inA.T * inB)
    # 计算两个向量的2范数再相乘
    denom = np.linalg.norm(inA) * np.linalg.norm(inB)
    # 通过0.5+0.5*cos(theta)这个计算公式将取值范围归一化为0到1之间.
    return 0.5 + 0.5 * (num / denom)


def stand_est(data_mat, user, sim_meas, item):
    '''
    在给定相似度计算方法的条件下, 计算用户user对物品item的估计评分值.
    遍历全部物品, 在用户user对某一物品有评分的情况下,计算该物品与物品item的相似度
    (基于对这2个物品都做过评分的所有用户给出的评分值计算相似度)
    得到某个物品与物品item的相似度以后, 再乘以用户user对于该物品的评分, 可以看作是
    用户对于该物品与物品item针对它们相似度的评分, 将这些评分进行累加, 再除以所有物品
    的相似度之和, 就得到了用户对于物品item的估计评分值.
    :param data_mat: 数据矩阵, 行对应用户, 列对应物品, 则行与行之间比较的是基于用户的相似度,
                     列与列之间比较的是基于物品的相似度.
    :param user: 用户编号
    :param sim_meas: 相似度计算方法
    :param item: 物品编号
    :return: 用户对于物品item的估计评分值
    '''
    n = np.shape(data_mat)[1]  # 物品数目
    sim_total = 0.0  # 总的相似度
    rat_sim_total = 0.0  # (相似度*评分)之和
    # 遍历每一个物品
    for j in range(n):
        user_rating = data_mat[user, j]  # 得到用户对该物品的评分值
        if user_rating == 0:  # 如果评分值为0, 就意味着用户没有对该物品评分, 那么就不计算这个物品
            continue
        # data_mat[:, item].A > 0: 编号为item的物品中有评分值(即评分值>0)的数据
        # data_mat[:, j].A > 0: 编号为j(j为遍历的每一个物品)的物品中有评分值(即评分值>0)的数据
        # np.logical_and(True, False) --> False
        # np.logical_and(False, False) --> False
        # np.logical_and(True, True) --> True
        # np.logical_and([True, False, True], [False, False, True]) --> array([False, False, True], dtype=bool)
        # 编号为item的物品和编号为j的物品都有评分值(即评分值>0)的那些行, 即用户
        over_lap = np.nonzero(np.logical_and(data_mat[:, item].A > 0, data_mat[:, j].A > 0))[0]
        if len(over_lap) == 0:  # 如果没有任何一个用户对这2个物品同时评过分, 则这2个物品的相似度为0
            similarity = 0
        else:  # 使用指定的相似度计算方法计算这2个物品的相似度(基于用户评分的相似度计算方法)
            similarity = sim_meas(data_mat[over_lap, item], data_mat[over_lap, j])
        if (np.isnan(similarity)):
            # print "相似度不是有效数字: j = {0}, 向量A = {1}, 向量B = {2}".format(j, data_mat[over_lap, item].T, data_mat[over_lap, j].T)
            similarity = 0
        # print 'the %d and %d similarity is : %f' % (item, j, similarity)
        sim_total += similarity  # 物品item和物品j的相似度做累加
        rat_sim_total += similarity * user_rating  # 物品item和物品j的相似度乘以用户对物品j的评分, 再做累加
    if sim_total == 0:  # 相似度之和为0, 则用户对于物品item的估计评分值也为0
        return 0
    else:  # 相似度之和非0
        return rat_sim_total / sim_total  # 返回用户对于物品item的估计评分值


def svd_est(data_mat, user, sim_meas, item):
    '''
    在给定相似度计算方法的条件下, 计算用户user对物品item的估计评分值.
    本函数是基于SVD进行估计评分的.
    :param data_mat: 数据矩阵, 行对应用户, 列对应物品, 则行与行之间比较的是基于用户的相似度,
                     列与列之间比较的是基于物品的相似度.
    :param user: 用户编号
    :param sim_meas: 相似度计算方法
    :param item: 物品编号
    :return: 用户对于物品item的估计评分值
    '''
    n = np.shape(data_mat)[1]  # 物品数目
    sim_total = 0.0  # 总的相似度
    rat_sim_total = 0.0  # (相似度*评分)之和
    # 执行SVD, Sigma是NumPy数组的形式
    U, Sigma, VT = np.linalg.svd(data_mat)
    # 使用包含90%能量值的奇异值建立对角矩阵
    Sig4 = np.mat(np.eye(4) * Sigma[:4])
    # data_mat: mxn, 行对应用户, 列对应物品
    # data_mat.T: nxm, data_mat转置之后, 行对应物品, 列对应用户
    # U: mxm, 则U[:, :4]: mx4
    # Sig4: 4x4
    # U矩阵会将物品映射到低维空间中, VT矩阵会将用户映射到低维空间中
    # 计算得到的矩阵, nx4, 行仍对应物品, 列仍对应用户, 物品总数未变, 减少的是用户
    xformed_items = data_mat.T * U[:, :4] * Sig4.I
    # 遍历每一个物品
    for j in range(n):
        user_rating = data_mat[user, j]  # 得到用户对该物品的评分值
        if user_rating == 0 or j == item:  # 如果评分值为0, 就意味着用户没有对该物品评分, 或者物品j和物品item是同一件物品, 那么就不计算这个物品
            continue
        # 在低维空间下, 使用指定的相似度计算方法计算这2个物品的相似度(基于用户评分的相似度计算方法)
        # 矩阵xformed_items的行对应物品, 列对应用户, 相似度计算方法的参数为列向量, 所以
        # xformed_items[item, :].T 是物品item的列向量,
        # xformed_items[j, :].T 是物品j的列向量.
        similarity = sim_meas(xformed_items[item, :].T, xformed_items[j, :].T)
        # print 'the %d and %d similarity is : %f' % (item, j, similarity)
        sim_total += similarity  # 物品item和物品j的相似度做累加
        rat_sim_total += similarity * user_rating  # 物品item和物品j的相似度乘以用户对物品j的评分, 再做累加
    if sim_total == 0:  # 相似度之和为0, 则用户对于物品item的估计评分值也为0
        return 0
    else:  # 相似度之和非0
        return rat_sim_total / sim_total  # 返回用户对于物品item的估计评分值


def recommend(data_mat, user, N=3, sim_meas=cos_sim, est_method=stand_est):
    '''
    基于物品相似度的推荐引擎
    :param data_mat: 数据矩阵, 行对应用户, 列对应物品, 则行与行之间比较的是基于用户的相似度,
                     列与列之间比较的是基于物品的相似度
    :param user: 用户编号
    :param N: 推荐物品的数量
    :param sim_meas: 相似度计算方法
    :param est_method: 评分估计方法
    :return:
    '''
    # 寻找用户未评分的物品, 建立未评分的物品列表
    unrated_items = np.nonzero(data_mat[user, :].A == 0)[1]
    if len(unrated_items) == 0:  # 不存在用户未评分的物品则直接退出
        return 'you rated everything'
    item_scores = []  # 估计评分值列表
    for item in unrated_items:  # 遍历所有的未评分物品
        # 计算用户对于每个未评分物品的估计评分值
        estimated_score = est_method(data_mat, user, sim_meas, item)
        # 将物品编号和用户对该物品的估计评分值保存到估计评分值列表中
        item_scores.append((item, estimated_score))
    # 对估计评分值列表按照估计评分值进行从大到小的排序, 返回指定的N个物品
    return sorted(item_scores, key=lambda jj: jj[1], reverse=True)[:N]


def run_local():
    np.set_printoptions(linewidth=1000)
    my_mat = load_local_rating()
    print my_mat

    euclid_s = euclid_sim(my_mat[:, 0], my_mat[:, 4])
    print euclid_s
    euclid_s = euclid_sim(my_mat[:, 0], my_mat[:, 0])
    print euclid_s
    cos_s = cos_sim(my_mat[:, 0], my_mat[:, 4])
    print cos_s
    cos_s = cos_sim(my_mat[:, 0], my_mat[:, 0])
    print cos_s
    pears_s = pears_sim(my_mat[:, 0], my_mat[:, 4])
    print pears_s
    pears_s = pears_sim(my_mat[:, 0], my_mat[:, 0])
    print pears_s

    # 执行SVD
    U, Sigma, VT = np.linalg.svd(my_mat)
    Sig2 = Sigma ** 2
    print 'U: \n', U
    print 'Sigma: \n', Sigma
    print 'Sig2: \n', Sig2
    print 'VT: \n', VT

    movie_titles = load_local_movie_titles()
    rec = recommend(my_mat, 789, N=10)
    print "通过计算余弦相似度获得的推荐电影："
    for index, r in enumerate(rec):
        print (index + 1), ".", movie_titles[r[0]], r[1]
    rec = recommend(my_mat, 789, N=10, sim_meas=euclid_sim)
    print "通过使用欧式距离计算相似度获得的推荐电影："
    for index, r in enumerate(rec):
        print (index + 1), ".", movie_titles[r[0]], r[1]
    rec = recommend(my_mat, 789, N=10, sim_meas=pears_sim)
    print "通过使用皮尔逊相关系数计算相似度获得的推荐电影："
    for index, r in enumerate(rec):
        print (index + 1), ".", movie_titles[r[0]], r[1]

    rec = recommend(my_mat, 789, est_method=svd_est)
    print "通过SVD评分估计方法及计算余弦相似度获得的推荐电影："
    for index, r in enumerate(rec):
        print (index + 1), ".", movie_titles[r[0]], r[1]
    rec = recommend(my_mat, 789, est_method=svd_est, sim_meas=pears_sim)
    print "通过SVD评分估计方法及使用皮尔逊相关系数计算相似度获得的推荐电影："
    for index, r in enumerate(rec):
        print (index + 1), ".", movie_titles[r[0]], r[1]


def run_hdfs():
    np.set_printoptions(linewidth=1000)
    my_mat = load_hdfs_rating()
    print my_mat

    euclid_s = euclid_sim(my_mat[:, 0], my_mat[:, 4])
    print euclid_s
    euclid_s = euclid_sim(my_mat[:, 0], my_mat[:, 0])
    print euclid_s
    cos_s = cos_sim(my_mat[:, 0], my_mat[:, 4])
    print cos_s
    cos_s = cos_sim(my_mat[:, 0], my_mat[:, 0])
    print cos_s
    pears_s = pears_sim(my_mat[:, 0], my_mat[:, 4])
    print pears_s
    pears_s = pears_sim(my_mat[:, 0], my_mat[:, 0])
    print pears_s

    # 执行SVD
    U, Sigma, VT = np.linalg.svd(my_mat)
    Sig2 = Sigma ** 2
    print 'U: \n', U
    print 'Sigma: \n', Sigma
    print 'Sig2: \n', Sig2
    print 'VT: \n', VT

    movie_titles = load_hdfs_movie_titles()
    rec = recommend(my_mat, 789, N=10)
    print "通过计算余弦相似度获得的推荐电影："
    for index, r in enumerate(rec):
        print (index + 1), ".", movie_titles(r[0]), r[1]
    rec = recommend(my_mat, 789, N=10, sim_meas=euclid_sim)
    print "通过使用欧式距离计算相似度获得的推荐电影："
    for index, r in enumerate(rec):
        print (index + 1), ".", movie_titles(r[0]), r[1]
    rec = recommend(my_mat, 789, N=10, sim_meas=pears_sim)
    print "通过使用皮尔逊相关系数计算相似度获得的推荐电影："
    for index, r in enumerate(rec):
        print (index + 1), ".", movie_titles(r[0]), r[1]

    rec = recommend(my_mat, 789, est_method=svd_est)
    print "通过SVD评分估计方法及计算余弦相似度获得的推荐电影："
    for index, r in enumerate(rec):
        print (index + 1), ".", movie_titles(r[0]), r[1]
    rec = recommend(my_mat, 789, est_method=svd_est, sim_meas=pears_sim)
    print "通过SVD评分估计方法及使用皮尔逊相关系数计算相似度获得的推荐电影："
    for index, r in enumerate(rec):
        print (index + 1), ".", movie_titles(r[0]), r[1]


if __name__ == '__main__':
    run_local()

    # from pyspark import SparkContext
    # sc = SparkContext()
    # run_hdfs()
    # sc.stop()
