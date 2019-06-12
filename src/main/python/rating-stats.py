#!/usr/bin/python
#  -*- coding:utf-8 -*-

from pyspark import SparkContext
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sc = SparkContext()
# 读取数据集
rating_data_raw = sc.textFile("hdfs:/PATH/ml-100k/u.data")
# 输出第1行数据
print(rating_data_raw.first())
# 统计数量
num_ratings = rating_data_raw.count()
# Ratings: 100000
print("Ratings: %d" % num_ratings)

rating_data = rating_data_raw.map(lambda line: line.split("\t"))
ratings = rating_data.map(lambda fields: int(fields[2]))
max_rating = ratings.reduce(lambda x, y: max(x, y))
min_rating = ratings.reduce(lambda x, y: min(x, y))
mean_rating = ratings.reduce(lambda x, y: x + y) / num_ratings
median_rating = np.median(ratings.collect())
num_users = 943
ratings_per_user = num_ratings / num_users
num_movies = 1682
ratings_per_movie = num_ratings / num_movies
# Min rating: 1
print("Min rating: %d" % min_rating)
# Max rating: 5
print("Max rating: %d" % max_rating)
# Average rating: 3.53
print("Average rating: %2.2f" % mean_rating)
# Median rating: 4
print("Median rating: %d" % median_rating)
# Average # of ratings per user: 106.00
print("Average # of ratings per user: %2.2f" % ratings_per_user)
# Average # of ratings per movie: 59.00
print("Average # of ratings per movie: %2.2f" % ratings_per_movie)

print(ratings.stats())

count_by_rating = ratings.countByValue()
x_axis = np.array(count_by_rating.keys())
y_axis = np.array([float(c) for c in count_by_rating.values()])
# 这里对y轴正则化，使它表示百分比
y_axis_normed = y_axis / y_axis.sum()
pos = np.arange(len(x_axis))
width = 1.0
ax = plt.axes()
ax.set_xticks(pos + (width / 2))
ax.set_xticklabels(x_axis)
plt.bar(pos, y_axis_normed, width, color='lightblue')
plt.xticks(rotation=30)
fig = plt.gcf()
fig.set_size_inches(16, 10)

# 按照用户进行分组
user_ratings_grouped = rating_data.map(lambda fields: (int(fields[0]), int(fields[2]))).groupByKey()
# 统计每一个用户的评级次数
user_ratings_byuser = user_ratings_grouped.map(lambda (k, v): (k, len(v)))
print(user_ratings_byuser.take(5))

user_ratings_byuser_local = user_ratings_byuser.map(lambda (k, v): v).collect()
plt.hist(user_ratings_byuser_local, bins=200, color='lightblue', normed=True)
fig = plt.gcf()
fig.set_size_inches(16,10)