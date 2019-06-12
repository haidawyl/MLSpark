#!/usr/bin/python
#  -*- coding:utf-8 -*-

from pyspark import SparkContext
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sc = SparkContext()
# 读取数据集
user_data = sc.textFile("hdfs:/PATH/ml-100k/u.user")
# 输出第1行数据
print(user_data.first())
# 输出前k行数据
print(user_data.take(10))

# 用户ID（user ID）、年龄（age）、性别（gender）、职业（occupation）和邮编（ZIP code）
user_fields = user_data.map(lambda line: line.split("|"))
# 统计用户数量
num_users = user_fields.map(lambda fields: fields[0]).count()
# 统计性别数量
num_genders = user_fields.map(lambda fields: fields[2]).distinct().count()
# 统计职业数量
num_occupations = user_fields.map(lambda fields: fields[3]).distinct().count()
# 统计邮编数量
num_zipcodes = user_fields.map(lambda fields: fields[4]).distinct().count()

# Users: 943, genders: 2, occupations: 21, ZIP codes: 795
print("Users: %d, genders: %d, occupations: %d, ZIP codes: %d" % (num_users, num_genders, num_occupations, num_zipcodes))

# 绘制用户年龄分布的直方图
ages = user_fields.map(lambda x: int(x[1])).collect()
# bins: 区间数
# normed=True: 正则化直方图
plt.hist(ages, bins=20, color='lightblue', normed=True)
fig = plt.gcf()
fig.set_size_inches(16, 10)

# 绘制用户职业分布的条形图
count_by_occupation = user_fields.map(lambda fields: (fields[3], 1)).reduceByKey(lambda x, y: x + y).collect()
x_axis1 = np.array([c[0] for c in count_by_occupation])
y_axis1 = np.array([c[1] for c in count_by_occupation])
x_axis = x_axis1[np.argsort(y_axis1)]
y_axis = y_axis1[np.argsort(y_axis1)]

pos = np.arange(len(x_axis))
width = 1.0
ax = plt.axes()
ax.set_xticks(pos + (width / 2))
ax.set_xticklabels(x_axis)
plt.bar(pos, y_axis, width, color='lightblue')
plt.xticks(rotation=30)
fig = plt.gcf()
fig.set_size_inches(16, 10)

count_by_occupation2 = user_fields.map(lambda fields: fields[3]).countByValue()
print("Map-reduce approach:")
print(dict(count_by_occupation2))
print("")
print("countByValue approach:")
print(dict(count_by_occupation))