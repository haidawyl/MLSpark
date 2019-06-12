#!/usr/bin/python
#  -*- coding:utf-8 -*-

from pyspark import SparkContext
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sc = SparkContext()
# 读取数据集
movie_data = sc.textFile("hdfs:/PATH/ml-100k/u.item")
# 输出第1行数据
print(movie_data.first())
# 输出前k行数据
print(movie_data.take(10))

# 统计电影数量
num_movies = movie_data.count()
# Movies: 1682
print("Movies: %d" % num_movies)

# 转换年份函数
def convert_year(x):
    try:
        return int(x[-4:])
    except:
        return 1900 # 若数据缺失年份则将其年份设为1900

movie_fields = movie_data.map(lambda lines: lines.split("|"))
years = movie_fields.map(lambda fields: fields[2]).map(lambda x: convert_year(x))
# 过滤掉1900年的数据
years_filtered = years.filter(lambda x: x != 1900)

movie_ages = years_filtered.map(lambda yr: 2019 - yr).countByValue()
values = movie_ages.values()
bins = movie_ages.keys()
# bins: 区间数
# normed=True: 正则化直方图
plt.hist(values, bins=bins, color='lightblue', normed=True)
fig = plt.gcf()
fig.set_size_inches(16,10)