#!/usr/bin/python
#  -*- coding:utf-8 -*-

from pyspark import SparkContext
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import datetime

sc = SparkContext()
# 读取数据集
movie_data = sc.textFile("hdfs://PATH/ml-100k/u.item")
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

movie_fields = movie_data.map(lambda line: line.split("|"))
# 提取出电影的年份信息
years = movie_fields.map(lambda fields: fields[2]).map(lambda x: convert_year(x))
# 过滤掉1900年(即未记录年份)的电影数据
years_filtered = years.filter(lambda x: x != 1900)
# 计算得到电影的年龄
today = datetime.date.today()
this_year = today.year
movie_ages = years_filtered.map(lambda year: this_year - year).countByValue()
values = movie_ages.values()
bins = movie_ages.keys()
# bins: 区间数
# normed=True: 正则化直方图
plt.hist(values, bins=bins, color='lightblue', normed=True)
#fig = plt.gcf()
#fig.set_size_inches(16,10)

# 年份集合
years_pre_processed = movie_fields.map(lambda fields: fields[2]).map(lambda x: convert_year(x)).collect()
years_pre_processed_array = np.array(years_pre_processed)
# 非1900年的全部年份的平均值
mean_year = np.mean(years_pre_processed_array[years_pre_processed_array!=1900])
# 非1900年的全部年份的中位数值
median_year = np.median(years_pre_processed_array[years_pre_processed_array!=1900])
# np.where(condition): 输出满足条件的元素的坐标
index_bad_data = np.where(years_pre_processed_array==1900)[0][0]
print("index_bad_data:")
print(index_bad_data)
years_pre_processed_array[index_bad_data] = median_year
# Mean year of release: 1989
print("Mean year of release: %d" % mean_year)
# Median year of release: 1995
print("Median year of release: %d" % median_year)
# Index of '1900' after assigning median: []
print("Index of '1900' after assigning median: %s" % np.where(years_pre_processed_array == 1900)[0])