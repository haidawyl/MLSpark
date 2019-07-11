#!/usr/bin/python
#  -*- coding:utf-8 -*-

from pyspark import SparkContext
from pyspark.mllib.feature import Normalizer
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

def extract_title(raw):
    import re
    # 该表达式找寻括号之间的非单词（数字）
    grps = re.search("\((\w+)\)", raw)
    if grps:
        # 只选取标题部分，并删除末尾的空白字符
        return raw[:grps.start()].strip()
    else:
        return raw

raw_titles = movie_fields.map(lambda fields: fields[1])
for raw_title in raw_titles.take(5):
    print extract_title(raw_title)

movie_titles = raw_titles.map(lambda m: extract_title(m))
# 下面用简单空白分词法将标题分词为词
title_terms = movie_titles.map(lambda t: t.split(" "))
print title_terms.take(5)

# 下面取回所有可能的词，以便构建一个词到序号的映射字典
all_terms = title_terms.flatMap(lambda x: x).distinct().collect()
# 创建一个新的字典来保存词，并分配k之1序号
idx = 0
all_terms_dict = {}
for term in all_terms:
    all_terms_dict[term] = idx
    idx +=1

# Total number of terms: 2645
print "Total number of terms: %d" % len(all_terms_dict)
# Index of term 'Dead': 147
print "Index of term 'Dead': %d" % all_terms_dict['Dead']
# Index of term 'Rooms': 1963
print "Index of term 'Rooms': %d" % all_terms_dict['Rooms']

all_terms_dict2 = title_terms.flatMap(lambda x: x).distinct().zipWithIndex().collectAsMap()
# Index of term 'Dead': 147
print "Index of term 'Dead': %d" % all_terms_dict2['Dead']
# Index of term 'Rooms': 1963
print "Index of term 'Rooms': %d" % all_terms_dict2['Rooms']

# 该函数输入一个词列表，并用k之1编码类似的方式将其编码为一个scipy稀疏向量
def create_vector(terms, term_dict):
    from scipy import sparse as sp
    num_terms = len(term_dict)
    x = sp.csc_matrix((1, num_terms))
    for t in terms:
        if t in term_dict:
            idx = term_dict[t]
            x[0, idx] = 1
    return x

all_terms_bcast = sc.broadcast(all_terms_dict)
term_vectors = title_terms.map(lambda terms: create_vector(terms, all_terms_bcast.value))
term_vectors.take(5)

# 将随机种子的值设为42，以保证每次运行的结果相同
np.random.seed(42)
x = np.random.randn(10)
norm_x_2 = np.linalg.norm(x)
normalized_x = x / norm_x_2
print "x:\n%s" % x
print "2-Norm of x: %2.4f" % norm_x_2
print "Normalized x:\n%s" % normalized_x
print "2-Norm of normalized_x: %2.4f" % np.linalg.norm(normalized_x)

normalizer = Normalizer()
vector = sc.parallelize([x])
normalized_x_mllib = normalizer.transform(vector).first().toArray()
print "x:\n%s" % x
print "2-Norm of x: %2.4f" % norm_x_2
print "Normalized x MLlib:\n%s" % normalized_x_mllib
print "2-Norm of normalized_x_mllib: %2.4f" % np.linalg.norm(normalized_x_mllib)
