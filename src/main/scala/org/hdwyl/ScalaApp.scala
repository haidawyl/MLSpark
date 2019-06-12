package org.hdwyl

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

/**
  * 用Scala编写的一个简单的Spark应用
  */
object ScalaApp {
  def main(args: Array[String]) {
    val sc = new SparkContext("local[2]", "First Spark App")
    // 将CSV格式的原始数据转化为(user,product,price)格式的记录集
    val data = sc.textFile("data/UserPurchaseHistory.csv")
      .map(line => line.split(","))
      .map(purchaseRecord => (purchaseRecord(0), purchaseRecord(1), purchaseRecord(2)))
    // 求购买次数
    val numPurchases = data.count()
    // 求有多少个不同客户购买过商品
    val uniqueUsers = data.map { case (user, product, price) => user }.distinct().count()
    // 求和得出总收入
    val totalRevenue = data.map { case (user, product, price) => price.toDouble }.sum()
    // 求最畅销的产品是什么
    val productsByPopularity = data
      .map { case (user, product, price) => (product, 1) }
      .reduceByKey(_ + _)
      .collect()
      .sortBy(_._2)
    val mostPopular = productsByPopularity(0)

    println("Total purchases: " + numPurchases)
    println("Unique users: " + uniqueUsers)
    println("Total revenue: " + totalRevenue)
    println("Most popular product: %s with %d purchases".format(mostPopular._1, mostPopular._2))
  }
}