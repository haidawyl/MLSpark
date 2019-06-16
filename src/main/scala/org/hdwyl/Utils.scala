package org.hdwyl

import org.apache.spark.rdd.RDD

/**
  * Created by wangyanl on 2019/6/15.
  */
object Utils {
  def getMedian(data: RDD[Int]): Int = {
    val sortedData = data.sortBy(x => x).collect().toList
    val count = data.count().toInt
    val median: Int = if (count % 2 == 0) {
      val l = count / 2 - 1
      val r = l + 1
      (sortedData(l) + sortedData(r)) / 2
    } else {
      sortedData(count / 2 + 1)
    }

    return median
  }

}
