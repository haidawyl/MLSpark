package org.hdwyl.kafka

import kafka.serializer.StringDecoder
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.streaming.kafka.KafkaUtils
import org.apache.spark.streaming.{Seconds, StreamingContext}


/**
  * Created by wangyanl on 2019/11/3.
  */
object KafkaDirector {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    val tenantPath = args(0)

    // 构建conf ssc 对象
    val conf = new SparkConf()
    val ssc = new StreamingContext(conf, Seconds(3))
    // 设置数据检查点进行累计统计单词
    ssc.checkpoint(s"hdfs://PATH/${tenantPath}/checkpoint")
    // kafka 需要Zookeeper 需要消费者组
    val topics = Set("mydemo2")
    // broker的原信息ip地址以及端口号
    val kafkaPrams = Map[String, String]("metadata.broker.list" -> "192.168.128.111:9092")
    // 数据的输入类型 数据的解码类型
    val data = KafkaUtils.createDirectStream[String, String, StringDecoder, StringDecoder](ssc, kafkaPrams, topics)
    val updateFunc = (curVal: Seq[Int], preVal: Option[Int]) => {
      // 进行数据统计当前值加上之前的值
      var total = curVal.sum
      // 最初的值应该是0
      var previous = preVal.getOrElse(0)
      // Some 代表最终的但会值
      Some(total + previous)
    }
    // 统计结果
    val result = data.map(_._2).flatMap(_.split(" ")).map(word => (word, 1)).updateStateByKey(updateFunc).print()
    // 启动程序
    ssc.start()
    ssc.awaitTermination()
  }
}
