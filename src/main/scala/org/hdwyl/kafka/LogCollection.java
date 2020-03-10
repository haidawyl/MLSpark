package org.hdwyl.kafka

import java.sql.SQLException

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.hbase.HBaseConfiguration
import org.apache.hadoop.hbase.client._
import org.apache.hadoop.hbase.io.ImmutableBytesWritable
import org.apache.hadoop.hbase.mapred.TableOutputFormat
import org.apache.hadoop.hbase.util.Bytes
import org.apache.hadoop.hive.ql.security.authorization.plugin.HiveAccessControlException
import org.apache.hadoop.mapred.JobConf
import org.apache.log4j.{Level, Logger}
import org.apache.spark.streaming.kafka010.{ConsumerStrategies, KafkaUtils, LocationStrategies}
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.{SparkConf, SparkContext}
import org.json4s.DefaultFormats

object LogCollection {
   @transient lazy val logger = Logger.getLogger(this.getClass)
   
   // throw exception
   @throws[Exception]
   @throws[SQLException]
   @throws[HiveAccessControlException]
   @throws[IndexOutOfBoundsException]
   @throws[ArrayIndexOutOfBoundsException]
   @throws[NullPointerException]
   def main(args: Array[String]): Unit = {
       Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
       Logger.getLogger("org.eclipse.jetty.sever").setLevel(Level.OFF)
       Logger.getLogger("org.hdwyl").setLevel(Level.INFO)
       
       val topics = args(0) // kafka topics
       val brokers = args(1) // kafka brokers
       val groupId = args(2) // kafka group id
       val tableName = args(3) // hbase table
       try {
           val batchTime: Int = 1
           val sparkConf = new SparkConf().setAppName("LogCollection")
           sparkConf.set("spark.streaming.stopGracefullyOnShutdown", true)
           sparkConf.set("spark.streaming.kafka.maxRatePerPartition", "3000")
           
           // kafka configuration
           val topicSet = Set(topics)
           val kafkaParams = Map[String, String](
             "bootstrap.servers" -> brokers,
             "key.deserializer" -> "org.apache.kafka.common.serialization.StringDeserializer",
             "value.deserializer" -> "org.apache.kafka.common.serialization.StringDeserializer",
             "group.id" -> groupId,
             "security.protocol" -> "PLAINTEXT",
             "sasl.kerberos.service.name" -> "kafka",
             "auto.offset.reset" -> "latest"
           )
           
           // spark initialization
           val sc = new SparkContext(sparkConf)
           val ssc = new StreamingContext(sc, Seconds(batchTime.toLong))
           val locationStrategy = LocationStrategies.PreferConsistent
           val consumerStrategy = ConsumerStrategies.Subscribe[String, String](topicSet, kafkaParams)
           val stream = KafkaUtils.createDirectStream[String, String](ssc, locationStrategy, consumerStrategy)
           
           // hbase configuration
           val hbaseConf: Configuration = HBaseConfiguration.create()
           hbaseConf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
           
           val jobConf = new JobConf(hbaseConf)
           jobConf.setOutputFormat(classOf[TableOutputFormat])
           jobConf.set(TableOutputFormat.OUTPUT_TABLE, talbeName)
           
       }
   }
}
