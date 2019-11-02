package org.hdwyl.flink

import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.api.java.functions.FunctionAnnotation.ForwardedFields
import org.apache.flink.api.java.utils.ParameterTool
import org.apache.flink.api.scala._
import org.apache.flink.configuration.Configuration

import scala.collection.JavaConverters._

/**
  * Created by wangyanl on 2019/11/1.
  */
object KMeans {

  def main(args: Array[String]) {

    // checking input parameters
    val params: ParameterTool = ParameterTool.fromArgs(args)

    // set up execution environment
    val env: ExecutionEnvironment = ExecutionEnvironment.getExecutionEnvironment

    // get input data:
    // read the points and centroids from the provided paths or fall back to default data
    val points: DataSet[Point] = getPointDataSet(params, env)
    val centroids: DataSet[Centroid] = getCentroidDataSet(params, env)

    val finalCentroids = centroids.iterate(params.getInt("iterations", 10)) {
      currentCentroids =>
        val newCentroids = points
          .map(new SelectNearestCenter).withBroadcastSet(currentCentroids, "centroids")
          .map { x => (x._1, x._2, 1L) }.withForwardedFields("_1; _2")
          .groupBy(0)
          .reduce { (p1, p2) => (p1._1, p1._2.add(p2._2), p1._3 + p2._3) }.withForwardedFields("_1")
          .map { x => new Centroid(x._1, x._2.div(x._3)) }.withForwardedFields("_1->id")
        newCentroids
    }

    val clusteredPoints: DataSet[(Int, Point)] =
      points.map(new SelectNearestCenter).withBroadcastSet(finalCentroids, "centroids")

    if (params.has("output")) {
      clusteredPoints.writeAsCsv(params.get("output"), "\n", " ")
      env.execute("Scala KMeans Example")
    } else {
      println("Printing result to stdout. Use --output to specify output path.")
      clusteredPoints.print()
    }
  }

  def getPointDataSet(params: ParameterTool, env: ExecutionEnvironment): DataSet[Point] = {
    if (params.has("points")) {
      env.readCsvFile[Point](
        params.get("points"),
        fieldDelimiter = " ",
        includedFields = Array(0, 1)
      )
    } else {
      println("Executing K-Means example with default points data set.")
      println("Use --points to specify file input.")
      env.fromCollection(KMeansData.POINTS.map {
        case Array(x, y) => new Point(x.asInstanceOf[Double], y.asInstanceOf[Double])
      })
    }
  }

  def getCentroidDataSet(params: ParameterTool, env: ExecutionEnvironment): DataSet[Centroid] = {
    if (params.has("centroids")) {
      env.readCsvFile[Centroid](
        params.get("centroids"),
        fieldDelimiter = " ",
        includedFields = Array(0, 1, 2)
      )
    } else {
      println("Executing K-Means example with default centroid data set.")
      println("Use --centroids to specify file input.")
      env.fromCollection(KMeansData.CENTROIDS.map {
        case Array(id, x, y) => new Centroid(id.asInstanceOf[Int], x.asInstanceOf[Double], y.asInstanceOf[Double])
      })
    }
  }

  /**
    * Common trait for operations supported by both points and centroids
    * Note: case class inheritance is not allowed in Scala
    */
  trait Coordinate extends Serializable {
    var x: Double
    var y: Double

    def add(other: Coordinate): this.type = {
      x += other.x
      y += other.y
      this
    }

    def div(other: Long): this.type = {
      x /= other
      y /= other
      this
    }

    def euclideanDistance(other: Coordinate): Double = Math.sqrt((x - other.x) * (x - other.x) + (y - other.y) * (y - other.y))

    def clear(): Unit = {
      x = 0
      y = 0
    }

    override def toString: String = s"$x $y"

  }

  /**
    * A simple two-dimensional point.
    */
  case class Point(var x: Double = 0, var y: Double = 0) extends Coordinate

  /**
    * A simple two-dimensional centroid, basically a point with an ID.
    */
  case class Centroid(var id: Int = 0, var x: Double = 0, var y: Double = 0) extends Coordinate {

    def this(id: Int, p: Point) {
      this(id, p.x, p.y)
    }

    override def toString: String = s"$id ${super.toString}"

  }

  /**
    * Determines the closest cluster center for a data point.
    */
  @ForwardedFields(Array("*->_2"))
  final class SelectNearestCenter extends RichMapFunction[Point, (Int, Point)] {
    private var centroids: Traversable[Centroid] = null

    /** Reads the centroid values from a broadcast variable into a collection. */
    override def open(parameters: Configuration) {
      centroids = getRuntimeContext.getBroadcastVariable[Centroid]("centroids").asScala
    }

    def map(p: Point): (Int, Point) = {
      var minDistance: Double = Double.MaxValue
      var closestCentroidId: Int = -1
      for (centroid <- centroids) {
        val distance = p.euclideanDistance(centroid)
        if (distance < minDistance) {
          minDistance = distance
          closestCentroidId = centroid.id
        }
      }
      (closestCentroidId, p)
    }
  }

}
