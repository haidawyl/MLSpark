package org.hdwyl

import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

import breeze.linalg.{DenseMatrix, DenseVector, csvwrite}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by wangyanl on 2019/7/16.
  */
object SparkDimensionReduction {

  def main(args: Array[String]) {
    val tenantPath = args(0)
    println(s"tenantPath = ${tenantPath}")

    val conf = new SparkConf()
    val sc = new SparkContext(conf)

    val path: String = s"hdfs://PATH/${tenantPath}/lfw/*"
    // wholeTextFiles将返回一个由键-值对组成的RDD，键是文件位置，值是整个文件的内容。
    val rdd = sc.wholeTextFiles(path)
    println(rdd.first())

    val files = rdd.map { case (fileName, content) => fileName }
    println(files.first())
    println(files.count())

    val aeImage = loadImageFromFile(files.first())
    println(aeImage)

    val grayImage = processImage(aeImage, 100, 100)
    println(grayImage)

    ImageIO.write(grayImage, "jpg", new File("hdfs://PATH/aeGray.jpg"))

    val pixels = files.map(f => extractPixels(f, 50, 50))
    println(pixels.take(10).map(_.take(10).mkString("", ",", ", ...")).mkString("\n"))

    // 为每一张图片创建MLlib向量对象
    val vectors = pixels.map(p => Vectors.dense(p))
    vectors.setName("image-vectors")
    vectors.cache()

    // 在运行降维模型尤其是PCA之前，通常会对输入数据进行标准化。
    // 对于稠密的输入数据可以提取平均值，但是对于稀疏数据，提取平均值将会使之变稠密。
    // 对于很高维度的输入，这将很可能耗尽可用内存资源，所以是不建议使用的。
    val scaler = new StandardScaler(withMean = true, withStd = false).fit(vectors)

    // 使用返回的scaler来转换原始的图像向量，让所有向量减去当前列的平均值
    val scaledVectors = vectors.map(v => scaler.transform(v))

    // 训练降维模型
    // MLlib中的降维模型需要向量作为输入。
    val matrix = new RowMatrix(scaledVectors)
    val K = 10
    val pc = matrix.computePrincipalComponents(K)

    // 可视化特征脸
    val rows = pc.numRows
    val cols = pc.numCols
    println(rows, cols)

    val pcBreeze = new DenseMatrix(rows, cols, pc.toArray)
    csvwrite(new File("hdfs://PATH/pc.csv"), pcBreeze)

    // 用矩阵乘法把图像矩阵和主成分矩阵相乘来实现投影
    val projected = matrix.multiply(pc)
    println(projected.numRows(), projected.numCols())

    println(projected.rows.take(5).mkString("\n"))

    // 在本例中，SVD计算产生的右奇异向量等同于我们计算得到的主成分。

    val svd = matrix.computeSVD(10, computeU = true)
    println(s"U dimension: (${svd.U.numRows}, ${svd.U.numCols})")
    println(s"S dimension: (${svd.s.size}, )")
    println(s"V dimension: (${svd.V.numRows}, ${svd.V.numCols})")

    println(approxEqual(Array(1.0, 2.0, 3.0), Array(1.0, 2.0, 3.0)))
    println(approxEqual(Array(1.0, 2.0, 3.0), Array(3.0, 2.0, 1.0)))
    println(approxEqual(svd.V.toArray, pc.toArray))

    val breezeS = DenseVector(svd.s.toArray)
    val projectedSVD = svd.U.rows.map { v =>
      val breezeV = DenseVector(v.toArray)
      // :* 运算符表示对向量执行对应元素和元素的乘法
      val multV = breezeV :* breezeS
      Vectors.dense(multV.data)
    }
    val cnt = projected.rows.zip(projectedSVD).map { case (v1, v2) =>
      approxEqual(v1.toArray, v2.toArray)
    }.filter(b => true).count()

    sc.stop()
  }

  /**
    * 从文件中读取图片
    *
    * @param path
    * @return
    */
  def loadImageFromFile(path: String): BufferedImage = {
    ImageIO.read(new File(path))
  }

  /**
    * 转换灰度和尺寸
    *
    * @param image
    * @param width
    * @param height
    * @return
    */
  def processImage(image: BufferedImage, width: Int, height: Int): BufferedImage = {
    // 创建一个指定宽、高和灰度模型的新图片
    val bwImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY)
    // 从原始图片绘制出灰度图片
    val g = bwImage.getGraphics
    // 颜色转换和尺寸变化
    g.drawImage(image, 0, 0, width, height, null)
    g.dispose()
    // 返回一个新的处理过的图片
    return bwImage
  }

  /**
    *
    * @param image
    * @return
    */
  def getPixelsFromImage(image: BufferedImage): Array[Double] = {
    val width = image.getWidth
    val height = image.getHeight
    val pixels = Array.ofDim[Double](width * height)
    image.getData.getPixels(0, 0, width, height, pixels)
  }

  /**
    *
    * @param path
    * @param width
    * @param height
    * @return
    */
  def extractPixels(path: String, width: Int, height: Int): Array[Double] = {
    val raw = loadImageFromFile(path)
    val processed = processImage(raw, width, height)
    getPixelsFromImage(processed)
  }


  def approxEqual(array1: Array[Double], array2: Array[Double], tolerance: Double = 1e-6): Boolean = {
    // note we ignore sign of the principal component / singular vector elements
    val bools = array1.zip(array2).map { case (v1, v2) => if (math.abs(math.abs(v1) - math.abs(v2)) > tolerance) false else true }
    bools.fold(true)(_ & _)
  }
}
