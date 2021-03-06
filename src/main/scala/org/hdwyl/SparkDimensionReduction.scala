package org.hdwyl

import java.awt.image.BufferedImage
import java.io._
import javax.imageio.ImageIO

import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.commons.compress.archivers.tar.{TarArchiveEntry, TarArchiveInputStream}
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream
import org.apache.commons.lang3.StringUtils
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable

/**
  * 降维方法从一个D维的数据输入提取出k维表示，k一般远远小于D。
  * 因此，降维方法本身是一种预处理方法，或者说是一种特征转换的方法，而不是模型预测的方法。
  * MLlib提供两种相似的降低维度的模型：
  * PCA（Principal Components Analysis，主成分分析法）和
  * SVD（Singular Value Decomposition，奇异值分解法）。
  * PCA处理一个数据矩阵，抽取矩阵中k个主向量，主向量彼此不相关。
  * 计算结果中，第一个主向量表示输入数据的最大变化方向。
  * 之后的每个主向量依次代表不考虑之前计算过的所有方向时最大的变化方向。
  * 因此，返回的k个主成分代表了输入数据可能的最大变化。
  * SVD试图将一个m×n的矩阵分解为三个主成分矩阵：
  *  m×k维矩阵U
  *  k×k维对角阵S，S中的元素是奇异值
  *  k×n维矩阵V
  * U = U x S x VT
  * PCA和SVD都是矩阵分解技术，从某种意义上来说，它们都把原来的矩阵分解成一些维度（或秩）较低的矩阵。
  * 很多降维技术都是基于矩阵分解的。
  *
  * Created by wangyanl on 2019/7/16.
  */
object SparkDimensionReduction {

  def main(args: Array[String]) {
    val tenantPath = args(0)
    println(s"tenantPath = ${tenantPath}")

    val conf = new SparkConf()
    val sc = new SparkContext(conf)
    val fs = FileSystem.get(sc.hadoopConfiguration)

    val destPath: scala.Predef.String = s"hdfs://PATH/${tenantPath}/"
    val inputTar = new ByteArrayInputStream(sc.binaryFiles(s"hdfs://PATH/${tenantPath}/lfw-a.tgz").first()._2.toArray())
    // decompress(fs, destPath, inputTar)

    val path: scala.Predef.String = s"hdfs://PATH/${tenantPath}/lfw/*"
    // wholeTextFiles将返回一个由键-值对组成的RDD，键是文件位置，值是整个文件的内容（文本）。
    // val rdd = sc.wholeTextFiles(path)
    // binaryFiles将返回一个由键-值对组成的RDD，键是文件位置，值是整个文件的内容（二进制）。
    val rdd = sc.binaryFiles(path)

    // val contents = rdd.map { case (fileName, content) => content }
    // val in = new ByteArrayInputStream(contents.first().toArray())

    val imagesMap = decompress(inputTar)
    val contents = sc.parallelize(imagesMap.values.toList)
    val in = new ByteArrayInputStream(contents.first())

    val aeImage = loadImageFromInputStream(in)
    println(aeImage)

    val grayImage = processImage(aeImage, 100, 100)
    println(grayImage)

    val baos = new ByteArrayOutputStream()
    ImageIO.write(grayImage, "jpg", baos)

    val imageOutputPath: scala.Predef.String = s"hdfs://PATH/${tenantPath}/lfw/aeGray.jpg"
    println(s"imageOutputPath = ${imageOutputPath}")
    val out = fs.create(new Path(imageOutputPath))
    out.write(baos.toByteArray)
    out.close()

    val pixels = contents.map { c =>
      // val in = new ByteArrayInputStream(c.toArray())
      val in = new ByteArrayInputStream(c)
      extractPixels(in, 50, 50)
    }
    println(pixels.take(10).map(_.take(10).mkString("", ",", ", ...")).mkString("\n"))

    // 为每一张图片创建MLlib向量对象
    val vectors = pixels.map(p => Vectors.dense(p))
    vectors.setName("image-vectors")
    vectors.cache()

    // 在运行降维模型尤其是PCA之前，通常会对输入数据进行标准化。
    // 使用MLlib的特征包提供的内建StandardScaler函数进行标准化。
    // 对于稠密的输入数据可以提取平均值，但是对于稀疏数据，提取平均值将会使之变稠密。
    // 对于很高维度的输入，这将很可能耗尽可用内存资源，所以是不建议使用的。
    // 调用fit函数会触发基于RDD[Vector]的计算。
    val scaler = new StandardScaler(withMean = true, withStd = false).fit(vectors)

    // 使用返回的scaler来转换原始的图像向量，让所有向量减去当前列的平均值
    val scaledVectors = vectors.map(v => scaler.transform(v))

    // 训练降维模型
    // MLlib中的降维模型需要向量作为输入。
    // PCA和SVD的计算是通过提供基于RowMatrix的方法实现的。
    val matrix = new RowMatrix(scaledVectors)
    // 调用computePrincipalComponents来计算分布式矩阵的前k个主成分
    val K = 10
    val pc = matrix.computePrincipalComponents(K)
    // 此时便训练得到了PCA模型

    // WARN LAPACK: Failed to load implementation from: com.github.fommil.netlib.NativeSystemLAPACK
    // WARN LAPACK: Failed to load implementation from: com.github.fommil.netlib. NativeRefLAPACK
    // 上述警告是说MLlib使用的线性代数库不能加载本地库。这时，基于Java的备选库会被使用。

    // 可视化特征脸
    val rows = pc.numRows
    val cols = pc.numCols
    // 2500, 10: 主成分矩阵有2500行, 10列
    // 每张图片的维度是50×50，我们计算得到了前K个主成分向量，每一个向量的维度都和输入图片的维度一样。
    // 我们可以认为这些主成分是一组包含了原始数据主要变化的隐层（隐藏）特征。
    println(rows, cols)

    // 因为每一个主成分都和原始图像有相同维度，所以每一个成分本身可以看作是一张图像。

    val pcBreeze = new DenseMatrix(rows, cols, pc.toArray)
    val pcFile: scala.Predef.String = s"hdfs://PATH/${tenantPath}/lfw/pc"
    val pcPath = new Path(pcFile)
    if (fs.exists(pcPath)) {
      fs.delete(pcPath, true)
    }
    val pcArray = pcBreeze.toArray
    val pcMatArray = Array.ofDim[Double](rows, cols)
    for (i <- 0 until rows) {
      pcMatArray(i) = pcArray.slice(i * cols, (i + 1) * cols)
    }
    sc.parallelize(pcMatArray).coalesce(1).saveAsTextFile(pcFile)
    Utils.printMatrix(pcBreeze, cols)

    // 降维方法最终的目标是要得到数据更加压缩化的表示，并能包含原始数据之中重要的特征和变化。
    // 为了做到这一点，我们需要通过使用训练好的模型，把原始数据投影到用主成分表示的新的低维空间上。

    // 使用PCA投影数据
    // 用矩阵乘法把图像矩阵和主成分矩阵相乘来实现投影
    // matrix: 图像矩阵
    // pc: 主成分矩阵
    val projected = matrix.multiply(pc)
    // 1055, 10，每幅2500维度的图像已经被转换成为一个大小为10的向量。
    println(projected.numRows(), projected.numCols())

    // 这些以向量形式表示的投影后的数据可以用来作为另一个机器学习模型的输入。
    // 例如我们可以通过使用这些投影后的脸的投影数据和一些没有脸的图像产生的投影数据，共同训练一个面部识别模型。
    println(projected.rows.take(5).mkString("\n"))

    // PCA和SVD有着密切的联系，可以使用SVD恢复出相同的主成分向量，并且应用相同的投影矩阵投射到主成分空间。
    // 在本例中，SVD计算产生的右奇异向量等同于我们计算得到的主成分。
    // 可以通过在图像矩阵上计算SVD并比较右奇异向量和PCA的结果说明这一点。

    val svd = matrix.computeSVD(10, computeU = true)
    println(s"U dimension: (${svd.U.numRows}, ${svd.U.numCols})")
    println(s"S dimension: (${svd.s.size}, )")
    println(s"V dimension: (${svd.V.numRows}, ${svd.V.numCols})")

    // 矩阵V和PCA的结果完全一样（不考虑正负号和浮点数误差）。
    println(approxEqual(Array(1.0, 2.0, 3.0), Array(1.0, 2.0, 3.0))) // true
    println(approxEqual(Array(1.0, 2.0, 3.0), Array(3.0, 2.0, 1.0))) // false
    println(approxEqual(svd.V.toArray, pc.toArray)) // true

    // 另外一个相关性体现在：矩阵U和向量S（或者严格来讲，对角矩阵S）的乘积和
    // PCA中用来把原始图像数据投影到K个主成分构成的空间中的投影矩阵相等
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
    println(s"cnt = ${cnt}") // 1055
    // 上述计算结果表明投影后的每一行和SVD投影后的每一行相等。

    // PCA和SVD都是确定性模型，就是对于给定输入数据，总可以产生确定结果的模型。
    // 这两个模型都确定可以返回多个主成分或者奇异值，因此控制模型的唯一参数就是k。
    // 就像聚类模型，增加k总是可以提高模型的表现（对于聚类，表现在相对误差函数值；
    // 对于PCA和SVD，整体的不确定性表现在k个成分上）。
    // 因此，选择k的值需要折中，看是要包含尽量多的数据的结构信息，还是要保持投影数据的低维度。

    // 在LFW数据集上估计SVD的k值
    // 通过观察在图像数据集上计算SVD得到的奇异值，可以确定奇异值每次运行结果相同，并且是按照递减的顺序返回的。
    val sValues = (1 to 5).map {
      i => matrix.computeSVD(i, computeU = false).s
    }
    sValues.foreach(println)

    // 为了估算SVD（和PCA）做聚类时的k值，以一个较大的k的变化范围绘制一个奇异值图是很有用的。
    // 可以看到每增加一个奇异值时增加的变化总量是否基本保持不变。
    // 计算最大的300个奇异值
    val svd300 = matrix.computeSVD(300, computeU = false)
    val sMatrix = new DenseMatrix(1, 300, svd300.s.toArray)
    val sFile: scala.Predef.String = s"hdfs://PATH/${tenantPath}/lfw/s"
    val sPath = new Path(sFile)
    if (fs.exists(sPath)) {
      fs.delete(sPath, true)
    }
    sc.parallelize(sMatrix.toArray).coalesce(1).saveAsTextFile(sFile)
    Utils.printMatrix(sMatrix, 300)

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
    * 从输入流中读取图片
    *
    * @param input
    * @return
    */
  def loadImageFromInputStream(input: InputStream): BufferedImage = {
    ImageIO.read(input)
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
    * 提取图片的像素值
    *
    * @param image
    * @return
    */
  def getPixelsFromImage(image: BufferedImage): Array[Double] = {
    // 图片宽度
    val width = image.getWidth
    // 图片高度
    val height = image.getHeight
    // 创建宽度*高度大小的Double类型数组
    val pixels = Array.ofDim[Double](width * height)
    // 提取图片的像素值
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

  /**
    *
    * @param input
    * @param width
    * @param height
    * @return
    */
  def extractPixels(input: InputStream, width: Int, height: Int): Array[Double] = {
    val raw = loadImageFromInputStream(input)
    val processed = processImage(raw, width, height)
    getPixelsFromImage(processed)
  }

  /**
    * 比较两个矩阵的向量数据
    *
    * @param array1
    * @param array2
    * @param tolerance
    * @return
    */
  def approxEqual(array1: Array[Double], array2: Array[Double], tolerance: Double = 1e-6): Boolean = {
    // note we ignore sign of the principal component / singular vector elements
    val bools = array1.zip(array2).map { case (v1, v2) => if (math.abs(math.abs(v1) - math.abs(v2)) > tolerance) false else true }
    bools.fold(true)(_ & _)
  }

  /**
    * 解压文件
    *
    * @param fs
    * @param destPath
    * @param inputTar
    */
  def decompress(fs: FileSystem, destPath: String, inputTar: InputStream) = {
    val tar = new TarArchiveInputStream(new GzipCompressorInputStream(inputTar))
    var entry: TarArchiveEntry = null

    do {
      entry = tar.getNextEntry().asInstanceOf[TarArchiveEntry]
      if (entry != null) {
        val fileName = entry.getName()
        // println(s"fileName = ${fileName}")
        val filePath = new Path(destPath + fileName)
        if (StringUtils.endsWithIgnoreCase(fileName, ".jpg")) {
          val out = fs.create(filePath)
          val byteFile = Array.ofDim[Byte](entry.getSize.toInt)
          tar.read(byteFile)
          out.write(byteFile)
          out.close()
        } else {
          if (!fs.exists(filePath)) {
            fs.mkdirs(filePath)
          }
        }
      }
    } while (entry != null)
  }

  /**
    * 解压文件
    *
    * @param inputTar
    */
  def decompress(inputTar: InputStream): mutable.Map[String, Array[Byte]] = {
    val imageMap = mutable.Map[String, Array[Byte]]()

    val tar = new TarArchiveInputStream(new GzipCompressorInputStream(inputTar))
    var entry: TarArchiveEntry = null

    do {
      entry = tar.getNextEntry().asInstanceOf[TarArchiveEntry]
      if (entry != null) {
        val fileName = entry.getName()
        // println(s"fileName = ${fileName}")
        if (StringUtils.endsWithIgnoreCase(fileName, ".jpg")) {
          val byteFile = Array.ofDim[Byte](entry.getSize.toInt)
          tar.read(byteFile)
          imageMap += (fileName -> byteFile)
        } else {
        }
      }
    } while (entry != null)

    imageMap
  }

}
