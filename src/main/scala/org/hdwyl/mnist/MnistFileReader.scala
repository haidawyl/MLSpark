package org.hdwyl.mnist

import java.io.{DataInputStream, FileInputStream}
import java.util.zip.GZIPInputStream

/**
  * Created by wangyanl on 2019/6/22.
  */
class MnistFileReader(path: String) {

  protected[this] var stream: DataInputStream = null
  if (path.endsWith(".gz")) {
    stream = new DataInputStream(new GZIPInputStream(new FileInputStream(path)))
  } else {
    stream = new DataInputStream(new FileInputStream(path))
  }

}
