package org.apache.spark.examples.mllib

import scopt.OptionParser
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{DenseMatrix, Vectors}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, _}
import breeze.linalg.{Matrix => BM, DenseMatrix => BDM}

object CosineSimilarityRecommender {
  case class Params(inputFile: String = "data/mahout_in_action.csv", threshold: Double = 0.0,
                    numRecommendations: Int = 2)
    extends AbstractParams[Params]

  def toBreeze(dm:DistributedMatrix):BM[Double] = dm match {
    case cm:CoordinateMatrix => {
      val m = cm.numRows().toInt
      val n = cm.numCols().toInt
      val mat = BDM.zeros[Double](m, n)
      cm.entries.collect().foreach { case MatrixEntry(i, j, value) =>
        mat(i.toInt, j.toInt) = value
      }
      mat
    }
    case im:IndexedRowMatrix => {
      val m = im.numRows().toInt
      val n = im.numCols().toInt
      val mat = BDM.zeros[Double](m, n)
      im.rows.collect().foreach { case IndexedRow(rowIndex, vector) =>
        val i = rowIndex.toInt
        vector.foreachActive { case (j, v) =>
          mat(i, j) = v
        }
      }
      mat
    }
    case _ =>  sys.error("invalid type")
  }

  def printMat(mat:BM[Double]) = {
    print("            ")
    for(j <- 0 to mat.cols-1) print("%-10d".format(j));
    println
    for(i <- 0 to mat.rows-1) {

      print("%-6d".format(i))
      val vec = for(j <- 0 to mat.cols-1) {
        print(" %9.3f".format(mat(i, j)))
      }
      println
    }
  }

  def main(args: Array[String]) {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("CosineSimilarity") {
      head("CosineSimilarity: an example app.")
      opt[Double]("threshold")
        .required()
        .text(s"threshold similarity: to tradeoff computation vs quality estimate")
        .action((x, c) => c.copy(threshold = x))
      arg[String]("<inputFile>")
        .required()
        .text(s"input file, one row per line, space-separated")
        .action((x, c) => c.copy(inputFile = x))
      note(
        """
          |For example, the following command runs this app on a dataset:
          |
          | ./bin/spark-submit  --class org.apache.spark.examples.mllib.CosineSimilarityRecommender \
          | examplesjar.jar \
          | --threshold 0.1 data/mahout_in_action.csv
        """.stripMargin)
    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(params: Params) {
    val conf = new SparkConf().setAppName("CosineSimilarity")
    val sc = new SparkContext(conf)
    val lines = sc.textFile(params.inputFile)
    val rdd = lines.map(x => Vectors.dense(x.split(",").map(_.trim().toDouble))).cache()

    // item-item類似度
    val irm = new IndexedRowMatrix(rdd.zipWithIndex.map{ case(value, index) => IndexedRow(index, value) })
    val itemSimilarity = irm.toRowMatrix().columnSimilarities(params.threshold)
    val userLog = irm.toBlockMatrix.toLocalMatrix.transpose

    // item-item類似度から評価行列Rを作る(上三角行列から対称行列を作成して内積を計算したかったがAPIが見つからなかった）
    // (5)式: https://cran.r-project.org/web/packages/recommenderlab/vignettes/recommenderlab.pdf
    val result1  = toBreeze(itemSimilarity.toIndexedRowMatrix.multiply(userLog))
    val result2  = toBreeze(itemSimilarity.transpose.toIndexedRowMatrix.multiply(userLog))
    val R = result1 + result2

    // mahout仕様に対応。userLogに存在するものはRから除外(->0に強制変換）してmaskedRを作成
    val newFeatures = new BDM(userLog.numCols, userLog.numRows, userLog.asInstanceOf[DenseMatrix].values.map {
      case 1.0 => 0.0
      case 0.0 => 1.0
    })
    val maskedR = R.toDenseMatrix :* newFeatures.t

    // maskedR を使ってスコア順トップNを表示する
    val N = params.numRecommendations
    printMat(maskedR)
    println("-------------")
    println("Top %s Similarity".format(N))
    val nRows = maskedR.rows -1
    val nCols = maskedR.cols -1
    for (i <- 0 to nRows) {
      val vec = for (j <- 0 to nCols) yield maskedR(i, j)
      print("%-6d".format(i))
      println(vec.sorted.reverse.filter { _ > 0.1 }.take(N))
    }

    sc.stop()
  }
}
