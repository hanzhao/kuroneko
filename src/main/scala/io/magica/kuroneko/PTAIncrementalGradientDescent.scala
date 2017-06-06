package io.magica.kuroneko

import breeze.linalg._
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.util.Random

object PTAIncrementalGradientDescent {

  def main(args: Array[String]): Unit = {
    require(args.length == 7,
      """PTA Prediction needs 6 args.
        |  0. record.csv file path
        |  1. iteration (T)
        |  2. learning rate u (gamma_u)
        |  3. learning rate i (gamma_i)
        |  4. regular value (lambda)
        |  5. latent factor rank (k)
        |  6. recall@R (r)""".stripMargin)

    val conf = new SparkConf().setAppName("PTA prediction using Gradient Descent")
    val sc = new SparkContext(conf)

    val ratings = sc.textFile(args(0)).map(_.split(',') match {
      case Array(rating, userId, productId, timestamp) =>
        PTARating(userId.toInt, productId.toInt, if (rating == "G") {
          1.0f
        } else {
          0.0f
        }, timestamp.toLong)
    }).collect()

    val model = new PTAIncrementalGradientDescentModel(args(5).toInt)

    val prediction = model.predictAndLearn(ratings, args(1).toInt, args(2).toFloat, args(3).toFloat, args(4).toFloat, args(6).toInt)
    var recallSum = 0.0
    var roundRecallSum = 0.0
    0.until(prediction.length).foreach { i =>
      val recall = prediction(i)
      recallSum += recall
      roundRecallSum += recall
      if (i % 500 == 499) {
        println(s"Round: ${(i + 1) / 500}, Recall: ${roundRecallSum / 500}")
        roundRecallSum = 0.0
      }
    }
    println(s"Recall: ${recallSum / prediction.length}")

    sc.stop()
  }

}

class PTAIncrementalGradientDescentModel(val rank: Int,
                                         val userFeatures: mutable.HashMap[Int, Vector[Float]] = new mutable.HashMap(),
                                         val productFeatures: mutable.HashMap[Int, Vector[Float]] = new mutable.HashMap()) {

  def train(ratings: Array[PTARating], T: Int, gammaU: Float, gammaI: Float, lambda: Float): Unit = {
    0.until(T).foreach { _ =>
      Random.shuffle(ratings.toIterator).foreach { rating =>
        train(rating, 1, gammaU, gammaI, lambda)
      }
    }
  }

  def train(rating: PTARating, T: Int, gammaU: Float, gammaI: Float, lambda: Float): Unit = {
    0.until(T).foreach { _ =>
      val uf = userFeatures.getOrElse(rating.user, PTAIncrementalGradientDescentUtil.randomVector(rank))
      val pf = productFeatures.getOrElse(rating.product, PTAIncrementalGradientDescentUtil.randomVector(rank))
      val e = rating.rating - (uf dot pf)
      val ug = gammaU :* ((e :* pf) - (lambda :* uf))
      val pg = gammaI :* ((e :* uf) - (lambda :* pf))
      userFeatures(rating.user) = uf + ug
      productFeatures(rating.product) = pf + pg
    }
  }

  def evaluateRecall(user: Int, k: Int, past: Array[Int], future: Array[Int]) = {
    val uf = userFeatures.get(user)
    if (uf.isEmpty) {
      0.0f
    } else if (future.size == 0) {
      2.0f
    } else {
      val recList = productFeatures.mapValues(x => x dot uf.get).toList.sortBy(_._2)
        .map(_._1).filter(x => !past.contains(x))
        .reverse.slice(0, k)
      println(s"pred: ${recList.mkString(",")}, future: ${future.mkString(",")}")
      recList.filter(x => future.contains(x)).length.toFloat / future.length
    }
  }

  def predictAndLearn(userProducts: Array[PTARating], T: Int, gammaU: Float, gammaI: Float, lambda: Float, r: Int): Array[Float] = {
    userProducts.zipWithIndex.map { case (rating, index) =>
      val past = PTAIncrementalGradientDescentUtil.removeDuplicates(
        userProducts.slice(0, index).filter(_.user == rating.user).map(_.product)
      )
      val future = PTAIncrementalGradientDescentUtil.removeDuplicates(
        userProducts.slice(index, userProducts.length).filter(x => x.user == rating.user && x.rating > 0.9f).map(_.product)
      )
      val result = evaluateRecall(rating.user, r, past, future)
      train(rating, T, gammaU, gammaI, lambda)
      result
    }.filter(_ < 1.1)
  }

}

object PTAIncrementalGradientDescentUtil extends Serializable {

  def randomVector(rank: Int) = Vector.fill(rank) {
    math.sqrt(math.random / rank).toFloat
  }

  def removeDuplicates(a: Array[Int]): Array[Int] = {
    val res = ListBuffer[Int]()
    val hash = new mutable.HashSet[Int]()
    a.foreach { x =>
      if (!hash.contains(x)) {
        hash.add(x)
        res += x
      }
    }
    res.toArray
  }

}
