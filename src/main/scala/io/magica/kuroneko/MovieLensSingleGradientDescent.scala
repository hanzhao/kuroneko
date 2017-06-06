package io.magica.kuroneko

import breeze.linalg._
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable
import scala.util.Random

object MovieLensSingleGradientDescent {

  def main(args: Array[String]): Unit = {
    require(args.length == 5,
      """MovieLens Prediction needs 4 args.
        |  0. ratings.csv file path
        |  1. iteration (T)
        |  2. learning rate (gamma)
        |  3. regular value (lambda)
        |  4. latent factor rank (k)""".stripMargin)

    val conf = new SparkConf().setAppName("MovieLens prediction using Gradient Descent")
    val sc = new SparkContext(conf)

    val data = sc.textFile(args(0)).map(_.split(',') match {
      case Array(userId, productId, rating, timestamp) =>
        MovieLensRating(userId.toInt, productId.toInt, rating.toFloat, timestamp.toInt)
    })
    val splits = data.randomSplit(Array(0.8, 0.2), Config.SEED)
    val training = splits(0).collect()
    val testing = splits(1).collect()

    val model = new MovieLensSingleGradientDescentModal(args(4).toInt)
    model.train(training, args(1).toInt, args(2).toFloat, args(3).toFloat)

    val prediction = model.predict(testing.map { r => (r.user, r.product) })
    var se = 0.0
    0.until(testing.length).foreach { i =>
      se += (testing(i).rating - prediction(i).rating) * (testing(i).rating - prediction(i).rating)
    }
    val rmse = math.sqrt(se / testing.length)
    println(s"RMSE: $rmse")

    sc.stop()
  }

}

class MovieLensSingleGradientDescentModal(val rank: Int,
                                          val userFeatures: mutable.HashMap[Int, Vector[Float]] = new mutable.HashMap(),
                                          val productFeatures: mutable.HashMap[Int, Vector[Float]] = new mutable.HashMap()) {

  def train(ratings: Array[MovieLensRating], T: Int, gamma: Float, lambda: Float) = {
    0.until(T).foreach { _ =>
      Random.shuffle(ratings.toIterator).foreach { rating =>
        val uf = userFeatures.getOrElse(rating.user, MovieLensSingleGradientDescentUtil.randomVector(rank))
        val pf = productFeatures.getOrElse(rating.product, MovieLensSingleGradientDescentUtil.randomVector(rank))
        val e = rating.rating - (uf dot pf)
        val ug = gamma :* ((e :* pf) - (lambda :* uf))
        val pg = gamma :* ((e :* uf) - (lambda :* pf))
        userFeatures(rating.user) = uf + ug
        productFeatures(rating.product) = pf + pg
      }
    }
  }

  def predict(userProduct: (Int, Int)): Float = {
    val uf = userFeatures.get(userProduct._1)
    val pf = productFeatures.get(userProduct._2)
    if (uf.isDefined && pf.isDefined) {
      uf.get dot pf.get
    } else {
      3.5f
    }
  }

  def predict(userProducts: Array[(Int, Int)]): Array[MovieLensRating] = {
    userProducts.map { case (user, product) =>
      new MovieLensRating(user, product, predict((user, product)), 0)
    }
  }

}

object MovieLensSingleGradientDescentUtil extends Serializable {

  def randomVector(rank: Int) = Vector.fill(rank) {
    math.sqrt(math.random * 5 / rank).toFloat
  }

}
