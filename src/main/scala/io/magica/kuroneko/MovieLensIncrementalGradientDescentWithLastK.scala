package io.magica.kuroneko

import breeze.linalg._
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable
import scala.util.Random

object MovieLensIncrementalGradientDescentWithLastK {

  def main(args: Array[String]): Unit = {
    require(args.length == 6,
      """MovieLens Prediction needs 6 args.
        |  0. ratings.csv file path
        |  1. ratings count to review (k)
        |  2. learning rate u (gamma_u)
        |  3. learning rate i (gamma_i)
        |  4. regular value (lambda)
        |  5. latent factor rank (k)""".stripMargin)

    val conf = new SparkConf().setAppName("MovieLens prediction using Gradient Descent")
    val sc = new SparkContext(conf)

    val ratings = sc.textFile(args(0)).map(_.split(',') match {
      case Array(userId, productId, rating, timestamp) =>
        MovieLensRating(userId.toInt, productId.toInt, rating.toFloat, timestamp.toInt)
    }).collect()

    // use [0, i) blocks to predict i block.
    val tail1 = (ratings.length / 50.0 * 1).toInt
    val tail2 = ratings.length
    println(s"tail1: $tail1, tail2: $tail2")
    val training = ratings.slice(0, tail1)
    val testing = ratings.slice(tail1, tail2)

    val model = new MovieLensIncrementalGradientDescentWithLastKModel(args(5).toInt, args(1).toInt)
    model.train(training, args(2).toFloat, args(3).toFloat, args(4).toFloat)

    val prediction = model.predictAndLearn(testing, args(2).toFloat, args(3).toFloat, args(4).toFloat)
    var se = 0.0
    var count = 0
    var roundse = 0.0
    0.until(testing.length).foreach { i =>
      val e = (testing(i).rating - prediction(i).rating) * (testing(i).rating - prediction(i).rating)
      se += e
      roundse += e
      if (prediction(i).rating == 3.5f) {
        count += 1
      }
      if (i % 400005 == 400004) {
        val rmse = math.sqrt(roundse / 400005)
        println(s"Round: ${(i + 1) / 400005}, RMSE: $rmse")
        roundse = 0.0
      }
    }
    val rmse = math.sqrt(se / testing.length)
    println(s"RMSE: $rmse, unknown: ${count}/${testing.length}")

    sc.stop()
  }

}

class MovieLensIncrementalGradientDescentWithLastKModel(val rank: Int,
                                                        val k: Int,
                                                        val userFeatures: mutable.HashMap[Int, Vector[Float]] = new mutable.HashMap(),
                                                        val productFeatures: mutable.HashMap[Int, Vector[Float]] = new mutable.HashMap(),
                                                        val lastRatings: mutable.HashMap[Int, mutable.Queue[MovieLensRating]] = new mutable.HashMap()) {

  private def addRatingToQueue(user: Int, rating: MovieLensRating, queueSize: Int): Unit = {
    val queue = lastRatings.getOrElseUpdate(user, new mutable.Queue[MovieLensRating]())
    queue.enqueue(rating)
    if (queue.size > queueSize) {
      queue.dequeue()
    }
  }

  def train(ratings: Array[MovieLensRating], gammaU: Float, gammaI: Float, lambda: Float): Unit = {
    ratings.foreach {
      rating => train(rating, gammaU, gammaI, lambda)
    }
  }

  def train(rating: MovieLensRating, gammaU: Float, gammaI: Float, lambda: Float): Unit = {
    addRatingToQueue(rating.user, rating, k)
    0.until(2).foreach { _ =>
      lastRatings(rating.user).reverseIterator.foreach { rating =>
        val uf = userFeatures.getOrElse(rating.user, MovieLensIncrementalGradientDescentWithLastKUtil.randomVector(rank))
        val pf = productFeatures.getOrElse(rating.product, MovieLensIncrementalGradientDescentWithLastKUtil.randomVector(rank))
        val e = rating.rating - (uf dot pf)
        val ug = gammaU :* ((e :* pf) - (lambda :* uf))
        val pg = gammaI :* ((e :* uf) - (lambda :* pf))
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

  def predictAndLearn(userProducts: Array[MovieLensRating], gammaU: Float, gammaI: Float, lambda: Float): Array[MovieLensRating] = {
    userProducts.map { rating =>
      val result = new MovieLensRating(rating.user, rating.product, predict((rating.user, rating.product)), 0)
      train(rating, gammaU, gammaI, lambda)
      result
    }
  }

}

object MovieLensIncrementalGradientDescentWithLastKUtil extends Serializable {

  def randomVector(rank: Int) = Vector.fill(rank) {
    math.sqrt(math.random * 5 / rank).toFloat
  }

}
