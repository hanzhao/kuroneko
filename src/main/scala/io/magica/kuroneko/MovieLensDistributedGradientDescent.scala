package io.magica.kuroneko

import breeze.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.redisson.Redisson
import org.redisson.codec.KryoCodec
import org.redisson.config.{Config => RedissionConfig}

import scala.util.Random

object MovieLensDistributedGradientDescent {

  def main(args: Array[String]): Unit = {
    require(args.length == 5,
      """MovieLens Prediction needs 4 args.
        |  0. ratings.csv file path
        |  1. iteration (T)
        |  2. learning rate (gamma)
        |  3. regular value (lambda)
        |  4. latent factor rank (k)""".stripMargin)

    val conf = new SparkConf().setAppName("MovieLens prediction using Distributed" +
      " Gradient Descent")
    val sc = new SparkContext(conf)

    val data = sc.textFile(args(0)).map(_.split(',') match {
      case Array(userId, productId, rating, timestamp) =>
        MovieLensRating(userId.toInt, productId.toInt, rating.toFloat, timestamp.toInt)
    })
    val splits = data.randomSplit(Array(0.8, 0.2), Config.SEED)
    val training = splits(0).coalesce(sc.defaultParallelism, true).cache()
    val testing = splits(1).cache()

    val startAt = System.currentTimeMillis()
    val model = new MovieLensDistributedGradientDescentModal(args(4).toInt)
    model.train(training, args(1).toInt, args(2).toFloat, args(3).toFloat)
    println(s"Training cost: ${System.currentTimeMillis() - startAt}ms")

    val prediction = model.predict(testing.map { r => (r.user, r.product) })
    val result = testing.map(r => ((r.user, r.product), r.rating))
      .join(prediction.map(r => ((r.user, r.product), r.rating))).map { case (_, (r1, r2)) =>
      (1, (r1 - r2) * (r1 - r2))
    }.treeReduce { (x, y) =>
      (x._1 + y._1, x._2 + y._2)
    }

    val rmse = math.sqrt(result._2 / result._1)
    println(s"RMSE: $rmse")

    sc.stop()
    MovieLensDistributedGradientDescentUtil.shutdown()
  }

}

class MovieLensDistributedGradientDescentModal(val rank: Int) extends Serializable {

  def train(ratings: RDD[MovieLensRating], T: Int, gamma: Float, lambda: Float) = {
    val rank = this.rank
    0.until(T).foreach { _ =>
      ratings.foreachPartition { partition =>
        Random.shuffle(partition).foreach { rating =>
          val ufpf = MovieLensDistributedGradientDescentUtil.getFeature(rating.user, rating.product)
          val uf = ufpf._1.getOrElse(MovieLensDistributedGradientDescentUtil.randomVector(rank))
          val pf = ufpf._2.getOrElse(MovieLensDistributedGradientDescentUtil.randomVector(rank))
          val e = rating.rating - (uf dot pf)
          val ug = gamma :* ((e :* pf) - (lambda :* uf))
          val pg = gamma :* ((e :* uf) - (lambda :* pf))
          MovieLensDistributedGradientDescentUtil.setFeature(
            (rating.user, uf + ug),
            (rating.product, pf + pg)
          )
        }
      }
    }
  }

  def predict(userProduct: (Int, Int)): Float = {
    val ufpf = MovieLensDistributedGradientDescentUtil.getFeature(userProduct._1, userProduct._2)
    val uf = ufpf._1
    val pf = ufpf._2
    if (uf.isDefined && pf.isDefined) {
      uf.get dot pf.get
    } else {
      3.5f
    }
  }

  def predict(userProducts: RDD[(Int, Int)]): RDD[MovieLensRating] = {
    userProducts.map { case (user, product) =>
      new MovieLensRating(user, product, predict((user, product)), 0)
    }
  }

}

object MovieLensDistributedGradientDescentUtil extends Serializable {

  def randomVector(rank: Int) = Vector.fill(rank) {
    math.sqrt(math.random * 5 / rank).toFloat
  }

  private val redissonConfig = new RedissionConfig()
  redissonConfig.useSingleServer()
    .setAddress("127.0.0.1:6379")
    .setConnectionPoolSize(128)
    .setConnectionMinimumIdleSize(16)
  private val redisClient = Redisson.create(redissonConfig)
  private val kryoCodec = new KryoCodec()

  def setFeature(userFeature: (Int, Vector[Float]),
                 productFeature: (Int, Vector[Float])) = {
    val uf = redisClient.getBucket[Array[Float]](s"uf:${userFeature._1}", kryoCodec)
    val pf = redisClient.getBucket[Array[Float]](s"pf:${productFeature._1}", kryoCodec)
    uf.setAsync(userFeature._2.toArray)
    pf.setAsync(productFeature._2.toArray)
  }

  def getFeature(user: Int, product: Int): (Option[Vector[Float]], Option[Vector[Float]]) = {
    val userFuture = redisClient.getBucket[Array[Float]](s"uf:$user", kryoCodec).getAsync
    val productFuture = redisClient.getBucket[Array[Float]](s"pf:$product", kryoCodec).getAsync
    val userFeature = userFuture.join()
    val productFeature = productFuture.join()
    (
      if (userFeature != null) {
        Option(Vector(userFeature))
      } else {
        Option.empty
      },
      if (productFeature != null) {
        Option(Vector(productFeature))
      } else {
        Option.empty
      }
    )
  }

  def shutdown() = redisClient.shutdown()

}
