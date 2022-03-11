package shared

package object predictions
{
  case class Rating(user: Int, item: Int, rating: Double)

  def timingInMs(f : ()=>Double ) : (Double, Double) = {
    val start = System.nanoTime() 
    val output = f()
    val end = System.nanoTime()
    return (output, (end-start)/1000000.0)
  }

  def mean(s :Seq[Double]): Double =  if (s.size > 0) s.reduce(_+_) / s.length else 0.0
  def std(s :Seq[Double]): Double = {
    if (s.size == 0) 0.0
    else {
      val m = mean(s)
      scala.math.sqrt(s.map(x => scala.math.pow(m-x, 2)).sum / s.length.toDouble)
    }
  }

  def toInt(s: String): Option[Int] = {
    try {
      Some(s.toInt)
    } catch {
      case e: Exception => None
    }
  }

  def load(spark : org.apache.spark.sql.SparkSession,  path : String, sep : String) : org.apache.spark.rdd.RDD[Rating] = {
       val file = spark.sparkContext.textFile(path)
       return file
         .map(l => {
           val cols = l.split(sep).map(_.trim)
           toInt(cols(0)) match {
             case Some(_) => Some(Rating(cols(0).toInt, cols(1).toInt, cols(2).toDouble))
             case None => None
           }
       })
         .filter({ case Some(_) => true 
                   case None => false })
         .map({ case Some(x) => x 
                case None => Rating(-1, -1, -1)})
  }

  def distributedGlobalAverage(rdd: org.apache.spark.rdd.RDD[Rating]): Double = {
    rdd.map(x => x.rating).reduce(_ + _) / rdd.count()
  }

  def distributedUserAverage(rdd: org.apache.spark.rdd.RDD[Rating], user: Int): Double = {
    val pair = rdd
          .filter(x => x.user == user)
          .map(x => x.rating)
          .map(x => (x,1))
          .reduce((x,y) => (x._1 + y._1, x._2 + y._2))

    pair._1 / pair._2
  }

  def distributedItemAverage(rdd: org.apache.spark.rdd.RDD[Rating], item: Int): Double = {
    val pair = rdd.filter(x => x.item == item)
          .map(x => x.rating)
          .map(x => (x,1))
          .reduce((x,y) => (x._1 + y._1, x._2 + y._2))

    pair._1 / pair._2
  }

  def scale(x: Double, userAvg: Double):Double = {
   if (x > userAvg) {5 - userAvg}
   else if (x < userAvg) {userAvg - 1}
   else 1
  }

  def distributedNormalizedDeviation(rdd: org.apache.spark.rdd.RDD[Rating], user: Int, item: Int): Double = {
    val userAvg = distributedUserAverage(rdd, user)
    val rating = rdd.filter(x => x.user == user || x.item == item).take(1).head.rating
    (rating - userAvg) / scale(rating, userAvg)
  }

  def distributedItemDeviation(rdd: org.apache.spark.rdd.RDD[Rating], item: Int): Double = {
    val users = rdd.filter(x => x.item == item).map(x => x.user).distinct.collect()
    //users.map(u => distributedNormalizedDeviation(rdd, u, item)).reduce(_ + _) / users.count()
    //TODO: fix this. it is really slow for the moment, i could not deal with nested RDDs... maybe broadcast variables?
    val userAvgs = for (u <- users) yield distributedNormalizedDeviation(rdd, u, item)
    userAvgs.reduce(_ + _) / userAvgs.size
  }

  def distributedPrediction(rdd: org.apache.spark.rdd.RDD[Rating], user: Int, item: Int): Double = {
    // i didn't manage to test this function since it uses distributedItemDeviation
    val userAvg = distributedUserAverage(rdd, user)
    val itemDev = distributedItemDeviation(rdd, item)
    val scaled = scale(userAvg + itemDev, userAvg)
    userAvg + itemDev * scaled
  }
}