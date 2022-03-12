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

  /////////////////////////////////////////////////////////////////////////////////////////////////////////
  //B.1
  /////////////////////////////////////////////////////////////////////////////////////////////////////////

  //Calculate global average
  def globalAvg(ratings: Array[Rating]): Double = {
    var total = 0.0
    for (r<-ratings) {
      total = total + r.rating
    }
    return total/ratings.size
  }

  //Calculate user average
  def computeUserAvg(uid: Int, ratings: Array[Rating]): Double = {
    val userRatings = ratings.filter(r => r.user == uid).map(r=>r.rating)
    return mean(userRatings)
  }
  
  //Calculate item average
  def computeItemAvg(iid: Int, ratings: Array[Rating]): Double = {
    var itemRatings = ratings.filter(r => r.item == iid).map(r=>r.rating)
    return mean(itemRatings)
  }

  /* 
  def scale(x: Double, userAvg: Double): Double = {
    if (x > userAvg) {5 - userAvg}
    else if (x < userAvg) {userAvg - 1}
    else { 1 }
  } */

  //Calculate the average item deviation
  //Takes as input a Map containing user IDs linked to their average rating
  //this is done to speed up the item deviation calculation, as there is no need
  //to recalculate the user averages at each loop. (See the evaluateBaseline function)
  def averageItemDeviation(iid: Int, ratings: Array[Rating], userAverages: Map[Int,Double]): Double = { 
    val itemRatings = ratings.filter(r => r.item == iid)
    val userRatings = itemRatings.map(r=>(r.user, r.rating))
    val devs = userRatings.map{case (uid, r)=>(r-userAverages(uid))/scale(r,userAverages(uid))}
    return devs.sum/devs.size
  }
  
  def predictRating(uid: Int, iid: Int, ratings: Array[Rating]): Double = {
    val userAvg = (for (uid <- ratings.map(r=>r.user).distinct; 
                     avgRating = computeUserAvg(uid, ratings)
                    ) yield (uid, avgRating)).toMap
    val meanDeviation = averageItemDeviation(iid, ratings, userAvg)
    val norm = scale(userAvg(uid) + meanDeviation, userAvg(uid))
    return userAvg(uid) + meanDeviation * norm
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////
  //B.2
  /////////////////////////////////////////////////////////////////////////////////////////////////////////
  def MAE(pred: Array[Double], r: Array[Double]): Double = {
    pred.zip(r).map{case (a,b) => (b-a).abs}.sum/r.size
  }

  //Calculate MAE when using global average to predict
  def evaluateGlobal(train: Array[Rating], test: Array[Rating]): Double = {
    val pred = globalAvg(train)
    val preds = (1 to test.size).map(_ => pred).toArray
    val target = test.map(r => r.rating).toArray
    return MAE(preds, target)
  }

  //Calculate MAE when using the user's average to predict
  def evaluateUserAverage(train: Array[Rating], test: Array[Rating]): Double = {
    val uids = train.map(r=>r.user).distinct
    val userAverages = (for (uid <- uids; 
                     avgRating = computeUserAvg(uid, train)
                    ) yield (uid, avgRating)).toMap
    val preds = test.map(r => userAverages(r.user))
    val target = test.map(r => r.rating)
    return MAE(preds, target)
  }

  //Calculate MAE when using the item's average to predict
  def evaluateItemAverage(train: Array[Rating], test: Array[Rating]): Double = {
    val train_iids = train.map(r=>r.item).distinct
    val itemAverages = (for (iid <- train_iids;
                             avgRating = computeItemAvg(iid, train)
                       ) yield (iid, avgRating)).toMap
    val default = globalAvg(train)
    val preds = test.map(r => itemAverages.get(r.item) getOrElse(default))
    val target = test.map(r => r.rating)
    return MAE(preds, target)
  }

  //Calculate MAE when using the baseline prediction formula
  def evaluateBaseline(train: Array[Rating], test: Array[Rating]): Double = {
    //Get the train user IDs and item IDs
    val uids = train.map(r=>r.user).distinct
    val iids = train.map(r=>r.item).distinct
    //Calculate the user averages and save as a map
    val userAverages = (for (uid <- uids; 
                     avgRating = computeUserAvg(uid, train)
                    ) yield (uid, avgRating)).toMap
    //Calculate average item deviations using the User-Average map we just created
    val itemDeviations = (for (iid <- iids;
                    dev = averageItemDeviation(iid, train, userAverages)
                    ) yield (iid, dev)).toMap
    //If a user doesn't exist in the train set, default to using the global average
    val default = globalAvg(train)
    println(test.size)
    //Iterate over the test rows and get the predicted ratings
    val preds = for (row <- test;
                    ru = userAverages.get(row.user) getOrElse(default);
                    ri = itemDeviations.get(row.item) getOrElse(0.0);
                    norm = scale(ru+ri, ru)
                    ) yield (ru + ri * norm)
    val target = test.map(r=>r.rating)
    return MAE(preds, target)
  }


  /////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Part D
  /////////////////////////////////////////////////////////////////////////////////////////////////////////

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

  def distributedAllUserAverage(rdd: org.apache.spark.rdd.RDD[Rating]): org.apache.spark.rdd.RDD[(Int, Double)] = {
    rdd.map{case Rating(u, i, r) => (u,(r,1))}
      .reduceByKey((x,y)=>(x._1 + y._1, x._2 + y._2))
      .map{case (k,v)=> (k, v._1/v._2)}
  }
  //TODO: do the same with item deviation
  //then we can implement the predictor with the two 'lookup' table the MAE function would look like this:
  //rddTest.map(case R(u,i,r)=> (allUserBroadcast(u), allItemBroadcast(i),r))
  //    .map(|globalAverageDeviation(x,y) - r|).reduce(_ + _) / rddTest.count() 

  def scale(rating: Double, userAvg: Double):Double = {
    if (rating > userAvg) {5 - userAvg}
    else if (rating < userAvg) {userAvg - 1}
    else 1
  }

  def normalizedDeviation(x: Double, y: Double) = {
    (x - y)/scale(x,y)
  }

  def distributedNormalizedDeviation(rdd: org.apache.spark.rdd.RDD[Rating], user: Int, item: Int): Double = {
    val userAvg = distributedUserAverage(rdd, user)
    val rating = rdd.filter(x => x.user == user || x.item == item).take(1).head.rating
    (rating - userAvg) / scale(rating, userAvg)
  }

  def distributedItemDeviation(
      rdd: org.apache.spark.rdd.RDD[Rating], 
      item: Int,
      userAvgMap: org.apache.spark.broadcast.Broadcast[scala.collection.immutable.Map[Int,Double]]): Double = {
    val devs = rdd.filter(x => x.item == item)
      .map{case Rating(u, i, r) => (r, userAvgMap.value.getOrElse(u,0.0))}
      .map{case (r, avg) => normalizedDeviation(r, avg)}

    devs.reduce(_ + _) / devs.count()
  }

  def globalAverageDeviation(userAvg: Double, itemDev: Double): Double = {
    userAvg + itemDev * scale(userAvg + itemDev, userAvg)
  }

}