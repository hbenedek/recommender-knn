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
    mean(ratings.map(r=>r.rating))
  }

  //Calculate user average
  def computeUserAvg(uid: Int, ratings: Array[Rating]): Double = {
    val userRatings = ratings.filter(r => r.user == uid).map(r=>r.rating)
    return mean(userRatings)
  }

  def computeAllUserAverages(ratings: Array[Rating]): Map[Int, Double] = {
    ratings.groupBy(r=>r.user).map{case (k,v)=>(k, mean(v.map(r=>r.rating)))}
    
  }

  def computeAllItemAverages(ratings: Array[Rating]): Map[Int, Double] = {
    val x = ratings.groupBy(r=>r.item).map{case (k,v)=>(k, mean(v.map(r=>r.rating)))}
    x
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

  //Calculate the average item deviation for an item
  def averageItemDeviation(iid: Int, ratings: Array[Rating], userAverages: Map[Int,Double]): Double = { 
    val itemRatings = ratings.filter(r => r.item == iid)
    val userRatings = itemRatings.map(r=>(r.user, r.rating))
    val devs = userRatings.map{case (uid, r)=>(r-userAverages(uid))/scale(r,userAverages(uid))}
    return devs.sum/devs.size
  }

  //Compute average item deviations of every item
  def computeAllItemDeviations(ratings: Array[Rating], userAverages: Map[Int,Double]): Map[Int,Double] = {
    //Use the global average if a user doesn't exist in the ratings
    val averages = userAverages.withDefaultValue(globalAvg(ratings))
    //Group by the items
    val groupedItems = ratings.groupBy(r => r.item)
    //Compute the average deviations
    val devs = groupedItems.map{case (k, v)=>(k, mean(v.map(r=>(r.user, r.rating))
                                                  .map{case (u,r)=>(r-averages(u))/scale(r, averages(u))}))}
    devs                    
  }
  
  def predictRating(uid: Int, iid: Int, ratings: Array[Rating]): Double = {
    val userAvg = computeAllUserAverages(ratings)
    val meanDeviation = averageItemDeviation(iid, ratings, userAvg)
    val norm = scale(userAvg(uid) + meanDeviation, userAvg(uid))
    return userAvg(uid) + meanDeviation * norm
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////
  //B.2
  /////////////////////////////////////////////////////////////////////////////////////////////////////////
  def mae(pred: Array[Double], r: Array[Double]): Double = {
    pred.zip(r).map{case (a,b) => (b-a).abs}.sum/r.size
  }

  //Calculate MAE when using global average to predict
  def evaluateGlobal(train: Array[Rating], test: Array[Rating]): Double = {
    val pred = globalAvg(train)
    val preds = (1 to test.size).map(_ => pred).toArray
    val target = test.map(r => r.rating).toArray
    return mae(preds, target)
  }

  //Calculate MAE when using the user's average to predict
  def evaluateUserAverage(train: Array[Rating], test: Array[Rating]): Double = {
    val userAverages = computeAllUserAverages(train)
    val default = globalAvg(train)
    val preds = test.map(r => userAverages.get(r.user) getOrElse(default))
    val target = test.map(r => r.rating)
    return mae(preds, target)
  }

  //Calculate MAE when using the item's average to predict
  def evaluateItemAverage(train: Array[Rating], test: Array[Rating]): Double = {
    val train_iids = train.map(r=>r.item).distinct
    val itemAverages = computeAllItemAverages(train)
    val default = globalAvg(train)
    val preds = test.map(r => itemAverages.get(r.item) getOrElse(default))
    val target = test.map(r => r.rating)
    return mae(preds, target)
  }

  //Calculate MAE when using the baseline prediction formula
  def evaluateBaseline(train: Array[Rating], test: Array[Rating]): Double = {    
    //Calculate the user averages and save as a map
    val userAverages = computeAllUserAverages(train)
    //Calculate average item deviations using the User-Average map we just created
    val itemDeviations = computeAllItemDeviations(train, userAverages)
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
    return mae(preds, target)
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

  def distributedItemDeviation(
    rdd: org.apache.spark.rdd.RDD[Rating], 
    item: Int,
    userAvgMap: org.apache.spark.broadcast.Broadcast[scala.collection.immutable.Map[Int,Double]]): Double = {
    val devs = rdd.filter(x => x.item == item)
      .map{case Rating(u, i, r) => (r, userAvgMap.value.getOrElse(u,0.0))}
      .map{case (r, avg) => normalizedDeviation(r, avg)}

    devs.reduce(_ + _) / devs.count()
  }

  def scale(rating: Double, userAvg: Double):Double = {
    if (rating > userAvg) {5 - userAvg}
    else if (rating < userAvg) {userAvg - 1}
    else 1
  }

  def normalizedDeviation(x: Double, y: Double) = {
    (x - y)/scale(x,y)
  }

  def predict(userAvg: Double, itemDev: Double): Double = {
    userAvg + itemDev * scale(userAvg + itemDev, userAvg)
  }

  def distributedAllUserAverage(rdd: org.apache.spark.rdd.RDD[Rating]): org.apache.spark.rdd.RDD[(Int, Double)] = {
    rdd.map{case Rating(u, i, r) => (u,(r,1))}
      .reduceByKey((x,y)=>(x._1 + y._1, x._2 + y._2))
      .map{case (k,v)=> (k, v._1/v._2)}
  }

  def distributedAllItemDeviation(
    rdd: org.apache.spark.rdd.RDD[Rating],
    userAvgMap: org.apache.spark.broadcast.Broadcast[scala.collection.immutable.Map[Int,Double]],
    default: Double): org.apache.spark.rdd.RDD[(Int, Double)] = {
    rdd.map{case Rating(u, i, r) => (i, (normalizedDeviation(r,userAvgMap.value.getOrElse(u,default)),1))}
      .reduceByKey((x,y)=>(x._1 + y._1, x._2 + y._2))
      .map{case (k,v)=> (k, v._1/v._2)}
  }
  
  def distributedBaselineMAE(
    rddTest: org.apache.spark.rdd.RDD[Rating],
    userAvgMap: org.apache.spark.broadcast.Broadcast[scala.collection.immutable.Map[Int,Double]],
    itemDevMap: org.apache.spark.broadcast.Broadcast[scala.collection.immutable.Map[Int,Double]],
    default: Double): Double ={
    rddTest.map{case Rating(u,i,r) => (userAvgMap.value.getOrElse(u,default), itemDevMap.value.getOrElse(i,0.0), r)}
          .map{case (avg, dev, r) => (predict(avg,dev) - r).abs}
          .reduce(_ + _) / rddTest.count() 
  } 


  /////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Part P
  /////////////////////////////////////////////////////////////////////////////////////////////////////////

  def computeAllNormalizedDevs(ratings: Array[Rating], userAverages: Map[Int,Double]): Array[Rating] = {
    ratings.map(r => Rating(r.user, r.item, normalizedDeviation(r.rating, userAverages(r.user))))
  }

  def preprocessRatings(ratings: Array[Rating], userAverages: Map[Int,Double]): Array[Rating] = {
    val devs = computeAllNormalizedDevs(ratings, userAverages)
    val denominators = devs.map(r => (r.user, r.rating * r.rating))
      .groupBy(_._1).mapValues( _.map(_._2).sum ).toList
      .map{case (u,v) => (u, scala.math.sqrt(v))}
      .toMap
    devs.map(r => Rating(r.user, r.item, r.rating/denominators(r.user)))
  }

   def cosineSimilarity(preprocessedRatings: Array[Rating], u: Int, v: Int): Double = {
    preprocessedRatings.filter(r => r.user == u || r.user == v)
      .map(r => (r.item,r.rating))
      .groupBy(_._1)
      .filter(x => x._2.size == 2)
      .mapValues(_.map(_._2).reduce(_ * _))
      .map(_._2)
      .reduce(_ + _)
  }

  def weightedAllItemDevForOneUser(normalizedRatings: Array[Rating], user: Int, item: Int, similarity: (Array[Rating], Int, Int) => Double): Double = {
    val weightedTuple = normalizedRatings.filter(r => r.item == item)
      .map(r => (r.rating, similarity(normalizedRatings, r.user, user)))
      .reduceOption((acc,elem) => (acc._1 + elem._1 * elem._2, acc._2 + elem._2)).getOrElse((0.0,1.0))
    weightedTuple._1 / weightedTuple._2
  }

  //For the moment I don't use this function. I was experimenting how to store similarity coeffs between users...
  def similarityMapper(ratings: Array[Rating], similarity: (Array[Rating], Int, Int) => Double): Map[(Int, Int), Double] = {
    val users = ratings.map(r => (r.user)).distinct
    val mapped = for {u1 <- users; u2 <- users} yield ((u1, u2), similarity(ratings, u1, u2))
    mapped.toMap
  }

  def jaccardIndexSimilarity(ratings: Array[Rating], u: Int, v: Int): Double = {
    val uRatings = ratings.filter(r => r.user == u).map(r => r.item)
    val vRatings = ratings.filter(r => r.user == v).map(r => r.item)
    uRatings.intersect(vRatings).size.toDouble / uRatings.union(vRatings).size.toDouble
  }

  def similarityMae(train: Array[Rating], test: Array[Rating], similarity: (Array[Rating], Int, Int) => Double): Double = {
    val global = globalAvg(train)
    val userAverages = computeAllUserAverages(train).withDefaultValue(global)
    val normalizedRatings = computeAllNormalizedDevs(train, userAverages)
    test.map(r => (userAverages(r.user), weightedAllItemDevForOneUser(normalizedRatings, r.user, r.item, similarity), r.rating))
      .map{case (avg, dev, r) => (predict(avg, dev) - r).abs}
      .reduce(_ + _) / test.size
  }

  /*
  Jaccard similarity test
  */

  //Compute all Jaccard similarities between users
  //TODO: probably nicer way of doing this without a for loop, and maybe having 
  //the user pair (u,v) as a key would be better,
  def allJaccardSimilarities(ratings: Array[Rating]): Map[Int, Map[Int, Double]] = {
    val userRatings = ratings.groupBy(r=>r.user)
    val uids = ratings.map(r=>r.user).distinct.toSet
    val similarityMap = for (uid <- uids; 
      //Iterate over users
      current = userRatings(uid).map(r=>r.item).toSet;
      //Get items the other users have rated, compute Jaccard
      others = (uids - uid).map(u => (u,userRatings(u).map(r=>r.item).toSet))
                               .map(r=>(r._1,r._2.intersect(current).size.toDouble/r._2.union(current).size.toDouble))
                               .toMap//.intersect(current))
    ) yield (uid, others)
    similarityMap.toMap
  }

  //Same function as computeAllItemDeviations (Part B.1), but doesn't average over items.
  def userItemDeviation(ratings: Array[Rating], userAverages: Map[Int,Double]): Map[Int,Map[Int,Double]] = {
    //Use the global average if a user doesn't exist in the ratings
    val averages = userAverages.withDefaultValue(globalAvg(ratings))
    //Group by the items
    val groupedItems = ratings.groupBy(r => r.item)
    //Compute the average deviations
    val devs = groupedItems.map{case (k, v)=>(k, v.map(r=>(r.user, r.rating))
                                                  .map{case (u,r)=>(u,(r-averages(u))/scale(r, averages(u)))}.toMap)}
    devs 
  }                  

  //Evaluate MAE when using Jaccard Similarity. 
  //TODO: Again, maybe getting rid of for loop could be nice
  def evaluateJaccardSimilarity(train: Array[Rating], test: Array[Rating]): Double = {
    val global = globalAvg(train)
    println("Computing user averages...")
    val userAverages = computeAllUserAverages(train).withDefaultValue(global)
    println("Computing User-Item Devations...")
    val itemDevs = userItemDeviation(train, userAverages)//computeAllItemDeviations(train, userAverages)
    println("Computing Jaccard Similarities...")
    val sims = allJaccardSimilarities(train)
    println("Predicting...")
    //val vs = train.groupBy(r=>r.item).map{case (k,v)=>(k, v.map(r=>r.user))}
    val preds = for (row <- test;
                 vs = train.filter(r=>r.item==row.item).map(r=>r.user);
                 u = row.user; 
                 uv_sum = vs.map(v=>sims(u)(v)).sum;
                 ri = if (uv_sum!=0) vs.map(v=>sims(u)(v) * itemDevs(row.item)(v)).sum/uv_sum else 0.0;
                 ru = userAverages(u);
                 norm = scale(ru+ri,ru)
                 ) yield(ru + ri * norm)
    val target = test.map(r=>r.rating)
    mae(preds, target)
  }

}
