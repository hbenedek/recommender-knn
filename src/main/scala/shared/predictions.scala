package shared

import org.apache.spark.rdd.RDD
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.SparkSession

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

  def scale(rating: Double, userAvg: Double):Double = {
    if (rating > userAvg) {5 - userAvg}
    else if (rating < userAvg) {userAvg - 1}
    else 1
  }

  def normalizedDeviation(x: Double, y: Double) = {
    (x - y) / scale(x,y)
  }

  def predict(userAvg: Double, itemDev: Double): Double = {
    userAvg + itemDev * scale(userAvg + itemDev, userAvg)
  }

  def globalAvg(ratings: Array[Rating]): Double = {
    mean(ratings.map(r => r.rating))
  }

  def computeUserAvg(uid: Int, ratings: Array[Rating]): Double = {
    mean(ratings.filter(r => r.user == uid).map(r => r.rating))
  }

  def computeAllUserAverages(ratings: Array[Rating]): Map[Int, Double] = {
    ratings.groupBy(r=>r.user).map{case (k,v) => (k, mean(v.map(r => r.rating)))}
  }

  def computeAllItemAverages(ratings: Array[Rating]): Map[Int, Double] = {
    ratings.groupBy(r=>r.item).map{case (k,v) => (k, mean(v.map(r => r.rating)))}
  }
  
  def computeItemAvg(iid: Int, ratings: Array[Rating]): Double = {
    mean(ratings.filter(r => r.item == iid).map(r => r.rating))
  }

  def averageItemDeviation(iid: Int, ratings: Array[Rating], userAverages: Map[Int,Double]): Double = { 
    val filtered = ratings.filter(r => r.item == iid)
    filtered.map(r=>(r.user, r.rating))
            .map{case (u, r)=> normalizedDeviation(r, userAverages(u))}.sum / filtered.size
  }

  def computeAllItemDeviations(ratings: Array[Rating], userAverages: Map[Int,Double]): Map[Int,Double] = {
    //Use the global average if a user doesn't exist in the ratings
    val averages = userAverages.withDefaultValue(globalAvg(ratings))
    //Group by the items
    val groupedItems = ratings.groupBy(r => r.item)
    //Compute the average deviations
    groupedItems.map{case (k, v) => (k, mean(v.map(r=>(r.user, r.rating))
                                                  .map{case (u,r)=> normalizedDeviation(r, averages(u))}))}                   
  }
  
  def predictRating(u: Int, i: Int, ratings: Array[Rating]): Double = {
    val userAvg = computeAllUserAverages(ratings)
    val meanDeviation = averageItemDeviation(i, ratings, userAvg)
    val norm = scale(userAvg(u) + meanDeviation, userAvg(u))
    userAvg(u) + meanDeviation * norm
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////
  //B.2
  /////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  def mae(pred: Array[Double], r: Array[Double]): Double = {
    pred.zip(r).map{case (a,b) => (b-a).abs}.sum/r.size
  }

  //Helper functions to fit recommender models using different prediction methods, all functions return
  //a predictor with the signiture (Int, Int) => Double, where the arguments are the user and movie id, and the 
  //returned double is the prediction
  def predictorGlobal(train: Array[Rating]): ((Int, Int) => Double) = {
    val global = globalAvg(train)
    (u: Int, i: Int) => global
  }

  def predictorUserAverage(train: Array[Rating]): ((Int, Int) => Double) = {
    val userMap = computeAllUserAverages(train)
    val global = globalAvg(train)
    (u: Int, i: Int) => userMap.getOrElse(u, global)
  }

  def predictorItemAverage(train: Array[Rating]): ((Int, Int) => Double) = {
    val userMap = computeAllUserAverages(train)
    val itemMap = computeAllItemAverages(train)
    val global = globalAvg(train)
    (u: Int, i: Int) => itemMap.getOrElse(i, global)
  }

  def predictorBaseline(train: Array[Rating]): ((Int, Int) => Double) = {
    val userAverages = computeAllUserAverages(train)
    val itemDeviations = computeAllItemDeviations(train, userAverages)
    val global = globalAvg(train)
    (u: Int, i: Int) =>  predict(userAverages.getOrElse(u, global), itemDeviations.getOrElse(i, 0.0))
  }

  //Calculates the MAE on a test set, given a fitted predictor function
  def evaluatePredictor(test: Array[Rating], predictor: (Int, Int) => Double): Double = {
    test.map(r => (predictor(r.user, r.item), r.rating))
        .map{case (pred, target) => (pred - target).abs}
        .reduce(_ + _ ) / test.size.toDouble
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Part D
  /////////////////////////////////////////////////////////////////////////////////////////////////////////

  def distributedGlobalAverage(rdd: RDD[Rating]): Double = {
    rdd.map(x => x.rating).reduce(_ + _) / rdd.count()
  }

  def distributedUserAverage(rdd: RDD[Rating], user: Int): Double = {
    val pair = rdd
          .filter(x => x.user == user)
          .map(x => x.rating)
          .map(x => (x,1))
          .reduce((x,y) => (x._1 + y._1, x._2 + y._2))

    pair._1 / pair._2
  }

  def distributedItemAverage(rdd: RDD[Rating], item: Int): Double = {
    val pair = rdd.filter(x => x.item == item)
          .map(x => x.rating)
          .map(x => (x,1))
          .reduce((x,y) => (x._1 + y._1, x._2 + y._2))

    pair._1 / pair._2
  }

  def distributedItemDeviation(
    rdd: RDD[Rating], 
    item: Int,
    userAvgMap: Broadcast[Map[Int,Double]]): Double = {
    val devs = rdd.filter(x => x.item == item)
      .map{case Rating(u, i, r) => (r, userAvgMap.value.getOrElse(u,0.0))}
      .map{case (r, avg) => normalizedDeviation(r, avg)}

    devs.reduce(_ + _) / devs.count()
  }

  def distributedAllUserAverage(rdd: RDD[Rating]): RDD[(Int, Double)] = {
    rdd.map{case Rating(u, i, r) => (u,(r,1))}
      .reduceByKey((x,y)=>(x._1 + y._1, x._2 + y._2))
      .map{case (k,v)=> (k, v._1/v._2)}
  }

  def distributedAllItemAverage(rdd: RDD[Rating]): RDD[(Int, Double)] = {
    rdd.map{case Rating(u, i, r) => (i,(r,1))}
      .reduceByKey((x,y)=>(x._1 + y._1, x._2 + y._2))
      .map{case (k,v)=> (k, v._1/v._2)}
  }

  def distributedAllItemDeviation(
    rdd: RDD[Rating],
    userAvgMap: Broadcast[Map[Int,Double]],
    default: Double): RDD[(Int, Double)] = {
    rdd.map{case Rating(u, i, r) => (i, (normalizedDeviation(r,userAvgMap.value.getOrElse(u,default)),1))}
      .reduceByKey((x,y)=>(x._1 + y._1, x._2 + y._2))
      .map{case (k,v)=> (k, v._1/v._2)}
  }

  //Helper functions to fit recommender models using different prediction methods, all functions return
  //a predictor with the signiture (Int, Int) => Double, where the arguments are the user and movie id, and the 
  //returned double is the prediction 
  def predictorDistributedGlobal(rddTrain: RDD[Rating], spark: SparkSession): ((Int, Int) => Double) = {
    val global = distributedGlobalAverage(rddTrain)
    (u: Int, i: Int) => global
  }

  def predictorDistributedUserAverage(rddTrain: RDD[Rating], spark: SparkSession): ((Int, Int) => Double) = {
    val global = distributedGlobalAverage(rddTrain)
    val allUser = distributedAllUserAverage(rddTrain)
    val allUserBroadcast = spark.sparkContext.broadcast(allUser.collect().toMap.withDefaultValue(global))
    (u: Int, i: Int) => allUserBroadcast.value.getOrElse(u , global)
  }

  def predictorDistributedItemAverage(rddTrain: RDD[Rating], spark: SparkSession): ((Int, Int) => Double) = {
    val global = distributedGlobalAverage(rddTrain)
    val allItem = distributedAllItemAverage(rddTrain)
    val allItemBroadcast = spark.sparkContext.broadcast(allItem.collect().toMap.withDefaultValue(global))
    (u: Int, i: Int) => allItemBroadcast.value.getOrElse(i , global)
  }

  def predictorDistributedBaseline(rddTrain: RDD[Rating], spark: SparkSession): ((Int, Int) => Double) = {
    val global = distributedGlobalAverage(rddTrain)
    val allUser = distributedAllUserAverage(rddTrain)
    val allUserBroadcast = spark.sparkContext.broadcast(allUser.collect().toMap.withDefaultValue(global))
    val allItemDev = distributedAllItemDeviation(rddTrain, allUserBroadcast, global)
    val allItemDevBroadcast = spark.sparkContext.broadcast(allItemDev.collect().toMap)
    (u: Int, i: Int) =>  predict(allUserBroadcast.value.getOrElse(u, global), allItemDevBroadcast.value.getOrElse(i, 0.0))
  }

  //Calculates the MAE on a test set, given a fitted predictor function
  def evaluateDistributedPredictor(rddTest: RDD[Rating], predictor: (Int, Int) => Double): Double = {
    rddTest.map(r => (predictor(r.user, r.item), r.rating))
        .map{case (pred, target) => (pred - target).abs}
        .reduce(_ + _ ) / rddTest.count()
  }


  /////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Part P
  /////////////////////////////////////////////////////////////////////////////////////////////////////////

  def computeAllNormalizedDevs(ratings: Array[Rating], userAverages: Map[Int,Double]): Array[Rating] = {
    ratings.map(r => Rating(r.user, r.item, normalizedDeviation(r.rating, userAverages(r.user))))
  }

  // Processes training set to calculate cosine similarity
  def preprocessRatings(ratings: Array[Rating], userAverages: Map[Int,Double]): Array[Rating] = {
    val devs = computeAllNormalizedDevs(ratings, userAverages)
    val denominators = devs.map(r => (r.user, r.rating * r.rating))
      .groupBy(_._1).mapValues( _.map(_._2).sum ).toList
      .map{case (u,v) => (u, scala.math.sqrt(v))}
      .toMap
    devs.map(r => Rating(r.user, r.item, r.rating/denominators(r.user)))
  }

  //Returns a map containing every item deviation, givan a training set
  def userItemDeviation(ratings: Array[Rating], userAverages: Map[Int,Double]): Map[Int,Map[Int,Double]] = {
    val averages = userAverages.withDefaultValue(globalAvg(ratings))
    val groupedItems = ratings.groupBy(r => r.item)
    groupedItems.map{case (k, v) => (k, v.map(r=>(r.user, r.rating))
                                        .map{case (u,r) => (u, normalizedDeviation(r, averages(u)))}.toMap)}
  }  
  

  // calculates the cosine similarity betwwen two users with PREPROCESSED ratings 
  //(for more detail see preprocessRatings function)
  def cosineSimilarity(u: Array[Rating], v: Array[Rating]): Double = {
    (u ++ v).map(r => (r.item,r.rating))
            .groupBy(_._1)
            .filter(x => x._2.size == 2)
            .mapValues(_.map(_._2).reduce(_ * _))
            .map(_._2)
            .reduceOption(_ + _).getOrElse((0.0))
  }

  def jaccardIndexSimilarity(u: Array[Rating], v: Array[Rating]): Double = {
    val uSet = u.map(r=>r.item).toSet
    val vSet = v.map(r=>r.item).toSet
    uSet.intersect(vSet).size.toDouble / uSet.union(vSet).size.toDouble
  }

  def oneSimilarity(u: Array[Rating], v: Array[Rating]): Double = 1

  // For a given training set and similarity function, retuns a map containing the calculated similarities between all user pairs
  def similarityMapper(ratings: Array[Rating], similarity: (Array[Rating], Array[Rating]) => Double): Map[Int, Map[Int, Double]] = {
    val averages = computeAllUserAverages(ratings).withDefaultValue(globalAvg(ratings))
    val processed = preprocessRatings(ratings, averages)
    val users = processed.map(r => r.user).distinct
    val userRatings = processed.groupBy(r => r.user)
    val mapped = for {u1 <- users;
      sims = users.map(v => (v, similarity(userRatings(u1), userRatings(v)))).toMap
    } yield (u1, sims)
    mapped.toMap
  }

  //Given a Rating object and a similarity metric, the function calculates the weighted item deviation for the item
  //and with the user average it predicts a rating for the item
  def predict(row: Rating, train: Array[Rating], sims: Map[Int, Map[Int, Double]], itemDevs: Map[Int, Map[Int, Double]], userAverages: Map[Int, Double]): Double ={
    val vs = train.filter(r => r.item == row.item).map(r => r.user)
    val u = row.user
    val uv_sum = vs.map(v => sims(u)(v)).sum
    val ri = if (uv_sum!=0) vs.map(v => sims(u)(v) * itemDevs(row.item)(v)).sum / uv_sum else 0.0
    val ru = userAverages(u)
    val norm = scale(ru + ri, ru)
    ru + ri * norm
  }

  //Create the predictors for the Uniform, Cosine, and Jaccard similarities.
  def computeOnesPredictor(train: Array[Rating]): ((Int,Int)=>Double) = {
    val global = globalAvg(train)
    val userAverages = computeAllUserAverages(train).withDefaultValue(global)
    val itemDevs = userItemDeviation(train, userAverages)
    val sims = similarityMapper(train, oneSimilarity)
    (u: Int, i: Int) => predict(Rating(u,i,0.0), train, sims, itemDevs, userAverages)
  }

  def computeCosinePredictor(train: Array[Rating]): ((Int,Int)=>Double) = {
    val global = globalAvg(train)
    val userAverages = computeAllUserAverages(train).withDefaultValue(global)
    val itemDevs = userItemDeviation(train, userAverages)
    val sims = similarityMapper(train, cosineSimilarity)
    (u: Int, i: Int) => predict(Rating(u,i,0.0), train, sims, itemDevs, userAverages)
  }

  def computeJaccardPredictor(train: Array[Rating]): ((Int,Int)=>Double) = {
    val global = globalAvg(train)
    val userAverages = computeAllUserAverages(train).withDefaultValue(global)
    val itemDevs = userItemDeviation(train, userAverages)
    val sims = similarityMapper(train, jaccardIndexSimilarity)
    (u: Int, i: Int) => predict(Rating(u,i,0.0), train, sims, itemDevs, userAverages)
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Part N
  /////////////////////////////////////////////////////////////////////////////////////////////////////////

  //Calculate the K nearest neighbours of user u given a similarity map
  def knn(u: Int, k: Int, sims: Map[Int, Map[Int, Double]]): Map[Int, Double] = {
    Array(sims(u).toSeq.sortWith(_._2 > _._2):_*).slice(1,k+1).toMap.withDefaultValue(0.0)
  }
  //Reduce a similarity map to only contain the K nearest neighbours for each user
  def computeAllKNN(k: Int, sims: Map[Int, Map[Int, Double]]): Map[Int, Map[Int, Double]] = {
    sims.keySet.map(u=>(u, knn(u,k,sims))).toMap
  }

  //Can probably make this function better, but basically compute the similarity mapping
  //before hand, and evaluate the MAE when using various K nearest neighbours. 
  //This is done to avoid re-computing the similarity map for each K
  def evaluateKValues(train: Array[Rating], test: Array[Rating], ks: List[Int]): Map[Int,Double] = {
    val global = globalAvg(train)
    println("Computing user averages...")
    val userAverages = computeAllUserAverages(train).withDefaultValue(global)
    println("Computing User-Item Devations...")
    val itemDevs = userItemDeviation(train, userAverages)
    println("Computing  Similarities...")
    val cosineMap = similarityMapper(train, cosineSimilarity)

    println("Evaluating the following k values: " + ks.mkString(", "))
    val target =  test.map(r => r.rating)
    val scores = for (k <- ks;
                      knnMap = computeAllKNN(k, cosineMap);
                      preds = test.map(row => predict(row, train, knnMap, itemDevs, userAverages))
                  ) yield (k, mae(preds,target))
    
    //val knnMap = computeAllKNN(k, cosineMap)
    scores.toMap
  }

  //Create the predictors for the KNNs
  def computeKnnPredictor(train: Array[Rating], k: Int): ((Int,Int)=>Double) = {
    val global = globalAvg(train)
    val userAverages = computeAllUserAverages(train).withDefaultValue(global)
    val itemDevs = userItemDeviation(train, userAverages)
    val sims = similarityMapper(train, cosineSimilarity)
    val knnMap = computeAllKNN(k, sims)
    (u: Int, i: Int) => predict(Rating(u,i,0.0), train, knnMap, itemDevs, userAverages)
  }

}
