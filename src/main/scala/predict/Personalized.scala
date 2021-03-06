package predict

import org.rogach.scallop._
import org.apache.spark.rdd.RDD
import ujson._

import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level

import scala.math
import shared.predictions._


class PersonalizedConf(arguments: Seq[String]) extends ScallopConf(arguments) {
  val train = opt[String](required = true)
  val test = opt[String](required = true)
  val separator = opt[String](default=Some("\t"))
  val num_measurements = opt[Int](default=Some(0))
  val json = opt[String]()
  verify()
}

object Personalized extends App {
  // Remove these lines if encountering/debugging Spark
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)
  val spark = SparkSession.builder()
    .master("local[1]")
    .getOrCreate()
  spark.sparkContext.setLogLevel("ERROR") 

  println("")
  println("******************************************************")

  var conf = new PersonalizedConf(args) 
  println("Loading training data from: " + conf.train()) 
  val train = load(spark, conf.train(), conf.separator()).collect()
  println("Loading test data from: " + conf.test()) 
  val test = load(spark, conf.test(), conf.separator()).collect()
  
  println("Number of users: " + train.map(r=>r.user).distinct.size)

  val globalAvgRating = globalAvg(train)
  val userAverages = computeAllUserAverages(train).withDefaultValue(globalAvgRating)
  val normalizedRatings = computeAllNormalizedDevs(train, userAverages)
  val userItemDevs = userItemDeviation(train, userAverages)

  println("Calculating results with similarity constant one")
  
  val onesPredictor = computeOnesPredictor(train)
  val onePredUser1Item1 = onesPredictor(1,1)
  val onesMae = evaluatePredictor(test, onesPredictor)
 

  println("Calculating results with Jaccard similarity")
  val u1 = preprocessRatings(train.filter(r => r.user == 1), userAverages)
  val u2 = preprocessRatings(train.filter(r => r.user == 2), userAverages)
  val jaccardUser1User2 = jaccardIndexSimilarity(u1, u2)

  val jaccardPredictor = computeJaccardPredictor(train)
  val jaccardPredUser1Item1 = jaccardPredictor(1,1)
  val jaccardMae = evaluatePredictor(test, jaccardPredictor)
  
  println("Calculating results with Cosine similarity")

  val adjustedCosineUser1User2 = cosineSimilarity(u1,u2)
  val cosinePredictor = computeCosinePredictor(train)
  val cosinePredUser1Item1 = cosinePredictor(1,1)
  val cosineMae = evaluatePredictor(test, cosinePredictor)

  // Save answers as JSON
  def printToFile(content: String, 
                  location: String = "./answers.json") =
    Some(new java.io.PrintWriter(location)).foreach{
      f => try{
        f.write(content)
      } finally{ f.close }
  }
  conf.json.toOption match {
    case None => ; 
    case Some(jsonFile) => {
      val answers = ujson.Obj(
        "Meta" -> ujson.Obj(
          "1.Train" -> ujson.Str(conf.train()),
          "2.Test" -> ujson.Str(conf.test()),
          "3.Measurements" -> ujson.Num(conf.num_measurements())
        ),
        "P.1" -> ujson.Obj(
          "1.PredUser1Item1" -> ujson.Num(onePredUser1Item1), // Prediction of item 1 for user 1 (similarity 1 between users)
          "2.OnesMAE" -> ujson.Num(onesMae)         // MAE when using similarities of 1 between all users
        ),
        "P.2" -> ujson.Obj(
          "1.AdjustedCosineUser1User2" -> ujson.Num(adjustedCosineUser1User2), // Similarity between user 1 and user 2 (adjusted Cosine)
          "2.PredUser1Item1" -> ujson.Num(cosinePredUser1Item1),  // Prediction item 1 for user 1 (adjusted cosine)
          "3.AdjustedCosineMAE" -> ujson.Num(cosineMae) // MAE when using adjusted cosine similarity
        ),
        "P.3" -> ujson.Obj(
          "1.JaccardUser1User2" -> ujson.Num(jaccardUser1User2), // Similarity between user 1 and user 2 (jaccard similarity)
          "2.PredUser1Item1" -> ujson.Num(jaccardPredUser1Item1),  // Prediction item 1 for user 1 (jaccard)
          "3.JaccardPersonalizedMAE" -> ujson.Num(jaccardMae) // MAE when using jaccard similarity
        )
      )
      val json = write(answers, 4)
      println(json)
      println("Saving answers in: " + jsonFile)
      printToFile(json, jsonFile)
    }
  }

  println("")
  spark.close()
}
