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
  
  // Compute here

  val globalAvgRating = globalAvg(train)
  val userAverages = computeAllItemAverages(train).withDefaultValue(globalAvgRating)
  val normalizedRatings = computeAllNormalizedDevs(train, userAverages)

  //Constant one similarity
  val devUser1Item1 = weightedAllItemDevForOneUser(normalizedRatings, allOnesSimilarity, 1, 1)
  val predUser1Item1 = predict(userAverages(1), devUser1Item1)
  val onesMae = similarityMae(train, test, allOnesSimilarity)

  //Jaccard
  val jaccardUser1User2 = jaccardIndexSimilarity(train, 1, 2)
  val jaccardDevUser1Item1 = weightedAllItemDevForOneUser(normalizedRatings, jaccardIndexSimilarity, 1, 1)
  val jaccardPredUser1Item1 = predict(userAverages(1), jaccardDevUser1Item1)
  //TODO: mae does not work with nontrivial similarity
  //val jaccardMae = similarityMae(train, test, jaccardIndexSimilarity)

  //Cosine
  val processed = preprocessRatings(train, userAverages)
  val adjustedCosineUser1User2 = cosineSimilarity(processed, 1, 2)
  val cosineDevUser1Item1 = weightedAllItemDevForOneUser(normalizedRatings, cosineSimilarity, 1, 1)
  val cosPredUser1Item1 = predict(userAverages(1), cosineDevUser1Item1)

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
          "1.PredUser1Item1" -> ujson.Num(predUser1Item1), // Prediction of item 1 for user 1 (similarity 1 between users)
          "2.OnesMAE" -> ujson.Num(onesMae)         // MAE when using similarities of 1 between all users
        ),
        "P.2" -> ujson.Obj(
          "1.AdjustedCosineUser1User2" -> ujson.Num(adjustedCosineUser1User2), // Similarity between user 1 and user 2 (adjusted Cosine)
          "2.PredUser1Item1" -> ujson.Num(cosPredUser1Item1),  // Prediction item 1 for user 1 (adjusted cosine)
          "3.AdjustedCosineMAE" -> ujson.Num(0.0) // MAE when using adjusted cosine similarity
        ),
        "P.3" -> ujson.Obj(
          "1.JaccardUser1User2" -> ujson.Num(jaccardUser1User2), // Similarity between user 1 and user 2 (jaccard similarity)
          "2.PredUser1Item1" -> ujson.Num(jaccardPredUser1Item1),  // Prediction item 1 for user 1 (jaccard)
          "3.JaccardPersonalizedMAE" -> ujson.Num(0.0) // MAE when using jaccard similarity
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
