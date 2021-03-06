package predict

import org.rogach.scallop._
import org.apache.spark.rdd.RDD
import ujson._

import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level

import scala.math
import shared.predictions._


class kNNConf(arguments: Seq[String]) extends ScallopConf(arguments) {
  val train = opt[String](required = true)
  val test = opt[String](required = true)
  val separator = opt[String](default=Some("\t"))
  val num_measurements = opt[Int](default=Some(0))
  val json = opt[String]()
  verify()
}

object kNN extends App {
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


  val measurements = (1 to conf.num_measurements()).map(x => timingInMs(() => {
    val predictor = computeKnnPredictor(train, 300) // Do everything here from train and test
    val out = evaluatePredictor(test, predictor)
    out        // Output answer as last value
  }))
  val timings = measurements.map(t => t._2) // Retrieve the timing measurements

  val cosineMap = similarityMapper(train, cosineSimilarity)
  
  val user1Top10 = knn(1,10, cosineMap)
  val user1SelfSim = user1Top10(1)
  val user1User864Sim = user1Top10(864)
  val user1User886Sim = user1Top10(886)

  val knnPredictor = computeKnnPredictor(train, 10)
  val predUser1Item1 = knnPredictor(1,1)

  val ks = List(10,100,50,100,200,300,400,800,943)
  val maes = evaluateKValues(train, test, ks).toList.map{case (k,m)=>List(k,m)}

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
          "1.Train" -> conf.train(),
          "2.Test" -> conf.test(),
          "3.Measurements" -> conf.num_measurements()
        ),
        "N.1" -> ujson.Obj(
          "1.k10u1v1" -> ujson.Num(user1SelfSim), // Similarity between user 1 and user 1 (k=10)
          "2.k10u1v864" -> ujson.Num(user1User864Sim), // Similarity between user 1 and user 864 (k=10)
          "3.k10u1v886" -> ujson.Num(user1User886Sim), // Similarity between user 1 and user 886 (k=10)
          "4.PredUser1Item1" -> ujson.Num(predUser1Item1) // Prediction of item 1 for user 1 (k=10)
        ),
        "N.2" -> ujson.Obj(
          "1.kNN-Mae" -> maes
        ),
        "N.3" -> ujson.Obj(
          "1.kNN" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(timings)),
            "stddev (ms)" -> ujson.Num(std(timings))
          )
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
