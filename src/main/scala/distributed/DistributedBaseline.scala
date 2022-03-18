package distributed

import org.rogach.scallop._
import org.apache.spark.rdd.RDD
import ujson._

import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level

import scala.math
import shared.predictions._

class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
  val train = opt[String](required = true)
  val test = opt[String](required = true)
  val separator = opt[String](default=Some("\t"))
  val master = opt[String](default=Some(""))
  val num_measurements = opt[Int](default=Some(0))
  val json = opt[String]()
  verify()
}

object DistributedBaseline extends App {
  var conf = new Conf(args) 

  // Remove these lines if encountering/debugging Spark
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)
  val spark = if (conf.master() != "") {
    SparkSession.builder().master(conf.master()).getOrCreate()
  } else {
    SparkSession.builder().getOrCreate()
  }
  spark.sparkContext.setLogLevel("ERROR") 

  println("")
  println("******************************************************")

  // sbt "runMain distributed.DistributedBaseline --train data/ml-25m/r2.train --test data/ml-25m/r2.test  --separator , --json distributed-25m-4.json --master local[4]" --num_measurements 3
  println("Loading training data from: " + conf.train()) 
  val train = load(spark, conf.train(), conf.separator())
  println("Loading test data from: " + conf.test()) 
  val test = load(spark, conf.test(), conf.separator())

  val measurements = (1 to conf.num_measurements()).map(x => timingInMs(() => {
      val predictor4 = predictorDistributedBaseline(train, spark)
      val mae4 = evaluateDistributedPredictor(test, predictor4)
      mae4
  }))
  val timings = measurements.map(t => t._2)
  val mae = mean(measurements.map(t => t._1))

  val globalAvg = distributedGlobalAverage(train)
  val avgUserOne = distributedUserAverage(train, 1)
  val avgItemOne = distributedItemAverage(train, 1)

  val allUser = distributedAllUserAverage(train)
  val allUserBroadcast = spark.sparkContext.broadcast(allUser.collect().toMap.withDefaultValue(globalAvg))

  val allItemDev = distributedAllItemDeviation(train, allUserBroadcast, globalAvg)
  val allItemDevBroadcast = spark.sparkContext.broadcast(allItemDev.collect().toMap)

  val item1AvgDev = allItemDevBroadcast.value.getOrElse(1,globalAvg)
  val predUser1Item1 = predict(avgUserOne,item1AvgDev)
  //val mae = distributedBaselineMAE(test, allUserBroadcast, allItemDevBroadcast, globalAvg)



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
          "3.Master" -> conf.master(),
          "4.Measurements" -> conf.num_measurements()
        ),
        "D.1" -> ujson.Obj(
          "1.GlobalAvg" -> ujson.Num(globalAvg), // Datatype of answer: Double
          "2.User1Avg" -> ujson.Num(avgUserOne),  // Datatype of answer: Double
          "3.Item1Avg" -> ujson.Num(avgItemOne),   // Datatype of answer: Double
          "4.Item1AvgDev" -> ujson.Num(item1AvgDev), // Datatype of answer: Double,
          "5.PredUser1Item1" -> ujson.Num(predUser1Item1), // Datatype of answer: Double
          "6.Mae" -> ujson.Num(mae) // Datatype of answer: Double
        ),
        "D.2" -> ujson.Obj(
          "1.DistributedBaseline" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(timings)), // Datatype of answer: Double
            "stddev (ms)" -> ujson.Num(std(timings)) // Datatype of answer: Double
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
