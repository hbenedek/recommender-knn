package predict

import org.rogach.scallop._
import org.apache.spark.rdd.RDD

import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level

import scala.math
import shared.predictions._


class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
  val train = opt[String](required = true)
  val test = opt[String](required = true)
  val separator = opt[String](default=Some("\t"))
  val num_measurements = opt[Int](default=Some(0))
  val json = opt[String]()
  verify()
}

object Baseline extends App {
  // Remove these lines if encountering/debugging Spark
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)
  val spark = SparkSession.builder()
    .master("local[1]")
    .getOrCreate()
  spark.sparkContext.setLogLevel("ERROR") 

  println("")
  println("******************************************************")

  var conf = new Conf(args) 
  // For these questions, data is collected in a scala Array 
  // to not depend on Spark
  println("Loading training data from: " + conf.train()) 
  val train = load(spark, conf.train(), conf.separator()).collect()
  println("Loading test data from: " + conf.test()) 
  val test = load(spark, conf.test(), conf.separator()).collect()

  //Time the different methods:
  // Using global average
  val measurements_global = (1 to conf.num_measurements()).map(x => timingInMs(() => {
    val out = evaluateGlobal(train, test)// Do everything here from train and test
    out     // Output answer as last value
  }))
  val timings_global = measurements_global.map(t => t._2) // Retrieve the timing measurements
  
  //Using user average
  val measurements_user = (1 to conf.num_measurements()).map(x => timingInMs(() => {
    val out = evaluateUserAverage(train, test)
    out     
  }))
  val timings_user = measurements_user.map(t => t._2) 

  //Using item average
  val measurements_item = (1 to conf.num_measurements()).map(x => timingInMs(() => {
    val out = evaluateItemAverage(train, test)
    out 
  }))
  val timings_item = measurements_item.map(t => t._2)

  //Using baseline prediction method
  val measurements_baseline = (1 to conf.num_measurements()).map(x => timingInMs(() => {
    val out = evaluateBaseline(train, test)
    out
  }))
  val timings_baseline = measurements_baseline.map(t => t._2) 
  // Save answers as JSON
  def printToFile(content: String, 
                  location: String = "./answers.json") =
    Some(new java.io.PrintWriter(location)).foreach{
      f => try{
        f.write(content)
      } finally{ f.close }
  }

  //Needed for predicting average item deviation for item 1 
  //(not needed for timings)
  val user_avgs = (for (uid <- train.map(r=>r.user).distinct; 
                     avgRating = computeUserAvg(uid, train)
                    ) yield (uid, avgRating)).toMap
  
  conf.json.toOption match {
    case None => ; 
    case Some(jsonFile) => {
      var answers = ujson.Obj(
        "Meta" -> ujson.Obj(
          "1.Train" -> ujson.Str(conf.train()),
          "2.Test" -> ujson.Str(conf.test()),
          "3.Measurements" -> ujson.Num(conf.num_measurements())
        ),
        "B.1" -> ujson.Obj(
          "1.GlobalAvg" -> ujson.Num(globalAvg(train)), // Datatype of answer: Double
          "2.User1Avg" -> ujson.Num(computeUserAvg(1, train)),  // Datatype of answer: Double
          "3.Item1Avg" -> ujson.Num(computeItemAvg(1, train)),   // Datatype of answer: Double
          "4.Item1AvgDev" -> ujson.Num(averageItemDeviation(1,train, user_avgs)), // Datatype of answer: Double
          "5.PredUser1Item1" -> ujson.Num(predictRating(1,1,train)) // Datatype of answer: Double
        ),
        "B.2" -> ujson.Obj(
          "1.GlobalAvgMAE" -> ujson.Num(evaluateGlobal(train, test)), // Datatype of answer: Double
          "2.UserAvgMAE" -> ujson.Num(evaluateUserAverage(train, test)),  // Datatype of answer: Double
          "3.ItemAvgMAE" -> ujson.Num(evaluateItemAverage(train, test)),   // Datatype of answer: Double
          "4.BaselineMAE" -> ujson.Num(evaluateBaseline(train, test))   // Datatype of answer: Double
        ),
        "B.3" -> ujson.Obj(
          "1.GlobalAvg" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(timings_global)), // Datatype of answer: Double
            "stddev (ms)" -> ujson.Num(std(timings_global)) // Datatype of answer: Double
          ),
          "2.UserAvg" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(timings_user)), // Datatype of answer: Double
            "stddev (ms)" -> ujson.Num(std(timings_user)) // Datatype of answer: Double
          ),
          "3.ItemAvg" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(timings_item)), // Datatype of answer: Double
            "stddev (ms)" -> ujson.Num(std(timings_item)) // Datatype of answer: Double
          ),
          "4.Baseline" -> ujson.Obj(
            "average (ms)" -> ujson.Num(mean(timings_baseline)), // Datatype of answer: Double
            "stddev (ms)" -> ujson.Num(std(timings_baseline)) // Datatype of answer: Double
          )
        )
      )

      val json = ujson.write(answers, 4)
      println(json)
      println("Saving answers in: " + jsonFile)
      printToFile(json.toString, jsonFile)
    }
  }

  println("")
  spark.close()
}
