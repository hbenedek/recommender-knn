package test.distributed

import org.scalatest._

import funsuite._

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level

import shared.predictions._
import tests.shared.helpers._

class DistributedBaselineTests extends AnyFunSuite with BeforeAndAfterAll {

   val separator = "\t"
   var spark : org.apache.spark.sql.SparkSession = _

   val train2Path = "data/ml-100k/u2.base"
   val test2Path = "data/ml-100k/u2.test"
   var train2 : org.apache.spark.rdd.RDD[shared.predictions.Rating] = null
   var test2 : org.apache.spark.rdd.RDD[shared.predictions.Rating] = null

   override def beforeAll {
       Logger.getLogger("org").setLevel(Level.OFF)
       Logger.getLogger("akka").setLevel(Level.OFF)
       spark = SparkSession.builder()
           .master("local[1]")
           .getOrCreate()
       spark.sparkContext.setLogLevel("ERROR")
       train2 = load(spark, train2Path, separator)
       test2 = load(spark, test2Path, separator)
   }

   // All the functions definitions for the tests below (and the tests in other suites) 
   // should be in a single library, 'src/main/scala/shared/predictions.scala'.

   // Provide tests to show how to call your code to do the following tasks (each in with their own test):
   // each method should be invoked with a single function call. 
   // Ensure you use the same function calls to produce the JSON outputs in
   // src/main/scala/predict/Baseline.scala.
   // Add assertions with the answer you expect from your code, up to the 4th
   // decimal after the (floating) point, on data/ml-100k/u2.base (as loaded above).
   test("Compute global average"){
     val result1 = distributedGlobalAverage(train2)
     assert(within(result1, 3.5264625, 0.0001)) 
     }

   test("Compute user 1 average"){ 
     val result2 = distributedUserAverage(train2, 1)
     assert(within(result2, 3.63302752293578, 0.0001)) 
     }

   test("Compute item 1 average"){ 
      val result3 = distributedItemAverage(train2, 1)
      assert(within(result3, 3.888268156424581, 0.0001)) 
     }

   test("Compute item 1 average deviation"){ 
      val globalAvg = distributedGlobalAverage(train2)
      val allUser = distributedAllUserAverage(train2)
      val allUserBroadcast = spark.sparkContext.broadcast(allUser.collect().toMap.withDefaultValue(globalAvg))

      val result4 = distributedItemDeviation(train2, 1, allUserBroadcast)
      assert(within(result4, 0.30270723414448747, 0.0001)) 
     }

   test("Compute baseline prediction for user 1 on item 1") {
      val globalAvg = distributedGlobalAverage(train2)
      val allUser = distributedAllUserAverage(train2)
      val allUserBroadcast = spark.sparkContext.broadcast(allUser.collect().toMap.withDefaultValue(globalAvg))

      val avgUserOne = allUserBroadcast.value.getOrElse(1,globalAvg)
      val item1AvgDev = distributedItemDeviation(train2, 1, allUserBroadcast)
      val result5 = predict(avgUserOne,item1AvgDev)
      assert(within(result5, 4.046819980619529, 0.0001)) 
      }

   // Show how to compute the MAE on all four non-personalized methods:
   // 1. There should be four different functions, one for each method, to create a predictor
   // with the following signature: ````predictor: (train: Seq[shared.predictions.Rating]) => ((u: Int, i: Int) => Double)````;
   // 2. There should be a single reusable function to compute the MAE on the test set, given a predictor;
   // 3. There should be invocations of both to show they work on the following datasets.
   test("MAE on all four non-personalized methods on data/ml-100k/u2.base and data/ml-100k/u2.test") {
      //GlobalAvg
      val predictor1 = predictorDistributedGlobal(train2, spark)
      val mae1 = evaluateDistributedPredictor(test2, predictor1)
      assert(within(mae1, 0.9489109899999697, 0.0001))

      //UserAvg
      val predictor2 = predictorDistributedUserAverage(train2, spark)
      val mae2 = evaluateDistributedPredictor(test2, predictor2)
      assert(within(mae2, 0.8383401457987351, 0.0001))

      //ItemAvg
      val predictor3 = predictorDistributedItemAverage(train2, spark)
      val mae3 = evaluateDistributedPredictor(test2, predictor3)
      assert(within(mae3, 0.8206951490543668, 0.0001))

      //Baseline
      val predictor4 = predictorDistributedBaseline(train2, spark)
      val mae4 = evaluateDistributedPredictor(test2, predictor4)
      assert(within(mae4, 0.7604467914538644, 0.0001))
   }
}
