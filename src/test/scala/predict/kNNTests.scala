package test.predict

import org.scalatest._
import funsuite._

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level

import shared.predictions._
import tests.shared.helpers._
import ujson._

class kNNTests extends AnyFunSuite with BeforeAndAfterAll {

   val separator = "\t"
   var spark : org.apache.spark.sql.SparkSession = _

   val train2Path = "data/ml-100k/u2.base"
   val test2Path = "data/ml-100k/u2.test"
   var train2 : Array[shared.predictions.Rating] = null
   var test2 : Array[shared.predictions.Rating] = null

   var adjustedCosine : Map[Int, Map[Int, Double]] = null

   override def beforeAll {
       Logger.getLogger("org").setLevel(Level.OFF)
       Logger.getLogger("akka").setLevel(Level.OFF)
       spark = SparkSession.builder()
           .master("local[1]")
           .getOrCreate()
       spark.sparkContext.setLogLevel("ERROR")

       // For these questions, train and test are collected in a scala Array
       // to not depend on Spark
       train2 = load(spark, train2Path, separator).collect()
       test2 = load(spark, test2Path, separator).collect()
   }

   // All the functions definitions for the tests below (and the tests in other suites) 
   // should be in a single library, 'src/main/scala/shared/predictions.scala'.

   // Provide tests to show how to call your code to do the following tasks.
   // Ensure you use the same function calls to produce the JSON outputs in
   // src/main/scala/predict/Baseline.scala.
   // Add assertions with the answer you expect from your code, up to the 4th
   // decimal after the (floating) point, on data/ml-100k/u2.base (as loaded above).
   test("kNN predictor with k=10") { 
     // Create predictor on train2
     val knnPredictor = computeKnnPredictor(train2, 10)

     // Similarity between user 1 and itself
     val cosineMap = similarityMapper(train2, cosineSimilarity)
     val user1Top10 = knn(1,10, cosineMap)
     val user1SelfSim = user1Top10(1)
     assert(within(user1SelfSim, 0.0000, 0.0001))
 
     // Similarity between user 1 and 864
     val user1User864Sim = user1Top10(864)
     assert(within(user1User864Sim, 0.2423, 0.0001))

     // Similarity between user 1 and 886
     val user1User886Sim = user1Top10(886)
     assert(within(user1User886Sim, 0.0000, 0.0001))

     // Prediction user 1 and item 1
     val predUser1Item1 = knnPredictor(1,1)
     assert(within(predUser1Item1, 4.3191, 0.0001))

     // MAE on test2
     val knnMae = evaluatePredictor(test2, knnPredictor)
     assert(within(knnMae, 0.8287, 0.0001))
   } 

   test("kNN Mae") {
     // Compute MAE for k around the baseline MAE
     val knnPredictor = computeKnnPredictor(train2, 100)
     val baselinePredictor = predictorBaseline(train2)

     val knnMae = evaluatePredictor(test2, knnPredictor)
     val baselineMae = evaluatePredictor(test2, baselinePredictor)
     // Ensure the MAEs are indeed lower/higher than baseline
     assert(knnMae < baselineMae)
   }
}
