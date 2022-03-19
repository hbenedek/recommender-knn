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

class PersonalizedTests extends AnyFunSuite with BeforeAndAfterAll {

   val separator = "\t"
   var spark : org.apache.spark.sql.SparkSession = _

   val train2Path = "data/ml-100k/u2.base"
   val test2Path = "data/ml-100k/u2.test"
   var train2 : Array[shared.predictions.Rating] = null
   var test2 : Array[shared.predictions.Rating] = null

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
   test("Test uniform unary similarities") { 
     // Create predictor with uniform similarities
     val uniformPredictor = computeOnesPredictor(train2)
     val predUser1Item1 = uniformPredictor(1,1)
     // Compute personalized prediction for user 1 on item 1
     assert(within(predUser1Item1, 4.0468, 0.0001))
    
     val uniformMae = evaluatePredictor(test2,uniformPredictor)
     // MAE 
     assert(within(uniformMae, 0.7604, 0.0001))
   } 

   test("Test ajusted cosine similarity") { 
     // Create predictor with adjusted cosine similarities
     val cosinePredictor = computeCosinePredictor(train2)

     // Similarity between user 1 and user 2
     val global = globalAvg(train2)
     val userAverages = computeAllUserAverages(train2).withDefaultValue(global)
     val u1 = preprocessRatings(train2.filter(r => r.user == 1), userAverages)
     val u2 = preprocessRatings(train2.filter(r => r.user == 2), userAverages)
     val adjustedCosineUser1User2 = cosineSimilarity(u1,u2)
     assert(within(adjustedCosineUser1User2, 0.0730, 0.0001))

     // Compute personalized prediction for user 1 on item 1
     val cosinePredUser1Item1 = cosinePredictor(1,1)
     assert(within(cosinePredUser1Item1, 4.0968, 0.0001))

     // MAE 
     val cosineMae = evaluatePredictor(test2, cosinePredictor)
     assert(within(cosineMae, 1.0582, 0.0001))
   }

   test("Test jaccard similarity") { 
     // Create predictor with jaccard similarities
     val jaccardPredictor = computeJaccardPredictor(train2)

     // Similarity between user 1 and user 2
     val u1 = train2.filter(r => r.user == 1)
     val u2 = train2.filter(r => r.user == 2)
     val jaccardUser1User2 = jaccardIndexSimilarity(u1, u2)
     assert(within(jaccardUser1User2, 0.0319, 0.0001))

     // Compute personalized prediction for user 1 on item 1
     val jaccardPredUser1Item1 = jaccardPredictor(1,1)
     assert(within(jaccardPredUser1Item1, 4.0982, 0.0001))

     // MAE
     val jaccardMae = evaluatePredictor(test2, jaccardPredictor)
     assert(within(jaccardMae, 0.7556, 0.0001))
   }
}
