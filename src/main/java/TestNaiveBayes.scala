//import com.mongodb.spark.MongoSpark
//import com.mongodb.spark.config.ReadConfig
//import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
//import org.apache.spark.sql.{DataFrame, Dataset, Row}
//import org.apache.spark.{SparkConf, SparkContext}
//import org.apache.spark.sql.types.{StringType, StructField, StructType}
//import org.apache.spark.ml.linalg.Vector
//import org.apache.spark.ml.linalg.Vectors
//import org.apache.spark.ml.feature.LabeledPoint
//import org.apache.spark.ml.classification.NaiveBayes
//import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
//import org.apache.spark.rdd.RDD
//import org.bson.Document
//
//import scala.collection.JavaConversions._
//case class Comment(comment:String)
//
//object TestNaiveBayes extends App{
//
//
//  def getCommentWithoutClassVector(data:RDD[Document]):Dataset[LabeledPoint] = {
//
//    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
//    import sqlContext.implicits._
//    val trainDF=data.map { comment =>
//      val words = comment.get("words").asInstanceOf[java.util.ArrayList[String]]
//      val wordStr = words.mkString(" ")
//      Comment( wordStr)
//    }.toDF()
//
//    val tokenizer = new Tokenizer().setInputCol("comment").setOutputCol("words")
//    val wordsData = tokenizer.transform(trainDF)
//
//    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures")
//      .setNumFeatures(500000)
//    val featurizedData = hashingTF.transform(wordsData)
//
//    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
//    val idfModel = idf.fit(featurizedData)
//
//    val rescaledData = idfModel.transform(featurizedData)
//
//    //转换成Bayes的输入格式
//
//    var trainDataRdd = rescaledData.select($"category",$"features").map {
//      case Row(label: String, features: Vector) =>
//        LabeledPoint(label.toDouble, Vectors.dense(features.toArray))
//    }
//    return trainDataRdd
//
//  }
////  def testNaiveBayes() :Unit={
////
////    val readTestData = ReadConfig(Map(
////      "uri" -> "mongodb://zc-slave:27017",
////      "database" -> "jd",
////      "collection" -> "test_word"), Some(ReadConfig(sc)))
////
////
////    val testData= getCommentWithoutClassVector(readTestData)
////
////    NaiveBayesModel.load("model");
////  }
////
////  testNaiveBayes();
//
//}