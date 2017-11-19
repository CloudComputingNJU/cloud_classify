import com.mongodb.spark.MongoSpark
import com.mongodb.spark.config.ReadConfig
import org.apache.spark.ml.feature._
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.rdd.RDD
import org.bson.Document

import scala.collection.JavaConversions._
//case class Comment(comment:String)

object TestNaiveBayes extends App{


  val sparkConf = new SparkConf()
    .setAppName("ClassifyPrediction")
    .setMaster("local[3]")
    //    .setMaster("spark://pyq-master:7077")
    .set("spark.driver.host", "localhost")
    .set("spark.mongodb.input.uri", "mongodb://zc-slave/jd.word_for_prediction")
    .set("spark.executor.memory","2g")
    .set("spark.executor.heartbeatInterval","20000")

  val sc = new SparkContext(sparkConf);

  def getCommentWithoutClassVector(data:RDD[Document]) = {

    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._
    val trainDF=data.map { comment =>
      val words = comment.get("words").asInstanceOf[java.util.ArrayList[String]]
      val wordStr = words.mkString(" ")
      rawComment(comment.get("classify").asInstanceOf[Int], wordStr)

    }.toDF()

    val tokenizer = new Tokenizer().setInputCol("comment").setOutputCol("words")
    val wordsData = tokenizer.transform(trainDF)

    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures")
      .setNumFeatures(10000)
    val featurizedData = hashingTF.transform(wordsData)

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)

    val rescaledData = idfModel.transform(featurizedData)

    val selector = new ChiSqSelector()
      .setNumTopFeatures(2)
      .setFeaturesCol("features")
      .setLabelCol("category")
      .setOutputCol("selectedFeatures")
    //
    val result = selector.fit(rescaledData).transform(rescaledData)
    var predictDataRdd=result.select($"category",$"selectedFeatures").map{
      case Row(label: Int, selectedFeatures: Vector) =>
        LabeledPoint(label.toDouble, Vectors.dense(selectedFeatures.toArray))

    }
    val model =NaiveBayesModel.load("model_naiveBayes10")
    model.transform(predictDataRdd).select($"label",$"prediction").show(10)



  }


  def testNaiveBayes() :Unit={

    val readTestData = ReadConfig(Map(
      "uri" -> "mongodb://zc-slave:27017",
      "database" -> "jd",
      "collection" -> "word_for_prediction"), Some(ReadConfig(sc)))


    val commentRDD = MongoSpark.load(sc, readTestData)


    getCommentWithoutClassVector(commentRDD)
  }
//
  testNaiveBayes();

}