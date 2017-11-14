import com.mongodb.spark.MongoSpark
import com.mongodb.spark.config.ReadConfig
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.collection.JavaConversions._

case class rawComment(category:String, comment:String)

object ClassifyComment extends App {
  val sparkConf = new SparkConf()
    .setAppName("SparkClassify")
    .setMaster("local[2]")
    .set("spark.driver.host", "localhost")
    .set("spark.mongodb.input.uri", "mongodb://zc-slave/jd.comment_word")
  val sc = new SparkContext(sparkConf);
  def getCommentVector(readConfig: ReadConfig) = {

    val commentRDD = MongoSpark.load(sc, readConfig)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._

    val sourceDF = commentRDD.map { comment =>
      val words = comment.get("words").asInstanceOf[java.util.ArrayList[String]]
      val wordStr = words.mkString(" ")
      rawComment(comment.get("classify").asInstanceOf[Int].toString(), wordStr)
    }.toDF()
    val tokenizer = new Tokenizer().setInputCol("comment").setOutputCol("words")
    val wordsData = tokenizer.transform(sourceDF)

    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures")
                    .setNumFeatures(10000000)
    val featurizedData = hashingTF.transform(wordsData)

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)

    val rescaledData = idfModel.transform(featurizedData)
    rescaledData.select("category", "features").show()
    //转换成Bayes的输入格式

    var trainDataRdd = rescaledData.select("category","features").map {
      case Row(label: String, features: Vector) =>
        (label.toDouble, Vectors.dense(features.toArray))
    }
//    return trainDataRdd

  }

  def naiveBayes(): Unit ={
    val readTrainData = ReadConfig(
      Map(
        "uri" -> "mongodb://zc-slave:27017",
        "database" -> "jd",
        "collection" -> "word_test"), Some(ReadConfig(sc)))
    val readTestData = ReadConfig(Map(
      "uri" -> "mongodb://zc-slave:27017",
      "database" -> "jd",
      "collection" -> "test_word"), Some(ReadConfig(sc)))
    getCommentVector(readTrainData)
//    val trainData = getCommentVector(readTrainData)
//    trainData.show()
//    val testData = getCommentVector(readTestData)
//    testData.show()
//    val model = new NaiveBayes()

  }


  naiveBayes()

}
