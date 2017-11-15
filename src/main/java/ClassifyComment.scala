import com.mongodb.spark.MongoSpark
import com.mongodb.spark.config.ReadConfig
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.rdd.RDD

import scala.collection.JavaConversions._

case class rawComment(category:String, comment:String)

object ClassifyComment extends App {
  val sparkConf = new SparkConf()
    .setAppName("SparkClassify")
     // .setMaster("local[3]")
    .setMaster("spark://pyq-master:7077")
    .set("spark.driver.host", "localhost")
    .set("spark.mongodb.input.uri", "mongodb://zc-slave/jd.comment_word")
    .set("spark.executor.memory","2g")
      .set("spark.executor.heartbeatInterval","20000")

  val sc = new SparkContext(sparkConf);
  sc.addJar("/home/chenzhang/work/update/cloud_classify/target/cloud_classify-1.0-SNAPSHOT.jar")
  sc.addJar("/home/chenzhang/work/update/cloud_classify/target/dependency/mongo-java-driver-3.4.2.jar")
  sc.addJar("/home/chenzhang/work/update/cloud_classify/target/dependency/mongo-spark-connector_2.11-2.2.0.jar")
  def getCommentVector(readConfig: ReadConfig):Dataset[LabeledPoint] = {

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

    //转换成Bayes的输入格式

    var trainDataRdd = rescaledData.select($"category",$"features").map {
      case Row(label: String, features: Vector) =>
        LabeledPoint(label.toDouble, Vectors.dense(features.toArray))
    }
//    trainDataRdd.show(10)
    return trainDataRdd

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
//    getCommentVector(readTrainData)
    val trainData = getCommentVector(readTrainData)
    trainData.show()
//    val testData = getCommentVector(readTestData)
////    testData.show()
    //建立模型
    val model =new NaiveBayes().fit(trainData)
    val predictions = model.transform(trainData)
    predictions.show()

    //评估模型
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("准确率:"+accuracy)

  }


  naiveBayes()
//  getCommentVector()

}
