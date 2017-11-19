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


case class rawComment(category:Int, comment:String)
case class Comment(comment:String)

object ClassifyComment extends App {
  val sparkConf = new SparkConf()
    .setAppName("SparkClassify")
      .setMaster("local[3]")
//    .setMaster("spark://pyq-master:7077")
    .set("spark.driver.host", "localhost")
    .set("spark.mongodb.input.uri", "mongodb://zc-slave/jd.comment_word")
    .set("spark.executor.memory","2g")
      .set("spark.executor.heartbeatInterval","20000")

  val sc = new SparkContext(sparkConf);

  def getCommentVector(data:RDD[Document]):Dataset[LabeledPoint] = {
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._
    val trainDF=data.map { comment =>
      val words = comment.get("words").asInstanceOf[java.util.ArrayList[String]]
      val wordStr = words.mkString(" ")
      rawComment(comment.get("classify").asInstanceOf[Int], wordStr)
    }.toDF()
//trainDF.show(1)
    val tokenizer = new Tokenizer().setInputCol("comment").setOutputCol("words")
    val wordsData = tokenizer.transform(trainDF)
//wordsData.show(1)
    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures")
                    .setNumFeatures(10000)
    val featurizedData = hashingTF.transform(wordsData)
//featurizedData.show(1)
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)

    val rescaledData = idfModel.transform(featurizedData)
//rescaledData.show(1)
    //转换成Bayes的输入格式

//    var trainDataRaw = rescaledData.select($"category",$"features").map {
//      case Row(label: String, features: Vector) =>
//        LabeledPoint(label.toDouble, Vectors.dense(features.toArray))
//    }

//
    val selector = new ChiSqSelector()
      .setNumTopFeatures(2)
      .setFeaturesCol("features")
      .setLabelCol("category")
      .setOutputCol("selectedFeatures")
//
    val result = selector.fit(rescaledData).transform(rescaledData)
    var trainDataRdd=result.select($"category",$"selectedFeatures").map{
      case Row(label: Int, selectedFeatures: Vector) =>
        LabeledPoint(label.toDouble, Vectors.dense(selectedFeatures.toArray))

    }
//    rescaledData.show(1)
//    print(result.col("selectedFeatures")(1))
//    trainDataRdd.show(1)
    return trainDataRdd

  }

  def getCommentWithoutClassVector(data:RDD[Document]) = {

    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._
    val trainDF=data.map { comment =>
      val words = comment.get("words").asInstanceOf[java.util.ArrayList[String]]
      val wordStr = words.mkString(" ")
      Comment( wordStr)
    }.toDF()

    val tokenizer = new Tokenizer().setInputCol("comment").setOutputCol("words")
    val wordsData = tokenizer.transform(trainDF)

    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures")
      .setNumFeatures(100000)
    val featurizedData = hashingTF.transform(wordsData)

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)

    val rescaledData = idfModel.transform(featurizedData)

    val model =NaiveBayesModel.load("model_naiveBayes10")
    model.transform(rescaledData).select("words","prediction").show(20)



  }
  def naiveBayes(): Unit ={

    val readConfig = ReadConfig(
      Map(
        "uri" -> "mongodb://127.0.0.1:27017",
        "database" -> "jd",
        "collection" -> "word_for_bayes"), Some(ReadConfig(sc)))
    val commentRDD = MongoSpark.load(sc, readConfig)


    val split = commentRDD.randomSplit(Array(0.7,0.3))

    val trainData = getCommentVector(split(0))
    val testData = getCommentVector(split(1))

//trainData.show(1)
//    建立模型
    val model =new NaiveBayes().fit(trainData)
    val predictions = model.transform(testData)
    predictions.show(1)

//    评估模型
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("准确率:"+accuracy.toString)

//    保存模型
    model.write.overwrite().save("model_naiveBayes1withSelector1")

//


  }


  naiveBayes()
//  getCommentWithoutClassVector(split(2))



}
