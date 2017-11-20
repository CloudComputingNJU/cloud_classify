import com.mongodb.spark.MongoSpark
import com.mongodb.spark.config.ReadConfig
import org.apache.spark.ml.feature._
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.rdd.RDD
import org.bson.Document

import scala.collection.JavaConversions._


case class rawComment(category:String, comment:String)
case class Comment(comment:String)

object ClassifyComment extends App {
  val sparkConf = new SparkConf()
    .setAppName("SparkClassify")
      .setMaster("local[3]")
//    .setMaster("spark://pyq-master:7077")
    .set("spark.driver.host", "localhost")
<<<<<<< HEAD
    .set("spark.mongodb.input.uri", "mongodb://127.0.0.1/jd.all_words")
=======
    .set("spark.mongodb.input.uri", "mongodb://172.19.165.137/jd.test_data")
>>>>>>> 903de5b0e748ee202f0dee7799f113ea8292f180
    .set("spark.executor.memory","2g")
    .set("spark.executor.heartbeatInterval","20000")

  val sc = new SparkContext(sparkConf);

  def getCommentVector(data:RDD[Document]):Dataset[LabeledPoint] = {
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._
    val trainDF=data.map { comment =>
      val words = comment.get("words").asInstanceOf[java.util.ArrayList[String]]
      val wordStr = words.mkString(" ")
<<<<<<< HEAD
      var classify = comment.get("classify").asInstanceOf[Int];
      rawComment(classify.toString(), wordStr)
=======
      rawComment(comment.get("classify").asInstanceOf[Int].toString, wordStr)
>>>>>>> 903de5b0e748ee202f0dee7799f113ea8292f180
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

    var trainDataRdd = rescaledData.select($"category",$"features").map {
      case Row(label: String, features: Vector) =>
        LabeledPoint(label.toDouble, Vectors.dense(features.toArray))
    }

<<<<<<< HEAD
=======
//
//    val selector = new ChiSqSelector()
//      .setNumTopFeatures(2)
//      .setFeaturesCol("features")
//      .setLabelCol("category")
//      .setOutputCol("selectedFeatures")
////
//    val result = selector.fit(rescaledData).transform(rescaledData)
//    var trainDataRdd=result.select($"category",$"selectedFeatures").map{
//      case Row(label: Double, selectedFeatures: Vector) =>
//        LabeledPoint(label, Vectors.dense(selectedFeatures.toArray))
//
//    }
//    rescaledData.show(1)
//    print(result.col("selectedFeatures")(1))
//    trainDataRdd.show(1)
>>>>>>> 903de5b0e748ee202f0dee7799f113ea8292f180
    return trainDataRdd

  }


  def naiveBayes(): Unit ={

    val readConfig = ReadConfig(
      Map(
        "uri" -> "mongodb://127.0.0.1:27017",
        "database" -> "jd",
        "collection" -> "all_words_cut"), Some(ReadConfig(sc)))
    val commentRDD = MongoSpark.load(sc, readConfig)


    val split = commentRDD.randomSplit(Array(0.7,0.3))

    val trainData = getCommentVector(split(0))
    val testData = getCommentVector(split(1))

//    val readConfig2 = ReadConfig(
//      Map(
//        "uri" -> "mongodb://127.0.0.1:27017",
//        "database" -> "jd",
//        "collection" -> "test_data"), Some(ReadConfig(sc)))
//    val dccRDD = MongoSpark.load(sc, readConfig2)
//    val testData = getCommentVector(dccRDD)

//trainData.show(1)
//    建立模型
    val model =new NaiveBayes().setModelType("multinomial").setSmoothing(1.0).fit(trainData)
    val predictions = model.transform(testData)
    predictions.show(100)

//    评估模型
<<<<<<< HEAD
//    val evaluator = new MulticlassClassificationEvaluator()
//      .setLabelCol("label")
//      .setPredictionCol("prediction")
//      .setMetricName("accuracy")
//    val accuracy = evaluator.evaluate(predictions)
//    println("准确率:"+accuracy.toString)
    val evaluator = new BinaryClassificationEvaluator()
    val accuracy = evaluator.evaluate(predictions)
//    保存模型
    model.write.overwrite().save("model_naiveBayes")

//    getCommentWithoutClassVector(split(2))
=======
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
//      .setMetricName("accuracy")
//    val accuracy = evaluator.evaluate(predictions)
    val weightedPrecision=evaluator.setMetricName("weightedPrecision").evaluate(predictions);

    println("精度:"+weightedPrecision.toString)

//    println("准确率:"+accuracy.toString)

//    保存模型
//    model.write.overwrite().save("model_naiveBayes1BalanceData")

//
>>>>>>> 903de5b0e748ee202f0dee7799f113ea8292f180


  }

<<<<<<< HEAD
  def validateModel():  Unit = {
    val readConfig2 = ReadConfig(
      Map(
        "uri" -> "mongodb://127.0.0.1:27017",
        "database" -> "jd",
        "collection" -> "test_data"), Some(ReadConfig(sc)))
    val dccRDD = MongoSpark.load(sc, readConfig2)
    val testData = getCommentVector(dccRDD)
    val model = NaiveBayesModel.load("model_naiveBayes");
    model.transform(testData).show(3)
  }
//  naiveBayes()
//  validateModel()


  naiveBayes()
//  getCommentWithoutClassVector(split(2))




}
