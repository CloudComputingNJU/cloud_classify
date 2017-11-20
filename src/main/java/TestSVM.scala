import com.mongodb.spark.MongoSpark
import com.mongodb.spark.config.ReadConfig
import org.apache.spark.ml.classification.{LinearSVCModel}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.{HashingTF, IDF, LabeledPoint, Tokenizer}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Row}
import org.bson.Document
import org.apache.spark.ml.classification._


//case class rawComment2(category:String, comment:String)

object TestSVM extends App{
  val sparkConf = new SparkConf()
    .setAppName("SparkClassify")
    .setMaster("local[3]")
    //    .setMaster("spark://pyq-master:7077")
    .set("spark.driver.host", "localhost")
    .set("spark.mongodb.input.uri", "mongodb://zc-slave/jd.words_Prediction")
    .set("spark.executor.memory","2g")
    .set("spark.executor.heartbeatInterval","20000")

  val sc = new SparkContext(sparkConf);

  def getCommentVector(data:RDD[Document]) {
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    import sqlContext.implicits._
    import collection.JavaConversions._
    val trainDF=data.map { comment =>
      val words = comment.get("words").asInstanceOf[java.util.ArrayList[String]]
      val wordStr = words.mkString(" ")
      var classify = comment.get("classify").asInstanceOf[Double]

      rawComment2(classify.toString(), wordStr)
    }.toDF()

    val tokenizer = new Tokenizer().setInputCol("comment").setOutputCol("words")
    val wordsData = tokenizer.transform(trainDF)

    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures")
      .setNumFeatures(100000)
    val featurizedData = hashingTF.transform(wordsData)

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)

    val rescaledData = idfModel.transform(featurizedData)


//    var trainDataRdd = rescaledData.select($"category",$"features").map {
//      case Row(label: String, features: Vector) =>
//        LabeledPoint(label.toDouble, Vectors.dense(features.toArray))
//    }

    val model =LinearSVCModel.load("model_svm")
    model.transform(rescaledData).select("words","prediction").show(20)

  }

  def testSVM() :Unit={

    val readTestData = ReadConfig(Map(
      "uri" -> "mongodb://zc-slave:27017",
      "database" -> "jd",
      "collection" -> "words_Prediction"), Some(ReadConfig(sc)))


    val commentRDD = MongoSpark.load(sc, readTestData)


   getCommentVector(commentRDD)


  }

  testSVM();
}
