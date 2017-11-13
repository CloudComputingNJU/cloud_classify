import com.mongodb.spark.MongoSpark
import com.mongodb.spark.config.ReadConfig
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{StructType,StructField,StringType};

case class rawComment(category:Int, comment:String)

object ClassifyComment extends App {
  val sparkConf = new SparkConf()
    .setAppName("SparkClassify")
    .setMaster("local[2]")
    .set("spark.driver.host", "localhost")
    .set("spark.mongodb.input.uri", "mongodb://zc-slave/jd.comment_word")

  def getCommentVector(){
    val sc = new SparkContext(sparkConf);
    val readConfig = ReadConfig(
    Map(
      "uri" -> "mongodb://zc-slave:27017",
      "database" -> "jd",
      "collection" -> "comment_word"), Some(ReadConfig(sc)))
    val commentRDD = MongoSpark.load(sc, readConfig)
    import sqlContext.implicits._
    val schemaString = "classify words"
    val schema = StructType(schemaString.split(" ").map(fieldName=>StructField(fieldName,StringType,true)))
    val sourceDF = commentRDD.map (comment =>
      rawComment(comment.get("classify").asInstanceOf[Int], comment.get("words").asInstanceOf[String])
    )
    sourceDF.count()
    SparkSession.
    val peopleRDD = SparkSession.sparkContext
      .textFile("file:/E:/scala_workspace/z_spark_study/people.txt",2)
      .map( x => x.split(",")).map( x => Person(x(0),x(1).trim().toInt)).toDF()
    peopleRDD

  }
  }


  getCommentVector()
}
