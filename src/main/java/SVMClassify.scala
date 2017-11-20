import ClassifyComment.{getCommentVector, sc}
import com.mongodb.spark.MongoSpark
import com.mongodb.spark.config.ReadConfig
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.rdd.RDD
import org.bson.Document

import scala.collection.JavaConversions._

case class rawComment2(category:String, comment:String)

object SVMClassify extends App {
  val sparkConf = new SparkConf()
    .setAppName("SparkClassify")
    .setMaster("local[3]")
    //    .setMaster("spark://pyq-master:7077")
    .set("spark.driver.host", "localhost")
    .set("spark.mongodb.input.uri", "mongodb://172.19.165.137/jd.all_words_data")
    .set("spark.executor.memory","2g")
    .set("spark.executor.heartbeatInterval","20000")

  val sc = new SparkContext(sparkConf);
  def getAllWords(commentData:Document):Array[AnyRef] = {
      val words= commentData.get("words").asInstanceOf[java.util.ArrayList[String]]
      return words.toArray()
  }
  def getAllWordsCount(commentData:RDD[Document]):Array[(Int,AnyRef)] = {
      val wordsRDD = commentData.flatMap(getAllWords)
      val wordCountRDD = wordsRDD.map(word => (word,1))
      val countRDD = wordCountRDD.reduceByKey((x1,x2)=>x1+x2)
      val wordCount = countRDD.map(weightedEdge => (weightedEdge._2, weightedEdge._1)).sortByKey(false)
      val topWord = wordCount.take(100)
      for( i <- 0 until 100){
         println(topWord(i))
      }
      return topWord
  }
//  def dealWithData(comments:RDD[Document]):Unit ={
//     val fiveRateComment = comments.filter(comment => comment.get("classify").asInstanceOf[Double] == 5.0).repartition(1)
//     val negetiveComment = comments.filter(comment => comment.get("classify").asInstanceOf[Double] <= 5.0).repartition(1)
//     val positiveRateComment = sc.parallelize(fiveRateComment.take(negetiveComment.count().toString().asInstanceOf[Int])).repartition(1)
//     val allComment = negetiveComment.union(positiveRateComment)
//     getCommentVector(allComment)
//  }

  def getCommentVector(data:RDD[Document]):Dataset[LabeledPoint] = {
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._
    val trainDF=data.map { comment =>
      val words = comment.get("words").asInstanceOf[java.util.ArrayList[String]]
      val wordStr = words.mkString(" ")
      var classify = comment.get("classify").asInstanceOf[Int];
      rawComment(classify.toString(), wordStr)

    }.toDF()

    val tokenizer = new Tokenizer().setInputCol("comment").setOutputCol("words")
    val wordsData = tokenizer.transform(trainDF)

    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures")
      .setNumFeatures(10000)
    val featurizedData = hashingTF.transform(wordsData)
    featurizedData.cache()

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)

    val rescaledData = idfModel.transform(featurizedData)


    var trainDataRdd = rescaledData.select($"category",$"features").map {
      case Row(label: String, features: Vector) =>
        LabeledPoint(label.toDouble, Vectors.dense(features.toArray))
    }

    return trainDataRdd

  }
  def wordDeal(): Unit ={

    val readConfig = ReadConfig(
      Map(
        "uri" -> "mongodb://127.0.0.1:27017",
        "database" -> "jd",
        "collection" -> "all_words"), Some(ReadConfig(sc)))
    val commentRDD = MongoSpark.load(sc, readConfig)
    getAllWordsCount(commentRDD)
  }

  def svmClassify(): Unit ={

    val readConfig = ReadConfig(
      Map(
        "uri" -> "mongodb://172.19.165.137:27017",
        "database" -> "jd",
        "collection" -> "all_words_data"), Some(ReadConfig(sc)))
    val commentRDD = MongoSpark.load(sc, readConfig)
    val split = commentRDD.randomSplit(Array(0.7,0.3))

    val trainData = getCommentVector(split(0))
    val testData = getCommentVector(split(1))
    println("------numIterations:5,stepSize:0.1")
    predictModel(trainData,testData,5,0.1)
//    println("------numIterations:10,stepSize:0.1")
//    predictModel(trainData,testData,10,0.1)
//    println("------numIterations:10,stepSize:0.01")
//    predictModel(trainData,testData,10,0.01)

  }

  def predictModel(trainData:Dataset[LabeledPoint],testData:Dataset[LabeledPoint],numIterations:Int,stepSize: Double):Unit={
    val lsvc = new LinearSVC().setMaxIter(numIterations).setRegParam(stepSize)
    val lsvcModel = lsvc.fit(trainData)
    lsvcModel.write.overwrite().save("model_svm2")

    val predictions = lsvcModel.transform(testData)
    predictions.show()
    //    评估模型
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("准确率:numIteration:"+numIterations.toString()+"stepSize:"+stepSize.toString()+accuracy.toString)

    //    保存模型

  }

  def decisionTreeClassify():Unit = {
      val readConfig = ReadConfig(
      Map(
      "uri" -> "mongodb://127.0.0.1:27017",
      "database" -> "jd",
      "collection" -> "all_words_data"), Some(ReadConfig(sc)))
      val commentRDD = MongoSpark.load(sc, readConfig)
      val split = commentRDD.randomSplit(Array(0.7,0.3))

      val trainData = getCommentVector(split(0))
      val testData = getCommentVector(split(1))
      val dt = new DecisionTreeClassifier().setMaxBins(2).setMaxMemoryInMB(512)fit(trainData)
      val predictions = dt.transform(testData)
      predictions.show(5)
      //    评估模型
      val evaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName("accuracy")
      val accuracy = evaluator.evaluate(predictions)
      println("准确率:"+accuracy.toString)


//      val labelIndexer = new StringIndexer()
//        .setInputCol("label")
//        .setOutputCol("indexedLabel")
//        .fit(data)
//      // Automatically identify categorical features, and index them.
//      val featureIndexer = new VectorIndexer()
//        .setInputCol("features")
//        .setOutputCol("indexedFeatures")
//        .setMaxCategories(2) // features with > 4 distinct values are treated as continuous.
//        .fit(data)
//      val Array(trainingData,testData) = data.randomSplit(Array(0.7,0.3))
//      // Train a DecisionTree model.
//      val dt = new DecisionTreeClassifier()
//        .setLabelCol("indexedLabel")
//        .setFeaturesCol("indexedFeatures")
//
//      // Convert indexed labels back to original labels.
//      val labelConverter = new IndexToString()
//        .setInputCol("prediction")
//        .setOutputCol("predictedLabel")
//        .setLabels(labelIndexer.labels)
//
//      // Chain indexers and tree in a Pipeline.
//      val pipeline = new Pipeline()
//        .setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
//
//      // Train model. This also runs the indexers.
//      val model = pipeline.fit(trainingData)
//
//      // Make predictions.
//      val predictions = model.transform(testData)
//
//      // Select example rows to display.
//      predictions.select("predictedLabel", "label", "features").show(5)
//
//      // Select (prediction, true label) and compute test error.
//      val evaluator = new MulticlassClassificationEvaluator()
//        .setLabelCol("indexedLabel")
//        .setPredictionCol("prediction")
//        .setMetricName("accuracy")
//      val accuracy = evaluator.evaluate(predictions)
//      println("Test Error = " + (1.0 - accuracy))
//
//      val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
//      println("Learned classification tree model:\n" + treeModel.toDebugString)


  }
def validateModel():  Unit = {
  val readConfig2 = ReadConfig(
    Map(
      "uri" -> "mongodb://127.0.0.1:27017",
      "database" -> "jd",
      "collection" -> "test_data"), Some(ReadConfig(sc)))
  val dccRDD = MongoSpark.load(sc, readConfig2)
  val testData = getCommentVector(dccRDD)
  val model = LinearSVCModel.load("model_svm");
  model.transform(testData).show(1)
}
//  validateModel()
  decisionTreeClassify()
//  svmClassify()
//  decisionTreeClassify()
//  wordDeal()





}
