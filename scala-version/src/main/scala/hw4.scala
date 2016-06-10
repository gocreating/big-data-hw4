package big.data

import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.SQLContext

object main {
  val FIELD_YEAR = 0
  val FIELD_MONTH = 1
  val FIELD_DAY_OF_MONTH = 2
  val FIELD_DAY_OF_WEEK = 3
  val FIELD_DEP_TIME = 4
  val FIELD_CRS_DEP_TIME = 5
  val FIELD_ARR_TIME = 6
  val FIELD_CRS_ARR_TIME = 7
  val FIELD_UNIQUE_CARRIER = 8
  val FIELD_FLIGHT_NUM = 9
  val FIELD_TAIL_NUM = 10
  val FIELD_ACTUAL_ELAPSED_TIME = 11
  val FIELD_CRS_ELAPSED_TIME = 12
  val FIELD_AIR_TIME = 13
  val FIELD_ARR_DELAY = 14
  val FIELD_DEP_DELAY = 15
  val FIELD_ORIGIN = 16
  val FIELD_DEST = 17
  val FIELD_DISTANCE = 18
  val FIELD_TAXI_IN = 19
  val FIELD_TAXI_OUT = 20
  val FIELD_CANCELLED = 21
  val FIELD_CANCELLATION_CODE = 22
  val FIELD_DIVERTED = 23
  val FIELD_CARRIER_DELAY = 24
  val FIELD_WEATHER_DELAY = 25
  val FIELD_NAS_DELAY = 26
  val FIELD_SECURITY_DELAY = 27
  val FIELD_LATE_AIRCRAFT_DELAY = 28

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("main")
    val sc = new SparkContext(conf)

    // prepare training data
    val RDDTrainData = sc.textFile(Array(
      // "1987.csv",
      // "1988.csv",
      // "1989.csv",
      // "1990.csv",
      // "1991.csv",
      // "1992.csv",
      // "1993.csv",
      // "1994.csv",
      // "1995.csv",
      // "1996.csv",
      // "1997.csv",
      // "1998.csv",
      // "1999.csv",
      // "2000.csv",
      // "2001.csv",
      // "2002.csv",
      // "2003.csv",
      // "2004.csv",
      // "2005.csv",
      // "2006.csv",
      "2007.csv"
    ).mkString(","))
    val RDDTrainHeader = RDDTrainData.first()
    val trainData = RDDTrainData
      .filter(_ != RDDTrainHeader)
      .map(_.split(","))
      .map(parts =>
        LabeledPoint(
          parts(FIELD_CANCELLED).toDouble,
          Vectors.dense(parts(FIELD_FLIGHT_NUM).toDouble)
        )
      )

    // prepare testing data
    val RDDTestData = sc.textFile("2008.csv")
    val RDDTestHeader = RDDTestData.first()
    val testData = RDDTestData
      .filter(_ != RDDTestHeader)
      .map(_.split(","))
      .map(parts =>
        LabeledPoint(
          parts(FIELD_CANCELLED).toDouble,
          Vectors.dense(parts(FIELD_FLIGHT_NUM).toDouble)
        )
      )

    // do prediction
    val numClasses = 100
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "gini"
    val maxDepth = 5
    val maxBins = 32
    val model = DecisionTree.trainClassifier(
      trainData,
      numClasses,
      categoricalFeaturesInfo,
      impurity,
      maxDepth,
      maxBins
    )

    val predictionData = testData.map(d =>
      (d.label, model.predict(d.features)))

    // evaluate error rate
    val errorCount = predictionData
      .filter(d => d._1 != d._2)
      .count()
    val totalCount = predictionData.count()
    println("error rate = " + errorCount + " / " + totalCount + " = " + errorCount.toDouble / totalCount.toDouble)
  }
}
