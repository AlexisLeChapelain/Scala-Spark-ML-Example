package ML

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.udf
import org.apache.spark.ml.linalg.Vector
import scala.math.sqrt

object ClassificationMetrics {

  type ClassificationMetricsObject = ClassificationMetrics.ClassificationMetrics

  class ClassificationMetrics extends java.io.Serializable   {

    private def getFirstElement(x: Vector): Double =  x(0).toFloat
    private val getFirstElementUDF = udf(getFirstElement _)
    private var predictions: DataFrame = _
    private var dataset_size: Float =  0
    private var columnLabelName: String = "label"
    private var columnPredictionName: String = "prediction"


    def getColumnLabelName: String = this.columnLabelName
    def setColumnLabelName(labelName: String): ClassificationMetricsObject  = {
      this.columnLabelName = labelName
      this
    }

    def getColumnPredictionName: String = columnPredictionName
    def setColumnPredictionName(predictionName: String): ClassificationMetricsObject = {
      this.columnPredictionName = predictionName
      this
    }

    def getPredictionDataFrame: DataFrame = predictions
    def fit(predictionsDF: DataFrame): ClassificationMetricsObject = {
      this.dataset_size = predictionsDF.count().toFloat
      this.predictions = predictionsDF.withColumn("probabilityTrue", getFirstElementUDF(col("probability")))
      this
    }

    def accuracy() : Float = {
      this.predictions.filter(col(columnLabelName) === col(columnPredictionName)).count().toFloat / this.dataset_size
    }

    def confusionMatrix() : Map[String,Float] = {
      val falsePositive : Float = this.predictions.filter((col(columnLabelName)===0).and(col(columnPredictionName)===1)).count().toFloat
      val truePositive : Float = this.predictions.filter((col(columnLabelName)===1).and(col(columnPredictionName)===1)).count().toFloat
      val falseNegative : Float = this.predictions.filter((col(columnLabelName)===1).and(col(columnPredictionName)===0)).count().toFloat
      val trueNegative : Float = this.predictions.filter((col(columnLabelName)===0).and(col(columnPredictionName)===0)).count().toFloat
      Map("True Positive" -> truePositive, "False Positive" -> falsePositive, "False Negative" -> falseNegative, "True negative" -> trueNegative)
    }

    def computeCustomGain(gain_true_positive: Float, cost_false_positive: Float, threshold: Float) : Float = {
      val falsePositive: Float = this.predictions.filter((col(columnLabelName) === 0).and(col("probabilityTrue") < threshold)).count().toFloat
      val truePositive: Float = this.predictions.filter((col(columnLabelName) === 1).and(col("probabilityTrue") < threshold)).count().toFloat
      truePositive * gain_true_positive - falsePositive * cost_false_positive
    }

    def accuracyBenchmark() : Float = {
      val meanAccuracy = predictions.filter(col(columnLabelName) === 1).count().toFloat / predictions.count().toFloat
      if (meanAccuracy>0.5) meanAccuracy else 1- meanAccuracy
    }

    def logLoss(): Float = {
      val logloss_df = predictions.withColumn("logloss", - (col(columnLabelName)*log(col("probabilityTrue")) + (lit(1)-col(columnLabelName))*(log(lit(1)-col("probabilityTrue"))))).select("logloss")
      //val logloss_df = this.predictions.withColumn("logloss", log(col("probabilityTrue")))
      logloss_df.agg(avg(col("logloss")).as("logloss")).first().getDouble(0).toFloat
    }

    def matthewsCorrelationCoefficient(): Float = {
      val falsePositive : Float = predictions.filter((col(columnLabelName)===0).and(col(columnPredictionName)===1)).count().toFloat
      val truePositive : Float = predictions.filter((col(columnLabelName)===1).and(col(columnPredictionName)===1)).count().toFloat
      val falseNegative : Float = predictions.filter((col(columnLabelName)===1).and(col(columnPredictionName)===0)).count().toFloat
      val trueNegative : Float = predictions.filter((col(columnLabelName)===0).and(col(columnPredictionName)===0)).count().toFloat
      val numerator = ((truePositive * trueNegative) - (falsePositive * falseNegative))
      val denominator = sqrt((truePositive+falsePositive)*(truePositive+falseNegative)*(trueNegative+falsePositive)*(trueNegative+falseNegative))
      (numerator / denominator).toFloat
    }

    def classificationReport(): Map[String, Any] = {
      Map("accuracy" -> this.accuracy() , "accuracy benchmark" -> this.accuracyBenchmark(),
          "confusion matrix" -> this.confusionMatrix, "logLoss" -> this.logLoss(),
          "matthewsCorrelationCoefficient" -> this.matthewsCorrelationCoefficient())
    }


  }

}




