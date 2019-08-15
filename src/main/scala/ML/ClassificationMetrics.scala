package ML

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.udf
import org.apache.spark.ml.linalg.Vector

object ClassificationMetrics {

  type ClassificationMetricsObject = ClassificationMetrics.ClassificationMetrics

  class ClassificationMetrics  {

    private def getFirstElement(x: Vector): Double =  x(0).toFloat
    private val getFirstElementUDF = udf(getFirstElement _)
    private var predictions: DataFrame = null
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

    def getPrediction: DataFrame = predictions
    def fit(predictionsDF: DataFrame): ClassificationMetricsObject = {
      this.dataset_size = predictionsDF.count().toFloat
      this.predictions = predictionsDF
      this
    }

    def Accuracy() : Float = {
      this.predictions.filter(col(columnLabelName) === col(columnPredictionName)).count().toFloat / this.dataset_size
    }

    def ConfusionMatrix() : List[Float] = {
      val falsePositive : Float = this.predictions.filter((col(columnLabelName)===0).and(col(columnPredictionName)===1)).count().toFloat
      val truePositive : Float = this.predictions.filter((col(columnLabelName)===1).and(col(columnPredictionName)===1)).count().toFloat
      val falseNegative : Float = this.predictions.filter((col(columnLabelName)===1).and(col(columnPredictionName)===0)).count().toFloat
      val trueNegative : Float = this.predictions.filter((col(columnLabelName)===0).and(col(columnPredictionName)===0)).count().toFloat
      List(truePositive, falsePositive, falseNegative, trueNegative)
    }

    def computeCustomGain(gain_true_positive: Float, cost_false_positive: Float, threshold: Float) : Float = {
      val falsePositive: Float = this.predictions.filter((col(columnLabelName) === 0).and(getFirstElementUDF(col("probability")) < threshold)).count().toFloat
      val truePositive: Float = this.predictions.filter((col(columnLabelName) === 1).and(getFirstElementUDF(col("probability")) < threshold)).count().toFloat
      truePositive * gain_true_positive - falsePositive * cost_false_positive
    }

  }

}




