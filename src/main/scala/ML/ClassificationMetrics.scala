package ML

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.udf
import org.apache.spark.ml.linalg.Vector

object ClassificationMetrics {

  class ClassificationMetrics(val predictions : DataFrame)  {

    private def getFirstElement(x: Vector): Double =  x(0).toFloat
    private val getFirstElementUDF = udf(getFirstElement _)
    private val dataset_size: Float =  predictions.count().toFloat
    private var columnLabelName = "label"
    private var columnPredictionName = "prediction"
/*
    def this(columnLabelName : String, columnPredictionName: String) = {
      this()
      this.columnLabelName = columnLabelName
      this.columnPredictionName = columnPredictionName
    }
*/
    def setColumnLabelName(labelName: String): Unit = {
      this.columnLabelName = labelName
    }

    def setColumnPredictionName(predictionName: String): Unit = {
      this.columnPredictionName = predictionName
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




