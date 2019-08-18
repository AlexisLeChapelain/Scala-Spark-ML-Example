package ML

import ClassificationMetrics.ClassificationMetrics

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.functions.lit
import org.apache.spark.ml.feature.Imputer
import org.apache.spark.ml.feature.OneHotEncoderEstimator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.log4j.Logger
import org.apache.log4j.Level

import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.mllib.regression.LassoWithSGD

import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel


object ML {

  // Turn off useless log
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  def main(args: Array[String]) = {

    println("Start")

    // Set up Spark
    val sparkConf = new SparkConf()
    val spark = SparkSession.builder.master("local").appName("ML on Scala / Spark").config(sparkConf).getOrCreate()

    import spark.implicits._

    // Loading data
    val census_data : DataFrame = spark.read.format("csv")
      .option("sep", ",")
      .option("inferSchema", "true")
      .option("header", "true")
      .load("/Users/az02234/Documents/Personnal_Git/Scala-Spark-ML-Example/src/main/resources/census_data.csv")

    // Some description
    census_data.show(5)
    println(census_data.count())
    census_data.printSchema()


    // Set list of column
    val target = "income"
    val categorical_variable_string = Array("workclass", "education", "occupation", "relationship", "sex", "marital_status", "native_country")
    val continuous_variable = Array("year_education", "capital_gain", "capital_loss", "hours_per_week")

    // Drop useless column and rename incorrect names
    val census_data_corrected: DataFrame = census_data.withColumnRenamed("education.num", "year_education").drop("fnlwgt")

    // Set up binary target variable
    val census_data_with_target: DataFrame = census_data_corrected.withColumn("income", when($"income"==="<=50K", 0).otherwise(1))

    // Set value to class "unknown" if not known for categorical variable
    val census_data_without_null: DataFrame = categorical_variable_string.foldLeft(census_data_with_target)((df: DataFrame, colname: String) => df.withColumn(colname, when(col(colname).isNull, "unknown").otherwise(col(colname))))

    // Set int column to float
    val census_data_recast: DataFrame = continuous_variable.foldLeft(census_data_without_null)((df, colname) => df.withColumn(colname, col(colname).cast(DoubleType)))

    // Set value to median if not known for numerical variable
    val imputer = new Imputer()
      .setInputCols(continuous_variable)
      .setOutputCols(continuous_variable.map(x => x + "_imputed"))

    // String vectorization
    val string_indexer_list = for (column_to_index <- categorical_variable_string) yield {
      new StringIndexer().setInputCol(column_to_index).setOutputCol(column_to_index+"_indexed")
    }

    // One-hot encoding
    val one_hot_encoder_model= new OneHotEncoderEstimator()
      .setInputCols(categorical_variable_string.map(x => x  +"_indexed"))
      .setOutputCols(categorical_variable_string.map(x => x  + "_encoded"))

    // Vector assembling
    val assembler = new VectorAssembler()
      .setInputCols(categorical_variable_string.map(x => x+"_encoded") ++ continuous_variable.map(x => x+"_imputed"))
      .setOutputCol("features")

    //val transformation_array =  string_indexer_list ++ Array(imputer) ++ Array(one_hot_encoder_model) ++ Array(assembler)
    val transformation_array = string_indexer_list ++ Array(one_hot_encoder_model, imputer, assembler)
    println(transformation_array)

    val transformation_pipeline: Pipeline = new Pipeline().setStages(transformation_array)
    val transformation_pipeline_model: PipelineModel = transformation_pipeline.fit(census_data_recast)

    val transformed_data: DataFrame = transformation_pipeline_model.transform(census_data_recast).select("income", "features")

    // Test to do here
    transformed_data.printSchema()
    transformed_data.show(5)

    // Start ML here

    val Array(trainingData, testData): Array[DataFrame] = transformed_data.randomSplit(Array(0.7, 0.3))

    val metricsComputer: ClassificationMetrics = new ClassificationMetrics().setColumnLabelName("income")


    val rf = new RandomForestClassifier()
      .setLabelCol("income")
      .setFeaturesCol("features")
      .setNumTrees(200)

    // Train model. This also runs the indexers.
    val rf_model = rf.fit(trainingData)

    // Make predictions.
    val predictions_rf = rf_model.transform(testData)

    val metrics_rf = metricsComputer.fit(predictions_rf)
    println(metrics_rf.classificationReport())


    // Define parameters of the gradient boosting
    val xgbParam  = Map("booster" -> "gbtree",
      "verbosity" -> 3,
      "eta" -> 0.3,
      "gamma" -> 0.5,
      "max_depth" -> 10,
      "subsample" -> 0.4,
      "colsample_bytree" -> 0.5,
      "colsample_bylevel" -> 1,
      "colsample_bynode" -> 1,
      "objective" -> "binary:logistic",
      "num_round" -> 100,
      "train_test_ratio " -> 0.9)

    // Create gradient boosting classifier object
    val xgbClassifier = new XGBoostClassifier(xgbParam)
      .setFeaturesCol("features")
      .setLabelCol("income")
      .setMissing(0)

    val XGBmodel = xgbClassifier.fit(trainingData)

    val predictionsXGB = XGBmodel.transform(testData)

    val metricsXGB = metricsComputer.fit(predictionsXGB)
    println(metricsXGB.classificationReport())

    spark.stop()
    println("Done")
  }

}

