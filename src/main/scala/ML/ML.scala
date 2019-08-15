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
//import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassifier, XGBoostClassificationModel}


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

    // Loading sata
    val census_data : DataFrame = spark.read.format("csv")
      .option("sep", ",")
      .option("inferSchema", "true")
      .option("header", "true")
      .load("/Users/az02234/Documents/Personnal_Git/Scala-Spark-ML-Example/src/main/resources/census_data.csv")

    // Some description
    census_data.show(5)
    println(census_data.count())
    census_data.printSchema()

    println(census_data.filter($"native_country".isNull).count())
    //print(census_data.filter($"native_country"==="nan").count())
    census_data.select("native_country").dropDuplicates().show()

    // Set list of column
    val target = "income"
    val categorical_variable_string = Array("workclass", "education", "occupation", "relationship", "sex", "marital_status", "native_country")
    //val categorical_variable_string = Array("workclass", "education" ,"native_country")
    val continuous_variable = Array("year_education", "capital_gain", "capital_loss", "hours_per_week")

    // Set up binary target variable
    val census_data_with_target = census_data.withColumn("income", when($"income"==="<=50K", 0).otherwise(1))

    // Set value to class "unknown" if not known for categorical variable
    // Create a function taking a list of column as argument
    val census_data_without_null = census_data_with_target.withColumn("education", when($"education".isNull, "unknown").otherwise($"education"))
      .withColumn("workclass", when($"workclass".isNull, "unknown").otherwise($"workclass"))
      .withColumn("marital_status", when($"marital_status".isNull, "unknown").otherwise($"marital_status"))
      .withColumn("occupation", when($"occupation".isNull, "unknown").otherwise($"occupation"))
      .withColumn("relationship", when($"relationship".isNull, "unknown").otherwise($"relationship"))
      .withColumn("sex", when($"sex".isNull, "unknown").otherwise($"sex"))
      .withColumn("native_country", when($"native_country".isNull, "unknown").otherwise($"native_country"))

    // Test to do here
    //println(census_data_without_null.filter($"native_country".isNull).count())
    //print(census_data.filter($"native_country"==="nan").count())
    // census_data_without_null.printSchema()

    // Set int column to float + rename column with name like xx.xx + drop useless column
    val census_data_recast = census_data_without_null
      .withColumnRenamed("education.num", "year_education")
      .withColumn("year_education", $"year_education".cast(DoubleType))
      .withColumn("capital_gain", $"capital_gain".cast(DoubleType))
      .withColumn("capital_loss", $"capital_loss".cast(DoubleType))
      .withColumn("hours_per_week", $"hours_per_week".cast(DoubleType))
      .drop("fnlwgt")

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
      //.setInputCols(categorical_variable_string.map(x => x+"_encoded"))
      .setOutputCol("features")

    //val transformation_array =  string_indexer_list ++ Array(imputer) ++ Array(one_hot_encoder_model) ++ Array(assembler)
    val transformation_array = string_indexer_list ++ Array(one_hot_encoder_model, imputer, assembler)
    println(transformation_array)

    val transformation_pipeline = new Pipeline().setStages(transformation_array)
    val transformation_pipeline_model = transformation_pipeline.fit(census_data_recast)

    val transformed_data = transformation_pipeline_model.transform(census_data_recast).select("income", "features")

    // Test to do here
    transformed_data.printSchema()
    transformed_data.show(5)

    // Start ML here

    val Array(trainingData, testData) = transformed_data.randomSplit(Array(0.7, 0.3))


    val rf = new RandomForestClassifier()
      .setLabelCol("income")
      .setFeaturesCol("features")
      .setNumTrees(200)

    // Train model. This also runs the indexers.
    val model = rf.fit(trainingData)

    val lasso = new LinearRegression()
      .setLabelCol("income")
      .setFeaturesCol("features")
      .setElasticNetParam(0)
      .setRegParam(0)

    val model_lasso = lasso.fit(trainingData)

    // Make predictions.
    val predictions_rf = model.transform(testData)

    val predictions = model_lasso.transform(testData)

    print(model_lasso.coefficients)

    predictions.show(5)
    val predictionsbis = predictions
      .withColumn("prediction", when($"prediction">0.5, 1).otherwise(0))
      .withColumn("prediction", $"prediction"cast("int"))
    predictionsbis.show(5)

    //val accuracy = predictionsbis.filter(col("income") === col("prediction")).count().toFloat / predictions.count().toFloat

    val metricsComputer = new ClassificationMetrics(predictionsbis).setColumnLabelName("income")

    val accuracy = metricsComputer.Accuracy()
    println("")
    println("Accuracy is:")
    println(accuracy)

    val benchmark = predictions.filter(col("income") === 1).count().toFloat / predictions.count().toFloat
    println("Benchmark is:")
    println(benchmark)

    val confusionMatrix = metricsComputer.ConfusionMatrix()
    println("")
    println("Confusion Matrix is:")
    println(confusionMatrix)

/*
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
      "num_round" -> 40)

    // Create gradient boosting classifier object
    val xgbClassifier = new XGBoostClassifier(xgbParam)
      .setFeaturesCol("features")
      .setLabelCol("income")
    println("INITIALIZE")

    val XGBmodel = xgbClassifier.fit(trainingData)
    println("ESTIMATION")

    val predictionsXGB = XGBmodel.transform(testData)
    println("PREDICTION")

    XGBmodel.write.overwrite().save("/Users/az02234/Documents/Personnal_Git/Scala-Spark-ML-Example/src/main/resources/xgbModel")
    //XGBmodel.write.overwrite().save("/Users/az02234/Documents/Personnal_Git/Scala-Spark-ML-Example/src/main/resources/xgbModel")
    println("SAVED")

    val loaded_model = XGBoostClassificationModel.load("/Users/az02234/Documents/Personnal_Git/Scala-Spark-ML-Example/src/main/resources/xgbModel")
    print("LOADED")

    val predictionsXGB2 = loaded_model.transform(testData)

*/


    spark.stop()
    println("Done")
  }

}

