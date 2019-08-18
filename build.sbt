name := "Scala-Spark-ML-Example"

version := "0.1"

scalaVersion := "2.11.0"

resolvers += "Local Maven Repository" at "file://"+Path.userHome.absolutePath+"/.m2/repository"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.4.3"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.4.3"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.4.3"
libraryDependencies += "ml.dmlc" % "xgboost4j-spark" % "0.90"

parallelExecution in Test := false
trapExit := false