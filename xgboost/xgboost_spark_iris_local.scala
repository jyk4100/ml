REM // scala codes xgboost4j-spark testing
REM // jyk4100 
REM // last modified: 2020-06-26
REM // spark-shell:spark-2.4.5-bin-hadoop2.6 
REM // other jars: akka-actor_2.11-2.3.4-spark, config-1.3.4, xgboost4j-spark-0.90-criteo-20190702_2.11, xgboost4j-0.90-criteo-20190702_2.11-win64
REM // https://github.com/dmlc/xgboost/issues/5370 num_workers = 1

REM // (windows shell comment short for remark) open spark shell
cd C:\spark-2.4.5-bin-hadoop2.6\bin
spark-shell --driver-memory 4g --executor-memory 8g --num-executors 1
REM --master local[2]

// import dependencies
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer, StringIndexer}
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType, IntegerType}
import org.apache.spark.sql.functions.{udf, round}
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassifier, TrackerConf} // "tracker_conf"->TrackerConf(60 * 60 * 1000, "scala")
// udf to get second element from the array
val getsecond = udf((v: Vector) => v.toArray(1))

// define schema/data type
val schema = ( new StructType().add("sepal_length",DoubleType,true).add("sepal_width",DoubleType,true)
							   .add("petal_length",DoubleType,true).add("petal_width",DoubleType,true)
							   .add("species",StringType,true) )
// load csv file
val iris = spark.read.schema(schema).option("header","true").format("csv").load("C:/Users/Kim Jungyoon/Documents/2.study/ml/xgboost/iris.csv")
// label encoding
val testDF = iris.withColumn("class", when(col("species") === "setosa",1).otherwise(0)) 
// feature vector
val vectorAssembler = new VectorAssembler().setInputCols(Array("sepal_length", "sepal_width", "petal_length", "petal_width")).setOutputCol("features")
val train = vectorAssembler.transform(testDF).selectExpr("features", "class")
// train.head(10)

// parameters toy example
val paramMap = ( List("objective"->"binary:logistic", "eta" -> 0.3, "scale_pos_weight" -> 1,
	 				  "lambda"->0.0, "alpha"->0, // no regularization
					  "max_depth"->1, "min_child_weight"->1.0, "num_round"->1, "tree_method"->"hist", // tree_method = 'hist'
					  "seed" -> 123, "silent" -> 1, "missing"-> 0.0, 
					  "tracker_conf"->TrackerConf(60 * 60 * 1000, "scala"), "num_workers"->1 // to run in windows local
					  ).toMap )
// initialize model fit and transform (such a CS syntax for model...)
val initmodel = new XGBoostClassifier(paramMap).setFeaturesCol("features").setLabelCol("class")
val model = initmodel.fit(train)
// model.transform(train).head(20)
val preds = model.transform(train).withColumn("probability", getsecond($"probability")).drop("rawPrediction")
preds.show(20)

// ++ suppose in real example we want to append scores to set of ids...
val iris_ids = iris.withColumn("id1", monotonically_increasing_id())
val preds_id = preds.drop("features", "class").withColumn("id2", monotonically_increasing_id()).select("id2", "probability")
val iris_preds = iris_ids.join(preds_id, iris_ids("id1") === preds_id("id2"), "left").sort($"id1").drop("id1", "id2")
iris_preds.show(10)
// iris_preds.coalesce(1).write.format("csv").save("iris_predictions.csv") 
// TBD // saved at C:\spark\bin under "XXX-123" folder (make sense if not collect/collapse/coalesced) how to move and delete directory?
