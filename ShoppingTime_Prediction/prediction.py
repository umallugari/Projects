# Python-3

from   pyspark     import SparkContext
from   pyspark.sql import SQLContext
from   time        import mktime, strptime
from   pyspark.sql.types     import IntegerType
from   pyspark.ml.feature    import VectorAssembler, StringIndexer
from   pyspark.ml.regression import RandomForestRegressionModel
from   pyspark.ml.regression import LinearRegressionModel
from   pyspark.ml.regression import GBTRegressionModel
from   pyspark.ml.regression import DecisionTreeRegressionModel
from   pyspark.sql.functions import monotonically_increasing_id
import pyspark.sql.functions as fn
import sys
from   model_info import *

PREDICTION_RESULTS_DIR = "./prediction_results"
feature_list = [
 'market_id',
 'created_at',
 'order_protocol',
 'total_items',
 'subtotal',
 'num_distinct_items',
 'total_onshift_runners',
 'total_busy_runners',
 'total_outstanding_orders',
 'estimated_order_place_duration',
 'estimated_store_to_consumer_driving_duration',
 'store_id_int',
 'store_primary_category_int']

def rdd_datetimesec_to_sec(input_time):
    return int(mktime(strptime(input_time, "%Y-%m-%d %H:%M:%S")))

def load_json_and_predict(spark, sqlContext, json_file) :

	# Load data to predict
	#predict_df = spark.read.json(JSON_DATA_TO_PREDICT)
	print("Loading prediction data from ", json_file)
	predict_df = spark.read.json(json_file)
	print("Done")

	# Apply same process as historical data to convert/map

	# Drop rows with NA columns
	print("Preprocessing...")
	predict_df_1 = predict_df.dropna()

	predict_df_1 = predict_df_1[(predict_df_1.subtotal>0) &
	                            (predict_df_1.min_item_price>0) &
	                            (predict_df_1.max_item_price>0) &
	                            (predict_df_1.total_onshift_runners>=0) &
	                            (predict_df_1.total_busy_runners>=0) &
	                            (predict_df_1.total_outstanding_orders>=0) &
	                            (predict_df_1.estimated_order_place_duration>0) &
	                            (predict_df_1.estimated_store_to_consumer_driving_duration>0) &
	                            (predict_df_1.market_id != "NA") &
	                            (predict_df_1.store_primary_category != "NA") &
	                            (predict_df_1.order_protocol != "NA")
	                           ]

	udf_rdd_datetimesec_to_sec = fn.udf(rdd_datetimesec_to_sec, IntegerType())  # LongType() not available for now

	predict_df_1 = predict_df_1.withColumn('created_at',
	                                        udf_rdd_datetimesec_to_sec(fn.col('created_at')))

	# Map store_id string to unique number
	stringindexer = StringIndexer().setInputCol("store_id").setOutputCol("store_id_int")
	modelc = stringindexer.fit(predict_df_1)
	predict_df_1 = modelc.transform(predict_df_1)

	# Map store_primary_category to unique number
	stringindexer = StringIndexer().setInputCol("store_primary_category").setOutputCol("store_primary_category_int")
	modelc = stringindexer.fit(predict_df_1)
	predict_df_1 = modelc.transform(predict_df_1)

	predict_df_1 = predict_df_1.withColumn("market_id", predict_df_1["market_id"].cast(IntegerType()))
	predict_df_1 = predict_df_1.withColumn("order_protocol", predict_df_1["order_protocol"].cast(IntegerType()))
	predict_df_1 = predict_df_1.withColumn("total_onshift_runners", predict_df_1["total_onshift_runners"].cast(IntegerType()))
	predict_df_1 = predict_df_1.withColumn("total_busy_runners", predict_df_1["total_busy_runners"].cast(IntegerType()))
	predict_df_1 = predict_df_1.withColumn("total_outstanding_orders", predict_df_1["total_outstanding_orders"].cast(IntegerType()))
	predict_df_1 = predict_df_1.withColumn("estimated_store_to_consumer_driving_duration", predict_df_1["estimated_store_to_consumer_driving_duration"].cast(IntegerType()))
	predict_df_1 = predict_df_1.withColumn("subtotal", predict_df_1["subtotal"].cast(IntegerType()))
	predict_df_1 = predict_df_1.withColumn("num_distinct_items", predict_df_1["num_distinct_items"].cast(IntegerType()))
	predict_df_1 = predict_df_1.withColumn("estimated_order_place_duration", predict_df_1["estimated_order_place_duration"].cast(IntegerType()))
	predict_df_1 = predict_df_1.withColumn("total_items", predict_df_1["total_items"].cast(IntegerType()))
	print("Done")

	# Use same features as in historical data
	# Other columns in test data ('store_id', 'store_primary_category', 'min_item_price', 'max_item_price')
	# will be dropped by VectorAssembler transformation

	print("Vectorize...")
	pvectorAssembler = VectorAssembler(inputCols = feature_list, outputCol = 'features')
	vectorized_predict_df = pvectorAssembler.transform(predict_df_1)
	vectorized_predict_df = vectorized_predict_df.select(['features'])
	print("Done...")

	txt_file = open(MODEL_NAME_FILE, "r")
	model_name = txt_file.read()
	print("Read model: ", model_name)
	txt_file.close()

	print("Loading model " + model_name + " from " + MODEL_DIR)

	if(model_name == DT_MODEL):
		predict_model = DecisionTreeRegressionModel.load(MODEL_DIR)

	if(model_name == GBT_MODEL):
		predict_model = GBTRegressionModel.load(MODEL_DIR)

	if(model_name == LR_MODEL):
		predict_model = LinearRegressionModel.load(MODEL_DIR)

	if(model_name == RF_MODEL):
		predict_model = RandomForestRegressionModel.load(MODEL_DIR)

	print("Done")

	print("Predicting...")
	model_predictions = predict_model.transform(vectorized_predict_df)
	print("Done")

	df1 = predict_df_1.select('delivery_id').withColumn("id", monotonically_increasing_id())
	df2 = model_predictions.select('prediction').withColumnRenamed('prediction', 'predicted_delivery_seconds').withColumn("id", monotonically_increasing_id())

	# Perform a join on the ids.
	prediction_results_df = df1.join(df2, "id", "left").drop("id")
	prediction_results_df = prediction_results_df.withColumn("predicted_delivery_seconds", prediction_results_df["predicted_delivery_seconds"].cast(IntegerType()))

	return prediction_results_df

if __name__ == "__main__":

	if( (len(sys.argv) < 2) or (len(sys.argv) > 2) ):
        	print("Supply JSON file to predict on.")
	else:
		# Create spark context

		sc = SparkContext(appName = "dd_prediction")
		sc.setLogLevel("WARN")

        	# Get sql context

		sqlContext = SQLContext(sc)
		spark = sqlContext.sparkSession

		prediction_results_df = load_json_and_predict(spark, sqlContext, sys.argv[1])

		print("Saving predictions to ", PREDICTION_RESULTS_DIR)
		prediction_results_df.repartition(1).write.format('csv').options(header=True, delimiter='\t').save(PREDICTION_RESULTS_DIR)
		print("Done")

		sc.stop()
