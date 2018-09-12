# Python-3

from   pyspark               import SparkContext
from   pyspark.sql           import SQLContext
from   time                  import mktime, strptime
from   pyspark.sql.types     import IntegerType
from   pyspark.ml.feature    import VectorAssembler
from   pyspark.ml.regression import LinearRegression
from   pyspark.ml.evaluation import RegressionEvaluator
from   pyspark.ml.regression import DecisionTreeRegressor
from   pyspark.ml.regression import GBTRegressor
from   pyspark.ml.regression import RandomForestRegressor
from   pyspark.ml.feature    import StringIndexer
from   pyspark.ml.tuning     import CrossValidator, ParamGridBuilder
import pyspark.sql.functions as fn
import model_info

CSV_TRAINING_DATA    = "./historical_data.csv"
CSV_TMP_FILE         = "./tmp.csv"

# Models dictionary and training scores populated by find_model()

model_dict = {}
model_rmse = []
model_count= 0

#######################################
# UDF to convert clock time to seconds
# Inputs:  Clock time
# Outputs: Seconds as an integer
#######################################

def rdd_datetime_to_sec(input_time):
    return int(mktime(strptime(input_time, "%m/%d/%y %H:%M")))


###################################################################################
# Performs Linear Regression on randomly split training data and saves score, model
# Inputs: split_input_train_df, split_input_validation_df, model_evaluator
# Global: model_rmse, model_dict, model_count
# Outputs:
###################################################################################

def score_lr(split_input_train_df, split_input_validation_df, model_evaluator):
    global  model_rmse, model_dict, model_count

    print("###################### Linear Regression #########################")
    lr = LinearRegression(featuresCol = 'features', labelCol='total_delivery_duration')

    print("CrossValidation...")
    lr_paramGrid = ParamGridBuilder()\
				.addGrid(lr.maxIter, [5, 10])\
				.addGrid(lr.regParam, [0.1, 1.0])\
				.addGrid(lr.elasticNetParam, [0.1, 1.0])\
				.build()
    lr_cross_val = CrossValidator(estimator=lr,
				estimatorParamMaps=lr_paramGrid,
				evaluator=model_evaluator,
				numFolds=3)
    print("Done")
    print("Fitting training data...")
    lr_cv_model  = lr_cross_val.fit(split_input_train_df)
    print("Done")

    rmse      = model_evaluator.evaluate(lr_cv_model.transform(split_input_validation_df))
    model_rmse.append(rmse)
    model_dict[model_count] = {}
    model_dict[model_count]["LR"] = lr_cv_model
    print("RMSE on validation data: %f" % rmse)


###################################################################################
# Performs Decision Tree Regression on randomly split training data and saves score, model
# Inputs: split_input_train_df, split_input_validation_df, model_evaluator
# Global: model_rmse, model_dict, model_count
# Outputs:
###################################################################################


def score_dt(split_input_train_df, split_input_validation_df, model_evaluator):
    global model_rmse, model_dict, model_count

    print("###################### Decision Tree Regression #########################")
    dt_regressor = DecisionTreeRegressor(featuresCol ='features', labelCol = 'total_delivery_duration')

    print("CrossValidation...")
    dt_paramGrid = ParamGridBuilder()\
				.addGrid(dt_regressor.maxBins, [5700, 6000])\
				.addGrid(dt_regressor.maxMemoryInMB, [256, 512])\
				.build()
    dt_cross_val = CrossValidator(estimator=dt_regressor,
				estimatorParamMaps=dt_paramGrid,
				evaluator=model_evaluator,
				numFolds=3)
    print("Done")
    print("Fitting training data...")
    dt_cv_model  = dt_cross_val.fit(split_input_train_df)
    print("Done")
    print("Evaluating on validation data...")
    rmse         = model_evaluator.evaluate(dt_cv_model.transform(split_input_validation_df))
    model_rmse.append(rmse)
    model_count += 1
    model_dict[model_count] = {}
    model_dict[model_count]["DT"] = dt_cv_model
    print("RMSE on validation data: %f" % rmse)


###################################################################################
# Performs Gradient Boosted Tree Regression on randomly split training data and saves score, model
# Inputs: split_input_train_df, split_input_validation_df, model_evaluator
# Global: model_rmse, model_dict, model_count
# Outputs:
###################################################################################


def score_gbt(split_input_train_df, split_input_validation_df, model_evaluator):
    global model_rmse, model_dict, model_count

    print("###################### Gradient Boosted Tree Regression #########################")
    gbt_regressor   = GBTRegressor(featuresCol='features', labelCol='total_delivery_duration')

    print("CrossValidation...")
    gbt_paramGrid = ParamGridBuilder()\
				.addGrid(gbt_regressor.maxIter, [5, 10])\
				.addGrid(gbt_regressor.maxBins, [5700, 6000])\
				.addGrid(gbt_regressor.maxMemoryInMB, [256, 512])\
				.addGrid(gbt_regressor.subsamplingRate, [0.1, 1.0])\
				.build()
    gbt_cross_val = CrossValidator(estimator=gbt_regressor,
				estimatorParamMaps=gbt_paramGrid,
				evaluator=model_evaluator,
				numFolds=3)
    print("Done")
    print("Fitting training data...")
    gbt_cv_model  = gbt_cross_val.fit(split_input_train_df)
    print("Done")
    print("Evaluating on validation data...")
    rmse          = model_evaluator.evaluate(gbt_cv_model.transform(split_input_validation_df))
    model_rmse.append(rmse)
    model_count += 1
    model_dict[model_count] = {}
    model_dict[model_count]["GBT"] = gbt_cv_model
    print("RMSE on validation data: %f" % rmse)


###################################################################################
# Performs Random Forest Regression on randomly split training data and saves score, model
# Inputs: split_input_train_df, split_input_validation_df, model_evaluator
# Global: model_rmse, model_dict, model_count
# Outputs:
###################################################################################


def	score_rf(split_input_train_df, split_input_validation_df, model_evaluator):
    global model_rmse, model_dict, model_count

    print("###################### Random Forest Regression #########################")
    rf_regressor   = RandomForestRegressor(featuresCol = 'features', labelCol = 'total_delivery_duration')

    print("CrossValidation...")
    rf_paramGrid = ParamGridBuilder()\
				.addGrid(rf_regressor.maxBins, [5700, 6000])\
				.addGrid(rf_regressor.maxMemoryInMB, [256, 512])\
				.addGrid(rf_regressor.subsamplingRate, [0.1, 1.0])\
				.build()
    rf_cross_val = CrossValidator(estimator=rf_regressor,
				estimatorParamMaps=rf_paramGrid,
				evaluator=model_evaluator,
				numFolds=3)
    print("Done")
    print("Fitting training data...")
    rf_cv_model  = rf_cross_val.fit(split_input_train_df)
    print("Done")
    print("Evaluating on validation data...")
    rmse         = model_evaluator.evaluate(rf_cv_model.transform(split_input_validation_df))
    model_rmse.append(rmse)
    model_count += 1
    model_dict[model_count] = {}
    model_dict[model_count]["RF"] = rf_cv_model
    print("RMSE on validation data: %f" % rmse)

#######################################################
# Find suitable model for our data
# Inputs:  SQL context, training data, orders data
# Outputs: Model index from dictionary 'model_dict'
#######################################################

def find_model( sqlContext ):

	print("Reading ", CSV_TRAINING_DATA)
	training_df = sqlContext.read.csv(CSV_TRAINING_DATA,
	                                    header=True,
	                                    inferSchema=True)
	print("Done")

	print("Preprocessing...")
	# Drop rows with NA columns
	training_df_1 = training_df.dropna()

	# Drop rows with subtotal=0
	# Drop rows where min_item_price is -ve
	# Drop rows where max_item_price is 0, assuming spoons and forks are not part of the order
	# Drop rows where total_onshift_runners is -ve
	# Drop rows where total_busy_runners is -ve
	# Drop rows where total_outstanding_orders is -ve
	# Drop rows where estimated_order_place_duration is 0 - is not practical
	# Drop rows where estimated_store_to_consumer_driving_duration is 0 - is not practical
	training_df_1 = training_df_1[(training_df_1.subtotal>0) &
					(training_df_1.min_item_price>0) &
					(training_df_1.max_item_price>0) &
					(training_df_1.total_onshift_runners>=0) &
					(training_df_1.total_busy_runners>=0) &
					(training_df_1.total_outstanding_orders>=0) &
					(training_df_1.estimated_order_place_duration>0) &
					(training_df_1.estimated_store_to_consumer_driving_duration>0) &
					(training_df_1.market_id != "NA") &
					(training_df_1.actual_delivery_time != "NA") &
					(training_df_1.store_primary_category != "NA") &
					(training_df_1.order_protocol != "NA")]

	# datetime string to seconds conversion
	udf_rdd_datetime_to_sec = fn.udf(rdd_datetime_to_sec, IntegerType())  # LongType() not available for now

	training_df_1 = training_df_1.withColumn('created_at',
	                                        udf_rdd_datetime_to_sec(fn.col('created_at')))
	training_df_1 = training_df_1.withColumn('actual_delivery_time',
	                                        udf_rdd_datetime_to_sec(fn.col('actual_delivery_time')))
	training_df_1 = training_df_1.withColumn('total_delivery_duration',
	                                        training_df_1['actual_delivery_time'] - training_df_1['created_at'])

	# Map store_id string to unique number
	stringindexer = StringIndexer().setInputCol("store_id").setOutputCol("store_id_int")
	modelc = stringindexer.fit(training_df_1)
	training_df_1 = modelc.transform(training_df_1)

	# Map store_primary_category to unique number
	stringindexer = StringIndexer().setInputCol("store_primary_category").setOutputCol("store_primary_category_int")
	modelc = stringindexer.fit(training_df_1)
	training_df_1 = modelc.transform(training_df_1)

	training_df_1 = training_df_1.withColumn("market_id", training_df_1["market_id"].cast(IntegerType()))
	training_df_1 = training_df_1.withColumn("order_protocol", training_df_1["order_protocol"].cast(IntegerType()))

	training_df_1 = training_df_1.withColumn("total_onshift_runners", training_df_1["total_onshift_runners"].cast(IntegerType()))
	training_df_1 = training_df_1.withColumn("total_busy_runners", training_df_1["total_busy_runners"].cast(IntegerType()))

	training_df_1 = training_df_1.withColumn("total_outstanding_orders", training_df_1["total_outstanding_orders"].cast(IntegerType()))
	training_df_1 = training_df_1.withColumn("estimated_store_to_consumer_driving_duration", training_df_1["estimated_store_to_consumer_driving_duration"].cast(IntegerType()))
	print("Done")

	print("Features...")
	# Drop string columns as we created respective columns with numerical values
	training_df_1 = training_df_1.drop('store_id', 'store_primary_category')

	# Drop actual_delivery_time as we have computed total_delivery_duration
	training_df_1 = training_df_1.drop('actual_delivery_time')

	# Drop min_item_price and max_item_price as they are factored into subtotal
	training_df_1 = training_df_1.drop('min_item_price', 'max_item_price')

	feature_list = training_df_1.columns
	feature_list.remove('total_delivery_duration')
	print("Done")

	print("Vectorize...")
	vectorAssembler = VectorAssembler(inputCols = feature_list, outputCol = 'features')
	vectorized_training_df = vectorAssembler.transform(training_df_1)
	vectorized_training_df = vectorized_training_df.select(['features', 'total_delivery_duration'])
	print("Done")

	splits = vectorized_training_df.randomSplit([0.7, 0.3], seed=12345)
	split_input_train_df      = splits[0]
	split_input_validation_df = splits[1]

	model_count = 0
	model_evaluator = RegressionEvaluator(labelCol="total_delivery_duration", predictionCol="prediction", metricName="rmse")

	'''
	############################################################################################
	# We will try multiple models and pick the one with best prediction accuracy (lowest error)#
	############################################################################################
	'''

	# LinearRegression

	score_lr(split_input_train_df, split_input_validation_df, model_evaluator)

	# DecisionTree

	score_dt(split_input_train_df, split_input_validation_df, model_evaluator)

	# GradientBoostedTree

	score_gbt(split_input_train_df, split_input_validation_df, model_evaluator)

	# RandomForestRegressor

	score_rf(split_input_train_df, split_input_validation_df, model_evaluator)

	# Identify best score and save corresponding model for later prediction

	least_rmse   = min(model_rmse)
	chosen_model_index = model_rmse.index(least_rmse)
	chosen_model_name  = list(model_dict.get(chosen_model_index).keys())[0]
	print("Least RMSE is " + str(least_rmse) + " for model " + chosen_model_name)
	print("Saving the model..")
	txt_file = open(MODEL_NAME_FILE, "w")
	txt_file.write(chosen_model_name)
	txt_file.close()
	model_dict[chosen_model_index][chosen_model_name].bestModel.write().overwrite().save(MODEL_DIR)

if __name__ == "__main__":

	# Create spark context

	sc = SparkContext(appName = "dd_prediction")
	sc.setLogLevel("WARN")

        # Get sql context

	sqlContext = SQLContext(sc)
	spark = sqlContext.sparkSession

	find_model(sqlContext)

	sc.stop()
