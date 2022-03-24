import os
import time
from pyspark import SparkContext
from pyspark.conf import SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.types import *

import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import ParameterGrid
import itertools

from fbprophet import Prophet

## function to process pandas dataframe
def process_data(pandas_df):
	demand_df = pandas_df.copy()

	#list of all possible series to train and forecast
	series_names = list(pandas_df['storeitem'].unique()) 
	total_time_series = len(series_names)

	# renaming columns as per FbProphet
	demand_df.rename(columns = {'date':'ds', 'sales':'y'}, inplace=True) 

	# Ensuring sales data has correct datatype
	demand_df['y'] = pd.to_numeric(demand_df['y'])
	demand_df['ds'] = pd.to_datetime(demand_df['ds'], format='%Y-%m-%d')

	return demand_df, sorted(series_names), total_time_series
  
## function to split single time series into train and test
def train_test_split(demand_df, i, train_end_date):
	final_df = demand_df[demand_df['storeitem']==i][['ds','y']]

	train = final_df[final_df['ds'].dt.strftime('%Y-%m-%d') <= train_end_date]
	test = final_df[final_df['ds'].dt.strftime('%Y-%m-%d') > train_end_date]

	return final_df, train, test

## function to generate parameter grid for hyperparameter tuning
def get_param_grid():

	# defining parameters grid for hyperparameter
	params_grid = {'yearly_seasonality': [True, False],
		       'weekly_seasonality': [True, False],
		       'monthly_seasonality': [True, False],
		       'seasonality_mode': ['multiplicative', 'additive'],
		       'changepoint_prior_scale': [0.05, 0.1],
		       'seasonality_prior_scale': [0.1, 1.0]
		      }

	grid = ParameterGrid(params_grid)

	grid_lst = []
	for ind, p in enumerate(grid):
		grid_lst.append(p)

	total_params = len(grid_lst)

	return grid, grid_lst, total_params

## Weighted Average Percentage Error 
def WAPE(y_true, y_pred):
	AE = np.abs(y_true - y_pred).sum()
	wape = AE / np.abs(y_true).sum()
	return wape*100

## Bias 
def BIAS(y_true, y_pred):
	total_diff = (y_pred - y_true).sum()
	total_actual = y_true.sum()
	bias = total_diff/total_actual
	return bias*100

## Root Mean Square Error 
def RMSE(y_true, y_pred):
	MSE = np.square(np.subtract(y_true,y_pred)).mean() 
	rmse = np.sqrt(MSE)
	return rmse

## function to measure the model's performance 
def get_metrics(y_true, y_pred):
	wape = WAPE(y_true, y_pred)
	bias = BIAS(y_true, y_pred)
	rmse = RMSE(y_true, y_pred)
	return wape, bias, rmse

## function to train a series on a single set of parameters
def train_single_model(series_name, param_dict, train_end_date, demand_df, start_time):

	"""
	Train the model for a single time-series with single paraneter set

	:param series_name: name of the series i.e. unique name for a store-item
	:param param_dict: model parameters dictionary for model training
	:param train_end_date: End date of the training period
	:param demand_df: array with forcasted values
	:param start_time: Time of stating spark application
	:returns: series_name, param_dict, test metrics (wape, bias, rmse) and start_time , end_time and time_taken for the mddel execution
	"""

	## capturing the start time of the execution
	start_time_local = time.time()

	## generating the train-test split for given time-series: series_name
	final_df, train, test = train_test_split(demand_df, series_name, train_end_date)
    
	## instantiating the prophet model with the given parameters: p
	p_model = Prophet(growth = 'linear',
		      	yearly_seasonality = param_dict['yearly_seasonality'],
		      	weekly_seasonality = param_dict['weekly_seasonality'],
		      	seasonality_mode = param_dict['seasonality_mode'],
		      	changepoint_prior_scale= param_dict['changepoint_prior_scale'],
		      	seasonality_prior_scale = param_dict['seasonality_prior_scale'],
		      	interval_width=0.8,
		      	changepoint_range=0.8
		      	)
    
	## adding monthly seasonality if required
	if param_dict['monthly_seasonality']:
		p_model.add_seasonality(name='monthly', period=30.5, fourier_order = 5, 
					prior_scale = param_dict['seasonality_prior_scale'], mode = param_dict['seasonality_mode'])

	## fitting the model on the train data and predicting on the test data
	p_model.fit(train)
	test_pred_df = p_model.predict(test)[['ds', 'yhat']]

	## merging actual test data with predicted test data
	test_final = test.merge(test_pred_df, on=['ds'], how='left')
	test_final = test_final[['ds', 'y', 'yhat']]

	## model evaluation
	y_true = test_final['y']
	y_pred = test_final['yhat']

	## getting the metric values for the prediction made on test data
	wape, bias, rmse = get_metrics(y_true, y_pred)

	## capturing the end time of the execution
	end_time_local = time.time()

	task_time = float(end_time_local - start_time_local)/60

	return (series_name, str(param_dict), float(wape), float(bias), float(rmse), float(task_time), float(start_time), float(start_time_local), float(end_time_local))

## function to train all models parallely using pyspark functionalities
def train_all_models_with_pyspark(series_params_lst, train_end_date, demand_df, num_partitions):

	## capturing the start time of the pyspark job
	start_time_pyspark =  time.time()

	## distributing these combinations across the worker nodes in the cluster
	series_param_rdd = sc.parallelize(series_params_lst, num_partitions)

	## transforming above rdd to train the all the models
	output_rdd = series_param_rdd.map(lambda x: train_single_model(x[0], x[1], train_end_date, demand_df, start_time_pyspark))

	## calling an action on the above transformation to start the process of training
	output_lst_pyspark = output_rdd.collect()

	## capturing the end time of the pyspark job
	end_time_pyspark = time.time()

	total_time_pyspark = end_time_pyspark - start_time_pyspark

	return output_lst_pyspark, total_time_pyspark

## function to train all models sequentially using python functionalities
def train_all_models_with_python(series_params_lst, train_end_date, demand_df):

	## capturing the start time of the python job
	start_time_python =  time.time()

	output_lst_python = []
	for series, param in series_params_lst:
		output = train_single_model(series, param, train_end_date, demand_df, start_time_python)
		output_lst_python.append(output)

	## capturing the end time of the python job
	end_time_python = time.time()
	
	total_time_python = end_time_python - start_time_python

	return output_lst_python, total_time_python


if __name__ == '__main__':

	spark_conf = SparkConf().setAppName("Forecasting at Scale with Prophet on EMR")
	sc = SparkContext.getOrCreate(conf=spark_conf)
	spark = SQLContext(sc)
	sc._jsc.hadoopConfiguration().set("mapreduce.fileoutputcommitter.algorithm.version", "2")

	print("\n ======================================  S E T T I N G   T H E   I N P U T   P A R A M E T E R S ======================================= \n")

	## path to the s3 bucket where the training data is located
	demand_df_path = "s3://Machine-Learning/forecasting_test/store_item_sales.csv"

	## path to the s3 bucket where the output needs to be saved
	output_df_path = "s3://Machine-Learning/forecasting_test/model_all_param_metrics"

	## deciding number of partitions the data is to be divided into
	num_partitions = int(sc.getConf().get('spark.default.parallelism')) # fetching from the cluster configurations
	# num_partitions = 80 # set it manually

	## deciding the train_end_date
	train_end_date = '2017-06-31'
	
	print("\n ====================================   R E A D I N G   &   P R O C E S S I N G    T H E   D A T A ======================================= \n")

	## reading the data from s3 in spark dataframe
	spark_df = spark.read.csv(demand_df_path, header = 'true')

	## converitng pyspark dataframe to pandas dataframe
	pandas_df = spark_df.toPandas()

	## processing the data to get the desired format and the list of series(storeitem) to train
	demand_df, series_names, total_time_series = process_data(pandas_df)

	## generating the parameter grid
	grid, grid_lst, total_params = get_param_grid()  # grid size = 64

	## creating all combination of models (series, parameters)
	series_to_train = series_names[:100]  # training first 100 time-series
	series_params_lst = list(itertools.product(series_to_train, grid_lst))  # size = 100*64
	total_models = len(series_params_lst)

	print("\n ================================================  G E N E R A L    P A R A M E T E R S  ================================================== \n")

	print("Total time-series ::", total_time_series)
	print("Total parameters  ::", total_params)
	print("Total models to train ::", total_models)
	print("Total partitions ::", num_partitions)

	print("\n =================================  T R A I N I N G     W I T H    P Y S P A R K  ( P A R A L L E L Y ) ==================================== \n")

	output_lst_pyspark, total_time_pyspark = train_all_models_with_pyspark(series_params_lst, train_end_date, demand_df, num_partitions)

	print("\n =================================  T R A I N I N G     W I T H    P Y T H O N  ( S E Q U E N T I A L L Y ) ================================ \n")

	output_lst_python, total_time_python = train_all_models_with_python(series_params_lst, train_end_date, demand_df)

	print("\n =============================================    T I M E    C O M P A R I S O N   ========================================================== \n")

	print(" Pyspark (parallely) :: %s mintutes " % (total_time_pyspark / 60))
	print(" Python (serially)   :: %s mintutes " % (total_time_python / 60))

	print("\n =============================================    S A V I N G    T H E    O U T P U T     =================================================== \n")

    
	schema = StructType([
			StructField('series', StringType(), True),
			StructField('Parameters', StringType(), True),
			StructField('WAPE', FloatType(), True),
			StructField('BIAS', FloatType(), True),
			StructField('RMSE', FloatType(), True),
			StructField('task_time', FloatType(), True),
			StructField('start_time', FloatType(), True),
			StructField('start_time_local', FloatType(), True),
			StructField('end_time_local', FloatType(), True)
			])

	output_lst = output_lst_pyspark # or output_lst = output_lst_python
	output_df = spark.createDataFrame(output_lst, schema)
	output_df.repartition(1).write.mode('overwrite').csv(output_df_path, header=True)

	print("\n =====================================    O U T P U T   S A V E D    T O   %s  =================================================== \n" % output_df_path)


# bootstrap_emr_v8.sh
# spark-submit --packages org.apache.spark:spark-avro_2.11:2.4.0 --master yarn --deploy-mode client test.py


