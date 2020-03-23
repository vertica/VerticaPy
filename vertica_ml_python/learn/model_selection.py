# (c) Copyright [2018] Micro Focus or one of its affiliates. 
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# AUTHOR: BADR OUALI
#
############################################################################################################ 
#  __ __   ___ ____  ______ ____   __  ____      ___ ___ _          ____  __ __ ______ __ __  ___  ____    #
# |  |  | /  _|    \|      |    | /  ]/    |    |   |   | |        |    \|  |  |      |  |  |/   \|    \   #
# |  |  |/  [_|  D  |      ||  | /  /|  o  |    | _   _ | |        |  o  |  |  |      |  |  |     |  _  |  #
# |  |  |    _|    /|_|  |_||  |/  / |     |    |  \_/  | |___     |   _/|  ~  |_|  |_|  _  |  O  |  |  |  #
# |  :  |   [_|    \  |  |  |  /   \_|  _  |    |   |   |     |    |  |  |___, | |  | |  |  |     |  |  |  #
#  \   /|     |  .  \ |  |  |  \     |  |  |    |   |   |     |    |  |  |     | |  | |  |  |     |  |  |  #
#   \_/ |_____|__|\_| |__| |____\____|__|__|    |___|___|_____|    |__|  |____/  |__| |__|__|\___/|__|__|  #
#                                                                                                          #
############################################################################################################
# Vertica-ML-Python allows user to create Virtual Dataframe. vDataframes simplify   #
# data exploration,   data cleaning   and   machine   learning   in    Vertica.     #
# It is an object which keeps in it all the actions that the user wants to achieve  # 
# and execute them when they are needed.    										#
#																					#
# The purpose is to bring the logic to the data and not the opposite                #
#####################################################################################
#
# Libraries
from vertica_ml_python import tablesample
import numpy as np
from vertica_ml_python.utilities import str_column

#
def best_k(X: list,
		   input_relation: str,
		   cursor,
		   n_cluster = (1, 100),
		   init = "kmeanspp",
		   max_iter: int = 50,
		   tol: float = 1e-4,
		   elbow_score_stop = 0.8):
	from vertica_ml_python.learn.cluster import KMeans
	L = range(n_cluster[0], n_cluster[1]) if not(type(n_cluster) == list) else n_cluster
	for i in L:
		cursor.execute("DROP MODEL IF EXISTS _vpython_kmeans_tmp_model")
		model = KMeans("_vpython_kmeans_tmp_model", cursor, i, init, max_iter, tol)
		model.fit(input_relation, X)
		score = model.metrics.values["value"][3]
		if (score > elbow_score_stop):
			return i
		score_prev = score
	print("/!\\ The K was not found. The last K (= {}) is returned with an elbow score of {}".format(i, score))
	return i
#
def cross_validate(estimator, 
				   input_relation: str, 
				   X: list, 
				   y: str, 
				   cv: int = 3, 
				   pos_label = None, 
				   cutoff: float = 0.5):
	if (estimator.type == "regressor"):
		result = {"index": ["explained_variance", "max_error", "median_absolute_error", "mean_absolute_error", "mean_squared_error", "r2"]} 
	elif (estimator.type == "classifier"):
		result = {"index": ["auc", "prc_auc", "accuracy", "log_loss", "precision", "recall", "f1-score", "mcc", "informedness", "markedness", "csi"]}
	else:
		raise ValueError("Cross Validation is only possible for Regressors and Classifiers")
	test_name, train_name = "{}_{}".format(input_relation, int(1 / cv * 100)), "{}_{}".format(input_relation, int(100 - 1 / cv * 100))
	estimator.cursor.execute("DROP TABLE IF EXISTS vpython_train_test_split_cv_{}".format(input_relation))
	query = "CREATE TABLE vpython_train_test_split_cv_{} AS SELECT *, RANDOMINT({}) AS test FROM {}".format(input_relation, cv, input_relation)
	estimator.cursor.execute(query)
	for i in range(cv):
		try:
			estimator.cursor.execute("DROP MODEL IF EXISTS {}".format(estimator.name))
		except:
			pass
		estimator.cursor.execute("DROP VIEW IF EXISTS vpython_train_test_split_cv_{}".format(test_name))
		estimator.cursor.execute("DROP VIEW IF EXISTS vpython_train_test_split_cv_{}".format(train_name))
		query = "CREATE VIEW vpython_train_test_split_cv_{} AS SELECT * FROM {} WHERE (test = {})".format(test_name, "vpython_train_test_split_cv_{}".format(input_relation), i)
		estimator.cursor.execute(query)
		query = "CREATE VIEW vpython_train_test_split_cv_{} AS SELECT * FROM {} WHERE (test != {})".format(train_name, "vpython_train_test_split_cv_{}".format(input_relation), i)
		estimator.cursor.execute(query)
		estimator.fit("vpython_train_test_split_cv_{}".format(train_name), X, y, "vpython_train_test_split_cv_{}".format(test_name))
		if (estimator.type == "regressor"):
			result["{}-fold".format(i + 1)] = estimator.regression_report().values["value"]
		else:
			if (len(estimator.classes) > 2) and (pos_label not in estimator.classes):
				raise ValueError("'pos_label' must be in the estimator classes, it must be the main class to study for the Cross Validation")
			try:
				result["{}-fold".format(i + 1)] = estimator.classification_report(labels = [pos_label], cutoff = cutoff).values["value"]
			except:
				result["{}-fold".format(i + 1)] = estimator.classification_report(cutoff = cutoff).values["value"]
		try:
			estimator.cursor.execute("DROP MODEL IF EXISTS {}".format(estimator.name))
		except:
			pass
	n = 6 if (estimator.type == "regressor") else 11
	total = [[] for item in range(n)]
	for i in range(cv):
		for k in range(n):
			total[k] += [result["{}-fold".format(i + 1)][k]]
	result["avg"] = [np.mean(item) for item in total]
	result["std"] = [np.std(item) for item in total]
	estimator.cursor.execute("DROP TABLE IF EXISTS vpython_train_test_split_cv_{}".format(input_relation))
	estimator.cursor.execute("DROP VIEW IF EXISTS vpython_train_test_split_cv_{}".format(test_name))
	estimator.cursor.execute("DROP VIEW IF EXISTS vpython_train_test_split_cv_{}".format(train_name))
	return (tablesample(values = result, table_info = False).transpose())
#
def fast_cv(algorithm: str, 
			input_relation: str, 
			cursor,
			X: list, 
			y: str, 
			cv: int = 3,
			metrics: list = [],
			params: dict = {},
			cutoff: float = -1):
	if (algorithm.lower() in ("logistic_reg", "logistic_regression", "logisticregression")):
		algorithm = "logistic_reg"
	elif (algorithm.lower() in ("linear_reg", "linear_regression", "linearregression")):
		algorithm = "linear_reg"
	elif (algorithm.lower() in ("svm_classifier", "svmclassifier", "linearsvc")):
		algorithm = "svm_classifier"
	elif (algorithm.lower() in ("svm_regressor", "svmregressor", "linearsvr")):
		algorithm = "svm_regressor"
	elif (algorithm.lower() in ("naive_bayes", "naivebayes", "multinomialnb")):
		algorithm = "naive_bayes"
	if not(metrics):
		if algorithm in ("naive_bayes", "svm_classifier", "logistic_reg"):
			metrics = ["accuracy", "auc_roc", "auc_prc", "fscore"]
		elif algorithm in ("svm_regressor", "linear_reg"):
			metrics = ["MSE", "MAE", "rsquared", "explained_variance"]
	sql = "SELECT CROSS_VALIDATE('{}', '{}', '{}', '{}' USING PARAMETERS cv_fold_count = {}, cv_metrics = '{}'".format(algorithm, input_relation, y, ", ".join([str_column(item) for item in X]), cv, ", ".join(metrics))
	if (params):
		sql += ", cv_hyperparams = '{}'".format(params)
	if (cutoff <= 1 and cutoff >= 0):
		sql += ", cv_prediction_cutoff = '{}'".format(cutoff)
	sql += ')'
	cursor.execute(sql)
	return (cursor.fetchone()[0])
#
def train_test_split(input_relation: str, cursor, test_size: float = 0.33):
	test_name, train_name = "{}_{}".format(input_relation, int(test_size * 100)), "{}_{}".format(input_relation, int(100 - test_size * 100))
	cursor.execute("DROP TABLE IF EXISTS vpython_train_test_split_{}".format(input_relation))
	cursor.execute("DROP VIEW IF EXISTS vpython_train_test_split_{}".format(test_name))
	cursor.execute("DROP VIEW IF EXISTS vpython_train_test_split_{}".format(train_name))
	query = "CREATE TABLE vpython_train_test_split_{} AS SELECT *, (CASE WHEN RANDOM() < {} THEN True ELSE False END) AS test FROM {}".format(input_relation, test_size, input_relation)
	cursor.execute(query)
	query = "CREATE VIEW vpython_train_test_split_{} AS SELECT * FROM {} WHERE test".format(test_name, "vpython_train_test_split_{}".format(input_relation))
	cursor.execute(query)
	query = "CREATE VIEW vpython_train_test_split_{} AS SELECT * FROM {} WHERE NOT(test)".format(train_name, "vpython_train_test_split_{}".format(input_relation))
	cursor.execute(query)
	return ("vpython_train_test_split_{}".format(train_name), "vpython_train_test_split_{}".format(test_name))
