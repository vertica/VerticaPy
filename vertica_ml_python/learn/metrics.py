# (c) Copyright [2018-2020] Micro Focus or one of its affiliates. 
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
from vertica_ml_python import tablesample, to_tablesample
from vertica_ml_python.learn.plot import roc_curve, prc_curve
import math
#
## REGRESSION
#
def explained_variance(y_true: str, 
			 		   y_score: str, 
			  		   input_relation: str,
			  		   cursor):
	query  = "SELECT 1 - VARIANCE({} - {}) / VARIANCE({}) FROM {}".format(y_true, y_score, y_true, input_relation)
	cursor.execute(query)
	return (cursor.fetchone()[0])
#
def max_error(y_true: str, 
			  y_score: str, 
			  input_relation: str,
			  cursor):
	query  = "SELECT MAX(ABS({} - {})) FROM {}".format(y_true, y_score, input_relation)
	cursor.execute(query)
	return (cursor.fetchone()[0])
#
def median_absolute_error(y_true: str, 
			  			  y_score: str, 
			  			  input_relation: str,
			  			  cursor):
	query  = "SELECT APPROXIMATE_MEDIAN(ABS({} - {})) FROM {}".format(y_true, y_score, input_relation)
	cursor.execute(query)
	return (cursor.fetchone()[0])
#
def mean_absolute_error(y_true: str, 
			 			y_score: str, 
			 			input_relation: str,
			 			cursor):
	query  = "SELECT AVG(ABS({} - {})) FROM {}".format(y_true, y_score, input_relation)
	cursor.execute(query)
	return (cursor.fetchone()[0])
#
def mean_squared_error(y_true: str, 
			 		   y_score: str, 
			 		   input_relation: str,
			 		   cursor):
	query  = "SELECT MSE({}, {}) OVER () FROM {}".format(y_true, y_score, input_relation)
	cursor.execute(query)
	return (cursor.fetchone()[0])
#
def mean_squared_log_error(y_true: str, 
			 		   	   y_score: str, 
			 		   	   input_relation: str,
			 		   	   cursor):
	query  = "SELECT AVG(POW(LOG({} + 1) - LOG({} + 1), 2)) FROM {}".format(y_true, y_score, input_relation)
	cursor.execute(query)
	return (cursor.fetchone()[0])
#
def regression_report(y_true: str, 
			 		  y_score: str, 
			 		  input_relation: str,
			 		  cursor):
	query  = "SELECT 1 - VARIANCE({} - {}) / VARIANCE({}), MAX(ABS({} - {})), ".format(y_true, y_score, y_true, y_true, y_score)
	query += "APPROXIMATE_MEDIAN(ABS({} - {})), AVG(ABS({} - {})), ".format(y_true, y_score, y_true, y_score)
	query += "AVG(POW({} - {}, 2)) FROM {}".format(y_true, y_score, input_relation)
	r2 = r2_score(y_true, y_score, input_relation, cursor)
	values = {"index": ["explained_variance", "max_error", "median_absolute_error", "mean_absolute_error", "mean_squared_error", "r2"]}
	cursor.execute(query)
	values["value"] = [item for item in cursor.fetchone()] + [r2]
	return (tablesample(values, table_info = False))
#
def r2_score(y_true: str, 
			 y_score: str, 
			 input_relation: str,
			 cursor):
	query  = "SELECT RSQUARED({}, {}) OVER() FROM {}".format(y_true, y_score, input_relation)
	cursor.execute(query)
	return (cursor.fetchone()[0])
#
## CLASSIFICATION
#
def accuracy_score(y_true: str, 
				   y_score: str, 
				   input_relation: str,
				   cursor):
	try:
		query = "SELECT AVG(CASE WHEN {} = {} THEN 1 ELSE 0 END) AS accuracy FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL"
		query = query.format(y_true, y_score, input_relation, y_true, y_score)
		cursor.execute(query)
		return (cursor.fetchone()[0])
	except:
		query = "SELECT AVG(CASE WHEN {}::varchar = {}::varchar THEN 1 ELSE 0 END) AS accuracy FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL"
		query = query.format(y_true, y_score, input_relation, y_true, y_score)
		cursor.execute(query)
		return (cursor.fetchone()[0])
#
def auc(y_true: str, 
		y_score: str, 
		input_relation: str,
		cursor,
		pos_label = 1):
	return (roc_curve(y_true, y_score, input_relation, cursor, pos_label, nbins = 10000, auc_roc = True))
#
def classification_report(y_true: str = "", 
						  y_score: str = "", 
						  input_relation: str = "",
						  cursor = None,
						  labels: list = [],
						  cutoff: float = 0.5,
						  estimator = None):
	if (estimator):
		num_classes = len(estimator.classes) 
		labels = labels if (num_classes > 2) else [estimator.classes[1]]
	else:
		labels = [1] if not (labels) else labels
		num_classes = len(labels) + 1
	values = {"index": ["auc", "prc_auc", "accuracy", "log_loss", "precision", "recall", "f1-score", "mcc", "informedness", "markedness", "csi"]}
	for idx, elem in enumerate(labels):
		pos_label = elem
		non_pos_label = 0 if (elem == 1) else "Non-{}".format(elem)
		if (estimator):
			try:
				matrix = estimator.confusion_matrix(pos_label, cutoff) 
			except:
				matrix = estimator.confusion_matrix(pos_label) 
		else:
			y_s, y_p, y_t = y_score[0].format(elem), y_score[1], "DECODE({}, '{}', 1, 0)".format(y_true, elem)
			matrix = confusion_matrix(y_true, y_p, input_relation, cursor, pos_label)
		tn, fp, fn, tp = matrix.values[non_pos_label][0], matrix.values[non_pos_label][1], matrix.values[pos_label][0], matrix.values[pos_label][1]
		ppv = tp / (tp + fp) if (tp + fp != 0) else 0 # precision
		tpr = tp / (tp + fn) if (tp + fn != 0) else 0 # recall
		tnr = tn / (tn + fp) if (tn + fp != 0) else 0 
		npv = tn / (tn + fn) if (tn + fn != 0) else 0
		f1 = 2 * (tpr * tnr) / (tpr + tnr) if (tpr + tnr != 0) else 0 # f1
		csi = tp / (tp + fn + fp) if (tp + fn + fp != 0) else 0 # csi
		bm = tpr + tnr - 1 # informedness
		mk = ppv + npv - 1 # markedness
		mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp != 0) and (tp + fn != 0) and (tn + fp != 0) and (tn + fn != 0) else 0
		if (estimator):
			try:
				accuracy = estimator.score(pos_label = pos_label, method = "acc", cutoff = cutoff)
			except:
				accuracy = estimator.score(pos_label = pos_label, method = "acc")
			auc_score, logloss, prc_auc_score = estimator.score(pos_label = pos_label, method = "auc"), estimator.score(pos_label = pos_label, method = "log_loss"), estimator.score(pos_label = pos_label, method = "prc_auc")
		else:
			auc_score = auc(y_t, y_s, input_relation, cursor, 1)
			prc_auc_score = prc_auc(y_t, y_s, input_relation, cursor, 1)
			y_p = "DECODE({}, '{}', 1, 0)".format(y_p, elem)
			logloss = log_loss(y_t, y_s, input_relation, cursor, 1)
			accuracy = accuracy_score(y_t, y_p, input_relation, cursor)
		elem = "value" if (len(labels) == 1) else elem
		values[elem] = [auc_score, prc_auc_score, accuracy, logloss, ppv, tpr, f1, mcc, bm, mk, csi]
	return (tablesample(values, table_info = False))
#
def confusion_matrix(y_true: str, 
					 y_score: str, 
					 input_relation: str,
					 cursor,
					 pos_label = 1):
	query  = "SELECT CONFUSION_MATRIX(obs, response USING PARAMETERS num_classes = 2) OVER() FROM (SELECT DECODE({}".format(y_true) 
	query += ", '{}', 1, NULL, NULL, 0) AS obs, DECODE({}, '{}', 1, NULL, NULL, 0) AS response FROM {}) x".format(pos_label, y_score, pos_label, input_relation) 
	result = to_tablesample(query, cursor)
	if (pos_label == 1):
		labels = [0, 1]
	else:
		labels = ["Non-{}".format(pos_label), pos_label]
	result.table_info = False
	del (result.values["comment"])
	result = result.transpose()
	result.values["actual_class"] = labels
	result = result.transpose()
	matrix = {"index": labels}
	for elem in result.values:
		if (elem != "actual_class"):
			matrix[elem] = result.values[elem]
	result.values = matrix
	return (result)
#
def critical_success_index(y_true: str, 
			 		  	   y_score: str, 
			 		  	   input_relation: str,
			 		  	   cursor,
			 		  	   pos_label = 1):
	matrix = confusion_matrix(y_true, y_score, input_relation, cursor, pos_label)
	non_pos_label = 0 if (pos_label == 1) else "Non-{}".format(pos_label)
	tn, fp, fn, tp = matrix.values[non_pos_label][0], matrix.values[non_pos_label][1], matrix.values[pos_label][0], matrix.values[pos_label][1]
	csi = tp / (tp + fn + fp) if (tp + fn + fp != 0) else 0
	return (csi)
#
def f1_score(y_true: str, 
			 y_score: str, 
			 input_relation: str,
			 cursor,
			 pos_label = 1):
	matrix = confusion_matrix(y_true, y_score, input_relation, cursor, pos_label)
	non_pos_label = 0 if (pos_label == 1) else "Non-{}".format(pos_label)
	tn, fp, fn, tp = matrix.values[non_pos_label][0], matrix.values[non_pos_label][1], matrix.values[pos_label][0], matrix.values[pos_label][1]
	recall = tp / (tp + fn) if (tp + fn != 0) else 0
	precision = tp / (tp + fp) if (tp + fp != 0) else 0
	f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall != 0) else 0
	return (f1)
#
def informedness(y_true: str, 
			     y_score: str, 
			     input_relation: str,
			     cursor,
			     pos_label = 1):
	matrix = confusion_matrix(y_true, y_score, input_relation, cursor, pos_label)
	non_pos_label = 0 if (pos_label == 1) else "Non-{}".format(pos_label)
	tn, fp, fn, tp = matrix.values[non_pos_label][0], matrix.values[non_pos_label][1], matrix.values[pos_label][0], matrix.values[pos_label][1]
	tpr = tp / (tp + fn) if (tp + fn != 0) else 0
	tnr = tn / (tn + fp) if (tn + fp != 0) else 0
	return (tpr + tnr - 1)
# 
def log_loss(y_true: str, 
			 y_score: str, 
			 input_relation: str,
			 cursor,
			 pos_label = 1):
	query= "SELECT AVG(CASE WHEN {} = '{}' THEN - LOG({}::float + 1e-90) else - LOG(1 - {}::float + 1e-90) END) FROM {};"
	query = query.format(y_true, pos_label, y_score, y_score, input_relation)
	cursor.execute(query)
	return (cursor.fetchone()[0])
#
def markedness(y_true: str, 
			   y_score: str, 
			   input_relation: str,
			   cursor,
			   pos_label = 1):
	matrix = confusion_matrix(y_true, y_score, input_relation, cursor, pos_label)
	non_pos_label = 0 if (pos_label == 1) else "Non-{}".format(pos_label)
	tn, fp, fn, tp = matrix.values[non_pos_label][0], matrix.values[non_pos_label][1], matrix.values[pos_label][0], matrix.values[pos_label][1]
	ppv = tp / (tp + fp) if (tp + fp != 0) else 0
	npv = tn / (tn + fn) if (tn + fn != 0) else 0
	return (ppv + npv - 1)
#
def matthews_corrcoef(y_true: str, 
			 		  y_score: str, 
			 		  input_relation: str,
			 		  cursor,
			 		  pos_label = 1):
	matrix = confusion_matrix(y_true, y_score, input_relation, cursor, pos_label)
	non_pos_label = 0 if (pos_label == 1) else "Non-{}".format(pos_label)
	tn, fp, fn, tp = matrix.values[non_pos_label][0], matrix.values[non_pos_label][1], matrix.values[pos_label][0], matrix.values[pos_label][1]
	mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp != 0) and (tp + fn != 0) and (tn + fp != 0) and (tn + fn != 0) else 0
	return (mcc)
#
def multilabel_confusion_matrix(y_true: str, 
								y_score: str, 
								input_relation: str,
								cursor,
								labels: list):
	num_classes = str(len(labels))
	query = "SELECT CONFUSION_MATRIX(obs, response USING PARAMETERS num_classes = {}) OVER() FROM (SELECT DECODE({}".format(num_classes, y_true) 
	for idx, item in enumerate(labels):
		query += ", '" + str(item) + "', " + str(idx)
	query += ") AS obs, DECODE({}".format(y_score)
	for idx,item in enumerate(labels):
		query += ", '" + str(item) + "', " + str(idx)
	query += ") AS response FROM {}) x".format(input_relation)
	result = to_tablesample(query, cursor)
	result.table_info = False
	del (result.values["comment"])
	result = result.transpose()
	result.values["actual_class"] = labels
	result = result.transpose()
	matrix = {"index": labels}
	for elem in result.values:
		if (elem != "actual_class"):
			matrix[elem] = result.values[elem]
	result.values = matrix
	return (result)
#
def negative_predictive_score(y_true: str, 
				      		  y_score: str, 
				     		  input_relation: str,
				      		  cursor,
				      		  pos_label = 1):
	matrix = confusion_matrix(y_true, y_score, input_relation, cursor, pos_label)
	non_pos_label = 0 if (pos_label == 1) else "Non-{}".format(pos_label)
	tn, fp, fn, tp = matrix.values[non_pos_label][0], matrix.values[non_pos_label][1], matrix.values[pos_label][0], matrix.values[pos_label][1]
	npv = tn / (tn + fn) if (tn + fn != 0) else 0
	return (npv)
#
def prc_auc(y_true: str, 
			y_score: str, 
			input_relation: str,
			cursor,
			pos_label = 1):
	return (prc_curve(y_true, y_score, input_relation, cursor, pos_label, nbins = 10000, auc_prc = True))
#
def precision_score(y_true: str, 
				    y_score: str, 
				    input_relation: str,
				    cursor,
				    pos_label = 1):
	matrix = confusion_matrix(y_true, y_score, input_relation, cursor, pos_label)
	non_pos_label = 0 if (pos_label == 1) else "Non-{}".format(pos_label)
	tn, fp, fn, tp = matrix.values[non_pos_label][0], matrix.values[non_pos_label][1], matrix.values[pos_label][0], matrix.values[pos_label][1]
	precision = tp / (tp + fp) if (tp + fp != 0) else 0
	return (precision)
#
def recall_score(y_true: str, 
				 y_score: str, 
				 input_relation: str,
				 cursor,
				 pos_label = 1):
	matrix = confusion_matrix(y_true, y_score, input_relation, cursor, pos_label)
	non_pos_label = 0 if (pos_label == 1) else "Non-{}".format(pos_label)
	tn, fp, fn, tp = matrix.values[non_pos_label][0], matrix.values[non_pos_label][1], matrix.values[pos_label][0], matrix.values[pos_label][1]
	recall = tp / (tp + fn) if (tp + fn != 0) else 0
	return (recall)
#
def specificity_score(y_true: str, 
				      y_score: str, 
				      input_relation: str,
				      cursor,
				      pos_label = 1):
	matrix = confusion_matrix(y_true, y_score, input_relation, cursor, pos_label)
	non_pos_label = 0 if (pos_label == 1) else "Non-{}".format(pos_label)
	tn, fp, fn, tp = matrix.values[non_pos_label][0], matrix.values[non_pos_label][1], matrix.values[pos_label][0], matrix.values[pos_label][1]
	tnr = tn / (tn + fp) if (tn + fp != 0) else 0
	return (tnr)