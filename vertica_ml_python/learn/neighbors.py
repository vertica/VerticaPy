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
from vertica_ml_python import drop_model
from vertica_ml_python import tablesample
from vertica_ml_python import to_tablesample

from vertica_ml_python.learn.metrics import accuracy_score
from vertica_ml_python.learn.metrics import auc
from vertica_ml_python.learn.metrics import prc_auc
from vertica_ml_python.learn.metrics import log_loss
from vertica_ml_python.learn.metrics import classification_report
from vertica_ml_python.learn.metrics import confusion_matrix
from vertica_ml_python.learn.metrics import critical_success_index
from vertica_ml_python.learn.metrics import f1_score
from vertica_ml_python.learn.metrics import informedness
from vertica_ml_python.learn.metrics import markedness
from vertica_ml_python.learn.metrics import matthews_corrcoef
from vertica_ml_python.learn.metrics import multilabel_confusion_matrix
from vertica_ml_python.learn.metrics import negative_predictive_score
from vertica_ml_python.learn.metrics import precision_score
from vertica_ml_python.learn.metrics import recall_score
from vertica_ml_python.learn.metrics import specificity_score
from vertica_ml_python.learn.metrics import r2_score
from vertica_ml_python.learn.metrics import mean_absolute_error
from vertica_ml_python.learn.metrics import mean_squared_error
from vertica_ml_python.learn.metrics import mean_squared_log_error
from vertica_ml_python.learn.metrics import median_absolute_error
from vertica_ml_python.learn.metrics import max_error
from vertica_ml_python.learn.metrics import explained_variance
from vertica_ml_python.learn.metrics import regression_report

from vertica_ml_python.learn.plot import lift_chart
from vertica_ml_python.learn.plot import roc_curve
from vertica_ml_python.learn.plot import prc_curve

#
class NearestCentroid:
	#
	def  __init__(self,
				  cursor,
				  p: int = 2):
		self.type = "classifier"
		self.cursor = cursor
		self.p = p
	# 
	def __repr__(self):
		try:
			rep = "<NearestCentroid>\n\ncentroids:\n" + self.centroids.__repr__() + "\n\nclasses: {}".format(self.classes)
			return (rep)
		except:
			return "<NearestCentroid>"
	#
	#
	#
	# METHODS
	# 
	#
	def classification_report(self, cutoff: float = 0.5, labels = []):
		labels = self.classes if not(labels) else labels
		return (classification_report(cutoff = cutoff, estimator = self, labels = labels))
	#
	def confusion_matrix(self, pos_label = None, cutoff: float = 0.5):
		pos_label = self.classes[1] if (pos_label == None and len(self.classes) == 2) else pos_label
		if (pos_label in self.classes and cutoff < 1 and cutoff > 0):
			input_relation = self.deploySQL() + " WHERE predict_nc = '{}'".format(pos_label)
			y_score = "(CASE WHEN proba_predict > {} THEN 1 ELSE 0 END)".format(cutoff)
			y_true = "DECODE({}, '{}', 1, 0)".format(self.y, pos_label)
			result = confusion_matrix(y_true, y_score, input_relation, self.cursor)
			if pos_label == 1:
				return (result)
			else:
				return(tablesample(values = {"index": ["Non-{}".format(pos_label), "{}".format(pos_label)], "Non-{}".format(pos_label): result.values[0],  "{}".format(pos_label): result.values[1]}, table_info = False))
		else:
			return (multilabel_confusion_matrix(self.y, "predict_nc", self.deploySQL(predict = True), self.cursor, self.classes))
	#
	def deploySQL(self, predict: bool = False):
		sql = ["POWER(ABS(x.{} - y.{}), {})".format(self.X[i], self.X[i], self.p) for i in range(len(self.X))] 
		distance = "POWER({}, 1 / {})".format(" + ".join(sql), self.p)
		sql = "ROW_NUMBER() OVER(PARTITION BY {}, row_id ORDER BY {})".format(", ".join(['x.{}'.format(item) for item in self.X]), distance)
		where = " AND ".join(["{} IS NOT NULL".format(item) for item in self.X])
		sql = "(SELECT {}, {} AS ordered_distance, {} AS distance, x.{}, y.{} AS predict_nc FROM (SELECT *, ROW_NUMBER() OVER() AS row_id FROM {} WHERE {}) x CROSS JOIN ({}) y) nc_distance_table".format(", ".join(['x.{}'.format(item) for item in self.X]), sql, distance, self.y, self.y, self.test_relation, where, self.centroids.to_sql())
		if (predict):
			sql = "(SELECT {}, {}, predict_nc FROM {} WHERE ordered_distance = 1) nc_table".format(", ".join(self.X), self.y, sql)
		else:
			sql = "(SELECT {}, {}, predict_nc, (1 - DECODE(distance, 0, 0, (distance / SUM(distance) OVER (PARTITION BY {})))) / {} AS proba_predict, ordered_distance FROM {}) nc_table".format(", ".join(self.X), self.y, ", ".join(self.X), len(self.classes) - 1, sql)
		return sql
	#
	def deploy_to_DB(self, name: str, view: bool = True, cutoff = -1, all_classes: bool = False):
		relation = "TABLE" if not(view) else "VIEW"
		if (all_classes):
			predict = ["ZEROIFNULL(AVG(DECODE(predict_nc, '{}', proba_predict, NULL))) AS \"{}_{}\"".format(elem, self.y.replace('"', ''), elem) for elem in self.classes]
			sql = "CREATE {} {} AS SELECT {}, {} FROM {} GROUP BY {}".format(relation, name, ", ".join(self.X), ", ".join(predict), self.deploySQL(), ", ".join(self.X))
		else:
			if ((len(self.classes) == 2) and (cutoff <= 1 and cutoff >= 0)):
				sql = "CREATE {} {} AS SELECT {}, (CASE WHEN proba_predict > {} THEN '{}' ELSE '{}' END) AS {} FROM {} WHERE predict_nc = '{}'".format(relation, name, ", ".join(self.X), cutoff, self.classes[1], self.classes[0], self.y, self.deploySQL(), self.classes[1])
			elif (len(self.classes) == 2):
				sql = "CREATE {} {} AS SELECT {}, proba_predict AS {} FROM {} WHERE predict_nc = '{}'".format(relation, name, ", ".join(self.X), self.y, self.deploySQL(), self.classes[1])
			else:
				sql = "CREATE {} {} AS SELECT {}, predict_nc AS {} FROM {}".format(relation, name, ", ".join(self.X), self.y, self.deploySQL(True))
		self.cursor.execute(sql)
	#
	def fit(self,
			input_relation: str, 
			X: list, 
			y: str,
			test_relation: str = ""):
		func = "APPROXIMATE_MEDIAN" if (self.p == 1) else "AVG"
		self.input_relation = input_relation
		self.test_relation = test_relation if (test_relation) else input_relation
		self.X = ['"' + column.replace('"', '') + '"' for column in X]
		self.y = '"' + y.replace('"', '') + '"'
		query = "SELECT {}, {} FROM {} WHERE {} IS NOT NULL GROUP BY {}".format(", ".join(["{}({}) AS {}".format(func, column, column) for column in self.X]), self.y, input_relation, self.y, self.y)
		self.centroids = to_tablesample(query = query, cursor = self.cursor)
		self.centroids.table_info = False
		self.classes = self.centroids.values[y]
		return (self)
	#
	def prc_curve(self, pos_label = None):
		pos_label = self.classes[1] if (pos_label == None and len(self.classes) == 2) else pos_label
		if (pos_label not in self.classes):
			raise ValueError("'pos_label' must be one of the response column classes")
		input_relation = self.deploySQL() + " WHERE predict_nc = '{}'".format(pos_label)
		y_proba = "proba_predict"
		return (prc_curve(self.y, y_proba, input_relation, self.cursor, pos_label))
	#
	def roc_curve(self, pos_label = None):
		pos_label = self.classes[1] if (pos_label == None and len(self.classes) == 2) else pos_label
		if (pos_label not in self.classes):
			raise ValueError("'pos_label' must be one of the response column classes")
		input_relation = self.deploySQL() + " WHERE predict_nc = '{}'".format(pos_label)
		y_proba = "proba_predict"
		return (roc_curve(self.y, y_proba, input_relation, self.cursor, pos_label))
	#
	def score(self, pos_label = None, cutoff: float = 0.5, method: str = "accuracy"):
		pos_label = self.classes[1] if (pos_label == None and len(self.classes) == 2) else pos_label
		input_relation = self.deploySQL() + " WHERE predict_nc = '{}'".format(pos_label)
		y_score = "(CASE WHEN proba_predict > {} THEN 1 ELSE 0 END)".format(cutoff)
		y_proba = "proba_predict"
		y_true = "DECODE({}, '{}', 1, 0)".format(self.y, pos_label)
		if (pos_label not in self.classes):
			raise ValueError("'pos_label' must be one of the response column classes")
		elif (cutoff >= 1 or cutoff <= 0):
			raise ValueError("'cutoff' must be in ]0;1[")
		if (method in ("accuracy", "acc")):
			return (accuracy_score(y_true, y_score, input_relation, self.cursor))
		elif (method == "auc"):
			return (auc(y_true, y_proba, input_relation, self.cursor))
		elif (method == "prc_auc"):
			return (prc_auc(y_true, y_proba, input_relation, self.cursor))
		elif (method in ("best_cutoff", "best_threshold")):
			return (roc_curve(y_true, y_proba, input_relation, self.cursor, best_threshold = True))
		elif (method in ("recall", "tpr")):
			return (recall_score(y_true, y_score, input_relation, self.cursor))
		elif (method in ("precision", "ppv")):
			return (precision_score(y_true, y_score, input_relation, self.cursor))
		elif (method in ("specificity", "tnr")):
			return (specificity_score(y_true, y_score, input_relation, self.cursor))
		elif (method in ("negative_predictive_value", "npv")):
			return (precision_score(y_true, y_score, input_relation, self.cursor))
		elif (method in ("log_loss", "logloss")):
			return (log_loss(y_true, y_proba, input_relation, self.cursor))
		elif (method == "f1"):
			return (f1_score(y_true, y_score, input_relation, self.cursor))
		elif (method == "mcc"):
			return (matthews_corrcoef(y_true, y_score, input_relation, self.cursor))
		elif (method in ("bm", "informedness")):
			return (informedness(y_true, y_score, input_relation, self.cursor))
		elif (method in ("mk", "markedness")):
			return (markedness(y_true, y_score, input_relation, self.cursor))
		elif (method in ("csi", "critical_success_index")):
			return (critical_success_index(y_true, y_score, input_relation, self.cursor))
		else:
			raise ValueError("The parameter 'method' must be in accuracy|auc|prc_auc|best_cutoff|recall|precision|log_loss|negative_predictive_value|specificity|mcc|informedness|markedness|critical_success_index")
#
class KNeighborsClassifier:
	#
	def  __init__(self,
				  cursor,
				  n_neighbors: int = 5,
				  p: int = 2):
		self.type = "classifier"
		self.cursor = cursor
		self.n_neighbors = n_neighbors
		self.p = p
	# 
	def __repr__(self):
		return "<KNeighborsClassifier>"
	#
	#
	#
	# METHODS
	# 
	#
	def classification_report(self, cutoff: float = 0.5, labels = []):
		labels = self.classes if not(labels) else labels
		return (classification_report(cutoff = cutoff, estimator = self, labels = labels))
	#
	def confusion_matrix(self, pos_label = None, cutoff: float = 0.5):
		pos_label = self.classes[1] if (pos_label == None and len(self.classes) == 2) else pos_label
		if (pos_label in self.classes and cutoff < 1 and cutoff > 0):
			input_relation = self.deploySQL() + " WHERE predict_knc = '{}'".format(pos_label)
			y_score = "(CASE WHEN proba_predict > {} THEN 1 ELSE 0 END)".format(cutoff)
			y_true = "DECODE({}, '{}', 1, 0)".format(self.y, pos_label)
			result = confusion_matrix(y_true, y_score, input_relation, self.cursor)
			if pos_label == 1:
				return (result)
			else:
				return(tablesample(values = {"index": ["Non-{}".format(pos_label), "{}".format(pos_label)], "Non-{}".format(pos_label): result.values[0],  "{}".format(pos_label): result.values[1]}, table_info = False))
		else:
			input_relation = "(SELECT *, ROW_NUMBER() OVER(PARTITION BY {}, row_id ORDER BY proba_predict DESC) AS pos FROM {}) knc_table_predict WHERE pos = 1".format(", ".join(self.X), self.deploySQL())
			return (multilabel_confusion_matrix(self.y, "predict_knc", input_relation, self.cursor, self.classes))
	#
	def deploySQL(self, predict: bool = False):
		sql = ["POWER(ABS(x.{} - y.{}), {})".format(self.X[i], self.X[i], self.p) for i in range(len(self.X))] 
		sql = "POWER({}, 1 / {})".format(" + ".join(sql), self.p)
		sql = "ROW_NUMBER() OVER(PARTITION BY {}, row_id ORDER BY {})".format(", ".join(['x.{}'.format(item) for item in self.X]), sql)
		where = " AND ".join(["{} IS NOT NULL".format(item) for item in self.X])
		sql = "SELECT {}, {} AS ordered_distance, x.{}, y.{} AS predict_knc, row_id FROM (SELECT *, ROW_NUMBER() OVER() AS row_id FROM {} WHERE {}) x CROSS JOIN (SELECT * FROM {} WHERE {}) y".format(", ".join(['x.{}'.format(item) for item in self.X]), sql, self.y, self.y, self.test_relation, where, self.input_relation, where)
		sql = "(SELECT row_id, {}, {}, predict_knc, COUNT(*) / {} AS proba_predict FROM ({}) z WHERE ordered_distance <= {} GROUP BY {}, {}, row_id, predict_knc) knc_table".format(", ".join(self.X), self.y, self.n_neighbors, sql, self.n_neighbors, ", ".join(self.X), self.y)
		if (predict):
			sql = "(SELECT {}, {}, predict_knc FROM (SELECT {}, {}, predict_knc, ROW_NUMBER() OVER (PARTITION BY {} ORDER BY proba_predict DESC) AS order_prediction FROM {}) x WHERE order_prediction = 1) predict_knc_table".format(", ".join(self.X), self.y, ", ".join(self.X), self.y, ", ".join(self.X), sql)
		return sql
	#
	def deploy_to_DB(self, name: str, view: bool = True, cutoff = -1, all_classes: bool = False):
		relation = "TABLE" if not(view) else "VIEW"
		if (all_classes):
			predict = ["ZEROIFNULL(AVG(DECODE(predict_knc, '{}', proba_predict, NULL))) AS \"{}_{}\"".format(elem, self.y.replace('"', ''), elem) for elem in self.classes]
			sql = "CREATE {} {} AS SELECT {}, {} FROM {} GROUP BY {}".format(relation, name, ", ".join(self.X), ", ".join(predict), self.deploySQL(), ", ".join(self.X))
		else:
			if ((len(self.classes) == 2) and (cutoff <= 1 and cutoff >= 0)):
				sql = "CREATE {} {} AS SELECT {}, (CASE WHEN proba_predict > {} THEN '{}' ELSE '{}' END) AS {} FROM {} WHERE predict_knc = '{}'".format(relation, name, ", ".join(self.X), cutoff, self.classes[1], self.classes[0], self.y, self.deploySQL(), self.classes[1])
			elif (len(self.classes) == 2):
				sql = "CREATE {} {} AS SELECT {}, proba_predict AS {} FROM {} WHERE predict_knc = '{}'".format(relation, name, ", ".join(self.X), self.y, self.deploySQL(), self.classes[1])
			else:
				sql = "CREATE {} {} AS SELECT {}, predict_knc AS {} FROM {}".format(relation, name, ", ".join(self.X), self.y, self.deploySQL(True))
		self.cursor.execute(sql)
	#
	def fit(self,
			input_relation: str,
			X: list, 
			y: str,
			test_relation: str = ""):
		self.input_relation = input_relation
		self.test_relation = test_relation if (test_relation) else input_relation
		self.X = ['"' + column.replace('"', '') + '"' for column in X]
		self.y = '"' + y.replace('"', '') + '"'
		self.cursor.execute("SELECT DISTINCT {} FROM {} WHERE {} IS NOT NULL ORDER BY 1".format(self.y, input_relation, self.y))
		classes = self.cursor.fetchall()
		self.classes = [item[0] for item in classes]
		return (self)
	#
	def lift_chart(self, pos_label = None):
		pos_label = self.classes[1] if (pos_label == None and len(self.classes) == 2) else pos_label
		if (pos_label not in self.classes):
			raise ValueError("'pos_label' must be one of the response column classes")
		input_relation = self.deploySQL() + " WHERE predict_knc = '{}'".format(pos_label)
		y_proba = "proba_predict"
		return (lift_chart(self.y, y_proba, input_relation, self.cursor, pos_label))
	#
	def prc_curve(self, pos_label = None):
		pos_label = self.classes[1] if (pos_label == None and len(self.classes) == 2) else pos_label
		if (pos_label not in self.classes):
			raise ValueError("'pos_label' must be one of the response column classes")
		input_relation = self.deploySQL() + " WHERE predict_knc = '{}'".format(pos_label)
		y_proba = "proba_predict"
		return (prc_curve(self.y, y_proba, input_relation, self.cursor, pos_label))
	#
	def roc_curve(self, pos_label = None):
		pos_label = self.classes[1] if (pos_label == None and len(self.classes) == 2) else pos_label
		if (pos_label not in self.classes):
			raise ValueError("'pos_label' must be one of the response column classes")
		input_relation = self.deploySQL() + " WHERE predict_knc = '{}'".format(pos_label)
		y_proba = "proba_predict"
		return (roc_curve(self.y, y_proba, input_relation, self.cursor, pos_label))
	#
	def score(self, pos_label = None, cutoff: float = 0.5, method: str = "accuracy"):
		pos_label = self.classes[1] if (pos_label == None and len(self.classes) == 2) else pos_label
		input_relation = self.deploySQL() + " WHERE predict_knc = '{}'".format(pos_label)
		y_score = "(CASE WHEN proba_predict > {} THEN 1 ELSE 0 END)".format(cutoff)
		y_proba = "proba_predict"
		y_true = "DECODE({}, '{}', 1, 0)".format(self.y, pos_label)
		if (pos_label not in self.classes):
			raise ValueError("'pos_label' must be one of the response column classes")
		elif (cutoff >= 1 or cutoff <= 0):
			raise ValueError("'cutoff' must be in ]0;1[")
		if (method in ("accuracy", "acc")):
			return (accuracy_score(y_true, y_score, input_relation, self.cursor))
		elif (method == "auc"):
			return (auc(y_true, y_proba, input_relation, self.cursor))
		elif (method == "prc_auc"):
			return (prc_auc(y_true, y_proba, input_relation, self.cursor))
		elif (method in ("best_cutoff", "best_threshold")):
			return (roc_curve(y_true, y_proba, input_relation, self.cursor, best_threshold = True))
		elif (method in ("recall", "tpr")):
			return (recall_score(y_true, y_score, input_relation, self.cursor))
		elif (method in ("precision", "ppv")):
			return (precision_score(y_true, y_score, input_relation, self.cursor))
		elif (method in ("specificity", "tnr")):
			return (specificity_score(y_true, y_score, input_relation, self.cursor))
		elif (method in ("negative_predictive_value", "npv")):
			return (precision_score(y_true, y_score, input_relation, self.cursor))
		elif (method in ("log_loss", "logloss")):
			return (log_loss(y_true, y_proba, input_relation, self.cursor))
		elif (method == "f1"):
			return (f1_score(y_true, y_score, input_relation, self.cursor))
		elif (method == "mcc"):
			return (matthews_corrcoef(y_true, y_score, input_relation, self.cursor))
		elif (method in ("bm", "informedness")):
			return (informedness(y_true, y_score, input_relation, self.cursor))
		elif (method in ("mk", "markedness")):
			return (markedness(y_true, y_score, input_relation, self.cursor))
		elif (method in ("csi", "critical_success_index")):
			return (critical_success_index(y_true, y_score, input_relation, self.cursor))
		else:
			raise ValueError("The parameter 'method' must be in accuracy|auc|prc_auc|best_cutoff|recall|precision|log_loss|negative_predictive_value|specificity|mcc|informedness|markedness|critical_success_index")
#
class KNeighborsRegressor:
	#
	def  __init__(self,
				  cursor,
				  n_neighbors: int = 5,
				  p: int = 2):
		self.type = "regressor"
		self.cursor = cursor
		self.n_neighbors = n_neighbors
		self.p = p
	# 
	def __repr__(self):
		return "<KNeighborsRegressor>"
	#
	#
	#
	# METHODS
	# 
	#
	def deploySQL(self):
		sql = ["POWER(ABS(x.{} - y.{}), {})".format(self.X[i], self.X[i], self.p) for i in range(len(self.X))] 
		sql = "POWER({}, 1 / {})".format(" + ".join(sql), self.p)
		sql = "ROW_NUMBER() OVER(PARTITION BY {}, row_id ORDER BY {})".format(", ".join(['x.{}'.format(item) for item in self.X]), sql)
		where = " AND ".join(["{} IS NOT NULL".format(item) for item in self.X])
		sql = "SELECT {}, {} AS ordered_distance, x.{}, y.{} AS predict_knr, row_id FROM (SELECT *, ROW_NUMBER() OVER() AS row_id FROM {} WHERE {}) x CROSS JOIN (SELECT * FROM {} WHERE {}) y".format(", ".join(['x.{}'.format(item) for item in self.X]), sql, self.y, self.y, self.test_relation, where, self.input_relation, where)
		sql = "(SELECT {}, {}, AVG(predict_knr) AS predict_knr FROM ({}) z WHERE ordered_distance <= {} GROUP BY {}, {}, row_id) knr_table".format(", ".join(self.X), self.y, sql, self.n_neighbors, ", ".join(self.X), self.y)
		return sql
	#
	def deploy_to_DB(self, name: str, view: bool = True):
		relation = "TABLE" if not(view) else "VIEW"
		sql = "CREATE {} {} AS SELECT {}, {} AS {} FROM {}".format(relation, name, ", ".join(self.X), "predict_knr", self.y, self.deploySQL())
		self.cursor.execute(sql)
	#
	def fit(self,
			input_relation: str, 
			X: list, 
			y: str,
			test_relation: str = ""):
		self.input_relation = input_relation
		self.test_relation = test_relation if (test_relation) else input_relation
		self.X = ['"' + column.replace('"', '') + '"' for column in X]
		self.y = '"' + y.replace('"', '') + '"'
		return (self)
	#
	def regression_report(self):
		return (regression_report(self.y, "predict_knr", self.deploySQL(), self.cursor))
	#
	def score(self, method: str = "r2"):
		if (method in ("r2", "rsquared")):
			return (r2_score(self.y, "predict_knr", self.deploySQL(), self.cursor))
		elif (method in ("mae", "mean_absolute_error")):
			return (mean_absolute_error(self.y, "predict_knr", self.deploySQL(), self.cursor))
		elif (method in ("mse", "mean_squared_error")):
			return (mean_squared_error(self.y, "predict_knr", self.deploySQL(), self.cursor))
		elif (method in ("msle", "mean_squared_log_error")):
			return (mean_squared_log_error(self.y, "predict_knr", self.deploySQL(), self.cursor))
		elif (method in ("max", "max_error")):
			return (max_error(self.y, "predict_knr", self.deploySQL(), self.cursor))
		elif (method in ("median", "median_absolute_error")):
			return (median_absolute_error(self.y, "predict_knr", self.deploySQL(), self.cursor))
		elif (method in ("var", "explained_variance")):
			return (explained_variance(self.y, "predict_knr", self.deploySQL(), self.cursor))
		else:
			raise ValueError("The parameter 'method' must be in r2|mae|mse|msle|max|median|var")
#
class LocalOutlierFactor:
	#
	def  __init__(self, 
				  name: str, 
				  cursor, 
				  n_neighbors: int = 20, 
				  p: int = 2):
		self.type = "anomaly_detection"
		self.name = name
		self.cursor = cursor
		self.n_neighbors = n_neighbors
		self.p = p
	# 
	def __repr__(self):
		return "<LocalOutlierFactor>"
	#
	#
	#
	# METHODS
	#
	#
	def fit(self, input_relation: str, X: list, key_columns: list = [], index = ""):
		X = ['"' + column.replace('"', '') + '"' for column in X]
		self.X = X
		self.key_columns = ['"' + column.replace('"', '') + '"' for column in key_columns]
		self.input_relation = input_relation
		cursor = self.cursor
		n_neighbors = self.n_neighbors
		p = self.p
		if not(index):
			index = "id"
			main_table = "main_{}_vpython".format(input_relation)
			cursor.execute("DROP TABLE IF EXISTS {}".format(main_table))
			sql = "CREATE TABLE {} AS SELECT ROW_NUMBER() OVER() AS id, {} FROM {} WHERE {}".format(main_table, ", ".join(X + key_columns), input_relation, " AND ".join(["{} IS NOT NULL".format(item) for item in X]))
			cursor.execute(sql)
		else:
			main_table = input_relation
		sql = ["POWER(ABS(x.{} - y.{}), {})".format(X[i], X[i], p) for i in range(len(X))] 
		distance = "POWER({}, 1 / {})".format(" + ".join(sql), p)
		sql = "SELECT x.{} AS node_id, y.{} AS nn_id, {} AS distance, ROW_NUMBER() OVER(PARTITION BY x.{} ORDER BY {}) AS knn FROM {} AS x CROSS JOIN {} AS y".format(index, index, distance, index, distance, main_table, main_table)
		sql = "SELECT node_id, nn_id, distance, knn FROM ({}) distance_table WHERE knn <= {}".format(sql, n_neighbors + 1)
		cursor.execute("DROP TABLE IF EXISTS distance_{}_vpython".format(input_relation))
		sql = "CREATE TABLE distance_{}_vpython AS {}".format(input_relation, sql)
		cursor.execute(sql)
		kdistance = "(SELECT node_id, nn_id, distance AS distance FROM distance_{}_vpython WHERE knn = {}) AS kdistance_table".format(input_relation, n_neighbors + 1)
		lrd = "SELECT distance_table.node_id, {} / SUM(CASE WHEN distance_table.distance > kdistance_table.distance THEN distance_table.distance ELSE kdistance_table.distance END) AS lrd FROM (distance_{}_vpython AS distance_table LEFT JOIN {} ON distance_table.nn_id = kdistance_table.node_id) x GROUP BY 1".format(n_neighbors, input_relation, kdistance)
		cursor.execute("DROP TABLE IF EXISTS lrd_{}_vpython".format(input_relation))
		sql = "CREATE TABLE lrd_{}_vpython AS {}".format(input_relation, lrd)
		cursor.execute(sql)
		sql = "SELECT x.node_id, SUM(y.lrd) / (MAX(x.node_lrd) * {}) AS LOF FROM (SELECT n_table.node_id, n_table.nn_id, lrd_table.lrd AS node_lrd FROM distance_{}_vpython AS n_table LEFT JOIN lrd_{}_vpython AS lrd_table ON n_table.node_id = lrd_table.node_id) x LEFT JOIN lrd_{}_vpython AS y ON x.nn_id = y.node_id GROUP BY 1".format(n_neighbors, input_relation, input_relation, input_relation)
		cursor.execute("DROP TABLE IF EXISTS lof_{}_vpython".format(input_relation))
		sql = "CREATE TABLE lof_{}_vpython AS {}".format(input_relation, sql)
		cursor.execute(sql)
		sql = "SELECT {}, (CASE WHEN lof > 1e100 OR lof != lof THEN 0 ELSE lof END) AS lof_score FROM {} AS x LEFT JOIN lof_{}_vpython AS y ON x.{} = y.node_id".format(", ".join(X + self.key_columns), main_table, input_relation, index)
		sql = "CREATE TABLE {} AS {}".format(self.name, sql)
		cursor.execute(sql)
		sql = "SELECT COUNT(*) FROM lof_{}_vpython z WHERE lof > 1e100 OR lof != lof".format(input_relation)
		cursor.execute(sql)
		self.n_errors = cursor.fetchone()[0]
		cursor.execute("DROP TABLE IF EXISTS main_{}_vpython".format(input_relation))
		cursor.execute("DROP TABLE IF EXISTS distance_{}_vpython".format(input_relation))
		cursor.execute("DROP TABLE IF EXISTS lrd_{}_vpython".format(input_relation))
		cursor.execute("DROP TABLE IF EXISTS lof_{}_vpython".format(input_relation))
		return (self)
	#
	def info(self):
		if (self.n_errors == 0):
			print("All the LOF scores were computed.")
		else:
			print("{} error(s) happened during the computation. These ones were imputed by 0 as it is highly probable that the {}-Neighbors of these points were confounded (usual problem of the LOF computation).\nIncrease the number of Neighbors to decrease the number of errors.".format(self.n_errors, self.n_neighbors))
	#
	def plot(self, X: list = [], tablesample: float = -1):
		X = self.X if not(X) else X
		from vertica_ml_python.learn.plot import lof_plot
		lof_plot(self.name, self.cursor, X, "lof_score", tablesample)
	#
	def to_vdf(self):
		from vertica_ml_python import vDataframe
		return (vDataframe(self.name, self.cursor))

