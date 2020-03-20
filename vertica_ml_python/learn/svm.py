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
from vertica_ml_python.learn.metrics import log_loss

from vertica_ml_python.learn.plot import lift_chart
from vertica_ml_python.learn.plot import plot_importance
from vertica_ml_python.learn.plot import regression_plot
from vertica_ml_python.learn.plot import svm_classifier_plot
from vertica_ml_python.learn.plot import roc_curve
from vertica_ml_python.learn.plot import prc_curve

#
class LinearSVC:
	#
	def  __init__(self,
				  name: str,
				  cursor,
				  tol: float = 1e-4, 
				  C: float = 1.0, 
				  fit_intercept: bool = True,
				  intercept_scaling: float = 1.0,
				  intercept_mode: str = "regularized",
				  class_weight: list = [1, 1],
				  max_iter: int = 100):
		self.type = "classifier"
		self.classes = [0, 1]
		self.cursor = cursor
		self.name = name
		self.tol = tol
		self.C = C 
		self.fit_intercept = fit_intercept 
		self.intercept_scaling = intercept_scaling
		self.intercept_mode = intercept_mode
		self.class_weight = class_weight
		self.max_iter = max_iter
	# 
	def __repr__(self):
		try:
			self.cursor.execute("SELECT GET_MODEL_SUMMARY(USING PARAMETERS model_name = '" + self.name + "')")
			return (self.cursor.fetchone()[0])
		except:
			return "<LinearSVC>"
	#
	#
	#
	# METHODS
	# 
	#
	def add_to_vdf(self,
				   vdf,
				   name: str = "",
				   cutoff: float = 0.5):
		name = "LinearSVC_" + self.name if not (name) else name
		return (vdf.eval(name, self.deploySQL(cutoff)))
	#
	def classification_report(self, cutoff: float = 0.5):
		return (classification_report(self.y, [self.deploySQL(), self.deploySQL(cutoff)], self.test_relation, self.cursor))
	#
	def confusion_matrix(self, cutoff: float = 0.5):
		return (confusion_matrix(self.y, self.deploySQL(cutoff), self.test_relation, self.cursor))
	#
	def deploySQL(self, cutoff: float = -1):
		sql = "PREDICT_SVM_CLASSIFIER({} USING PARAMETERS model_name = '{}', type = 'probability', match_by_pos = 'true')"
		if (cutoff <= 1 and cutoff >= 0):
			sql = "(CASE WHEN {} > {} THEN 1 ELSE 0 END)".format(sql, cutoff)
		return (sql.format(", ".join(self.X), self.name))
	#
	def deploy_to_DB(self, name: str, view: bool = True, cutoff: float = -1):
		relation = "TABLE" if not(view) else "VIEW"
		sql = "CREATE {} {} AS SELECT {}, {} AS {} FROM {}".format(relation, name, ", ".join(self.X), self.deploySQL(cutoff), self.y, self.test_relation)
		self.cursor.execute(sql)
	#
	def drop(self):
		drop_model(self.name, self.cursor, print_info = False)
	#
	def features_importance(self):
		query  = "SELECT predictor, ROUND(100 * importance / SUM(importance) OVER(), 2) AS importance, sign FROM "
		query += "(SELECT stat.predictor AS predictor, ABS(coefficient * (max - min)) AS importance, SIGN(coefficient) AS sign FROM "
		query += "(SELECT LOWER(\"column\") AS predictor, min, max FROM (SELECT SUMMARIZE_NUMCOL({}) OVER() ".format(", ".join(self.X))
		query += " FROM {}) x) stat NATURAL JOIN (SELECT GET_MODEL_ATTRIBUTE (USING PARAMETERS model_name = '{}', ".format(self.input_relation, self.name)
		query += "attr_name = 'details')) coeff) importance_t ORDER BY 2 DESC;"
		self.cursor.execute(query)
		result = self.cursor.fetchall()
		coeff_importances, coeff_sign = {}, {}
		for elem in result:
			coeff_importances[elem[0]] = elem[1]
			coeff_sign[elem[0]] = elem[2]
		try:
			plot_importance(coeff_importances, coeff_sign)
		except:
			pass
		importances = {"index": ["importance"]}
		for elem in coeff_importances:
			importances[elem] = [coeff_importances[elem]]
		return (tablesample(values = importances, table_info = False).transpose())
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
		query = "SELECT SVM_CLASSIFIER('{}', '{}', '{}', '{}' USING PARAMETERS C = {}, epsilon = {}, max_iterations = {}"
		query = query.format(self.name, input_relation, self.y, ", ".join(self.X), self.C, self.tol, self.max_iter)
		query += ", class_weights = '{}'".format(", ".join([str(item) for item in self.class_weight]))
		if (self.fit_intercept):
			query += ", intercept_mode = '{}', intercept_scaling = {}".format(self.intercept_mode, self.intercept_scaling)
		query += ")"
		self.cursor.execute(query)
		self.coef = to_tablesample(query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'details')".format(self.name), cursor = self.cursor)
		self.coef.table_info = False
		return (self)
	#
	def lift_chart(self):
		return (lift_chart(self.y, self.deploySQL(), self.test_relation, self.cursor))
	#
	def plot(self, max_nb_points: int = 50):
		coefficients = self.coef.values["coefficient"]
		svm_classifier_plot(self.X, self.y, self.input_relation, self.cursor, coefficients, max_nb_points)
	#
	def prc_curve(self):
		return (prc_curve(self.y, self.deploySQL(), self.test_relation, self.cursor))
	#
	def roc_curve(self):
		return (roc_curve(self.y, self.deploySQL(), self.test_relation, self.cursor))
	#
	def score(self, cutoff: float = 0.5, method: str = "accuracy"):
		if (method in ("accuracy", "acc")):
			return (accuracy_score(self.y, self.deploySQL(cutoff), self.test_relation, self.cursor))
		elif (method == "auc"):
			return auc(self.y, self.deploySQL(), self.test_relation, self.cursor)
		elif (method == "prc_auc"):
			return prc_auc(self.y, self.deploySQL(), self.test_relation, self.cursor)
		elif (method in ("best_cutoff", "best_threshold")):
			return (roc_curve(self.y, self.deploySQL(), self.test_relation, self.cursor, best_threshold = True))
		elif (method in ("recall", "tpr")):
			return (recall_score(self.y, self.deploySQL(cutoff), self.test_relation, self.cursor))
		elif (method in ("precision", "ppv")):
			return (precision_score(self.y, self.deploySQL(cutoff), self.test_relation, self.cursor))
		elif (method in ("specificity", "tnr")):
			return (specificity_score(self.y, self.deploySQL(cutoff), self.test_relation, self.cursor))
		elif (method in ("negative_predictive_value", "npv")):
			return (precision_score(self.y, self.deploySQL(cutoff), self.test_relation, self.cursor))
		elif (method in ("log_loss", "logloss")):
			return (log_loss(self.y, self.deploySQL(), self.test_relation, self.cursor))
		elif (method == "f1"):
			return (f1_score(self.y, self.deploySQL(cutoff), self.test_relation, self.cursor))
		elif (method == "mcc"):
			return (matthews_corrcoef(self.y, self.deploySQL(cutoff), self.test_relation, self.cursor))
		elif (method in ("bm", "informedness")):
			return (informedness(self.y, self.deploySQL(cutoff), self.test_relation, self.cursor))
		elif (method in ("mk", "markedness")):
			return (markedness(self.y, self.deploySQL(cutoff), self.test_relation, self.cursor))
		elif (method in ("csi", "critical_success_index")):
			return (critical_success_index(self.y, self.deploySQL(cutoff), self.test_relation, self.cursor))
		else:
			raise ValueError("The parameter 'method' must be in accuracy|auc|prc_auc|best_cutoff|recall|precision|log_loss|negative_predictive_value|specificity|mcc|informedness|markedness|critical_success_index")
#
class LinearSVR:
	#
	def  __init__(self,
				  name: str,
				  cursor,
				  tol: float = 1e-4, 
				  C: float = 1.0, 
				  fit_intercept: bool = True,
				  intercept_scaling: float = 1.0,
				  intercept_mode: str = "regularized",
				  acceptable_error_margin: float = 0.1,
				  max_iter: int = 100):
		self.type = "regressor"
		self.cursor = cursor
		self.name = name
		self.tol = tol
		self.C = C 
		self.fit_intercept = fit_intercept 
		self.intercept_scaling = intercept_scaling
		self.intercept_mode = intercept_mode
		self.acceptable_error_margin = acceptable_error_margin
		self.max_iter = max_iter
	# 
	def __repr__(self):
		try:
			self.cursor.execute("SELECT GET_MODEL_SUMMARY(USING PARAMETERS model_name = '" + self.name + "')")
			return (self.cursor.fetchone()[0])
		except:
			return "<LinearSVR>"
	#
	#
	#
	# METHODS
	# 
	#
	def add_to_vdf(self,
				   vdf,
				   name: str = ""):
		name = "LinearSVR_" + self.name if not (name) else name
		return (vdf.eval(name, self.deploySQL()))
	#
	def deploySQL(self):
		sql = "PREDICT_SVM_REGRESSOR({} USING PARAMETERS model_name = '{}', match_by_pos = 'true')"
		return (sql.format(", ".join(self.X), self.name))
	#
	def deploy_to_DB(self, name: str, view: bool = True):
		relation = "TABLE" if not(view) else "VIEW"
		sql = "CREATE {} {} AS SELECT {}, {} AS {} FROM {}".format(relation, name, ", ".join(self.X), self.deploySQL(), self.y, self.test_relation)
		self.cursor.execute(sql)
	#
	def drop(self):
		drop_model(self.name, self.cursor, print_info = False)
	#
	def features_importance(self):
		query  = "SELECT predictor, ROUND(100 * importance / SUM(importance) OVER(), 2) AS importance, sign FROM "
		query += "(SELECT stat.predictor AS predictor, ABS(coefficient * (max - min)) AS importance, SIGN(coefficient) AS sign FROM "
		query += "(SELECT LOWER(\"column\") AS predictor, min, max FROM (SELECT SUMMARIZE_NUMCOL({}) OVER() ".format(", ".join(self.X))
		query += " FROM {}) x) stat NATURAL JOIN (SELECT GET_MODEL_ATTRIBUTE (USING PARAMETERS model_name = '{}', ".format(self.input_relation, self.name)
		query += "attr_name = 'details')) coeff) importance_t ORDER BY 2 DESC;"
		self.cursor.execute(query)
		result = self.cursor.fetchall()
		coeff_importances, coeff_sign = {}, {}
		for elem in result:
			coeff_importances[elem[0]] = elem[1]
			coeff_sign[elem[0]] = elem[2]
		try:
			plot_importance(coeff_importances, coeff_sign)
		except:
			pass
		importances = {"index": ["importance"]}
		for elem in coeff_importances:
			importances[elem] = [coeff_importances[elem]]
		return (tablesample(values = importances, table_info = False).transpose())
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
		query = "SELECT SVM_REGRESSOR('{}', '{}', '{}', '{}' USING PARAMETERS C = {}, epsilon = {}, max_iterations = {}"
		query = query.format(self.name, input_relation, self.y, ", ".join(self.X), self.C, self.tol, self.max_iter)
		query += ", error_tolerance = {}".format(self.acceptable_error_margin)
		if (self.fit_intercept):
			query += ", intercept_mode = '{}', intercept_scaling = {}".format(self.intercept_mode, self.intercept_scaling)
		query += ")"
		self.cursor.execute(query)
		self.coef = to_tablesample(query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'details')".format(self.name), cursor = self.cursor)
		self.coef.table_info = False
		return (self)
	#
	def plot(self, max_nb_points: int = 50):
		coefficients = self.coef.values["coefficient"]
		regression_plot(self.X, self.y, self.input_relation, self.cursor, coefficients, max_nb_points)
	#
	def regression_report(self):
		return (regression_report(self.y, self.deploySQL(), self.test_relation, self.cursor))
	#
	def score(self, method: str = "r2"):
		if (method in ("r2", "rsquared")):
			return (r2_score(self.y, self.deploySQL(), self.test_relation, self.cursor))
		elif (method in ("mae", "mean_absolute_error")):
			return (mean_absolute_error(self.y, self.deploySQL(), self.test_relation, self.cursor))
		elif (method in ("mse", "mean_squared_error")):
			return (mean_squared_error(self.y, self.deploySQL(), self.test_relation, self.cursor))
		elif (method in ("msle", "mean_squared_log_error")):
			return (mean_squared_log_error(self.y, self.deploySQL(), self.test_relation, self.cursor))
		elif (method in ("max", "max_error")):
			return (max_error(self.y, self.deploySQL(), self.test_relation, self.cursor))
		elif (method in ("median", "median_absolute_error")):
			return (median_absolute_error(self.y, self.deploySQL(), self.test_relation, self.cursor))
		elif (method in ("var", "explained_variance")):
			return (explained_variance(self.y, self.deploySQL(), self.test_relation, self.cursor))
		else:
			raise ValueError("The parameter 'method' must be in r2|mae|mse|msle|max|median|var")
