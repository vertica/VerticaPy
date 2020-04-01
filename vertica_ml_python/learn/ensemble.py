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
from vertica_ml_python.learn.metrics import accuracy_score, auc, prc_auc, log_loss, classification_report, confusion_matrix, critical_success_index, f1_score, informedness, negative_predictive_score, precision_score, recall_score, markedness, matthews_corrcoef, multilabel_confusion_matrix, specificity_score, r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, max_error, explained_variance, regression_report
from vertica_ml_python.learn.plot import lift_chart, plot_importance, roc_curve, prc_curve, plot_tree
from vertica_ml_python.utilities import str_column, drop_model, tablesample, to_tablesample
#
class RandomForestClassifier:
	#
	def  __init__(self,
				  name: str,
				  cursor,
				  n_estimators: int = 10,
				  max_features = "auto",
				  max_leaf_nodes: int = 1e9, 
				  sample: float = 0.632, 
				  max_depth: int = 5,
				  min_samples_leaf: int = 1,
				  min_info_gain: float = 0.0,
				  nbins: int = 32):
		self.type = "classifier"
		self.cursor = cursor
		self.name = name
		self.n_estimators = n_estimators
		self.max_features = max_features
		self.max_leaf_nodes = max_leaf_nodes
		self.sample = sample 
		self.max_depth = max_depth
		self.min_samples_leaf = min_samples_leaf
		self.min_info_gain = min_info_gain
		self.nbins = nbins
	# 
	def __repr__(self):
		try:
			self.cursor.execute("SELECT GET_MODEL_SUMMARY(USING PARAMETERS model_name = '" + self.name + "')")
			return (self.cursor.fetchone()[0])
		except:
			return "<RandomForestClassifier>"
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
		name = "RandomForestClassifier_" + self.name if not (name) else name
		pos_label = self.classes[1] if (len(self.classes) == 2) else None
		return (vdf.eval(name, self.deploySQL(pos_label, cutoff)))
	#
	def classification_report(self, cutoff: float = 0.5, labels = []):
		labels = self.classes if not(labels) else labels
		return (classification_report(cutoff = cutoff, estimator = self, labels = labels))
	#
	def confusion_matrix(self, pos_label = None, cutoff: float = 0.5):
		pos_label = self.classes[1] if (pos_label == None and len(self.classes) == 2) else pos_label
		if (pos_label in self.classes and cutoff < 1 and cutoff > 0):
			return (confusion_matrix(self.y, self.deploySQL(pos_label, cutoff), self.test_relation, self.cursor, pos_label = pos_label))
		else:
			return (multilabel_confusion_matrix(self.y, self.deploySQL(), self.test_relation, self.cursor, self.classes))
	#
	def deploySQL(self, pos_label = None, cutoff: float = -1, allSQL: bool = False):
		if (allSQL):
			sql = "PREDICT_RF_CLASSIFIER({} USING PARAMETERS model_name = '{}', class = '{}', type = 'probability', match_by_pos = 'true')".format(", ".join(self.X), self.name, "{}")
			sql = [sql, "PREDICT_RF_CLASSIFIER({} USING PARAMETERS model_name = '{}', match_by_pos = 'true')".format(", ".join(self.X), self.name)]
		else:
			if (pos_label in self.classes and cutoff <= 1 and cutoff >= 0):
				sql = "PREDICT_RF_CLASSIFIER({} USING PARAMETERS model_name = '{}', class = '{}', type = 'probability', match_by_pos = 'true')".format(", ".join(self.X), self.name, pos_label)
				if (len(self.classes) > 2):
					sql = "(CASE WHEN {} >= {} THEN '{}' WHEN {} IS NULL THEN NULL ELSE 'Non-{}' END)".format(sql, cutoff, pos_label, sql, pos_label)
				else:
					non_pos_label = self.classes[0] if (self.classes[0] != pos_label) else self.classes[1]
					sql = "(CASE WHEN {} >= {} THEN '{}' WHEN {} IS NULL THEN NULL ELSE '{}' END)".format(sql, cutoff, pos_label, sql, non_pos_label)
			elif (pos_label in self.classes):
				sql = "PREDICT_RF_CLASSIFIER({} USING PARAMETERS model_name = '{}', class = '{}', type = 'probability', match_by_pos = 'true')".format(", ".join(self.X), self.name, pos_label)
			else:
				sql = "PREDICT_RF_CLASSIFIER({} USING PARAMETERS model_name = '{}', match_by_pos = 'true')".format(", ".join(self.X), self.name)
		return (sql)
	#
	def deploy_to_DB(self, name: str, view: bool = True, cutoff = -1, all_classes: bool = False):
		relation = "TABLE" if not(view) else "VIEW"
		sql = "CREATE {} {} AS SELECT {}, {} FROM {}".format(relation, name, ", ".join(self.X), "{}", self.test_relation)
		if (all_classes):
			predict = []
			for elem in self.classes:
				if elem not in (self.classes):
					raise ValueError("All the elements of 'pos_label' must be in the estimator classes")
				alias = '"{}_{}"'.format(self.y.replace('"', ''), elem) 
				predict += ["{} AS {}".format(self.deploySQL(elem), alias)]
			predict += ["{} AS {}".format(self.deploySQL(), self.y)]
		else:
			if (len(self.classes) == 2):
				predict = ["{} AS {}".format(self.deploySQL(self.classes[1], cutoff), self.y)]
			else:
				predict = ["{} AS {}".format(self.deploySQL(), self.y)]
		self.cursor.execute(sql.format(", ".join(predict)))
	#
	def drop(self):
		drop_model(self.name, self.cursor, print_info = False)
	#
	def export_graphviz(self, tree_id: int = 0):
		query = "SELECT READ_TREE ( USING PARAMETERS model_name = '{}', tree_id = {}, format = 'graphviz');".format(self.name, tree_id)
		self.cursor.execute(query)
		return (self.cursor.fetchone()[1])
	#
	def features_importance(self):
		query  = "SELECT predictor_name AS predictor, ROUND(100 * importance_value / SUM(importance_value) OVER (), 2) AS importance, SIGN(importance_value) AS sign FROM (SELECT RF_PREDICTOR_IMPORTANCE ( USING PARAMETERS model_name = '{}')) x ORDER BY 2 DESC;".format(self.name)
		self.cursor.execute(query)
		result = self.cursor.fetchall()
		coeff_importances, coeff_sign = {}, {}
		for elem in result:
			coeff_importances[elem[0]] = elem[1]
			coeff_sign[elem[0]] = elem[2]
		try:
			plot_importance(coeff_importances, coeff_sign, print_legend = False)
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
		self.X = [str_column(column) for column in X]
		self.y = str_column(y)
		if (self.max_features == "auto"):
			self.max_features = int(len(self.X) / 3 + 1)
		elif (self.max_features == "max"):
			self.max_features = len(self.X)
		query = "SELECT RF_CLASSIFIER('{}', '{}', '{}', '{}' USING PARAMETERS ntree = {}, mtry = {}, sampling_size = {}"
		query = query.format(self.name, input_relation, self.y, ", ".join(self.X), self.n_estimators, self.max_features, self.sample)
		query += ", max_depth = {}, max_breadth = {}, min_leaf_size = {}, min_info_gain = {}, nbins = {})".format(self.max_depth, int(self.max_leaf_nodes), self.min_samples_leaf, self.min_info_gain, self.nbins)
		self.cursor.execute(query)
		self.cursor.execute("SELECT DISTINCT {} FROM {} WHERE {} IS NOT NULL ORDER BY 1".format(self.y, input_relation, self.y))
		classes = self.cursor.fetchall()
		self.classes = [item[0] for item in classes]
		return (self)
	#
	def get_tree(self, tree_id: int = 0):
		query = "SELECT READ_TREE ( USING PARAMETERS model_name = '{}', tree_id = {}, format = 'tabular');".format(self.name, tree_id)
		result = to_tablesample(query = query, cursor = self.cursor)
		result.table_info = False
		return (result)
	#
	def lift_chart(self, pos_label = None):
		pos_label = self.classes[1] if (pos_label == None and len(self.classes) == 2) else pos_label
		if (pos_label not in self.classes):
			raise ValueError("'pos_label' must be one of the response column classes")
		return (lift_chart(self.y, self.deploySQL(allSQL = True)[0].format(pos_label), self.test_relation, self.cursor, pos_label))
	#
	def plot_tree(self, tree_id: int = 0, pic_path: str = ""):
		plot_tree(self.get_tree(tree_id = tree_id).values, metric = "probability", pic_path = pic_path)
	#
	def prc_curve(self, pos_label = None):
		pos_label = self.classes[1] if (pos_label == None and len(self.classes) == 2) else pos_label
		if (pos_label not in self.classes):
			raise ValueError("'pos_label' must be one of the response column classes")
		return (prc_curve(self.y, self.deploySQL(allSQL = True)[0].format(pos_label), self.test_relation, self.cursor, pos_label))
	#
	def roc_curve(self, pos_label = None):
		pos_label = self.classes[1] if (pos_label == None and len(self.classes) == 2) else pos_label
		if (pos_label not in self.classes):
			raise ValueError("'pos_label' must be one of the response column classes")
		return (roc_curve(self.y, self.deploySQL(allSQL = True)[0].format(pos_label), self.test_relation, self.cursor, pos_label))
	#
	def score(self, pos_label = None, cutoff: float = 0.5, method: str = "accuracy"):
		pos_label = self.classes[1] if (pos_label == None and len(self.classes) == 2) else pos_label
		if (pos_label not in self.classes):
			raise ValueError("'pos_label' must be one of the response column classes")
		elif (cutoff >= 1 or cutoff <= 0):
			raise ValueError("'cutoff' must be in ]0;1[")
		if (method in ("accuracy", "acc")):
			return (accuracy_score(self.y, self.deploySQL(pos_label, cutoff), self.test_relation, self.cursor))
		elif (method == "auc"):
			return auc("DECODE({}, '{}', 1, 0)".format(self.y, pos_label), self.deploySQL(allSQL = True)[0].format(pos_label), self.test_relation, self.cursor)
		elif (method == "prc_auc"):
			return prc_auc("DECODE({}, '{}', 1, 0)".format(self.y, pos_label), self.deploySQL(allSQL = True)[0].format(pos_label), self.test_relation, self.cursor)
		elif (method in ("best_cutoff", "best_threshold")):
			return (roc_curve("DECODE({}, '{}', 1, 0)".format(self.y, pos_label), self.deploySQL(allSQL = True)[0].format(pos_label), self.test_relation, self.cursor, best_threshold = True))
		elif (method in ("recall", "tpr")):
			return (recall_score(self.y, self.deploySQL(pos_label, cutoff), self.test_relation, self.cursor))
		elif (method in ("precision", "ppv")):
			return (precision_score(self.y, self.deploySQL(pos_label, cutoff), self.test_relation, self.cursor))
		elif (method in ("specificity", "tnr")):
			return (specificity_score(self.y, self.deploySQL(pos_label, cutoff), self.test_relation, self.cursor))
		elif (method in ("negative_predictive_value", "npv")):
			return (precision_score(self.y, self.deploySQL(pos_label, cutoff), self.test_relation, self.cursor))
		elif (method in ("log_loss", "logloss")):
			return (log_loss("DECODE({}, '{}', 1, 0)".format(self.y, pos_label), self.deploySQL(allSQL = True)[0].format(pos_label), self.test_relation, self.cursor))
		elif (method == "f1"):
			return (f1_score(self.y, self.deploySQL(pos_label, cutoff), self.test_relation, self.cursor))
		elif (method == "mcc"):
			return (matthews_corrcoef(self.y, self.deploySQL(pos_label, cutoff), self.test_relation, self.cursor))
		elif (method in ("bm", "informedness")):
			return (informedness(self.y, self.deploySQL(pos_label, cutoff), self.test_relation, self.cursor))
		elif (method in ("mk", "markedness")):
			return (markedness(self.y, self.deploySQL(pos_label, cutoff), self.test_relation, self.cursor))
		elif (method in ("csi", "critical_success_index")):
			return (critical_success_index(self.y, self.deploySQL(pos_label, cutoff), self.test_relation, self.cursor))
		else:
			raise ValueError("The parameter 'method' must be in accuracy|auc|prc_auc|best_cutoff|recall|precision|log_loss|negative_predictive_value|specificity|mcc|informedness|markedness|critical_success_index")

#
class RandomForestRegressor:
	#
	def  __init__(self,
				  name: str,
				  cursor,
				  n_estimators: int = 10,
				  max_features = "auto",
				  max_leaf_nodes: int = 1e9, 
				  sample: float = 0.632, 
				  max_depth: int = 5,
				  min_samples_leaf: int = 1,
				  min_info_gain: float = 0.0,
				  nbins: int = 32):
		self.type = "regressor"
		self.cursor = cursor
		self.name = name
		self.n_estimators = n_estimators
		self.max_features = max_features
		self.max_leaf_nodes = max_leaf_nodes
		self.sample = sample 
		self.max_depth = max_depth
		self.min_samples_leaf = min_samples_leaf
		self.min_info_gain = min_info_gain
		self.nbins = nbins
	# 
	def __repr__(self):
		try:
			self.cursor.execute("SELECT GET_MODEL_SUMMARY(USING PARAMETERS model_name = '" + self.name + "')")
			return (self.cursor.fetchone()[0])
		except:
			return "<RandomForestRegressor>"
	#
	#
	#
	# METHODS
	# 
	#
	def add_to_vdf(self,
				   vdf,
				   name: str = ""):
		name = "RandomForestRegressor_" + self.name if not (name) else name
		return (vdf.eval(name, self.deploySQL()))
	#
	def deploySQL(self):
		sql = "PREDICT_RF_REGRESSOR({} USING PARAMETERS model_name = '{}', match_by_pos = 'true')"
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
	def export_graphviz(self, tree_id: int = 0):
		query = "SELECT READ_TREE ( USING PARAMETERS model_name = '{}', tree_id = {}, format = 'graphviz');".format(self.name, tree_id)
		self.cursor.execute(query)
		return (self.cursor.fetchone()[1])
	#
	def features_importance(self):
		query  = "SELECT predictor_name AS predictor, ROUND(100 * importance_value / SUM(importance_value) OVER (), 2) AS importance, SIGN(importance_value) AS sign FROM (SELECT RF_PREDICTOR_IMPORTANCE ( USING PARAMETERS model_name = '{}')) x ORDER BY 2 DESC;".format(self.name)
		self.cursor.execute(query)
		result = self.cursor.fetchall()
		coeff_importances, coeff_sign = {}, {}
		for elem in result:
			coeff_importances[elem[0]] = elem[1]
			coeff_sign[elem[0]] = elem[2]
		try:
			plot_importance(coeff_importances, coeff_sign, print_legend = False)
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
		self.X = [str_column(column) for column in X]
		self.y = str_column(y)
		if (self.max_features == "auto"):
			self.max_features = int(len(self.X) / 3 + 1)
		elif (self.max_features == "max"):
			self.max_features = len(self.X)
		query = "SELECT RF_REGRESSOR('{}', '{}', '{}', '{}' USING PARAMETERS ntree = {}, mtry = {}, sampling_size = {}"
		query = query.format(self.name, input_relation, self.y, ", ".join(self.X), self.n_estimators, self.max_features, self.sample)
		query += ", max_depth = {}, max_breadth = {}, min_leaf_size = {}, min_info_gain = {}, nbins = {})".format(self.max_depth, int(self.max_leaf_nodes), self.min_samples_leaf, self.min_info_gain, self.nbins)
		self.cursor.execute(query)
		return (self)
	#
	def get_tree(self, tree_id: int = 0):
		query = "SELECT READ_TREE ( USING PARAMETERS model_name = '{}', tree_id = {}, format = 'tabular');".format(self.name, tree_id)
		result = to_tablesample(query = query, cursor = self.cursor)
		result.table_info = False
		return (result)
	#
	def plot_tree(self, tree_id: int = 0, pic_path: str = ""):
		plot_tree(self.get_tree(tree_id = tree_id).values, metric = "variance", pic_path = pic_path)
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