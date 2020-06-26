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
# |_     |~) _  _| _  /~\    _ |.
# |_)\/  |_)(_|(_||   \_/|_|(_|||
#    /                           
#              ____________       ______
#             / __        `\     /     /
#            |  \/         /    /     /
#            |______      /    /     /
#                   |____/    /     /
#          _____________     /     /
#          \           /    /     /
#           \         /    /     /
#            \_______/    /     /
#             ______     /     /
#             \    /    /     /
#              \  /    /     /
#               \/    /     /
#                    /     /
#                   /     /
#                   \    /
#                    \  /
#                     \/
#                    _
# \  / _  __|_. _ _ |_)
#  \/ (/_|  | |(_(_|| \/
#                     /  
# VerticaPy allows user to create vDataFrames (Virtual Dataframes). 
# vDataFrames simplify data exploration, data cleaning and MACHINE LEARNING     
# in VERTICA. It is an object which keeps in it all the actions that the user 
# wants to achieve and execute them when they are needed.    										
#																					
# The purpose is to bring the logic to the data and not the opposite !
#
# 
# Modules
#
# VerticaPy Modules
from verticapy.learn.metrics import *
from verticapy.learn.plot import *
from verticapy.utilities import *
from verticapy.toolbox import *
from verticapy import vDataFrame
from verticapy.learn.plot import lof_plot
from verticapy.connections.connect import read_auto_connect
#---#
class NearestCentroid:
	"""
---------------------------------------------------------------------------
[Beta Version]
Creates a NearestCentroid object by using the K Nearest Centroid Algorithm. 
This object is using pure SQL to compute all the distances and final score. 
As NearestCentroid is using the p-distance, it is highly sensible to 
un-normalized data.  

Parameters
----------
cursor: DBcursor, optional
	Vertica DB cursor. 
p: int, optional
	The p corresponding to the one of the p-distance (distance metric used 
	during the model computation).

Attributes
----------
After the object creation, all the parameters become attributes. 
The model will also create extra attributes when fitting the model:

centroids: tablesample
	The final centroids.
classes: list
	List of all the response classes.
input_relation: str
	Train relation.
X: list
	List of the predictors.
y: str
	Response column.
test_relation: str
	Relation used to test the model. All the model methods are abstractions
	which will simplify the process. The test relation will be used by many
	methods to evaluate the model. If empty, the train relation will be 
	used as test. You can change it anytime by changing the test_relation
	attribute of the object.
	"""
	#
	# Special Methods
	#
	#---#
	def  __init__(self,
				  cursor = None,
				  p: int = 2):
		check_types([("p", p, [int, float], False)])
		if not(cursor):
			cursor = read_auto_connect().cursor()
		else:
			check_cursor(cursor)
		self.type = "classifier"
		self.cursor = cursor
		self.p = p
	#---#
	def __repr__(self):
		try:
			rep = "<NearestCentroid>\n\ncentroids:\n" + self.centroids.__repr__() + "\n\nclasses: {}".format(self.classes)
			return (rep)
		except:
			return "<NearestCentroid>"
	#
	# Methods
	#
	#---# 
	def classification_report(self, 
							  cutoff = [], 
							  labels: list = []):
		"""
	---------------------------------------------------------------------------
	Computes a classification report using multiple metrics to evaluate the model
	(AUC, accuracy, PRC AUC, F1...). In case of multiclass classification, it will 
	consider each category as positive and switch to the next one during the computation.

	Parameters
	----------
	cutoff: float/list, optional
		Cutoff for which the tested category will be accepted as prediction. 
		In case of multiclass classification, each tested category becomes 
		the positives and the others are merged into the negatives. The list will 
		represent the classes threshold. If it is empty, the best cutoff will be used.
	labels: list, optional
		List of the different labels to be used during the computation.

	Returns
	-------
	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.
		"""
		check_types([
			("cutoff", cutoff, [int, float, list], False),
			("labels", labels, [list], False)])
		if not(labels): labels = self.classes
		return (classification_report(cutoff = cutoff, estimator = self, labels = labels))
	#---#
	def confusion_matrix(self, 
						 pos_label = None, 
						 cutoff: float = -1):
		"""
	---------------------------------------------------------------------------
	Computes the model confusion matrix.

	Parameters
	----------
	pos_label: int/float/str, optional
		Label to consider as positive. All the other classes will be merged and
		considered as negative in case of multi classification.
	cutoff: float, optional
		Cutoff for which the tested category will be accepted as prediction. If the 
		cutoff is not between 0 and 1, the entire confusion matrix will be drawn.

	Returns
	-------
	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.
		"""
		check_types([("cutoff", cutoff, [int, float], False)])
		pos_label = self.classes[1] if (pos_label == None and len(self.classes) == 2) else pos_label
		if (pos_label in self.classes and cutoff <= 1 and cutoff >= 0):
			input_relation = self.deploySQL() + " WHERE predict_nc = '{}'".format(pos_label)
			y_score = "(CASE WHEN proba_predict > {} THEN 1 ELSE 0 END)".format(cutoff)
			y_true = "DECODE({}, '{}', 1, 0)".format(self.y, pos_label)
			result = confusion_matrix(y_true, y_score, input_relation, self.cursor)
			if pos_label == 1:
				return (result)
			else:
				return(tablesample(values = {"index": ["Non-{}".format(pos_label), "{}".format(pos_label)], "Non-{}".format(pos_label): result.values[0],  "{}".format(pos_label): result.values[1]}, table_info = False))
		else:
			return (multilabel_confusion_matrix(self.y, "predict_nc", self.deploySQL(predict = True), self.classes, self.cursor))
	#---#
	def deploySQL(self, 
				  predict: bool = False):
		"""
	---------------------------------------------------------------------------
	Returns the SQL code needed to deploy the model. 

	Parameters
	----------
	predict: bool, optional
		If set to True, returns the prediction instead of the probability.

	Returns
	-------
	str/list
 		the SQL code needed to deploy the model.
		"""
		check_types([("predict", predict, [bool], False)])
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
	#---#
	def fit(self,
			input_relation: str, 
			X: list, 
			y: str,
			test_relation: str = ""):
		"""
	---------------------------------------------------------------------------
	Trains the model.

	Parameters
	----------
	input_relation: str
		Train relation.
	X: list
		List of the predictors.
	y: str
		Response column.
	test_relation: str, optional
		Relation used to test the model.

	Returns
	-------
	object
 		self
		"""
		check_types([
			("input_relation", input_relation, [str], False),
			("X", X, [list], False),
			("y", y, [str], False),
			("test_relation", test_relation, [str], False)])
		func = "APPROXIMATE_MEDIAN" if (self.p == 1) else "AVG"
		self.input_relation = input_relation
		self.test_relation = test_relation if (test_relation) else input_relation
		self.X = [str_column(column) for column in X]
		self.y = str_column(y)
		query = "SELECT {}, {} FROM {} WHERE {} IS NOT NULL GROUP BY {} ORDER BY {} ASC".format(", ".join(["{}({}) AS {}".format(func, column, column) for column in self.X]), self.y, input_relation, self.y, self.y, self.y)
		self.centroids = to_tablesample(query = query, cursor = self.cursor)
		self.centroids.table_info = False
		self.classes = self.centroids.values[y]
		return (self)
	#---#
	def lift_chart(self, 
				   pos_label = None):
		"""
	---------------------------------------------------------------------------
	Draws the model Lift Chart.

	Parameters
	----------
	pos_label: int/float/str
		To draw a lift chart, one of the response column class has to be the 
		positive one. The parameter 'pos_label' represents this class.

	Returns
	-------
	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.
		"""
		pos_label = self.classes[1] if (pos_label == None and len(self.classes) == 2) else pos_label
		if (pos_label not in self.classes):
			raise ValueError("'pos_label' must be one of the response column classes")
		input_relation = self.deploySQL() + " WHERE predict_nc = '{}'".format(pos_label)
		y_proba = "proba_predict"
		return (lift_chart(self.y, y_proba, input_relation, self.cursor, pos_label))
	#---#
	def prc_curve(self, 
				  pos_label = None):
		"""
	---------------------------------------------------------------------------
	Draws the model PRC curve.

	Parameters
	----------
	pos_label: int/float/str
		To draw the PRC curve, one of the response column class has to be the 
		positive one. The parameter 'pos_label' represents this class.

	Returns
	-------
	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.
		"""
		pos_label = self.classes[1] if (pos_label == None and len(self.classes) == 2) else pos_label
		if (pos_label not in self.classes):
			raise ValueError("'pos_label' must be one of the response column classes")
		input_relation = self.deploySQL() + " WHERE predict_nc = '{}'".format(pos_label)
		y_proba = "proba_predict"
		return (prc_curve(self.y, y_proba, input_relation, self.cursor, pos_label))
	#---#
	def roc_curve(self, 
				  pos_label = None):
		"""
	---------------------------------------------------------------------------
	Draws the model ROC curve.

	Parameters
	----------
	pos_label: int/float/str
		To draw the ROC curve, one of the response column class has to be the 
		positive one. The parameter 'pos_label' represents this class.

	Returns
	-------
	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.
		"""
		pos_label = self.classes[1] if (pos_label == None and len(self.classes) == 2) else pos_label
		if (pos_label not in self.classes):
			raise ValueError("'pos_label' must be one of the response column classes")
		input_relation = self.deploySQL() + " WHERE predict_nc = '{}'".format(pos_label)
		y_proba = "proba_predict"
		return (roc_curve(self.y, y_proba, input_relation, self.cursor, pos_label))
	#---#
	def score(self, 
			  pos_label = None, 
			  cutoff: float = -1, 
			  method: str = "accuracy"):
		"""
	---------------------------------------------------------------------------
	Computes the model score.

	Parameters
	----------
	pos_label: int/float/str, optional
		Label to consider as positive. All the other classes will be merged and
		considered as negative in case of multi classification.
	cutoff: float, optional
		Cutoff for which the tested category will be accepted as prediction. 
	method: str, optional
		The method used to compute the score.
			accuracy    : Accuracy
			auc         : Area Under the Curve (ROC)
			best_cutoff : Cutoff which optimised the ROC Curve prediction.
			bm          : Informedness = tpr + tnr - 1
			csi         : Critical Success Index = tp / (tp + fn + fp)
			f1          : F1 Score 
			logloss     : Log Loss
			mcc         : Matthews Correlation Coefficient 
			mk          : Markedness = ppv + npv - 1
			npv         : Negative Predictive Value = tn / (tn + fn)
			prc_auc     : Area Under the Curve (PRC)
			precision   : Precision = tp / (tp + fp)
			recall      : Recall = tp / (tp + fn)
			specificity : Specificity = tn / (tn + fp) 

	Returns
	-------
	float
 		score
		"""
		check_types([
			("cutoff", cutoff, [int, float], False),
			("method", method, [str], False)])
		pos_label = self.classes[1] if (pos_label == None and len(self.classes) == 2) else pos_label
		input_relation = "(SELECT * FROM {} WHERE predict_nc = '{}') final_centroids_relation".format(self.deploySQL(), pos_label)
		y_score = "(CASE WHEN proba_predict > {} THEN 1 ELSE 0 END)".format(cutoff)
		y_proba = "proba_predict"
		y_true = "DECODE({}, '{}', 1, 0)".format(self.y, pos_label)
		if (pos_label not in self.classes) and (method != "accuracy"):
			raise ValueError("'pos_label' must be one of the response column classes")
		elif (cutoff >= 1 or cutoff <= 0) and (method != "accuracy"):
			cutoff = self.score(pos_label, 0.5, "best_cutoff")
		if (method in ("accuracy", "acc")):
			if (pos_label not in self.classes):
				return (accuracy_score(self.y, "predict_nc", self.deploySQL(True), self.cursor, pos_label = None))
			else:
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
	#---#
	def to_vdf(self, 
			   cutoff: float = -1, 
			   all_classes: bool = False):
		"""
	---------------------------------------------------------------------------
	Returns the vDataFrame of the Prediction.

	Parameters
	----------
	cutoff: float, optional
		Cutoff used in case of binary classification. It is the probability to
		accept the category 1.
	all_classes: bool, optional
		If set to True, all the classes probabilities will be generated (one 
		column per category).

	Returns
	-------
	vDataFrame
 		the vDataFrame of the prediction
		"""
		check_types([
			("cutoff", cutoff, [int, float], False),
			("all_classes", all_classes, [bool], False)])
		if (all_classes):
			predict = ["ZEROIFNULL(AVG(DECODE(predict_nc, '{}', proba_predict, NULL))) AS \"{}_{}\"".format(elem, self.y.replace('"', ''), elem) for elem in self.classes]
			sql = "SELECT {}, {} FROM {} GROUP BY {}".format(", ".join(self.X), ", ".join(predict), self.deploySQL(), ", ".join(self.X))
		else:
			if ((len(self.classes) == 2) and (cutoff <= 1 and cutoff >= 0)):
				sql = "SELECT {}, (CASE WHEN proba_predict > {} THEN '{}' ELSE '{}' END) AS {} FROM {} WHERE predict_nc = '{}'".format(", ".join(self.X), cutoff, self.classes[1], self.classes[0], self.y, self.deploySQL(), self.classes[1])
			elif (len(self.classes) == 2):
				sql = "SELECT {}, proba_predict AS {} FROM {} WHERE predict_nc = '{}'".format(", ".join(self.X), self.y, self.deploySQL(), self.classes[1])
			else:
				sql = "SELECT {}, predict_nc AS {} FROM {}".format(", ".join(self.X), self.y, self.deploySQL(True))
		sql = "({}) x".format(sql)
		return vdf_from_relation(name = "NearestCentroid", relation = sql, cursor = self.cursor)
#---#
class KNeighborsClassifier:
	"""
---------------------------------------------------------------------------
[Beta Version]
Creates a KNeighborsClassifier object by using the K Nearest Neighbors Algorithm. 
This object is using pure SQL to compute all the distances and final score. 
It is using CROSS JOIN and may be really expensive in some cases. As 
KNeighborsClassifier is using the p-distance, it is highly sensible to 
un-normalized data. 

Parameters
----------
cursor: DBcursor, optional
	Vertica DB cursor. 
n_neighbors: int, optional
	Number of neighbors to consider when computing the score.
p: int, optional
	The p corresponding to the one of the p-distance (distance metric used during 
	the model computation).

Attributes
----------
After the object creation, all the parameters become attributes. 
The model will also create extra attributes when fitting the model:

classes: list
	List of all the response classes.
input_relation: str
	Train relation.
X: list
	List of the predictors.
y: str
	Response column.
test_relation: str
	Relation used to test the model. All the model methods are abstractions
	which will simplify the process. The test relation will be used by many
	methods to evaluate the model. If empty, the train relation will be 
	used as test. You can change it anytime by changing the test_relation
	attribute of the object.
	"""
	#
	# Special Methods
	#
	#---#
	def  __init__(self,
				  cursor = None,
				  n_neighbors: int = 5,
				  p: int = 2):
		check_types([
			("n_neighbors", n_neighbors, [int, float], False),
			("p", p, [int, float], False)])
		if not(cursor):
			cursor = read_auto_connect().cursor()
		else:
			check_cursor(cursor)
		self.type = "classifier"
		self.cursor = cursor
		self.n_neighbors = n_neighbors
		self.p = p
	# 
	def __repr__(self):
		return "<KNeighborsClassifier>"
	#
	# Methods
	#
	#---# 
	def classification_report(self, 
							  cutoff = [], 
							  labels: list = []):
		"""
	---------------------------------------------------------------------------
	Computes a classification report using multiple metrics to evaluate the model
	(AUC, accuracy, PRC AUC, F1...). In case of multiclass classification, it will 
	consider each category as positive and switch to the next one during the computation.

	Parameters
	----------
	cutoff: float/list, optional
		Cutoff for which the tested category will be accepted as prediction. 
		In case of multiclass classification, each tested category becomes 
		the positives and the others are merged into the negatives. The list will 
		represent the classes threshold. If it is empty, the best cutoff will be used.
	labels: list, optional
		List of the different labels to be used during the computation.

	Returns
	-------
	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.
		"""
		check_types([
			("cutoff", cutoff, [int, float, list], False),
			("labels", labels, [list], False)])
		if not(labels): labels = self.classes
		return (classification_report(cutoff = cutoff, estimator = self, labels = labels))
	#---#
	def confusion_matrix(self, 
						 pos_label = None, 
						 cutoff: float = -1):
		"""
	---------------------------------------------------------------------------
	Computes the model confusion matrix.

	Parameters
	----------
	pos_label: int/float/str, optional
		Label to consider as positive. All the other classes will be merged and
		considered as negative in case of multi classification.
	cutoff: float, optional
		Cutoff for which the tested category will be accepted as prediction. If the 
		cutoff is not between 0 and 1, the entire confusion matrix will be drawn.

	Returns
	-------
	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.
		"""
		check_types([("cutoff", cutoff, [int, float], False)])
		pos_label = self.classes[1] if (pos_label == None and len(self.classes) == 2) else pos_label
		if (pos_label in self.classes and cutoff <= 1 and cutoff >= 0):
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
			return (multilabel_confusion_matrix(self.y, "predict_knc", input_relation, self.classes, self.cursor))
	#---#
	def deploySQL(self, 
				  predict: bool = False):
		"""
	---------------------------------------------------------------------------
	Returns the SQL code needed to deploy the model. 

	Parameters
	----------
	predict: bool, optional
		If set to true, returns the prediction instead of the probability.

	Returns
	-------
	str/list
 		the SQL code needed to deploy the model.
		"""
		check_types([("predict", predict, [bool], False)])
		sql = ["POWER(ABS(x.{} - y.{}), {})".format(self.X[i], self.X[i], self.p) for i in range(len(self.X))] 
		sql = "POWER({}, 1 / {})".format(" + ".join(sql), self.p)
		sql = "ROW_NUMBER() OVER(PARTITION BY {}, row_id ORDER BY {})".format(", ".join(['x.{}'.format(item) for item in self.X]), sql)
		where = " AND ".join(["{} IS NOT NULL".format(item) for item in self.X])
		sql = "SELECT {}, {} AS ordered_distance, x.{}, y.{} AS predict_knc, row_id FROM (SELECT *, ROW_NUMBER() OVER() AS row_id FROM {} WHERE {}) x CROSS JOIN (SELECT * FROM {} WHERE {}) y".format(", ".join(['x.{}'.format(item) for item in self.X]), sql, self.y, self.y, self.test_relation, where, self.input_relation, where)
		sql = "(SELECT row_id, {}, {}, predict_knc, COUNT(*) / {} AS proba_predict FROM ({}) z WHERE ordered_distance <= {} GROUP BY {}, {}, row_id, predict_knc) knc_table".format(", ".join(self.X), self.y, self.n_neighbors, sql, self.n_neighbors, ", ".join(self.X), self.y)
		if (predict):
			sql = "(SELECT {}, {}, predict_knc FROM (SELECT {}, {}, predict_knc, ROW_NUMBER() OVER (PARTITION BY {} ORDER BY proba_predict DESC) AS order_prediction FROM {}) x WHERE order_prediction = 1) predict_knc_table".format(", ".join(self.X), self.y, ", ".join(self.X), self.y, ", ".join(self.X), sql)
		return sql
	#---#
	def fit(self,
			input_relation: str,
			X: list, 
			y: str,
			test_relation: str = ""):
		"""
	---------------------------------------------------------------------------
	Trains the model.

	Parameters
	----------
	input_relation: str
		Train relation.
	X: list
		List of the predictors.
	y: str
		Response column.
	test_relation: str, optional
		Relation used to test the model.

	Returns
	-------
	object
 		self
		"""
		check_types([
			("input_relation", input_relation, [str], False),
			("X", X, [list], False),
			("y", y, [str], False),
			("test_relation", test_relation, [str], False)])
		self.input_relation = input_relation
		self.test_relation = test_relation if (test_relation) else input_relation
		self.X = [str_column(column) for column in X]
		self.y = str_column(y)
		self.cursor.execute("SELECT DISTINCT {} FROM {} WHERE {} IS NOT NULL ORDER BY {} ASC".format(self.y, input_relation, self.y, self.y))
		classes = self.cursor.fetchall()
		self.classes = [item[0] for item in classes]
		return (self)
	#---#
	def lift_chart(self, 
				   pos_label = None):
		"""
	---------------------------------------------------------------------------
	Draws the model Lift Chart.

	Parameters
	----------
	pos_label: int/float/str
		To draw a lift chart, one of the response column class has to be the 
		positive one. The parameter 'pos_label' represents this class.

	Returns
	-------
	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.
		"""
		pos_label = self.classes[1] if (pos_label == None and len(self.classes) == 2) else pos_label
		if (pos_label not in self.classes):
			raise ValueError("'pos_label' must be one of the response column classes")
		input_relation = self.deploySQL() + " WHERE predict_knc = '{}'".format(pos_label)
		y_proba = "proba_predict"
		return (lift_chart(self.y, y_proba, input_relation, self.cursor, pos_label))
	#---#
	def prc_curve(self, 
				  pos_label = None):
		"""
	---------------------------------------------------------------------------
	Draws the model PRC curve.

	Parameters
	----------
	pos_label: int/float/str
		To draw the PRC curve, one of the response column class has to be the 
		positive one. The parameter 'pos_label' represents this class.

	Returns
	-------
	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.
		"""
		pos_label = self.classes[1] if (pos_label == None and len(self.classes) == 2) else pos_label
		if (pos_label not in self.classes):
			raise ValueError("'pos_label' must be one of the response column classes")
		input_relation = self.deploySQL() + " WHERE predict_knc = '{}'".format(pos_label)
		y_proba = "proba_predict"
		return (prc_curve(self.y, y_proba, input_relation, self.cursor, pos_label))
	#---#
	def roc_curve(self, 
				  pos_label = None):
		"""
	---------------------------------------------------------------------------
	Draws the model ROC curve.

	Parameters
	----------
	pos_label: int/float/str
		To draw the ROC curve, one of the response column class has to be the 
		positive one. The parameter 'pos_label' represents this class.

	Returns
	-------
	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.
		"""
		pos_label = self.classes[1] if (pos_label == None and len(self.classes) == 2) else pos_label
		if (pos_label not in self.classes):
			raise ValueError("'pos_label' must be one of the response column classes")
		input_relation = self.deploySQL() + " WHERE predict_knc = '{}'".format(pos_label)
		y_proba = "proba_predict"
		return (roc_curve(self.y, y_proba, input_relation, self.cursor, pos_label))
	#---#
	def score(self, 
			  pos_label = None, 
			  cutoff: float = -1, 
			  method: str = "accuracy"):
		"""
	---------------------------------------------------------------------------
	Computes the model score.

	Parameters
	----------
	pos_label: int/float/str, optional
		Label to consider as positive. All the other classes will be merged and
		considered as negative in case of multi classification.
	cutoff: float, optional
		Cutoff for which the tested category will be accepted as prediction. 
	method: str, optional
		The method used to compute the score.
			accuracy    : Accuracy
			auc         : Area Under the Curve (ROC)
			best_cutoff : Cutoff which optimised the ROC Curve prediction.
			bm          : Informedness = tpr + tnr - 1
			csi         : Critical Success Index = tp / (tp + fn + fp)
			f1          : F1 Score 
			logloss     : Log Loss
			mcc         : Matthews Correlation Coefficient 
			mk          : Markedness = ppv + npv - 1
			npv         : Negative Predictive Value = tn / (tn + fn)
			prc_auc     : Area Under the Curve (PRC)
			precision   : Precision = tp / (tp + fp)
			recall      : Recall = tp / (tp + fn)
			specificity : Specificity = tn / (tn + fp) 

	Returns
	-------
	float
 		score
		"""
		check_types([
			("cutoff", cutoff, [int, float], False),
			("method", method, [str], False)])
		pos_label = self.classes[1] if (pos_label == None and len(self.classes) == 2) else pos_label
		input_relation = "(SELECT * FROM {} WHERE predict_knc = '{}') final_knn_table".format(self.deploySQL(), pos_label)
		y_score = "(CASE WHEN proba_predict > {} THEN 1 ELSE 0 END)".format(cutoff)
		y_proba = "proba_predict"
		y_true = "DECODE({}, '{}', 1, 0)".format(self.y, pos_label)
		if (pos_label not in self.classes) and (method != "accuracy"):
			raise ValueError("'pos_label' must be one of the response column classes")
		elif (cutoff >= 1 or cutoff <= 0) and (method != "accuracy"):
			cutoff = self.score(pos_label, 0.5, "best_cutoff")
		if (method in ("accuracy", "acc")):
			if (pos_label not in self.classes):
				return (accuracy_score(self.y, "predict_knc", self.deploySQL(True), self.cursor, pos_label = None))
			else:
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
	#---#
	def to_vdf(self, 
			   cutoff: float = -1, 
			   all_classes: bool = False):
		"""
	---------------------------------------------------------------------------
	Returns the vDataFrame of the Prediction.

	Parameters
	----------
	cutoff: float, optional
		Cutoff used in case of binary classification. It is the probability to
		accept the category 1.
	all_classes: bool, optional
		If set to true, all the classes probabilities will be generated (one 
		column per category).

	Returns
	-------
	vDataFrame
 		the vDataFrame of the prediction
		"""
		check_types([
			("cutoff", cutoff, [int, float], False),
			("all_classes", all_classes, [bool], False)])
		if (all_classes):
			predict = ["ZEROIFNULL(AVG(DECODE(predict_knc, '{}', proba_predict, NULL))) AS \"{}_{}\"".format(elem, self.y.replace('"', ''), elem) for elem in self.classes]
			sql = "SELECT {}, {} FROM {} GROUP BY {}".format(", ".join(self.X), ", ".join(predict), self.deploySQL(), ", ".join(self.X))
		else:
			if ((len(self.classes) == 2) and (cutoff <= 1 and cutoff >= 0)):
				sql = "SELECT {}, (CASE WHEN proba_predict > {} THEN '{}' ELSE '{}' END) AS {} FROM {} WHERE predict_knc = '{}'".format(", ".join(self.X), cutoff, self.classes[1], self.classes[0], self.y, self.deploySQL(), self.classes[1])
			elif (len(self.classes) == 2):
				sql = "SELECT {}, proba_predict AS {} FROM {} WHERE predict_knc = '{}'".format(", ".join(self.X), self.y, self.deploySQL(), self.classes[1])
			else:
				sql = "SELECT {}, predict_knc AS {} FROM {}".format(", ".join(self.X), self.y, self.deploySQL(True))
		sql = "({}) x".format(sql)
		return vdf_from_relation(name = "KNN", relation = sql, cursor = self.cursor)
#---#
class KNeighborsRegressor:
	"""
---------------------------------------------------------------------------
[Beta Version]
Creates a KNeighborsRegressor object by using the K Nearest Neighbors Algorithm. 
This object is using pure SQL to compute all the distances and final score. 
It is using CROSS JOIN and may be really expensive in some cases. As 
KNeighborsRegressor is using the p-distance, it is highly sensible to 
un-normalized data. 

Parameters
----------
cursor: DBcursor, optional
	Vertica DB cursor. 
n_neighbors: int, optional
	Number of neighbors to consider when computing the score.
p: int, optional
	The p corresponding to the one of the p-distance (distance metric used during 
	the model computation).

Attributes
----------
After the object creation, all the parameters become attributes. 
The model will also create extra attributes when fitting the model:

input_relation: str
	Train relation.
X: list
	List of the predictors.
y: str
	Response column.
test_relation: str
	Relation used to test the model. All the model methods are abstractions
	which will simplify the process. The test relation will be used by many
	methods to evaluate the model. If empty, the train relation will be 
	used as test. You can change it anytime by changing the test_relation
	attribute of the object.
	"""
	#
	# Special Methods
	#
	#---#
	def  __init__(self,
				  cursor = None,
				  n_neighbors: int = 5,
				  p: int = 2):
		check_types([
			("n_neighbors", n_neighbors, [int, float], False),
			("p", p, [int, float], False)])
		if not(cursor):
			cursor = read_auto_connect().cursor()
		else:
			check_cursor(cursor)
		self.type = "regressor"
		self.cursor = cursor
		self.n_neighbors = n_neighbors
		self.p = p
	#---#
	def __repr__(self):
		return "<KNeighborsRegressor>"
	#
	# Methods
	#
	#---#
	def deploySQL(self):
		"""
	---------------------------------------------------------------------------
	Returns the SQL code needed to deploy the model. 

	Returns
	-------
	str
 		the SQL code needed to deploy the model.
		"""
		sql = ["POWER(ABS(x.{} - y.{}), {})".format(self.X[i], self.X[i], self.p) for i in range(len(self.X))] 
		sql = "POWER({}, 1 / {})".format(" + ".join(sql), self.p)
		sql = "ROW_NUMBER() OVER(PARTITION BY {}, row_id ORDER BY {})".format(", ".join(['x.{}'.format(item) for item in self.X]), sql)
		where = " AND ".join(["{} IS NOT NULL".format(item) for item in self.X])
		sql = "SELECT {}, {} AS ordered_distance, x.{}, y.{} AS predict_knr, row_id FROM (SELECT *, ROW_NUMBER() OVER() AS row_id FROM {} WHERE {}) x CROSS JOIN (SELECT * FROM {} WHERE {}) y".format(", ".join(['x.{}'.format(item) for item in self.X]), sql, self.y, self.y, self.test_relation, where, self.input_relation, where)
		sql = "(SELECT {}, {}, AVG(predict_knr) AS predict_knr FROM ({}) z WHERE ordered_distance <= {} GROUP BY {}, {}, row_id) knr_table".format(", ".join(self.X), self.y, sql, self.n_neighbors, ", ".join(self.X), self.y)
		return sql
	#---#
	def fit(self,
			input_relation: str, 
			X: list, 
			y: str,
			test_relation: str = ""):
		"""
	---------------------------------------------------------------------------
	Trains the model.

	Parameters
	----------
	input_relation: str
		Train relation.
	X: list
		List of the predictors.
	y: str
		Response column.
	test_relation: str, optional
		Relation used to test the model.

	Returns
	-------
	object
 		self
		"""
		check_types([
			("input_relation", input_relation, [str], False),
			("X", X, [list], False),
			("y", y, [str], False),
			("test_relation", test_relation, [str], False)])
		self.input_relation = input_relation
		self.test_relation = test_relation if (test_relation) else input_relation
		self.X = [str_column(column) for column in X]
		self.y = str_column(y)
		return (self)
	#---#
	def regression_report(self):
		"""
	---------------------------------------------------------------------------
	Computes a regression report using multiple metrics to evaluate the model
	(r2, mse, max error...). 

	Returns
	-------
	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.
		"""
		return (regression_report(self.y, "predict_knr", self.deploySQL(), self.cursor))
	#---#
	def score(self, 
			  method: str = "r2"):
		"""
	---------------------------------------------------------------------------
	Computes the model score.

	Parameters
	----------
	method: str, optional
		The method used to compute the score.
			max    : Max Error
			mae    : Mean Absolute Error
			median : Median Absolute Error
			mse    : Mean Squared Error
			msle   : Mean Squared Log Error
			r2     : R squared coefficient
			var    : Explained Variance 

	Returns
	-------
	float
 		score
		"""
		check_types([("method", method, [str], False)])
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
	#---#
	def to_vdf(self):
		"""
	---------------------------------------------------------------------------
	Returns the vDataFrame of the Prediction.

	Returns
	-------
	vDataFrame
 		the vDataFrame of the prediction
		"""
		sql = "(SELECT {}, {} AS {} FROM {}) x".format(", ".join(self.X), "predict_knr", self.y, self.deploySQL())
		return vdf_from_relation(name = "KNeighborsRegressor", relation = sql, cursor = self.cursor)
#---#
class LocalOutlierFactor:
	"""
---------------------------------------------------------------------------
[Beta Version]
Creates a LocalOutlierFactor object by using the Local Outlier Factor algorithm 
as defined by Markus M. Breunig, Hans-Peter Kriegel, Raymond T. Ng and Jörg 
Sander. This object is using pure SQL to compute all the distances and final 
score. It is using CROSS JOIN and may be really expensive in some cases. It 
will index all the elements of the table in order to be optimal (the CROSS 
JOIN will happen only with IDs which are integers). As LocalOutlierFactor 
is using the p-distance, it is highly sensible to un-normalized data. 

Parameters
----------
name: str
	Name of the the model. As it is not a built in model, this name will be used
	to build the final table.
cursor: DBcursor, optional
	Vertica DB cursor.
n_neighbors: int, optional
	Number of neighbors to consider when computing the score.
p: int, optional
	The p of the p-distance (distance metric used during the model computation).

Attributes
----------
After the object creation, all the parameters become attributes. 
The model will also create extra attributes when fitting the model:

n_errors: int
	Number of errors during the LOF computation.
input_relation: str
	Train relation.
X: list
	List of the predictors.
key_columns: list
	Columns not used during the algorithm computation but which will be used
	to create the final relation.
	"""
	#
	# Special Methods
	#
	#---#
	def  __init__(self, 
				  name: str, 
				  cursor = None, 
				  n_neighbors: int = 20, 
				  p: int = 2):
		check_types([
			("name", name, [str], False),
			("n_neighbors", n_neighbors, [int, float], False),
			("p", p, [int, float], False)])
		if not(cursor):
			cursor = read_auto_connect().cursor()
		else:
			check_cursor(cursor)
		self.type = "anomaly_detection"
		self.name = name
		self.cursor = cursor
		self.n_neighbors = n_neighbors
		self.p = p
	#---#
	def __repr__(self):
		return "<LocalOutlierFactor>"
	#
	# Methods
	#
	#---#
	def fit(self, 
			input_relation: str, 
			X: list, 
			key_columns: list = [], 
			index: str = ""):
		"""
	---------------------------------------------------------------------------
	Trains the model.

	Parameters
	----------
	input_relation: str
		Train relation.
	X: list
		List of the predictors.
	key_columns: list, optional
		Columns not used during the algorithm computation but which will be used
		to create the final relation.
	index: str, optional
		Index used to identify each row separately. It is highly recommanded to
		have one already in the main table to avoid creation of temporary tables.

	Returns
	-------
	object
 		self
		"""
		check_types([
			("input_relation", input_relation, [str], False),
			("X", X, [list], False),
			("key_columns", key_columns, [list], False),
			("index", index, [str], False)])
		X = [str_column(column) for column in X]
		self.X = X
		self.key_columns = [str_column(column) for column in key_columns]
		self.input_relation = input_relation
		cursor = self.cursor
		n_neighbors = self.n_neighbors
		p = self.p
		relation_alpha = ''.join(ch for ch in input_relation if ch.isalnum())
		schema, relation = schema_relation(input_relation)
		if not(index):
			index = "id"
			relation_alpha = ''.join(ch for ch in relation if ch.isalnum())
			main_table = "VERTICAPY_MAIN_{}".format(relation_alpha)
			schema = "v_temp_schema"
			try:
				cursor.execute("DROP TABLE IF EXISTS v_temp_schema.{}".format(main_table))
			except:
				pass
			sql = "CREATE LOCAL TEMPORARY TABLE {} ON COMMIT PRESERVE ROWS AS SELECT ROW_NUMBER() OVER() AS id, {} FROM {} WHERE {}".format(main_table, ", ".join(X + key_columns), input_relation, " AND ".join(["{} IS NOT NULL".format(item) for item in X]))
			cursor.execute(sql)
		else:
			main_table = input_relation
		sql = ["POWER(ABS(x.{} - y.{}), {})".format(X[i], X[i], p) for i in range(len(X))] 
		distance = "POWER({}, 1 / {})".format(" + ".join(sql), p)
		sql = "SELECT x.{} AS node_id, y.{} AS nn_id, {} AS distance, ROW_NUMBER() OVER(PARTITION BY x.{} ORDER BY {}) AS knn FROM {}.{} AS x CROSS JOIN {}.{} AS y".format(index, index, distance, index, distance, schema, main_table, schema, main_table)
		sql = "SELECT node_id, nn_id, distance, knn FROM ({}) distance_table WHERE knn <= {}".format(sql, n_neighbors + 1)
		try:
			cursor.execute("DROP TABLE IF EXISTS v_temp_schema.VERTICAPY_DISTANCE_{}".format(relation_alpha))
		except:
			pass
		sql = "CREATE LOCAL TEMPORARY TABLE VERTICAPY_DISTANCE_{} ON COMMIT PRESERVE ROWS AS {}".format(relation_alpha, sql)
		cursor.execute(sql)
		kdistance = "(SELECT node_id, nn_id, distance AS distance FROM v_temp_schema.VERTICAPY_DISTANCE_{} WHERE knn = {}) AS kdistance_table".format(relation_alpha, n_neighbors + 1)
		lrd = "SELECT distance_table.node_id, {} / SUM(CASE WHEN distance_table.distance > kdistance_table.distance THEN distance_table.distance ELSE kdistance_table.distance END) AS lrd FROM (v_temp_schema.VERTICAPY_DISTANCE_{} AS distance_table LEFT JOIN {} ON distance_table.nn_id = kdistance_table.node_id) x GROUP BY 1".format(n_neighbors, relation_alpha, kdistance)
		try:
			cursor.execute("DROP TABLE IF EXISTS v_temp_schema.VERTICAPY_LRD_{}".format(relation_alpha))
		except:
			pass
		sql = "CREATE LOCAL TEMPORARY TABLE VERTICAPY_LRD_{} ON COMMIT PRESERVE ROWS AS {}".format(relation_alpha, lrd)
		cursor.execute(sql)
		sql = "SELECT x.node_id, SUM(y.lrd) / (MAX(x.node_lrd) * {}) AS LOF FROM (SELECT n_table.node_id, n_table.nn_id, lrd_table.lrd AS node_lrd FROM v_temp_schema.VERTICAPY_DISTANCE_{} AS n_table LEFT JOIN v_temp_schema.VERTICAPY_LRD_{} AS lrd_table ON n_table.node_id = lrd_table.node_id) x LEFT JOIN v_temp_schema.VERTICAPY_LRD_{} AS y ON x.nn_id = y.node_id GROUP BY 1".format(n_neighbors, relation_alpha, relation_alpha, relation_alpha)
		try:
			cursor.execute("DROP TABLE IF EXISTS v_temp_schema.VERTICAPY_LOF_{}".format(relation_alpha))
		except:
			pass
		sql = "CREATE LOCAL TEMPORARY TABLE VERTICAPY_LOF_{} ON COMMIT PRESERVE ROWS AS {}".format(relation_alpha, sql)
		cursor.execute(sql)
		sql = "SELECT {}, (CASE WHEN lof > 1e100 OR lof != lof THEN 0 ELSE lof END) AS lof_score FROM {} AS x LEFT JOIN v_temp_schema.VERTICAPY_LOF_{} AS y ON x.{} = y.node_id".format(", ".join(X + self.key_columns), main_table, relation_alpha, index)
		cursor.execute("CREATE TABLE {} AS {}".format(self.name, sql))
		cursor.execute("SELECT COUNT(*) FROM {}.VERTICAPY_LOF_{} z WHERE lof > 1e100 OR lof != lof".format(schema, relation_alpha))
		self.n_errors = cursor.fetchone()[0]
		cursor.execute("DROP TABLE IF EXISTS v_temp_schema.VERTICAPY_MAIN_{}".format(relation_alpha))
		cursor.execute("DROP TABLE IF EXISTS v_temp_schema.VERTICAPY_DISTANCE_{}".format(relation_alpha))
		cursor.execute("DROP TABLE IF EXISTS v_temp_schema.VERTICAPY_LRD_{}".format(relation_alpha))
		cursor.execute("DROP TABLE IF EXISTS v_temp_schema.VERTICAPY_LOF_{}".format(relation_alpha))
		return (self)
	#---#
	def info(self):
		"""
	---------------------------------------------------------------------------
	Displays some information about the model.
		"""
		if (self.n_errors == 0):
			print("All the LOF scores were computed.")
		else:
			print("{} error(s) happened during the computation. These ones were imputed by 0 as it is highly probable that the {}-Neighbors of these points were confounded (usual problem of the LOF computation).\nIncrease the number of Neighbors to decrease the number of errors.".format(self.n_errors, self.n_neighbors))
	#---#
	def plot(self, 
			 X: list = [], 
			 tablesample: float = -1):
		"""
	---------------------------------------------------------------------------
	Draws the model is the number of predictors is 2 or 3.

	Parameters
	----------
	X: list, optional
		List of the predictors to display. The score will stay the same. 
	tablesample: float, optional
		Sample of data to display. If this number is not between 0 and 1 all
		the data points will be displayed.
		"""
		check_types([
			("X", X, [list], False),
			("tablesample", tablesample, [int, float], False)])
		X = self.X if not(X) else X
		lof_plot(self.name, X, "lof_score", self.cursor, tablesample)
	#---#
	def to_vdf(self):
		"""
	---------------------------------------------------------------------------
	Creates a vDataFrame of the model.

	Returns
	-------
	vDataFrame
 		model vDataFrame
		"""
		return (vDataFrame(self.name, self.cursor))