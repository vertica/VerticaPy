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
# Standard Python Modules
import math
# Other Python Modules
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
# VerticaPy Modules
from verticapy.utilities import *
from verticapy.toolbox import *
from verticapy.learn.cluster import KMeans
from verticapy.connections.connect import read_auto_connect
#---#
def elbow(X: list,
		  input_relation: str,
		  cursor = None,
		  n_cluster = (1, 15),
		  init = "kmeanspp",
		  max_iter: int = 50,
		  tol: float = 1e-4):
	"""
---------------------------------------------------------------------------
Draws the Elbow Curve.

Parameters
----------
X: list
	List of the predictor columns.
input_relation: str
	Relation used to train the model.
cursor: DBcursor, optional
	Vertica DB cursor.
n_cluster: int, optional
	Tuple representing the number of cluster to start with and to end with.
	It can also be customized list with the different K to test.
init: str/list, optional
	The method used to find the initial cluster centers.
		kmeanspp : Use the KMeans++ method to initialize the centers.
		random   : The initial centers
	It can be also a list with the initial cluster centers to use.
max_iter: int, optional
	The maximum number of iterations the algorithm performs.
tol: float, optional
	Determines whether the algorithm has converged. The algorithm is considered 
	converged after no center has moved more than a distance of 'tol' from the 
	previous iteration.

Returns
-------
tablesample
 	An object containing the result. For more information, check out
 	utilities.tablesample.
	"""
	check_types([
		("X", X, [list], False), 
		("input_relation", input_relation, [str], False), 
		("n_cluster", n_cluster, [list, tuple], False),
		("init", init, ["kmeanspp", "random"], True),
		("max_iter", max_iter, [int, float], False),
		("tol", tol, [int, float], False)])
	if not(cursor):
		conn = read_auto_connect()
		cursor = conn.cursor()
	else:
		conn = False
		check_cursor(cursor)
	schema, relation = schema_relation(input_relation)
	schema = str_column(schema)
	relation_alpha = ''.join(ch for ch in relation if ch.isalnum())
	all_within_cluster_SS = []
	if not(type(n_cluster) == list):
		L = [i for i in range(n_cluster[0], n_cluster[1])] 
	else:
		L = n_cluster
		L.sort()
	for i in L:
		cursor.execute("DROP MODEL IF EXISTS {}.VERTICAPY_KMEANS_TMP_{}".format(schema, relation_alpha))
		model = KMeans("{}.VERTICAPY_KMEANS_TMP_{}".format(schema, relation_alpha), cursor, i, init, max_iter, tol)
		model.fit(input_relation, X)
		all_within_cluster_SS += [float(model.metrics.values["value"][3])]
		model.drop()
	if (conn):
		conn.close()
	plt.figure(figsize = (10,8))
	plt.rcParams['axes.facecolor'] = '#F4F4F4'
	plt.grid()
	plt.plot(L, all_within_cluster_SS, marker = "s", color = "#FE5016")
	plt.title("Elbow Curve")
	plt.xlabel('Number of Clusters')
	plt.ylabel('Between-Cluster SS / Total SS')
	plt.subplots_adjust(left = 0.2)
	plt.show()
	values = {"index": L, "Within-Cluster SS": all_within_cluster_SS}
	return tablesample(values = values, table_info = False)
#---#
def lift_chart(y_true: str, 
			   y_score: str, 
			   input_relation: str,
			   cursor = None,
			   pos_label = 1, 
			   nbins: int = 1000):
	"""
---------------------------------------------------------------------------
Draws the Lift Chart.

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction Probability.
input_relation: str
	Relation used to do the scoring. The relation can be a view or a table
	or even a customized relation. For example, you could write:
	"(SELECT ... FROM ...) x" as long as an alias is given at the end of the
	relation.
cursor: DBcursor, optional
	Vertica DB cursor.
pos_label: int/float/str, optional
	To compute the Lift Chart, one of the response column class has to be the 
	positive one. The parameter 'pos_label' represents this class.
nbins: int, optional
	Curve number of bins.

Returns
-------
tablesample
 	An object containing the result. For more information, check out
 	utilities.tablesample.
	"""
	check_types([
		("y_true", y_true, [str], False), 
		("y_score", y_score, [str], False), 
		("input_relation", input_relation, [str], False),
		("nbins", nbins, [int, float], False)])
	if not(cursor):
		conn = read_auto_connect()
		cursor = conn.cursor()
	else:
		conn = False
		check_cursor(cursor)
	query = "SELECT LIFT_TABLE(obs, prob USING PARAMETERS num_bins = {}) OVER() FROM (SELECT (CASE WHEN {} = '{}' THEN 1 ELSE 0 END) AS obs, {}::float AS prob FROM {}) AS prediction_output"
	query = query.format(nbins, y_true, pos_label, y_score, input_relation)
	cursor.execute(query)
	query_result = cursor.fetchall()
	if (conn):
		conn.close()
	decision_boundary, positive_prediction_ratio, lift = [item[0] for item in query_result], [item[1] for item in query_result], [item[2] for item in query_result]
	decision_boundary.reverse()
	plt.figure(figsize = (10,8))
	plt.rcParams['axes.facecolor'] = '#F5F5F5'
	plt.xlabel('Cumulative Data Fraction')
	plt.plot(decision_boundary, lift, color = "#FE5016")
	plt.plot(decision_boundary, positive_prediction_ratio, color = "#444444")
	plt.title("Lift Table")
	plt.gca().set_axisbelow(True)
	plt.grid()
	color1 = mpatches.Patch(color = "#FE5016", label = 'Cumulative Lift')
	color2 = mpatches.Patch(color = "#444444", label = 'Cumulative Capture Rate')
	plt.legend(handles = [color1, color2])
	plt.show()
	return (tablesample(values = {"decision_boundary": decision_boundary, "positive_prediction_ratio": positive_prediction_ratio, "lift": lift}, table_info = False))
#---#
def prc_curve(y_true: str, 
			  y_score: str, 
			  input_relation: str,
			  cursor = None,
			  pos_label = 1, 
			  nbins: int = 1000,
			  auc_prc: bool = False):
	"""
---------------------------------------------------------------------------
Draws the PRC Curve.

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction Probability.
input_relation: str
	Relation used to do the scoring. The relation can be a view or a table
	or even a customized relation. For example, you could write:
	"(SELECT ... FROM ...) x" as long as an alias is given at the end of the
	relation.
cursor: DBcursor, optional
	Vertica DB cursor.
pos_label: int/float/str, optional
	To compute the PRC Curve, one of the response column class has to be the 
	positive one. The parameter 'pos_label' represents this class.
nbins: int, optional
	Curve number of bins.
auc_prc: bool, optional
	If set to True, the function will return the PRC AUC without drawing the 
	curve.

Returns
-------
tablesample
 	An object containing the result. For more information, check out
 	utilities.tablesample.
	"""
	check_types([
		("y_true", y_true, [str], False),
		("y_score", y_score, [str], False),
		("input_relation", input_relation, [str], False),
		("nbins", nbins, [int, float], False),
		("auc_prc", auc_prc, [bool], False)])
	if not(cursor):
		conn = read_auto_connect()
		cursor = conn.cursor()
	else:
		conn = False
		check_cursor(cursor)
	query = "SELECT PRC(obs, prob USING PARAMETERS num_bins = {}) OVER() FROM (SELECT (CASE WHEN {} = '{}' THEN 1 ELSE 0 END) AS obs, {}::float AS prob FROM {}) AS prediction_output"
	query = query.format(nbins, y_true, pos_label, y_score, input_relation)
	cursor.execute(query)
	query_result = cursor.fetchall()
	if (conn):
		conn.close()
	threshold, recall, precision = [0] + [item[0] for item in query_result] + [1], [1] + [item[1] for item in query_result] + [0], [0] +  [item[2] for item in query_result] + [1]
	auc=0
	for i in range(len(recall) - 1):
		if (recall[i + 1] - recall[i] != 0.0):
			a = (precision[i + 1] - precision[i]) / (recall[i + 1] - recall[i])
			b = precision[i + 1] - a * recall[i + 1]
			auc = auc + a * (recall[i + 1] * recall[i + 1] - recall[i] * recall[i]) / 2 + b * (recall[i + 1] - recall[i]);
	auc = - auc
	if (auc_prc):
		return (auc)
	plt.figure(figsize = (10,8))
	plt.rcParams['axes.facecolor'] = '#F5F5F5'
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.plot(recall, precision, color = "#FE5016")
	plt.ylim(0,1)
	plt.xlim(0,1)
	plt.title("PRC Curve\nAUC = " + str(auc))
	plt.gca().set_axisbelow(True)
	plt.grid()
	plt.show()
	return (tablesample(values = {"threshold": threshold, "recall": recall, "precision": precision}, table_info = False))
#---#
def roc_curve(y_true: str, 
			  y_score: str, 
			  input_relation: str,
			  cursor = None,
			  pos_label = 1, 
			  nbins: int = 1000,
			  auc_roc: bool = False,
			  best_threshold: bool = False):
	"""
---------------------------------------------------------------------------
Draws the ROC Curve.

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction Probability.
input_relation: str
	Relation used to do the scoring. The relation can be a view or a table
	or even a customized relation. For example, you could write:
	"(SELECT ... FROM ...) x" as long as an alias is given at the end of the
	relation.
cursor: DBcursor, optional
	Vertica DB cursor.
pos_label: int/float/str, optional
	To compute the PRC Curve, one of the response column class has to be the 
	positive one. The parameter 'pos_label' represents this class.
nbins: int, optional
	Curve number of bins.
auc_roc: bool, optional
	If set to true, the function will return the ROC AUC without drawing the 
	curve.
best_threshold: bool, optional
	If set to True, the function will return the best threshold without drawing 
	the curve. The best threshold is the threshold of the point which is the 
	farest from the random line.

Returns
-------
tablesample
 	An object containing the result. For more information, check out
 	utilities.tablesample.
	"""
	check_types([
		("y_true", y_true, [str], False),
		("y_score", y_score, [str], False),
		("input_relation", input_relation, [str], False),
		("nbins", nbins, [int, float], False),
		("auc_roc", auc_roc, [bool], False),
		("best_threshold", best_threshold, [bool], False)])
	if not(cursor):
		conn = read_auto_connect()
		cursor = conn.cursor()
	else:
		conn = False
		check_cursor(cursor)
	query = "SELECT ROC(obs, prob USING PARAMETERS num_bins = {}) OVER() FROM (SELECT (CASE WHEN {} = '{}' THEN 1 ELSE 0 END) AS obs, {}::float AS prob FROM {}) AS prediction_output"
	query = query.format(nbins, y_true, pos_label, y_score, input_relation)
	cursor.execute(query)
	query_result = cursor.fetchall()
	if (conn):
		conn.close()
	threshold, false_positive, true_positive = [item[0] for item in query_result], [item[1] for item in query_result], [item[2] for item in query_result]
	auc=0
	for i in range(len(false_positive) - 1):
		if (false_positive[i + 1] - false_positive[i] != 0.0):
			a = (true_positive[i + 1] - true_positive[i]) / (false_positive[i + 1] - false_positive[i])
			b = true_positive[i + 1] - a * false_positive[i + 1]
			auc = auc + a * (false_positive[i + 1] * false_positive[i + 1] - false_positive[i] * false_positive[i]) / 2 + b * (false_positive[i + 1] - false_positive[i]);
	auc = - auc
	auc = min(auc, 1.0)
	if (auc_roc):
		return (auc)
	if (best_threshold):
		l = [abs(y - x) for x, y in zip(false_positive, true_positive)]
		best_threshold_arg = max(zip(l, range(len(l))))[1]
		best = max(threshold[best_threshold_arg], 0.001)
		best = min(best, 0.999)
		return (best)
	plt.figure(figsize = (10,8))
	plt.rcParams['axes.facecolor'] = '#F5F5F5'
	plt.xlabel('False Positive Rate (1-Specificity)')
	plt.ylabel('True Positive Rate (Sensitivity)')
	plt.plot(false_positive, true_positive, color = "#FE5016")
	plt.plot([0,1], [0,1], color = "#444444")
	plt.ylim(0,1)
	plt.xlim(0,1)
	plt.title("ROC Curve\nAUC = " + str(auc))
	plt.gca().set_axisbelow(True)
	plt.grid()
	plt.show()
	return (tablesample(values = {"threshold": threshold, "false_positive": false_positive, "true_positive": true_positive}, table_info = False))
#
#
# Functions used by models to draw graphics which are not useful independantly.
#
#---#
def logit_plot(X: list, 
		  	   y: str, 
		  	   input_relation: str, 
		  	   coefficients: list,
		  	   cursor = None, 
		  	   max_nb_points = 50):
	check_types([
		("X", X, [list], False), 
		("y", y, [str], False), 
		("input_relation", input_relation, [str], False),
		("coefficients", coefficients, [list], False),
		("max_nb_points", max_nb_points, [int, float], False)])
	if not(cursor):
		conn = read_auto_connect()
		cursor = conn.cursor()
	else:
		conn = False
		check_cursor(cursor)
	def logit(x):
		return (1 / (1 + math.exp( - x)))
	if (len(X) == 1):
		query  = "(SELECT {}, {} FROM {} WHERE {} IS NOT NULL AND {} = 0 LIMIT {})".format(X[0], y, input_relation, X[0], y, int(max_nb_points / 2))
		query += " UNION ALL (SELECT {}, {} FROM {} WHERE {} IS NOT NULL AND {} = 1 LIMIT {})".format(X[0], y, input_relation, X[0], y, int(max_nb_points / 2))
		cursor.execute(query)
		all_points = cursor.fetchall()
		plt.figure(figsize = (10, 8), facecolor = '#F9F9F9')
		x0, x1 = [], []
		for idx, item in enumerate(all_points):
			if (item[1] == 0):
				x0 += [float(item[0])]
			else:
				x1 += [float(item[0])]
		min_logit, max_logit  = min(x0 + x1), max(x0 + x1)
		step = (max_logit - min_logit) / 40.0
		x_logit = arange(min_logit - 5 * step, max_logit + 5 * step, step) if (step > 0) else [max_logit]
		y_logit = [logit(coefficients[0] + coefficients[1] * item) for item in x_logit]
		plt.plot(x_logit, y_logit, alpha = 1, color = "black")
		all_scatter  = [plt.scatter(x0, [logit(coefficients[0] + coefficients[1] * item) for item in x0], alpha = 1, marker = "o", color = "#263133")]
		all_scatter += [plt.scatter(x1, [logit(coefficients[0] + coefficients[1] * item) for item in x1], alpha = 0.8, marker = "^", color = "#FE5016")]
		plt.gca().grid()
		plt.gca().set_axisbelow(True)
		plt.xlabel(X[0])
		plt.ylabel("logit")
		plt.legend(all_scatter, [0,1], scatterpoints = 1)
		plt.title(y + ' = logit(' + X[0] + ")")
	elif (len(X) == 2):
		try:
			import numpy
		except:
			raise Exception("You must install the numpy module to be able to plot 3D surfaces.")
		query  = "(SELECT {}, {}, {} FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL AND {} = 0 LIMIT {})".format(X[0], X[1], y, input_relation, X[0], X[1], y, int(max_nb_points / 2))
		query += " UNION (SELECT {}, {}, {} FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL AND {} = 1 LIMIT {})".format(X[0], X[1], y, input_relation, X[0], X[1], y, int(max_nb_points / 2))
		cursor.execute(query)
		all_points = cursor.fetchall()
		x0, x1, y0, y1 = [], [], [], []
		for idx, item in enumerate(all_points):
			if (item[2] == 0):
				x0 += [float(item[0])]
				y0 += [float(item[1])]
			else:
				x1 += [float(item[0])]
				y1 += [float(item[1])]
		min_logit_x, max_logit_x  = min(x0 + x1), max(x0 + x1)
		step_x = (max_logit_x - min_logit_x) / 40.0
		min_logit_y, max_logit_y  = min(y0 + y1), max(y0 + y1)
		step_y = (max_logit_y - min_logit_y) / 40.0
		X_logit = arange(min_logit_x - 5 * step_x, max_logit_x + 5 * step_x, step_x) if (step_x > 0) else [max_logit_x]
		Y_logit = arange(min_logit_y - 5 * step_y, max_logit_y + 5 * step_y, step_y) if (step_y > 0) else [max_logit_y]
		X_logit, Y_logit = numpy.meshgrid(X_logit, Y_logit)
		Z_logit = 1 / (1 + numpy.exp( - (coefficients[0] + coefficients[1] * X_logit + coefficients[2] * Y_logit)))
		fig = plt.figure(figsize = (10, 8))
		ax = fig.add_subplot(111, projection = '3d')
		ax.plot_surface(X_logit, Y_logit, Z_logit, rstride = 1, cstride = 1, alpha = 0.5, color = "gray")
		all_scatter  = [ax.scatter(x0, y0, [logit(coefficients[0] + coefficients[1] * x0[i] + coefficients[2] * y0[i]) for i in range(len(x0))], alpha = 1, marker = "o", color = "#263133")]
		all_scatter += [ax.scatter(x1, y1, [logit(coefficients[0] + coefficients[1] * x1[i] + coefficients[2] * y1[i]) for i in range(len(x1))], alpha = 0.8, marker = "^", color = "#FE5016")]
		ax.set_xlabel(X[0])
		ax.set_ylabel(X[1])
		ax.set_zlabel(y + ' = logit(' + X[0] + ", " + X[1] + ")")
		plt.legend(all_scatter, [0,1], scatterpoints = 1, loc = "lower left", title = y, bbox_to_anchor = (0.9, 1), ncol = 2, fontsize = 8)
	else:
		raise ValueError("The number of predictors is too big.")
	if (conn):
		conn.close()
	plt.show()
#---#
def lof_plot(input_relation: str,
			 columns: list,
			 lof: str,
			 cursor = None,
			 tablesample: float = -1):
	check_types([
		("input_relation", input_relation, [str], False),
		("columns", columns, [list], False),
		("lof", lof, [str], False),
		("tablesample", tablesample, [int, float], False)])
	if not(cursor):
		conn = read_auto_connect()
		cursor = conn.cursor()
	else:
		conn = False
		check_cursor(cursor)
	tablesample = "TABLESAMPLE({})".format(tablesample) if (tablesample > 0 and tablesample < 100) else ""
	if (len(columns) == 1):
		column = str_column(columns[0])
		query = "SELECT {}, {} FROM {} {} WHERE {} IS NOT NULL".format(column, lof, input_relation, tablesample, column)
		cursor.execute(query)
		query_result = cursor.fetchall()
		column1, lof = [item[0] for item in query_result], [item[1] for item in query_result]
		column2 = [0] * len(column1)
		plt.figure(figsize = (10,2))
		plt.gca().grid()
		plt.gca().set_axisbelow(True)
		plt.title('Local Outlier Factor (LOF)')
		plt.xlabel(column)
		radius = [1000 * (item - min(lof)) / (max(lof) - min(lof)) for item in lof]
		plt.scatter(column1, column2, color = "#263133", s = 14, label = 'Data points')
		plt.scatter(column1, column2, color = "#FE5016", s = radius, label = 'Outlier scores', facecolors = 'none')
	elif (len(columns) == 2):
		columns = [str_column(column) for column in columns]
		query = "SELECT {}, {}, {} FROM {} {} WHERE {} IS NOT NULL AND {} IS NOT NULL".format(columns[0], columns[1], lof, input_relation, tablesample, columns[0], columns[1])
		cursor.execute(query)
		query_result = cursor.fetchall()
		column1, column2, lof = [item[0] for item in query_result], [item[1] for item in query_result], [item[2] for item in query_result]
		plt.figure(figsize = (10,8))
		plt.gca().grid()
		plt.gca().set_axisbelow(True)
		plt.title('Local Outlier Factor (LOF)')
		plt.ylabel(columns[1])
		plt.xlabel(columns[0])
		radius = [1000 * (item - min(lof)) / (max(lof) - min(lof)) for item in lof]
		plt.scatter(column1, column2, color = "#263133", s = 14, label = 'Data points')
		plt.scatter(column1, column2, color = "#FE5016", s = radius, label = 'Outlier scores', facecolors = 'none')
	elif (len(columns) == 3):
		query = "SELECT {}, {}, {}, {} FROM {} {} WHERE {} IS NOT NULL AND {} IS NOT NULL AND {} IS NOT NULL".format(
					columns[0], columns[1], columns[2], lof, input_relation, tablesample, columns[0], columns[1], columns[2])
		cursor.execute(query)
		query_result = cursor.fetchall()
		column1, column2, column3, lof = [float(item[0]) for item in query_result], [float(item[1]) for item in query_result], [float(item[2]) for item in query_result], [float(item[3]) for item in query_result]
		fig = plt.figure(figsize = (10,8))
		ax = fig.add_subplot(111, projection = '3d')
		plt.title('Local Outlier Factor (LOF)')
		ax.set_xlabel(columns[0])
		ax.set_ylabel(columns[1])
		ax.set_zlabel(columns[2])
		radius = [1000 * (item - min(lof)) / (max(lof) - min(lof)) for item in lof]
		ax.scatter(column1, column2, column3, color = "#263133", label = 'Data points')
		ax.scatter(column1, column2, column3, color = "#FE5016", s = radius)
		ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
		ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
		ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
	else:
		raise ValueError("LocalOutlierFactor Plot is available for a maximum of 3 columns")
	if (conn):
		conn.close()
	plt.show()
#---#
def plot_importance(coeff_importances: dict, 
					coeff_sign: dict = {},
					print_legend: bool = True):
	check_types([
		("coeff_importances", coeff_importances, [dict], False),
		("coeff_sign", coeff_sign, [dict], False),
		("print_legend", print_legend, [bool], False)])
	coefficients, importances, signs = [], [], []
	for coeff in coeff_importances:
		coefficients += [coeff]
		importances += [coeff_importances[coeff]]
		signs += [coeff_sign[coeff]] if (coeff in coeff_sign) else [1]
	importances, coefficients, signs = zip( * sorted(zip(importances, coefficients, signs)))
	plt.figure(figsize = (12, int(len(importances) / 2) + 1))
	plt.rcParams['axes.facecolor'] = '#F5F5F5'
	color = []
	for item in signs:
		color += ['#263133'] if (item == 1) else ['#FE5016']
	plt.barh(range(0, len(importances)), importances, 0.9, color = color, alpha = 0.86)
	if (print_legend):
		orange = mpatches.Patch(color = '#FE5016', label = 'sign -')
		blue = mpatches.Patch(color = '#263133', label = 'sign +')
		plt.legend(handles = [orange,blue], loc = "lower right")
	plt.ylabel("Features")
	plt.xlabel("Importance")
	plt.gca().xaxis.grid()
	plt.gca().set_axisbelow(True)
	plt.yticks(range(0, len(importances)), coefficients)
	plt.show()
#---#
def plot_tree(tree, 
			  metric: str = "probability", 
			  pic_path: str = ""):
	try:
		from anytree import Node, RenderTree
	except:
		raise Exception("You must install the anytree module to be able to plot trees.")
	check_types([
		("metric", metric, [str], False),
		("pic_path", pic_path, [str], False)])
	try:
		import shutil
		screen_columns = shutil.get_terminal_size().columns
	except:
		import os
		screen_rows, screen_columns = os.popen('stty size', 'r').read().split()
	tree_id, nb_nodes, tree_depth, tree_breadth = tree["tree_id"][0], len(tree["node_id"]), max(tree["node_depth"]), sum([1 if item else 0 for item in tree["is_leaf"]])
	print("-" * int(screen_columns))
	print("Tree Id: {}".format(tree_id))
	print("Number of Nodes: {}".format(nb_nodes))
	print("Tree Depth: {}".format(tree_depth))
	print("Tree Breadth: {}".format(tree_breadth))
	print("-" * int(screen_columns))
	tree_nodes = {}
	for idx in range(nb_nodes):
		op = "<" if not(tree["is_categorical_split"][idx]) else "="
		if (tree["is_leaf"][idx]):
			tree_nodes[tree["node_id"][idx]] = Node('[{}] => {} ({} = {})'.format(tree["node_id"][idx], tree["prediction"][idx], metric, tree["probability/variance"][idx]))
		else:
			tree_nodes[tree["node_id"][idx]] = Node('[{}] ({} {} {} ?)'.format(tree["node_id"][idx], tree["split_predictor"][idx], op, tree["split_value"][idx]))
	for idx, node_id in enumerate(tree["node_id"]):
		if not(tree["is_leaf"][idx]):
			tree_nodes[node_id].children = [tree_nodes[tree["left_child_id"][idx]], tree_nodes[tree["right_child_id"][idx]]]
	for pre, fill, node in RenderTree(tree_nodes[1]):
		print("%s%s" % (pre,node.name)) 
	if (pic_path): 
		from anytree.dotexport import RenderTreeGraph
		RenderTreeGraph(tree_nodes[1]).to_picture(pic_path)
		if (isnotebook()):
			from IPython.core.display import HTML, display
			display(HTML("<img src='{}'>".format(pic_path)))
#---#
def regression_plot(X: list, 
		  	   		y: str, 
		  	   		input_relation: str,
		  	   		coefficients: list,
		  	   		cursor = None,
		  	   		max_nb_points: int = 50):
	check_types([
		("X", X, [list], False),
		("y", y, [str], False),
		("input_relation", input_relation, [str], False),
		("coefficients", coefficients, [list], False),
		("max_nb_points", max_nb_points, [int, float], False)])
	if not(cursor):
		conn = read_auto_connect()
		cursor = conn.cursor()
	else:
		conn = False
		check_cursor(cursor)
	if (len(X) == 1):
		query  = "SELECT {}, {} FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL LIMIT {}".format(X[0], y, input_relation, X[0], y, int(max_nb_points))
		cursor.execute(query)
		all_points = cursor.fetchall()
		plt.figure(figsize = (10, 8), facecolor = '#F9F9F9')
		x0, y0 = [float(item[0]) for item in all_points], [float(item[1]) for item in all_points]
		min_reg, max_reg  = min(x0), max(x0)
		x_reg = [min_reg, max_reg]
		y_reg = [coefficients[0] + coefficients[1] * item for item in x_reg]
		plt.plot(x_reg, y_reg, alpha = 1, color = "black")
		plt.scatter(x0, y0, alpha = 1, marker = "o", color = "#263133")
		plt.gca().grid()
		plt.gca().set_axisbelow(True)
		plt.xlabel(X[0])
		plt.ylabel(y)
		plt.title(y + ' = f(' + X[0] + ")")
	elif (len(X) == 2):
		try:
			import numpy
		except:
			raise Exception("You must install the numpy module to be able to plot 3D surfaces.")
		query  = "(SELECT {}, {}, {} FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL AND {} IS NOT NULL LIMIT {})".format(X[0], X[1], y, input_relation, X[0], X[1], y, int(max_nb_points))
		cursor.execute(query)
		all_points = cursor.fetchall()
		x0, y0, z0 = [float(item[0]) for item in all_points], [float(item[1]) for item in all_points], [float(item[2]) for item in all_points]
		min_reg_x, max_reg_x  = min(x0), max(x0)
		step_x = (max_reg_x - min_reg_x) / 40.0
		min_reg_y, max_reg_y  = min(y0), max(y0)
		step_y = (max_reg_y - min_reg_y) / 40.0
		X_reg = arange(min_reg_x - 5 * step_x, max_reg_x + 5 * step_x, step_x) if (step_x > 0) else [max_reg_x]
		Y_reg = arange(min_reg_y - 5 * step_y, max_reg_y + 5 * step_y, step_y) if (step_y > 0) else [max_reg_y]
		X_reg, Y_reg = numpy.meshgrid(X_reg, Y_reg)
		Z_reg = coefficients[0] + coefficients[1] * X_reg + coefficients[2] * Y_reg
		fig = plt.figure(figsize=(10, 8))
		ax = fig.add_subplot(111, projection = '3d')
		ax.plot_surface(X_reg, Y_reg, Z_reg, rstride = 1, cstride = 1, alpha = 0.5, color = "gray")
		ax.scatter(x0, y0, z0, alpha = 1, marker = "o", color = "#263133")
		ax.set_xlabel(X[0])
		ax.set_ylabel(X[1])
		ax.set_zlabel(y + ' = f(' + X[0] + ", " + X[1] + ")")
	else:
		raise ValueError("The number of predictors is too big.")
	if (conn):
		conn.close()
	plt.show()
#---#
def svm_classifier_plot(X: list, 
		  	   			y: str, 
		  	   			input_relation: str, 
		  	   			coefficients: list, 
		  	   			cursor = None,
		  	   			max_nb_points: int = 500):
	check_types([
		("X", X, [list], False),
		("y", y, [str], False),
		("input_relation", input_relation, [str], False),
		("coefficients", coefficients, [list], False),
		("max_nb_points", max_nb_points, [int, float], False)])
	if not(cursor):
		conn = read_auto_connect()
		cursor = conn.cursor()
	else:
		conn = False
		check_cursor(cursor)
	if (len(X) == 1):
		query  = "(SELECT {}, {} FROM {} WHERE {} IS NOT NULL AND {} = 0 LIMIT {})".format(X[0], y, input_relation, X[0], y, int(max_nb_points / 2))
		query += " UNION ALL (SELECT {}, {} FROM {} WHERE {} IS NOT NULL AND {} = 1 LIMIT {})".format(X[0], y, input_relation, X[0], y, int(max_nb_points / 2))
		cursor.execute(query)
		all_points = cursor.fetchall()
		plt.figure(figsize = (10, 2), facecolor = '#F9F9F9')
		x0, x1 = [], []
		for idx, item in enumerate(all_points):
			if (item[1] == 0):
				x0 += [float(item[0])]
			else:
				x1 += [float(item[0])]
		x_svm, y_svm = [ - coefficients[0] / coefficients[1], - coefficients[0] / coefficients[1]], [-1, 1]
		plt.plot(x_svm, y_svm, alpha = 1, color = "black")
		all_scatter  = [plt.scatter(x0, [0 for item in x0], marker = "o", color = "#263133")]
		all_scatter += [plt.scatter(x1, [0 for item in x1], marker = "^", color = "#FE5016")]
		plt.gca().grid()
		plt.gca().set_axisbelow(True)
		plt.xlabel(X[0])
		plt.legend(all_scatter, [0, 1], scatterpoints = 1)
		plt.title('svm(' + X[0] + ")")
	elif (len(X) == 2):
		query  = "(SELECT {}, {}, {} FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL AND {} = 0 LIMIT {})".format(X[0], X[1], y, input_relation, X[0], X[1], y, int(max_nb_points / 2))
		query += " UNION (SELECT {}, {}, {} FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL AND {} = 1 LIMIT {})".format(X[0], X[1], y, input_relation, X[0], X[1], y, int(max_nb_points / 2))
		cursor.execute(query)
		all_points = cursor.fetchall()
		plt.figure(figsize = (10, 8), facecolor = '#F9F9F9')
		x0, x1, y0, y1 = [], [], [], []
		for idx, item in enumerate(all_points):
			if (item[2] == 0):
				x0 += [float(item[0])]
				y0 += [float(item[1])]
			else:
				x1 += [float(item[0])]
				y1 += [float(item[1])]
		min_svm, max_svm  = min(x0 + x1), max(x0 + x1)
		x_svm, y_svm = [min_svm, max_svm], [ - (coefficients[0] + coefficients[1] * min_svm) / coefficients[2], - (coefficients[0] + coefficients[1] * max_svm) / coefficients[2]]
		plt.plot(x_svm, y_svm, alpha = 1, color = "black")
		all_scatter  = [plt.scatter(x0, y0, marker = "o", color = "#263133")]
		all_scatter += [plt.scatter(x1, y1, marker = "^", color = "#FE5016")]
		plt.gca().grid()
		plt.gca().set_axisbelow(True)
		plt.xlabel(X[0])
		plt.ylabel(X[1])
		plt.legend(all_scatter, [0, 1], scatterpoints = 1)
		plt.title('svm(' + X[0] + ", " + X[1] + ")")
	elif (len(X) == 3):
		try:
			import numpy
		except:
			raise Exception("You must install the numpy module to be able to plot 3D surfaces.")
		query  = "(SELECT {}, {}, {}, {} FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL AND {} IS NOT NULL AND {} = 0 LIMIT {})".format(X[0], X[1], X[2], y, input_relation, X[0], X[1], X[2], y, int(max_nb_points / 2))
		query += " UNION (SELECT {}, {}, {}, {} FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL AND {} IS NOT NULL AND {} = 1 LIMIT {})".format(X[0], X[1], X[2], y, input_relation, X[0], X[1], X[2], y, int(max_nb_points / 2))
		cursor.execute(query)
		all_points = cursor.fetchall()
		x0, x1, y0, y1, z0, z1 = [], [], [], [], [], []
		for idx, item in enumerate(all_points):
			if (item[3] == 0):
				x0 += [float(item[0])]
				y0 += [float(item[1])]
				z0 += [float(item[2])]
			else:
				x1 += [float(item[0])]
				y1 += [float(item[1])]
				z1 += [float(item[2])]
		min_svm_x, max_svm_x  = min(x0 + x1), max(x0 + x1)
		step_x = (max_svm_x - min_svm_x) / 40.0
		min_svm_y, max_svm_y  = min(y0 + y1), max(y0 + y1)
		step_y = (max_svm_y - min_svm_y) / 40.0
		X_svm = arange(min_svm_x - 5 * step_x, max_svm_x + 5 * step_x, step_x) if (step_x > 0) else [max_svm_x]
		Y_svm = arange(min_svm_y - 5 * step_y, max_svm_y + 5 * step_y, step_y) if (step_y > 0) else [max_svm_y]
		X_svm, Y_svm = numpy.meshgrid(X_svm, Y_svm)
		Z_svm = coefficients[0] + coefficients[1] * X_svm + coefficients[2] * Y_svm
		fig = plt.figure(figsize=(10, 8))
		ax = fig.add_subplot(111, projection = '3d')
		ax.plot_surface(X_svm, Y_svm, Z_svm, rstride = 1, cstride = 1, alpha = 0.5, color = "gray")
		all_scatter  = [ax.scatter(x0, y0, z0, alpha = 1, marker = "o", color = "#263133")]
		all_scatter += [ax.scatter(x1, y1, z1, alpha = 0.8, marker = "^", color = "#FE5016")]
		ax.set_xlabel(X[0])
		ax.set_ylabel(X[1])
		ax.set_zlabel(X[2])
		plt.title('svm(' + X[0] + ", " + X[1] + ", " + X[2] + ")")
		plt.legend(all_scatter, [0, 1], scatterpoints = 1, loc = "lower left", title = y, bbox_to_anchor = (0.9, 1), ncol = 2, fontsize = 8)
	else:
		raise ValueError("The number of predictors is too big.")
	if (conn):
		conn.close()
	plt.show()
#---#
def voronoi_plot(clusters: list, columns: list):
	check_types([
		("clusters", clusters, [list], False),
		("columns", columns, [list], False)])
	from scipy.spatial import voronoi_plot_2d, Voronoi
	v = Voronoi(clusters)
	voronoi_plot_2d(v, show_vertices = 0)
	plt.xlabel(columns[0])
	plt.ylabel(columns[1])
	plt.show()