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
import statistics
# VerticaPy Modules
from verticapy.utilities import *
from verticapy.toolbox import *
from verticapy.learn.cluster import KMeans
from verticapy.connections.connect import read_auto_connect
#---#
def best_k(X: list,
		   input_relation: str,
		   cursor = None,
		   n_cluster = (1, 100),
		   init = "kmeanspp",
		   max_iter: int = 50,
		   tol: float = 1e-4,
		   elbow_score_stop: float = 0.8):
	"""
---------------------------------------------------------------------------
Finds the KMeans K based on a score.

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
elbow_score_stop: float, optional
	Stops the Parameters Search when this Elbow score is reached.

Returns
-------
int
	the KMeans K
	"""
	check_types([
		("X", X, [list], False), 
		("input_relation", input_relation, [str], False), 
		("n_cluster", n_cluster, [list, tuple], False),
		("init", init, ["kmeanspp", "random"], True),
		("max_iter", max_iter, [int, float], False),
		("tol", tol, [int, float], False),
		("elbow_score_stop", elbow_score_stop, [int, float], False)])
	if not(cursor):
		conn = read_auto_connect()
		cursor = conn.cursor()
	else:
		conn = False
		check_cursor(cursor)
	if not(type(n_cluster) == list):
		L = range(n_cluster[0], n_cluster[1])
	else:
		L = n_cluster
		L.sort()
	schema, relation = schema_relation(input_relation)
	schema = str_column(schema)
	relation_alpha = ''.join(ch for ch in relation if ch.isalnum())
	for i in L:
		cursor.execute("DROP MODEL IF EXISTS {}.__vpython_kmeans_tmp_model_{}__".format(schema, relation_alpha))
		model = KMeans("{}.__vpython_kmeans_tmp_model_{}__".format(schema, relation_alpha), cursor, i, init, max_iter, tol)
		model.fit(input_relation, X)
		score = model.metrics.values["value"][3]
		if (score > elbow_score_stop):
			return i
		score_prev = score
	if (conn):
		conn.close()
	print("\u26A0 The K was not found. The last K (= {}) is returned with an elbow score of {}".format(i, score))
	return i
#---#
def cross_validate(estimator, 
				   input_relation: str, 
				   X: list, 
				   y: str, 
				   cv: int = 3, 
				   pos_label = None, 
				   cutoff: float = -1):
	"""
---------------------------------------------------------------------------
Computes the K-Fold cross validation of an estimator.

Parameters
----------
estimator: object
	Vertica estimator having a fit method and a DB cursor.
input_relation: str
	Relation used to train the model.
X: list
	List of the predictor columns.
y: str
	Response Column.
cv: int, optional
	Number of folds.
pos_label: int/float/str, optional
	The main class to be considered as positive (classification only).
cutoff: float, optional
	The model cutoff (classification only).

Returns
-------
tablesample
 	An object containing the result. For more information, check out
 	utilities.tablesample.
	"""
	check_types([
		("X", X, [list], False), 
		("input_relation", input_relation, [str], False), 
		("y", y, [str], False),
		("cv", cv, [int, float], False),
		("cutoff", cutoff, [int, float], False)])
	if (cv < 2):
		raise ValueError("Cross Validation is only possible with at least 2 folds")
	if (estimator.type == "regressor"):
		result = {"index": ["explained_variance", "max_error", "median_absolute_error", "mean_absolute_error", "mean_squared_error", "r2"]} 
	elif (estimator.type == "classifier"):
		result = {"index": ["auc", "prc_auc", "accuracy", "log_loss", "precision", "recall", "f1_score", "mcc", "informedness", "markedness", "csi"]}
	else:
		raise ValueError("Cross Validation is only possible for Regressors and Classifiers")
	try:
		schema, relation = schema_relation(estimator.name)
		schema = str_column(schema)
	except:
		schema, relation = schema_relation(input_relation)
		schema, relation = str_column(schema), "model_{}".format(relation)
	relation_alpha = ''.join(ch for ch in relation if ch.isalnum())
	test_name, train_name = "{}_{}".format(relation_alpha, int(1 / cv * 100)), "{}_{}".format(relation_alpha, int(100 - 1 / cv * 100))
	try:
		estimator.cursor.execute("DROP TABLE IF EXISTS v_temp_schema.VERTICAPY_CV_SPLIT_{}".format(relation_alpha))
	except:
		pass
	query = "CREATE LOCAL TEMPORARY TABLE VERTICAPY_CV_SPLIT_{} ON COMMIT PRESERVE ROWS AS SELECT *, RANDOMINT({}) AS test FROM {}".format(relation_alpha, cv, input_relation)
	estimator.cursor.execute(query)
	for i in range(cv):
		try:
			estimator.cursor.execute("DROP MODEL IF EXISTS {}".format(estimator.name))
		except:
			pass
		estimator.cursor.execute("DROP VIEW IF EXISTS {}.VERTICAPY_CV_SPLIT_{}_TEST".format(schema, test_name))
		estimator.cursor.execute("DROP VIEW IF EXISTS {}.VERTICAPY_CV_SPLIT_{}_TRAIN".format(schema, train_name))
		query = "CREATE VIEW {}.VERTICAPY_CV_SPLIT_{}_TEST AS SELECT * FROM {} WHERE (test = {})".format(schema, test_name, "v_temp_schema.VERTICAPY_CV_SPLIT_{}".format(relation_alpha), i)
		estimator.cursor.execute(query)
		query = "CREATE VIEW {}.VERTICAPY_CV_SPLIT_{}_TRAIN AS SELECT * FROM {} WHERE (test != {})".format(schema, train_name, "v_temp_schema.VERTICAPY_CV_SPLIT_{}".format(relation_alpha), i)
		estimator.cursor.execute(query)
		estimator.fit("{}.VERTICAPY_CV_SPLIT_{}_TRAIN".format(schema, train_name), X, y, "{}.VERTICAPY_CV_SPLIT_{}_TEST".format(schema, test_name))
		if (estimator.type == "regressor"):
			result["{}-fold".format(i + 1)] = estimator.regression_report().values["value"]
		else:
			if (len(estimator.classes) > 2) and (pos_label not in estimator.classes):
				raise ValueError("'pos_label' must be in the estimator classes, it must be the main class to study for the Cross Validation")
			elif (len(estimator.classes) == 2) and (pos_label not in estimator.classes):
				pos_label = estimator.classes[1]
			try:
				result["{}-fold".format(i + 1)] = estimator.classification_report(labels = [pos_label], cutoff = cutoff).values["value"][0:-1]
			except:
				result["{}-fold".format(i + 1)] = estimator.classification_report(cutoff = cutoff).values["value"][0:-1]
		try:
			estimator.cursor.execute("DROP MODEL IF EXISTS {}".format(estimator.name))
		except:
			pass
	n = 6 if (estimator.type == "regressor") else 11
	total = [[] for item in range(n)]
	for i in range(cv):
		for k in range(n):
			total[k] += [result["{}-fold".format(i + 1)][k]]
	result["avg"], result["std"] = [], []
	for item in total:
		result["avg"] += [statistics.mean([float(elem) for elem in item])] 
		result["std"] += [statistics.stdev([float(elem) for elem in item])] 
	estimator.cursor.execute("DROP TABLE IF EXISTS v_temp_schema.VERTICAPY_CV_SPLIT_{}".format(relation_alpha))
	estimator.cursor.execute("DROP VIEW IF EXISTS {}.VERTICAPY_CV_SPLIT_{}_TEST".format(schema, test_name))
	estimator.cursor.execute("DROP VIEW IF EXISTS {}.VERTICAPY_CV_SPLIT_{}_TRAIN".format(schema, train_name))
	return (tablesample(values = result, table_info = False).transpose())
#---#
def train_test_split(input_relation: str, 
					 cursor = None, 
					 test_size: float = 0.33, 
					 schema_writing: str = ""):
	"""
---------------------------------------------------------------------------
Creates a temporary table and 2 views which can be used to evaluate a model. 
The table will include all the main relation information with a test column 
(boolean) which represents if the data belong to the test or train set.

Parameters
----------
input_relation: str
	Input Relation.
cursor: DBcursor, optional
	Vertica DB cursor.
test_size: float, optional
	Proportion of the test set comparint to the training set.
schema_writing: str, optional
	Schema used to write the main relation.

Returns
-------
tuple
 	(name of the train view, name of the test view)
	"""
	check_types([
		("test_size", test_size, [float], False),
		("schema_writing", schema_writing, [str], False), 
		("input_relation", input_relation, [str], False)])
	if not(cursor):
		conn = read_auto_connect()
		cursor = conn.cursor()
	else:
		conn = False
		check_cursor(cursor)
	schema, relation = schema_relation(input_relation)
	schema = str_column(schema) if not(schema_writing) else schema_writing
	relation_alpha = ''.join(ch for ch in relation if ch.isalnum())
	test_name, train_name = "{}_{}".format(relation_alpha, int(test_size * 100)), "{}_{}".format(relation_alpha, int(100 - test_size * 100))
	try:
		cursor.execute("DROP TABLE IF EXISTS {}.VERTICAPY_SPLIT_{}".format(schema, relation_alpha))
	except:
		pass
	cursor.execute("DROP VIEW IF EXISTS {}.VERTICAPY_SPLIT_{}_TEST".format(schema, test_name))
	cursor.execute("DROP VIEW IF EXISTS {}.VERTICAPY_SPLIT_{}_TRAIN".format(schema, train_name))
	query = "CREATE TABLE {}.VERTICAPY_SPLIT_{} AS SELECT *, (CASE WHEN RANDOM() < {} THEN True ELSE False END) AS test FROM {}".format(schema, relation_alpha, test_size, input_relation)
	cursor.execute(query)
	query = "CREATE VIEW {}.VERTICAPY_SPLIT_{}_TEST AS SELECT * FROM {} WHERE test".format(schema, test_name, "{}.VERTICAPY_SPLIT_{}".format(schema, relation_alpha))
	cursor.execute(query)
	query = "CREATE VIEW {}.VERTICAPY_SPLIT_{}_TRAIN AS SELECT * FROM {} WHERE NOT(test)".format(schema, train_name, "{}.VERTICAPY_SPLIT_{}".format(schema, relation_alpha))
	cursor.execute(query)
	if (conn):
		conn.close()
	return ("{}.VERTICAPY_SPLIT_{}_TRAIN".format(schema, train_name), "{}.VERTICAPY_SPLIT_{}_TEST".format(schema, test_name))