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
from verticapy.connections.connect import read_auto_connect
#---#
class RandomForestClassifier:
	"""
---------------------------------------------------------------------------
Creates a RandomForestClassifier object by using the Vertica Highly Distributed 
and Scalable Random Forest on the data. It is one of the ensemble learning 
method for classification that operate by constructing a multitude of decision 
trees at training time and outputting the class that is the mode of the classes.

Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
cursor: DBcursor, optional
	Vertica DB cursor. 
n_estimators: int, optional
	The number of trees in the forest, an integer between 0 and 1000, inclusive.
max_features: str, optional
	The number of randomly chosen features from which to pick the best feature 
	to split on a given tree node. It can be an integer or one of the two following
	methods.
		auto : square root of the total number of predictors.
		max  : number of predictors.
max_leaf_nodes: int, optional
	The maximum number of leaf nodes a tree in the forest can have, an integer 
	between 1 and 1e9, inclusive.
sample: float, optional
	The portion of the input data set that is randomly picked for training each tree, 
	a float between 0.0 and 1.0, inclusive. 
max_depth: int, optional
	The maximum depth for growing each tree, an integer between 1 and 100, inclusive.
min_samples_leaf: int, optional
	The minimum number of samples each branch must have after splitting a node, an 
	integer between 1 and 1e6, inclusive. A split that causes fewer remaining samples 
	is discarded. 
min_info_gain: float, optional
	The minimum threshold for including a split, a float between 0.0 and 1.0, inclusive. 
	A split with information gain less than this threshold is discarded.
nbins: int, optional 
	The number of bins to use for continuous features, an integer between 2 and 1000, 
	inclusive.

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
				  name: str,
				  cursor = None,
				  n_estimators: int = 10,
				  max_features = "auto",
				  max_leaf_nodes: int = 1e9, 
				  sample: float = 0.632, 
				  max_depth: int = 5,
				  min_samples_leaf: int = 1,
				  min_info_gain: float = 0.0,
				  nbins: int = 32):
		check_types([
			("name", name, [str], False),
			("n_estimators", n_estimators, [int, float], False),
			("max_features", max_features, [str, int, float], False),
			("max_leaf_nodes", max_leaf_nodes, [int, float], False),
			("sample", sample, [int, float], False),
			("max_depth", max_depth, [int, float], False),
			("min_samples_leaf", min_samples_leaf, [int, float], False),
			("min_info_gain", min_info_gain, [int, float], False),
			("nbins", nbins, [int, float], False)])
		if not(cursor):
			cursor = read_auto_connect().cursor()
		else:
			check_cursor(cursor)
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
	#---#
	def __repr__(self):
		try:
			self.cursor.execute("SELECT GET_MODEL_SUMMARY(USING PARAMETERS model_name = '" + self.name + "')")
			return (self.cursor.fetchone()[0])
		except:
			return "<RandomForestClassifier>"
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
			return (confusion_matrix(self.y, self.deploySQL(pos_label, cutoff), self.test_relation, self.cursor, pos_label = pos_label))
		else:
			return (multilabel_confusion_matrix(self.y, self.deploySQL(), self.test_relation, self.classes, self.cursor))
	#---#
	def deploySQL(self, 
				  pos_label = None, 
				  cutoff: float = -1, 
				  allSQL: bool = False):
		"""
	---------------------------------------------------------------------------
	Returns the SQL code needed to deploy the model. 

	Parameters
	----------
	pos_label: int/float/str, optional
		Label to consider as positive. All the other classes will be merged and
		considered as negative in case of multi classification.
	cutoff: float, optional
		Cutoff for which the tested category will be accepted as prediction. If 
		the cutoff is not between 0 and 1, a probability will be returned.
	allSQL: bool, optional
		If set to True, the output will be a list of the different SQL codes 
		needed to deploy the different categories score.

	Returns
	-------
	str/list
 		the SQL code needed to deploy the model.
		"""
		check_types([
			("cutoff", cutoff, [int, float], False), 
			("allSQL", allSQL, [bool], False)])
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
	#---#
	def drop(self):
		"""
	---------------------------------------------------------------------------
	Drops the model from the Vertica DB.
		"""
		drop_model(self.name, self.cursor, print_info = False)
	#---#
	def export_graphviz(self, 
						tree_id: int = 0):
		"""
	---------------------------------------------------------------------------
	Converts the input tree to graphviz.

	Parameters
	----------
	tree_id: int, optional
		Unique tree identifier. It is an integer between 0 and n_estimators - 1

	Returns
	-------
	str
 		graphviz formatted tree.
		"""
		check_types([("tree_id", tree_id, [int, float], False)])
		query = "SELECT READ_TREE ( USING PARAMETERS model_name = '{}', tree_id = {}, format = 'graphviz');".format(self.name, tree_id)
		self.cursor.execute(query)
		return (self.cursor.fetchone()[1])
	#---#
	def features_importance(self):
		"""
	---------------------------------------------------------------------------
	Computes the model features importance using the Gini Index.

	Returns
	-------
	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.
		"""
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
	#---#
	def get_tree(self, 
				 tree_id: int = 0):
		"""
	---------------------------------------------------------------------------
	Returns a tablesample with all the input tree information.

	Parameters
	----------
	tree_id: int, optional
		Unique tree identifier. It is an integer between 0 and n_estimators - 1

	Returns
	-------
	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.
		"""
		check_types([("tree_id", tree_id, [int, float], False)])
		query = "SELECT READ_TREE ( USING PARAMETERS model_name = '{}', tree_id = {}, format = 'tabular');".format(self.name, tree_id)
		result = to_tablesample(query = query, cursor = self.cursor)
		result.table_info = False
		return (result)
	#---#
	def lift_chart(self, 
				   pos_label = None):
		"""
	---------------------------------------------------------------------------
	Draws the model Lift Chart.

	Parameters
	----------
	pos_label: int/float/str, optional
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
		return (lift_chart(self.y, self.deploySQL(allSQL = True)[0].format(pos_label), self.test_relation, self.cursor, pos_label))
	#---#
	def plot_tree(self, 
				  tree_id: int = 0, 
				  pic_path: str = ""):
		"""
	---------------------------------------------------------------------------
	Draws the input tree. The module anytree must be installed in the machine.

	Parameters
	----------
	tree_id: int, optional
		Unique tree identifier. It is an integer between 0 and n_estimators - 1
	pic_path: str, optional
		Absolute path to save the image of the tree. 
		"""
		check_types([
			("tree_id", tree_id, [int, float], False),
			("pic_path", pic_path, [str], False)])
		plot_tree(self.get_tree(tree_id = tree_id).values, metric = "probability", pic_path = pic_path)
	#---#
	def prc_curve(self, 
				  pos_label = None):
		"""
	---------------------------------------------------------------------------
	Draws the model PRC curve.

	Parameters
	----------
	pos_label: int/float/str, optional
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
		return (prc_curve(self.y, self.deploySQL(allSQL = True)[0].format(pos_label), self.test_relation, self.cursor, pos_label))
	#---# 
	def predict(self,
				vdf,
				name: str = "",
				cutoff: float = -1,
				pos_label = None):
		"""
	---------------------------------------------------------------------------
	Predicts using the input relation.

	Parameters
	----------
	vdf: vDataFrame
		Object used to insert the prediction as a vcolumn.
	name: str, optional
		Name of the added vcolumn. If empty, a name will be generated.
	cutoff: float, optional
		Cutoff for which the tested category will be accepted as prediction. 
		If the parameter is not between 0 and 1, the class probability will
		be returned.
	pos_label: int/float/str, optional
		Class label.

	Returns
	-------
	vDataFrame
		the input object.
		"""
		check_types([
			("name", name, [str], False),
			("cutoff", cutoff, [int, float], False)],
			vdf = ["vdf", vdf])
		name = "RandomForestClassifier_" + ''.join(ch for ch in self.name if ch.isalnum()) if not (name) else name
		if (len(self.classes) == 2 and pos_label == None): pos_label = self.classes[1] 
		return (vdf.eval(name, self.deploySQL(pos_label, cutoff)))
	#---#
	def roc_curve(self, 
				  pos_label = None):
		"""
	---------------------------------------------------------------------------
	Draws the model ROC curve.

	Parameters
	----------
	pos_label: int/float/str, optional
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
		return (roc_curve(self.y, self.deploySQL(allSQL = True)[0].format(pos_label), self.test_relation, self.cursor, pos_label))
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
		If the parameter is not between 0 and 1, an automatic cutoff is 
		computed.
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
		if (pos_label not in self.classes) and (method != "accuracy"):
			raise ValueError("'pos_label' must be one of the response column classes")
		elif (cutoff >= 1 or cutoff <= 0) and (method != "accuracy"):
			cutoff = self.score(pos_label, 0.5, "best_cutoff")
		if (method in ("accuracy", "acc")):
			return accuracy_score(self.y, self.deploySQL(pos_label, cutoff), self.test_relation, self.cursor, pos_label)
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

#---# 
class RandomForestRegressor:
	"""
---------------------------------------------------------------------------
Creates a RandomForestRegressor object by using the Vertica Highly Distributed 
and Scalable Random Forest on the data. It is one of the ensemble learning 
method for regression that operate by constructing a multitude of decision 
trees at training time and outputting the mean prediction.

Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
cursor: DBcursor, optional
	Vertica DB cursor. 
n_estimators: int, optional
	The number of trees in the forest, an integer between 0 and 1000, inclusive.
max_features: str, optional
	The number of randomly chosen features from which to pick the best feature 
	to split on a given tree node. It can be an integer or one of the two following
	methods.
		auto : square root of the total number of predictors.
		max  : number of predictors.
max_leaf_nodes: int, optional
	The maximum number of leaf nodes a tree in the forest can have, an integer 
	between 1 and 1e9, inclusive.
sample: float, optional
	The portion of the input data set that is randomly picked for training each tree, 
	a float between 0.0 and 1.0, inclusive. 
max_depth: int, optional
	The maximum depth for growing each tree, an integer between 1 and 100, inclusive.
min_samples_leaf: int, optional
	The minimum number of samples each branch must have after splitting a node, an 
	integer between 1 and 1e6, inclusive. A split that causes fewer remaining samples 
	is discarded. 
min_info_gain: float, optional
	The minimum threshold for including a split, a float between 0.0 and 1.0, inclusive. 
	A split with information gain less than this threshold is discarded.
nbins: int, optional 
	The number of bins to use for continuous features, an integer between 2 and 1000, 
	inclusive.

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
				  name: str,
				  cursor = None,
				  n_estimators: int = 10,
				  max_features = "auto",
				  max_leaf_nodes: int = 1e9, 
				  sample: float = 0.632, 
				  max_depth: int = 5,
				  min_samples_leaf: int = 1,
				  min_info_gain: float = 0.0,
				  nbins: int = 32):
		check_types([
			("name", name, [str], False),
			("n_estimators", n_estimators, [int, float], False),
			("max_features", max_features, [str, int, float], False),
			("max_leaf_nodes", max_leaf_nodes, [int, float], False),
			("sample", sample, [int, float], False),
			("max_depth", max_depth, [int, float], False),
			("min_samples_leaf", min_samples_leaf, [int, float], False),
			("min_info_gain", min_info_gain, [int, float], False),
			("nbins", nbins, [int, float], False)])
		if not(cursor):
			cursor = read_auto_connect().cursor()
		else:
			check_cursor(cursor)
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
		sql = "PREDICT_RF_REGRESSOR({} USING PARAMETERS model_name = '{}', match_by_pos = 'true')"
		return (sql.format(", ".join(self.X), self.name))
	#---#
	def drop(self):
		"""
	---------------------------------------------------------------------------
	Drops the model from the Vertica DB.
		"""
		drop_model(self.name, self.cursor, print_info = False)
	#---#
	def export_graphviz(self, 
						tree_id: int = 0):
		"""
	---------------------------------------------------------------------------
	Converts the input tree to graphviz.

	Parameters
	----------
	tree_id: int, optional
		Unique tree identifier. It is an integer between 0 and n_estimators - 1

	Returns
	-------
	str
 		graphviz formatted tree.
		"""
		check_types([("tree_id", tree_id, [int, float], False)])
		query = "SELECT READ_TREE ( USING PARAMETERS model_name = '{}', tree_id = {}, format = 'graphviz');".format(self.name, tree_id)
		self.cursor.execute(query)
		return (self.cursor.fetchone()[1])
	#---# 
	def features_importance(self):
		"""
	---------------------------------------------------------------------------
	Computes the model features importance using the Gini Index.

	Returns
	-------
	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.
		"""
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
		if (self.max_features == "auto"):
			self.max_features = int(len(self.X) / 3 + 1)
		elif (self.max_features == "max"):
			self.max_features = len(self.X)
		query = "SELECT RF_REGRESSOR('{}', '{}', '{}', '{}' USING PARAMETERS ntree = {}, mtry = {}, sampling_size = {}"
		query = query.format(self.name, input_relation, self.y, ", ".join(self.X), self.n_estimators, self.max_features, self.sample)
		query += ", max_depth = {}, max_breadth = {}, min_leaf_size = {}, min_info_gain = {}, nbins = {})".format(self.max_depth, int(self.max_leaf_nodes), self.min_samples_leaf, self.min_info_gain, self.nbins)
		self.cursor.execute(query)
		return (self)
	#---#
	def get_tree(self, 
				 tree_id: int = 0):
		"""
	---------------------------------------------------------------------------
	Returns a table with all the input tree information.

	Parameters
	----------
	tree_id: int, optional
		Unique tree identifier. It is an integer between 0 and n_estimators - 1

	Returns
	-------
	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.
		"""
		check_types([("tree_id", tree_id, [int, float], False)])
		query = "SELECT READ_TREE ( USING PARAMETERS model_name = '{}', tree_id = {}, format = 'tabular');".format(self.name, tree_id)
		result = to_tablesample(query = query, cursor = self.cursor)
		result.table_info = False
		return (result)
	#---#
	def plot_tree(self, 
				  tree_id: int = 0, 
				  pic_path: str = ""):
		"""
	---------------------------------------------------------------------------
	Draws the input tree. The module anytree must be installed in the machine.

	Parameters
	----------
	tree_id: int, optional
		Unique tree identifier. It is an integer between 0 and n_estimators - 1
	pic_path: str, optional
		Absolute path to save the image of the tree. 
		"""
		check_types([
			("tree_id", tree_id, [int, float], False),
			("pic_path", pic_path, [str], False)])
		plot_tree(self.get_tree(tree_id = tree_id).values, metric = "variance", pic_path = pic_path)
	#---# 
	def predict(self,
				vdf,
				name: str = ""):
		"""
	---------------------------------------------------------------------------
	Adds the prediction in a vDataFrame.

	Parameters
	----------
	vdf: vDataFrame
		Object used to insert the prediction as a vcolumn.
	name: str, optional
		Name of the added vcolumn. If empty, a name will be generated.

	Returns
	-------
	vDataFrame
		the input object.
		"""
		check_types([
			("name", name, [str], False)],
			vdf = ["vdf", vdf])
		name = "RandomForestRegressor_" + ''.join(ch for ch in self.name if ch.isalnum()) if not (name) else name
		return (vdf.eval(name, self.deploySQL()))
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
		return (regression_report(self.y, self.deploySQL(), self.test_relation, self.cursor))
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