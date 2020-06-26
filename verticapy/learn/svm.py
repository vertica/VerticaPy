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
from verticapy import vDataFrame
from verticapy.learn.metrics import *
from verticapy.learn.plot import *
from verticapy.utilities import *
from verticapy.toolbox import *
from verticapy.connections.connect import read_auto_connect
#---#
class LinearSVC:
	"""
---------------------------------------------------------------------------
Creates a LinearSVC object by using the Vertica Highly Distributed and 
Scalable SVM on the data. Given a set of training examples, each marked as 
belonging to one or the other of two categories, an SVM training algorithm 
builds a model that assigns new examples to one category or the other, making 
it a non-probabilistic binary linear classifier.

Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
cursor: DBcursor, optional
	Vertica DB cursor.
tol: float, optional
	Used to control accuracy.
C: float, optional
	The weight for misclassification cost. The algorithm minimizes the 
	regularization cost and the misclassification cost.
fit_intercept: bool, optional
	A bool to fit also the intercept.
intercept_scaling: float
	A float value, serves as the value of a dummy feature whose coefficient 
	Vertica uses to calculate the model intercept. Because the dummy feature 
	is not in the training data, its values are set to a constant, by default 
	set to 1. 
intercept_mode: str, optional
	Specify how to treat the intercept.
		regularized   : Fits the intercept and applies a regularization on it.
		unregularized : Fits the intercept but does not include it in regularization. 
class_weight: list, optional
	Specifies how to determine weights of the two classes. It can be a list of 2 
	elements or one of the following method:
		auto : Weights each class according to the number of samples.
		none : No weights are used.
max_iter: int, optional
	The maximum number of iterations that the algorithm performs.

Attributes
----------
After the object creation, all the parameters become attributes. 
The model will also create extra attributes when fitting the model:

coef: tablesample
	Coefficients and their mathematical information (pvalue, std, value...)
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
				  tol: float = 1e-4, 
				  C: float = 1.0, 
				  fit_intercept: bool = True,
				  intercept_scaling: float = 1.0,
				  intercept_mode: str = "regularized",
				  class_weight: list = [1, 1],
				  max_iter: int = 100):
		check_types([
			("name", name, [str], False),
			("tol", tol, [int, float], False),
			("C", C, [int, float], False),
			("fit_intercept", fit_intercept, [bool], False),
			("intercept_scaling", intercept_scaling, [int, float], False),
			("intercept_mode", intercept_mode, ["unregularized", "regularized"], True),
			("max_iter", max_iter, [int, float], False)])
		if not(cursor):
			cursor = read_auto_connect().cursor()
		else:
			check_cursor(cursor)
		self.type = "classifier"
		self.classes = [0, 1]
		self.cursor = cursor
		self.name = name
		self.tol = tol
		self.C = C 
		self.fit_intercept = fit_intercept 
		self.intercept_scaling = intercept_scaling
		self.intercept_mode = intercept_mode.lower()
		self.class_weight = class_weight
		self.max_iter = max_iter
	#---#
	def __repr__(self):
		try:
			self.cursor.execute("SELECT GET_MODEL_SUMMARY(USING PARAMETERS model_name = '{}')".format(self.name))
			return (self.cursor.fetchone()[0])
		except:
			return "<LinearSVC>"
	#
	# Methods
	#
	#---#
	def classification_report(self, 
							  cutoff: float = 0.5):
		"""
	---------------------------------------------------------------------------
	Computes a classification report using multiple metrics to evaluate the model
	(AUC, accuracy, PRC AUC, F1...). 

	Parameters
	----------
	cutoff: float, optional
		Probability cutoff.

	Returns
	-------
	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.
		"""
		check_types([("cutoff", cutoff, [int, float], False)])
		if (cutoff > 1 or cutoff < 0):
			cutoff = self.score(method = "best_cutoff")
		return (classification_report(self.y, [self.deploySQL(), self.deploySQL(cutoff)], self.test_relation, self.cursor))
	#---#
	def confusion_matrix(self, 
						 cutoff: float = 0.5):
		"""
	---------------------------------------------------------------------------
	Computes the model confusion matrix.

	Parameters
	----------
	cutoff: float, optional
		Probability cutoff.

	Returns
	-------
	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.
		"""
		check_types([("cutoff", cutoff, [int, float], False)])
		return (confusion_matrix(self.y, self.deploySQL(cutoff), self.test_relation, self.cursor))
	#---#
	def deploySQL(self, 
				  cutoff: float = -1):
		"""
	---------------------------------------------------------------------------
	Returns the SQL code needed to deploy the model. 

	Parameters
	----------
	cutoff: float, optional
		Probability cutoff. If this number is not between 0 and 1, the method 
		will return the probability to be of class 1.

	Returns
	-------
	str/list
 		the SQL code needed to deploy the model.
		"""
		check_types([("cutoff", cutoff, [int, float], False)])
		sql = "PREDICT_SVM_CLASSIFIER({} USING PARAMETERS model_name = '{}', type = 'probability', match_by_pos = 'true')"
		if (cutoff <= 1 and cutoff >= 0):
			sql = "(CASE WHEN {} > {} THEN 1 ELSE 0 END)".format(sql, cutoff)
		return (sql.format(", ".join(self.X), self.name))
	#---#
	def deploy_to_DB(self, 
					 name: str, 
					 view: bool = True, 
					 cutoff: float = -1):
		"""
	---------------------------------------------------------------------------
	Deploys the model in the Vertica DB by creating a relation. 

	Parameters
	----------
	name: str
		Relation name. It must include the schema (the default schema is public).
	view: bool
		If set to false, it will create a table instead of a view.
	cutoff: float, optional
		Probability cutoff. If this number is not between 0 and 1, a column
		corresponding to the probability to be of class 1 will be generated.

	Returns
	-------
	vDataFrame
 		the vDataFrame of the new relation.
		"""
		check_types([
			("name", name, [str], False),
			("view", view, [bool], False),
			("cutoff", cutoff, [int, float], False)])
		relation = "TABLE" if not(view) else "VIEW"
		sql = "CREATE {} {} AS SELECT {}, {} AS {} FROM {}".format(relation, name, ", ".join(self.X), self.deploySQL(cutoff), self.y, self.test_relation)
		self.cursor.execute(sql)
		return vDataFrame(name, self.cursor)
	#---#
	def drop(self):
		"""
	---------------------------------------------------------------------------
	Drops the model from the Vertica DB.
		"""
		drop_model(self.name, self.cursor, print_info = False)
	#---#
	def features_importance(self):
		"""
	---------------------------------------------------------------------------
	Computes the model features importance by normalizing the LinearSVC hyperplan
	coefficients.

	Returns
	-------
	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.
		"""
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
	#---#
	def lift_chart(self):
		"""
	---------------------------------------------------------------------------
	Draws the model Lift Chart.

	Returns
	-------
	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.
		"""
		return (lift_chart(self.y, self.deploySQL(), self.test_relation, self.cursor))
	#---#
	def plot(self, 
			 max_nb_points: int = 50):
		"""
	---------------------------------------------------------------------------
	Draws the LinearSVC if the number of predictors is lesser than 3.

	Parameters
	----------
	max_nb_points: int
		Maximum number of points to display.
		"""
		check_types([("max_nb_points", max_nb_points, [int, float], False)])
		coefficients = self.coef.values["coefficient"]
		svm_classifier_plot(self.X, self.y, self.input_relation, coefficients, self.cursor, max_nb_points)
	#---#
	def prc_curve(self):
		"""
	---------------------------------------------------------------------------
	Draws the model PRC curve.

	Returns
	-------
	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.
		"""
		return (prc_curve(self.y, self.deploySQL(), self.test_relation, self.cursor))
	#---#
	def predict(self,
				vdf,
				name: str = "",
				cutoff: float = 0.5):
		"""
	---------------------------------------------------------------------------
	Adds the prediction in a vDataFrame.

	Parameters
	----------
	vdf: vDataFrame
		Object used to insert the prediction as a vcolumn.
	name: str, optional
		Name of the added vcolumn. If empty, a name will be generated.
	cutoff: float, optional
		Probability cutoff.

	Returns
	-------
	vDataFrame
		the input object.
		"""
		check_types([
			("name", name, [str], False),
			("cutoff", cutoff, [int, float], False)],
			vdf = ["vdf", vdf])
		name = "LinearSVC_" + ''.join(ch for ch in self.name if ch.isalnum()) if not (name) else name
		return (vdf.eval(name, self.deploySQL(cutoff)))
	#---#
	def roc_curve(self):
		"""
	---------------------------------------------------------------------------
	Draws the model ROC curve.

	Returns
	-------
	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.
		"""
		return (roc_curve(self.y, self.deploySQL(), self.test_relation, self.cursor))
	#---#
	def score(self, 
			  cutoff: float = 0.5, 
			  method: str = "accuracy"):
		"""
	---------------------------------------------------------------------------
	Computes the model score.

	Parameters
	----------
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
#---#
class LinearSVR:
	"""
---------------------------------------------------------------------------
Creates a LinearSVR object by using the Vertica Highly Distributed and Scalable 
SVM on the data. This algorithm will find the hyperplan which will approximate
the data distribution.

Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
cursor: DBcursor, optional
	Vertica DB cursor.
tol: float, optional
	Used to control accuracy.
C: float, optional
	The weight for misclassification cost. The algorithm minimizes the 
	regularization cost and the misclassification cost.
fit_intercept: bool, optional
	A bool to fit also the intercept.
intercept_scaling: float
	A float value, serves as the value of a dummy feature whose coefficient 
	Vertica uses to calculate the model intercept. Because the dummy feature 
	is not in the training data, its values are set to a constant, by default 
	set to 1. 
intercept_mode: str, optional
	Specify how to treat the intercept.
		regularized   : Fits the intercept and applies a regularization on it.
		unregularized : Fits the intercept but does not include it in regularization. 
acceptable_error_margin: float, optional
	Defines the acceptable error margin. Any data points outside this region add a 
	penalty to the cost function. 
max_iter: int, optional
	The maximum number of iterations that the algorithm performs.

Attributes
----------
After the object creation, all the parameters become attributes. 
The model will also create extra attributes when fitting the model:

coef: tablesample
	Coefficients and their mathematical information (pvalue, std, value...)
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
				  tol: float = 1e-4, 
				  C: float = 1.0, 
				  fit_intercept: bool = True,
				  intercept_scaling: float = 1.0,
				  intercept_mode: str = "regularized",
				  acceptable_error_margin: float = 0.1,
				  max_iter: int = 100):
		check_types([
			("name", name, [str], False),
			("tol", tol, [int, float], False),
			("C", C, [int, float], False),
			("fit_intercept", fit_intercept, [bool], False),
			("intercept_scaling", intercept_scaling, [int, float], False),
			("intercept_mode", intercept_mode, ["unregularized", "regularized"], True),
			("acceptable_error_margin", acceptable_error_margin, [int, float], False),
			("max_iter", max_iter, [int, float], False)])
		if not(cursor):
			cursor = read_auto_connect().cursor()
		else:
			check_cursor(cursor)
		self.type = "regressor"
		self.cursor = cursor
		self.name = name
		self.tol = tol
		self.C = C 
		self.fit_intercept = fit_intercept 
		self.intercept_scaling = intercept_scaling
		self.intercept_mode = intercept_mode.lower()
		self.acceptable_error_margin = acceptable_error_margin
		self.max_iter = max_iter
	#---#
	def __repr__(self):
		try:
			self.cursor.execute("SELECT GET_MODEL_SUMMARY(USING PARAMETERS model_name = '" + self.name + "')")
			return (self.cursor.fetchone()[0])
		except:
			return "<LinearSVR>"
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
		sql = "PREDICT_SVM_REGRESSOR({} USING PARAMETERS model_name = '{}', match_by_pos = 'true')"
		return (sql.format(", ".join(self.X), self.name))
	#---#
	def drop(self):
		"""
	---------------------------------------------------------------------------
	Drops the model from the Vertica DB.
		"""
		drop_model(self.name, self.cursor, print_info = False)
	#---#
	def features_importance(self):
		"""
	---------------------------------------------------------------------------
	Computes the model features importance by normalizing the LinearSVC hyperplan
	coefficients.

	Returns
	-------
	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.
		"""
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
	#---#
	def plot(self, 
			 max_nb_points: int = 50):
		"""
	---------------------------------------------------------------------------
	Draws the LinearSVR if the number of predictors is lesser than 2.

	Parameters
	----------
	max_nb_points: int
		Maximum number of points to display.
		"""
		check_types([("max_nb_points", max_nb_points, [int, float], False)])
		coefficients = self.coef.values["coefficient"]
		regression_plot(self.X, self.y, self.input_relation, coefficients, self.cursor, max_nb_points)
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
		name = "LinearSVR_" + ''.join(ch for ch in self.name if ch.isalnum()) if not (name) else name
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
	def score(self, method: str = "r2"):
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