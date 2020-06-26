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
class ElasticNet:
	"""
---------------------------------------------------------------------------
Creates a ElasticNet object by using the Vertica Highly Distributed and 
Scalable Linear Regression on the data. The Elastic Net is a regularized 
regression method that linearly combines the L1 and L2 penalties of the 
Lasso and Ridge methods. 

Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
cursor: DBcursor, optional
	Vertica DB cursor.
penalty: str, optional
	Determines the method of regularization.
		None : No Regularization
		L1   : L1 Regularization
		L2   : L2 Regularization
		ENet : Combination between L1 and L2
tol: float, optional
	Determines whether the algorithm has reached the specified accuracy result.
C: float, optional
	The regularization parameter value. The value must be zero or non-negative.
max_iter: int, optional
	Determines the maximum number of iterations the algorithm performs before 
	achieving the specified accuracy result.
solver: str, optional
	The optimizer method used to train the model. 
		Newton : Newton Method
		BFGS   : Broyden Fletcher Goldfarb Shanno
		CGD    : Coordinate Gradient Descent
l1_ratio: float, optional
	ENet mixture parameter that defines how much L1 versus L2 regularization 
	to provide. 

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
				  penalty: str = 'ENet', 
				  tol: float = 1e-4, 
				  C: float = 1.0, 
				  max_iter: int = 100, 
				  solver: str = 'CGD', 
				  l1_ratio: float = 0.5):
		check_types([
			("name", name, [str], False),
			("solver", solver, ['newton', 'bfgs', 'cgd'], True),
			("tol", tol, [int, float], False),
			("C", C, [int, float], False),
			("max_iter", max_iter, [int, float], False),
			("penalty", penalty, ['enet', 'l1', 'l2', 'none'], True),
			("l1_ratio", l1_ratio, [int, float], False)])
		if not(cursor):
			cursor = read_auto_connect().cursor()
		else:
			check_cursor(cursor)
		self.type = "regressor"
		self.cursor = cursor
		self.name = name
		self.penalty = penalty.lower()
		self.tol = tol
		self.C = C 
		self.max_iter = max_iter
		self.solver = solver.lower()
		self.l1_ratio = l1_ratio
	#---#
	def __repr__(self):
		try:
			self.cursor.execute("SELECT GET_MODEL_SUMMARY(USING PARAMETERS model_name = '" + self.name + "')")
			return (self.cursor.fetchone()[0])
		except:
			return "<LinearRegression>"
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
		sql = "PREDICT_LINEAR_REG({} USING PARAMETERS model_name = '{}', match_by_pos = 'true')"
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
	Computes the model features importance by normalizing the Linear Regression
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
		query = "SELECT LINEAR_REG('{}', '{}', '{}', '{}' USING PARAMETERS optimizer = '{}', epsilon = {}, max_iterations = {}"
		query = query.format(self.name, input_relation, self.y, ", ".join(self.X), self.solver, self.tol, self.max_iter)
		query += ", regularization = '{}', lambda = {}".format(self.penalty, self.C)
		if (self.penalty == 'ENet'):
			query += ", alpha = {}".format(self.l1_ratio)
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
	Draws the Linear Regression if the number of predictors is equal to 1 or 2.

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
		name = "LinearRegression_" + ''.join(ch for ch in self.name if ch.isalnum()) if not (name) else name
		return (vdf.eval(name, self.deploySQL()))
	#---#
	def regression_report(self):
		return (regression_report(self.y, self.deploySQL(), self.test_relation, self.cursor))
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
#---#
def Lasso(name: str,
		  cursor = None,
		  tol: float = 1e-4, 
		  max_iter: int = 100, 
		  solver: str = 'CGD'):
	"""
---------------------------------------------------------------------------
Creates a Lasso object by using the Vertica Highly Distributed and Scalable 
Linear Regression on the data. The Lasso is a regularized regression method 
which uses L1 penalty. 

Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
cursor: DBcursor, optional
	Vertica DB cursor.
tol: float, optional
	Determines whether the algorithm has reached the specified accuracy result.
max_iter: int, optional
	Determines the maximum number of iterations the algorithm performs before 
	achieving the specified accuracy result.
solver: str, optional
	The optimizer method used to train the model. 
		Newton : Newton Method
		BFGS   : Broyden Fletcher Goldfarb Shanno
		CGD    : Coordinate Gradient Descent
	"""
	return ElasticNet(name = name,
		  		 	  cursor = cursor,
		  			  penalty = 'L1', 
		  			  tol = tol, 
		  			  max_iter = max_iter, 
		  			  solver = solver)
#---#
def LinearRegression(name: str,
		  			 cursor = None,
		  			 tol: float = 1e-4,
		  			 max_iter: int = 100, 
		  			 solver: str = 'Newton'):
	"""
---------------------------------------------------------------------------
Creates a LinearRegression object by using the Vertica Highly Distributed and 
Scalable Linear Regression on the data. 

Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
cursor: DBcursor, optional
	Vertica DB cursor.
tol: float, optional
	Determines whether the algorithm has reached the specified accuracy result.
max_iter: int, optional
	Determines the maximum number of iterations the algorithm performs before 
	achieving the specified accuracy result.
solver: str, optional
	The optimizer method used to train the model. 
		Newton : Newton Method
		BFGS   : Broyden Fletcher Goldfarb Shanno
		CGD    : Coordinate Gradient Descent

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
	return ElasticNet(name = name,
		  		 	  cursor = cursor,
		  			  penalty = 'None', 
		  			  tol = tol, 
		  			  max_iter = max_iter, 
		  			  solver = solver)
#---#
class LogisticRegression:
	"""
---------------------------------------------------------------------------
Creates a LogisticRegression object by using the Vertica Highly Distributed 
and Scalable Logistic Regression on the data.

Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
cursor: DBcursor, optional
	Vertica DB cursor.
penalty: str, optional
	Determines the method of regularization.
		None : No Regularization
		L1   : L1 Regularization
		L2   : L2 Regularization
		ENet : Combination between L1 and L2
tol: float, optional
	Determines whether the algorithm has reached the specified accuracy result.
C: float, optional
	The regularization parameter value. The value must be zero or non-negative.
max_iter: int, optional
	Determines the maximum number of iterations the algorithm performs before 
	achieving the specified accuracy result.
solver: str, optional
	The optimizer method used to train the model. 
		Newton : Newton Method
		BFGS   : Broyden Fletcher Goldfarb Shanno
		CGD    : Coordinate Gradient Descent
l1_ratio: float, optional
	ENet mixture parameter that defines how much L1 versus L2 regularization 
	to provide. 

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
				  penalty: str = 'L2', 
				  tol: float = 1e-4, 
				  C: int = 1, 
				  max_iter: int = 100, 
				  solver: str = 'CGD', 
				  l1_ratio: float = 0.5):
		check_types([
			("name", name, [str], False),
			("solver", solver, ['newton', 'bfgs', 'cgd'], True),
			("tol", tol, [int, float], False),
			("C", C, [int, float], False),
			("max_iter", max_iter, [int, float], False),
			("penalty", penalty, ['enet', 'l1', 'l2', 'none'], True),
			("l1_ratio", l1_ratio, [int, float], False)])
		if not(cursor):
			cursor = read_auto_connect().cursor()
		else:
			check_cursor(cursor)
		self.type = "classifier"
		self.classes = [0, 1]
		self.cursor = cursor
		self.name = name
		self.penalty = penalty.lower()
		self.tol = tol
		self.C = C 
		self.max_iter = max_iter
		self.solver = solver.lower()
		self.l1_ratio = l1_ratio
	#---#
	def __repr__(self):
		try:
			self.cursor.execute("SELECT GET_MODEL_SUMMARY(USING PARAMETERS model_name = '" + self.name + "')")
			return (self.cursor.fetchone()[0])
		except:
			return "<LogisticRegression>"
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
		if (cutoff <= 1 and cutoff >= 0):
			sql = "PREDICT_LOGISTIC_REG({} USING PARAMETERS model_name = '{}', cutoff = " + str(cutoff) + ", match_by_pos = 'true')"
		else:
			sql = "PREDICT_LOGISTIC_REG({} USING PARAMETERS model_name = '{}', type = 'probability', match_by_pos = 'true')"
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
	Computes the model features importance by normalizing the Logistic Regression
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
	Train the model.

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
		query = "SELECT LOGISTIC_REG('{}', '{}', '{}', '{}' USING PARAMETERS optimizer = '{}', epsilon = {}, max_iterations = {}"
		query = query.format(self.name, input_relation, self.y, ", ".join(self.X), self.solver, self.tol, self.max_iter)
		query += ", regularization = '{}', lambda = {}".format(self.penalty, self.C)
		if (self.penalty == 'ENet'):
			query += ", alpha = {}".format(self.l1_ratio)
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
	Draws the Logistic Regression if the number of predictors is equal to 1 or 2.

	Parameters
	----------
	max_nb_points: int
		Maximum number of points to display.
		"""
		check_types([("max_nb_points", max_nb_points, [int, float], False)])
		coefficients = self.coef.values["coefficient"]
		logit_plot(self.X, self.y, self.input_relation, coefficients, self.cursor, max_nb_points)
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
				cutoff: float = -1):
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
		name = "LogisticRegression_" + ''.join(ch for ch in self.name if ch.isalnum()) if not (name) else name
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
def Ridge(name: str,
		  cursor = None,
		  tol: float = 1e-4, 
		  max_iter: int = 100, 
		  solver: str = 'Newton'):
	"""
---------------------------------------------------------------------------
Creates a Ridge object by using the Vertica Highly Distributed and Scalable 
Linear Regression on the data. The Ridge is a regularized regression method 
which uses L2 penalty. 

Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
cursor: DBcursor, optional
	Vertica DB cursor.
tol: float, optional
	Determines whether the algorithm has reached the specified accuracy result.
max_iter: int, optional
	Determines the maximum number of iterations the algorithm performs before 
	achieving the specified accuracy result.
solver: str, optional
	The optimizer method used to train the model. 
		Newton : Newton Method
		BFGS   : Broyden Fletcher Goldfarb Shanno
		CGD    : Coordinate Gradient Descent
	"""
	return ElasticNet(name = name,
		  		 	  cursor = cursor,
		  			  penalty = 'L2', 
		  			  tol = tol, 
		  			  max_iter = max_iter, 
		  			  solver = solver)