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
#             /           `\     /     /
#            |   O         /    /     /
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
#
#
# \  / _  __|_. _ _   |\/||   |~)_|_|_  _  _ 
#  \/ (/_|  | |(_(_|  |  ||_  |~\/| | |(_)| |
#                               /            
# Vertica-ML-Python allows user to create vDataFrames (Virtual Dataframes). 
# vDataFrames simplify data exploration, data cleaning and MACHINE LEARNING     
# in VERTICA. It is an object which keeps in it all the actions that the user 
# wants to achieve and execute them when they are needed.    										
#																					
# The purpose is to bring the logic to the data and not the opposite !
#
# Modules
#
# Standard Python Modules
import random
# Other Python Modules
import matplotlib.pyplot as plt
# Vertica ML Python Modules
from vertica_ml_python.learn.linear_model import *
from vertica_ml_python.learn.ensemble import *
from vertica_ml_python.learn.tree import *
from vertica_ml_python.learn.neighbors import *
from vertica_ml_python.learn.svm import *
from vertica_ml_python.learn.naive_bayes import *
from vertica_ml_python.learn.cluster import *
from vertica_ml_python.learn.decomposition import *
from vertica_ml_python.learn.datasets import *
from vertica_ml_python import vDataframe
#
#---#
class MLTest:
	#---#
	def  __init__(self, 
				  input_relation: str, 
				  X: list, 
				  y: str, 
				  cursor, 
				  schema: str = "public"):
		self.input_relation = input_relation
		self.X = X
		self.y = y
		self.cursor = cursor
		self.schema = schema
	#---#
	def test_regression(self):
		vdf = vDataframe(self.input_relation, self.cursor)
		name = '"{}".VERTICA_ML_PYTHON_MODEL_TEST_REGRESSION'.format(self.schema)
		for i in range(9):
			self.cursor.execute("DROP MODEL IF EXISTS {}".format(name))
			if (i == 0):
				print("Test Elastic Net Creation")
				model = ElasticNet(name, self.cursor)
			elif (i == 1):
				print("Test Linear Regression Creation")
				model = LinearRegression(name, self.cursor)
			elif (i == 2):
				print("Test Lasso Creation")
				model = Lasso(name, self.cursor)
			elif (i == 3):
				print("Test Ridge Creation")
				model = Ridge(name, self.cursor)
			elif (i == 4):
				print("Test Decision Tree Regressor Creation")
				model = DecisionTreeRegressor(name, self.cursor)
			elif (i == 5):
				print("Test Dummy Tree Regressor Creation")
				model = DummyTreeRegressor(name, self.cursor)
			elif (i == 6):
				print("Test Random Forest Regressor Creation")
				model = RandomForestRegressor(name, self.cursor)
			elif (i == 7):
				print("Test K Neighbors Regressor Creation")
				model = KNeighborsRegressor(self.cursor)
			elif (i == 8):
				print("Test Linear SVR Creation")
				model = LinearSVR(name, self.cursor)
			model.fit(self.input_relation, self.X, self.y)
			if (i != 7):
				print("Success\nTest Features Importance")
				model.features_importance()
				plt.close()
				print("Success\nTest Add to vdf")
				model.add_to_vdf(vdf, name = "VERTICA_ML_PYTHON_MODEL_TEST_{}".format(i))
			if (len(self.X) <= 2) and (i not in [4, 5, 6, 7]):
				print("Success\nTest Plot")
				model.plot()
				plt.close()
			print("Success\nTest Regression Report")
			model.regression_report()
			for metric in ["r2", "mae", "mse", "msle", "max", "median", "var"]:
				print("Success\nTest Metric ({})".format(metric))
				model.score(metric)
			print("Success")
			print("--------------------")
			self.cursor.execute("DROP MODEL IF EXISTS {}".format(name))
	#---#
	def test_binary_classification(self):
		vdf = vDataframe(self.input_relation, self.cursor)
		name = '"{}".VERTICA_ML_PYTHON_MODEL_TEST_CLASSIFICATION'.format(self.schema)
		for i in range(8):
			self.cursor.execute("DROP MODEL IF EXISTS {}".format(name))
			if (i == 0):
				print("Test Logistic Regression Creation")
				model = LogisticRegression(name, self.cursor)
			elif (i == 1):
				print("Test Linear SVC Creation")
				model = LinearSVC(name, self.cursor)
			elif (i == 2):
				print("Test MultinomialNB Creation")
				model = MultinomialNB(name, self.cursor)
			elif (i == 3):
				print("Test Nearest Centroid Creation")
				model = NearestCentroid(self.cursor)
			elif (i == 4):
				print("Test Decision Tree Classifier Creation")
				model = DecisionTreeClassifier(name, self.cursor)
			elif (i == 5):
				print("Test Dummy Tree Classifier Creation")
				model = DummyTreeClassifier(name, self.cursor)
			elif (i == 6):
				print("Test Random Forest Classifier Creation")
				model = RandomForestClassifier(name, self.cursor)
			elif (i == 7):
				print("Test K Neighbors Classifier Creation")
				model = KNeighborsClassifier(self.cursor)
			model.fit(self.input_relation, self.X, self.y)
			if (i not in [2, 3, 7]):
				print("Success\nTest Features Importance")
				model.features_importance()
				plt.close()
				print("Success\nTest Add to vdf")
				model.add_to_vdf(vdf, name = "VERTICA_ML_PYTHON_MODEL_TEST_{}".format(i))
			if (len(self.X) <= 2 and i == 0) or (len(self.X) <= 3 and i == 1):
				print("Success\nTest Plot")
				model.plot()
				plt.close()
			print("Success\nTest Classification Report")
			model.classification_report()
			print("Success\nTest ROC")
			model.roc_curve()
			plt.close()
			print("Success\nTest PRC")
			model.prc_curve()
			plt.close()
			print("Success\nTest Lift Chart")
			model.lift_chart()
			plt.close()
			print("Success\nTest Confusion Matrix")
			model.confusion_matrix()
			for metric in ["accuracy", 
						   "auc", 
						   "prc_auc", 
						   "best_cutoff", 
						   "recall", "precision", 
						   "log_loss", "negative_predictive_value", 
						   "specificity", 
						   "mcc", 
						   "informedness", 
						   "markedness", 
						   "critical_success_index"]:
				print("Success\nTest Metric ({})".format(metric))
				model.score(method = metric)
			print("Success")
			print("--------------------")
			self.cursor.execute("DROP MODEL IF EXISTS {}".format(name))
	#---#
	def test_multi_classification(self):
		vdf = vDataframe(self.input_relation, self.cursor)
		name = '"{}".VERTICA_ML_PYTHON_MODEL_TEST_MULTI_CLASSIFICATION'.format(self.schema)
		for i in range(6):
			self.cursor.execute("DROP MODEL IF EXISTS {}".format(name))
			if (i == 0):
				print("Test MultinomialNB Creation")
				model = MultinomialNB(name, self.cursor)
			elif (i == 1):
				print("Test Nearest Centroid Creation")
				model = NearestCentroid(self.cursor)
			elif (i == 2):
				print("Test Decision Tree Classifier Creation")
				model = DecisionTreeClassifier(name, self.cursor)
			elif (i == 3):
				print("Test Dummy Tree Classifier Creation")
				model = DummyTreeClassifier(name, self.cursor)
			elif (i == 4):
				print("Test Random Forest Classifier Creation")
				model = RandomForestClassifier(name, self.cursor)
			elif (i == 5):
				print("Test K Neighbors Classifier Creation")
				model = KNeighborsClassifier(self.cursor)
			model.fit(self.input_relation, self.X, self.y)
			if (i not in [0, 1, 5]):
				print("Success\nTest Features Importance")
				model.features_importance()
				plt.close()
				print("Success\nTest Add to vdf")
				model.add_to_vdf(vdf, name = "VERTICA_ML_PYTHON_MODEL_TEST_{}".format(i))
			print("Success\nTest Classification Report")
			model.classification_report()
			print("Success\nTest ROC")
			model.roc_curve(pos_label = model.classes[1])
			plt.close()
			print("Success\nTest PRC")
			model.prc_curve(pos_label = model.classes[1])
			plt.close()
			print("Success\nTest Lift Chart")
			model.lift_chart(pos_label = model.classes[1])
			plt.close()
			print("Success\nTest Confusion Matrix")
			model.confusion_matrix()
			for metric in ["accuracy", 
						   "auc", 
						   "prc_auc", 
						   "best_cutoff", 
						   "recall", "precision", 
						   "log_loss", "negative_predictive_value", 
						   "specificity", 
						   "mcc", 
						   "informedness", 
						   "markedness", 
						   "critical_success_index"]:
				print("Success\nTest Metric ({})".format(metric))
				model.score(pos_label = model.classes[1], method = metric)
			print("Success")
			print("--------------------")
			self.cursor.execute("DROP MODEL IF EXISTS {}".format(name))
	#---#
	def test_unsupervised(self):
		vdf = vDataframe(self.input_relation, self.cursor)
		name = '"{}".VERTICA_ML_PYTHON_MODEL_TEST_UNSUPERVISED'.format(self.schema)
		self.cursor.execute("DROP TABLE IF EXISTS {}".format(name))
		self.cursor.execute("DROP MODEL IF EXISTS {}".format(name))
		for i in range(6):
			if (i == 0):
				print("Test KMeans Creation")
				model = KMeans(name, self.cursor)
			elif (i == 1):
				print("Test Local Outlier Factor Creation")
				model = LocalOutlierFactor(name, self.cursor)
			elif (i == 2):
				print("Test DBSCAN Creation")
				model = DBSCAN(name, self.cursor)
			elif (i == 3):
				print("Test PCA Creation")
				model = PCA(name, self.cursor)
			elif (i == 4):
				print("Test SVD Creation")
				model = SVD(name, self.cursor)
			model.fit(self.input_relation, self.X)
			if (len(self.X) <= 3 and i < 3):
				print("Success\nTest Plot")
				model.plot()
				plt.close()
			if (i != 0):
				print("Success\nTest to vdf")
				model.to_vdf()
			else:
				print("Success\nTest Add to vdf")
				model.add_to_vdf(vdf)
			print("Success")
			print("--------------------")
			self.cursor.execute("DROP TABLE IF EXISTS {}".format(name))
			self.cursor.execute("DROP MODEL IF EXISTS {}".format(name))
#---#
def DataTest(cursor, schema: str = "public"):
	name = "{}.VERTICA_ML_PYTHON_LOAD_TEST".format(schema)
	cursor.execute("DROP TABLE IF EXISTS {}".format(name))
	print("Test Load Titanic")
	load_titanic(cursor, name = name)
	print("Success")
	cursor.execute("DROP TABLE IF EXISTS {}".format(name))
	print("Test Load Amazon")
	load_amazon(cursor, name = name)
	print("Success")
	cursor.execute("DROP TABLE IF EXISTS {}".format(name))
	print("Test Load WineQuality")
	load_winequality(cursor, name = name)
	print("Success")
	cursor.execute("DROP TABLE IF EXISTS {}".format(name))
	print("Test Load Iris")
	load_iris(cursor, name = name)
	print("Success")
	cursor.execute("DROP TABLE IF EXISTS {}".format(name))
	print("Test Load Smart Meters")
	load_smart_meters(cursor, name = name)
	print("Success")
	cursor.execute("DROP TABLE IF EXISTS {}".format(name))