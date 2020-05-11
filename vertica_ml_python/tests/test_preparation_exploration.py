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
# 
# Modules
#
# Standard Python Modules
import random
# Other Python Modules
import matplotlib.pyplot as plt
# Vertica ML Python Modules
from vertica_ml_python import vDataframe
#
#---#
class ExplorationPreparationTest:
	#---#
	def  __init__(self, vdf):
		self.vdf = vdf
	#---#
	def test_stats(self):
		numcols = self.shuffle_columns(self.numcol(), get_n_cols = 10)
		columns = self.shuffle_columns(self.allcol(), get_n_cols = 10)
		functions = ["APPROXIMATE_COUNT_DISTINCT", "APPROXIMATE_MEDIAN", "AVG", "COUNT", "MAX", "MIN", "STDDEV", "SUM", "VARIANCE", "SEM", "PROD", "PERCENT"]
		print("Current Relation = '{}'\n".format(self.vdf._VERTICA_ML_PYTHON_VARIABLES_["main_relation"]))
		print("Test Statistics\n--------------\nParameters\n----------\ncolumns = {}\nnumcols = {}".format(columns, numcols))
		print("Test Agg")
		self.vdf.aggregate(functions, numcols)
		for col in numcols:
			self.vdf[col].aggregate(functions)
		print("Success\nTest Statistics")
		self.vdf.statistics(numcols)
		print("Success\nTest Describe")
		self.vdf.describe(method = "numerical", columns = numcols)
		self.vdf.describe(method = "categorical", columns = columns)
		for col in columns:
			self.vdf[col].describe()
		print("Success\nTest Shape")
		self.vdf.shape()
		print("Success\nTest Memory Usage")
		self.vdf.memory_usage()
		print("Success\nTest Expected Store Usage")
		self.vdf.expected_store_usage()
		print("Success\nTest Duplicated")
		self.vdf.duplicated(columns = columns)
		print("Success")
	#---#
	def test_preparation(self):
		numcols = self.shuffle_columns(self.numcol(), get_n_cols = 10)
		columns = self.shuffle_columns(self.allcol(), get_n_cols = 10)
		bar_type = self.shuffle_columns(["auto", "fully_stacked", "stacked", "fully", "fully stacked"], get_n_cols = 1)[0]
		functions = ["APPROXIMATE_COUNT_DISTINCT", "APPROXIMATE_MEDIAN", "AVG", "COUNT", "MAX", "MIN", "STDDEV", "SUM", "VARIANCE", "SEM", "PROD", "PERCENT"]
		print("Current Relation = '{}'\n".format(self.vdf._VERTICA_ML_PYTHON_VARIABLES_["main_relation"]))
		print("Test Statistics\n--------------\nParameters\n----------\ncolumns = {}\nnumcols = {}".format(columns, numcols))
		print("Test Sort")
		self.vdf.sort(columns = numcols)
		print("Success\nTest Get Dummies")
		self.vdf.get_dummies(columns = columns)
		print("Success\nTest Normalize")
		for column in numcols:
			method = self.shuffle_columns(["minmax", "robust_zscore", "zscore"], get_n_cols = 1)[0]
			by = self.shuffle_columns(self.catcol(), get_n_cols = 1)
			self.vdf[column].normalize(method = method, by = by)
		print("Success\nTest Fill NAs")
		for column in numcols:
			method = self.shuffle_columns(["auto", "mode", "0ifnull", 'mean', 'avg', 'median', 'ffill', 'pad', 'bfill', 'backfill'], get_n_cols = 1)[0]
			by = self.shuffle_columns(self.catcol(), get_n_cols = random.randint(0, 3))
			order_by = self.shuffle_columns(self.catcol(), get_n_cols = random.randint(0, 2))
			self.vdf[column].fillna(method = method, by = by, order_by = order_by)
		print("Success\nTest Drop NAs")
		self.vdf.dropna(columns = columns)
		print("Success\nTest Drop Outliers")
		for col in numcols:
			self.vdf[col].drop_outliers()
		print("Success")
	#
	#
	#---#
	def allcol(self):
		return self.vdf.get_columns()
	#---#
	def catcol(self):
		all_cols = self.vdf.get_columns()
		numcol = self.vdf.numcol()
		for col in numcol:
			all_cols.remove(col)
		return all_cols
	#---#
	def numcol(self):
		return self.vdf.numcol()
	#---#
	def shuffle_columns(self, columns: list = [], get_n_cols: int = -1):
		for k in range(3):
			random.shuffle(columns)
		if (get_n_cols > 0) and (len(columns) > get_n_cols):
				columns = columns[0:get_n_cols]
		return columns