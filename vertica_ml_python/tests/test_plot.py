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
from vertica_ml_python import vDataframe
#
#---#
class PlotTest:
	#---#
	def  __init__(self, vdf):
		self.vdf = vdf
	#---#
	def test_1D(self):
		method = self.shuffle_columns(["density", "count", "avg", "min", "max", "sum"], get_n_cols = 1)[0]
		of = self.shuffle_columns(self.numcol(), get_n_cols = 1)[0]
		col = self.shuffle_columns(self.allcol(), get_n_cols = 2)
		try:
			columns.remove(of)
		except:
			pass
		col = col[0] if col[0] != of else col[1]
		catcol = self.shuffle_columns(self.catcol(), get_n_cols = 1)[0]
		numcol = self.shuffle_columns(self.numcol(), get_n_cols = 1)[0]
		max_cardinality = random.randint(4, 1000)
		kernel = self.shuffle_columns(["gaussian", "logistic", "sigmoid", "silverman"], get_n_cols = 1)[0]
		print("Current Relation = '{}'\n".format(self.vdf._VERTICA_ML_PYTHON_VARIABLES_["main_relation"]))
		print("Test Charts 1D\n--------------\nParameters\n----------\ncolumn = '{}'\ncatcol = '{}'\nnumcol = '{}'\nmethod = '{}'\nof = '{}'\nmax_cardinality = '{}'\nkernel = '{}'".format(col, catcol, numcol, method, of, max_cardinality, kernel))
		print("Test Bar")
		self.vdf[col].bar(method = method,
					 	  of = of,
					 	  max_cardinality = max_cardinality)
		plt.close()
		print("Success\nTest Hist")
		self.vdf[col].hist(method = method,
					 	   of = of,
					 	   max_cardinality = max_cardinality)
		plt.close()
		print("Success\nTest Pie/Donut")
		self.vdf[col].pie(method = method,
					 	  of = of,
					 	  max_cardinality = max_cardinality)
		plt.close()
		print("Success\nTest Density")
		self.vdf[numcol].density(kernel = kernel)
		plt.close()
		print("Success\nTest BoxPlot")
		self.vdf[numcol].boxplot(by = catcol, max_cardinality = max_cardinality)
		plt.close()
		print("Success")
	#---#
	def test_2D(self):
		method = self.shuffle_columns(["density", "count", "avg", "min", "max", "sum"], get_n_cols = 1)[0]
		bar_type = self.shuffle_columns(["auto", "fully_stacked", "stacked", "fully", "fully stacked"], get_n_cols = 1)[0]
		hist_type = self.shuffle_columns(["auto", "stacked"], get_n_cols = 1)[0]
		of = self.shuffle_columns(self.numcol(), get_n_cols = 1)[0]
		numcols = self.shuffle_columns(self.numcol(), get_n_cols = 4)
		columns = self.allcol()
		catcol = self.shuffle_columns(self.catcol(), get_n_cols = 1)[0]
		try:
			columns.remove(of)
			numcols.remove(of)
		except:
			pass
		columns = self.shuffle_columns(columns, get_n_cols = 2)
		print("Current Relation = '{}'\n".format(self.vdf._VERTICA_ML_PYTHON_VARIABLES_["main_relation"]))
		print("Test Charts 2D\n--------------\nParameters\n----------\ncolumns = {}\ncatcol = '{}'\nnumcols = {}\nmethod = '{}'\nof = '{}'\nbar_type = '{}'\nhist_type = '{}'".format(columns, catcol, numcols, method, of, bar_type, hist_type))
		print("Test Bar")
		self.vdf.bar(columns = columns, 
					 method = method,
					 of = of,
					 max_cardinality = (6, 6),
					 h = (None, None),
					 hist_type = bar_type)
		print("Success\nTest Hist")
		self.vdf.hist(columns = columns, 
					  method = method,
					  of = of,
					  max_cardinality = (6, 6),
					  h = (None, None),
					  hist_type = hist_type)
		print("Success\nTest Multiple Hist")
		self.vdf.hist(columns = numcols, 
					  method = method,
					  of = of,
					  max_cardinality = (6, 6),
					  h = (None, None),
					  hist_type = "multi")
		print("Success\nTest Hexbin")
		self.vdf.hexbin(columns = numcols[0:2], 
					    method = method,
					    of = of)
		print("Success\nTest Pivot")
		self.vdf.pivot_table(columns = columns,
							 method = method,
							 of = of,
							 h = (None, None),
							 max_cardinality = (20, 20))
		print("Success\nTest Scatter Matrix")
		self.vdf.scatter_matrix(numcols)
		plt.close()
		print("Success\nTest Scatter Plot 2D")
		self.vdf.scatter(numcols[0:2])
		plt.close()
		print("Success\nTest Scatter Plot 2D with Category")
		self.vdf.scatter(numcols[0:2], catcol = catcol)
		plt.close()
		print("Success\nTest Scatter Plot 3D")
		self.vdf.scatter(numcols[0:3])
		plt.close()
		print("Success\nTest Scatter Plot 3D with Category")
		self.vdf.scatter(numcols[0:3], catcol = catcol)
		plt.close()
		print("Success\nBoxPlot")
		self.vdf.boxplot(numcols)
		plt.close()
		print("Success")
	#---#
	def test_matrix(self):
		numcol = self.shuffle_columns(self.numcol(), get_n_cols = 10)
		catcol = self.shuffle_columns(self.catcol(), get_n_cols = 10)
		focus_num = self.shuffle_columns(numcol, get_n_cols = 1)[0]
		focus_cat = self.shuffle_columns(catcol, get_n_cols = 1)[0]
		methods = ["spearman", "kendall", "pearson", "biserial"]
		print("Current Relation = '{}'\n".format(self.vdf._VERTICA_ML_PYTHON_VARIABLES_["main_relation"]))
		print("Test Matrix\n-----------\nParameters\n----------\nnumcol = {}\ncatcol = {}\nfocus_num = '{}'\nfocus_cat = '{}'".format(numcol, catcol, focus_num, focus_cat))
		for method in methods:
			print("Test Corr {}".format(method))
			self.vdf.corr(columns = numcol, method = method)
			plt.close()
			self.vdf.corr(columns = numcol, method = method, focus = focus_num)
			plt.close()
			print("Success")
		print("Test Cov")
		self.vdf.cov(columns = numcol)
		plt.close()
		self.vdf.cov(columns = numcol, focus = focus_num)
		plt.close()
		print("Success\nTest Corr cramer")
		self.vdf.corr(columns = catcol, method = "cramer")
		plt.close()
		self.vdf.corr(columns = catcol, focus = focus_cat, method = "cramer")
		plt.close()
		print("Success\nTest Regr")
		self.vdf.regr(columns = numcol, method = "slope")
		plt.close()
		self.vdf.regr(columns = numcol, method = "intercept")
		plt.close()
		print("Success")
	#---#
	def test_TS(self):
		date_cols = self.allcol()
		for col in date_cols:
			if (self.vdf[col].category() not in ("date", "float")):
				date_cols.remove(col)
		ts =  self.shuffle_columns(date_cols, get_n_cols = 1)[0]
		columns = self.shuffle_columns(self.numcol(), get_n_cols = 10)
		catcol = self.shuffle_columns(self.catcol(), get_n_cols = 1)[0]
		try:
			columns.remove(of)
			columns.remove(ts)
		except:
			pass
		print("Current Relation = '{}'\n".format(self.vdf._VERTICA_ML_PYTHON_VARIABLES_["main_relation"]))
		print("Test TS Plot\n------------\nParameters\n----------\ncolumns = {}\ncatcol = '{}'\nts = '{}'".format(columns, catcol, ts))
		print("Test TS 2D")
		self.vdf.plot(ts = ts,
					  columns = columns)
		plt.close()
		print("Success\nTest TS 1D")
		self.vdf[columns[0]].plot(ts = ts, 
					  			  by = catcol)
		plt.close()
		print("Success")
		plt.close()
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