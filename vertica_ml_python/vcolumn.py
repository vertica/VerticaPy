# (c) Copyright [2018] Micro Focus or one of its affiliates. 
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
import numpy as np
import math
import time
import datetime
from vertica_ml_python.utilities import tablesample
from vertica_ml_python.utilities import to_tablesample
from vertica_ml_python.utilities import category_from_type
##
#
#   __   __   ______     ______     __         __  __     __    __     __   __    
#  /\ \ / /  /\  ___\   /\  __ \   /\ \       /\ \/\ \   /\ "-./  \   /\ "-.\ \   
#  \ \ \'/   \ \ \____  \ \ \/\ \  \ \ \____  \ \ \_\ \  \ \ \-./\ \  \ \ \-.  \  
#   \ \__|    \ \_____\  \ \_____\  \ \_____\  \ \_____\  \ \_\ \ \_\  \ \_\\"\_\ 
#    \/_/      \/_____/   \/_____/   \/_____/   \/_____/   \/_/  \/_/   \/_/ \/_/ 
#                                                                                
#
##
class vColumn:
	#
	def  __init__(self, alias: str, transformations = None, parent = None):
		self.parent, self.alias = parent, alias
		if (transformations == None):
			# COLUMN DATA TYPE
			query = "(SELECT data_type FROM columns WHERE table_name='{}' AND column_name='{}')".format(self.parent.input_relation, self.alias.replace('"', ''))
			query += " UNION (SELECT data_type FROM view_columns WHERE table_name='{}' AND column_name='{}')".format(self.parent.input_relation, self.alias.replace('"', ''))
			result = self.parent.cursor.execute(query).fetchone()
			ctype = str(result[0]) if (result) else "undefined"
			category = category_from_type(ctype)
			# TRANSFORMATIONS
			self.transformations = [(alias, ctype, category)]
		else:
			self.transformations = transformations
	#
	def __repr__(self):
		return self.head(limit = 5).__repr__()
	# 
	def __setattr__(self,attr,val):
		self.__dict__[attr] = val
	#
	#
	#
	# METHODS
	#  
	def abs(self):
		return (self.apply(func = "ABS({})"))
	# 
	def add(self, x: float):
		if (self.isdate()):
			return (self.apply(func = "TIMESTAMPADD(SECOND, {}, {})".format(x, "{}")))
		else:
			return (self.apply(func = "{} + ({})".format("{}", x)))
	#
	def add_copy(self, name: str):
		if not(name):
			raise ValueError("The parameter 'name' must not be empty")
		name = '"' + name.replace('"', '') + '"'
		if (name in self.parent.get_columns()):
			raise ValueError("The column '{}' already exist".format(name))
		new_vColumn = vColumn(name, parent = self.parent, transformations = [item for item in self.transformations])
		setattr(self.parent, name, new_vColumn)
		setattr(self.parent, name[1:-1], new_vColumn)
		self.parent.columns += [name]
		self.parent.history += ["{" + time.strftime("%c") + "} " + "[ADD COPY]: A copy of the vColumn '{}' named '{}' was added to the vDataframe.".format(self.alias, name)]
		return (new_vColumn)
	# 
	def add_prefix(self, prefix: str):
		return (self.apply(func = "'{}' || {}".format(prefix, "{}")))
	#
	def add_suffix(self, suffix: str):
		return (self.apply(func = "{} || '{}'".format("{}", suffix)))
	#
	def agg(self, func: list):
		return (self.aggregate(func = func))
	def aggregate(self, func: list):
		return (self.parent.aggregate(func = func, columns = [self.alias]).transpose())
	#
	def apply(self, func: str, copy: bool = False, copy_name: str = ""):
		try:
			self.executeSQL(query = "DROP TABLE IF EXISTS _vpython_apply_test_; CREATE TEMPORARY TABLE _vpython_apply_test_ AS SELECT {} FROM {} WHERE {} IS NOT NULL LIMIT 10;".format(func.replace("{}", self.alias), self.parent.genSQL(), self.alias), title = "TEST FUNC {}".format(func))
			self.executeSQL(query = "SELECT data_type FROM columns where table_name = '_vpython_apply_test_'", title = "SELECT NEW DATA TYPE")
			ctype = self.parent.cursor.fetchone()[0]
			self.executeSQL(query = "DROP TABLE IF EXISTS _vpython_apply_test_;", title = "DROP TEMPORARY TABLE")
			category = category_from_type(ctype = ctype)
			if (copy):
				self.add_copy(name = copy_name)
				self.parent[copy_name].transformations += [(func, ctype, category)]
				self.parent.history += ["{" + time.strftime("%c") + "} " + "[{}]: The vColumn '{}' was transformed with the func 'x -> {}'.".format(func.replace("{}", ""), copy_name.replace('"', ''), func.replace("{}", "x"))]
			else:
				self.transformations += [(func, ctype, category)]
				self.parent.history += ["{" + time.strftime("%c") + "} " + "[{}]: The vColumn '{}' was transformed with the func 'x -> {}'.".format(func.replace("{}", ""), self.alias.replace('"', ''), func.replace("{}", "x"))]
			return (self.parent)
		except:
			raise Exception("Error when applying the func 'x -> {}' to '{}'".format(func.replace("{}", "x"), self.alias.replace('"', '')))
	# 
	def astype(self, dtype: str):
		try:
			query = "SELECT {}::{} AS {} FROM {} WHERE {} IS NOT NULL LIMIT 20".format(
				self.alias, dtype, self.alias, self.parent.genSQL(), self.alias)
			self.executeSQL(query, title = "Data Type Conversion - TEST")
			self.transformations += [("{}::" + dtype, dtype, category_from_type(ctype = dtype))]
			self.parent.history += ["{" + time.strftime("%c") + "} " + "[AsType]: The vColumn '{}' was converted to {}.".format(self.alias, dtype)]
			return (self.parent)
		except:
			raise Exception("The column {} can not be converted to '{}'".format(self.alias, dtype))
	# 
	def avg(self):
		return (self.mean())
	# 
	def bar(self,
			method: str = "density",
			of = None,
			max_cardinality: int = 6,
			bins: int = 0,
			h: float = 0,
			color: str = '#214579'):
		from vertica_ml_python.plot import bar
		bar(self, method, of, max_cardinality, bins, h, color)	
		return (self.parent)
	# 
	def boxplot(self, 
				by: str = "", 
				h: float = 0, 
				max_cardinality: int = 8, 
				cat_priority: list = []):
		from vertica_ml_python.plot import boxplot
		boxplot(self, by, h, max_cardinality, cat_priority)
		return (self.parent)
	# 
	def category(self):
		return (self.transformations[-1][2])
	#
	def clip(self, lower = None, upper = None):
		if ((lower == None) and (upper == None)):
			raise ValueError("At least 'lower' or 'upper' must have a numerical value")
		lower_when = "WHEN {} < {} THEN {} ".format("{}", lower, lower) if (type(lower) in (float, int)) else ""
		upper_when = "WHEN {} > {} THEN {} ".format("{}", upper, upper) if (type(upper) in (float, int)) else "" 
		func = "(CASE " + lower_when + upper_when + "ELSE {} END)"
		self.apply(func = func)
		return (self.parent)
	# 
	def count(self):
		query = "SELECT COUNT({}) FROM {}".format(self.alias, self.parent.genSQL())
		self.executeSQL(query = query, title = "Compute the vColumn '" + self.alias + "' number of non-missing elements")
		missing_data=self.parent.cursor.fetchone()[0]
		return (missing_data)
	#
	def ctype(self):
		return (self.transformations[-1][1])
	# 
	def date_part(self, field: str):
		return (self.apply(func = "DATE_PART('{}', {})".format(field, "{}")))
	# 
	def decode(self, values: dict, others = None):
		new_dict = {}
		for elem in values:
			if (type(values[elem]) == str):
				val = "'" + values[elem] + "'"
			elif (values[elem] == None):
				val = "NULL"
			else:
				val = values[elem]
			if str(elem).upper() in ('NULL', 'NONE'):
				new_dict["NULL"] = val
			else:
				new_dict["'" + elem + "'"] = val
		others = "NULL" if (others == None) else others
		others = "'{}'".format(others) if (type(others) == str) else others
		fun = "DECODE({}, " + ", ".join(["{}, {}".format(item, new_dict[item]) for item in new_dict]) + ", {})".format(others)
		return (self.apply(func = fun))
	# 
	def density(self,
				a = None,
				kernel: str = "gaussian",
				smooth: int = 200,
				color: str = '#214579'):
		from vertica_ml_python.plot import density
		density(self, a, kernel, smooth, color)
		return (self.parent)
	# 
	def describe(self, 
				 method: str = "auto", 
				 max_cardinality: int = 6):
		if (method not in ["auto", "numerical", "categorical", "cat_stats"]):
			raise ValueError("The parameter 'method' must be in auto|categorical|numerical|cat_stats")
		distinct_count = self.nunique()
		is_numeric = self.isnum()
		is_date = self.isdate()
		if ((is_date) and not(method == "categorical")):
			query=("SELECT COUNT({}) AS count, MIN({}) AS min, MAX({}) AS max FROM {};").format(
						self.alias,self.alias,self.alias,self.parent.genSQL())
			self.executeSQL(query = query, title = "Compute the descriptive statistics of "+self.alias)
			result = self.parent.cursor.fetchall()
			result = [item for sublist in result for item in sublist]
			index = ['count', 'min', 'max']
		elif (method == "cat_stats"):
			query = []
			cat = self.distinct()
			lp = "(" if (len(cat) == 1) else ""
			rp = ")" if (len(cat) == 1) else ""
			for category in cat:
				tmp_query = "SELECT '{}' AS 'index', COUNT({}) AS count, MIN({}) AS min, APPROXIMATE_PERCENTILE ({} USING PARAMETERS percentile = 0.25) AS '25%', APPROXIMATE_PERCENTILE ({}".format(category, self.alias, self.alias, self.alias, self.alias)
				tmp_query += " USING PARAMETERS percentile = 0.5) AS '50%', APPROXIMATE_PERCENTILE ({} USING PARAMETERS percentile = 0.75) AS '75%', MAX".format(self.alias)
				tmp_query += "({}) AS max FROM {}".format(self.alias, self.parent.genSQL())
				tmp_query += " WHERE {} IS NULL".format(self.alias) if (category in ('None', None)) else " WHERE {} = '{}'".format(self.alias, category)
				query += [lp + tmp_query + rp]
			query = " UNION ALL ".join(query)
			result = to_tablesample(query, self.parent.cursor)
			result.table_info = False
			return (result)
		elif (((distinct_count < max_cardinality + 1) and (method != "numerical")) or not(is_numeric) or (method == "categorical")):
			query = "(SELECT {}||'', COUNT(*) FROM {} GROUP BY {} ORDER BY COUNT(*) DESC LIMIT {})".format(
					self.alias, self.parent.genSQL(), self.alias, max_cardinality)
			if (distinct_count > max_cardinality):
				query += ("UNION (SELECT 'Others', SUM(count) FROM (SELECT COUNT(*) AS count FROM {} WHERE {} IS NOT NULL GROUP BY {}" +
							" ORDER BY COUNT(*) DESC OFFSET {}) x) ORDER BY count DESC").format(
							self.parent.genSQL(), self.alias, self.alias, max_cardinality + 1)
			self.executeSQL(query = query, title = "Compute the descriptive statistics of " + self.alias)
			query_result = self.parent.cursor.fetchall()
			result = [distinct_count] + [item[1] for item in query_result]
			index = ['unique'] + [item[0] for item in query_result]
		else:
			result = self.summarize_numcol().values["value"]
			result = [distinct_count] + result
			index = ['unique', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
		values = {"index" : ["name", "dtype"] + index, "value" : [self.alias, self.ctype()] + result}
		return (tablesample(values, table_info = False))
	# 
	def distinct(self):
		query="SELECT {} FROM {} WHERE {} IS NOT NULL GROUP BY {} ORDER BY {}".format(
				self.alias, self.parent.genSQL(), self.alias, self.alias, self.alias)
		self.executeSQL(query = query, title = "Compute the distinct categories of {}".format(self.alias))
		query_result = self.parent.cursor.fetchall()
		return ([item for sublist in query_result for item in sublist])
	# 
	def div(self, x: float):
		return (self.divide(x = x))
	def divide(self, x: float):
		if (x == 0):
			raise ZeroDivisionError("Division by 0 is forbidden")
		return (self.apply(func = "{} / ({})".format("{}", x)))
	# 
	def donut(self,
			  method: str = "density",
			  of: str = "",
			  max_cardinality: int = 6,
			  h: float = 0):
		from vertica_ml_python.plot import pie
		pie(self, method, of, max_cardinality, h, True)	
		return (self.parent)
	# 
	def drop(self, add_history: bool = True):
		try:
			parent = self.parent
			force_columns = [column for column in self.parent.columns]
			force_columns.remove(self.alias)
			self.parent.cursor.execute("SELECT * FROM {} LIMIT 10".format(self.parent.genSQL(force_columns = force_columns)))
			self.parent.columns.remove(self.alias)
			delattr(self.parent, self.alias)
		except:
			self.parent.exclude_columns += [self.alias]
		if (add_history):
			print("vColumn '{}' deleted from the vDataframe.".format(self.alias))
			self.parent.history += ["{" + time.strftime("%c") + "} " + "[Drop]: vColumn '{}' was deleted from the vDataframe.".format(self.alias)]
		return (parent)
	# 
	def dropna(self, print_info: bool = True):
		count = self.parent.shape()[0]
		self.parent.where += [("{} IS NOT NULL".format(self.alias), len(self.transformations) - 1)]
		total = abs(count - self.parent.shape()[0])
		if (total > 1):
			if (print_info):
				print("{} elements were dropped".format(total))
			self.parent.history += ["{" + time.strftime("%c") + "} " + "[Dropna]: The {} missing elements of column '{}' were dropped from the vDataframe.".format(total, self.alias)]
		elif (total == 1):
			if (print_info):
				print("1 element was dropped")
			self.parent.history += ["{" + time.strftime("%c") + "} " + "[Dropna]: The only missing element of column '{}' was dropped from the vDataframe.".format(self.alias)]
		else:
			del self.parent.where[-1]
			if (print_info):
				print("/!\\ Warning: Nothing was dropped")
		return (self.parent) 
	# 
	def drop_outliers(self, alpha: float = 0.05, use_threshold: bool = True, threshold: float = 4.0):
		if (use_threshold):
			result = self.aggregate(func = ["std", "avg"]).transpose().values
			self.parent.filter(expr = "ABS({} - {}) / {} < {}".format(self.alias, result["avg"][0], result["std"][0], threshold))
		else:
			query = "SELECT PERCENTILE_CONT({}) WITHIN GROUP (ORDER BY {}) OVER (), PERCENTILE_CONT(1 - {}) WITHIN GROUP (ORDER BY {}) OVER () FROM {} LIMIT 1".format(alpha, self.alias, alpha, self.alias, self.parent.genSQL())
			self.executeSQL(query = query, title = "Compute the PERCENTILE_CONT of " + self.alias)
			p_alpha, p_1_alpha = self.parent.cursor.fetchone()
			self.parent.filter(expr = "({} BETWEEN {} AND {})".format(self.alias, p_alpha, p_1_alpha))
		return (self.parent)
	#
	def dtype(self):
		print("col".ljust(6) + self.ctype().rjust(12))
		print("dtype: object")
		return (self.ctype())
	#
	def ema(self, ts: str, by: list = [], alpha: float = 0.5):
		by = ['"' + column.replace('"', '') + '"' for column in by]
		by = "PARTITION BY {}".format(", ".join(by)) if (by) else ""
		return (self.apply(func = "EXPONENTIAL_MOVING_AVERAGE({}, {}) OVER ({} ORDER BY {})".format("{}", alpha, by, '"' + ts.replace('"', '') + '"')))
	# 
	def eq(self, x):
		return (self.equals(x))
	def equals(self, x):
		if (x == None):
			return (self.apply(func = "{} IS NULL"))
		else:
			return (self.apply(func = "{} = {}".format("{}", x)))
	# 
	def executeSQL(self, query: str, title: str = ""):
		return (self.parent.executeSQL(query = query, title = title))
	# 
	def fillna(self,
			   val = None,
			   method: str = "auto",
			   by: list = [],
			   order_by: list = [],
			   print_info: bool = True):
		if (method == "auto"):
			method = "mean" if (self.category() in ['float','int']) else "mode"
		total = self.count()
		if ((method == "mode") and (val == None)):
			val = self.mode()
		if (val != None):
			new_column = "COALESCE({}, '{}')".format("{}", val)
		elif (method == "0ifnull"):
			new_column = "DECODE({}, NULL, 0, 1)"
		elif (method in ('mean', 'avg', 'median')):
			fun = "MEDIAN" if (method == "median") else "AVG"
			if (by == []):
				if (fun == "AVG"):
					val = self.mean()
				elif (fun == "MEDIAN"):
					val = self.median()
				new_column = "COALESCE({}, {})".format("{}", val)
			else:
				new_column = "COALESCE({}, {}({}) OVER (PARTITION BY {}))".format("{}", fun, "{}", ", ".join(by))
		elif (method in ('ffill', 'pad', 'bfill', 'backfill')):
			if not(order_by):
				raise ValueError("If the method is in ffill|pad|bfill|backfill then 'order_by' must be a list of at least one element used to order the data")
			desc = " DESC" if (method in ("ffill", "pad")) else ""
			by = "PARTITION BY {}".format(", ".join(['"' + column.replace('"', '') + '"' for column in by])) if (by) else ""
			order_by = ", ".join(['"' + column.replace('"', '') + '"' for column in order_by])
			new_column = "COALESCE({}, LAST_VALUE({} IGNORE NULLS) OVER ({} ORDER BY {}{}))".format("{}", "{}", by, order_by, desc)
		else:
			raise ValueError("The method '{}' does not exist or is not available".format(method) + "\nPlease use a method in auto|mean|median|mode|ffill|bfill|0ifnull")
		if (method in ("mean", "median") or (type(val) == float)):
			category, ctype = "float", "float"
		elif (method == "0ifnull"):
			category, ctype = "int", "bool"
		else:
			category, ctype = self.category(), self.ctype()
		self.transformations += [(new_column, ctype, category)]
		total = abs(self.count() - total)
		if (total > 1):
			if (print_info):
				print("{} elements were filled".format(total))
			self.parent.history += ["{" + time.strftime("%c")+"} " + "[Fillna]: {} missing values of the vColumn '{}' were filled.".format(total, self.alias)]
		elif (total == 0):
			if (print_info):
				print("Nothing was filled")
			del self.transformations[-1]
		else:
			if (print_info):
				print("1 element was filled")
			self.parent.history += ["{" + time.strftime("%c") + "} " + "[Fillna]: 1 missing value of the vColumn '{}' was filled.".format(self.alias)]
		return (self.parent)
	#
	def fill_outliers(self, method: str = "winsorize", alpha: float = 0.05, use_threshold: bool = True, threshold: float = 4.0):
		if (method not in ("winsorize", "null", "mean")):
			raise ValueError("The parameter 'method' must be in winsorize|null|mean")
		else:
			if (use_threshold):
				result = self.aggregate(func = ["std", "avg"]).transpose().values
				p_alpha, p_1_alpha = - threshold * result["std"][0] + result["avg"][0], threshold * result["std"][0] + result["avg"][0]
			else:
				query = "SELECT PERCENTILE_CONT({}) WITHIN GROUP (ORDER BY {}) OVER (), PERCENTILE_CONT(1 - {}) WITHIN GROUP (ORDER BY {}) OVER () FROM {} LIMIT 1".format(alpha, self.alias, alpha, self.alias, self.parent.genSQL())
				self.executeSQL(query = query, title = "Compute the PERCENTILE_CONT of " + self.alias)
				p_alpha, p_1_alpha = self.parent.cursor.fetchone()
			if (method == "winsorize"):
				self.clip(lower = p_alpha, upper = p_1_alpha)
			elif (method == "null"):
				self.apply(func = "(CASE WHEN ({} BETWEEN {} AND {}) THEN {} ELSE NULL END)".format('{}', p_alpha, p_1_alpha, '{}'))
			elif (method == "mean"):
				query = "(SELECT AVG({}) FROM {} WHERE {} < {}) UNION ALL (SELECT AVG({}) FROM {} WHERE {} > {})".format(
					self.alias, self.parent.genSQL(), self.alias, p_alpha, self.alias, self.parent.genSQL(), self.alias, p_1_alpha)
				self.executeSQL(query = query, title = "Compute the MEAN of the {}'s lower and upper outliers".format(self.alias))
				mean_alpha, mean_1_alpha = [item[0] for item in self.parent.cursor.fetchall()]
				self.apply(func = "(CASE WHEN {} < {} THEN {} WHEN {} > {} THEN {} ELSE {} END)".format('{}', p_alpha, mean_alpha, '{}', p_1_alpha, mean_1_alpha, '{}'))
		return (self.parent)
	#
	def ge(self, x: float):
		return (self.apply(func = "{} >= ({})".format("{}", x)))
	# 
	def get_dummies(self, 
					prefix: str = "", 
					prefix_sep: str = "_", 
					drop_first: bool = False, 
					use_numbers_as_suffix: bool = False):
		if (self.nunique() < 3):
			print("/!\\ Warning: The column has already a limited number of elements.")
			print("Please use the label_encode func in order to code the elements if it is needed.")
		else:
			distinct_elements = self.distinct()
			all_new_features = []
			prefix = self.alias.replace('"', '') + prefix_sep if not(prefix) else prefix.replace('"', '') + prefix_sep
			n = 1 if drop_first else 0
			for k in range(len(distinct_elements) - n):
				if (use_numbers_as_suffix):
					name = '"{}{}"'.format(prefix, k)
				else:
					name = '"{}{}"'.format(prefix, distinct_elements[k])
				expr = "DECODE({}, '{}', 1, 0)".format("{}", distinct_elements[k])
				transformations = self.transformations + [(expr, "bool", "int")]
				new_vColumn = vColumn(name, parent = self.parent, transformations = transformations)
				setattr(self.parent, name, new_vColumn)
				setattr(self.parent, name.replace('"', ''), new_vColumn)
				self.parent.columns += [name]
				all_new_features += [name]
			print("{} new features: {}".format(len(all_new_features), ", ".join(all_new_features)))
			self.parent.history += ["{" + time.strftime("%c") + "} " + "[Get Dummies]: One hot encoder was applied to the vColumn '{}' and {} features were created: {}".format(self.alias, len(all_new_features), ", ".join(all_new_features)) + "."]
		return (self.parent)
	#
	def gt(self, x: float):
		return (self.apply(func = "{} > ({})".format("{}", x)))
	# 
	def head(self, limit: int = 5):
		return (self.tail(limit = limit))
	# 
	def hist(self,
			 method: str = "density",
			 of: str = "",
			 max_cardinality: int = 6,
			 bins: int = 0,
			 h: float = 0,
			 color: str = '#214579'):
		from vertica_ml_python.plot import hist
		hist(self, method, of, max_cardinality, bins, h, color)	
		return (self.parent)
	#
	def isdate(self):
		return(self.category() == "date")
	#
	def isin(self, val: list):
		val = {self.alias: val}
		return (self.parent.isin(val))
	#
	def isnum(self):
		return (self.category() in ("float", "int"))
	#
	def kurt(self):
		return self.kurtosis()
	def kurtosis(self):
		stats = self.parent.statistics(columns = [self.alias], skew_kurt_only = True)
		return (stats.values[self.alias][1])
	# 
	def label_encode(self):
		if (self.category() in ["date","float"]):
			print("/!\\ Warning: label_encode is only available for categorical variables.")
		else:
			distinct_elements = self.distinct()
			expr = ["DECODE({}"]
			text_info = "\n"
			for k in range(len(distinct_elements)):
				expr += ["'{}', {}".format(distinct_elements[k], k)]
				text_info += "\t{} => {}".format(distinct_elements[k], k)
			expr = ", ".join(expr) + ", {})".format(len(distinct_elements))
			self.transformations += [(expr, 'int', 'int')]
			self.parent.history += ["{" + time.strftime("%c") + "} " + "[Label Encoding]: Label Encoding was applied to the vColumn '{}' using the following mapping:{}".format(self.alias, text_info)]
		return (self.parent)
	#
	def le(self, x: float):
		return (self.apply(func = "{} <= ({})".format("{}", x)))
	#
	def lt(self, x: float):
		return (self.apply(func = "{} < ({})".format("{}", x)))
	# 
	def mad(self):
		return (self.aggregate(["mad"]).values[self.alias][0])
	# 
	def max(self):
		return (self.aggregate(["max"]).values[self.alias][0])
	# 
	def mean(self):
		return (self.aggregate(["avg"]).values[self.alias][0])
	# 
	def mean_encode(self, response_column: str):
		if (response_column not in self.parent.get_columns()) and ('"' + response_column.replace('"', '') + '"' not in self.parent.get_columns()):
			raise NameError("The response column doesn't exist")
		elif not(self.parent[response_column].isnum()):
			raise TypeError("The response column must be numerical to use a mean encoding")
		else:
			self.transformations += [("AVG({}) OVER (PARTITION BY {})".format(response_column,"{}"), "int", "float")]
			self.parent.history += ["{" + time.strftime("%c") + "} " + "[Mean Encode]: The vColumn '{}' was transformed using a mean encoding with as response column '{}'.".format(self.alias, response_column)]
			print("The mean encoding was successfully done.")
		return (self.parent)
	# 
	def median(self):
		return (self.quantile(0.5))
	#
	def memory_usage(self):
		import sys
		return (sys.getsizeof(self) + sys.getsizeof(self.alias) + sys.getsizeof(self.transformations))
	# 
	def min(self):
		return (self.aggregate(["min"]).values[self.alias][0])
	# 
	def mod(self, n: int):
		return (self.apply(func = "MOD({}, {})".format("{}", n)))
	# 
	def mode(self, dropna: bool = True):
		return (self.topk(k = 1, dropna = dropna).values["index"][0])
	# 
	def mul(self, x: float):
		return (self.apply(func = "{} * ({})".format("{}", x)))
	# 
	def neq(self, x):
		return (self.apply(func = "{} != {}".format("{}", x)))
	#
	def next(self, order_by: list, by: list = []):
		order_by = ['"' + column.replace('"', '') + '"' for column in order_by]
		by = ['"' + column.replace('"', '') + '"' for column in by]
		by = "PARTITION BY {}".format(", ".join(by)) if (by) else ""
		return (self.apply(func = "LEAD({}) OVER ({} ORDER BY {})".format("{}", by, ", ".join(order_by))))
	# 
	def nlargest(self, n: int = 10):
		query = "SELECT * FROM {} WHERE {} IS NOT NULL ORDER BY {} DESC LIMIT {}".format(self.parent.genSQL(), self.alias, self.alias, n)
		return (to_tablesample(query, self.parent.cursor, name = "nlargest"))
	# 
	def normalize(self, method = "zscore"):
		if (self.category() in ("int","float")):
			if (method == "zscore"):
				query = "SELECT AVG(" + self.alias + "), STDDEV(" + self.alias + ") FROM " + self.parent.genSQL()
				self.executeSQL(query = query, title = "Compute the AVG and STDDEV of " + self.alias + " for normalization")
				avg, stddev = self.parent.cursor.fetchone()
				self.transformations += [("({} - {}) / ({})".format("{}", avg, stddev), "float", "float")]
			elif (method == "robust_zscore"):
				query = "SELECT MIN(median) AS median, APPROXIMATE_MEDIAN(ABS({} - median)) AS mad FROM (SELECT MEDIAN({}) OVER () AS median, {} FROM {}) x".format(self.alias, self.alias, self.alias, self.parent.genSQL()) 
				self.executeSQL(query = query, title = "Compute the MEDIAN and MAD of " + self.alias + " for normalization")
				med, mad = self.parent.cursor.fetchone()
				mad *= 1.4826
				self.transformations += [("({} - {}) / ({})".format("{}", med, mad), "float", "float")]
			elif (method == "minmax"):
				query = "SELECT MIN(" + self.alias + "), MAX(" + self.alias + ") FROM " + self.parent.genSQL()
				self.executeSQL(query = query, title = "Compute the MIN and MAX of " + self.alias + " for normalization")
				cmin, cmax = self.parent.cursor.fetchone()
				self.transformations+=[("({} - {}) / ({} - {})".format("{}", cmin, cmax, cmin), "float", "float")]
			else:
				raise ValueError("The method '{}' doesn't exist\nPlease use a method in zscore|robust_zscore|minmax".format(method))
			print("The vColumn '" +self.alias+ "' was successfully normalized.")
			self.parent.history += ["{" + time.strftime("%c") + "} " + "[Normalize]: The vColumn '{}' was normalized with the method '{}'.".format(self.alias,method)]
		else:
			raise TypeError("The vColumn must be numerical for Normalization")
		return (self.parent)
	# 
	def nsmallest(self, n: int = 10):
		query = "SELECT * FROM {} WHERE {} IS NOT NULL ORDER BY {} ASC LIMIT {}".format(self.parent.genSQL(), self.alias, self.alias, n)
		return (to_tablesample(query, self.parent.cursor, name = "nsmallest"))
	# 
	def numh(self, method: str = "auto"):
		if (self.category() in ["int","float"]):
			result = self.summarize_numcol().values["value"]
			count, vColumn_min, vColumn_025, vColumn_075, vColumn_max = result[0], result[3], result[4], result[6], result[7]
		elif (self.category() in ["date"]):
			min_date = self.min()
			table = "(SELECT DATEDIFF('second','" + str(min_date) + "'::timestamp," + self.alias + ") AS " + self.alias + " FROM " + self.parent.genSQL()
			table += ") best_h_date_table"
			query = ("SELECT (SELECT COUNT(*) FROM {} WHERE {} IS NOT NULL) AS NAs, MIN({}) AS min, (SELECT PERCENTILE_CONT(0.25)"
					+ " WITHIN GROUP (ORDER BY {}) OVER () FROM {} LIMIT 1) AS Q1, (SELECT PERCENTILE_CONT(0.75) WITHIN GROUP" 
					+ " (ORDER BY {}) OVER () FROM {} LIMIT 1) AS Q3, MAX({}) AS max from {} GROUP BY Q1, Q3, NAs")
			query = query.format(table, self.alias, self.alias, self.alias, table, self.alias, table, self.alias, table, self.alias, table)
			self.executeSQL(query, title = "AGGR to compute h")
			result = self.parent.cursor.fetchone()
			count, vColumn_min, vColumn_025, vColumn_075, vColumn_max  = result
		else:
			raise TypeError("numh is only available on type float|date")
		sturges = math.floor(float(vColumn_max - vColumn_min) / int(math.floor(math.log(count, 2) + 1))) + 1
		fd = math.floor(2.0 * (vColumn_075 - vColumn_025) / (count) ** (1.0 / 3.0)) + 1
		if (method.lower() == "sturges"):
			return (sturges)
		elif (method.lower() in ("freedman_diaconis", "fd")):
			return (fd) 
		else:
			return (max(sturges, fd))
	# 
	def nunique(self):
		query = "SELECT COUNT(DISTINCT {}) FROM {} WHERE {} IS NOT NULL".format(self.alias, self.parent.genSQL(), self.alias)
		self.executeSQL(query = query, title = "Compute the feature "+self.alias+" cardinality")
		return (self.parent.cursor.fetchone()[0])
	#
	def pct_change(self, order_by: list, by: list = []):
		by = "PARTITION BY {}".format(", ".join(['"' + column.replace('"', '') + '"' for column in by])) if (by) else ""
		return (self.apply(func = "LEAD({}) OVER ({} ORDER BY {}) / {}".format("{}", by, ", ".join(['"' + column.replace('"', '') + '"' for column in order_by]), "{}")))
	# 
	def pie(self,
			method: str = "density",
			of: str = "",
			max_cardinality: int = 6,
			h: float = 0):
		from vertica_ml_python.plot import pie
		pie(self, method, of, max_cardinality, h, False)
		return (self.parent)
	# 
	def plot(self, 
			 ts: str, 
			 by: str = "",
			 start_date: str = "",
			 end_date: str = "",
			 color: str = '#214579', 
			 area: bool = False):
		from vertica_ml_python.plot import ts_plot
		ts_plot(self, ts, by, start_date, end_date, color, area)
		return (self.parent)	
	# 
	def pow(self, x: float):
		return (self.apply(func = "POWER({}, {})".format("{}", x)))
	#
	def prev(self, order_by: list, by: list = []):
		order_by = ['"' + column.replace('"', '') + '"' for column in order_by]
		by = ['"' + column.replace('"', '') + '"' for column in by]
		by = "PARTITION BY {}".format(", ".join(by)) if (by) else ""
		return (self.apply(func = "LAG({}) OVER ({} ORDER BY {})".format("{}", by, ", ".join(order_by))))
	#
	def prod(self):
		return (self.product())
	def product(self):
		return (self.aggregate(func = ["prod"]).values[self.alias][0])
	# 
	def quantile(self, x: float):
		query="SELECT PERCENTILE_CONT({}) WITHIN GROUP (ORDER BY {}) OVER () FROM {} LIMIT 1".format(x, self.alias, self.parent.genSQL())
		self.executeSQL(query = query, title = "Compute the PERCENTILE_CONT of " + self.alias)
		return (self.parent.cursor.fetchone()[0])
	#
	def rename(self, new_name: str):
		new_name = new_name.replace('"', '')
		if ('"' + new_name + '"' in self.parent.columns) or (new_name in self.parent.columns):
			error = "The column '{}' is already in the vDataframe".format(new_name)
			raise Exception(error)
		try:
			old_name = self.alias
			self.alias = new_name
			for idx, column in enumerate(self.parent.columns):
				if (column == old_name):
					self.parent.columns[idx] = new_name
					break
			self.parent[new_name] = self
			self.parent['"' + new_name + '"'] = self
			self.parent.cursor.execute("SELECT * FROM {} LIMIT 10".format(self.parent.genSQL()))
			print("vColumn {} was renamed {}".format(old_name, new_name))
			self.parent.history += ["{" + time.strftime("%c") + "} " + "[Rename]: The vColumn {} was renamed '{}'.".format(old_name, new_name)]
		except:
			self.alias = old_name
			for idx, column in enumerate(self.parent.columns):
				if (column == new_name):
					self.parent.columns[idx] = old_name
					break
			print("/!\\ Warning: The name wasn't change because of some columns dependencies")
		return (self.parent)
	# 
	def round(self, n: int):
		return (self.apply(func = "ROUND({}, {})".format("{}", n)))
	# 
	def sem(self):
		return (self.aggregate(["sem"]).values[self.alias][0])
	#
	def skew(self):
		return (self.skewness())
	def skewness(self):
		stats = self.parent.statistics(columns = [self.alias], skew_kurt_only = True)
		return (stats.values[self.alias][0])
	#
	def slice(self, length: int, unit: str = "second", start: bool = True):
		start_or_end = "START" if (start) else "END"
		return (self.apply(func = "TIME_SLICE({}, {}, '{}', '{}')".format("{}", length, unit.upper(), start_or_end)))
	#
	def store_usage(self):
		return (self.parent.cursor.execute("SELECT SUM(LENGTH({}::varchar)) FROM {}".format(self.alias, self.parent.genSQL())).fetchone()[0])
	#
	def summarize_numcol(self):
		# For Vertica 9.0 and higher
		try:
			query = "SELECT SUMMARIZE_NUMCOL({}) OVER () FROM {}".format(self.alias, self.parent.genSQL())
			self.executeSQL(query = query, title = "Compute a direct SUMMARIZE_NUMCOL(" + self.alias + ")")
			result = self.parent.cursor.fetchone()
			val = [float(result[i]) for i in range(1, len(result))]
		# For all versions of Vertica
		except:
			try:
				query = ("SELECT COUNT({}) AS count, AVG({}) AS mean, STDDEV({}) AS std, MIN({}) AS min, (SELECT " 
					    + "PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {}) OVER () FROM {} LIMIT 1)"
						+ " AS Q1, (SELECT PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY {}) OVER () FROM {} LIMIT 1) AS Median, "
						+ "(SELECT PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {}) OVER () FROM {} LIMIT 1) AS Q3, max({}) AS max "
						+ " FROM {} GROUP BY Q1, Median, Q3")
				query = query.format(self.alias, self.alias, self.alias, self.alias, self.alias, self.parent.genSQL(),
								     self.alias, self.parent.genSQL(), self.alias, self.parent.genSQL(), self.alias, 
								     self.parent.genSQL())
				self.executeSQL(query = query, title = "Compute a manual SUMMARIZE_NUMCOL(" + self.alias + ")")
				result = self.parent.cursor.fetchone()
				val = [float(item) for item in result]
			except:
				val = [self.count(), self.mean(), self.std(), self.min(), self.quantile(0.25), self.median(), self.quantile(0.75), self.max()]
		values = {"index" : ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'], "value" : val}
		return (tablesample(values, table_info = False))
	# 
	def std(self):
		return (self.aggregate(["stddev"]).values[self.alias][0])
	# 
	def str_contains(self, pat: str):
		return (self.apply(func = "REGEXP_COUNT({}, '{}') > 0".format("{}", pat))) 
	# 
	def str_count(self, pat: str):
		return (self.apply(func = "REGEXP_COUNT({}, '{}')".format("{}", pat))) 
	# 
	def str_extract(self, pat: str):
		return (self.apply(func = "REGEXP_SUBSTR({}, '{}')".format("{}", pat))) 
	#
	def str_replace(self, to_replace: str, value: str = ""):
		return (self.apply(func = "REGEXP_REPLACE({}, '{}', '{}')".format("{}", to_replace, value))) 
	# 
	def str_slice(self, start: int, step: int):
		return (self.apply(func = "SUBSTR({}, {}, {})".format("{}", start, step))) 
	# 
	def sub(self, x: float):
		if (self.isdate()):
			return (self.apply(func = "TIMESTAMPADD(SECOND, -({}), {})".format(x, "{}")))
		else:
			return (self.apply(func = "{} - ({})".format("{}", x)))
	# 
	def sum(self):
		return (self.aggregate(["sum"]).values[self.alias][0])
	# 
	def tail(self, limit: int = 5, offset: int = 0):
		tail = to_tablesample("SELECT {} FROM {} LIMIT {} OFFSET {}".format(self.alias, self.parent.genSQL(), limit, offset), self.parent.cursor)
		tail.count = self.parent.shape()[0]
		tail.offset = offset
		tail.dtype[self.alias.replace('"', '')] = self.ctype()
		tail.name = self.alias.replace('"', '')
		return (tail)
	# 
	def to_enum(self, 
				h: float = 0, 
				return_enum_trans: bool = False):
		if (self.isnum()):
			h = round(self.numh(), 2) if (h <= 0) else h
			h = int(max(math.floor(h), 1)) if (self.category == "int") else h
			floor_end = - 1 if (self.category == "int") else ""
			if (h > 1) or (self.category == "float"):
				trans = ("'[' || FLOOR({} / {}) * {} || ';' || (FLOOR({} / {}) * {} + {}{}) || ']'".format(
					"{}", h, h, "{}", h, h, h, floor_end), "varchar", "text")
			else:
				trans = ("FLOOR({}) || ''", "varchar", "text")
		else:
			trans = ("{} || ''", "varchar", "text")
		if (return_enum_trans):
			return(trans)
		else:
			self.transformations += [trans]
			self.parent.history += [ "{" + time.strftime("%c") + "} " + "[Enum]: The vColumn '{}' was converted to enum.".format(self.alias)]
		return (self.parent)
	# 
	def topk(self, k: int = -1, dropna: bool = True):
		if (k < 1):
			topk = ""
		else:
			topk = "TOPK = {},".format(k)
		query = "SELECT SUMMARIZE_CATCOL({}::varchar USING PARAMETERS {} WITH_TOTALCOUNT = False) OVER () FROM {}".format(self.alias, topk, self.parent.genSQL())
		if (dropna):
			query += " WHERE {} IS NOT NULL".format(self.alias)
		self.executeSQL(query, title = "Compute the TOPK categories of "+self.alias)
		result = self.parent.cursor.fetchall()
		values = {"index" : [item[0] for item in result], "count" : [item[1] for item in result], "percent" : [item[2] for item in result]}
		return (tablesample(values, table_info = False))
	#
	def to_timestamp(self):
		return (self.astype(dtype = "timestamp"))
	# 
	def value_counts(self, k: int = 30):
		return (self.describe(method = "categorical", max_cardinality = k))
	#
	def var(self):
		return (self.aggregate(["variance"]).values[self.alias][0])




