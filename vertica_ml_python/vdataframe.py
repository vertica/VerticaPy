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
import os
import math
import time
from vertica_ml_python.vcolumn import vColumn
from vertica_ml_python.utilities import print_table
from vertica_ml_python.utilities import isnotebook
from vertica_ml_python.utilities import tablesample
from vertica_ml_python.utilities import to_tablesample
from vertica_ml_python.utilities import category_from_type
##
#                                           _____    
#   _______    ______ ____________    ____  \    \   
#   \      |  |      |\           \   \   \ /____/|  
#    |     /  /     /| \           \   |  |/_____|/  
#    |\    \  \    |/   |    /\     |  |  |    ___   
#    \ \    \ |    |    |   |  |    |  |   \__/   \  
#     \|     \|    |    |    \/     | /      /\___/| 
#      |\         /|   /           /|/      /| | | | 
#      | \_______/ |  /___________/ ||_____| /\|_|/  
#       \ |     | /  |           | / |     |/        
#        \|_____|/   |___________|/  |_____|         
#                                                   
##
# 
class vDataframe:
	#
	def  __init__(self,
				  input_relation: str,
				  cursor = None,
				  dsn: str = "",
				  usecols: list = [],
				  schema: str = "",
				  empty: bool = False):
		if not(empty):
			if (cursor == None):
				from vertica_ml_python import vertica_cursor
				cursor = vertica_cursor(dsn)
			self.dsn = dsn
			if not(schema):
				schema_input_relation = input_relation.split(".")
				if (len(schema_input_relation) == 1):
					self.schema = "public"
					self.input_relation = input_relation
				else:
					self.input_relation = schema_input_relation[1]
					self.schema = schema_input_relation[0]
			else:
				self.schema = schema
				self.input_relation = input_relation
			# Cursor to the Vertica Database
			self.cursor = cursor
			# All the columns of the vDataframe
			if (usecols == []):
				query = "(SELECT column_name FROM columns WHERE table_name='{}' AND table_schema='{}')".format(self.input_relation, self.schema)
				query += " UNION (SELECT column_name FROM view_columns WHERE table_name='{}' AND table_schema='{}')".format(self.input_relation, self.schema)
				cursor.execute(query)
				columns = cursor.fetchall()
				columns = [str(item) for sublist in columns for item in sublist]
				columns = ['"' + item + '"' for item in columns]
				if (columns != []):
					self.columns = columns
				else:
					print("/!\\ Warning: No table or views '{}' found.\nNothing was created.".format(self.input_relation))
					del self
					return None
			else:
				self.columns = usecols
			for column in self.columns:
				new_vColumn = vColumn(column, parent = self)
				setattr(self, column, new_vColumn)
				setattr(self, column[1:-1], new_vColumn)
			# Columns to not consider for the final query
			self.exclude_columns = []
			# Rules for the cleaned data
			self.where = []
			# Rules to sort the data
			self.order_by = []
			# Display the elapsed time during the query
			self.time_on = False
			# Display or not the sequal queries that are used during the vDataframe manipulation
			self.query_on = False
			# vDataframe history
			self.history = []
			# vDataframe saving
			self.saving = []
			# vDataframe main relation
			self.main_relation = '"{}"."{}"'.format(self.schema, self.input_relation)
	# 
	def __getitem__(self, index):
		return getattr(self, index)
	def __setitem__(self, index, val):
		setattr(self, index, val)
	# 
	def __repr__(self):
		return self.head(limit = 5).__repr__()
	# 
	def __setattr__(self, attr, val):
		self.__dict__[attr] = val
	#
	# SQL GEN = THE MOST IMPORTANT METHOD
	#
	def genSQL(self, 
			   split: bool = False, 
			   transformations: dict = {}, 
			   force_columns = [],
			   final_table_name = "final_table"):
		# FINDING MAX FLOOR
		all_imputations_grammar = []
		force_columns = self.columns if not(force_columns) else force_columns
		for column in force_columns:
		    all_imputations_grammar += [[item[0] for item in self[column].transformations]]
		for column in transformations:
			all_imputations_grammar += [transformations[column]]
		# MAX FLOOR
		max_len = len(max(all_imputations_grammar, key=len))
		# TRANSFORMATIONS COMPLETION
		for imputations in all_imputations_grammar:
		    diff = max_len - len(imputations)
		    if diff > 0:
		        imputations += ["{}"]*diff
		# FILTER
		where_positions = [item[1] for item in self.where]
		max_where_pos = max(where_positions+[0])
		all_where = [[] for item in range(max_where_pos+1)]
		for i in range(0, len(self.where)):
			all_where[where_positions[i]] += [self.where[i][0]]
		all_where = [" AND ".join(item) for item in all_where]
		for i in range(len(all_where)):
			if (all_where[i] != ''):
				all_where[i] = " WHERE " + all_where[i]
		# ORDER BY
		order_by_positions = [item[1] for item in self.order_by]
		max_order_by_pos = max(order_by_positions+[0])
		all_order_by = [[] for item in range(max_order_by_pos+1)]
		for i in range(0, len(self.order_by)):
			all_order_by[order_by_positions[i]] += [self.order_by[i][0]]
		all_order_by = [", ".join(item) for item in all_order_by]
		for i in range(len(all_order_by)):
			if (all_order_by[i] != ''):
				all_order_by[i] = " ORDER BY " + all_order_by[i]
		# FIRST FLOOR
		columns = force_columns + [column for column in transformations]
		first_values = [item[0] for item in all_imputations_grammar]
		for i in range(0, len(first_values)):
		    first_values[i] = "{} AS {}".format(first_values[i], columns[i]) 
		table = "SELECT " + ", ".join(first_values) + " FROM " + self.main_relation
		# OTHER FLOORS
		for i in range(1, max_len):
		    values = [item[i] for item in all_imputations_grammar]
		    for j in range(0, len(values)):
		        values[j] = values[j].replace("{}", columns[j]) + " AS " + columns[j]
		    table = "SELECT " + ", ".join(values) + " FROM (" + table + ") t" + str(i)
		    try:
		    	table += all_where[i - 1]
		    except:
		    	pass
		    try:
		    	table += all_order_by[i - 1]
		    except:
		    	pass
		try:
			where_final = all_where[max_len - 1]
		except:
			where_final = ""
		try:
			order_final = all_order_by[max_len - 1]
		except:
			order_final = ""
		split = ", RANDOM() AS __split_vpython__" if (split) else ""
		if (where_final == "") and (order_final == ""):
			if (split):
				table = "(SELECT *{} FROM (".format(split) + table + ") " + final_table_name + ") split_final_table"
			else:
				table = "(" + table + ") " + final_table_name
		else:
			table = "(" + table + ") t" + str(max_len)
			table += where_final + order_final
			table = "(SELECT *{} FROM " + table + ") ".format(split) + final_table_name
		if (self.exclude_columns):
			table = "(SELECT " + ", ".join(self.get_columns()) + split + " FROM " + table + ") " + final_table_name
		return table
	#
	#
	#
	# METHODS
	# 
	def abs(self, columns: list = []):
		func = {}
		if not(columns):
			columns = self.numcol()
		for column in columns:
			func[column] = "abs({})"
		return (self.apply(func))
	#
	def agg(self, func: list, columns: list = []):
		return (self.aggregate(func = func, columns = columns))
	def aggregate(self, func: list, columns: list = []):
		columns = self.numcol() if not(columns) else ['"' + column.replace('"', '') + '"' for column in columns]
		query = []
		for column in columns:
			cast = "::int" if (self[column].ctype()[0:4] == "bool") else ""
			for fun in func:
				if (fun.lower() == "median"):
					expr = "APPROXIMATE_MEDIAN({}{})".format(column, cast)
				elif (fun.lower() == "std"):
					expr = "STDDEV({}{})".format(column, cast)
				elif (fun.lower() == "var"):
					expr = "VARIANCE({}{})".format(column, cast)
				elif (fun.lower() == "mean"):
					expr = "AVG({}{})".format(column, cast)
				elif ('%' in fun):
					expr = "APPROXIMATE_PERCENTILE({}{} USING PARAMETERS percentile = {})".format(column, cast, float(fun[0:-1]) / 100)
				elif (fun.lower() == "sem"):
					expr = "STDDEV({}{}) / SQRT(COUNT({}))".format(column, cast, column)
				elif (fun.lower() == "mad"):
					mean = self[column].mean()
					expr = "SUM(ABS({}{} - {})) / COUNT({})".format(column, cast, mean, column)
				elif (fun.lower() in ("prod", "product")):
					expr = "DECODE(ABS(MOD(SUM(CASE WHEN {}{} < 0 THEN 1 ELSE 0 END), 2)), 0, 1, -1) * POWER(10, SUM(LOG(ABS({}{}))))".format(column, cast, column, cast)
				elif (fun.lower() in ("percent", "count_percent")):
					expr = "ROUND(COUNT({}) / {} * 100, 3)".format(column, self.shape()[0])
				else:
					expr = "{}({}{})".format(fun.upper(), column, cast)
				query += [expr]
		query = "SELECT {} FROM {}".format(', '.join(query), self.genSQL())
		self.executeSQL(query, title = "COMPUTE AGGREGATION(S)")
		result = [item for item in self.cursor.fetchone()]
		try:
			result = [float(item) for item in result]
		except:
			pass
		values = {"index": func}
		i = 0
		for column in columns:
			values[column] = result[i:i + len(func)]
			i += len(func)
		return (tablesample(values = values, table_info = False).transpose())
	# 
	def aggregate_matrix(self, 
						 method: str = "pearson",
			 	   		 columns: list = [], 
			 	   		 cmap: str = "",
			 	   		 round_nb: int = 3,
			 	   		 show: bool = True):
		columns = ['"' + column.replace('"', '') + '"' for column in columns]
		for column_name in columns:
			if not(column_name in self.get_columns()):
				raise NameError("The parameter 'columns' must be a list of different columns name")
		if (len(columns) == 1):
			if (method in ("pearson", "beta", "spearman", "kendall", "biserial", "cramer")):
				return 1.0
			elif (method == "cov"):
				return self[columns[0]].var()
		elif (len(columns) == 2):
			cast_0 = "::int" if (self[columns[0]].ctype()[0:4] == "bool") else ""
			cast_1 = "::int" if (self[columns[1]].ctype()[0:4] == "bool") else ""
			if (method in ("pearson", "spearman")):
				if (columns[1] == columns[0]):
					return 1
				table = self.genSQL() if (method == "pearson") else "(SELECT RANK() OVER (ORDER BY {}) AS {}, RANK() OVER (ORDER BY {}) AS {} FROM {}) rank_spearman_table".format(columns[0], columns[0], columns[1], columns[1], self.genSQL())
				query = "SELECT CORR({}{}, {}{}) FROM {}".format(columns[0], cast_0, columns[1], cast_1, table)
				title = "Compute the {} Correlation between the two variables".format(method)
			elif (method == "biserial"):
				if (self[columns[1]].nunique() == 2 and self[columns[1]].min() == 0 and self[columns[1]].max() == 1):
					if (columns[1] == columns[0]):
						return 1
					column_b, column_n = columns[1], columns[0]
					cast_b, cast_n = cast_1, cast_0
				elif (self[columns[0]].nunique() == 2 and self[columns[0]].min() == 0 and self[columns[0]].max() == 1):
					if (columns[1] == columns[0]):
						return 1
					column_b, column_n = columns[0], columns[1]
					cast_b, cast_n = cast_0, cast_1
				else:
					return None
				query = "SELECT (AVG(DECODE({}{}, 1, {}{}, NULL)) - AVG(DECODE({}{}, 0, {}{}, NULL))) / STDDEV({}{}) * SQRT(SUM({}{}) * SUM(1 - {}{}) / COUNT(*) / COUNT(*)) FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL;".format(
					column_b, cast_b, column_n, cast_n, column_b, cast_b, column_n, cast_n, column_n, cast_n, column_b, cast_b, column_b, cast_b, self.genSQL(), column_n, column_b)
				title = "Compute the biserial Correlation between the two variables"
			elif (method == "cramer"):
				if (columns[1] == columns[0]):
					return 1
				k, r = self[columns[0]].nunique(), self[columns[1]].nunique()
				table_0_1 = "SELECT {}, {}, COUNT(*) AS nij FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL GROUP BY 1, 2".format(columns[0], columns[1], self.genSQL(), columns[0], columns[1])
				table_0 = "SELECT {}, COUNT(*) AS ni FROM {} WHERE {} IS NOT NULL GROUP BY 1".format(columns[0], self.genSQL(), columns[0])
				table_1 = "SELECT {}, COUNT(*) AS nj FROM {} WHERE {} IS NOT NULL GROUP BY 1".format(columns[1], self.genSQL(), columns[1])
				query_count = "SELECT COUNT(*) AS n FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL".format(self.genSQL(), columns[0], columns[1])
				self.cursor.execute(query_count)
				n = self.cursor.fetchone()[0]
				query = "SELECT SUM((nij - ni * nj / {}) * (nij - ni * nj / {}) / (ni * nj)) AS phi2 FROM (SELECT * FROM ({}) table_0_1 LEFT JOIN ({}) table_0 ON table_0_1.{} = table_0.{}) x LEFT JOIN ({}) table_1 ON x.{} = table_1.{}"
				query = query.format(n, n, table_0_1, table_0, columns[0], columns[0], table_1, columns[1], columns[1])
				self.cursor.execute(query)
				phi2 = self.cursor.fetchone()[0]
				phi2 = max(0, float(phi2) - (r - 1) * (k - 1) / (n - 1))
				k = k - (k - 1) * (k - 1) / (n - 1)
				r = r - (r - 1) * (r - 1) / (n - 1)
				return math.sqrt(phi2 / min(k, r))
			elif (method == "kendall"):
				if (columns[1] == columns[0]):
					return 1
				query = "SELECT (SUM(((x.{}{} < y.{}{} AND x.{}{} < y.{}{}) OR (x.{}{} > y.{}{} AND x.{}{} > y.{}{}))::int) - SUM(((x.{}{} > y.{}{} AND x.{}{} < y.{}{}) OR (x.{}{} < y.{}{} AND x.{}{} > y.{}{}))::int)) / SUM((x.{}{} != y.{}{} AND x.{}{} != y.{}{})::int) FROM (SELECT {}, {} FROM {}) x CROSS JOIN (SELECT {}, {} FROM {}) y"
				query = query.format(columns[0], cast_0, columns[0], cast_0, columns[1], cast_1, columns[1], cast_1, columns[0], cast_0, columns[0], cast_0, columns[1], cast_1, columns[1], cast_1, columns[0], cast_0, columns[0], cast_0, columns[1], cast_1, columns[1], cast_1, columns[0], cast_0, columns[0], cast_0, columns[1], cast_1, columns[1], cast_1, columns[0], cast_0, columns[0], cast_0, columns[1], cast_1, columns[1], cast_1, columns[0], columns[1], self.genSQL(), columns[0], columns[1], self.genSQL())
				title = "Compute the kendall Correlation between the two variables"
			elif (method == "cov"):
				query = "SELECT COVAR_POP({}{}, {}{}) FROM {}".format(columns[0], cast_0, columns[1], cast_1, self.genSQL())
				title = "Compute the Covariance between the two variables"
			elif (method == "beta"):
				if (columns[1] == columns[0]):
					return 1
				query = "SELECT COVAR_POP({}{}, {}{}) / VARIANCE({}{}) FROM {}".format(columns[0], cast_0, columns[1], cast_1, columns[1], cast_1, self.genSQL())
				title = "Compute the elasticity Beta between the two variables"
			try:
				self.executeSQL(query = query, title = title)
				return self.cursor.fetchone()[0]
			except:
				return None
		elif (len(columns) >= 2):
			if (method in ("pearson", "spearman", "kendall", "biserial", "cramer")):
				title_query = "Compute all the Correlations in a single query"
				title = 'Correlation Matrix ({})'.format(method)
				if (method == "biserial"):
					i0, step = 0, 1
				else:
					i0, step = 1, 0 
			elif (method == "cov"):
				title_query = "Compute all the Covariances in a single query"
				title = 'Covariance Matrix'
				i0, step = 0, 1
			elif (method == "beta"):
				title_query = "Compute all the Beta Coefficients in a single query"
				title = 'Elasticity Matrix'
				i0, step = 1, 0
			try:
				all_list = []
				n = len(columns)
				for i in range(i0, n):
					for j in range(0, i + step):
						cast_i = "::int" if (self[columns[i]].ctype()[0:4] == "bool") else ""
						cast_j = "::int" if (self[columns[j]].ctype()[0:4] == "bool") else ""
						if (method in ("pearson", "spearman")):
							all_list += ["ROUND(CORR({}{}, {}{}), {})".format(columns[i], cast_i, columns[j], cast_j, round_nb)]
						elif (method == "kendall"):
							all_list += ["(SUM(((x.{}{} < y.{}{} AND x.{}{} < y.{}{}) OR (x.{}{} > y.{}{} AND x.{}{} > y.{}{}))::int) - SUM(((x.{}{} > y.{}{} AND x.{}{} < y.{}{}) OR (x.{}{} < y.{}{} AND x.{}{} > y.{}{}))::int)) / NULLIFZERO(SUM((x.{}{} != y.{}{} AND x.{}{} != y.{}{})::int))".format(
								columns[i], cast_i, columns[i], cast_i, columns[j], cast_j, columns[j], cast_j, columns[i], cast_i, columns[i], cast_i, columns[j], cast_j, columns[j], cast_j, columns[i], cast_i, columns[i], cast_i, columns[j], cast_j, columns[j], cast_j, columns[i], cast_i, columns[i], cast_i, columns[j], cast_j, columns[j], cast_j, columns[i], cast_i, columns[i], cast_i, columns[j], cast_j, columns[j], cast_j)]
						elif (method == "cov"):
							all_list += ["COVAR_POP({}{}, {}{})".format(columns[i], cast_i, columns[j], cast_j)]
						elif (method == "beta"):
							all_list += ["COVAR_POP({}{}, {}{}) / VARIANCE({}{})".format(columns[i], cast_i, columns[j], cast_j, columns[j], cast_j)]
						else:
							raise
				if (method == "spearman"):
					rank = ["RANK() OVER (ORDER BY {}) AS {}".format(column, column) for column in columns]
					table = "(SELECT {} FROM {}) rank_spearman_table".format(", ".join(rank), self.genSQL())
				elif (method == "kendall"):
					table = "(SELECT {} FROM {}) x CROSS JOIN (SELECT {} FROM {}) y".format(", ".join(columns), self.genSQL(), ", ".join(columns), self.genSQL())
				else:
					table = self.genSQL()
				self.executeSQL(query = "SELECT {} FROM {}".format(", ".join(all_list), table), title = title_query)
				result = self.cursor.fetchone()
			except:
				n = len(columns)
				result = []
				for i in range(i0, n):
					for j in range(0, i + step):
						result += [self.aggregate_matrix(method, [columns[i], columns[j]])]
			matrix = [[1 for i in range(0, n + 1)] for i in range(0, n + 1)]
			matrix[0] = [""] + columns
			for i in range(0, n + 1):
				matrix[i][0] = columns[i - 1]
			k = 0
			for i in range(i0, n):
				for j in range(0, i + step):
					current = result[k]
					k += 1
					if (current == None):
						current = ""
					matrix[i + 1][j + 1] = current
					matrix[j + 1][i + 1] = 1 / current if ((method == "beta") and (current != 0)) else current
			if ((show) and (method in ("pearson", "spearman", "kendall", "biserial", "cramer"))):
				from vertica_ml_python.plot import cmatrix
				vmin = 0 if (method == "cramer") else -1
				if not(cmap):
					from matplotlib.colors import LinearSegmentedColormap
					cm1 = LinearSegmentedColormap.from_list("vml", ["#FFFFFF", "#214579"], N = 1000)
					cm2 = LinearSegmentedColormap.from_list("vml", ["#FFCC01", "#FFFFFF", "#214579"], N = 1000)
					cmap = cm1 if (method == "cramer") else cm2
				cmatrix(matrix, columns, columns, n, n, vmax = 1, vmin = vmin, cmap = cmap, title = title, mround = round_nb)
			values = {"index" : matrix[0][1:len(matrix[0])]}
			del(matrix[0])
			for column in matrix:
				values[column[0]] = column[1:len(column)]
			return tablesample(values = values, table_info = False)
		else:
			if (method == "cramer"):
				cols = self.catcol(100)
				if (len(cols) == 0):
			 		raise Exception("No categorical column found")
			else:
				cols = self.numcol()
				if (len(cols) == 0):
			 		raise Exception("No numerical column found")
			return (self.aggregate_matrix(method = method, columns = cols, cmap = cmap, round_nb = round_nb, show = show))
	# 
	def aggregate_vector(self, 
						 focus: str,
						 method: str = "pearson",
			 	   		 columns: list = [], 
			 	   		 cmap: str = "",
			 	   		 round_nb: int = 3,
			 	   		 show: bool = True):
		if (len(columns) == 0):
			if (method == "cramer"):
				cols = self.catcol(100)
				if (len(cols) == 0):
			 		raise Exception("No categorical column found")
			else:
				cols = self.numcol()
				if (len(cols) == 0):
			 		raise Exception("No numerical column found")
		else:
			cols = [column for column in columns]
		if (method in ('spearman', 'pearson', 'kendall') and (len(cols) > 1)):
			cast_i = "::int" if (self[focus].ctype()[0:4] == "bool") else ""
			all_list, all_cols  = [], [focus]
			for column in cols:
				if (column.replace('"', '').lower() != focus.replace('"', '').lower()):
					all_cols += [column]
				cast_j = "::int" if (self[column].ctype()[0:4] == "bool") else ""
				if (method in ("pearson", "spearman")):
					all_list += ["ROUND(CORR({}{}, {}{}), {})".format(focus, cast_i, column, cast_j, round_nb)]
				elif (method == "kendall"):
					all_list += ["(SUM(((x.{}{} < y.{}{} AND x.{}{} < y.{}{}) OR (x.{}{} > y.{}{} AND x.{}{} > y.{}{}))::int) - SUM(((x.{}{} > y.{}{} AND x.{}{} < y.{}{}) OR (x.{}{} < y.{}{} AND x.{}{} > y.{}{}))::int)) / NULLIFZERO(SUM((x.{}{} != y.{}{} AND x.{}{} != y.{}{})::int))".format(
						focus, cast_i, focus, cast_i, column, cast_j, column, cast_j, focus, cast_i, focus, cast_i, column, cast_j, column, cast_j, focus, cast_i, focus, cast_i, column, cast_j, column, cast_j, focus, cast_i, focus, cast_i, column, cast_j, column, cast_j, focus, cast_i, focus, cast_i, column, cast_j, column, cast_j)]
				elif (method == "cov"):
					all_list += ["COVAR_POP({}{}, {}{})".format(focus, cast_i, column, cast_j)]
				elif (method == "beta"):
					all_list += ["COVAR_POP({}{}, {}{}) / VARIANCE({}{})".format(focus, cast_i, column, cast_j, column, cast_j)]
			if (method == "spearman"):
				rank = ["RANK() OVER (ORDER BY {}) AS {}".format(column, column) for column in all_cols]
				table = "(SELECT {} FROM {}) rank_spearman_table".format(", ".join(rank), self.genSQL())
			elif (method == "kendall"):
				table = "(SELECT {} FROM {}) x CROSS JOIN (SELECT {} FROM {}) y".format(", ".join(all_cols), self.genSQL(), ", ".join(all_cols), self.genSQL())
			else:
				table = self.genSQL()
			self.executeSQL(query = "SELECT {} FROM {}".format(", ".join(all_list), table), title = "Compute the Correlation Vector")
			result = self.cursor.fetchone()
			vector = [elem for elem in result]
		else: 
			vector = []
			for column in cols:
				if (column.replace('"', '').lower() == focus.replace('"', '').lower()):
					vector += [1]
				else:
					vector += [self.aggregate_matrix(method = method, columns = [column, focus])]
		if ((show) and (method in ("pearson", "spearman", "kendall", "biserial", "cramer"))):
			from vertica_ml_python.plot import cmatrix
			vmin = 0 if (method == "cramer") else -1
			if not(cmap):
				from matplotlib.colors import LinearSegmentedColormap
				cm1 = LinearSegmentedColormap.from_list("vml", ["#FFFFFF", "#214579"], N = 1000)
				cm2 = LinearSegmentedColormap.from_list("vml", ["#FFCC01", "#FFFFFF", "#214579"], N = 1000)
				cmap = cm1 if (method == "cramer") else cm2
			title = "Correlation Vector of {} ({})".format(focus, method)
			cmatrix([cols, [focus] + vector], cols, [focus], len(cols), 1, vmax = 1, vmin = vmin, cmap = cmap, title = title, mround = round_nb)
		return tablesample(values = {"index" : cols, focus : vector}, table_info = False)
	#
	def append(self, 
			   vdf = None, 
			   input_relation: str = ""):
		first_relation = self.genSQL()
		second_relation = input_relation if not(vdf) else vdf.genSQL()
		table = "(SELECT * FROM {}) UNION ALL (SELECT * FROM {})".format(first_relation, second_relation)
		query = "SELECT * FROM ({}) append_table LIMIT 1".format(table)
		self.executeSQL(query = query, title = "Merging the two relation")
		self.main_relation = "({}) append_table".format(table)
		return (self)
	#
	def all(self, columns: list):
		return (self.aggregate(func = ["bool_and"], columns = columns))
	#
	def any(self, columns: list):
		return (self.aggregate(func = ["bool_or"], columns = columns))
	# 
	def apply(self, func: dict):
		for column in func:
			self[column].apply(func[column])
		return (self)
	# 
	def applymap(self, func: str, numeric_only: bool = True):
		function = {}
		columns = self.numcol() if numeric_only else self.get_columns()
		for column in columns:
			function[column] = func
		return (self.apply(function))
	#
	def asfreq(self,
			   ts: str,
			   rule: str,
			   method: dict,
			   by: list = []):
		all_elements = []
		for column in method:
			if (method[column] not in ('bfill', 'backfill', 'pad', 'ffill', 'linear')):
				raise ValueError("Each element of the 'method' dictionary must be in bfill|backfill|pad|ffill|linear")
			if (method[column] in ('bfill', 'backfill')):
				func = "TS_FIRST_VALUE"
				interp = 'const'
			elif (method[column] in ('pad', 'ffill')):
				func = "TS_LAST_VALUE"
				interp = 'const'
			else:
				func = "TS_FIRST_VALUE"
				interp = 'linear'
			all_elements += ["{}({}, '{}') AS {}".format(func, '"' + column.replace('"', '') + '"', interp, '"' + column.replace('"', '') + '"')]
		table = "SELECT {} FROM {}".format("{}", self.genSQL())
		tmp_query = ["slice_time AS {}".format('"' + ts.replace('"', '') + '"')]
		tmp_query += ['"' + column.replace('"', '') + '"' for column in by]
		tmp_query += all_elements
		table = table.format(", ".join(tmp_query))
		table += " TIMESERIES slice_time AS '{}' OVER (PARTITION BY {} ORDER BY {})".format(rule, ", ".join(['"' + column.replace('"', '') + '"' for column in by]), '"' + ts.replace('"', '') + '"')
		return (self.vdf_from_relation("(" + table + ') resample_table', "resample", "[Resample]: The data was resampled"))
	#
	def astype(self, dtype: dict):
		for column in dtype:
			self[column].astype(dtype = dtype[column])
		return (self)
	#
	def at_time(self,
				ts: str, 
				time: str):
		expr = "{}::time = '{}'".format('"' + ts.replace('"', '') + '"', time)
		self.filter(expr)
		return (self)
	#
	def avg(self, columns = []):
		return (self.mean(columns = columns))
	# 
	def bar(self,
			columns: list,
			method: str = "density",
			of: str = "",
			max_cardinality: tuple = (6, 6),
			h: tuple = (None, None),
			limit_distinct_elements: int = 200,
			hist_type: str = "auto"):
		if (len(columns) == 1):
			self[columns[0]].bar(method, of, 6, 0, 0)
		else:
			stacked, fully_stacked = False, False
			if (hist_type.lower() in ("fully", "fully stacked", "fully_stacked")):
				fully_stacked = True
			elif (hist_type.lower() == "stacked"):
				stacked = True
			from vertica_ml_python.plot import bar2D
			bar2D(self, columns, method, of, max_cardinality, h, limit_distinct_elements, stacked, fully_stacked)
		return (self)
	# 
	def beta(self, columns: list = [], focus: str = ""):
		if (focus == ""):
			return (self.aggregate_matrix(method = "beta", columns = columns))
		else:
			return (self.aggregate_vector(focus, method = "beta", columns = columns))
	#
	def between_time(self,
					 ts: str, 
					 start_time: str, 
					 end_time: str):
		expr = "{}::time BETWEEN '{}' AND '{}'".format('"' + ts.replace('"', '') + '"', start_time, end_time)
		self.filter(expr)
		return (self)
	#
	def bool_to_int(self):
		columns = self.get_columns()
		for column in columns:
			if (self[column].ctype()[0:4] == "bool"):
				self[column].astype("int")
		return (self)
	#
	def boxplot(self, columns: list = []):
		from vertica_ml_python.plot import boxplot2D
		boxplot2D(self, columns)
		return (self)
	#
	def catcol(self, max_cardinality: int = 12):
		columns = []
		for column in self.get_columns():
			if (self[column].nunique() <= max_cardinality):
				columns += [column]
		return (columns)
	#
	def copy(self):
		copy_vDataframe = vDataframe("", empty = True)
		copy_vDataframe.dsn = self.dsn
		copy_vDataframe.input_relation = self.input_relation
		copy_vDataframe.main_relation = self.main_relation
		copy_vDataframe.schema = self.schema
		copy_vDataframe.cursor = self.cursor
		copy_vDataframe.columns = [item for item in self.columns]
		copy_vDataframe.where = [item for item in self.where]
		copy_vDataframe.order_by = [item for item in self.order_by]
		copy_vDataframe.exclude_columns = [item for item in self.exclude_columns]
		copy_vDataframe.history = [item for item in self.history]
		copy_vDataframe.saving = [item for item in self.saving]
		copy_vDataframe.query_on = self.query_on
		copy_vDataframe.time_on = self.time_on
		for column in self.columns:
			new_vColumn = vColumn(column, parent = copy_vDataframe, transformations = self[column].transformations)
			setattr(copy_vDataframe, column, new_vColumn)
			setattr(copy_vDataframe, column[1:-1], new_vColumn)
		return (copy_vDataframe)
	# 
	def corr(self, 
			 columns: list = [],
			 method: str = "pearson", 
			 cmap: str = "",
			 round_nb: int = 3, 
			 focus: str = "",
			 show: bool = True):
		if not(method in ("pearson", "kendall", "spearman", "biserial", "cramer")):
			raise ValueError("The parameter 'method' must be in pearson|kendall|spearman|biserial|cramer")
		if (focus == ""):
			return (self.aggregate_matrix(method = method, columns = columns, cmap = cmap, round_nb = round_nb, show = show))
		else:
			return (self.aggregate_vector(focus, method = method, columns = columns, cmap = cmap, round_nb = round_nb, show = show))
	# 
	def cov(self, columns: list = [], focus: str = ""):
		if (focus == ""):
			return (self.aggregate_matrix(method = "cov", columns = columns))
		else:
			return (self.aggregate_vector(focus, method = "cov", columns = columns))
	# 
	def count(self, 
			  columns = [], 
			  percent: bool = True):
		columns = self.get_columns() if not(columns) else columns
		func = ["count", "percent"] if (percent) else ["count"]
		return (self.aggregate(func = func, columns = columns))
	#
	def cummax(self, 
			   name: str, 
			   column: str, 
			   by: list = [], 
			   order_by: list = []):
		return (self.rolling(name = name, aggr = "max", column = column, preceding = "UNBOUNDED", following = 0, by = by, order_by = order_by))
	#
	def cummin(self, 
			   name: str, 
			   column: str, 
			   by: list = [], 
			   order_by: list = []):
		return (self.rolling(name = name, aggr = "min", column = column, preceding = "UNBOUNDED", following = 0, by = by, order_by = order_by))
	#
	def cumprod(self, 
			    name: str, 
			    column: str, 
			    by: list = [], 
			    order_by: list = []):
		return (self.rolling(name = name, aggr = "", column = column, preceding = "UNBOUNDED", following = 0, expr = "DECODE(ABS(MOD(SUM(CASE WHEN {} < 0 THEN 1 ELSE 0 END) #, 2)), 0, 1, -1) * POWER(10, SUM(LOG(ABS({}))) #)", by = by, order_by = order_by))
	#
	def cumsum(self, 
			    name: str, 
			    column: str, 
			    by: list = [], 
			    order_by: list = []):
		return (self.rolling(name = name, aggr = "sum", column = column, preceding = "UNBOUNDED", following = 0, by = by, order_by = order_by))
	# 
	def current_relation(self):
		try:
			import sqlparse
			return (sqlparse.format(self.genSQL(), reindent=True))
		except:
			return (self.genSQL())
	#
	def datecol(self):
		columns = []
		for column in self.get_columns():
			if self[column].isdate():
				columns += [column]
		return (columns)
	# 
	def describe(self, 
				 method: str = "auto", 
				 columns: list = [], 
				 unique: bool = True):
		for i in range(len(columns)):
			columns[i] = '"' + columns[i].replace('"', '') + '"'
		if (method == "auto"):
			if not(columns):
				method = "categorical" if not(self.numcol()) else "numerical"
			else:
				method = "numerical"
				for column in columns:
					if (column.category() not in ['float','int']):
						method = "categorical"
						break
		if (method == "numerical"):
			if not(columns):
				columns = self.numcol()
			query = []
			for column in columns:
				if self[column].isnum():
					if (self[column].transformations[-1][1] == "boolean"):
						query += [column + "::int"]
					else:
						query += [column]
				else:
					print("/!\\ Warning: The column '{}' is not numerical, it was ignored. To get information about all the different variables, please use the parameter method = 'categorical'.".format(column))
			if not(query):
				raise ValueError("There is no numerical columns in the vDataframe")
			query = "SELECT SUMMARIZE_NUMCOL({}) OVER () FROM {}".format(", ".join(query), self.genSQL())
			self.executeSQL(query, title = "Compute the descriptive statistics of all the numerical columns")
			query_result = self.cursor.fetchall()
			data = [item for item in query_result]
			matrix = [['column'], ['count'], ['mean'], ['std'], ['min'], ['25%'], ['50%'], ['75%'], ['max']]
			for row in data:
				for idx,val in enumerate(row):
					matrix[idx] += [val]
			if (unique):
				query = []
				try:
					for column in matrix[0][1:]:
						query += ["COUNT(DISTINCT {})".format('"' + column + '"')]
					query= "SELECT "+",".join(query) + " FROM " + self.genSQL()
					self.executeSQL(query, title = "Compute the cardinalities of all the elements in a single query")
					cardinality=self.cursor.fetchone()
					cardinality=[item for item in cardinality]
				except:
					cardinality = []
					for column in matrix[0][1:]:
						query = "SELECT COUNT(DISTINCT {}) FROM {}".format('"' + column + '"',self.genSQL())
						self.executeSQL(query, title = "New attempt: Compute one per one all the cardinalities")
						cardinality += [self.cursor.fetchone()[0]]
				matrix += [['unique'] + cardinality]
			values = {"index" : matrix[0][1:len(matrix[0])]}
			del(matrix[0])
			for column in matrix:
				values[column[0]] = column[1:len(column)]
		elif (method == "categorical"):
			if not(columns):
				columns = self.get_columns()
			values = {"index" : [], "dtype" : [], "unique" : [], "count" : [], "top" : [], "top_percent" : []}
			for column in columns:
				values["index"] += [column]
				values["dtype"] += [self[column].ctype()]
				values["unique"] += [self[column].nunique()]
				values["count"] += [self[column].count()]
				query = "SELECT SUMMARIZE_CATCOL({}::varchar USING PARAMETERS TOPK = 1) OVER () FROM {} WHERE {} IS NOT NULL OFFSET 1".format(column, self.genSQL(), column)
				self.executeSQL(query, title = "Compute the TOP1 feature")
				result = self.cursor.fetchone()
				if (result == None):
					values["top"] += [None]
					values["top_percent"] += [None]
				else:
					values["top"] += [result[0]]
					values["top_percent"] += [round(result[2], 3)]
		else:
			raise ValueError("The parameter 'method' must be in auto|numerical|categorical")
		return (tablesample(values, table_info = False))
	#
	def drop(self, columns: list = []):
		for column in columns:
			if ('"' + column.replace('"', '') + '"' in self.get_columns()):
				self[column].drop()
			else:
				print("/!\\ Warning: Column '{}' is not in the vDataframe.".format(column))
		return (self)
	#
	def drop_duplicates(self, columns: list = []):
		count = self.duplicated(columns = columns, count = True)
		if (count):
			name = "_vpython_duplicated_index" + str(np.random.randint(10000000)) + "_"
			columns = self.get_columns() if not(columns) else ['"' + column.replace('"', '') + '"' for column in columns]
			self.eval(name = name, expr = "ROW_NUMBER() OVER (PARTITION BY {})".format(", ".join(columns)), print_info = False)
			self.filter(expr = '"{}" = 1'.format(name))
			self.exclude_columns += ['"{}"'.format(name)]
		else:
			print("/!\\ Warning: No duplicates detected")
		return (self)
	# 
	def dropna(self, columns: list = [], print_info: bool = True):
		if (columns == []):
			columns = self.get_columns()
		total = self.shape()[0]
		for column in columns:
			self[column].dropna(print_info = False)
		if (print_info):
			total -= self.shape()[0]
			if (total == 0):
				print("/!\\ Warning: Nothing was dropped")
			elif (total == 1):
				print("1 element was dropped")
			else:
				print("{} elements were dropped".format(total))
		return (self)
	# 
	def dsn_restart(self):
		from vertica_ml_python import vertica_cursor
		self.cursor = vertica_cursor(self.dsn)
		return (self)
	# 
	def dtypes(self):
		values = {"index" : [], "dtype" : []}
		for column in self.get_columns():
			values["index"] += [column]
			values["dtype"] += [self[column].ctype()]
		return (tablesample(values, table_info = False))
	#
	def duplicated(self, columns: list = [], count: bool = False):
		columns = self.get_columns() if not(columns) else ['"' + column.replace('"', '') + '"' for column in columns]
		query = "(SELECT *, ROW_NUMBER() OVER (PARTITION BY {}) AS duplicated_index FROM {}) duplicated_index_table WHERE duplicated_index > 1"
		query = query.format(", ".join(columns), self.genSQL())
		if (count):
			query = "SELECT COUNT(*) FROM " + query
			self.executeSQL(query = query, title = "Computing the Number of duplicates")
			return self.cursor.fetchone()[0]
		else:
			query = "SELECT {}, MAX(duplicated_index) AS occurrence FROM ".format(", ".join(columns)) + query + " GROUP BY {}".format(", ".join(columns))
			return (to_tablesample(query, self.cursor, name = "Duplicated Rows"))
	#
	def empty(self):
		return (self.get_columns() == [])
	# 
	def eval(self, name: str, expr: str, print_info: bool = True):
		try:
			name = '"' + name.replace('"', '') + '"'
			tmp_name = "_vpython" + str(np.random.randint(10000000)) + "_"
			self.executeSQL(query = "DROP TABLE IF EXISTS " + tmp_name, title = "Drop the existing generated table")
			query = "CREATE TEMPORARY TABLE {} AS SELECT {} AS {} FROM {} LIMIT 20".format(tmp_name, expr, name, self.genSQL())
			self.executeSQL(query = query, title = "Create a temporary table to test if the new feature is correct")
			query = "SELECT data_type FROM columns WHERE column_name='{}' AND table_name='{}'".format(name.replace('"', ''), tmp_name)
			self.executeSQL(query = query, title = "Catch the new feature's type")
			ctype = self.cursor.fetchone()[0]
			self.executeSQL(query = "DROP TABLE IF EXISTS " + tmp_name, title = "Drop the temporary table")
			ctype = ctype if (ctype) else "undefined"
			category = category_from_type(ctype = ctype)
			vDataframe_maxfloor_length = len(max([self[column].transformations for column in self.get_columns()], key = len))
			vDataframe_minfloor_length = 0
			for column in self.get_columns():
				if ((column in expr) or (column.replace('"','') in expr)):
					vDataframe_minfloor_length = max(len(self[column].transformations), vDataframe_minfloor_length)
			for eval_floor_length in range(vDataframe_minfloor_length, vDataframe_maxfloor_length):
				try:
					self.cursor.execute("SELECT * FROM {} LIMIT 0".format(
						self.genSQL(transformations = {name : [1 for i in range(eval_floor_length)] + [expr]})))
					floor_length = eval_floor_length
					break
				except:
					floor_length = vDataframe_maxfloor_length
			floor_length = vDataframe_maxfloor_length if (vDataframe_minfloor_length >= vDataframe_maxfloor_length) else floor_length
			transformations = [('0', "undefined", "undefined") for i in range(floor_length)] + [(expr, ctype, category)]
			new_vColumn = vColumn(name, parent = self, transformations = transformations)
			setattr(self, name, new_vColumn)
			setattr(self, name.replace('"', ''), new_vColumn)
			self.columns += [name]
			if (print_info):
				print("The new vColumn {} was added to the vDataframe.".format(name))
			self.history += ["{" + time.strftime("%c") + "} " + "[Eval]: A new vColumn '{}' was added to the vDataframe.".format(name)]
			return (self)
		except:
			raise Exception("An error occurs during the creation of the new feature")
	# 
	def executeSQL(self, query: str, title: str = ""):
		if (self.query_on):
			try:
				import shutil
				screen_columns = shutil.get_terminal_size().columns
			except:
				screen_rows, screen_columns = os.popen('stty size', 'r').read().split()
			try:
				import sqlparse
				query_print = sqlparse.format(query, reindent = True)
			except:
				query_print = query
			if (isnotebook()):
				from IPython.core.display import HTML, display
				display(HTML("<h4 style = 'color : #444444; text-decoration : underline;'>" + title + "</h4>"))
				query_print = query_print.replace('\n',' <br>').replace('  ',' &emsp; ')
				display(HTML(query_print))
				display(HTML("<div style = 'border : 1px dashed black; width : 100%'></div>"))
			else:
				print("$ " + title + " $\n")
				print(query_print)
				print("-" * int(screen_columns) + "\n")
		start_time = time.time()
		self.cursor.execute(query)
		elapsed_time = time.time() - start_time
		if (self.time_on):
			try:
				import shutil
				screen_columns = shutil.get_terminal_size().columns
			except:
				screen_rows, screen_columns = os.popen('stty size', 'r').read().split()
			if (isnotebook()):
				from IPython.core.display import HTML,display
				display(HTML("<div><b>Elapsed Time : </b> " + str(elapsed_time) + "</div>"))
				display(HTML("<div style = 'border : 1px dashed black; width : 100%'></div>"))
			else:
				print("Elapsed Time: " + str(elapsed_time))
				print("-" * int(screen_columns) + "\n")
		return (self.cursor)
	#
	def expected_store_usage(self, unit = 'b'):
		if (unit.lower() == 'kb'):
			div_unit = 1024
		elif (unit.lower() == 'mb'):
			div_unit = 1024 * 1024
		elif (unit.lower() == 'gb'):
			div_unit = 1024 * 1024 * 1024
		elif (unit.lower() == 'tb'):
			div_unit = 1024 * 1024 * 1024 * 1024
		else:
			unit = 'b'
			div_unit = 1
		total = 0
		total_expected = 0
		columns = self.get_columns()
		values = self.aggregate(func = ["count"], columns = columns).transpose().values
		values["index"] = ["expected_size ({})".format(unit), "max_size ({})".format(unit), "type"]
		for column in columns:
			ctype = self[column].ctype()
			if (ctype[0:4] == "date") or (ctype[0:4] == "time") or (ctype[0:8] == "interval"):
				maxsize, expsize = 8, 8
			elif (ctype[0:3] == "int"):
				maxsize, expsize = 64, self[column].store_usage()
			elif (ctype[0:4] == "bool"):
				maxsize, expsize = 1, 1
			elif (ctype[0:5] == "float"):
				maxsize, expsize = 8, 8
			elif (ctype[0:7] == "numeric"):
				size = sum([int(item) for item in ctype.split("(")[1].split(")")[0].split(",")])
				maxsize, expsize = size, size
			elif (ctype[0:7] == "varchar"):
				size = int(ctype.split("(")[1].split(")")[0])
				maxsize, expsize = size, self[column].store_usage()
			elif (ctype[0:4] == "char"):
				size = int(ctype.split("(")[1].split(")")[0])
				maxsize, expsize = size, size
			else:
				maxsize, expsize = 80, self[column].store_usage()
			maxsize /= div_unit
			expsize /= div_unit
			values[column] = [expsize, values[column][0] * maxsize, ctype]
			total_expected += values[column][0]
			total += values[column][1]
		values["separator"] = [len(columns) * self.shape()[0] / div_unit, len(columns) * self.shape()[0] / div_unit, ""]
		total += values["separator"][0]
		total_expected += values["separator"][0]
		values["header"] = [(sum([len(item) for item in columns]) + len(columns)) / div_unit, (sum([len(item) for item in columns]) + len(columns)) / div_unit, ""]
		total += values["header"][0]
		total_expected += values["header"][0]
		values["rawsize"] = [total_expected, total, ""]
		return (tablesample(values = values, table_info = False).transpose())
	# 
	def fillna(self,
			   val: dict = {},
			   method: dict = {},
			   numeric_only: bool = False,
			   print_info: bool = False):
		if (not(val) and not(method)):
			for column in self.get_columns():
				if (numeric_only):
					if self[column].isnum():
						self[column].fillna(method = "auto", print_info = print_info)
				else:
					self[column].fillna(method = "auto", print_info = print_info)
		else:
			for column in val:
				self[column].fillna(val = val[column], print_info = print_info)
			for column in method:
				self[column].fillna(method = method[column], print_info = print_info)
		return (self)
	# 
	def filter(self, 
			   expr: str = "", 
			   conditions: list = [],
			   print_info: bool = True):
		count = self.shape()[0]
		if not(expr):
			for condition in conditions:
				self.filter(expr = condition, print_info = False)
			count -= self.shape()[0]
			if (count > 1):
				if (print_info):
					print("{} elements were filtered".format(count))
				self.history += ["{" + time.strftime("%c") + "} " + "[Filter]: {} elements were filtered using the filter '{}'".format(count, conditions)]
			elif (count == 1):
				if (print_info):
					print("{} element was filtered".format(count))
				self.history += ["{" + time.strftime("%c") + "} " + "[Filter]: {} element was filtered using the filter '{}'".format(count, conditions)]
			else:
				if (print_info):
					print("Nothing was filtered.")
		else:
			max_pos = 0
			for column in self.columns:
				max_pos = max(max_pos, len(self[column].transformations) - 1)
			self.where += [(expr, max_pos)]
			try:
				count -= self.shape()[0]
			except:
				del self.where[-1]
				if (print_info):
					print("/!\\ Warning: The expression '{}' is incorrect.\nNothing was filtered.".format(expr))
			if (count > 1):
				if (print_info):
					print("{} elements were filtered".format(count))
				self.history += ["{" + time.strftime("%c") + "} " + "[Filter]: {} elements were filtered using the filter '{}'".format(count, expr)]
			elif (count == 1):
				if (print_info):
					print("{} element was filtered".format(count))
				self.history += ["{" + time.strftime("%c") + "} " + "[Filter]: {} element was filtered using the filter '{}'".format(count, expr)]
			else:
				del self.where[-1]
				if (print_info):
					print("Nothing was filtered.")
		return (self)
	#
	def first(self, ts: str, offset: str):
		ts = '"' + ts.replace('"', '') + '"'
		query = "SELECT (MIN({}) + '{}'::interval)::varchar FROM {}".format(ts, offset, self.genSQL())
		self.cursor.execute(query)
		first_date = self.cursor.fetchone()[0]
		expr = "{} <= '{}'".format(ts, first_date)
		self.filter(expr)
		return (self)
	#
	def get_columns(self):
		columns = [column for column in self.columns]
		for column in self.exclude_columns:
			if column in columns:
				columns.remove(column)
		return(columns)
	# 
	def get_dummies(self, 
					columns: list = [],
					max_cardinality: int = 12,
					prefix: str = "", 
					prefix_sep: str = "_", 
					drop_first: bool = False, 
					use_numbers_as_suffix: bool = False):
		columns = self.get_columns() if not(columns) else ['"' + column.replace('"', '') + '"' for column in columns]
		for column in columns:
			if (self[column].nunique() < max_cardinality):
				self[column].get_dummies()
		return (self)
	# 
	def groupby(self, columns: list, expr: list = []):
		columns = ['"' + column.replace('"', '') + '"' for column in columns]
		query = "SELECT {}".format(", ".join(columns + expr)) + " FROM {} GROUP BY {}".format(self.genSQL(), ", ".join(columns))
		return (self.vdf_from_relation("(" + query + ") groupby_table", "groupby", "[Groupby]: The columns were group by {}".format(", ".join(columns))))
	# 
	def head(self, limit: int = 5):
		return (self.tail(limit = limit))
	# 
	def hexbin(self,
		   	   columns: list,
		   	   method: str = "count",
		       of: str = "",
		       cmap: str = '',
		       gridsize: int = 10,
		       color: str = "white"):
		if not(cmap):
			from matplotlib.colors import LinearSegmentedColormap
			cmap = LinearSegmentedColormap.from_list("vml", ["#FFFFFF", "#214579"], N = 1000)
		from vertica_ml_python.plot import hexbin
		hexbin(self, columns, method, of, cmap, gridsize, color)
		return (self)
	# 
	def hist(self,
			 columns: list,
			 method: str = "density",
			 of: str = "",
			 max_cardinality: tuple = (6, 6),
			 h: tuple = (None, None),
			 limit_distinct_elements: int = 200,
			 hist_type: str = "auto"):
		stacked = True if (hist_type.lower() == "stacked") else False
		multi = True if (hist_type.lower() == "multi") else False
		if (len(columns) == 1):
			self[columns[0]].hist(method, of, 6, 0, 0)
		else:
			if (multi):
				from vertica_ml_python.plot import multiple_hist
				h_0 = h[0] if (h[0]) else 0
				multiple_hist(self, columns, method, of, h_0)
			else:
				from vertica_ml_python.plot import hist2D
				hist2D(self, columns, method, of, max_cardinality, h, limit_distinct_elements, stacked)
		return (self)
	# 
	def info(self):
		if (len(self.history) == 0):
			print("The vDataframe was never modified.")
		elif (len(self.history)==1):
			print("The vDataframe was modified with only one action: ")
			print(" * "+self.history[0])
		else:
			print("The vDataframe was modified many times: ")
			for modif in self.history:
				print(" * "+modif)
		return (self)
	#
	def isin(self, val: dict):
		n = len(val[list(val.keys())[0]])
		isin = []
		for i in range(n):
			tmp_query = []
			for column in val:
				if (val[column][i] == None):
					tmp_query += ['"' + column.replace('"', '') + '"' + " IS NULL"]
				else:
					tmp_query += ['"' + column.replace('"', '') + '"' + " = '{}'".format(val[column][i])]
			query = "SELECT * FROM {} WHERE ".format(self.genSQL()) + " AND ".join(tmp_query) + " LIMIT 1"
			self.cursor.execute(query)
			isin += [self.cursor.fetchone() != None] 
		return (isin)
	#
	def join(self, 
			 input_relation: str = "", 
			 vdf = None, 
			 on: dict = {},
			 how: str = 'natural',
			 expr1: list = [],
			 expr2: list = []):
		on_join = ["x." + elem + " = y." + on[elem] for elem in on]
		on_join = " AND ".join(on_join)
		on_join = " ON " + on_join if (on_join) else ""
		first_relation = self.genSQL(final_table_name = "x")
		second_relation = input_relation + " AS y" if not(vdf) else vdf.genSQL(final_table_name = "y")
		expr1 = ["x.{}".format(elem) for elem in expr1]
		expr2 = ["y.{}".format(elem) for elem in expr2]
		expr = expr1 + expr2
		expr = "*" if not(expr) else ", ".join(expr)
		table = "SELECT {} FROM {} {} JOIN {} {}".format(expr, first_relation, how.upper(), second_relation, on_join)
		return (self.vdf_from_relation("(" + table + ") join_table", "join", "[Join]: Two relations were joined together"))
	#
	def kurt(self, columns: list = []):
		return self.kurtosis(columns = columns)
	def kurtosis(self, columns: list = []):
		stats = self.statistics(columns = columns, skew_kurt_only = True)
		for column in stats.values:
			del(stats.values[column][0])
		return (stats.transpose())
	#
	def last(self, ts: str, offset: str):
		ts = '"' + ts.replace('"', '') + '"'
		query = "SELECT (MAX({}) - '{}'::interval)::varchar FROM {}".format(ts, offset, self.genSQL())
		self.cursor.execute(query)
		last_date = self.cursor.fetchone()[0]
		expr = "{} >= '{}'".format(ts, last_date)
		self.filter(expr)
		return (self)
	#
	def load(self, offset: int = -1):
		save =  self.saving[offset]
		vdf = {}
		exec(save, globals(), vdf)
		vdf = vdf["vdf_save"]
		vdf.cursor = self.cursor
		return (vdf)
	#
	def mad(self, columns = []):
		return (self.aggregate(func = ["mad"], columns = columns))
	#
	def max(self, columns = []):
		return (self.aggregate(func = ["max"], columns = columns))
	#
	def mean(self, columns = []):
		return (self.aggregate(func = ["avg"], columns = columns))
	#
	def median(self, columns = []):
		return (self.aggregate(func = ["median"], columns = columns))
	#
	def memory_usage(self):
		import sys
		total = sys.getsizeof(self.columns) + sys.getsizeof(self.where) + sys.getsizeof(self.history) + sys.getsizeof(self.main_relation)
		total += sys.getsizeof(self.input_relation) + sys.getsizeof(self.schema) + sys.getsizeof(self.dsn) + sys.getsizeof(self) 
		total += sys.getsizeof(self.query_on) + sys.getsizeof(self.exclude_columns)
		values = {"index": [], "value": []}
		values["index"] += ["object"]
		values["value"] += [total]
		for column in self.columns:
			values["index"] += [column] 
			values["value"] += [self[column].memory_usage()]
			total += self[column].memory_usage()
		values["index"] += ["total"]
		values["value"] += [total]
		return (tablesample(values = values, table_info = False))
	#
	def min(self, columns = []):
		return (self.aggregate(func = ["min"], columns = columns))
	# 
	def normalize(self, columns = [], method = "zscore"):
		columns = self.numcol() if not(columns) else ['"' + column.replace('"', '') + '"' for column in columns]
		for column in columns:
			self[column].normalize(method = method)
		return (self)
	#
	def numcol(self):
		columns = []
		for column in self.get_columns():
			if self[column].isnum():
				columns += [column]
		return (columns)
	# 
	def outliers(self,
				 columns: list = [],
				 name: str = "distribution_outliers",
				 threshold: float = 3.0):
		columns = ['"' + column.replace('"', '') + '"' for column in columns] if (columns) else self.numcol()
		result = self.aggregate(func = ["std", "avg"], columns = columns).values
		conditions = []
		for idx, elem in enumerate(result["index"]):
			conditions += ["ABS({} - {}) / {} > {}".format(elem, result["avg"][idx], result["std"][idx], threshold)]
		self.eval(name, "(CASE WHEN {} THEN 1 ELSE 0 END)".format(" OR ".join(conditions)))
		return (self)
	# 
	def pivot_table(self,
					columns: list,
					method: str = "count",
					of: str = "",
					h: tuple = (None, None),
					max_cardinality: tuple = (20, 20),
					show: bool = True,
					cmap: str = '',
					limit_distinct_elements: int = 1000,
					with_numbers: bool = True):
		if not(cmap):
			from matplotlib.colors import LinearSegmentedColormap
			cmap = LinearSegmentedColormap.from_list("vml", ["#FFFFFF", "#214579"], N = 1000)
		from vertica_ml_python.plot import pivot_table
		return (pivot_table(self, columns, method, of, h, max_cardinality, show, cmap, limit_distinct_elements, with_numbers))
	#
	def plot(self, 
		     ts: str,
			 columns: list = [], 
			 start_date: str = "",
		     end_date: str = ""):
		from vertica_ml_python.plot import multi_ts_plot
		multi_ts_plot(self, ts, columns, start_date, end_date)	
		return (self)
	#
	def prod(self, columns = []):
		return (self.product(columns = columns))
	def product(self, columns = []):
		return (self.aggregate(func = ["prod"], columns = columns))
	#
	def quantile(self, q: list, columns = []):
		return (self.aggregate(func = ["{}%".format(float(item)*100) for item in q], columns = columns))
	#
	def rank(self, order_by: list, method: str = "first", by: list = [], name = ""):
		if (method not in ('first', 'dense', 'percent')):
			raise ValueError("The parameter 'method' must be in first|dense|percent")
		elif (method == "first"):
			func = "RANK"
		elif (method == "dense"):
			func = "DENSE_RANK"
		elif (method == "percent"):
			func = "PERCENT_RANK"
		partition = "" if not(by) else "PARTITION BY " + ", ".join(['"' + column.replace('"', '') + '"' for column in by])
		name_prefix = "_".join([column.replace('"', '') for column in order_by]) 
		name_prefix += "_by_" + "_".join([column.replace('"', '') for column in by]) if (by) else ""
		name = name if (name) else method + "_rank_" + name_prefix
		order = "ORDER BY {}".format(", ".join(['"' + column.replace('"', '') + '"' for column in order_by]))
		expr = "{}() OVER ({} {})".format(func, partition, order)
		return (self.eval(name = name, expr = expr))
	#
	def rolling(self, 
				name: str,
				aggr: str, 
				column: str, 
				preceding, 
				following, 
				expr: str = "",
				by: list = [], 
				order_by: list = [], 
				method: str = "rows",
				rule: str = "auto"):
		if not(rule.lower() in ("auto", "past", "future")):
			raise ValueError("The parameter 'rule' must be in auto|past|future\n'auto' will create a moving window from the past to the future (preceding-following)\n'past' will create a moving window in the past (preceding-preceding)\n'future' will create a moving window in the future (following-following)\n")
		if (rule.lower() == "past"):
			rule_p, rule_f = "PRECEDING", "PRECEDING"
		elif (rule.lower() == "future"):
			rule_p, rule_f = "FOLLOWING", "FOLLOWING"
		else:
			rule_p, rule_f = "PRECEDING", "FOLLOWING"
		if (method not in ('range', 'rows')):
			raise ValueError("The parameter 'method' must be in rows|range")
		column = '"' + column.replace('"', '') + '"'
		by = "" if not(by) else "PARTITION BY " + ", ".join(['"' + col.replace('"', '') + '"' for col in by])
		order_by = [column] if not(order_by) else ['"' + column.replace('"', '') + '"' for column in order_by]
		expr = "{}({})".format(aggr.upper(), column) if not(expr) else expr.replace("{}", column)
		expr = expr + " #" if '#' not in expr else expr
		if (method == "rows"):
			preceding = "{}".format(preceding) if (str(preceding).upper() != "UNBOUNDED") else "UNBOUNDED"
			following = "{}".format(following) if (str(following).upper() != "UNBOUNDED") else "UNBOUNDED"
		else:
			preceding = "'{}'".format(preceding) if (str(preceding).upper() != "UNBOUNDED") else "UNBOUNDED"
			following = "'{}'".format(following) if (str(following).upper() != "UNBOUNDED") else "UNBOUNDED"
		preceding, following = "{} {}".format(preceding, rule_p), "{} {}".format(following, rule_f)
		expr = expr.replace('#'," OVER ({} ORDER BY {} {} BETWEEN {} AND {})".format(by, ", ".join(order_by), method.upper(), preceding, following))
		return (self.eval(name = name, expr = expr))
	# 
	def sample(self, x: float):
		query = "SELECT {} FROM (SELECT *, RANDOM() AS random FROM {}) x WHERE random < {}".format(", ".join(self.get_columns()), self.genSQL(), x)
		sample = to_tablesample(query, self.cursor)
		sample.name = "Sample({}) of ".format(x) + self.input_relation
		for column in sample.values:
			sample.dtype[column] = self[column].ctype()
		return (sample)
	#
	def save(self):
		save = 'vdf_save = vDataframe("", empty = True)'
		save += '\nvdf_save.dsn = \'{}\''.format(self.dsn)
		save += '\nvdf_save.input_relation = \'{}\''.format(self.input_relation)
		save += '\nvdf_save.main_relation = \'{}\''.format(self.main_relation)
		save += '\nvdf_save.schema = \'{}\''.format(self.schema)
		save += '\nvdf_save.columns = {}'.format(self.columns)
		save += '\nvdf_save.exclude_columns = {}'.format(self.exclude_columns)
		save += '\nvdf_save.where = {}'.format(self.where)
		save += '\nvdf_save.query_on = {}'.format(self.query_on)
		save += '\nvdf_save.time_on = {}'.format(self.time_on)
		save += '\nvdf_save.order_by = {}'.format(self.order_by)
		save += '\nvdf_save.history = {}'.format(self.history)
		save += '\nvdf_save.saving = {}'.format(self.saving)
		for column in self.columns:
			save += '\nsave_vColumn = vColumn(\'{}\', parent = vdf_save, transformations = {})'.format(column, self[column].transformations)
			save += '\nsetattr(vdf_save, \'{}\', save_vColumn)'.format(column)
			save += '\nsetattr(vdf_save, \'{}\', save_vColumn)'.format(column[1:-1])
		self.saving += [save]
		return (self)
	# 
	def scatter(self,
			  	columns: list,
			  	catcol: str = "",
			  	max_cardinality: int = 3,
			  	cat_priority: list = [],
			  	with_others: bool = True,
			  	max_nb_points: int = 20000):
		catcol = [catcol] if (catcol) else []
		if (len(columns) == 2):
			from vertica_ml_python.plot import scatter2D
			scatter2D(self, columns + catcol, max_cardinality, cat_priority, with_others, max_nb_points)	
		elif (len(columns) == 3):
			from vertica_ml_python.plot import scatter3D
			scatter3D(self, columns + catcol, max_cardinality, cat_priority, with_others, max_nb_points)	
		else:
			raise ValueError("The parameter 'columns' must be a list of 2 or 3 columns")
		return (self)
	# 
	def scatter_matrix(self, columns: list = []):
		from vertica_ml_python.plot import scatter_matrix
		scatter_matrix(self, columns)
		return (self)	
	#
	def select(self, columns: list):
		copy_vDataframe = self.copy()
		columns = ['"' + column.replace('"', '') + '"' for column in columns]
		for column in copy_vDataframe.get_columns():
			column_tmp = '"' + column.replace('"', '') + '"'
			if (column_tmp not in columns):
				copy_vDataframe[column_tmp].drop(add_history = False) 
		return (copy_vDataframe)
	#
	def sem(self, columns = []):
		return (self.aggregate(func = ["sem"], columns = columns))
	#
	def sessionize(self,
				   ts: str,
				   by: list = [],
				   session_threshold = "30 minutes",
				   name = "session_id"):
		partition = "PARTITION BY {}".format(", ".join(['"' + column.replace('"', '') + '"' for column in by])) if (by) else ""
		expr = "CONDITIONAL_TRUE_EVENT({} - LAG({}) > '{}') OVER ({} ORDER BY {})".format(ts, ts, session_threshold, partition, ts)
		return (self.eval(name = name, expr = expr))
	# 
	def set_cursor(self, cursor):
		self.cursor = cursor
		return (self)
	# 
	def set_dsn(self, dsn: str):
		self.dsn = dsn
		return (self)
	# 
	def shape(self):
		query = "SELECT COUNT(*) FROM {}".format(self.genSQL())
		self.cursor.execute(query)
		return (self.cursor.fetchone()[0], len(self.get_columns()))
	#
	def skew(self, columns: list = []):
		return self.skewness(columns = columns)
	def skewness(self, columns: list = []):
		stats = self.statistics(columns = columns, skew_kurt_only = True)
		for column in stats.values:
			del(stats.values[column][1])
		return (stats.transpose())
	# 
	def sort(self, columns: list):
		columns = ['"' + column.replace('"', '') + '"' for column in columns]
		max_pos = 0
		for column in columns:
			if (column not in self.get_columns()):
				raise NameError("The column {} doesn't exist".format(column))
		for column in self.columns:
			max_pos = max(max_pos, len(self[column].transformations) - 1)
		for column in columns:
			self.order_by += [(column, max_pos)]
		return (self)
	# 
	def sql_on_off(self):
		self.query_on = not(self.query_on)
		return (self)
	#
	def statistics(self, 
				   columns: list = [], 
				   skew_kurt_only: bool = False):
		columns = self.numcol() if not(columns) else ['"' + column.replace('"', '') + '"' for column in columns]
		query = []
		if (skew_kurt_only):
			stats = self.aggregate(func = ["count", "avg", "stddev"], columns = columns).transpose()
		else:
			stats = self.aggregate(func = ["count", "avg", "stddev", "min", "10%", "25%", "median", "75%", "90%", "max"], columns = columns).transpose()
		for column in columns:
			cast = "::int" if (self[column].ctype()[0:4] == "bool") else ""
			count, avg, std = stats.values[column][0], stats.values[column][1], stats.values[column][2]
			if (count == 0): 
				query += ["NULL", "NULL"]
			elif ((count == 1) or (std == 0)): 
				query += ["0", "-3"]
			else:
				for k in range(3, 5):
					expr = "AVG(POWER(({}{} - {}) / {}, {}))".format(column, cast, avg, std, k) 
					if (count > 3):
						expr += "* {} - 3 * {}".format(count * count * (count + 1) / (count - 1) / (count - 2) / (count - 3), (count -1) * (count - 1) / (count - 2) / (count - 3)) if (k == 4) else "* {}".format(count * count / (count - 1) / (count - 2))
					else:
						expr += "* - 3" if (k == 4) else ""
						expr += "* {}".format(count * count / (count - 1) / (count - 2)) if (count == 3) else ""
					query += [expr]
		query = "SELECT {} FROM {}".format(', '.join(query), self.genSQL())
		self.executeSQL(query, title = "COMPUTE KURTOSIS AND SKEWNESS")
		result = [item for item in self.cursor.fetchone()]
		if (skew_kurt_only):
			values = {"index" : []}
			for column in columns:
				values[column] = []
			stats = tablesample(values = values, table_info = False)
		i = 0
		for column in columns:
			stats.values[column] += result[i:i + 2]
			i += 2
		stats.values["index"] += ["skewness", "kurtosis"]
		return (stats)
	#
	def std(self, columns = []):
		return (self.aggregate(func = ["stddev"], columns = columns))
	#
	def sum(self, columns = []):
		return (self.aggregate(func = ["sum"], columns = columns))
	# 
	def tail(self, limit: int = 5, offset: int = 0):
		tail = to_tablesample("SELECT * FROM {} LIMIT {} OFFSET {}".format(self.genSQL(), limit, offset), self.cursor)
		tail.count = self.shape()[0]
		tail.offset = offset
		tail.name = self.input_relation
		for column in tail.values:
			tail.dtype[column] = self[column].ctype()
		return (tail)
	# 
	def time_on_off(self):
		self.time_on = not(self.time_on)
		return (self)
	#
	def to_csv(self, 
			   name: str,
			   path: str = '', 
			   sep: str = ',',
			   na_rep: str = '',
			   quotechar: str = '"',
			   usecols: list = [],
			   header: bool = True,
			   new_header: list = [],
			   order_by: list = [],
			   nb_row_per_work: int = 0):
		file = open("{}{}.csv".format(path, name), "w+") 
		columns = self.get_columns() if not(usecols) else ['"' + column.replace('"', '') + '"' for column in usecols]
		if (new_header) and (len(new_header) != len(columns)):
			raise ValueError("The header has an incorrect number of columns")
		elif (new_header):
			file.write(sep.join(new_header))
		elif (header):
			file.write(sep.join([column.replace('"', '') for column in columns]))
		total = self.shape()[0]
		current_nb_rows_written = 0
		limit = total if (nb_row_per_work <= 0) else nb_row_per_work
		while (current_nb_rows_written < total):
			self.cursor.execute("SELECT {} FROM {} LIMIT {} OFFSET {}".format(", ".join(columns), self.genSQL(), limit, current_nb_rows_written))
			result = self.cursor.fetchall()
			for row in result:
				tmp_row = []
				for item in row:
					if (type(item) == str):
						tmp_row += [quotechar + item + quotechar]
					elif (item == None):
						tmp_row += [na_rep]
					else:
						tmp_row += [str(item)]
				file.write("\n" + sep.join(tmp_row))
			current_nb_rows_written += limit
		file.close()
		return (self)
	# 
	def to_db(self,
			  name: str,
			  usecols: list = [],
			  relation_type: str = "view",
			  inplace: bool = False):
		if (relation_type not in ["view","temporary","table"]):
			raise ValueError("The parameter mode must be in view|temporary|table\nNothing was saved.")
		if (relation_type == "temporary"):
			relation_type += " table" 
		usecols = "*" if not(usecols) else ", ".join(['"' + column.replace('"', '') + '"' for column in usecols])
		query = "CREATE {} {} AS SELECT {} FROM {}".format(relation_type.upper(), name, usecols, self.genSQL())
		self.executeSQL(query = query, title = "Create a new " + relation_type + " to save the vDataframe")
		self.history += ["{" + time.strftime("%c") + "} " + "[Save]: The vDataframe was saved into a {} named '{}'.".format(relation_type, name)]
		if (inplace):
			query_on, time_on, history, saving = self.query_on, self.time_on, self.history, self.saving
			self.__init__(name, self.cursor)
			self.history = history
			self.query_on = query_on
			self.time_on = time_on
		return (self)
	# 
	def to_pandas(self):
		import pandas as pd
		query = "SELECT * FROM {}".format(self.genSQL())
		self.cursor.execute(query)
		column_names = [column[0] for column in self.cursor.description]
		query_result = self.cursor.fetchall()
		data = [list(item) for item in query_result]
		df = pd.DataFrame(data)
		df.columns = column_names
		return (df)
	#
	def to_vdf(self, name: str):
		self.save()
		file = open("{}.vdf".format(name), "w+") 
		file.write(self.saving[-1])
		file.close()
		return (self)
	#
	def var(self, columns = []):
		return (self.aggregate(func = ["variance"], columns = columns))
	#
	def vdf_from_relation(self, table: str, func: str, history: str):
		vdf = vDataframe("", empty = True)
		vdf.dsn = self.dsn
		vdf.input_relation = self.input_relation
		vdf.main_relation = table
		vdf.schema = self.schema
		vdf.cursor = self.cursor
		vdf.query_on = self.query_on
		vdf.time_on = self.time_on
		self.executeSQL(query = "DROP TABLE IF EXISTS _vpython_{}_test_; CREATE TEMPORARY TABLE _vpython_{}_test_ AS SELECT * FROM {} LIMIT 10;".format(func, func, table), title = "Test {}".format(func))
		self.executeSQL(query = "SELECT column_name, data_type FROM columns where table_name = '_vpython_{}_test_'".format(func), title = "SELECT NEW DATA TYPE AND THE COLUMNS NAME")
		result = self.cursor.fetchall()
		self.executeSQL(query = "DROP TABLE IF EXISTS _vpython_{}_test_;".format(func), title = "DROP TEMPORARY TABLE")
		vdf.columns = ['"' + item[0] + '"' for item in result]
		vdf.where = []
		vdf.order_by = []
		vdf.exclude_columns = []
		vdf.history = [item for item in self.history] + [history]
		vdf.saving = [item for item in self.saving]
		for column, ctype in result:
			column = '"' + column + '"'
			new_vColumn = vColumn(column, parent = vdf, transformations = [(column, ctype, category_from_type(ctype = ctype))])
			setattr(vdf, column, new_vColumn)
			setattr(vdf, column[1:-1], new_vColumn)
		return (vdf)
	# 
	#
	#
	#
	#
	#
	#
	def help(self):
		print("#############################")
		print("#                           #")
		print("# Vertica Virtual Dataframe #")
		print("#                           #")
		print("#############################")
		print("")
		print("The vDataframe is a Python object which will keep in mind all the user modifications in order "
				+"to use an optimised SQL query. It will send the query to the database which will use its "
				+"aggregations to compute fast results. It is created using a view or a table stored in the "
				+"user database and a database cursor. It will create for each column of the table a vColumn (Vertica"
				+" Virtual Column) which will store for each column its name, its imputations and allows to do easy "
				+"modifications and explorations.")
		print("")
		print("vColumn and vDataframe coexist and one can not live without the other. vColumn will use the vDataframe information and reciprocally." 
				+" It is imperative to understand both structures to know how to use the entire object.")
		print("")
		print("When the user imputes or filters the data, the vDataframe gets in memory all the transformations to select for each query "
				+"the needed data in the input relation.")
		print("")
		print("As the vDataframe will try to keep in mind where the transformations occurred in order to use the appropriate query," 
				+" it is highly recommended to save the vDataframe when the user has done a lot of transformations in order to gain in efficiency" 
				+" (using the to_db method). We can also see all the modifications using the history method.")
		print("")
		print("If you find any difficulties using vertica_ml_python, please contact me: badr.ouali@microfocus.com / I'll be glad to help.")
		print("")
		print("For more information about the different methods or the entire vDataframe structure, please see the entire documentation")
	# 
	def version(self):
		self.cursor.execute("SELECT version();")
		version = self.cursor.fetchone()[0]
		print("############################################################################################################") 
		print("#  __ __   ___ ____  ______ ____   __  ____      ___ ___ _          ____  __ __ ______ __ __  ___  ____    #")
		print("# |  |  | /  _|    \|      |    | /  ]/    |    |   |   | |        |    \|  |  |      |  |  |/   \|    \   #")
		print("# |  |  |/  [_|  D  |      ||  | /  /|  o  |    | _   _ | |        |  o  |  |  |      |  |  |     |  _  |  #")
		print("# |  |  |    _|    /|_|  |_||  |/  / |     |    |  \_/  | |___     |   _/|  ~  |_|  |_|  _  |  O  |  |  |  #")
		print("# |  :  |   [_|    \  |  |  |  /   \_|  _  |    |   |   |     |    |  |  |___, | |  | |  |  |     |  |  |  #")
		print("#  \   /|     |  .  \ |  |  |  \     |  |  |    |   |   |     |    |  |  |     | |  | |  |  |     |  |  |  #")
		print("#   \_/ |_____|__|\_| |__| |____\____|__|__|    |___|___|_____|    |__|  |____/  |__| |__|__|\___/|__|__|  #")
		print("#                                                                                                          #")
		print("############################################################################################################")
		print("#")
		print("# Author: Badr Ouali, Datascientist at Vertica")
		print("#")
		print("# You are currently using "+version)
		print("#")
		version = version.split("Database v")
		version_id = int(version[1][0])
		version_release = int(version[1][2])
		if (version_id > 8):
			print("# You have a perfectly adapted version for using vDataframe and Vertica ML")
		elif (version_id == 8):
			if (version_release > 0):
				print("# Your Vertica version is adapted for using vDataframe but you are quite limited for Vertica ML")
				print("# Go to your Vertica version documentation for more information")
				print("# /!\\ Some vDataframe queries can be really big because of the unavailability of a lot of functions")
				print("# /!\\ Some vDataframe functions could not work")
		else:
			print("# Your Vertica version is adapted for using vDataframe but you can not use Vertica ML")
			print("# Go to your Vertica version documentation for more information")
			print("# /!\\ Some vDataframe queries can be really big because of the unavailability of a lot of functions")
			print("# /!\\ Some vDataframe functions could not work")
		print("#")
		print("# For more information about the vDataframe you can use the help() method")
		return (version_id, version_release)

		




