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
import random, time
from vertica_ml_python.vcolumn import vColumn
from vertica_ml_python.utilities import print_table, isnotebook, tablesample, to_tablesample, category_from_type, str_column, column_check_ambiguous, check_types, columns_check, vdf_columns_names, schema_relation, convert_special_type
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
		check_types([("input_relation", input_relation, [str], False), ("dsn", dsn, [str], False), ("usecols", usecols, [list], False), ("schema", schema, [str], False), ("empty", empty, [bool], False)])
		self.VERTICA_ML_PYTHON_VARIABLES = {}
		self.VERTICA_ML_PYTHON_VARIABLES["*"] = -1
		if not(empty):
			if (cursor == None):
				from vertica_ml_python import vertica_cursor
				cursor = vertica_cursor(dsn)
			self.VERTICA_ML_PYTHON_VARIABLES["dsn"] = dsn
			if not(schema):
				schema, input_relation = schema_relation(input_relation)
			self.VERTICA_ML_PYTHON_VARIABLES["schema"], self.VERTICA_ML_PYTHON_VARIABLES["input_relation"] = schema.replace('"', ''), input_relation.replace('"', '')
			# Cursor to the Vertica Database
			self.VERTICA_ML_PYTHON_VARIABLES["cursor"] = cursor
			# All the columns of the vDataframe
			where = " AND LOWER(column_name) IN ({})".format(", ".join(["'{}'".format(elem.lower().replace("'", "''")) for elem in usecols])) if (usecols) else ""
			query = "(SELECT column_name, data_type FROM columns WHERE table_name = '{}' AND table_schema = '{}'{})".format(self.VERTICA_ML_PYTHON_VARIABLES["input_relation"].replace("'", "''"), self.VERTICA_ML_PYTHON_VARIABLES["schema"].replace("'", "''"), where)
			query += " UNION (SELECT column_name, data_type FROM view_columns WHERE table_name = '{}' AND table_schema = '{}'{})".format(self.VERTICA_ML_PYTHON_VARIABLES["input_relation"].replace("'", "''"), self.VERTICA_ML_PYTHON_VARIABLES["schema"].replace("'", "''"), where)
			cursor.execute(query)
			columns_dtype = cursor.fetchall()
			columns_dtype = [(str(item[0]), str(item[1])) for item in columns_dtype]
			columns = ['"{}"'.format(elem[0].replace('"', '_')) for elem in columns_dtype]
			if (columns != []):
				self.VERTICA_ML_PYTHON_VARIABLES["columns"] = columns
			else:
				raise ValueError("No table or views '{}' found.".format(self.VERTICA_ML_PYTHON_VARIABLES["input_relation"]))
			for col_dtype in columns_dtype:
				column, dtype = col_dtype[0], col_dtype[1]
				if ('"' in column):
					print("\u26A0 Warning: A double quote \" was found in the column {}, its alias was changed using underscores '_' to {}".format(column, column.replace('"', '_')))
				new_vColumn = vColumn('"{}"'.format(column.replace('"', '_')), parent = self, transformations = [('"{}"'.format(column.replace('"', '""')), dtype, category_from_type(dtype))])
				setattr(self, '"{}"'.format(column.replace('"', '_')), new_vColumn)
				setattr(self, column.replace('"', '_'), new_vColumn)
			# Number of columns of the main relation To know if the it was never modified
			self.VERTICA_ML_PYTHON_VARIABLES["*"] = -1 if (usecols) else len(columns_dtype)
			# Columns to not consider for the final query
			self.VERTICA_ML_PYTHON_VARIABLES["exclude_columns"] = []
			# Rules to filter
			self.VERTICA_ML_PYTHON_VARIABLES["where"] = []
			# Rules to sort the data
			self.VERTICA_ML_PYTHON_VARIABLES["order_by"] = ['' for i in range(100)]
			# Display the elapsed time during the query
			self.VERTICA_ML_PYTHON_VARIABLES["time_on"] = False
			# Display or not the sequal queries that are used during the vDataframe manipulation
			self.VERTICA_ML_PYTHON_VARIABLES["query_on"] = False
			# vDataframe history
			self.VERTICA_ML_PYTHON_VARIABLES["history"] = []
			# vDataframe saving
			self.VERTICA_ML_PYTHON_VARIABLES["saving"] = []
			# vDataframe main relation
			self.VERTICA_ML_PYTHON_VARIABLES["main_relation"] = '"{}"."{}"'.format(self.VERTICA_ML_PYTHON_VARIABLES["schema"], self.VERTICA_ML_PYTHON_VARIABLES["input_relation"])
	# 
	def __getitem__(self, index):
		try:
			return getattr(self, index)
		except:
			new_index = vdf_columns_names([index], self)
			if (len(new_index) == 1):
				return getattr(self, new_index[0])
			else: 
				raise
	def __setitem__(self, index, val):
		setattr(self, index, val)
	# 
	def __repr__(self):
		return self.head(limit = 5).__repr__()
	# 
	def __setattr__(self, attr, val):
		self.__dict__[attr] = val
	# 
	def genSQL(self, 
			   split: bool = False, 
			   transformations: dict = {}, 
			   force_columns: list = [],
			   final_table_name: str = "final_table",
			   return_without_alias: bool = False):
		if (self.isintact() and not(split)):
			if (return_without_alias):
				return "(SELECT * FROM {})".format(self.VERTICA_ML_PYTHON_VARIABLES["main_relation"])
			else:
				return self.VERTICA_ML_PYTHON_VARIABLES["main_relation"]
		# FINDING MAX FLOOR
		all_imputations_grammar = []
		force_columns = self.VERTICA_ML_PYTHON_VARIABLES["columns"] if not(force_columns) else force_columns
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
		        imputations += ["{}"] * diff
		# FILTER
		where_positions = [item[1] for item in self.VERTICA_ML_PYTHON_VARIABLES["where"]]
		max_where_pos = max(where_positions + [0])
		all_where = [[] for item in range(max_where_pos + 1)]
		for i in range(0, len(self.VERTICA_ML_PYTHON_VARIABLES["where"])):
			all_where[where_positions[i]] += [self.VERTICA_ML_PYTHON_VARIABLES["where"][i][0]]
		all_where = [" AND ".join(item) for item in all_where]
		for i in range(len(all_where)):
			if (all_where[i] != ''):
				all_where[i] = " WHERE {}".format(all_where[i])
		# FIRST FLOOR
		columns = force_columns + [column for column in transformations]
		first_values = [item[0] for item in all_imputations_grammar]
		for i in range(0, len(first_values)):
		    first_values[i] = "{} AS {}".format(first_values[i], columns[i]) 
		table = "SELECT {} FROM {}".format(", ".join(first_values), self.VERTICA_ML_PYTHON_VARIABLES["main_relation"])
		# OTHER FLOORS
		for i in range(1, max_len):
		    values = [item[i] for item in all_imputations_grammar]
		    for j in range(0, len(values)):
		        values[j] = "{} AS {}".format(values[j].replace("{}", columns[j]), columns[j])
		    table = "SELECT {} FROM ({}) t{}".format(", ".join(values), table, i)
		    try:
		    	table += all_where[i - 1]
		    except:
		    	pass
		    try:
		    	table += self.VERTICA_ML_PYTHON_VARIABLES["order_by"][i - 1]
		    except:
		    	pass
		try:
			where_final = all_where[max_len - 1]
		except:
			where_final = ""
		try:
			order_final = self.VERTICA_ML_PYTHON_VARIABLES["order_by"][max_len - 1]
		except:
			order_final = ""
		split = ", RANDOM() AS __vertica_ml_python_split__" if (split) else ""
		if (where_final == "") and (order_final == ""):
			if (split):
				if (return_without_alias):
					return "SELECT *{} FROM ({}) {}".format(split, table, final_table_name)
				table = "(SELECT *{} FROM ({}) {}) split_final_table".format(split, table, final_table_name)
			else:
				if (return_without_alias):
					return table
				table = "({}) {}".format(table, final_table_name)
		else:
			table = "({}) t{}{}{}".format(table, max_len, where_final, order_final)
			if (return_without_alias):
				return table
			table = "(SELECT *{} FROM {}) {}".format(split, table, final_table_name)
		if (self.VERTICA_ML_PYTHON_VARIABLES["exclude_columns"]):
			table = "(SELECT {}{} FROM {}) {}".format(", ".join(self.get_columns()), split, table, final_table_name)
		return table
	# 
	def abs(self, columns: list = []):
		check_types([("columns", columns, [list], False)])
		columns_check(columns, self)
		columns = self.numcol() if not(columns) else vdf_columns_names(columns, self)
		func = {}
		for column in columns:
			if (self[column].ctype() != "boolean"):
				func[column] = "ABS({})"
		return (self.apply(func))
	#
	def add_to_history(self, message: str):
		check_types([("message", message, [str], False)])
		self.VERTICA_ML_PYTHON_VARIABLES["history"] += ["{}{}{} {}".format("{", time.strftime("%c"), "}", message)]
		return (self)
	#
	def agg(self, func: list, columns: list = []):
		return (self.aggregate(func = func, columns = columns))
	def aggregate(self, func: list, columns: list = []):
		check_types([("func", func, [list], False), ("columns", columns, [list], False)])
		columns_check(columns, self)
		columns = self.numcol() if not(columns) else vdf_columns_names(columns, self)
		agg = [[] for i in range(len(columns))]
		for idx, column in enumerate(columns):
			cast = "::int" if (self[column].ctype() == "boolean") else ""
			for fun in func:
				if (fun.lower() == "unique"):
					expr = "COUNT(DISTINCT {})".format(column)
				elif (fun.lower() == "approx_unique"):
					expr = "APPROXIMATE_COUNT_DISTINCT({})".format(column)
				elif (fun.lower() == "count"):
					expr = "COUNT({})".format(column)
				elif (fun.lower() == "median"):
					expr = "APPROXIMATE_MEDIAN({}{})".format(column, cast)
				elif (fun.lower() in ("std", "stddev")):
					expr = "STDDEV({}{})".format(column, cast)
				elif (fun.lower() in ("var", "variance")):
					expr = "VARIANCE({}{})".format(column, cast)
				elif (fun.lower() in ("mean", "avg")):
					expr = "AVG({}{})".format(column, cast)
				elif ('%' in fun):
					expr = "APPROXIMATE_PERCENTILE({}{} USING PARAMETERS percentile = {})".format(column, cast, float(fun[0:-1]) / 100)
				elif (fun.lower() == "sem"):
					expr = "STDDEV({}{}) / SQRT(COUNT({}))".format(column, cast, column)
				elif (fun.lower() == "mae"):
					mean = self[column].mean()
					expr = "SUM(ABS({}{} - {})) / COUNT({})".format(column, cast, mean, column)
				elif (fun.lower() == "mad"):
					median = self[column].median()
					expr = "APPROXIMATE_MEDIAN(ABS({}{} - {}))".format(column, cast, median)
				elif (fun.lower() in ("prod", "product")):
					expr = "DECODE(ABS(MOD(SUM(CASE WHEN {}{} < 0 THEN 1 ELSE 0 END), 2)), 0, 1, -1) * POWER(10, SUM(LOG(ABS({}{}))))".format(column, cast, column, cast)
				elif (fun.lower() in ("percent", "count_percent")):
					expr = "ROUND(COUNT({}) / {} * 100, 3)".format(column, self.shape()[0])
				else:
					expr = "{}({}{})".format(fun.upper(), column, cast)
				agg[idx] += [expr]
		values = {"index": func}
		try:
			self.executeSQL("SELECT {} FROM {}".format(', '.join([item for sublist in agg for item in sublist]), self.genSQL()), title = "COMPUTE AGGREGATION(S)")
			result = [item for item in self.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchone()]
			try:
				result = [float(item) for item in result]
			except:
				pass
			values = {"index": func}
			i = 0
			for column in columns:
				values[column] = result[i:i + len(func)]
				i += len(func)
		except:
			query = ["SELECT {} FROM vdf_table".format(', '.join(elem)) for elem in agg]
			query = " UNION ALL ".join(["({})".format(elem) for elem in query]) if (len(query) != 1) else query[0]
			query = "WITH vdf_table AS ({}) {}".format(self.genSQL(return_without_alias = True), query)
			self.executeSQL(query, title = "COMPUTE AGGREGATION(S) WITH UNION ALL")
			result = self.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchall()
			for idx, elem in enumerate(result):
				values[columns[idx]] = [item for item in elem]
		return (tablesample(values = values, table_info = False).transpose())
	# 
	def aggregate_matrix(self, 
						 method: str = "pearson",
			 	   		 columns: list = [], 
			 	   		 cmap: str = "",
			 	   		 round_nb: int = 3,
			 	   		 show: bool = True):
		columns = vdf_columns_names(columns, self)
		if (len(columns) == 1):
			if (method in ("pearson", "beta", "spearman", "kendall", "biserial", "cramer")):
				return 1.0
			elif (method == "cov"):
				return self[columns[0]].var()
		elif (len(columns) == 2):
			cast_0 = "::int" if (self[columns[0]].ctype() == "boolean") else ""
			cast_1 = "::int" if (self[columns[1]].ctype() == "boolean") else ""
			if (method in ("pearson", "spearman")):
				if (columns[1] == columns[0]):
					return 1
				table = self.genSQL() if (method == "pearson") else "(SELECT RANK() OVER (ORDER BY {}) AS {}, RANK() OVER (ORDER BY {}) AS {} FROM {}) rank_spearman_table".format(columns[0], columns[0], columns[1], columns[1], self.genSQL())
				query = "SELECT CORR({}{}, {}{}) FROM {}".format(columns[0], cast_0, columns[1], cast_1, table)
				title = "Compute the {} Correlation between the two variables".format(method)
			elif (method == "biserial"):
				if (columns[1] == columns[0]):
						return 1
				elif (self[columns[1]].category() != 'int') and (self[columns[0]].category() != 'int'):
					return float('nan')
				elif (self[columns[1]].category() == 'int'):
					if (self[columns[1]].ctype() != 'boolean'):
						agg = self[columns[1]].aggregate(['approx_unique', 'min', 'max']).values[columns[1]]
						if ((agg[0] != 2) or (agg[1] != 0) or (agg[2] != 1)):
							return float('nan')
					column_b, column_n = columns[1], columns[0]
					cast_b, cast_n = cast_1, cast_0
				elif (self[columns[0]].category() == 'int'):
					if (self[columns[0]].ctype() != 'boolean'):
						agg = self[columns[0]].aggregate(['approx_unique', 'min', 'max']).values[columns[0]]
						if ((agg[0] != 2) or (agg[1] != 0) or (agg[2] != 1)):
							return float('nan')
					column_b, column_n = columns[0], columns[1]
					cast_b, cast_n = cast_0, cast_1
				else:
					return float('nan')
				query = "SELECT (AVG(DECODE({}{}, 1, {}{}, NULL)) - AVG(DECODE({}{}, 0, {}{}, NULL))) / STDDEV({}{}) * SQRT(SUM({}{}) * SUM(1 - {}{}) / COUNT(*) / COUNT(*)) FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL;".format(
					column_b, cast_b, column_n, cast_n, column_b, cast_b, column_n, cast_n, column_n, cast_n, column_b, cast_b, column_b, cast_b, self.genSQL(), column_n, column_b)
				title = "Compute the biserial Correlation between the two variables"
			elif (method == "cramer"):
				if (columns[1] == columns[0]):
					return 1
				table_0_1 = "SELECT {}, {}, COUNT(*) AS nij FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL GROUP BY 1, 2".format(columns[0], columns[1], self.genSQL(), columns[0], columns[1])
				table_0 = "SELECT {}, COUNT(*) AS ni FROM {} WHERE {} IS NOT NULL GROUP BY 1".format(columns[0], self.genSQL(), columns[0])
				table_1 = "SELECT {}, COUNT(*) AS nj FROM {} WHERE {} IS NOT NULL GROUP BY 1".format(columns[1], self.genSQL(), columns[1])
				self.VERTICA_ML_PYTHON_VARIABLES["cursor"].execute("SELECT COUNT(*) AS n FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL".format(self.genSQL(), columns[0], columns[1]))
				n = self.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchone()[0]
				phi = "SELECT SUM((nij - ni * nj / {}) * (nij - ni * nj / {}) / (ni * nj)) AS phi2 FROM (SELECT * FROM ({}) table_0_1 LEFT JOIN ({}) table_0 ON table_0_1.{} = table_0.{}) x LEFT JOIN ({}) table_1 ON x.{} = table_1.{}".format(n, n, table_0_1, table_0, columns[0], columns[0], table_1, columns[1], columns[1])
				r_k_n_phi = "WITH cardinality AS (SELECT APPROXIMATE_COUNT_DISTINCT({}) AS k, APPROXIMATE_COUNT_DISTINCT({}) AS r FROM {}), phi2_table AS ({}) ".format(columns[0], columns[1], self.genSQL(), phi)
				r_k_n_phi += "SELECT SQRT((CASE WHEN phi2_adjusted < 0 THEN 0 ELSE phi2_adjusted END) / NULLIFZERO(CASE WHEN k_adjusted < r_adjusted THEN k_adjusted ELSE r_adjusted END)) AS kendall FROM (SELECT phi2, k, r, phi2 - (r - 1) * (k - 1) / ({} - 1) AS phi2_adjusted, k - (k - 1) * (k - 1) / ({} - 1) AS k_adjusted, r - (r - 1) * (r - 1) / ({} - 1) AS r_adjusted FROM cardinality, phi2_table) x".format(n, n, n)
				self.executeSQL(r_k_n_phi, title = "Compute the CramerV Correlation between the two variables ({} and {})".format(columns[0], columns[1]))
				result = self.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchone()[0]
				try:
					result = float(result)
				except:
					result = float('nan')
				result = float('nan') if (result > 1 or result < 0) else result
				return result
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
				return self.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchone()[0]
			except:
				return float('nan')
		elif (len(columns) >= 2):
			try:
				if (method in ("pearson", "spearman")):
					table = self.genSQL() if (method == "pearson") else "(SELECT {} FROM {}) spearman_table".format(", ".join(["RANK() OVER (ORDER BY {}) AS {}".format(column, column) for column in columns]), self.genSQL())
					self.executeSQL(query = "SELECT CORR_MATRIX({}) OVER () FROM {}".format(", ".join(columns), table), title = "Computing the Corr Matrix")
					result = self.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchall()
					corr_dict = {}
					for idx, column in enumerate(columns):
						corr_dict[column] = idx
					n = len(columns)
					matrix = [[1 for i in range(0, n + 1)] for i in range(0, n + 1)]
					for elem in result:
						i, j = corr_dict[str_column(elem[0])], corr_dict[str_column(elem[1])]
						matrix[i + 1][j + 1] = elem[2]
					matrix[0] = [''] + columns
					for idx, column in enumerate(columns):
						matrix[idx + 1][0] = column
					title = 'Correlation Matrix ({})'.format(method)
				else:
					raise
			except:
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
							cast_i = "::int" if (self[columns[i]].ctype() == "boolean") else ""
							cast_j = "::int" if (self[columns[j]].ctype() == "boolean") else ""
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
					result = self.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchone()
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
							current = float('nan')
						matrix[i + 1][j + 1] = current
						matrix[j + 1][i + 1] = 1 / current if ((method == "beta") and (current != 0)) else current
			if ((show) and (method in ("pearson", "spearman", "kendall", "biserial", "cramer"))):
				from vertica_ml_python.plot import cmatrix
				vmin = 0 if (method == "cramer") else -1
				if not(cmap):
					from vertica_ml_python.plot import gen_cmap
					cm1, cm2 = gen_cmap()
					cmap = cm1 if (method == "cramer") else cm2
				cmatrix(matrix, columns, columns, n, n, vmax = 1, vmin = vmin, cmap = cmap, title = title, mround = round_nb)
			values = {"index" : matrix[0][1:len(matrix[0])]}
			del(matrix[0])
			for column in matrix:
				values[column[0]] = column[1:len(column)]
			return tablesample(values = values, table_info = False)
		else:
			if (method == "cramer"):
				cols = self.catcol()
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
				cols = self.catcol()
				if (len(cols) == 0):
			 		raise Exception("No categorical column found")
			else:
				cols = self.numcol()
				if (len(cols) == 0):
			 		raise Exception("No numerical column found")
		else:
			cols = vdf_columns_names(columns, self)
		if (method in ('spearman', 'pearson', 'kendall') and (len(cols) > 1)):
			try:
				fail = 0
				cast_i = "::int" if (self[focus].ctype() == "boolean") else ""
				all_list, all_cols  = [], [focus]
				for column in cols:
					if (column.replace('"', '').lower() != focus.replace('"', '').lower()):
						all_cols += [column]
					cast_j = "::int" if (self[column].ctype() == "boolean") else ""
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
				result = self.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchone()
				vector = [elem for elem in result]
			except:
				fail = 1
		if not(method in ('spearman', 'pearson', 'kendall') and (len(cols) > 1)) or (fail): 
			vector = []
			for column in cols:
				if (column.replace('"', '').lower() == focus.replace('"', '').lower()):
					vector += [1]
				else:
					vector += [self.aggregate_matrix(method = method, columns = [column, focus])]
		vector = [0 if (elem == None) else elem for elem in vector]
		data = [(cols[i], vector[i]) for i in range(len(vector))]
		data.sort(key = lambda tup: abs(tup[1]), reverse = True)
		cols, vector = [elem[0] for elem in data], [elem[1] for elem in data]
		if ((show) and (method in ("pearson", "spearman", "kendall", "biserial", "cramer"))):
			from vertica_ml_python.plot import cmatrix
			vmin = 0 if (method == "cramer") else -1
			if not(cmap):
				from vertica_ml_python.plot import gen_cmap
				cm1, cm2 = gen_cmap()
				cmap = cm1 if (method == "cramer") else cm2
			title = "Correlation Vector of {} ({})".format(focus, method)
			cmatrix([cols, [focus] + vector], cols, [focus], len(cols), 1, vmax = 1, vmin = vmin, cmap = cmap, title = title, mround = round_nb)
		return tablesample(values = {"index" : cols, focus : vector}, table_info = False)
	#
	def append(self, 
			   vdf = None, 
			   input_relation: str = ""):
		check_types([("vdf", vdf, [type(None), type(self)], False), ("input_relation", input_relation, [str], False)])
		first_relation = self.genSQL()
		second_relation = input_relation if not(vdf) else vdf.genSQL()
		columns = ", ".join(self.get_columns())
		table = "(SELECT {} FROM {}) UNION ALL (SELECT {} FROM {})".format(columns, first_relation, columns, second_relation)
		query = "SELECT * FROM ({}) append_table LIMIT 1".format(table)
		self.executeSQL(query = query, title = "Merging the two relation")
		self.VERTICA_ML_PYTHON_VARIABLES["main_relation"] = "({}) append_table".format(table)
		return (self)
	#
	def all(self, columns: list):
		return (self.aggregate(func = ["bool_and"], columns = columns))
	#
	def any(self, columns: list):
		return (self.aggregate(func = ["bool_or"], columns = columns))
	# 
	def apply(self, func: dict):
		check_types([("func", func, [dict], False)])
		columns_check([elem for elem in func], self)
		for column in func:
			self[vdf_columns_names([column], self)[0]].apply(func[column])
		return (self)
	# 
	def applymap(self, func: str, numeric_only: bool = True):
		check_types([("func", func, [str], False), ("numeric_only", numeric_only, [bool], False)])
		function = {}
		columns = self.numcol() if numeric_only else self.get_columns()
		for column in columns:
			function[column] = func if (self[column].ctype() != "boolean") else func.replace("{}", "{}::int")
		return (self.apply(function))
	#
	def asfreq(self,
			   ts: str,
			   rule: str,
			   method: dict,
			   by: list = []):
		check_types([("ts", ts, [str], False), ("rule", rule, [str], False), ("method", method, [dict], False), ("by", by, [list], False)])
		columns_check(by + [elem for elem in method], self)
		ts, by = vdf_columns_names([ts], self)[0], vdf_columns_names(by, self)
		all_elements = []
		for column in method:
			if (method[column] not in ('bfill', 'backfill', 'pad', 'ffill', 'linear')):
				raise ValueError("Each element of the 'method' dictionary must be in bfill|backfill|pad|ffill|linear")
			if (method[column] in ('bfill', 'backfill')):
				func, interp = "TS_FIRST_VALUE", 'const'
			elif (method[column] in ('pad', 'ffill')):
				func, interp = "TS_LAST_VALUE", 'const'
			else:
				func, interp = "TS_FIRST_VALUE", 'linear'
			all_elements += ["{}({}, '{}') AS {}".format(func, vdf_columns_names([column], self)[0], interp, vdf_columns_names([column], self)[0])]
		table = "SELECT {} FROM {}".format("{}", self.genSQL())
		tmp_query = ["slice_time AS {}".format(str_column(ts))]
		tmp_query += [str_column(column) for column in by]
		tmp_query += all_elements
		table = table.format(", ".join(tmp_query))
		partition = "PARTITION BY {} ".format(", ".join([str_column(column) for column in by])) if (by) else ""
		table += " TIMESERIES slice_time AS '{}' OVER ({}ORDER BY {})".format(rule, partition, str_column(ts))
		return (self.vdf_from_relation("(" + table + ') resample_table', "resample", "[Resample]: The data was resampled"))
	#
	def astype(self, dtype: dict):
		check_types([("dtype", dtype, [dict], False)])
		columns_check([elem for elem in dtype], self)
		for column in dtype:
			self[vdf_columns_names([column], self)[0]].astype(dtype = dtype[column])
		return (self)
	#
	def at_time(self,
				ts: str, 
				time: str):
		check_types([("ts", ts, [str], False), ("time", time, [str], False)])
		columns_check([ts], self)
		self.filter("{}::time = '{}'".format(str_column(ts), time))
		return (self)
	#
	def avg(self, columns: list = []):
		return (self.mean(columns = columns))
	# 
	def bar(self,
			columns: list,
			method: str = "density",
			of: str = "",
			max_cardinality: tuple = (6, 6),
			h: tuple = (None, None),
			hist_type: str = "auto"):
		check_types([("columns", columns, [list], False), ("method", method, ["density", "count", "avg", "min", "max", "sum"], True), ("of", of, [str], False), ("max_cardinality", max_cardinality, [tuple], False), ("h", h, [tuple], False), ("hist_type", hist_type, ["auto", "fully_stacked", "stacked", "fully", "fully stacked"], True)])
		columns_check(columns, self, [1, 2])
		columns = vdf_columns_names(columns, self)
		if (of):
			columns_check([of], self)
			of = vdf_columns_names([of], self)[0]
		if (len(columns) == 1):
			self[columns[0]].bar(method, of, 6, 0, 0)
		else:
			stacked, fully_stacked = False, False
			if (hist_type.lower() in ("fully", "fully stacked", "fully_stacked")):
				fully_stacked = True
			elif (hist_type.lower() == "stacked"):
				stacked = True
			from vertica_ml_python.plot import bar2D
			bar2D(self, columns, method, of, max_cardinality, h, stacked, fully_stacked)
		return (self)
	# 
	def beta(self, columns: list = [], focus: str = ""):
		check_types([("columns", columns, [list], False), ("focus", focus, [str], False)])
		columns_check(columns, self)
		columns = vdf_columns_names(columns, self)
		if (focus == ""):
			return (self.aggregate_matrix(method = "beta", columns = columns))
		else:
			columns_check([focus], self)
			focus = vdf_columns_names([focus], self)[0]
			return (self.aggregate_vector(focus, method = "beta", columns = columns))
	#
	def between_time(self,
					 ts: str, 
					 start_time: str, 
					 end_time: str):
		check_types([("ts", ts, [str], False), ("start_time", start_time, [str], False), ("end_time", end_time, [str], False)])
		columns_check([ts], self)
		self.filter("{}::time BETWEEN '{}' AND '{}'".format(str_column(ts), start_time, end_time))
		return (self)
	#
	def bool_to_int(self):
		columns = self.get_columns()
		for column in columns:
			if (self[column].ctype() == "boolean"):
				self[column].astype("int")
		return (self)
	#
	def boxplot(self, columns: list = []):
		check_types([("columns", columns, [list], False)])
		columns_check(columns, self)
		columns = vdf_columns_names(columns, self) if (columns) else self.numcol()
		from vertica_ml_python.plot import boxplot2D
		boxplot2D(self, columns)
		return (self)
	#
	def catcol(self, max_cardinality: int = 12):
		check_types([("max_cardinality", max_cardinality, [int, float], False)])
		columns = []
		for column in self.get_columns():
			if ((self[column].category() == "int") and (self[column].ctype() != "boolean")):
				self.VERTICA_ML_PYTHON_VARIABLES["cursor"].execute("SELECT (APPROXIMATE_COUNT_DISTINCT({}) < {}) FROM {}".format(column, max_cardinality, self.genSQL()))
				is_cat = self.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchall()[0]
			elif (self[column].category() == "float"):
				is_cat = False
			else:
				is_cat = True
			if (is_cat):
				columns += [column]
		return (columns)
	#
	def copy(self):
		copy_vDataframe = vDataframe("", empty = True)
		copy_vDataframe.VERTICA_ML_PYTHON_VARIABLES["dsn"] = self.VERTICA_ML_PYTHON_VARIABLES["dsn"]
		copy_vDataframe.VERTICA_ML_PYTHON_VARIABLES["input_relation"] = self.VERTICA_ML_PYTHON_VARIABLES["input_relation"]
		copy_vDataframe.VERTICA_ML_PYTHON_VARIABLES["main_relation"] = self.VERTICA_ML_PYTHON_VARIABLES["main_relation"]
		copy_vDataframe.VERTICA_ML_PYTHON_VARIABLES["schema"] = self.VERTICA_ML_PYTHON_VARIABLES["schema"]
		copy_vDataframe.VERTICA_ML_PYTHON_VARIABLES["cursor"] = self.VERTICA_ML_PYTHON_VARIABLES["cursor"]
		copy_vDataframe.VERTICA_ML_PYTHON_VARIABLES["columns"] = [item for item in self.VERTICA_ML_PYTHON_VARIABLES["columns"]]
		copy_vDataframe.VERTICA_ML_PYTHON_VARIABLES["where"] = [item for item in self.VERTICA_ML_PYTHON_VARIABLES["where"]]
		copy_vDataframe.VERTICA_ML_PYTHON_VARIABLES["order_by"] = [item for item in self.VERTICA_ML_PYTHON_VARIABLES["order_by"]]
		copy_vDataframe.VERTICA_ML_PYTHON_VARIABLES["exclude_columns"] = [item for item in self.VERTICA_ML_PYTHON_VARIABLES["exclude_columns"]]
		copy_vDataframe.VERTICA_ML_PYTHON_VARIABLES["history"] = [item for item in self.VERTICA_ML_PYTHON_VARIABLES["history"]]
		copy_vDataframe.VERTICA_ML_PYTHON_VARIABLES["saving"] = [item for item in self.VERTICA_ML_PYTHON_VARIABLES["saving"]]
		copy_vDataframe.VERTICA_ML_PYTHON_VARIABLES["query_on"] = self.VERTICA_ML_PYTHON_VARIABLES["query_on"]
		copy_vDataframe.VERTICA_ML_PYTHON_VARIABLES["time_on"] = self.VERTICA_ML_PYTHON_VARIABLES["time_on"]
		for column in self.VERTICA_ML_PYTHON_VARIABLES["columns"]:
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
		check_types([("columns", columns, [list], False), ("method", method, ["pearson", "kendall", "spearman", "biserial", "cramer"], True), ("cmap", cmap, [str], False), ("round_nb", round_nb, [int, float], False), ("focus", focus, [str], False), ("show", show, [bool], False)])
		columns_check(columns, self)
		columns = vdf_columns_names(columns, self)
		if (focus == ""):
			return (self.aggregate_matrix(method = method, columns = columns, cmap = cmap, round_nb = round_nb, show = show))
		else:
			columns_check([focus], self)
			focus = vdf_columns_names([focus], self)[0]
			return (self.aggregate_vector(focus, method = method, columns = columns, cmap = cmap, round_nb = round_nb, show = show))
	# 
	def cov(self, columns: list = [], focus: str = ""):
		check_types([("columns", columns, [list], False), ("focus", focus, [str], False)])
		columns_check(columns, self)
		columns = vdf_columns_names(columns, self)
		if (focus == ""):
			return (self.aggregate_matrix(method = "cov", columns = columns))
		else:
			columns_check([focus], self)
			focus = vdf_columns_names([focus], self)[0]
			return (self.aggregate_vector(focus, method = "cov", columns = columns))
	# 
	def count(self, 
			  columns: list = [], 
			  percent: bool = True,
			  sort_result: bool = True,
			  desc: bool = True):
		check_types([("columns", columns, [list], False), ("percent", percent, [bool], False), ("desc", desc, [bool], False), ("sort_result", sort_result, [bool], False)])
		columns_check(columns, self)
		columns = vdf_columns_names(columns, self)
		columns = self.get_columns() if not(columns) else columns
		func = ["count", "percent"] if (percent) else ["count"]
		result = self.aggregate(func = func, columns = columns)
		if (sort_result):
			sort = []
			for i in range(len(result.values["index"])):
				if percent:
					sort += [(result.values["index"][i], result.values["count"][i], result.values["percent"][i])] 
				else:
					sort += [(result.values["index"][i], result.values["count"][i])] 
			sort.sort(key = lambda tup: tup[1], reverse = desc)
			result.values["index"] = [elem[0] for elem in sort]
			result.values["count"] = [elem[1] for elem in sort]
			if percent:
				result.values["percent"] = [elem[2] for elem in sort]
		return (result)
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
		cols = self.get_columns()
		for column in cols:
			if self[column].isdate():
				columns += [column]
		return (columns)
	# 
	def describe(self, 
				 method: str = "auto", 
				 columns: list = [], 
				 unique: bool = True):
		check_types([("method", method, ["auto", "numerical", "categorical"], True), ("columns", columns, [list], False), ("unique", unique, [bool], False)])
		columns_check(columns, self)
		columns = vdf_columns_names(columns, self)
		for i in range(len(columns)):
			columns[i] = str_column(columns[i])
		if (method == "auto"):
			if not(columns):
				method = "categorical" if not(self.numcol()) else "numerical"
			else:
				method = "numerical"
				for column in columns:
					if not(self[column].isnum()):
						method = "categorical"
						break
		if (method == "numerical"):
			if not(columns):
				columns = self.numcol()
			query = []
			for column in columns:
				if self[column].isnum():
					if (self[column].ctype() == "boolean"):
						query += [column + "::int"]
					else:
						query += [column]
				else:
					print("\u26A0 Warning: The Virtual Column {} is not numerical, it was ignored.\nTo get statistical information about all the different variables, please use the parameter method = 'categorical'.".format(column))
			if not(query):
				raise ValueError("There is no numerical Virtual Column in the vDataframe.")
			try:
				query = "SELECT SUMMARIZE_NUMCOL({}) OVER () FROM {}".format(", ".join(query), self.genSQL())
				self.executeSQL(query, title = "Compute the descriptive statistics of all the numerical columns using SUMMARIZE_NUMCOL")
			except:
				query = []
				for column in columns:
					if self[column].isnum():
						cast = "::int" if (self[column].ctype() == "boolean") else ""
						query += ["SELECT '{}' AS column, COUNT({}) AS count, AVG({}{}) AS mean, STDDEV({}{}) AS std, MIN({}{}) AS min, APPROXIMATE_PERCENTILE ({}{} USING PARAMETERS percentile = 0.25) AS '25%', APPROXIMATE_PERCENTILE ({}{} USING PARAMETERS percentile = 0.5) AS '50%', APPROXIMATE_PERCENTILE ({}{} USING PARAMETERS percentile = 0.75) AS '75%', MAX({}{}) AS max FROM vdf_table".format(
									column.replace('"', '').replace("'", "''"), column, column, cast, column, cast, column, cast, column, cast, column, cast, column, cast, column, cast)]
				query = query[0] if (len(query) == 1) else " UNION ALL ".join(["({})".format(elem) for elem in query])
				query = "WITH vdf_table AS ({}) {}".format(self.genSQL(return_without_alias = True), query)
				self.executeSQL(query, title = "Compute the descriptive statistics of all the numerical columns using standard SQL")
			query_result = self.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchall()
			data = [item for item in query_result]
			matrix = [['column'], ['count'], ['mean'], ['std'], ['min'], ['25%'], ['50%'], ['75%'], ['max']]
			for row in data:
				for idx,val in enumerate(row):
					matrix[idx] += [val]
			if (unique):
				query = []
				try:
					cardinality = self.aggregate(['approx_unique'], matrix[0][1:]).values['approx_unique']
				except:
					cardinality = []
					for column in matrix[0][1:]:
						cardinality += self.aggregate(['unique'], [column]).values['unique']
				matrix += [['unique'] + cardinality]
			values = {"index" : matrix[0][1:len(matrix[0])]}
			del(matrix[0])
			for column in matrix:
				values[column[0]] = column[1:len(column)]
		elif (method == "categorical"):
			if not(columns):
				columns = self.get_columns()
			try:
				values = {"index" : [column for column in columns], "dtype" : [self[column].ctype() for column in columns], "unique" : [], "count" : [], "top" : [], "top_percent" : []}
				information = self.aggregate(["count", "approx_unique"], columns)
				values["count"], values["unique"]  = information.values["count"], information.values["approx_unique"]
				try:
					cnt = self.shape()[0]
					query = []
					for column in columns:
						query += ["(SELECT {}::varchar, 100 * COUNT(*) / {} AS percent FROM vdf_table GROUP BY {} ORDER BY percent DESC LIMIT 1)".format(convert_special_type(self[column].category(), False, column), cnt, column)]
					query = "WITH vdf_table AS ({}) {}".format(self.genSQL(return_without_alias = True), " UNION ALL ".join(query))
					self.executeSQL(query, title = "Compute the MODE of all the selected features")
					result = self.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchall()
					values["top"], values["top_percent"] = [elem[0] for elem in result], [round(elem[1], 3) for elem in result]
				except:
					for column in columns:
						topk = self[column].topk(1, False).values
						values["top"] += topk["index"]
						values["top_percent"] += topk["percent"]
			except:
				values = {"index" : [], "dtype" : [], "unique" : [], "count" : [], "top" : [], "top_percent" : []}
				for column in columns:
					information = self.aggregate(["count", "unique"], [column])
					values["index"] += [column]
					values["dtype"] += [self[column].ctype()]
					values["unique"] += information.values["unique"]
					values["count"] += information.values["count"]
					result = self[column].topk(1, False).values
					if (len(result["index"]) == 0):
						values["top"] += [None]
						values["top_percent"] += [None]
					else:
						values["top"] += result["index"]
						values["top_percent"] += result["percent"]
		else:
			raise ValueError("The parameter 'method' must be in auto|numerical|categorical")
		return (tablesample(values, table_info = False))
	#
	def drop(self, columns: list = []):
		check_types([("columns", columns, [list], False)])
		columns_check(columns, self)
		columns = vdf_columns_names(columns, self)
		for column in columns:
			self[column].drop()
		return (self)
	#
	def drop_duplicates(self, columns: list = []):
		check_types([("columns", columns, [list], False)])
		columns_check(columns, self)
		count = self.duplicated(columns = columns, count = True)
		if (count):
			columns = self.get_columns() if not(columns) else vdf_columns_names(columns, self)
			name = "__vertica_ml_python_duplicated_index__" + str(random.randint(0, 10000000)) + "_"
			self.eval(name = name, expr = "ROW_NUMBER() OVER (PARTITION BY {})".format(", ".join(columns)))
			self.filter(expr = '"{}" = 1'.format(name))
			self.VERTICA_ML_PYTHON_VARIABLES["exclude_columns"] += ['"{}"'.format(name)]
		else:
			print("\u26A0 Warning: No duplicates detected")
		return (self)
	# 
	def dropna(self, columns: list = [], print_info: bool = True):
		check_types([("columns", columns, [list], False), ("print_info", print_info, [bool], False)])
		columns_check(columns, self)
		columns = self.get_columns() if not(columns) else vdf_columns_names(columns, self)
		total = self.shape()[0]
		for column in columns:
			self[column].dropna(print_info = False)
		if (print_info):
			total -= self.shape()[0]
			if (total == 0):
				print("\u26A0 Warning: Nothing was dropped")
			elif (total == 1):
				print("1 element was dropped")
			else:
				print("{} elements were dropped".format(total))
		return (self)
	# 
	def dsn_restart(self):
		from vertica_ml_python import vertica_cursor
		self.VERTICA_ML_PYTHON_VARIABLES["cursor"] = vertica_cursor(self.VERTICA_ML_PYTHON_VARIABLES["dsn"])
		return (self)
	# 
	def dtypes(self):
		values = {"index" : [], "dtype" : []}
		for column in self.get_columns():
			values["index"] += [column]
			values["dtype"] += [self[column].ctype()]
		return (tablesample(values, table_info = False))
	#
	def duplicated(self, columns: list = [], count: bool = False, limit: int = 30):
		check_types([("columns", columns, [list], False), ("count", count, [bool], False)])
		columns_check(columns, self)
		columns = self.get_columns() if not(columns) else vdf_columns_names(columns, self)
		query = "(SELECT *, ROW_NUMBER() OVER (PARTITION BY {}) AS duplicated_index FROM {}) duplicated_index_table WHERE duplicated_index > 1".format(", ".join(columns), self.genSQL())
		self.executeSQL(query = "SELECT COUNT(*) FROM {}".format(query), title = "Computing the Number of duplicates")
		total = self.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchone()[0]
		if (count):
			return total
		result = to_tablesample("SELECT {}, MAX(duplicated_index) AS occurrence FROM {} GROUP BY {} ORDER BY occurrence DESC LIMIT {}".format(", ".join(columns), query, ", ".join(columns), limit), self.VERTICA_ML_PYTHON_VARIABLES["cursor"], name = "Duplicated Rows (total = {})".format(total))
		self.executeSQL(query = "SELECT COUNT(*) FROM (SELECT {}, MAX(duplicated_index) AS occurrence FROM {} GROUP BY {}) t".format(", ".join(columns), query, ", ".join(columns)), title = "Computing the Number of different duplicates")
		result.count = self.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchone()[0]
		return (result)
	#
	def empty(self):
		return not(self.get_columns())
	# 
	def eval(self, name: str, expr: str):
		check_types([("name", name, [str], False), ("expr", expr, [str], False)])
		name = str_column(name.replace('"', '_'))
		if column_check_ambiguous(name, self.get_columns()):
			raise ValueError("A Virtual Column has already the alias {}.\nBy changing the parameter 'name', you'll be able to solve this issue.".format(name))
		tmp_name = "VERTICA_ML_PYTHON_" + str(random.randint(0, 10000000))
		self.executeSQL(query = "DROP TABLE IF EXISTS v_temp_schema.{}".format(tmp_name), title = "Drop the existing generated table")
		try:
			query = "CREATE LOCAL TEMPORARY TABLE {} ON COMMIT PRESERVE ROWS AS SELECT {} AS {} FROM {} LIMIT 20".format(tmp_name, expr, name, self.genSQL())
			self.executeSQL(query = query, title = "Create a temporary table to test if the new feature is correct")
		except:
			self.executeSQL(query = "DROP TABLE IF EXISTS v_temp_schema.{}".format(tmp_name), title = "Drop the temporary table")
			raise ValueError("The expression '{}' seems to be incorrect.\nBy turning on the SQL with the 'sql_on_off' method, you'll print the SQL code generation and probably see why the evaluation didn't work.".format(expr))
		query = "SELECT data_type FROM columns WHERE column_name = '{}' AND table_name = '{}' AND table_schema = 'v_temp_schema'".format(name.replace('"', '').replace("'", "''"), tmp_name)
		self.executeSQL(query = query, title = "Catch the new feature's type")
		ctype = self.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchone()[0]
		self.executeSQL(query = "DROP TABLE IF EXISTS v_temp_schema.{}".format(tmp_name), title = "Drop the temporary table")
		ctype = ctype if (ctype) else "undefined"
		category = category_from_type(ctype = ctype)
		vDataframe_maxfloor_length, vDataframe_minfloor_length = len(max([self[column].transformations for column in self.get_columns()], key = len)), 0
		for column in self.get_columns():
			if ((column in expr) or (column.replace('"', '') in expr)):
				vDataframe_minfloor_length = max(len(self[column].transformations), vDataframe_minfloor_length)
		for eval_floor_length in range(vDataframe_minfloor_length, vDataframe_maxfloor_length):
			try:
				self.VERTICA_ML_PYTHON_VARIABLES["cursor"].execute("SELECT * FROM {} LIMIT 0".format(
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
		self.VERTICA_ML_PYTHON_VARIABLES["columns"] += [name]
		self.add_to_history("[Eval]: A new Virtual Column {} was added to the vDataframe.".format(name)) 
		return (self)
	# 
	def executeSQL(self, query: str, title: str = ""):
		check_types([("query", query, [str], False), ("title", title, [str], False)])
		if (self.VERTICA_ML_PYTHON_VARIABLES["query_on"]):
			try:
				import shutil
				screen_columns = shutil.get_terminal_size().columns
			except:
				import os
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
		self.VERTICA_ML_PYTHON_VARIABLES["cursor"].execute(query)
		elapsed_time = time.time() - start_time
		if (self.VERTICA_ML_PYTHON_VARIABLES["time_on"]):
			try:
				import shutil
				screen_columns = shutil.get_terminal_size().VERTICA_ML_PYTHON_VARIABLES["columns"]
			except:
				import os
				screen_rows, screen_columns = os.popen('stty size', 'r').read().split()
			if (isnotebook()):
				from IPython.core.display import HTML,display
				display(HTML("<div><b>Elapsed Time : </b> " + str(elapsed_time) + "</div>"))
				display(HTML("<div style = 'border : 1px dashed black; width : 100%'></div>"))
			else:
				print("Elapsed Time: " + str(elapsed_time))
				print("-" * int(screen_columns) + "\n")
		return (self.VERTICA_ML_PYTHON_VARIABLES["cursor"])
	#
	def expected_store_usage(self, unit: str = 'b'):
		check_types([("unit", unit, [str], False)])
		if (unit.lower() == 'kb'):
			div_unit = 1024
		elif (unit.lower() == 'mb'):
			div_unit = 1024 * 1024
		elif (unit.lower() == 'gb'):
			div_unit = 1024 * 1024 * 1024
		elif (unit.lower() == 'tb'):
			div_unit = 1024 * 1024 * 1024 * 1024
		else:
			unit, div_unit = 'b', 1
		total, total_expected = 0, 0
		columns = self.get_columns()
		values = self.aggregate(func = ["count"], columns = columns).transpose().values
		values["index"] = ["expected_size ({})".format(unit), "max_size ({})".format(unit), "type"]
		for column in columns:
			ctype = self[column].ctype()
			if (ctype[0:4] == "date") or (ctype[0:4] == "time") or (ctype[0:8] == "interval") or (ctype == "smalldatetime"):
				maxsize, expsize = 8, 8
			elif ("int" in ctype):
				maxsize, expsize = 8, self[column].store_usage()
			elif (ctype[0:4] == "bool"):
				maxsize, expsize = 1, 1
			elif (ctype[0:5] == "float") or (ctype[0:6] == "double") or (ctype[0:4] == "real"):
				maxsize, expsize = 8, 8
			elif (ctype[0:7] in ("numeric", "decimal")) or (ctype[0:6] == "number") or (ctype[0:5] == "money"):
				size = sum([int(item) for item in ctype.split("(")[1].split(")")[0].split(",")])
				maxsize, expsize = size, size
			elif (ctype[0:7] == "varchar"):
				size = int(ctype.split("(")[1].split(")")[0])
				maxsize, expsize = size, self[column].store_usage()
			elif (ctype[0:4] == "char") or (ctype[0:3] == "geo") or ("binary" in ctype):
				try:
					size = int(ctype.split("(")[1].split(")")[0])
					maxsize, expsize = size, size
				except:
					if (ctype[0:3] == "geo"):
						maxsize, expsize = 10000, 10000000
					elif ("long" in ctype):
						maxsize, expsize = 10000, 32000000
					else:
						maxsize, expsize = 1000, 65000
			elif (ctype[0:4] == "uuid"):
				maxsize, expsize = 16, 16
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
		check_types([("val", val, [dict], False), ("method", method, [dict], False), ("numeric_only", numeric_only, [bool], False), ("print_info", print_info, [bool], False)])
		columns_check([elem for elem in val] + [elem for elem in method], self)
		if (not(val) and not(method)):
			cols = self.get_columns()
			for column in cols:
				if (numeric_only):
					if self[column].isnum():
						self[column].fillna(method = "auto", print_info = print_info)
				else:
					self[column].fillna(method = "auto", print_info = print_info)
		else:
			for column in val:
				self[vdf_columns_names([column], self)[0]].fillna(val = val[column], print_info = print_info)
			for column in method:
				self[vdf_columns_names([column], self)[0]].fillna(method = method[column], print_info = print_info)
		return (self)
	# 
	def filter(self, 
			   expr: str = "", 
			   conditions: list = [],
			   print_info: bool = True):
		check_types([("expr", expr, [str], False), ("conditions", conditions, [list], False), ("print_info", print_info, [bool], False)])
		count = self.shape()[0]
		if not(expr):
			for condition in conditions:
				self.filter(expr = condition, print_info = False)
			count -= self.shape()[0]
			if (count > 1):
				if (print_info):
					print("{} elements were filtered".format(count))
				self.add_to_history("[Filter]: {} elements were filtered using the filter '{}'".format(count, conditions))
			elif (count == 1):
				if (print_info):
					print("{} element was filtered".format(count))
				self.add_to_history("[Filter]: {} element was filtered using the filter '{}'".format(count, conditions))
			else:
				if (print_info):
					print("Nothing was filtered.")
		else:
			max_pos = 0
			for column in self.VERTICA_ML_PYTHON_VARIABLES["columns"]:
				max_pos = max(max_pos, len(self[column].transformations) - 1)
			self.VERTICA_ML_PYTHON_VARIABLES["where"] += [(expr, max_pos)]
			try:
				count -= self.shape()[0]
			except:
				del self.VERTICA_ML_PYTHON_VARIABLES["where"][-1]
				if (print_info):
					print("\u26A0 Warning: The expression '{}' is incorrect.\nNothing was filtered.".format(expr))
			if (count > 1):
				if (print_info):
					print("{} elements were filtered".format(count))
				self.add_to_history("[Filter]: {} elements were filtered using the filter '{}'".format(count, expr))
			elif (count == 1):
				if (print_info):
					print("{} element was filtered".format(count))
				self.add_to_history("[Filter]: {} element was filtered using the filter '{}'".format(count, expr))
			else:
				del self.VERTICA_ML_PYTHON_VARIABLES["where"][-1]
				if (print_info):
					print("Nothing was filtered.")
		return (self)
	#
	def first(self, ts: str, offset: str):
		check_types([("ts", ts, [str], False), ("offset", offset, [str], False)])
		ts = vdf_columns_names([ts], self)[0]
		query = "SELECT (MIN({}) + '{}'::interval)::varchar FROM {}".format(ts, offset, self.genSQL())
		self.VERTICA_ML_PYTHON_VARIABLES["cursor"].execute(query)
		first_date = self.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchone()[0]
		self.filter("{} <= '{}'".format(ts, first_date))
		return (self)
	#
	def get_columns(self, exclude_columns: list = []):
		columns = [column for column in self.VERTICA_ML_PYTHON_VARIABLES["columns"]]
		for column in columns:
			if (column in self.VERTICA_ML_PYTHON_VARIABLES["exclude_columns"] + exclude_columns):
				columns.remove(column)
		return(columns)
	# 
	def get_dummies(self, 
					columns: list = [],
					max_cardinality: int = 12, 
					prefix_sep: str = "_", 
					drop_first: bool = True, 
					use_numbers_as_suffix: bool = False):
		check_types([("columns", columns, [list], False), ("max_cardinality", max_cardinality, [int, float], False), ("prefix_sep", prefix_sep, [str], False), ("drop_first", drop_first, [bool], False), ("use_numbers_as_suffix", use_numbers_as_suffix, [bool], False)])
		columns_check(columns, self)
		cols_hand = True if (columns) else False
		columns = self.get_columns() if not(columns) else vdf_columns_names(columns, self)
		for column in columns:
			if (self[column].nunique(True) < max_cardinality):
				self[column].get_dummies("", prefix_sep, drop_first, use_numbers_as_suffix)
			elif (cols_hand):
				print("\u26A0 Warning: The Virtual Column {} was ignored because of its high cardinality\nIncrease the parameter 'max_cardinality' to solve this issue or use directly the Virtual Column get_dummies method".format(column))
		return (self)
	# 
	def groupby(self, columns: list, expr: list = []):
		check_types([("columns", columns, [list], False), ("expr", expr, [list], False)])
		columns_check(columns, self)
		columns = vdf_columns_names(columns, self)
		relation = "(SELECT {} FROM {} GROUP BY {}) groupby_table".format(", ".join(columns + expr), self.genSQL(), ", ".join(columns)) 
		return (self.vdf_from_relation(relation, "groupby", "[Groupby]: The columns were group by {}".format(", ".join(columns))))
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
		check_types([("columns", columns, [list], False), ("method", method, ["density", "count", "avg", "min", "max", "sum"], True), ("of", of, [str], False), ("cmap", cmap, [str], False), ("gridsize", gridsize, [int, float], False), ("color", color, [str], False)])
		columns_check(columns, self, [2])
		columns = vdf_columns_names(columns, self)
		if (of):
			columns_check([of], self)
			of = vdf_columns_names([of], self)[0]
		if not(cmap):
			from vertica_ml_python.plot import gen_cmap
			cmap = gen_cmap()[0]
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
			 hist_type: str = "auto"):
		check_types([("columns", columns, [list], False), ("method", method, ["density", "count", "avg", "min", "max", "sum"], True), ("of", of, [str], False), ("max_cardinality", max_cardinality, [tuple], False), ("h", h, [tuple], False), ("hist_type", hist_type, ["auto", "multi", "stacked"], True)])
		columns_check(columns, self, [2])
		columns = vdf_columns_names(columns, self)
		if (of):
			columns_check([of], self)
			of = vdf_columns_names([of], self)[0]
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
				hist2D(self, columns, method, of, max_cardinality, h, stacked)
		return (self)
	# 
	def info(self):
		if (len(self.VERTICA_ML_PYTHON_VARIABLES["history"]) == 0):
			print("The vDataframe was never modified.")
		elif (len(self.VERTICA_ML_PYTHON_VARIABLES["history"]) == 1):
			print("The vDataframe was modified with only one action: ")
			print(" * " + self.VERTICA_ML_PYTHON_VARIABLES["history"][0])
		else:
			print("The vDataframe was modified many times: ")
			for modif in self.VERTICA_ML_PYTHON_VARIABLES["history"]:
				print(" * " + modif)
		return (self)
	#
	def isin(self, val: dict):
		check_types([("val", val, [dict], False)])
		columns_check([elem for elem in val], self)
		n = len(val[list(val.keys())[0]])
		isin = []
		for i in range(n):
			tmp_query = []
			for column in val:
				if (val[column][i] == None):
					tmp_query += [str_column(column) + " IS NULL"]
				else:
					tmp_query += [str_column(column) + " = '{}'".format(str(val[column][i]).replace("'", "''"))]
			query = "SELECT * FROM {} WHERE ".format(self.genSQL()) + " AND ".join(tmp_query) + " LIMIT 1"
			self.VERTICA_ML_PYTHON_VARIABLES["cursor"].execute(query)
			isin += [self.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchone() != None] 
		return (isin)
	#
	def isintact(self):
		if ((self.VERTICA_ML_PYTHON_VARIABLES["*"] < 1) or self.VERTICA_ML_PYTHON_VARIABLES["where"] != [] or self.VERTICA_ML_PYTHON_VARIABLES["order_by"] != ['' for i in range(100)]):
			return False
		columns = self.get_columns()
		if (len(columns) != self.VERTICA_ML_PYTHON_VARIABLES["*"]):
			return False
		else:
			for column in columns:
				transformations = self[column].transformations
				if ((len(transformations) != 1) or (transformations[0][0] != '{}'.format(column))):
					return False
		return True
	#
	def join(self, 
			 input_relation: str = "", 
			 vdf = None, 
			 on: dict = {},
			 how: str = 'natural',
			 expr1: list = [],
			 expr2: list = []):
		check_types([("input_relation", input_relation, [str], False), ("vdf", vdf, [type(None), type(self)], False), ("on", on, [dict], False), ("how", how.lower(), ["left", "right", "cross", "full", "natural", "self", "inner", ""], True), ("expr1", expr1, [list], False), ("expr2", expr2, [list], False)])
		columns_check([elem for elem in on], self)
		if (vdf != None):
			columns_check([on[elem] for elem in on], vdf)
		vdf_cols = []
		if (vdf):
			for elem in on:
				vdf_cols += [on[elem]]
			columns_check(vdf_cols, vdf)
		on_join = " AND ".join(["x." + elem + " = y." + on[elem] for elem in on])
		on_join = " ON {}".format(on_join) if (on_join) else ""
		first_relation = self.genSQL(final_table_name = "x")
		second_relation = input_relation + " AS y" if not(vdf) else vdf.genSQL(final_table_name = "y")
		expr1, expr2 = ["x.{}".format(elem) for elem in expr1], ["y.{}".format(elem) for elem in expr2]
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
		check_types([("ts", ts, [str], False), ("offset", offset, [str], False)])
		ts = vdf_columns_names([ts], self)[0]
		query = "SELECT (MAX({}) - '{}'::interval)::varchar FROM {}".format(ts, offset, self.genSQL())
		self.VERTICA_ML_PYTHON_VARIABLES["cursor"].execute(query)
		last_date = self.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchone()[0]
		self.filter("{} >= '{}'".format(ts, last_date))
		return (self)
	#
	def load(self, offset: int = -1):
		check_types([("offset", offset, [int, float], False)])
		save =  self.VERTICA_ML_PYTHON_VARIABLES["saving"][offset]
		vdf = {}
		exec(save, globals(), vdf)
		vdf = vdf["vdf_save"]
		vdf.VERTICA_ML_PYTHON_VARIABLES["cursor"] = self.VERTICA_ML_PYTHON_VARIABLES["cursor"]
		return (vdf)
	#
	def mad(self, columns: list = []):
		return (self.aggregate(func = ["mad"], columns = columns))
	#
	def mae(self, columns: list = []):
		return (self.aggregate(func = ["mae"], columns = columns))
	#
	def max(self, columns: list = []):
		return (self.aggregate(func = ["max"], columns = columns))
	#
	def mean(self, columns: list = []):
		return (self.aggregate(func = ["avg"], columns = columns))
	#
	def median(self, columns: list = []):
		return (self.aggregate(func = ["median"], columns = columns))
	#
	def memory_usage(self):
		import sys
		total =  sum([sys.getsizeof(elem) for elem in self.VERTICA_ML_PYTHON_VARIABLES]) + sys.getsizeof(self) 
		values = {"index": ["object"], "value": [total]}
		for column in self.VERTICA_ML_PYTHON_VARIABLES["columns"]:
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
	def normalize(self, columns: list = [], method = "zscore"):
		check_types([("columns", columns, [list], False), ("method", method, ["zscore", "robust_zscore", "minmax"], True), ("by", by, [list], False)])
		columns_check(columns, self)
		no_cols = True if not(columns) else False
		columns = self.numcol() if not(columns) else vdf_columns_names(columns, self)
		for column in columns:
			if (self[column].isnum() and (self[column].ctype() != "boolean")):
				self[column].normalize(method = method)
			elif ((no_cols) and (self[column].ctype() == "boolean")):
				pass
			else:
				print("\u26A0 Warning: The Virtual Column {} was skipped\nNormalize only accept numerical data types".format(column))
		return (self)
	#
	def numcol(self):
		columns, cols = [], self.get_columns()
		for column in cols:
			if self[column].isnum():
				columns += [column]
		return (columns)
	# 
	def outliers(self,
				 columns: list = [],
				 name: str = "distribution_outliers",
				 threshold: float = 3.0,
				 robust: bool = False):
		check_types([("columns", columns, [list], False), ("name", name, [str], False), ("threshold", threshold, [int, float], False)])
		columns_check(columns, self)
		columns = vdf_columns_names(columns, self) if (columns) else self.numcol()
		if not(robust):
			result = self.aggregate(func = ["std", "avg"], columns = columns).values
		else:
			result = self.aggregate(func = ["mad", "median"], columns = columns).values
		conditions = []
		for idx, elem in enumerate(result["index"]):
			if not(robust):
				conditions += ["ABS({} - {}) / NULLIFZERO({}) > {}".format(elem, result["avg"][idx], result["std"][idx], threshold)]
			else:
				conditions += ["ABS({} - {}) / NULLIFZERO({} * 1.4826) > {}".format(elem, result["median"][idx], result["mad"][idx], threshold)]
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
					with_numbers: bool = True):
		check_types([("columns", columns, [list], False), ("method", method, ["density", "count", "avg", "min", "max", "sum"], True), ("of", of, [str], False), ("max_cardinality", max_cardinality, [tuple], False), ("h", h, [tuple], False), ("cmap", cmap, [str], False), ("show", show, [bool], False), ("with_numbers", with_numbers, [bool], False)])
		columns_check(columns, self, [1, 2])
		columns = vdf_columns_names(columns, self)
		if (of):
			columns_check([of], self)
			of = vdf_columns_names([of], self)[0]
		if not(cmap):
			from vertica_ml_python.plot import gen_cmap
			cmap = gen_cmap()[0]
		from vertica_ml_python.plot import pivot_table
		return (pivot_table(self, columns, method, of, h, max_cardinality, show, cmap, with_numbers))
	#
	def plot(self, 
		     ts: str,
			 columns: list = [], 
			 start_date: str = "",
		     end_date: str = ""):
		check_types([("columns", columns, [list], False), ("ts", ts, [str], False), ("start_date", start_date, [str], False), ("end_date", end_date, [str], False)])
		columns_check(columns + [ts], self)
		columns = vdf_columns_names(columns, self)
		ts = vdf_columns_names([ts], self)[0]
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
	def rank(self, order_by: list, method: str = "first", by: list = [], name: str = ""):
		check_types([("order_by", order_by, [list], False), ("method", method, ["first", "dense", "percent"], True), ("by", by, [list], False), ("name", name, [str], False)])
		columns_check(order_by + by, self)
		if (method == "first"):
			func = "RANK"
		elif (method == "dense"):
			func = "DENSE_RANK"
		elif (method == "percent"):
			func = "PERCENT_RANK"
		partition = "" if not(by) else "PARTITION BY " + ", ".join(vdf_columns_names(by, self))
		name_prefix = "_".join([column.replace('"', '') for column in vdf_columns_names(order_by, self)]) 
		name_prefix += "_by_" + "_".join([column.replace('"', '') for column in vdf_columns_names(by, self)]) if (by) else ""
		name = name if (name) else method + "_rank_" + name_prefix
		order = "ORDER BY {}".format(", ".join(vdf_columns_names(order_by, self)))
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
		check_types([("name", name, [str], False), ("aggr", aggr, [str], False), ("column", column, [str], False), ("preceding", preceding, [str, int], False), ("following", following, [str, int], False), ("expr", expr, [str], False), ("by", by, [list], False), ("order_by", order_by, [list], False), ("method", method, ["rows", "range"], True), ("rule", rule, ["auto", "past", "future"], True)])
		columns_check([column] + by + order_by, self)
		if (rule.lower() == "past"):
			rule_p, rule_f = "PRECEDING", "PRECEDING"
		elif (rule.lower() == "future"):
			rule_p, rule_f = "FOLLOWING", "FOLLOWING"
		else:
			rule_p, rule_f = "PRECEDING", "FOLLOWING"
		column = vdf_columns_names([column], self)[0]
		by = "" if not(by) else "PARTITION BY " + ", ".join(vdf_columns_names(by, self))
		order_by = [column] if not(order_by) else vdf_columns_names(order_by, self)
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
		check_types([("x", x, [int, float], False)])
		if ((x <= 0) or (x >= 1)):
			raise ValueError("Parameter 'x' must be between 0 and 1")
		name = "__vertica_ml_python_random_" + str(random.randint(0, 10000000)) + "__"
		self.eval(name, "RANDOM()")
		self.filter("{} < {}".format(name, x), print_info = False)
		self[name].drop()
		return (self)
	#
	def save(self):
		save = 'vdf_save = vDataframe("", empty = True)'
		save += '\nvdf_save.VERTICA_ML_PYTHON_VARIABLES["dsn"] = \'{}\''.format(self.VERTICA_ML_PYTHON_VARIABLES["dsn"])
		save += '\nvdf_save.VERTICA_ML_PYTHON_VARIABLES["input_relation"] = \'{}\''.format(self.VERTICA_ML_PYTHON_VARIABLES["input_relation"])
		save += '\nvdf_save.VERTICA_ML_PYTHON_VARIABLES["main_relation"] = \'{}\''.format(self.VERTICA_ML_PYTHON_VARIABLES["main_relation"])
		save += '\nvdf_save.VERTICA_ML_PYTHON_VARIABLES["schema"] = \'{}\''.format(self.VERTICA_ML_PYTHON_VARIABLES["schema"])
		save += '\nvdf_save.VERTICA_ML_PYTHON_VARIABLES["columns"] = {}'.format(self.VERTICA_ML_PYTHON_VARIABLES["columns"])
		save += '\nvdf_save.VERTICA_ML_PYTHON_VARIABLES["exclude_columns"] = {}'.format(self.VERTICA_ML_PYTHON_VARIABLES["exclude_columns"])
		save += '\nvdf_save.VERTICA_ML_PYTHON_VARIABLES["where"] = {}'.format(self.VERTICA_ML_PYTHON_VARIABLES["where"])
		save += '\nvdf_save.VERTICA_ML_PYTHON_VARIABLES["query_on"] = {}'.format(self.VERTICA_ML_PYTHON_VARIABLES["query_on"])
		save += '\nvdf_save.VERTICA_ML_PYTHON_VARIABLES["time_on"] = {}'.format(self.VERTICA_ML_PYTHON_VARIABLES["time_on"])
		save += '\nvdf_save.VERTICA_ML_PYTHON_VARIABLES["order_by"] = {}'.format(self.VERTICA_ML_PYTHON_VARIABLES["order_by"])
		save += '\nvdf_save.VERTICA_ML_PYTHON_VARIABLES["history"] = {}'.format(self.VERTICA_ML_PYTHON_VARIABLES["history"])
		save += '\nvdf_save.VERTICA_ML_PYTHON_VARIABLES["saving"] = {}'.format(self.VERTICA_ML_PYTHON_VARIABLES["saving"])
		for column in self.VERTICA_ML_PYTHON_VARIABLES["columns"]:
			save += '\nsave_vColumn = vColumn(\'{}\', parent = vdf_save, transformations = {})'.format(column, self[column].transformations)
			save += '\nsetattr(vdf_save, \'{}\', save_vColumn)'.format(column)
			save += '\nsetattr(vdf_save, \'{}\', save_vColumn)'.format(column[1:-1])
		self.VERTICA_ML_PYTHON_VARIABLES["saving"] += [save]
		return (self)
	# 
	def scatter(self,
			  	columns: list,
			  	catcol: str = "",
			  	max_cardinality: int = 3,
			  	cat_priority: list = [],
			  	with_others: bool = True,
			  	max_nb_points: int = 20000):
		check_types([("columns", columns, [list], False), ("catcol", catcol, [str], False), ("max_cardinality", max_cardinality, [int, float], False), ("cat_priority", cat_priority, [list], False), ("with_others", with_others, [bool], False), ("max_nb_points", max_nb_points, [int, float], False)])
		columns_check(columns, self, [2, 3])
		columns = vdf_columns_names(columns, self)
		if (catcol):
			columns_check([catcol], self)
			catcol = vdf_columns_names([catcol], self) 
		else:
			catcol = []
		if (len(columns) == 2):
			from vertica_ml_python.plot import scatter2D
			scatter2D(self, columns + catcol, max_cardinality, cat_priority, with_others, max_nb_points)	
		else:
			from vertica_ml_python.plot import scatter3D
			scatter3D(self, columns + catcol, max_cardinality, cat_priority, with_others, max_nb_points)
		return (self)
	# 
	def scatter_matrix(self, columns: list = []):
		check_types([("columns", columns, [list], False)])
		columns_check(columns, self)
		columns = vdf_columns_names(columns, self)
		from vertica_ml_python.plot import scatter_matrix
		scatter_matrix(self, columns)
		return (self)	
	#
	def select(self, columns: list):
		check_types([("columns", columns, [list], False)])
		columns_check(columns, self)
		columns = vdf_columns_names(columns, self)
		copy_vDataframe = self.copy()
		for column in copy_vDataframe.get_columns():
			column_tmp = str_column(column)
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
		check_types([("ts", ts, [str], False), ("by", by, [list], False), ("session_threshold", session_threshold, [str], False), ("name", name, [str], False)])
		columns_check(by + [ts], self)
		by = vdf_columns_names(by, self)
		ts = vdf_columns_names([ts], self)[0]
		partition = "PARTITION BY {}".format(", ".join(by)) if (by) else ""
		expr = "CONDITIONAL_TRUE_EVENT({} - LAG({}) > '{}') OVER ({} ORDER BY {})".format(ts, ts, session_threshold, partition, ts)
		return (self.eval(name = name, expr = expr))
	# 
	def set_cursor(self, cursor):
		try:
			cursor.execute("SELECT 1;")
			cursor.fetchone()
		except Exception as e:
			raise TypeError("{}\nThe parameter 'cursor' must be a DB cursor having the methods fetchall, fetchone and execute.".format(e))
		self.VERTICA_ML_PYTHON_VARIABLES["cursor"] = cursor
		return (self)
	# 
	def set_dsn(self, dsn: str):
		check_types([("dsn", dsn, [str], False)])
		self.VERTICA_ML_PYTHON_VARIABLES["dsn"] = dsn
		return (self)
	# 
	def shape(self):
		query = "SELECT COUNT(*) FROM {}".format(self.genSQL())
		self.VERTICA_ML_PYTHON_VARIABLES["cursor"].execute(query)
		return (self.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchone()[0], len(self.get_columns()))
	#
	def skew(self, columns: list = []):
		return self.skewness(columns = columns)
	def skewness(self, columns: list = []):
		stats = self.statistics(columns = columns, skew_kurt_only = True)
		for column in stats.values:
			del(stats.values[column][1])
		return (stats.transpose())
	# 
	def sort(self, columns: list, desc: bool = False):
		check_types([("columns", columns, [list], False), ("desc", desc, [bool], False)])
		columns_check(columns, self)
		columns, max_pos, vdf_columns = vdf_columns_names(columns, self), 0, self.get_columns()
		for column in columns:
			if not(column_check_ambiguous(column, vdf_columns)):
				raise NameError("The Virtual Column {} doesn't exist".format(column))
		for column in self.VERTICA_ML_PYTHON_VARIABLES["columns"]:
			max_pos = max(max_pos, len(self[column].transformations) - 1)
		self.VERTICA_ML_PYTHON_VARIABLES["order_by"][max_pos] = " ORDER BY {} {}".format(", ".join(columns), "DESC" if (desc) else "ASC")
		return (self)
	# 
	def sql_on_off(self):
		self.VERTICA_ML_PYTHON_VARIABLES["query_on"] = not(self.VERTICA_ML_PYTHON_VARIABLES["query_on"])
		return (self)
	#
	def statistics(self, 
				   columns: list = [], 
				   skew_kurt_only: bool = False):
		check_types([("columns", columns, [list], False), ("skew_kurt_only", skew_kurt_only, [bool], False)])
		columns_check(columns, self)
		columns = self.numcol() if not(columns) else vdf_columns_names(columns, self)
		query = []
		if (skew_kurt_only):
			stats = self.aggregate(func = ["count", "avg", "stddev"], columns = columns).transpose()
		else:
			stats = self.aggregate(func = ["count", "avg", "stddev", "min", "10%", "25%", "median", "75%", "90%", "max"], columns = columns).transpose()
		for column in columns:
			cast = "::int" if (self[column].ctype() == "boolean") else ""
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
		result = [item for item in self.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchone()]
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
		check_types([("limit", limit, [int, float], False), ("offset", offset, [int, float], False)])
		columns = self.get_columns()
		all_columns = []
		for column in columns:
			all_columns += ["{} AS {}".format(convert_special_type(self[column].category(), True, column), column)]
		tail = to_tablesample("SELECT {} FROM {} LIMIT {} OFFSET {}".format(", ".join(all_columns), self.genSQL(), limit, offset), self.VERTICA_ML_PYTHON_VARIABLES["cursor"])
		tail.count = self.shape()[0]
		tail.offset = offset
		tail.name = self.VERTICA_ML_PYTHON_VARIABLES["input_relation"]
		for column in tail.values:
			tail.dtype[column] = self[column].ctype()
		return (tail)
	# 
	def time_on_off(self):
		self.VERTICA_ML_PYTHON_VARIABLES["time_on"] = not(self.VERTICA_ML_PYTHON_VARIABLES["time_on"])
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
		check_types([("name", name, [str], False), ("path", path, [str], False), ("sep", sep, [str], False), ("na_rep", na_rep, [str], False), ("quotechar", quotechar, [str], False), ("usecols", usecols, [list], False), ("header", header, [bool], False), ("new_header", new_header, [list], False), ("order_by", order_by, [list], False), ("nb_row_per_work", nb_row_per_work, [int, float], False)])
		file = open("{}{}.csv".format(path, name), "w+") 
		columns = self.get_columns() if not(usecols) else [str_column(column) for column in usecols]
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
			self.VERTICA_ML_PYTHON_VARIABLES["cursor"].execute("SELECT {} FROM {} LIMIT {} OFFSET {}".format(", ".join(columns), self.genSQL(), limit, current_nb_rows_written))
			result = self.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchall()
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
		check_types([("name", name, [str], False), ("usecols", usecols, [list], False), ("relation_type", relation_type, ["view", "temporary", "table"], True), ("inplace", inplace, [bool], False)])
		columns_check(usecols, self)
		usecols = vdf_columns_names(usecols, self)
		if (relation_type == "temporary"):
			relation_type += " table" 
		usecols = "*" if not(usecols) else ", ".join([str_column(column) for column in usecols])
		query = "CREATE {} {} AS SELECT {} FROM {}".format(relation_type.upper(), name, usecols, self.genSQL())
		self.executeSQL(query = query, title = "Create a new " + relation_type + " to save the vDataframe")
		self.add_to_history("[Save]: The vDataframe was saved into a {} named '{}'.".format(relation_type, name))
		if (inplace):
			query_on, time_on, history, saving = self.VERTICA_ML_PYTHON_VARIABLES["query_on"], self.VERTICA_ML_PYTHON_VARIABLES["time_on"], self.VERTICA_ML_PYTHON_VARIABLES["history"], self.VERTICA_ML_PYTHON_VARIABLES["saving"]
			self.__init__(name, self.VERTICA_ML_PYTHON_VARIABLES["cursor"])
			self.VERTICA_ML_PYTHON_VARIABLES["history"] = history
			self.VERTICA_ML_PYTHON_VARIABLES["query_on"] = query_on
			self.VERTICA_ML_PYTHON_VARIABLES["time_on"] = time_on
		return (self)
	# 
	def to_pandas(self):
		import pandas as pd
		query = "SELECT * FROM {}".format(self.genSQL())
		self.VERTICA_ML_PYTHON_VARIABLES["cursor"].execute(query)
		column_names = [column[0] for column in self.VERTICA_ML_PYTHON_VARIABLES["cursor"].description]
		query_result = self.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchall()
		data = [list(item) for item in query_result]
		df = pd.DataFrame(data)
		df.VERTICA_ML_PYTHON_VARIABLES["columns"] = column_names
		return (df)
	#
	def to_vdf(self, name: str):
		check_types([("name", name, [str], False)])
		self.save()
		file = open("{}.vdf".format(name), "w+") 
		file.write(self.VERTICA_ML_PYTHON_VARIABLES["saving"][-1])
		file.close()
		return (self)
	#
	def var(self, columns = []):
		return (self.aggregate(func = ["variance"], columns = columns))
	#
	def vdf_from_relation(self, table: str, func: str, history: str):
		check_types([("table", table, [str], False), ("func", func, [str], False), ("history", history, [str], False)])
		vdf = vDataframe("", empty = True)
		vdf.VERTICA_ML_PYTHON_VARIABLES["dsn"] = self.VERTICA_ML_PYTHON_VARIABLES["dsn"]
		vdf.VERTICA_ML_PYTHON_VARIABLES["input_relation"] = self.VERTICA_ML_PYTHON_VARIABLES["input_relation"]
		vdf.VERTICA_ML_PYTHON_VARIABLES["main_relation"] = table
		vdf.VERTICA_ML_PYTHON_VARIABLES["schema"] = self.VERTICA_ML_PYTHON_VARIABLES["schema"]
		vdf.VERTICA_ML_PYTHON_VARIABLES["cursor"] = self.VERTICA_ML_PYTHON_VARIABLES["cursor"]
		vdf.VERTICA_ML_PYTHON_VARIABLES["query_on"] = self.VERTICA_ML_PYTHON_VARIABLES["query_on"]
		vdf.VERTICA_ML_PYTHON_VARIABLES["time_on"] = self.VERTICA_ML_PYTHON_VARIABLES["time_on"]
		vdf.VERTICA_ML_PYTHON_VARIABLES["where"] = []
		vdf.VERTICA_ML_PYTHON_VARIABLES["order_by"] = ['' for i in range(100)]
		vdf.VERTICA_ML_PYTHON_VARIABLES["exclude_columns"] = []
		vdf.VERTICA_ML_PYTHON_VARIABLES["history"] = [item for item in self.VERTICA_ML_PYTHON_VARIABLES["history"]] + [history]
		vdf.VERTICA_ML_PYTHON_VARIABLES["saving"] = [item for item in self.VERTICA_ML_PYTHON_VARIABLES["saving"]]
		self.executeSQL(query = "DROP TABLE IF EXISTS v_temp_schema.VERTICA_ML_PYTHON_{}_TEST;".format(func), title = "Drop the Existing Temp Table")
		self.executeSQL(query = "CREATE LOCAL TEMPORARY TABLE VERTICA_ML_PYTHON_{}_TEST ON COMMIT PRESERVE ROWS AS SELECT * FROM {} LIMIT 10;".format(func, table))
		self.executeSQL(query = "SELECT column_name, data_type FROM columns where table_name = 'VERTICA_ML_PYTHON_{}_TEST' AND table_schema = 'v_temp_schema'".format(func), title = "SELECT NEW DATA TYPE AND THE COLUMNS NAME")
		result = self.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchall()
		self.executeSQL(query = "DROP TABLE IF EXISTS v_temp_schema.VERTICA_ML_PYTHON_{}_TEST;".format(func), title = "Drop the Temp Table")
		vdf.VERTICA_ML_PYTHON_VARIABLES["columns"] = ['"{}"'.format(item[0]) for item in result]
		for column, ctype in result:
			if ('"' in column):
				print("\u26A0 Warning: A double quote \" was found in the column {}, its alias was changed using underscores '_' to {}".format(column, column.replace('"', '_')))
			new_vColumn = vColumn('"{}"'.format(column.replace('"', '_')), parent = self, transformations = [('"{}"'.format(column.replace('"', '""')), ctype, category_from_type(ctype))])
			setattr(vdf, '"{}"'.format(column.replace('"', '_')), new_vColumn)
			setattr(vdf, column.replace('"', '_'), new_vColumn)
		return (vdf)
	# 
	def version(self):
		self.VERTICA_ML_PYTHON_VARIABLES["cursor"].execute("SELECT version();")
		return self.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchone()[0]