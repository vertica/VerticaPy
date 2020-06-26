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
#
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
import random, time, shutil, re
# VerticaPy Modules
from verticapy.vcolumn import vColumn
from verticapy.utilities import *
from verticapy.connections.connect import read_auto_connect
from verticapy.toolbox import *
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
#---#
class vDataFrame:
	"""
---------------------------------------------------------------------------
Python object which will keep in mind all the user modifications 
in order to use the correct SQL code generation. It will send SQL 
queries to the Vertica DB which will aggregate and return the final 
result. vDataFrame will create for each column of the relation a 
Virtual Column (vcolumn) which will store the column alias and all 
the user transformations. Thus, vDataFrame allows to do easy data 
preparation and exploration without modifying the data.

Parameters
----------
input_relation: str
	Relation (View, Table or Temporary Table) used to create the object.
	To get a specific schema relation, your string must include both
	relation and schema: 'schema.relation' or '"schema"."relation"'.
	You can also use the 'schema' parameter to be less ambiguous.
	In this case input_relation must only be the relation name (it must 
	not include a schema).
cursor: DBcursor, optional
	Vertica DB cursor. 
	For a cursor designed by Vertica, search for vertica_python
	For ODBC, search for pyodbc.
	For JDBC, search for jaydebeapi.
	Check out utilities.vHelp function, it may help you.
dsn: str, optional
	Data Base DSN. OS File including the DB credentials.
	VERTICAPY will try to create a vertica_python cursor first.
	If it didn't find the library, it will try to create a pyodbc cursor.
	Check out utilities.vHelp function, it may help you.
usecols: list, optional
	List of columns used to create the object. As Vertica is a columnar DB
	including less columns is easy and makes the process faster. Do not hesitate
	to not include useless columns.
schema: str, optional
	Relation schema. It can be used to be less ambiguous and allow to create schema 
	and relation name with dots '.' inside.
empty: bool, optional
	If set to True, the created object will be empty. It can be used to create customized 
	vDataFrame without going through the initialization check.

Attributes
----------
_VERTICAPY_VARIABLES_: dict
	Dictionary containing all the vDataFrame attributes.
		allcols_ind, int      : Int used to optimize the SQL code generation.
		columns, list         : List of the vcolumns names.
		count, int            : Number of elements of the vDataFrame (catalog).
		cursor, DBcursor      : Vertica Database cursor.
		dsn, str              : Vertica Database DSN.
		exclude_columns, list : vcolumns to exclude from the final relation.
		history, list         : vDataFrame history (user modifications).
		input_relation, str   : Name of the vDataFrame.
		main_relation, str    : Relation used to build the vDataFrame (first floor).
		order_by, dict        : Dictionary of all the rules to sort the vDataFrame.
		query_on, bool        : If set to True, all the query will be printed.
		saving, list          : List used to reconstruct the vDataFrame.
		schema, str           : Schema of the input relation.
		schema_writing, str   : Schema used to create temporary tables when needed.
		time_on, bool         : If set to True, all the query elapsed time will be printed.
		where, list           : List of all the rules to filter the vDataFrame.
vcolumns : vcolumn
	Each vcolumn of the vDataFrame is accessible by entering its name between brackets
	For example to access to "myVC", you can write vDataFrame["myVC"].
	"""
	#
	# Special Methods
	#
	#---#
	def  __init__(self,
				  input_relation: str,
				  cursor = None,
				  dsn: str = "",
				  usecols: list = [],
				  schema: str = "",
				  empty: bool = False):
		check_types([
			("input_relation", input_relation, [str], False), 
			("dsn", dsn, [str], False), 
			("usecols", usecols, [list], False), 
			("schema", schema, [str], False), 
			("empty", empty, [bool], False)])
		self._VERTICAPY_VARIABLES_ = {}
		self._VERTICAPY_VARIABLES_["count"] = -1
		self._VERTICAPY_VARIABLES_["allcols_ind"] = -1
		if not(empty):
			if not(cursor) and not(dsn):
				cursor = read_auto_connect().cursor()
			elif not(cursor):
				from verticapy import vertica_cursor
				cursor = vertica_cursor(dsn)
			else:
				check_cursor(cursor)
			self._VERTICAPY_VARIABLES_["dsn"] = dsn
			if not(schema):
				schema, input_relation = schema_relation(input_relation)
			self._VERTICAPY_VARIABLES_["schema"] = schema.replace('"', '')
			self._VERTICAPY_VARIABLES_["input_relation"] = input_relation.replace('"', '')
			self._VERTICAPY_VARIABLES_["cursor"] = cursor
			where = " AND LOWER(column_name) IN ({})".format(", ".join(["'{}'".format(elem.lower().replace("'", "''")) for elem in usecols])) if (usecols) else ""
			query = "(SELECT column_name, data_type FROM columns WHERE table_name = '{}' AND table_schema = '{}'{})".format(
				self._VERTICAPY_VARIABLES_["input_relation"].replace("'", "''"), self._VERTICAPY_VARIABLES_["schema"].replace("'", "''"), where)
			query += " UNION (SELECT column_name, data_type FROM view_columns WHERE table_name = '{}' AND table_schema = '{}'{})".format(
				self._VERTICAPY_VARIABLES_["input_relation"].replace("'", "''"), self._VERTICAPY_VARIABLES_["schema"].replace("'", "''"), where)
			cursor.execute(query)
			columns_dtype = cursor.fetchall()
			columns_dtype = [(str(item[0]), str(item[1])) for item in columns_dtype]
			columns = ['"{}"'.format(elem[0].replace('"', '_')) for elem in columns_dtype]
			if not(usecols): self._VERTICAPY_VARIABLES_["allcols_ind"] = len(columns)
			if (columns != []):
				self._VERTICAPY_VARIABLES_["columns"] = columns
			else:
				raise ValueError("No table or views '{}' found.".format(self._VERTICAPY_VARIABLES_["input_relation"]))
			for col_dtype in columns_dtype:
				column, dtype = col_dtype[0], col_dtype[1]
				if ('"' in column):
					print("\u26A0 Warning: A double quote \" was found in the column {}, its alias was changed using underscores '_' to {}".format(column, column.replace('"', '_')))
				new_vColumn = vColumn('"{}"'.format(column.replace('"', '_')), parent = self, transformations = [('"{}"'.format(column.replace('"', '""')), dtype, category_from_type(dtype))])
				setattr(self, '"{}"'.format(column.replace('"', '_')), new_vColumn)
				setattr(self, column.replace('"', '_'), new_vColumn)
			self._VERTICAPY_VARIABLES_["exclude_columns"] = []
			self._VERTICAPY_VARIABLES_["where"] = []
			self._VERTICAPY_VARIABLES_["order_by"] = {}
			self._VERTICAPY_VARIABLES_["time_on"] = False
			self._VERTICAPY_VARIABLES_["query_on"] = False
			self._VERTICAPY_VARIABLES_["history"] = []
			self._VERTICAPY_VARIABLES_["saving"] = []
			self._VERTICAPY_VARIABLES_["main_relation"] = '"{}"."{}"'.format(self._VERTICAPY_VARIABLES_["schema"], self._VERTICAPY_VARIABLES_["input_relation"])
			self._VERTICAPY_VARIABLES_["schema_writing"] = ""
	#---#
	def __getitem__(self, index):
		try:
			return getattr(self, index)
		except:
			new_index = vdf_columns_names([index], self)
			if (len(new_index) == 1):
				return getattr(self, new_index[0])
			else: 
				raise
	#---#
	def __repr__(self):
		return self.head(limit = 5).__repr__()
	#---#
	def __setattr__(self, attr, val):
		self.__dict__[attr] = val
	#---#
	def __setitem__(self, index, val):
		setattr(self, index, val)
	#---#
	def __add_to_history__(self, 
					   	   message: str):
		"""
	---------------------------------------------------------------------------
	VERTICAPY stores the user modification and help the user to look at 
	what he/she did. This method is used to add a customized message in the 
	vDataFrame history attribute.
		"""
		check_types([("message", message, [str], False)])
		self._VERTICAPY_VARIABLES_["history"] += ["{}{}{} {}".format("{", time.strftime("%c"), "}", message)]
		return (self)
	#---#
	def __aggregate_matrix__(self, 
							 method: str = "pearson",
			 	   		 	 columns: list = [], 
			 	   		 	 cmap: str = "",
			 	   		 	 round_nb: int = 3,
			 	   		 	 show: bool = True):
		"""
	---------------------------------------------------------------------------
	Global method used to compute the Correlation/Cov/Beta Matrix.

	See Also
	--------
	vDataFrame.corr : Computes the Correlation Matrix of the vDataFrame.
	vDataFrame.cov  : Computes the Covariance Matrix of the vDataFrame.
	vDataFrame.regr : Computes the Regression Matrix of the vDataFrame.
		"""
		columns = vdf_columns_names(columns, self)
		if (len(columns) == 1):
			if (method in ("pearson", "spearman", "kendall", "biserial", "cramer")):
				return 1.0
			elif (method == "cov"):
				return self[columns[0]].var()
		elif (len(columns) == 2):
			pre_comp_val = self.__get_catalog_value__(method = method, columns = columns)
			if (pre_comp_val != "VERTICAPY_NOT_PRECOMPUTED"):
				return (pre_comp_val)
			cast_0 = "::int" if (self[columns[0]].ctype() == "boolean") else ""
			cast_1 = "::int" if (self[columns[1]].ctype() == "boolean") else ""
			if (method in ("pearson", "spearman")):
				if (columns[1] == columns[0]):
					return 1
				table = self.__genSQL__() if (method == "pearson") else "(SELECT RANK() OVER (ORDER BY {}) AS {}, RANK() OVER (ORDER BY {}) AS {} FROM {}) rank_spearman_table".format(columns[0], columns[0], columns[1], columns[1], self.__genSQL__())
				query = "SELECT CORR({}{}, {}{}) FROM {}".format(columns[0], cast_0, columns[1], cast_1, table)
				title = "Computes the {} correlation between {} and {}.".format(method, columns[0], columns[1])
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
					column_b, cast_b, column_n, cast_n, column_b, cast_b, column_n, cast_n, column_n, cast_n, column_b, cast_b, column_b, cast_b, self.__genSQL__(), column_n, column_b)
				title = "Computes the biserial correlation between {} and {}.".format(column_b, column_n)
			elif (method == "cramer"):
				if (columns[1] == columns[0]):
					return 1
				table_0_1 = "SELECT {}, {}, COUNT(*) AS nij FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL GROUP BY 1, 2".format(columns[0], columns[1], self.__genSQL__(), columns[0], columns[1])
				table_0 = "SELECT {}, COUNT(*) AS ni FROM {} WHERE {} IS NOT NULL GROUP BY 1".format(columns[0], self.__genSQL__(), columns[0])
				table_1 = "SELECT {}, COUNT(*) AS nj FROM {} WHERE {} IS NOT NULL GROUP BY 1".format(columns[1], self.__genSQL__(), columns[1])
				self._VERTICAPY_VARIABLES_["cursor"].execute("SELECT COUNT(*) AS n FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL".format(self.__genSQL__(), columns[0], columns[1]))
				n = self._VERTICAPY_VARIABLES_["cursor"].fetchone()[0]
				phi = "SELECT SUM((nij - ni * nj / {}) * (nij - ni * nj / {}) / (ni * nj)) AS phi2 FROM (SELECT * FROM ({}) table_0_1 LEFT JOIN ({}) table_0 ON table_0_1.{} = table_0.{}) x LEFT JOIN ({}) table_1 ON x.{} = table_1.{}".format(n, n, table_0_1, table_0, columns[0], columns[0], table_1, columns[1], columns[1])
				r_k_n_phi = "WITH cardinality AS (SELECT APPROXIMATE_COUNT_DISTINCT({}) AS k, APPROXIMATE_COUNT_DISTINCT({}) AS r FROM {}), phi2_table AS ({}) ".format(columns[0], columns[1], self.__genSQL__(), phi)
				r_k_n_phi += "SELECT SQRT((CASE WHEN phi2_adjusted < 0 THEN 0 ELSE phi2_adjusted END) / NULLIFZERO(CASE WHEN k_adjusted < r_adjusted THEN k_adjusted ELSE r_adjusted END)) AS kendall FROM (SELECT phi2, k, r, phi2 - (r - 1) * (k - 1) / ({} - 1) AS phi2_adjusted, k - (k - 1) * (k - 1) / ({} - 1) AS k_adjusted, r - (r - 1) * (r - 1) / ({} - 1) AS r_adjusted FROM cardinality, phi2_table) x".format(n, n, n)
				self.__executeSQL__(r_k_n_phi, title = "Computes the CramerV correlation between {} and {}.".format(columns[0], columns[1]))
				result = self._VERTICAPY_VARIABLES_["cursor"].fetchone()[0]
				try:
					result = float(result)
				except:
					result = float('nan')
				if (result > 1 or result < 0): result = float('nan') 
				return result
			elif (method == "kendall"):
				if (columns[1] == columns[0]):
					return 1
				query = "SELECT (SUM(((x.{}{} < y.{}{} AND x.{}{} < y.{}{}) OR (x.{}{} > y.{}{} AND x.{}{} > y.{}{}))::int) - SUM(((x.{}{} > y.{}{} AND x.{}{} < y.{}{}) OR (x.{}{} < y.{}{} AND x.{}{} > y.{}{}))::int)) / SUM((x.{}{} != y.{}{} AND x.{}{} != y.{}{})::int) FROM (SELECT {}, {} FROM {}) x CROSS JOIN (SELECT {}, {} FROM {}) y"
				query = query.format(columns[0], cast_0, columns[0], cast_0, columns[1], cast_1, columns[1], cast_1, columns[0], cast_0, columns[0], cast_0, columns[1], cast_1, columns[1], cast_1, columns[0], cast_0, columns[0], cast_0, columns[1], cast_1, columns[1], cast_1, columns[0], cast_0, columns[0], cast_0, columns[1], cast_1, columns[1], cast_1, columns[0], cast_0, columns[0], cast_0, columns[1], cast_1, columns[1], cast_1, columns[0], columns[1], self.__genSQL__(), columns[0], columns[1], self.__genSQL__())
				title = "Computes the kendall correlation between {} and {}.".format(columns[0], columns[1])
			elif (method == "cov"):
				query = "SELECT COVAR_POP({}{}, {}{}) FROM {}".format(columns[0], cast_0, columns[1], cast_1, self.__genSQL__())
				title = "Computes the covariance between {} and {}.".format(columns[0], columns[1])
			try:
				self.__executeSQL__(query = query, title = title)
				result = self._VERTICAPY_VARIABLES_["cursor"].fetchone()[0]
			except:
				result = float('nan')
			self.__update_catalog__(values = {columns[1]: result}, matrix = method, column = columns[0])
			self.__update_catalog__(values = {columns[0]: result}, matrix = method, column = columns[1])
			return (result)
		elif (len(columns) > 2):
			try:
				nb_precomputed, n = 0, len(columns)
				for column1 in columns:
					for column2 in columns:
						pre_comp_val = self.__get_catalog_value__(method = method, columns = [column1, column2])
						if (pre_comp_val != "VERTICAPY_NOT_PRECOMPUTED"):
							nb_precomputed += 1
				if (nb_precomputed > n * n / 3) or (method not in ("pearson", "spearman")):
					raise
				else:
					table = self.__genSQL__() if (method == "pearson") else "(SELECT {} FROM {}) spearman_table".format(", ".join(["RANK() OVER (ORDER BY {}) AS {}".format(column, column) for column in columns]), self.__genSQL__())
					self.__executeSQL__(query = "SELECT CORR_MATRIX({}) OVER () FROM {}".format(", ".join(columns), table), title = "Computes the {} Corr Matrix.".format(method))
					result = self._VERTICAPY_VARIABLES_["cursor"].fetchall()
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
			except:
				if (method in ("pearson", "spearman", "kendall", "biserial", "cramer")):
					title_query = "Compute all the Correlations in a single query"
					title = 'Correlation Matrix ({})'.format(method)
					if (method == "biserial"):
						i0, step = 0, 1
					else:
						i0, step = 1, 0 
				elif (method == "cov"):
					title_query = "Compute all the covariances in a single query"
					title = 'Covariance Matrix'
					i0, step = 0, 1
				try:
					all_list = []
					n = len(columns)
					nb_precomputed = 0
					nb_loop = 0
					for i in range(i0, n):
						for j in range(0, i + step):
							nb_loop += 1
							cast_i = "::int" if (self[columns[i]].ctype() == "boolean") else ""
							cast_j = "::int" if (self[columns[j]].ctype() == "boolean") else ""
							pre_comp_val = self.__get_catalog_value__(method = method, columns = [columns[i], columns[j]])
							if (pre_comp_val == None or pre_comp_val != pre_comp_val): pre_comp_val = 'NULL'
							if (pre_comp_val != "VERTICAPY_NOT_PRECOMPUTED"):
								all_list += [str(pre_comp_val)]
								nb_precomputed += 1
							elif (method in ("pearson", "spearman")):
								all_list += ["ROUND(CORR({}{}, {}{}), {})".format(columns[i], cast_i, columns[j], cast_j, round_nb)]
							elif (method == "kendall"):
								all_list += ["(SUM(((x.{}{} < y.{}{} AND x.{}{} < y.{}{}) OR (x.{}{} > y.{}{} AND x.{}{} > y.{}{}))::int) - SUM(((x.{}{} > y.{}{} AND x.{}{} < y.{}{}) OR (x.{}{} < y.{}{} AND x.{}{} > y.{}{}))::int)) / NULLIFZERO(SUM((x.{}{} != y.{}{} AND x.{}{} != y.{}{})::int))".format(
									columns[i], cast_i, columns[i], cast_i, columns[j], cast_j, columns[j], cast_j, columns[i], cast_i, columns[i], cast_i, columns[j], cast_j, columns[j], cast_j, columns[i], cast_i, columns[i], cast_i, columns[j], cast_j, columns[j], cast_j, columns[i], cast_i, columns[i], cast_i, columns[j], cast_j, columns[j], cast_j, columns[i], cast_i, columns[i], cast_i, columns[j], cast_j, columns[j], cast_j)]
							elif (method == "cov"):
								all_list += ["COVAR_POP({}{}, {}{})".format(columns[i], cast_i, columns[j], cast_j)]
							else:
								raise
					if (method == "spearman"):
						rank = ["RANK() OVER (ORDER BY {}) AS {}".format(column, column) for column in columns]
						table = "(SELECT {} FROM {}) rank_spearman_table".format(", ".join(rank), self.__genSQL__())
					elif (method == "kendall"):
						table = "(SELECT {} FROM {}) x CROSS JOIN (SELECT {} FROM {}) y".format(", ".join(columns), self.__genSQL__(), ", ".join(columns), self.__genSQL__())
					else:
						table = self.__genSQL__()
					if (nb_precomputed == nb_loop):
						self._VERTICAPY_VARIABLES_["cursor"].execute("SELECT {}".format(", ".join(all_list)))
					else:
						self.__executeSQL__(query = "SELECT {} FROM {}".format(", ".join(all_list), table), title = title_query)
					result = self._VERTICAPY_VARIABLES_["cursor"].fetchone()
				except:
					n = len(columns)
					result = []
					for i in range(i0, n):
						for j in range(0, i + step):
							result += [self.__aggregate_matrix__(method, [columns[i], columns[j]])]
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
						matrix[j + 1][i + 1] = current
			if (show):
				from verticapy.plot import cmatrix
				vmin = 0 if (method == "cramer") else -1
				if (method in ("cov")): vmin = None
				vmax = 1 if (method in ("pearson", "spearman", "kendall", "biserial", "cramer")) else None
				if not(cmap):
					from verticapy.plot import gen_cmap
					cm1, cm2 = gen_cmap()
					cmap = cm1 if (method == "cramer") else cm2
				cmatrix(matrix, columns, columns, n, n, vmax = vmax, vmin = vmin, cmap = cmap, title = title, mround = round_nb)
			values = {"index" : matrix[0][1:len(matrix[0])]}
			del(matrix[0])
			for column in matrix:
				values[column[0]] = column[1:len(column)]
			for column1 in values:
				if (column1 != "index"):
					val = {}
					for idx, column2 in enumerate(values["index"]):
						val[column2] = values[column1][idx]
					self.__update_catalog__(values = val, matrix = method, column = column1)
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
			return (self.__aggregate_matrix__(method = method, columns = cols, cmap = cmap, round_nb = round_nb, show = show))
	#---# 
	def __aggregate_vector__(self, 
						 	 focus: str,
						 	 method: str = "pearson",
			 	   		 	 columns: list = [], 
			 	   		 	 cmap: str = "",
			 	   		 	 round_nb: int = 3,
			 	   		 	 show: bool = True):
		"""
	---------------------------------------------------------------------------
	Global method used to compute the Correlation/Cov/Beta Vector.

	See Also
	--------
	vDataFrame.corr : Computes the Correlation Matrix of the vDataFrame.
	vDataFrame.cov  : Computes the Covariance Matrix of the vDataFrame.
	vDataFrame.regr : Computes the Regression Matrix of the vDataFrame.
		"""
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
		if (method in ('spearman', 'pearson', 'kendall') and (len(cols) >= 1)):
			try:
				fail = 0
				cast_i = "::int" if (self[focus].ctype() == "boolean") else ""
				all_list, all_cols  = [], [focus]
				nb_precomputed = 0
				for column in cols:
					if (column.replace('"', '').lower() != focus.replace('"', '').lower()):
						all_cols += [column]
					cast_j = "::int" if (self[column].ctype() == "boolean") else ""
					pre_comp_val = self.__get_catalog_value__(method = method, columns = [focus, column])
					if (pre_comp_val == None or pre_comp_val != pre_comp_val): pre_comp_val = 'NULL' 
					if (pre_comp_val != "VERTICAPY_NOT_PRECOMPUTED"):
						all_list += [str(pre_comp_val)]
						nb_precomputed += 1
					elif (method in ("pearson", "spearman")):
						all_list += ["ROUND(CORR({}{}, {}{}), {})".format(focus, cast_i, column, cast_j, round_nb)]
					elif (method == "kendall"):
						all_list += ["(SUM(((x.{}{} < y.{}{} AND x.{}{} < y.{}{}) OR (x.{}{} > y.{}{} AND x.{}{} > y.{}{}))::int) - SUM(((x.{}{} > y.{}{} AND x.{}{} < y.{}{}) OR (x.{}{} < y.{}{} AND x.{}{} > y.{}{}))::int)) / NULLIFZERO(SUM((x.{}{} != y.{}{} AND x.{}{} != y.{}{})::int))".format(
							focus, cast_i, focus, cast_i, column, cast_j, column, cast_j, focus, cast_i, focus, cast_i, column, cast_j, column, cast_j, focus, cast_i, focus, cast_i, column, cast_j, column, cast_j, focus, cast_i, focus, cast_i, column, cast_j, column, cast_j, focus, cast_i, focus, cast_i, column, cast_j, column, cast_j)]
					elif (method == "cov"):
						all_list += ["COVAR_POP({}{}, {}{})".format(focus, cast_i, column, cast_j)]
				if (method == "spearman"):
					rank = ["RANK() OVER (ORDER BY {}) AS {}".format(column, column) for column in all_cols]
					table = "(SELECT {} FROM {}) rank_spearman_table".format(", ".join(rank), self.__genSQL__())
				elif (method == "kendall"):
					table = "(SELECT {} FROM {}) x CROSS JOIN (SELECT {} FROM {}) y".format(", ".join(all_cols), self.__genSQL__(), ", ".join(all_cols), self.__genSQL__())
				else:
					table = self.__genSQL__()
				if (nb_precomputed == len(cols)):
					self._VERTICAPY_VARIABLES_["cursor"].execute("SELECT {}".format(", ".join(all_list)))
				else:
					self.__executeSQL__(query = "SELECT {} FROM {} LIMIT 1".format(", ".join(all_list), table), title = "Computes the Correlation Vector ({})".format(method))
				result = self._VERTICAPY_VARIABLES_["cursor"].fetchone()
				vector = [elem for elem in result]
			except:
				fail = 1
		if not(method in ('spearman', 'pearson', 'kendall') and (len(cols) >= 1)) or (fail): 
			vector = []
			for column in cols:
				if (column.replace('"', '').lower() == focus.replace('"', '').lower()):
					vector += [1]
				else:
					vector += [self.__aggregate_matrix__(method = method, columns = [column, focus])]
		vector = [0 if (elem == None) else elem for elem in vector]
		data = [(cols[i], vector[i]) for i in range(len(vector))]
		data.sort(key = lambda tup: abs(tup[1]), reverse = True)
		cols, vector = [elem[0] for elem in data], [elem[1] for elem in data]
		if (show):
			from verticapy.plot import cmatrix
			vmin = 0 if (method == "cramer") else -1
			if (method in ("cov")): vmin = None 
			vmax = 1 if (method in ("pearson", "spearman", "kendall", "biserial", "cramer")) else None
			if not(cmap):
				from verticapy.plot import gen_cmap
				cm1, cm2 = gen_cmap()
				cmap = cm1 if (method == "cramer") else cm2
			title = "Correlation Vector of {} ({})".format(focus, method)
			cmatrix([cols, [focus] + vector], cols, [focus], len(cols), 1, vmax = vmax, vmin = vmin, cmap = cmap, title = title, mround = round_nb)
		for idx, column in enumerate(cols):
			self.__update_catalog__(values = {focus: vector[idx]}, matrix = method, column = column)
			self.__update_catalog__(values = {column: vector[idx]}, matrix = method, column = focus)
		return tablesample(values = {"index" : cols, focus : vector}, table_info = False)
	#---#
	def __executeSQL__(self, 
				   	   query: str, 
				   	   title: str = ""):
		"""
	---------------------------------------------------------------------------
	Executes the input SQL Query.

	Parameters
 	----------
 	query: str
 		Input query.
 	title: str, optional
 		Query title. It is the tip used to indicate the query meaning when
 		turning on the SQL using to the 'sql_on_off' method. 

 	Returns
 	-------
 	DBcursor
 		The DB cursor.

 	See Also
 	--------
 	vDataFrame.sql_on_off : Displays each query generated by the vDataFrame.
		"""
		check_types([
			("query", query, [str], False), 
			("title", title, [str], False)])
		if (self._VERTICAPY_VARIABLES_["query_on"]):
			print_query(query, title)
		start_time = time.time()
		self._VERTICAPY_VARIABLES_["cursor"].execute(query)
		elapsed_time = time.time() - start_time
		if (self._VERTICAPY_VARIABLES_["time_on"]):
			print_time(elapsed_time)
		return (self._VERTICAPY_VARIABLES_["cursor"])	
	#---#
	def __genSQL__(self, 
			   	   split: bool = False, 
			   	   transformations: dict = {}, 
			   	   force_columns: list = []):
		"""
	---------------------------------------------------------------------------
	Method used to generate the SQL final relation. It will look at all the 
	transformations to build a nested query where each transformation will 
	be associated to a specific floor.

	Parameters
 	----------
 	split: bool, optional
 		Adds a split column __verticapy_split__ in the relation 
 		which can be used to downsample the data.
 	transformations: dict, optional
 		Dictionary of columns and their respective transformation. It 
 		will be used to test if an expression is correct and can be 
 		added it in the final relation.
 	force_columns: list, optional
 		Columns used to generate the final relation.

	Returns
	-------
	str
		The SQL final relation.
		"""
		# The First step is to find the Max Floor
		all_imputations_grammar = []
		if not(force_columns): force_columns = self._VERTICAPY_VARIABLES_["columns"]
		for column in force_columns:
			all_imputations_grammar += [[item[0] for item in self[column].transformations]]
		for column in transformations:
			all_imputations_grammar += [transformations[column]]
		max_transformation_floor = len(max(all_imputations_grammar, key=len))
		# We complete all the virtual columns transformations which do not have enough floors
		# with the identity transformation x :-> x in order to generate the correct SQL query
		for imputations in all_imputations_grammar:
			diff = max_transformation_floor - len(imputations)
			if diff > 0:
				imputations += ["{}"] * diff
		# We find the position of all the filters in order to write them at the correct floor
		where_positions = [item[1] for item in self._VERTICAPY_VARIABLES_["where"]]
		max_where_pos = max(where_positions + [0])
		all_where = [[] for item in range(max_where_pos + 1)]
		for i in range(0, len(self._VERTICAPY_VARIABLES_["where"])):
			all_where[where_positions[i]] += [self._VERTICAPY_VARIABLES_["where"][i][0]]
		all_where = [" AND ".join(["({})".format(elem) for elem in item]) for item in all_where]
		for i in range(len(all_where)):
			if (all_where[i] != ''):
				all_where[i] = " WHERE {}".format(all_where[i])
		# We compute the first floor
		columns = force_columns + [column for column in transformations]
		first_values = [item[0] for item in all_imputations_grammar]
		transformations_first_floor = False
		for i in range(0, len(first_values)):
			if (first_values[i] != "___VERTICAPY_UNDEFINED___") and (first_values[i] != columns[i]):
				first_values[i] = "{} AS {}".format(first_values[i], columns[i])
				transformations_first_floor = True
		if ((transformations_first_floor) or (self._VERTICAPY_VARIABLES_["allcols_ind"] != len(first_values))):
			table = "SELECT {} FROM {}".format(", ".join(first_values), self._VERTICAPY_VARIABLES_["main_relation"])
		else:
			table = "SELECT * FROM {}".format(self._VERTICAPY_VARIABLES_["main_relation"])
		# We compute the other floors
		for i in range(1, max_transformation_floor):
			values = [item[i] for item in all_imputations_grammar]
			for j in range(0, len(values)):
				if (values[j] == "{}"):
					values[j] = columns[j]
				elif (values[j] != "___VERTICAPY_UNDEFINED___"):
					values[j] = "{} AS {}".format(values[j].replace("{}", columns[j]), columns[j])
			table = "SELECT {} FROM ({}) VERTICAPY_SUBTABLE".format(", ".join(values), table)
			if (len(all_where) > i - 1):
				table += all_where[i - 1]
			if ((i - 1) in self._VERTICAPY_VARIABLES_["order_by"]):
				table += self._VERTICAPY_VARIABLES_["order_by"][i - 1]
		where_final = all_where[max_transformation_floor - 1] if (len(all_where) > max_transformation_floor - 1) else ""
		# Only the last order_by matters as the order_by will never change the final relation
		try:
			order_final = self._VERTICAPY_VARIABLES_["order_by"][max_transformation_floor - 1]
		except:
			order_final = ""
		for vml_undefined in [", ___VERTICAPY_UNDEFINED___", 
							  "___VERTICAPY_UNDEFINED___, ",
							  "___VERTICAPY_UNDEFINED___"]:
			table = table.replace(vml_undefined, "")
		split = ", RANDOM() AS __verticapy_split__" if (split) else ""
		if (where_final == "") and (order_final == ""):
			if (split):
				table = "SELECT *{} FROM ({}) VERTICAPY_SUBTABLE".format(split, table)
			table = "({}) VERTICAPY_SUBTABLE".format(table)
		else:
			table = "({}) VERTICAPY_SUBTABLE{}{}".format(table, where_final, order_final)
			table = "(SELECT *{} FROM {}) VERTICAPY_SUBTABLE".format(split, table)
		if (self._VERTICAPY_VARIABLES_["exclude_columns"]) and not(split):
			table = "(SELECT {}{} FROM {}) VERTICAPY_SUBTABLE".format(", ".join(self.get_columns()), split, table)
		main_relation = self._VERTICAPY_VARIABLES_["main_relation"]
		all_main_relation = "(SELECT * FROM {}) VERTICAPY_SUBTABLE".format(main_relation)
		table = table.replace(all_main_relation, main_relation)
		return table
	#---#
	def __get_catalog_value__(self,
						  	  column: str  = "",
						  	  key: str = "",
						  	  method: str = "",
						  	  columns: list = []):
		"""
	---------------------------------------------------------------------------
	VERTICAPY stores the already computed aggregations to avoid useless 
	computations. This method returns the stored aggregation if it was already 
	computed.
		"""
		if (column == "VERTICAPY_COUNT"):
			if (self._VERTICAPY_VARIABLES_["count"] < 0):
				return "VERTICAPY_NOT_PRECOMPUTED"
			total = self._VERTICAPY_VARIABLES_["count"]
			if (type(total) not in [int, float]):
				return "VERTICAPY_NOT_PRECOMPUTED"
			return (total)
		elif (method):
			method = str_function(method.lower())
			if (columns[1] in self[columns[0]].catalog[method]):
				return self[columns[0]].catalog[method][columns[1]]
			else:
				return "VERTICAPY_NOT_PRECOMPUTED"
		key = str_function(key.lower())
		column = vdf_columns_names([column], self)[0]
		try:
			if (key == "approx_unique") and ("unique" in self[column].catalog):
				key = "unique"
			result = "VERTICAPY_NOT_PRECOMPUTED" if key not in self[column].catalog else self[column].catalog[key]
		except:
			result = "VERTICAPY_NOT_PRECOMPUTED"
		if (result != result): result = None 
		if (("top" not in key) and (result == None)):
			return "VERTICAPY_NOT_PRECOMPUTED"
		return result
	#---#
	def __update_catalog__(self, 
					   	   values: dict = {},
					   	   erase: bool = False,
					   	   columns: list = [],
					   	   matrix: str = "",
					   	   column: str = ""):
		"""
	---------------------------------------------------------------------------
	VERTICAPY stores the already computed aggregations to avoid useless 
	computations. This method stores the input aggregation in the vcolumn catalog.
		"""
		columns = vdf_columns_names(columns, self)
		if (erase):
			if not(columns): columns = self._VERTICAPY_VARIABLES_["columns"] 
			all_cols = self.get_columns()
			for column in columns:
				self[column].catalog = {}
				for method in ["cov", "pearson", "spearman", "kendall", "cramer", "biserial", "regr_avgx", "regr_avgy", "regr_count", "regr_intercept", "regr_r2", "regr_slope", "regr_sxx", "regr_sxy", "regr_syy"]: 
					self[column].catalog[method] = {}
					for col in all_cols:
						if (column in self[col].catalog[method]):
							del self[col].catalog[method][column]
			self._VERTICAPY_VARIABLES_["count"] = -1 	
		elif (matrix):
			matrix = str_function(matrix.lower())
			if (matrix in ["cov", "pearson", "spearman", "kendall", "cramer", "biserial", "regr_avgx", "regr_avgy", "regr_count", "regr_intercept", "regr_r2", "regr_slope", "regr_sxx", "regr_sxy", "regr_syy"]):
				for elem in values:
					val = values[elem]
					try:
						val = float(val)
					except:
						pass
					self[column].catalog[matrix][elem] = val
		else:
			columns = [elem for elem in values]
			columns.remove("index")
			for column in columns:
				for i in range(len(values["index"])):
					key, val = values["index"][i].lower(), values[column][i]
					if key not in ["listagg"]:
						key = str_function(key)
						try:
							val = float(val)
							if (val - int(val) == 0):
								val = int(val)
						except:
							pass
						if (val != val):
							val = None
						self[column].catalog[key] = val
	#---#
	def __vdf_from_relation__(self, 
						  	  table: str, 
						  	  func: str, 
						  	  history: str):
		"""
	---------------------------------------------------------------------------
	This method is used to build a vDataFrame based on a relation
		"""
		check_types([
			("table", table, [str], False),
			("func", func, [str], False),
			("history", history, [str], False)])
		cursor = self._VERTICAPY_VARIABLES_["cursor"]
		dsn = self._VERTICAPY_VARIABLES_["dsn"]
		schema = self._VERTICAPY_VARIABLES_["schema"]
		schema_writing = self._VERTICAPY_VARIABLES_["schema_writing"]
		history = self._VERTICAPY_VARIABLES_["history"] + [history]
		saving = self._VERTICAPY_VARIABLES_["saving"]
		query_on = self._VERTICAPY_VARIABLES_["query_on"]
		time_on = self._VERTICAPY_VARIABLES_["time_on"]
		return (vdf_from_relation(table, func, cursor, dsn, schema, schema_writing, history, saving, query_on, time_on))
	#
	# Methods
	#
	#---# 
	def abs(self, 
			columns: list = []):
		"""
	---------------------------------------------------------------------------
	Applies the absolute value function to all the input vcolumns. 

	Parameters
 	----------
 	columns: list, optional
 		List of the vcolumns names. If empty, all the numerical vcolumns will 
 		be used.

 	Returns
 	-------
 	vDataFrame
		self

	See Also
	--------
	vDataFrame.apply    : Applies functions to the input vcolumns.
	vDataFrame.applymap : Applies a function to all the vcolumns.
		"""
		check_types([("columns", columns, [list], False)])
		columns_check(columns, self)
		columns = self.numcol() if not(columns) else vdf_columns_names(columns, self)
		func = {}
		for column in columns:
			if (self[column].ctype() != "boolean"):
				func[column] = "ABS({})"
		return (self.apply(func))
	#---#
	def agg(self, 
			func: list, 
			columns: list = []):
		"""
	---------------------------------------------------------------------------
	Aggregates the vDataFrame using the input functions.

	Parameters
 	----------
 	func: list
 		List of the different aggregation.
 			approx_unique  : approximative cardinality
 			count          : number of non-missing elements
			cvar           : conditional value at risk
			dtype          : virtual column type
			iqr            : interquartile range
			kurtosis       : kurtosis
			jb             : Jarque Bera index 
			mad            : median absolute deviation
			mae            : mean absolute error (deviation)
			max            : maximum
			mean           : average
			median         : median
			min            : minimum
			mode           : most occurent element
			percent        : percent of non-missing elements
			q%             : q quantile (ex: 50% for the median)
			prod           : product
			range          : difference between the max and the min
			sem            : standard error of the mean
			skewness       : skewness
			sum            : sum
			std            : standard deviation
			topk           : kth most occurent element (ex: top1 for the mode)
			topk_percent   : kth most occurent element density
			unique         : cardinality (count distinct)
			var            : variance
				Other aggregations could work if it is part of 
				the DB version you are using.
 	columns: list, optional
 		List of the vcolumns names. If empty, all the vcolumns 
 		or only numerical vcolumns will be used depending on the
 		aggregations.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.

 	See Also
 	--------
 	vDataFrame.analytic : Adds a new vcolumn to the vDataFrame by using an advanced 
 		analytical function on a specific vcolumn.
		"""
		return (self.aggregate(func = func, columns = columns))
	#---#
	def aggregate(self, 
				  func: list, 
				  columns: list = []):
		"""
	---------------------------------------------------------------------------
	Aggregates the vDataFrame using the input functions.

	Parameters
 	----------
 	func: list
 		List of the different aggregation.
 			approx_unique  : approximative cardinality
 			count          : number of non-missing elements
			cvar           : conditional value at risk
			dtype          : virtual column type
			iqr            : interquartile range
			kurtosis       : kurtosis
			jb             : Jarque Bera index 
			mad            : median absolute deviation
			mae            : mean absolute error (deviation)
			max            : maximum
			mean           : average
			median         : median
			min            : minimum
			mode           : most occurent element
			percent        : percent of non-missing elements
			q%             : q quantile (ex: 50% for the median)
			prod           : product
			range          : difference between the max and the min
			sem            : standard error of the mean
			skewness       : skewness
			sum            : sum
			std            : standard deviation
			topk           : kth most occurent element (ex: top1 for the mode)
			topk_percent   : kth most occurent element density
			unique         : cardinality (count distinct)
			var            : variance
				Other aggregations could work if it is part of 
				the DB version you are using.
 	columns: list, optional
 		List of the vcolumns names. If empty, all the vcolumns 
 		or only numerical vcolumns will be used depending on the
 		aggregations.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.

 	See Also
 	--------
 	vDataFrame.analytic : Adds a new vcolumn to the vDataFrame by using an advanced 
 		analytical function on a specific vcolumn.
		"""
		check_types([
			("func", func, [list], False), 
			("columns", columns, [list], False)])
		columns_check(columns, self)
		if not(columns): 
			columns = self.get_columns()
			for fun in func:
				cat_agg = ["count", "unique", "approx_unique", "approximate_count_distinct", "dtype"]
				if ("top" not in fun) and (fun not in cat_agg):
					columns = self.numcol()
					break
		else:
			columns = vdf_columns_names(columns, self)
		agg = [[] for i in range(len(columns))]
		nb_precomputed = 0
		for idx, column in enumerate(columns):
			cast = "::int" if (self[column].ctype() == "boolean") else ""
			for fun in func:
				pre_comp = self.__get_catalog_value__(column, fun)
				if (pre_comp != "VERTICAPY_NOT_PRECOMPUTED"):
					nb_precomputed += 1 
					expr = "'{}'".format(str(pre_comp).replace("'", "''")) if (pre_comp != None) else "NULL"
				elif ("_percent" in fun.lower()) and (fun.lower()[0:3] == "top"):
					n = fun.lower().replace("top", "").replace("_percent", "")
					if (n == ""): n = 1
					try:
						n = int(n)
						if (n < 1): raise
					except:
						raise ValueError("The aggregation '{}' doesn't exist. If you want to compute the frequence of the nth most occurent element please write 'topn_percent' with n > 0. Example: top2_percent for the frequency of the second most occurent element.".format(fun))
					try:
						expr = str(self[column].topk(k = n, dropna = False).values["percent"][n - 1])
					except:
						expr = "0.0"
				elif (len(fun.lower()) > 2) and (fun.lower()[0:3] == "top"):
					n = fun.lower()[3:] if (len(fun.lower()) > 3) else 1
					try:
						n = int(n)
						if (n < 1): raise
					except:
						raise ValueError("The aggregation '{}' doesn't exist. If you want to compute the nth most occurent element please write 'topn' with n > 0. Example: top2 for the second most occurent element.".format(fun))
					expr = "'{}'".format(self[column].mode(n = n))
				elif (fun.lower() in ("mode")):
					expr = "'{}'".format(self[column].mode(n = 1))
				elif (fun.lower() in ("kurtosis", "kurt")):
					count, avg, std = self.aggregate(func = ["count", "avg", "stddev"], columns = columns).transpose().values[column]
					if (count == 0 or (std != std) or (avg != avg) or (std == None) or (avg == None)): 
						expr = "NULL"
					elif ((count == 1) or (std == 0)): 
						expr = "-3"
					else:
						expr = "AVG(POWER(({}{} - {}) / {}, 4))".format(column, cast, avg, std) 
						if (count > 3):
							expr += "* {} - 3 * {}".format(count * count * (count + 1) / (count - 1) / (count - 2) / (count - 3), (count - 1) * (count - 1) / (count - 2) / (count - 3)) 
						else:
							expr += "* - 3"
							expr += "* {}".format(count * count / (count - 1) / (count - 2)) if (count == 3) else ""
				elif (fun.lower() in ("skewness", "skew")):
					count, avg, std = self.aggregate(func = ["count", "avg", "stddev"], columns = columns).transpose().values[column]
					if (count == 0 or (std != std) or (avg != avg) or (std == None) or (avg == None)): 
						expr = "NULL"
					elif ((count == 1) or (std == 0)): 
						expr = "0"
					else:
						expr = "AVG(POWER(({}{} - {}) / {}, 3))".format(column, cast, avg, std) 
						if (count >= 3):
							expr += "* {}".format(count * count / (count - 1) / (count - 2))
				elif (fun.lower() in ("jb")):
					count, avg, std = self.aggregate(func = ["count", "avg", "stddev"], columns = columns).transpose().values[column]
					if (count < 4) or (std == 0): 
						expr = "NULL"
					else:
						expr = "{} / 6 * (POWER(AVG(POWER(({}{} - {}) / {}, 3)) * {}, 2) + POWER(AVG(POWER(({}{} - {}) / {}, 4)) - 3 * {}, 2) / 4)".format(
							count, column, cast, avg, std, count * count / (count - 1) / (count - 2), column, cast, avg, std, count * count / (count - 1) / (count - 2))
				elif (fun.lower() == "dtype"):
					expr = "'{}'".format(self[column].ctype())
				elif (fun.lower() == "range"):
					expr = "MAX({}{}) - MIN({}{})".format(column, cast, column, cast)
				elif (fun.lower() == "unique"):
					expr = "COUNT(DISTINCT {})".format(column)
				elif (fun.lower() in ("approx_unique", "approximate_count_distinct")):
					expr = "APPROXIMATE_COUNT_DISTINCT({})".format(column)
				elif (fun.lower() == "count"):
					expr = "COUNT({})".format(column)
				elif (fun.lower() == "median"):
					expr = "APPROXIMATE_MEDIAN({}{})".format(column, cast)
				elif (fun.lower() in ("std", "stddev", "stdev")):
					expr = "STDDEV({}{})".format(column, cast)
				elif (fun.lower() in ("var", "variance")):
					expr = "VARIANCE({}{})".format(column, cast)
				elif (fun.lower() in ("mean", "avg")):
					expr = "AVG({}{})".format(column, cast)
				elif (fun.lower() == "iqr"):
					expr = "APPROXIMATE_PERCENTILE({}{} USING PARAMETERS percentile = 0.75) - APPROXIMATE_PERCENTILE({}{} USING PARAMETERS percentile = 0.25)".format(column, cast, column, cast)
				elif ('%' == fun[-1]):
					try:
						expr = "APPROXIMATE_PERCENTILE({}{} USING PARAMETERS percentile = {})".format(column, cast, float(fun[0:-1]) / 100)
					except:
						raise ValueError("The aggregation '{}' doesn't exist. If you want to compute the percentile x of the element please write 'x%' with x > 0. Example: 50% for the median.".format(fun))
				elif (fun.lower() == "cvar"):
					q95 = self[column].quantile(0.95)
					expr = "AVG(CASE WHEN {}{} > {} THEN {}{} ELSE NULL END)".format(column, cast, q95, column, cast)
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
			if (nb_precomputed == len(func) * len(columns)):
				self._VERTICAPY_VARIABLES_["cursor"].execute("SELECT {}".format(', '.join([item for sublist in agg for item in sublist])))
			else:
				self.__executeSQL__("SELECT {} FROM {} LIMIT 1".format(', '.join([item for sublist in agg for item in sublist]), self.__genSQL__()), title = "Computes the different aggregations.")
			result = [item for item in self._VERTICAPY_VARIABLES_["cursor"].fetchone()]
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
			try:
				query = ["SELECT {} FROM vdf_table LIMIT 1".format(', '.join(elem)) for elem in agg]
				query = " UNION ALL ".join(["({})".format(elem) for elem in query]) if (len(query) != 1) else query[0]
				query = "WITH vdf_table AS (SELECT * FROM {}) {}".format(self.__genSQL__(), query)
				if (nb_precomputed == len(func) * len(columns)):
					self._VERTICAPY_VARIABLES_["cursor"].execute(query)
				else:
					self.__executeSQL__(query, title = "Computes the different aggregations using UNION ALL.")
				result = self._VERTICAPY_VARIABLES_["cursor"].fetchall()
				for idx, elem in enumerate(result):
					values[columns[idx]] = [item for item in elem]
			except:
				try:
					for i, elem in enumerate(agg):
						pre_comp_val = []
						for fun in func:
							pre_comp = self.__get_catalog_value__(columns[i], fun)
							if (pre_comp == "VERTICAPY_NOT_PRECOMPUTED"):
								query = "SELECT {} FROM {}".format(", ".join(elem), self.__genSQL__())
								self.__executeSQL__(query, title = "Computes the different aggregations one vcolumn at a time.")
								pre_comp_val = []
								break
							pre_comp_val += [pre_comp]
						if (pre_comp_val):
							values[columns[i]] = pre_comp_val
						else:
							values[columns[i]] = [elem for elem in self._VERTICAPY_VARIABLES_["cursor"].fetchone()]
				except:
					for i, elem in enumerate(agg):
						values[columns[i]] = []
						for j, agg_fun in enumerate(elem):
							pre_comp = self.__get_catalog_value__(columns[i], func[j])
							if (pre_comp == "VERTICAPY_NOT_PRECOMPUTED"):
								query = "SELECT {} FROM {}".format(agg_fun, self.__genSQL__())
								self.__executeSQL__(query, title = "Computes the different aggregations one vcolumn at a time.")
								result = self._VERTICAPY_VARIABLES_["cursor"].fetchone()[0]
							else:
								result = pre_comp
							values[columns[i]] += [result]
		self.__update_catalog__(values)
		return (tablesample(values = values, table_info = False).transpose())
	#---#
	def all(self, 
			columns: list):
		"""
	---------------------------------------------------------------------------
	Aggregates the vDataFrame using 'bool_and'.

	Parameters
 	----------
 	columns: list
 		List of the vcolumns names.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
		return (self.aggregate(func = ["bool_and"], columns = columns))
	#---#
	def analytic(self,
				 func: str, 
				 column: str = "",
				 by: list = [], 
				 order_by = [], 
				 column2: str = "", 
				 name: str = "",
				 offset: int = 1,
				 x_smoothing: float = 0.5,
				 add_count: bool = True):
		"""
	---------------------------------------------------------------------------
	Adds a new vcolumn to the vDataFrame by using an advanced analytical function 
	on one or two specific vcolumns.

	Parameters
 	----------
 	func: str
 		Function to apply.
 			beta         : Beta Coefficient between 2 vcolumns
 			count        : number of non-missing elements
 			corr         : Pearson correlation between 2 vcolumns
 			cov          : covariance between 2 vcolumns
			dense_rank   : dense rank
			ema          : exponential moving average
			first_value  : first non null lead
			iqr          : interquartile range
			kurtosis     : kurtosis
			jb           : Jarque Bera index 
			lead         : next element
			lag          : previous element
			last_value   : first non null lag
			mad          : median absolute deviation
			mae          : mean absolute error (deviation)
			max          : maximum
			mean         : average
			median       : median
			min          : minimum
			mode         : most occurent element
			q%           : q quantile (ex: 50% for the median)
			pct_change   : ratio between the current value and the previous one
			percent_rank : percent rank
			prod         : product
			range        : difference between the max and the min
			rank         : rank
			row_number   : row number
			sem          : standard error of the mean
			skewness     : skewness
			sum          : sum
			std          : standard deviation
			unique       : cardinality (count distinct)
			var          : variance
				Other analytical functions could work if it is part of 
				the DB version you are using.
 	column: str, optional
 		Input vcolumn.
 	by: list, optional
 		vcolumns used in the partition.
 	order_by: dict / list, optional
 		List of the vcolumns used to sort the data using asc order or
 		dictionary of all the sorting methods. For example, to sort by "column1"
 		ASC and "column2" DESC, write {"column1": "asc", "column2": "desc"}
 	column2: str, optional
		Second input vcolumn in case of functions using 2 parameters.
 	name: str, optional
 		Name of the new vcolumn. If empty a default name based on the other
 		parameters will be generated.
 	offset: int, optional
 		Lead/Lag offset if parameter 'func' is the function 'lead'/'lag'.
 	x_smoothing: float, optional
 		The smoothing parameter of the 'ema' if the function is 'ema'. It must be in [0;1]
 	add_count: bool, optional
 		If the function is the 'mode' and this parameter is True then another column will 
 		be added to the vDataFrame with the mode number of occurences.

 	Returns
 	-------
 	vDataFrame
		self

	See Also
	--------
	vDataFrame.eval    : Evaluates a customized expression.
	vDataFrame.rolling : Computes a customized moving window.
		"""
		check_types([
			("func", func, [str], False),
			("by", by, [list], False), 
			("name", name, [str], False),
			("order_by", order_by, [list, dict], False),
			("column", column, [str], False),
			("add_count", add_count, [bool], False),
			("offset", offset, [int , float], False),
			("x_smoothing", x_smoothing, [int, float], False)])
		columns_check([elem for elem in order_by] + by, self)
		if (column):
			columns_check([column], self)
			column = vdf_columns_names([column], self)[0]
		if (column2):
			columns_check([column2], self)
			column2 = vdf_columns_names([column2], self)[0]
		by_name = ["by"] + by if (by) else []
		by_order = ["order_by"] + [elem for elem in order_by] if (order_by) else []
		if not(name):
			name = gen_name([func, column, column2] + by_name + by_order)
		by_order_name = gen_name(by_name + by_order)
		by = vdf_columns_names(by, self)
		func = func.lower()
		by = ", ".join(by)
		by = "PARTITION BY {}".format(by) if (by) else ""
		order_by = sort_str(order_by, self)
		func = str_function(func.lower())
		if (func in ("max", "min", "avg", "sum", "count", "std", "median", "var", "unique", "top", "kurtosis", "skewness", "mad", "mae", "range", "prod", "jb") or ("%" in func)):
			if (order_by):
				print("\u26A0 '{}' analytic function doesn't need an order by clause, it was ignored".format(func))
			elif not(column):
				raise ValueError("The parameter 'column' must be a vDataFrame Column when using analytic function '{}'".format(func))
			if (func in ("skewness", "kurtosis", "mae", "mad", "jb")):
				mean_name = "{}_mean{}".format(column.replace('"', ''), by_order_name)
				median_name = "{}_median{}".format(column.replace('"', ''), by_order_name)
				std_name = "{}_std{}".format(column.replace('"', ''), by_order_name)
				count_name = "{}_count{}".format(column.replace('"', ''), by_order_name)
				all_cols = self.get_columns()
				if (str_column(median_name) not in all_cols) and (func == "mad"):
					self.eval(median_name, "MEDIAN({}) OVER ({})".format(column, by))
				if (str_column(mean_name) not in all_cols) and (func != "mad"):
					self.eval(mean_name, "AVG({}) OVER ({})".format(column, by))
				if (str_column(std_name) not in all_cols) and (func not in ("mae", "mad")):
					self.eval(std_name, "STDDEV({}) OVER ({})".format(column, by))
				if (str_column(count_name) not in all_cols) and (func not in ("mae", "mad")):
					self.eval(count_name, "COUNT({}) OVER ({})".format(column, by))
				if (func == "kurtosis"):
					self.eval("{}_kurtosis".format(name), "AVG(POWER(({} - {}) / NULLIFZERO({}), 4)) OVER ({}) * POWER({}, 2) * ({} + 1) / NULLIFZERO(({} - 1) * ({} - 2) * ({} - 3)) - 3 * POWER({} - 1, 2) / NULLIFZERO(({} - 2) * ({} - 3))".format(
						column, mean_name, std_name, by, count_name, count_name, count_name, count_name, count_name, count_name, count_name, count_name))
				elif (func == "skewness"):
					self.eval("{}_skewness".format(name), "AVG(POWER(({} - {}) / NULLIFZERO({}), 3)) OVER ({}) * POWER({}, 2) / NULLIFZERO(({} - 1) * ({} - 2))".format(
						column, mean_name, std_name, by, count_name, count_name, count_name))
				elif (func == "jb"):
					self.eval("{}_jb".format(name), "{} / 6 * (POWER(AVG(POWER(({} - {}) / NULLIFZERO({}), 3)) OVER ({}) * POWER({}, 2) / NULLIFZERO(({} - 1) * ({} - 2)), 2) + POWER(AVG(POWER(({} - {}) / NULLIFZERO({}), 4)) OVER ({}) * POWER({}, 2) * ({} + 1) / NULLIFZERO(({} - 1) * ({} - 2) * ({} - 3)) - 3 * POWER({} - 1, 2) / NULLIFZERO(({} - 2) * ({} - 3)), 2) / 4)".format(
						count_name, column, mean_name, std_name, by, count_name, count_name, count_name, column, mean_name, std_name, by, count_name, count_name, count_name, count_name, count_name, count_name, count_name, count_name))
				elif (func == "mae"):
					self.eval("{}_mae".format(name), "AVG(ABS({} - {})) OVER ({})".format(column, mean_name, by))
				elif (func == "mad"):
					self.eval("{}_mad".format(name), "AVG(ABS({} - {})) OVER ({})".format(column, median_name, by))
			elif (func == "top"):
				self.eval(name, "ROW_NUMBER() OVER ({})".format("PARTITION BY {}".format(column) if not(by) else "{}, {}".format(by, column)))
				if (add_count):
					self.eval("{}_count".format(name.replace('"', '')), "NTH_VALUE({}, 1) OVER ({} ORDER BY {} DESC)".format(name, by, name))
				self[name].apply("NTH_VALUE({}, 1) OVER ({} ORDER BY {} DESC)".format(column, by, "{}"))
			elif (func == "unique"):
				self.eval(name, "DENSE_RANK() OVER ({} ORDER BY {} ASC) + DENSE_RANK() OVER ({} ORDER BY {} DESC) - 1".format(by, column, by, column))
			elif ('%' == func[-1]):
				try:
					x = float(func[0:-1]) / 100
				except:
					raise ValueError("The aggregate function '{}' doesn't exist. If you want to compute the percentile x of the element please write 'x%' with x > 0. Example: 50% for the median.".format(func))
				self.eval(name, "PERCENTILE_CONT({}) WITHIN GROUP(ORDER BY {}) OVER ({})".format(x, column, by))
			elif (func == "range"):
				self.eval(name, "MAX({}) OVER ({}) - MIN({}) OVER ({})".format(column, by, column, by))
			elif (func == "prod"):
				self.eval(name, "DECODE(ABS(MOD(SUM(CASE WHEN {} < 0 THEN 1 ELSE 0 END) OVER ({}), 2)), 0, 1, -1) * POWER(10, SUM(LOG(ABS({}))) OVER ({}))".format(column, by, column, by))
			else:
				self.eval(name, "{}({}) OVER ({})".format(func.upper(), column, by))
		elif func in ("lead", "lag", "row_number", "percent_rank", "dense_rank", "rank", "first_value", "last_value", "exponential_moving_average", "pct_change"):
			if (not(column) and func in ("lead", "lag", "first_value", "last_value", "pct_change")):
				raise ValueError("The parameter 'column' must be a vDataFrame Column when using analytic function '{}'".format(func))
			elif ((column) and func not in ("lead", "lag", "first_value", "last_value", "pct_change")):
				raise ValueError("The parameter 'column' must be empty when using analytic function '{}'".format(func))
			if (by) and (order_by):
				order_by = " {}".format(order_by)
			if (func in ("lead", "lag")):
				info_param = ", {}".format(offset)
			elif (func in ("last_value", "first_value")):
				info_param = " IGNORE NULLS"
			elif (func == "exponential_moving_average"):
				info_param = ", {}".format(x_smoothing)
			else:
				info_param = ""
			if (func == "pct_change"):
				self.eval(name, "{} / (LAG({}) OVER ({}{}))".format(column, column, by, order_by))
			else:
				self.eval(name, "{}({}{}) OVER ({}{})".format(func.upper(), column, info_param, by, order_by))
		elif func in ("corr", "cov", "beta"):
			if (order_by):
				print("\u26A0 '{}' analytic function doesn't need an order by clause, it was ignored".format(func))
			if not(column):
				raise ValueError("The parameter 'column' must be a vcolumn when using analytic function '{}'".format(func))
			elif not(column2):
				raise ValueError("The parameter 'column2' must be a vcolumn when using analytic function '{}'".format(func))
			all_cols = self.get_columns()
			mean_name  = "{}_mean{}".format(column.replace('"', ''), by_order_name)
			if (str_column(mean_name) not in all_cols):
				self.eval(mean_name, "AVG({}) OVER ({})".format(column, by))
			mean_name2 = "{}_mean{}".format(column2.replace('"', ''), by_order_name)
			if (str_column(mean_name2) not in all_cols):
				self.eval(mean_name2, "AVG({}) OVER ({})".format(column, by))
			if (func == "corr"):
				den = " / (STDDEV({}) OVER ({}) * STDDEV({}) OVER ({}))".format(column, by, column2, by)
			elif (func == "beta"):
				den = " / (VARIANCE({}) OVER ({}))".format(column2, by)
			else:
				den = ""
			self.eval(name, "AVG(({} - {}) * ({} - {})) OVER ({}){}".format(column, mean_name, column2, mean_name2, by, den))
		else:
			try:
				self.eval(name, "{}({}) OVER ({}{})".format(func.upper(), column, info_param, by, order_by))
			except:
				raise ValueError("The aggregate function '{}' doesn't exist or is not managed by the analytic function. If you want more flexibility use the 'eval' function".format(func))
		return (self)
	#---#
	def any(self, 
			columns: list):
		"""
	---------------------------------------------------------------------------
	Aggregates the vDataFrame using 'bool_or'.

	Parameters
 	----------
 	columns: list
 		List of the vcolumns names.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
		return (self.aggregate(func = ["bool_or"], columns = columns))
	#---#
	def append(self, 
			   input_relation,
			   expr1: list = [],
			   expr2: list = [],
			   union_all: bool = True):
		"""
	---------------------------------------------------------------------------
	Merges the vDataFrame with another one or an input relation and returns 
	a new vDataFrame.

	Parameters
 	----------
 	input_relation: str / vDataFrame
 		Relation used to do the merging.
 	expr1: list, optional
		List of expressions from the current vDataFrame to use during the merging.
		It must be pure SQL. For example, 'CASE WHEN "column" > 3 THEN 2 ELSE NULL END' 
		and 'POWER("column", 2)' will work. If empty all the vDataFrame vcolumns will
		be used. It is highly recommended to write aliases to avoid auto-naming.
	expr2: list, optional
		List of expressions from the input relation to use during the merging.
		It must be pure SQL. For example, 'CASE WHEN "column" > 3 THEN 2 ELSE NULL END' 
		and 'POWER("column", 2)' will work. If empty all the input relation columns will
		be used. It is highly recommended to write aliases to avoid auto-naming.
	union_all: bool, optional
		If set to True, the vDataFrame will be merged with the input relation using an
		'UNION ALL' instead of an 'UNION'.

 	Returns
 	-------
 	vDataFrame
		vDataFrame of the Union

	See Also
	--------
	vDataFrame.groupby : Aggregates the vDataFrame.
	vDataFrame.join    : Joins the vDataFrame with another relation.
	vDataFrame.sort    : Sorts the vDataFrame.
		"""
		check_types([
			("expr1", expr1, [list], False),
			("expr2", expr2, [list], False),
			("union_all", union_all, [bool], False)])
		first_relation = self.__genSQL__()
		if (type(input_relation) == str):
			second_relation = input_relation
		elif (type(input_relation) == type(self)):
			second_relation = input_relation.__genSQL__()
		else:
			raise TypeError("Parameter 'input_relation' type must be one of the following [{}, {}], found {}".format(str, type(self), type(input_relation)))
		columns  = ", ".join(self.get_columns()) if not(expr1) else ", ".join(expr1)
		columns2 = columns if not(expr2) else ", ".join(expr2)
		table = "(SELECT {} FROM {}) UNION ALL (SELECT {} FROM {})".format(columns, first_relation, columns2, second_relation)
		query = "SELECT * FROM ({}) append_table".format(table)
		self.__executeSQL__(query = query, title = "Merges the two relations.")
		return (self.__vdf_from_relation__("({}) append_table".format(table), self._VERTICAPY_VARIABLES_["input_relation"], "[Append]: Union of two relations"))
	#---#
	def apply(self, 
			  func: dict):
		"""
	---------------------------------------------------------------------------
	Applies each function of the dictionary to the input vcolumns.

	Parameters
 	----------
 	func: dict
 		Dictionary of functions.
 		The dictionary must be like the following: 
		{column1: func1, ..., columnk: funck}. Each function variable must
		be composed of two flower brackets {}. For example to apply the 
		function: x -> x^2 + 2 use "POWER({}, 2) + 2".

 	Returns
 	-------
 	vDataFrame
		self

	See Also
	--------
	vDataFrame.applymap : Applies a function to all the vcolumns.
	vDataFrame.eval     : Evaluates a customized expression.
		"""
		check_types([("func", func, [dict], False)])
		columns_check([elem for elem in func], self)
		for column in func:
			self[vdf_columns_names([column], self)[0]].apply(func[column])
		return (self)
	#---#
	def applymap(self, 
				 func: str, 
				 numeric_only: bool = True):
		"""
	---------------------------------------------------------------------------
	Applies a function to all the vcolumns. 

	Parameters
 	----------
 	func: str
 		The function.
 		The function variable must be composed of two flower brackets {}. 
 		For example to apply the function: x -> x^2 + 2 use "POWER({}, 2) + 2".
 	numeric_only: bool, optional
 		If set to True, only the numerical columns will be used.

 	Returns
 	-------
 	vDataFrame
		self

	See Also
	--------
	vDataFrame.apply : Applies functions to the input vcolumns.
		"""
		check_types([
			("func", func, [str], False), 
			("numeric_only", numeric_only, [bool], False)])
		function = {}
		columns = self.numcol() if numeric_only else self.get_columns()
		for column in columns:
			function[column] = func if (self[column].ctype() != "boolean") else func.replace("{}", "{}::int")
		return (self.apply(function))
	#---#
	def asfreq(self,
			   ts: str,
			   rule: str,
			   method: dict,
			   by: list = []):
		"""
	---------------------------------------------------------------------------
	Computes a regular time interval vDataFrame by interpolating the missing 
	values using different techniques.

	Parameters
 	----------
 	ts: str
 		TS (Time Series) vcolumn used to order the data. The vcolumn type must be
 		date like (date, datetime, timestamp...)
 	rule: str
 		Interval used to slice the time. For example, '5 minutes' will create records
 		separated by '5 minutes' time interval.
 	method: dict
 		Dictionary of all the different methods of interpolation. The dict must be 
 		similar to the following:
 		{"column1": "interpolation1" ..., "columnk": "interpolationk"}
 		3 types of interpolations are possible:
 			bfill  : Constant propagation of the next value (Back Propagation).
 			ffill  : Constant propagation of the first value (First Propagation).
 			linear : Linear Interpolation.
 	by: list, optional
 		vcolumns used in the partition.

 	Returns
 	-------
 	vDataFrame
 		object result of the interpolation.

	See Also
	--------
	vDataFrame[].fillna  : Fills the vcolumn missing values.
	vDataFrame[].slice   : Slices the vcolumn.
		"""
		check_types([
			("ts", ts, [str], False), 
			("rule", rule, [str], False), 
			("method", method, [dict], False), 
			("by", by, [list], False)])
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
		table = "SELECT {} FROM {}".format("{}", self.__genSQL__())
		tmp_query = ["slice_time AS {}".format(str_column(ts))]
		tmp_query += [str_column(column) for column in by]
		tmp_query += all_elements
		table = table.format(", ".join(tmp_query))
		partition = "PARTITION BY {} ".format(", ".join([str_column(column) for column in by])) if (by) else ""
		table += " TIMESERIES slice_time AS '{}' OVER ({}ORDER BY {}::timestamp)".format(rule, partition, str_column(ts))
		return (self.__vdf_from_relation__('({}) asfreq'.format(table), "asfreq", "[Asfreq]: The data was resampled"))
	#---#
	def astype(self, 
			   dtype: dict):
		"""
	---------------------------------------------------------------------------
	Converts the vcolumns to the input types.

	Parameters
 	----------
 	dtype: dict
 		Dictionary of the different types. Each key of the dictionary must 
 		represent a vcolumn. The dictionary must be similar to the 
 		following: {"column1": "type1", ... "columnk": "typek"}

 	Returns
 	-------
 	vDataFrame
		self
		"""
		check_types([("dtype", dtype, [dict], False)])
		columns_check([elem for elem in dtype], self)
		for column in dtype:
			self[vdf_columns_names([column], self)[0]].astype(dtype = dtype[column])
		return (self)
	#---#
	def at_time(self, 
				ts: str, 
				time: str):
		"""
	---------------------------------------------------------------------------
	Filters the vDataFrame by only keeping the records at the input time.

	Parameters
 	----------
 	ts: str
 		TS (Time Series) vcolumn used to filter the data. The vcolumn type must be
 		date like (date, datetime, timestamp...)
 	time: str
 		Input Time. For example, time = '12:00' will filter the data when time('ts') 
 		is equal to 12:00.

 	Returns
 	-------
 	vDataFrame
		self

	See Also
	--------
	vDataFrame.between_time : Filters the data between two time ranges.
	vDataFrame.first        : Filters the data by only keeping the first records.
	vDataFrame.filter       : Filters the data using the input expression.
	vDataFrame.last         : Filters the data by only keeping the last records.
		"""
		check_types([
			("ts", ts, [str], False), 
			("time", time, [str], False)])
		columns_check([ts], self)
		self.filter("{}::time = '{}'".format(str_column(ts), time))
		return (self)
	#---#
	def autocorr(self, 
				 ts: str, 
				 column: str, 
				 by: list = [],
				 p = 12,
				 unit: str = "rows",
				 method: str = "pearson", 
				 cmap: str = "",
			 	 round_nb: int = 3, 
			 	 focus: bool = False,
			 	 show: bool = True,
			 	 matrix: bool = False):
		"""
	---------------------------------------------------------------------------
	Computes the correlations of the input vcolumn and its lags. 

	Parameters
 	----------
 	ts: str
 		TS (Time Series) vcolumn used to order the data. It can be of type date
 		or a numerical vcolumn.
 	column: str
 		Input vcolumn used to compute the Auto Correlation Plot.
 	by: list, optional
 		vcolumns used in the partition.
 	p: int/list, optional
 		Int equals to the maximum number of lag to consider during the computation
 		or List of the different lags to include during the computation.
 		p must be positive or a list of positive integers.
 	unit: str, optional
 		Unit used to compute the lags.
 			rows: Natural lags
 			else : Any time unit, for example you can write 'hour' to compute the hours
 				lags or 'day' to compute the days lags.
 	method: str, optional
 		Method used to compute the correlation.
 			pearson   : Pearson correlation coefficient (linear).
 			spearmann : Spearmann correlation coefficient (monotonic - rank based).
 			kendall   : Kendall correlation coefficient (similar trends).
 			cramer    : Cramer's V (correlation between categories).
 			biserial  : Biserial Point (correlation between binaries and a numericals).
 	cmap: str, optional
 		Color Map.
 	round_nb: int, optional
 		Round the coefficient using the input number of digits. It is used only if parameter 'matrix' 
 		is set to True.
 	focus: bool, optional
 		Computes a correlation vector instead of the matrix. It is used only if parameter 'matrix' 
 		is set to True.
 	show: bool, optional
 		If set to True, the Auto Correlation Plot will be drawn using Matplotlib.
 	matrix: bool, optional
 		If set to True, draw a correlation matrix instead of a regular plot.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.

	See Also
	--------
	vDataFrame.asfreq : Interpolates and computes a regular time interval vDataFrame.
	vDataFrame.corr   : Computes the Correlation Matrix of a vDataFrame.
	vDataFrame.cov    : Computes the Covariance Matrix of the vDataFrame.
		"""
		check_types([
			("by", by, [list], False),
			("ts", ts, [str], False),
			("column", column, [str], False), 
			("p", p, [int, float, list], False),
			("unit", unit, [str], False),
			("matrix", matrix, [bool], False), 
			("method", method, ["pearson", "kendall", "spearman", "biserial", "cramer"], True), 
			("cmap", cmap, [str], False), 
			("round_nb", round_nb, [int, float], False), 
			("focus", focus, [bool], False), 
			("show", show, [bool], False)])
		method = method.lower()
		columns_check([column, ts] + by, self)
		by = vdf_columns_names(by, self)
		column = vdf_columns_names([column], self)[0]
		ts = vdf_columns_names([ts], self)[0]
		if unit == "rows":
			table = self.__genSQL__()
		else:
			table = self.asfreq(ts = ts, rule = "1 {}".format(unit), method = {column: "linear"}, by = by).__genSQL__()
		if (type(p) in [int, float]): p = range(1, p)
		by = "PARTITION BY {} ".format(", ".join(by)) if (by) else ""
		columns = ["LAG({}, {}) OVER ({}ORDER BY {}) AS lag_{}_{}".format(column, i, by, ts, i, gen_name([column])) for i in p]
		relation = "(SELECT {} FROM {}) autocorr".format(", ".join([column] + columns), table)
		if (matrix):
			return (self.__vdf_from_relation__(relation, "autocorr", "").corr([], method, cmap, round_nb, "" if not(focus) else column, show))
		else:
			result = self.__vdf_from_relation__(relation, "autocorr", "").corr([], method, focus = column, show = False)
			columns = [elem for elem in result.values["index"]]
			autocorr = [elem for elem in result.values[column]]
			if (columns[0] == column):
				columns[0] = 0
			for i in range(1, len(columns)):
				columns[i] = int(columns[i].split("_")[1])
			data = [(columns[i], autocorr[i]) for i in range(len(columns))]
			data.sort(key = lambda tup: tup[0])
			result.values["index"] = [elem[0] for elem in data]
			result.values[column] = [elem[1] for elem in data]
			if (show):
				from verticapy.plot import autocorr_plot
				autocorr_plot(result.values["index"], result.values[column], title = column)
			return (result)
	#---#
	def avg(self, columns: list = []):
		"""
	---------------------------------------------------------------------------
	Aggregates the vDataFrame using 'avg' (Average).

	Parameters
 	----------
 	columns: list, optional
 		List of the vcolumns names. If empty, all the numerical vcolumns will be 
 		used.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
		return (self.mean(columns = columns))
	#---# 
	def bar(self,
			columns: list,
			method: str = "density",
			of: str = "",
			max_cardinality: tuple = (6, 6),
			h: tuple = (None, None),
			hist_type: str = "auto"):
		"""
	---------------------------------------------------------------------------
	Draws the Bar Chart of the input vcolumns based on an aggregation.

	Parameters
 	----------
 	columns: list
 		List of the vcolumns names. The list must have one or two elements.
 	method: str, optional
 		The method used to aggregate the data.
 			count   : Number of elements.
 			density : Percentage of the distribution.
 			mean    : Average of the vcolumn 'of'.
 			min     : Minimum of the vcolumn 'of'.
 			max     : Maximum of the vcolumn 'of'.
 			sum     : Sum of the vcolumn 'of'.
 			q%      : q Quantile of the vcolumn 'of' (ex: 50% to get the median).
 	of: str, optional
 		The vcolumn used to compute the aggregation.
 	h: tuple, optional
 		Interval width of the vcolumns 1 and 2 bars. It is only valid if the 
 		vcolumns are numerical. Optimized h will be computed if the parameter 
 		is empty or invalid.
 	max_cardinality: tuple, optional
 		Maximum number of distinct elements for vcolumns 1 and 2 to be used as 
 		categorical (No h will be picked or computed)
 	hist_type: str, optional
 		The Histogram Type.
 			auto          : Regular Bar Chart based on 1 or 2 vcolumns.
			stacked       : Stacked Bar Chart based on 2 vcolumns.
			fully_stacked : Fully Stacked Bar Chart based on 2 vcolumns.

 	Returns
 	-------
 	vDataFrame
		self

 	See Also
 	--------
 	vDataFrame.boxplot     : Draws the Box Plot of the input vcolumns.
 	vDataFrame.hist        : Draws the Histogram of the input vcolumns based on an aggregation.
 	vDataFrame.pivot_table : Draws the Pivot Table of vcolumns based on an aggregation.
		"""
		check_types([
			("columns", columns, [list], False), 
			("method", method, [str], False),  
			("of", of, [str], False), 
			("max_cardinality", max_cardinality, [tuple], False), 
			("h", h, [tuple], False), 
			("hist_type", hist_type, ["auto", "fully_stacked", "stacked", "fully", "fully stacked"], True)])
		method, hist_type = method.lower(), hist_type.lower()
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
			from verticapy.plot import bar2D
			bar2D(self, columns, method, of, max_cardinality, h, stacked, fully_stacked)
		return (self)
	#---#
	def between_time(self,
					 ts: str, 
					 start_time: str, 
					 end_time: str):
		"""
	---------------------------------------------------------------------------
	Filters the vDataFrame by only keeping the records between two input times.

	Parameters
 	----------
 	ts: str
 		TS (Time Series) vcolumn used to filter the data. The vcolumn type must be
 		date like (date, datetime, timestamp...)
 	start_time: str
 		Input Start Time. For example, time = '12:00' will filter the data when 
 		time('ts') is lesser than 12:00.
 	end_time: str
 		Input End Time. For example, time = '14:00' will filter the data when 
 		time('ts') is greater than 14:00.

 	Returns
 	-------
 	vDataFrame
		self

	See Also
	--------
	vDataFrame.at_time : Filters the data at the input time.
	vDataFrame.first   : Filters the data by only keeping the first records.
	vDataFrame.filter  : Filters the data using the input expression.
	vDataFrame.last    : Filters the data by only keeping the last records.
		"""
		check_types([
			("ts", ts, [str], False), 
			("start_time", start_time, [str], False), 
			("end_time", end_time, [str], False)])
		columns_check([ts], self)
		self.filter("{}::time BETWEEN '{}' AND '{}'".format(str_column(ts), start_time, end_time))
		return (self)
	#---#
	def bool_to_int(self):
		"""
	---------------------------------------------------------------------------
	Converts all the booleans vcolumns to integers.

 	Returns
 	-------
 	vDataFrame
		self
	
	See Also
	--------
	vDataFrame.astype : Converts the vcolumns to the input types.
		"""
		columns = self.get_columns()
		for column in columns:
			if (self[column].ctype() == "boolean"):
				self[column].astype("int")
		return (self)
	#---#
	def boxplot(self, 
				columns: list = []):
		"""
	---------------------------------------------------------------------------
	Draws the Box Plot of the input vcolumns. 

	Parameters
 	----------
 	columns: list, optional
 		List of the vcolumns names. If empty, all the numerical vcolumns will 
 		be used.

 	Returns
 	-------
 	vDataFrame
		self

	See Also
	--------
	vDataFrame.bar         : Draws the Bar Chart of the input vcolumns based on an aggregation.
	vDataFrame.boxplot     : Draws the vcolumn Box Plot.
 	vDataFrame.hist        : Draws the Histogram of the input vcolumns based on an aggregation.
 	vDataFrame.pivot_table : Draws the Pivot Table of vcolumns based on an aggregation.
		"""
		check_types([("columns", columns, [list], False)])
		columns_check(columns, self)
		columns = vdf_columns_names(columns, self) if (columns) else self.numcol()
		from verticapy.plot import boxplot2D
		boxplot2D(self, columns)
		return (self)
	#---#
	def bubble(self,
			   columns: list,
			   size_bubble_col: str,
			   catcol: str = "",
			   max_nb_points: int = 20000,
			   bbox: list = [],
			   img: str = ""):
		"""
	---------------------------------------------------------------------------
	Draws the Bubble Plot of the input vcolumns.

	Parameters
 	----------
 	columns: list
 		List of the vcolumns names. The list must have two elements.
 	size_bubble_col: str
 		Numerical vcolumn used to represent the Bubble size.
 	catcol: str, optional
 		Categorical column used as color.
 	max_nb_points: int, optional
 		Maximum number of points to display.
 	bbox: list, optional
 		List of 4 elements to delimit the boundaries of the final Plot. 
 		It must be similar the following list: [xmin, xmax, ymin, ymax]
 	img: str, optional
 		Path to the image to display as background.

 	Returns
 	-------
 	vDataFrame
		self

 	See Also
 	--------
 	vDataFrame.scatter : Draws the Scatter Plot of the input vcolumns.
		"""
		check_types([
			("columns", columns, [list], False), 
			("size_bubble_col", size_bubble_col, [str], False),
			("max_nb_points", max_nb_points, [int, float], False),
			("bbox", bbox, [list], False),
			("img", img, [str], False)])
		columns_check(columns, self, [2])
		columns = vdf_columns_names(columns, self)
		if catcol:
			columns_check([catcol], self)
			catcol = vdf_columns_names([catcol], self)[0]
		columns_check([size_bubble_col], self)
		size_bubble_col = vdf_columns_names([size_bubble_col], self)[0]
		from verticapy.plot import bubble
		bubble(self, columns + [size_bubble_col], catcol, max_nb_points, bbox, img)
		return (self)
	#---#
	def catcol(self, 
			   max_cardinality: int = 12):
		"""
	---------------------------------------------------------------------------
	Returns the vDataFrame categorical vcolumns.
	
	Parameters
	----------
	max_cardinality: int, optional
		Maximum number of unique values to consider integer vcolumns as categorical.

 	Returns
 	-------
 	List
 		List of the categorical vcolumns names.
	
	See Also
	--------
	vDataFrame.get_columns : Returns all the vDataFrame vcolumns.
	vDataFrame.numcol      : Returns all the vDataFrame numerical vcolumns.
		"""
		check_types([("max_cardinality", max_cardinality, [int, float], False)])
		columns = []
		for column in self.get_columns():
			if ((self[column].category() == "int") and (self[column].ctype() != "boolean")):
				self._VERTICAPY_VARIABLES_["cursor"].execute("SELECT (APPROXIMATE_COUNT_DISTINCT({}) < {}) FROM {}".format(column, max_cardinality, self.__genSQL__()))
				is_cat = self._VERTICAPY_VARIABLES_["cursor"].fetchone()[0]
			elif (self[column].category() == "float"):
				is_cat = False
			else:
				is_cat = True
			if (is_cat):
				columns += [column]
		return (columns)
	#---#
	def copy(self):
		"""
	---------------------------------------------------------------------------
	Returns a copy of the vDataFrame.

 	Returns
 	-------
 	vDataFrame
 		The copy of the vDataFrame.
		"""
		copy_vDataFrame = vDataFrame("", empty = True)
		copy_vDataFrame._VERTICAPY_VARIABLES_["dsn"] = self._VERTICAPY_VARIABLES_["dsn"]
		copy_vDataFrame._VERTICAPY_VARIABLES_["input_relation"] = self._VERTICAPY_VARIABLES_["input_relation"]
		copy_vDataFrame._VERTICAPY_VARIABLES_["main_relation"] = self._VERTICAPY_VARIABLES_["main_relation"]
		copy_vDataFrame._VERTICAPY_VARIABLES_["schema"] = self._VERTICAPY_VARIABLES_["schema"]
		copy_vDataFrame._VERTICAPY_VARIABLES_["cursor"] = self._VERTICAPY_VARIABLES_["cursor"]
		copy_vDataFrame._VERTICAPY_VARIABLES_["columns"] = [item for item in self._VERTICAPY_VARIABLES_["columns"]]
		copy_vDataFrame._VERTICAPY_VARIABLES_["where"] = [item for item in self._VERTICAPY_VARIABLES_["where"]]
		copy_vDataFrame._VERTICAPY_VARIABLES_["order_by"] = {}
		for item in self._VERTICAPY_VARIABLES_["order_by"]:
			copy_vDataFrame._VERTICAPY_VARIABLES_["order_by"][item] = self._VERTICAPY_VARIABLES_["order_by"][item]
		copy_vDataFrame._VERTICAPY_VARIABLES_["exclude_columns"] = [item for item in self._VERTICAPY_VARIABLES_["exclude_columns"]]
		copy_vDataFrame._VERTICAPY_VARIABLES_["history"] = [item for item in self._VERTICAPY_VARIABLES_["history"]]
		copy_vDataFrame._VERTICAPY_VARIABLES_["saving"] = [item for item in self._VERTICAPY_VARIABLES_["saving"]]
		copy_vDataFrame._VERTICAPY_VARIABLES_["query_on"] = self._VERTICAPY_VARIABLES_["query_on"]
		copy_vDataFrame._VERTICAPY_VARIABLES_["time_on"] = self._VERTICAPY_VARIABLES_["time_on"]
		copy_vDataFrame._VERTICAPY_VARIABLES_["schema_writing"] = self._VERTICAPY_VARIABLES_["schema_writing"]
		for column in self._VERTICAPY_VARIABLES_["columns"]:
			new_vColumn = vColumn(column, parent = copy_vDataFrame, transformations = [elem for elem in self[column].transformations], catalog = {})
			setattr(copy_vDataFrame, column, new_vColumn)
			setattr(copy_vDataFrame, column[1:-1], new_vColumn)
		return (copy_vDataFrame)
	#---#
	def case_when(self, 
				  name: str, 
				  conditions: dict, 
				  others = None,
				  auto_quote: bool = True):
		"""
	---------------------------------------------------------------------------
	Creates a new feature by evaluating some conditions.
	
	Parameters
	----------
	name: str
		Name of the new feature.
	conditions: dict
		Dictionary of the conditions. The dictionary must be similar to the 
		following: {"condition1": "value1", ... "conditionk": "valuek"}
		Each key is a condition and each value is the value returned when
		the condition is valid.
	others: str / int / float, optional
		last condition: The ELSE in the CASE WHEN statement.
	auto_quote: bool, optional
		If set to True, all the str values of the dictionary will not be considered
		as expressions. Otherwise, you'll have to manage by yourself all the quotations.

 	Returns
 	-------
 	vDataFrame
 		self
	
	See Also
	--------
	vDataFrame[].decode : Encodes the vcolumn using a User Defined Encoding.
	vDataFrame.eval : Evaluates a customized expression.
		"""
		check_types([
			("conditions", conditions, [dict], False), 
			("name", name, [str], False)])
		for elem in conditions:
			if (type(conditions[elem]) == str) and auto_quote:
				val = conditions[elem]
				conditions[elem] = "'{}'".format(val.replace("'", "''"))
			elif (conditions[elem] == None):
				conditions[elem] = 'NULL'
		if (type(others) == str) and auto_quote:
			others = "'{}'".format(others.replace("'", "''"))
		elif (others == None):
			others = 'NULL'
		expr = " ".join(["WHEN {} THEN {}".format(elem, conditions[elem]) for elem in conditions])
		expr = "(CASE {} ELSE {} END)".format(expr, others)
		return (self.eval(name = name, expr = expr))
	#---#
	def corr(self, 
			 columns: list = [],
			 method: str = "pearson", 
			 cmap: str = "",
			 round_nb: int = 3, 
			 focus: str = "",
			 show: bool = True):
		"""
	---------------------------------------------------------------------------
	Computes the Correlation Matrix of the vDataFrame. 

	Parameters
 	----------
 	columns: list, optional
 		List of the vcolumns names. If empty, all the numerical vcolumns will be 
 		used.
 	method: str, optional
 		Method used to compute the correlation.
 			pearson   : Pearson correlation coefficient (linear).
 			spearmann : Spearmann correlation coefficient (monotonic - rank based).
 			kendall   : Kendall correlation coefficient (similar trends).
 			cramer    : Cramer's V (correlation between categories).
 			biserial  : Biserial Point (correlation between binaries and a numericals).
 	cmap: str, optional
 		Color Map.
 	round_nb: int, optional
 		Rounds the coefficient using the input number of digits.
 	focus: str, optional
 		Focus the computation on only one vcolumn.
 	show: bool, optional
 		If set to True, the Correlation Matrix will be drawn using Matplotlib.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.

	See Also
	--------
	vDataFrame.autocorr : Computes the Correlations between a vcolumn and its lags.
	vDataFrame.cov      : Computes the Covariance Matrix of the vDataFrame.
	vDataFrame.regr     : Computes the Regression Matrix of the vDataFrame. 
		"""
		check_types([
			("columns", columns, [list], False), 
			("method", method, ["pearson", "kendall", "spearman", "biserial", "cramer"], True), 
			("cmap", cmap, [str], False), 
			("round_nb", round_nb, [int, float], False), 
			("focus", focus, [str], False), 
			("show", show, [bool], False)])
		method = method.lower()
		columns_check(columns, self)
		columns = vdf_columns_names(columns, self)
		if (focus == ""):
			return (self.__aggregate_matrix__(method = method, columns = columns, cmap = cmap, round_nb = round_nb, show = show))
		else:
			columns_check([focus], self)
			focus = vdf_columns_names([focus], self)[0]
			return (self.__aggregate_vector__(focus, method = method, columns = columns, cmap = cmap, round_nb = round_nb, show = show))
	#---#
	def count(self, 
			  columns: list = [], 
			  percent: bool = True,
			  sort_result: bool = True,
			  desc: bool = True):
		"""
	---------------------------------------------------------------------------
	Aggregates the vDataFrame using a list of 'count' (Number of missing values).

	Parameters
 	----------
 	columns: list, optional
 		List of the vcolumns names. If empty, all the vcolumns will be used.
 	percent: bool, optional
 		If set to True, the percentage of missing value will be also computed.
 	sort_result: bool, optional
 		If set to True, the result will be sorted.
 	desc: bool, optional
 		If set to True and 'sort_result' is set to True, the result will be sorted desc.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
		check_types([
			("columns", columns, [list], False), 
			("percent", percent, [bool], False), 
			("desc", desc, [bool], False), 
			("sort_result", sort_result, [bool], False)])
		columns_check(columns, self)
		columns = vdf_columns_names(columns, self)
		if not(columns): columns = self.get_columns()
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
	#---#
	def cov(self, 
			columns: list = [], 
			cmap: str = "",
			focus: str = "",
			show: bool = True):
		"""
	---------------------------------------------------------------------------
	Computes the Covariance Matrix of the vDataFrame. 

	Parameters
 	----------
 	columns: list, optional
 		List of the vcolumns names. If empty, all the numerical vcolumns will be 
 		used.
 	cmap: str, optional
 		Color Map.
 	focus: str, optional
 		Focus the computation on only one vcolumn.
 	show: bool, optional
 		If set to True, the Covariance Matrix will be drawn using Matplotlib.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.

	See Also
	--------
	vDataFrame.autocorr : Computes the Correlations between a vcolumn and its lags.
	vDataFrame.corr     : Computes the Correlation Matrix of the vDataFrame.
	vDataFrame.regr     : Computes the Regression Matrix of the vDataFrame.
		"""
		check_types([
			("columns", columns, [list], False), 
			("cmap", cmap, [str], False), 
			("focus", focus, [str], False), 
			("show", show, [bool], False)])
		columns_check(columns, self)
		columns = vdf_columns_names(columns, self)
		if (focus == ""):
			return (self.__aggregate_matrix__(method = "cov", columns = columns, cmap = cmap, show = show))
		else:
			columns_check([focus], self)
			focus = vdf_columns_names([focus], self)[0]
			return (self.__aggregate_vector__(focus, method = "cov", columns = columns, cmap = cmap, show = show))
	#---#
	def cummax(self,  
			   column: str, 
			   by: list = [], 
			   order_by = [],
			   name: str = ""):
		"""
	---------------------------------------------------------------------------
	Adds a new vcolumn to the vDataFrame by computing the cumulative maximum of
	the input vcolumn.

	Parameters
 	----------
 	column: str
 		Input vcolumn.
 	by: list, optional
 		vcolumns used in the partition.
 	order_by: dict / list, optional
 		List of the vcolumns used to sort the data using asc order or
 		dictionary of all the sorting methods. For example, to sort by "column1"
 		ASC and "column2" DESC, write {"column1": "asc", "column2": "desc"}
 	name: str, optional
 		Name of the new vcolumn. If empty, a default name will be generated.

 	Returns
 	-------
 	vDataFrame
		self

	See Also
	--------
	vDataFrame.rolling : Computes a customized moving window.
		"""
		return (self.rolling(func = "max", column = column, preceding = "UNBOUNDED", following = 0, by = by, order_by = order_by, name = name))
	#---#
	def cummin(self, 
			   column: str, 
			   by: list = [], 
			   order_by = [],
			   name: str = ""):
		"""
	---------------------------------------------------------------------------
	Adds a new vcolumn to the vDataFrame by computing the cumulative minimum of
	the input vcolumn.

	Parameters
 	----------
 	column: str
 		Input vcolumn.
 	by: list, optional
 		vcolumns used in the partition.
 	order_by: dict / list, optional
 		List of the vcolumns used to sort the data using asc order or
 		dictionary of all the sorting methods. For example, to sort by "column1"
 		ASC and "column2" DESC, write {"column1": "asc", "column2": "desc"}
 	name: str, optional
 		Name of the new vcolumn. If empty, a default name will be generated.

 	Returns
 	-------
 	vDataFrame
		self

	See Also
	--------
	vDataFrame.rolling : Computes a customized moving window.
		"""
		return (self.rolling(func = "min", column = column, preceding = "UNBOUNDED", following = 0, by = by, order_by = order_by, name = name))
	#---#
	def cumprod(self,  
			    column: str, 
			    by: list = [], 
			    order_by = [],
			    name: str = ""):
		"""
	---------------------------------------------------------------------------
	Adds a new vcolumn to the vDataFrame by computing the cumulative product of 
	the input vcolumn.

	Parameters
 	----------
 	column: str
 		Input vcolumn.
 	by: list, optional
 		vcolumns used in the partition.
 	order_by: dict / list, optional
 		List of the vcolumns used to sort the data using asc order or
 		dictionary of all the sorting methods. For example, to sort by "column1"
 		ASC and "column2" DESC, write {"column1": "asc", "column2": "desc"}
 	name: str, optional
 		Name of the new vcolumn. If empty, a default name will be generated.

 	Returns
 	-------
 	vDataFrame
		self

	See Also
	--------
	vDataFrame.rolling : Computes a customized moving window.
		"""
		return (self.rolling(func = "prod", column = column, preceding = "UNBOUNDED", following = 0, by = by, order_by = order_by, name = name))
	#---#
	def cumsum(self, 
			   column: str, 
			   by: list = [], 
			   order_by = [],
			   name: str = ""):
		"""
	---------------------------------------------------------------------------
	Adds a new vcolumn to the vDataFrame by computing the cumulative sum of the 
	input vcolumn.

	Parameters
 	----------
 	column: str
 		Input vcolumn.
 	by: list, optional
 		vcolumns used in the partition.
 	order_by: dict / list, optional
 		List of the vcolumns used to sort the data using asc order or
 		dictionary of all the sorting methods. For example, to sort by "column1"
 		ASC and "column2" DESC, write {"column1": "asc", "column2": "desc"}
 	name: str, optional
 		Name of the new vcolumn. If empty, a default name will be generated.

 	Returns
 	-------
 	vDataFrame
		self

	See Also
	--------
	vDataFrame.rolling : Computes a customized moving window.
		"""
		return (self.rolling(func = "sum", column = column, preceding = "UNBOUNDED", following = 0, by = by, order_by = order_by, name = name))
	#---#
	def current_relation(self, 
						 reindent: bool = True):
		"""
	---------------------------------------------------------------------------
	Returns the current vDataFrame relation.

	Parameters
 	----------
 	reindent: bool, optional
 		Reindent the text to be more readable. 

 	Returns
 	-------
 	str
 		The formatted current vDataFrame relation.
		"""
		if (reindent):
			return (indentSQL(self.__genSQL__()))
		else:
			return (self.__genSQL__())
	#---#
	def datecol(self):
		"""
	---------------------------------------------------------------------------
	Returns all the vDataFrame vcolumns of type date.

 	Returns
 	-------
 	List
 		List of all the vcolumns of type date.

 	See Also
 	--------
 	vDataFrame.catcol : Returns all the vDataFrame categorical vcolumns.
 	vDataFrame.numcol : Returns all the vDataFrame numerical vcolumns.
		"""
		columns = []
		cols = self.get_columns()
		for column in cols:
			if self[column].isdate():
				columns += [column]
		return (columns)
	#---#
	def describe(self, 
				 method: str = "auto", 
				 columns: list = [], 
				 unique: bool = True):
		"""
	---------------------------------------------------------------------------
	Aggregates the vDataFrame using multiple statistical aggregations: 
	min, max, median, unique... depending on the vcolumns types.

	Parameters
 	----------
 	method: str, optional
 		The describe method.
 			auto 	    : It is set to 'numerical' if at least one of the vDataFrame 
				vcolumns is numerical, 'categorical' otherwise.
			categorical : Uses only categorical aggregations during the computation.
 			numerical   : Descriptive Statistics are computed in a faster way than
				the 'aggregate' method.
 	columns: list, optional
 		List of the vcolumns names. If empty, the vcolumns will be selected
 		depending on the parameter 'method'.
 	unique: bool, optional
 		If set to True, the cardinality of each element will be computed.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
		check_types([
			("method", method, ["auto", "numerical", "categorical"], True), 
			("columns", columns, [list], False), 
			("unique", unique, [bool], False)])
		method = method.lower()
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
			try:
				idx = ["index", "count", "mean", "std", "min", "25%", "50%", "75%", "max"]
				values = {}
				for key in idx:
					values[key] = []
				col_to_compute = []
				for column in columns:
					if (self[column].isnum()):
						for fun in idx[1:]:
							pre_comp = self.__get_catalog_value__(column, fun)
							if (pre_comp == "VERTICAPY_NOT_PRECOMPUTED"):
								col_to_compute += [column]
								break
					else:
						print("\u26A0 Warning: The vcolumn {} is not numerical, it was ignored.\nTo get statistical information about all the different variables, please use the parameter method = 'categorical'.".format(column))
				for column in columns:
					if column not in col_to_compute:
						values["index"] += [column.replace('"', '')] 
						for fun in idx[1:]:
							values[fun] += [self.__get_catalog_value__(column, fun)]
				if (col_to_compute):
					query = "SELECT SUMMARIZE_NUMCOL({}) OVER () FROM {}".format(", ".join([elem if self[elem].ctype() != "boolean" else "{}::int".format(elem) for elem in col_to_compute]), self.__genSQL__())
					self.__executeSQL__(query, title = "Computes the descriptive statistics of all the numerical columns using SUMMARIZE_NUMCOL.")
					query_result = self._VERTICAPY_VARIABLES_["cursor"].fetchall()
					for i, key in enumerate(idx):
						values[key] += [elem[i] for elem in query_result]
					columns = [elem for elem in values["index"]]
			except:
				values = self.aggregate(["count", "mean", "std", "min", "25%", "50%", "75%", "max"], columns = columns).values
			if (unique):
				values["unique"] = self.aggregate(["unique"], columns = columns).values["unique"]
		elif (method == "categorical"):
			values = self.aggregate(["dtype", "unique", "count", "top", "top_percent"], columns = columns).values
		else:
			raise ValueError("The parameter 'method' must be in auto|numerical|categorical")
		self.__update_catalog__(tablesample(values, table_info = False).transpose().values)
		return (tablesample(values, table_info = False))
	#---#
	def drop(self, 
			 columns: list = []):
		"""
	---------------------------------------------------------------------------
	Drops the input vcolumns from the vDataFrame. Dropping vcolumns means 
	not selecting them in the final SQL code generation.
	Be Careful when using this method. It can make the vDataFrame structure 
	heavier if some other vcolumns are computed using the dropped vcolumns.

	Parameters
 	----------
 	columns: list, optional
 		List of the vcolumns names.

 	Returns
 	-------
 	vDataFrame
		self
		"""
		check_types([("columns", columns, [list], False)])
		columns_check(columns, self)
		columns = vdf_columns_names(columns, self)
		for column in columns:
			self[column].drop()
		return (self)
	#---#
	def drop_duplicates(self, 
						columns: list = []):
		"""
	---------------------------------------------------------------------------
	Filters the duplicated using a partition by the input vcolumns.

	Parameters
 	----------
 	columns: list, optional
 		List of the vcolumns names. If empty, all the vcolumns will be selected.

 	Returns
 	-------
 	vDataFrame
		self
		"""
		check_types([("columns", columns, [list], False)])
		columns_check(columns, self)
		count = self.duplicated(columns = columns, count = True)
		if (count):
			columns = self.get_columns() if not(columns) else vdf_columns_names(columns, self)
			name = "__verticapy_duplicated_index__" + str(random.randint(0, 10000000)) + "_"
			self.eval(name = name, expr = "ROW_NUMBER() OVER (PARTITION BY {})".format(", ".join(columns)))
			self.filter(expr = '"{}" = 1'.format(name))
			self._VERTICAPY_VARIABLES_["exclude_columns"] += ['"{}"'.format(name)]
		else:
			print("\u26A0 Warning: No duplicates detected")
		return (self)
	#---#
	def dropna(self, 
			   columns: list = [], 
			   print_info: bool = True):
		"""
	---------------------------------------------------------------------------
	Filters the vDataFrame where the input vcolumns are missing.

	Parameters
 	----------
 	columns: list, optional
 		List of the vcolumns names. If empty, all the vcolumns will be selected.
 	print_info: bool, optional
 		If set to True, it will display the result.

 	Returns
 	-------
 	vDataFrame
		self

 	See Also
	--------
	vDataFrame.filter: Filters the data using the input expression.
		"""
		check_types([
			("columns", columns, [list], False), 
			("print_info", print_info, [bool], False)])
		columns_check(columns, self)
		columns = self.get_columns() if not(columns) else vdf_columns_names(columns, self)
		total = self.shape()[0]
		for column in columns:
			self[column].dropna(print_info = False)
		if (print_info):
			total -= self.shape()[0]
			if (total == 0):
				print("\u26A0 Warning: Nothing was dropped")
			else:
				print("{} element(s) was/were dropped".format(total))
		return (self)
	#---#
	def dsn_restart(self):
		"""
	---------------------------------------------------------------------------
	Creates a new cursor based on the DSN used to build the vDataFrame.

 	Returns
 	-------
 	vDataFrame
		self

	See Also
 	--------
 	vDataFrame.set_dsn : Sets a new DB DSN.
		"""
		from verticapy import vertica_cursor
		if (self._VERTICAPY_VARIABLES_["dsn"]):
			self._VERTICAPY_VARIABLES_["cursor"] = vertica_cursor(self._VERTICAPY_VARIABLES_["dsn"])
		else:
			print("\u26A0 Warning: No DSN found in the vDataFrame.")
		return (self)
	#---#
	def dtypes(self):
		"""
	---------------------------------------------------------------------------
	Returns the different vcolumns types.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.
		"""
		values = {"index" : [], "dtype" : []}
		for column in self.get_columns():
			values["index"] += [column]
			values["dtype"] += [self[column].ctype()]
		return (tablesample(values, table_info = False))
	#---#
	def duplicated(self, 
				   columns: list = [], 
				   count: bool = False, 
				   limit: int = 30):
		"""
	---------------------------------------------------------------------------
	Returns the duplicated values.

	Parameters
 	----------
 	columns: list, optional
 		List of the vcolumns names. If empty, all the vcolumns will be selected.
 	count: bool, optional
 		If set to True, the method will also return the count of each duplicates.
 	limit: int, optional
 		The limited number of elements to be displayed.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.

 	See Also
 	--------
 	vDataFrame.drop_duplicates : Filters the duplicated values.
		"""
		check_types([
			("columns", columns, [list], False), 
			("count", count, [bool], False),
			("limit", limit, [int, float], False)])
		columns_check(columns, self)
		columns = self.get_columns() if not(columns) else vdf_columns_names(columns, self)
		query = "(SELECT *, ROW_NUMBER() OVER (PARTITION BY {}) AS duplicated_index FROM {}) duplicated_index_table WHERE duplicated_index > 1".format(", ".join(columns), self.__genSQL__())
		self.__executeSQL__(query = "SELECT COUNT(*) FROM {}".format(query), title = "Computes the number of duplicates.")
		total = self._VERTICAPY_VARIABLES_["cursor"].fetchone()[0]
		if (count):
			return total
		result = to_tablesample("SELECT {}, MAX(duplicated_index) AS occurrence FROM {} GROUP BY {} ORDER BY occurrence DESC LIMIT {}".format(", ".join(columns), query, ", ".join(columns), limit), self._VERTICAPY_VARIABLES_["cursor"], name = "Duplicated Rows (total = {})".format(total))
		self.__executeSQL__(query = "SELECT COUNT(*) FROM (SELECT {}, MAX(duplicated_index) AS occurrence FROM {} GROUP BY {}) t".format(", ".join(columns), query, ", ".join(columns)), title = "Computes the number of distinct duplicates.")
		result.count = self._VERTICAPY_VARIABLES_["cursor"].fetchone()[0]
		return (result)
	#---#
	def empty(self):
		"""
	---------------------------------------------------------------------------
	Returns True if the vDataFrame is empty.

 	Returns
 	-------
 	bool
 		True if the vDataFrame has no vcolumns.
		"""
		return not(self.get_columns())
	#---# 
	def eval(self, 
			 name: str, 
			 expr: str):
		"""
	---------------------------------------------------------------------------
	Evaluates a customized expression.

	Parameters
 	----------
 	name: str
 		Name of the new vcolumn.
 	expr: str
 		Expression used to compute the new feature. It must be pure SQL. 
 		For example, 'CASE WHEN "column" > 3 THEN 2 ELSE NULL END' and
 		'POWER("column", 2)' will work.

 	Returns
 	-------
 	vDataFrame
		self

	See Also
	--------
	vDataFrame.analytic : Adds a new vcolumn to the vDataFrame by using an advanced 
		analytical function on a specific vcolumn.
		"""
		check_types([
			("name", name, [str], False), 
			("expr", expr, [str], False)])
		name = str_column(name.replace('"', '_'))
		if column_check_ambiguous(name, self.get_columns()):
			raise ValueError("A vcolumn has already the alias {}.\nBy changing the parameter 'name', you'll be able to solve this issue.".format(name))
		try:
			ctype = get_data_types("SELECT {} AS {} FROM {} LIMIT 0".format(expr, name, self.__genSQL__()), self._VERTICAPY_VARIABLES_["cursor"], name.replace('"', '').replace("'", "''"))
		except:
			try:
				ctype = get_data_types("SELECT {} AS {} FROM {} LIMIT 0".format(expr, name, self.__genSQL__()), self._VERTICAPY_VARIABLES_["cursor"], name.replace('"', '').replace("'", "''"), self._VERTICAPY_VARIABLES_["schema_writing"])
			except:
				raise ValueError("The expression '{}' seems to be incorrect.\nBy turning on the SQL with the 'sql_on_off' method, you'll print the SQL code generation and probably see why the evaluation didn't work.".format(expr))
		ctype = ctype if (ctype) else "undefined"
		category = category_from_type(ctype = ctype)
		all_cols, max_floor = self.get_columns(), 0
		for column in all_cols:
			if ((str_column(column) in expr) or (re.search(re.compile('\\b{}\\b'.format(column.replace('"', ''))), expr))):
				max_floor = max(len(self[column].transformations), max_floor)
		transformations = [('___VERTICAPY_UNDEFINED___', "___VERTICAPY_UNDEFINED___", "___VERTICAPY_UNDEFINED___") for i in range(max_floor)] + [(expr, ctype, category)]
		new_vColumn = vColumn(name, parent = self, transformations = transformations)
		setattr(self, name, new_vColumn)
		setattr(self, name.replace('"', ''), new_vColumn)
		self._VERTICAPY_VARIABLES_["columns"] += [name]
		self.__add_to_history__("[Eval]: A new vcolumn {} was added to the vDataFrame.".format(name)) 
		return (self)
	#---#
	def expected_store_usage(self, 
							 unit: str = 'b'):
		"""
	---------------------------------------------------------------------------
	Returns the vDataFrame expected store usage. 

	Parameters
	----------
	unit: str, optional
		unit used for the computation
		b : byte
		kb: kilo byte
		gb: giga byte
		tb: tera byte

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.

	See Also
	--------
	vDataFrame.memory_usage : Returns the vDataFrame memory usage.
		"""
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
	#---#
	def explain(self,
				digraph: bool = False):
		"""
	---------------------------------------------------------------------------
	Provides information on how Vertica is computing the current vDataFrame
	relation.

	Parameters
	----------
	digraph: bool, optional
		If set to True, returns only the digraph of the explain plan.

 	Returns
 	-------
 	str
		explain plan
		"""
		query = "EXPLAIN SELECT * FROM {}".format(self.__genSQL__())
		self.__executeSQL__(query = query, title = "Explaining the Current Relation")
		result = self._VERTICAPY_VARIABLES_["cursor"].fetchall()
		result = [elem[0] for elem in result]
		result = "\n".join(result)
		if not(digraph):
			result = result.replace("------------------------------\n", "")
			result = result.replace("\\n", "\n\t")
			result = result.replace(", ", ",").replace(",", ", ").replace("\n}", "}")
		else:
			result = "digraph G {" + result.split("digraph G {")[1]
		return (result)
	#---#
	def fillna(self,
			   val: dict = {},
			   method: dict = {},
			   numeric_only: bool = False,
			   print_info: bool = False):
		"""
	---------------------------------------------------------------------------
	Fills the vcolumns missing elements using specific rules.

	Parameters
 	----------
 	val: dict, optional
 		Dictionary of values. The dictionary must be similar to the following:
 		{"column1": val1 ..., "columnk": valk}. Each key of the dictionary must
 		be a vcolumn. The missing values of the input vcolumns will be replaced
 		by the input value.
 	method: dict, optional
 		Method used to impute the missing values.
 			auto    : Mean for the numerical and Mode for the categorical vcolumns.
			mean    : Average.
			median  : Median.
			mode    : Mode (most occurent element).
			0ifnull : 0 when the vcolumn is null, 1 otherwise.
				More Methods are available on the vDataFrame[].fillna method.
	numeric_only: bool, optional
		If parameters 'val' and 'method' are empty and 'numeric_only' is set
		to True then all the numerical vcolumns will be imputed by their average.
		If set to False, all the categorical vcolumns will be also imputed by their
		mode.
	print_info: bool, optional
		If set to True, the information of the imputations will be displayed.

 	Returns
 	-------
 	vDataFrame
		self

	See Also
	--------
	vDataFrame[].fillna : Fills the vcolumn missing values. This method is more 
		complete than the vDataFrame.fillna method by allowing more parameters.
		"""
		check_types([
			("val", val, [dict], False), 
			("method", method, [dict], False), 
			("numeric_only", numeric_only, [bool], False), 
			("print_info", print_info, [bool], False)])
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
	#---#
	def filter(self, 
			   expr: str = "", 
			   conditions: list = [],
			   print_info: bool = True):
		"""
	---------------------------------------------------------------------------
	Filters the vDataFrame using the input expressions.

	Parameters
 	----------
 	expr: str, optional
 		Customized SQL expression used to filter the data. For example to keep
 		only the records where the vcolumn 'column' is greater than 5 you can
 		write 'column > 5' or '"column" > 5'. Try to always keep the double 
 		quotes if possible, it will make the parsing easier. 
 	conditions: list, optional
 		List of expressions. For example to keep only the records where the 
 		vcolumn 'column' is greater than 5 and lesser than 10 you can write 
 		['"column" > 5', '"column" < 10'].
 	print_info: bool, optional
 		If set to True, the result of the filtering will be displayed.

 	Returns
 	-------
 	vDataFrame
		self

	See Also
	--------
	vDataFrame.at_time      : Filters the data at the input time.
	vDataFrame.between_time : Filters the data between two time ranges.
	vDataFrame.first        : Filters the data by only keeping the first records.
	vDataFrame.last         : Filters the data by only keeping the last records.
	vDataFrame.search       : Searches the elements which matches with the input 
		conditions.
		"""
		check_types([
			("expr", expr, [str], False), 
			("conditions", conditions, [list], False), 
			("print_info", print_info, [bool], False)])
		count = self.shape()[0]
		if not(expr):
			for condition in conditions:
				self.filter(expr = condition, print_info = False)
			count -= self.shape()[0]
			if (count > 0):
				if (print_info):
					print("{} element(s) was/were filtered".format(count))
				self.__add_to_history__("[Filter]: {} element(s) was/were filtered using the filter '{}'".format(count, conditions))
			else:
				if (print_info):
					print("Nothing was filtered.")
		else:
			max_pos = 0
			for column in self._VERTICAPY_VARIABLES_["columns"]:
				max_pos = max(max_pos, len(self[column].transformations) - 1)
			new_count = self.shape()[0]
			self._VERTICAPY_VARIABLES_["where"] += [(expr, max_pos)]
			try:
				self._VERTICAPY_VARIABLES_["cursor"].execute("SELECT COUNT(*) FROM {}".format(self.__genSQL__()))
				new_count = self._VERTICAPY_VARIABLES_["cursor"].fetchone()[0]
				count -= new_count
			except:
				del self._VERTICAPY_VARIABLES_["where"][-1]
				if (print_info):
					print("\u26A0 Warning: The expression '{}' is incorrect.\nNothing was filtered.".format(expr))
				return (self)
			if (count > 0):
				self.__update_catalog__(erase = True)
				self._VERTICAPY_VARIABLES_["count"] = new_count
				if (print_info):
					print("{} element(s) was/were filtered".format(count))
				self.__add_to_history__("[Filter]: {} element(s) was/were filtered using the filter '{}'".format(count, expr))
			else:
				del self._VERTICAPY_VARIABLES_["where"][-1]
				if (print_info):
					print("\u26A0 Warning: Nothing was filtered.")
		return (self)
	#---#
	def first(self, 
			  ts: str, 
			  offset: str):
		"""
	---------------------------------------------------------------------------
	Filters the vDataFrame by only keeping the first records.

	Parameters
 	----------
 	ts: str
 		TS (Time Series) vcolumn used to filter the data. The vcolumn type must be
 		date like (date, datetime, timestamp...)
 	offset: str
 		Interval offset. For example, to filter and keep only the first 6 months of
 		records, offset should be set to '6 months'. 

 	Returns
 	-------
 	vDataFrame
		self

	See Also
	--------
	vDataFrame.at_time      : Filters the data at the input time.
	vDataFrame.between_time : Filters the data between two time ranges.
	vDataFrame.filter       : Filters the data using the input expression.
	vDataFrame.last         : Filters the data by only keeping the last records.
		"""
		check_types([
			("ts", ts, [str], False), 
			("offset", offset, [str], False)])
		ts = vdf_columns_names([ts], self)[0]
		query = "SELECT (MIN({}) + '{}'::interval)::varchar FROM {}".format(ts, offset, self.__genSQL__())
		self._VERTICAPY_VARIABLES_["cursor"].execute(query)
		first_date = self._VERTICAPY_VARIABLES_["cursor"].fetchone()[0]
		self.filter("{} <= '{}'".format(ts, first_date))
		return (self)
	#---#
	def gmaps_scatter(self,
					  api_key: str,
					  lat: str,
		   	 	  	  lon: str,
		   	 	  	  column_map: str = "",
		   	 	  	  max_nb_points: int = 200):
		"""
	---------------------------------------------------------------------------
	Draws the Scatter Plot of the input vcolumns using the Plotly module. 
	Plotly must be installed.

	Parameters
 	----------
 	api_key: str
 		Google API key.
 	lat: str
 		Latitude vcolumn
 	lon: str
 		Longitude vcolumn
 	column_map: str, optional
 		vcolumn used to draw the correct cmap color.
 	max_nb_points: int, optional
 		Maximum number of points to display.

 	Returns
 	-------
 	vDataFrame
		self

 	See Also
 	--------
 	vDataFrame.bubble  : Draws the Bubble Plot of the input vcolumns.
 	vDataFrame.scatter : Draws the Scatter Plot of the input vcolumns.
		"""
		check_types([
			("lat", lat, [str], False), 
			("lon", lon, [str], False),
			("api_key", api_key, [str], False),
			("column_map", column_map, [str], False),
			("max_nb_points", max_nb_points, [int], False)])
		columns_check([lat, lon], self)
		lat, lon = vdf_columns_names([lat, lon], self)
		if (column_map):
			columns_check([column_map], self)
			column_map = vdf_columns_names([column_map], self)[0]
			from verticapy.plot import gmaps_geo_plot_cmap
			fig = gmaps_geo_plot_cmap(self, lat = lat, lon = lon, api_key = api_key, column_map = column_map, max_nb_points = max_nb_points)
		else:
			from verticapy.plot import gmaps_geo_plot_regular
			fig = gmaps_geo_plot_regular(self, lat = lat, lon = lon, api_key = api_key, max_nb_points = max_nb_points) 
		return (fig)
	#---#
	def get_columns(self, 
					exclude_columns: list = []):
		"""
	---------------------------------------------------------------------------
	Returns the vDataFrame vcolumns.

	Parameters
 	----------
 	exclude_columns: list, optional
 		List of the vcolumns names to exclude from the final list. 

 	Returns
 	-------
 	List
 		List of all the vDataFrame columns.

	See Also
	--------
	vDataFrame.catcol  : Returns all the categorical vDataFrame vcolumns.
	vDataFrame.datecol : Returns all the vDataFrame vcolumns of type date.
	vDataFrame.numcol  : Returns all the numerical vDataFrame vcolumns.

		"""
		check_types([("exclude_columns", exclude_columns, [list], False)])
		columns = self._VERTICAPY_VARIABLES_["columns"]
		result = []
		exclude_columns += self._VERTICAPY_VARIABLES_["exclude_columns"]
		exclude_columns = [elem.replace('"', '').lower() for elem in exclude_columns]
		for column in columns:
			if (column.replace('"', '').lower() not in exclude_columns):
				result += [column]
		return (result)
	#---#
	def get_dummies(self, 
					columns: list = [],
					max_cardinality: int = 12, 
					prefix_sep: str = "_", 
					drop_first: bool = True, 
					use_numbers_as_suffix: bool = False):
		"""
	---------------------------------------------------------------------------
	Encodes the vcolumns using the One Hot Encoding algorithm.

	Parameters
 	----------
 	columns: list, optional
 		List of the vcolumns used to train the One Hot Encoding model. If empty, 
 		only the vcolumns having a cardinality lesser than 'max_cardinality' will 
 		be used.
 	max_cardinality: int, optional
 		Cardinality threshold used to determine if the vcolumn will be taken into
 		account during the encoding. This parameter is used only if the parameter 
 		'columns' is empty.
 	prefix_sep: str, optional
 		Prefix delimitor of the dummies names.
 	drop_first: bool, optional
 		Drops the first dummy to avoid the creation of correlated features.
 	use_numbers_as_suffix: bool, optional
 		Uses numbers as suffix instead of the vcolumns categories.

 	Returns
 	-------
 	vDataFrame
 		self

	See Also
	--------
	vDataFrame[].decode       : Encodes the vcolumn using a user defined Encoding.
	vDataFrame[].discretize   : Discretizes the vcolumn.
	vDataFrame[].get_dummies  : Computes the vcolumns result of One Hot Encoding.
	vDataFrame[].label_encode : Encodes the vcolumn using the Label Encoding.
	vDataFrame[].mean_encode  : Encodes the vcolumn using the Mean Encoding of a response.
		"""
		check_types([
			("columns", columns, [list], False), 
			("max_cardinality", max_cardinality, [int, float], False), 
			("prefix_sep", prefix_sep, [str], False), 
			("drop_first", drop_first, [bool], False), 
			("use_numbers_as_suffix", use_numbers_as_suffix, [bool], False)])
		columns_check(columns, self)
		cols_hand = True if (columns) else False
		columns = self.get_columns() if not(columns) else vdf_columns_names(columns, self)
		for column in columns:
			if (self[column].nunique(True) < max_cardinality):
				self[column].get_dummies("", prefix_sep, drop_first, use_numbers_as_suffix)
			elif (cols_hand):
				print("\u26A0 Warning: The vcolumn {} was ignored because of its high cardinality\nIncrease the parameter 'max_cardinality' to solve this issue or use directly the vcolumn get_dummies method".format(column))
		return (self)
	#---#
	def groupby(self, 
				columns: list, 
				expr: list = [],
				check: bool = True):
		"""
	---------------------------------------------------------------------------
	Aggregates the vDataFrame by grouping the elements.

	Parameters
 	----------
 	columns: list
 		List of the vcolumns used for the grouping.
 	expr: list, optional
 		List of the different aggregations. Pure SQL must be written. Aliases can
 		also be given. 'SUM(column)' or 'AVG(column) AS my_new_alias' are correct
 		whereas 'AVG' is incorrect. Aliases are recommended to keep the track of 
 		the different features and not have ambiguous names. The function MODE does
 		not exist in SQL for example but can be obtained using the 'analytic' method
 		first and then by grouping the result.
 	check: bool, optional
 		Do a columns checking before the computation. Set it to False to do
 		customized selection.

 	Returns
 	-------
 	vDataFrame
 		object result of the grouping.

	See Also
	--------
	vDataFrame.append   : Merges the vDataFrame with another relation.
	vDataFrame.analytic : Adds a new vcolumn to the vDataFrame by using an advanced 
 		analytical function on a specific vcolumn.
	vDataFrame.join     : Joins the vDataFrame with another relation.
	vDataFrame.sort     : Sorts the vDataFrame.
		"""
		check_types([
			("columns", columns, [list], False), 
			("expr", expr, [list], False)])
		if (check):
			columns_check(columns, self)
			columns = vdf_columns_names(columns, self)
		relation = "(SELECT {} FROM {} GROUP BY {}) groupby_table".format(", ".join(columns + expr), self.__genSQL__(), ", ".join(columns)) 
		return (self.__vdf_from_relation__(relation, "groupby", "[Groupby]: The columns were grouped by {}".format(", ".join(columns))))
	#---#
	def hchart(self, 
			   x = None,
			   y = None,
			   z = None,
			   c = None,
			   aggregate: bool = True,
			   kind = "boxplot",
			   width: int = 600,
			   height: int = 400,
			   options: dict = {},
			   h: float = -1,
			   max_cardinality: int = 10,
			   limit = 10000,
			   drilldown: bool = False,
			   stock: bool = False,
			   alpha: float = 0.25):
		"""
	---------------------------------------------------------------------------
	Computes an interactive graphic using the High Chart API
	https://api.highcharts.com/highcharts/
	The object returned can be customized using the API parameters and the
	set_dict_options method.

	Parameters
 	----------
 	x / y / z / c: str / list
 		vcolumns and aggregations used to draw the Chart. They will depend on the 
 		Chart type. An expression must be written entirely in SQL. For example: 
 		AVG(column1) + SUM(column2) AS new_name will work.
 			area / area_ts / line / spline
 				x: numerical or type date like vcolumn
 				y: Can be => list of the expressions used to draw the Plot
 						  => single expression
			    z: [OPTIONAL] vcolumn representing the different categories (only 
			    			  if y is a single vcolumn)
 			area_range
 				x: numerical or type date like vcolumn
 				y: list of 3 expressions [expression, lower bound, upper bound]
 			bar (single) / donut / donut3d / hist (single) / pie / pie_half / pie3d
 				x: vcolumn used to compute the categories
 				y: [OPTIONAL] numerical expression representing the categories values.
 							  If empty, COUNT(*) is used as default aggregation.
 			bar (double / drilldown) / donut (drilldown) / donut3d (drilldown)
 			hist (double / drilldown) / pie (drilldown) / pie_half (drilldown)
 			pie3d (drilldown) / stacked_bar / stacked_hist
 				x: vcolumn used to compute the first category
				y: vcolumn used to compute the second category
				z: [OPTIONAL] numerical expression representing the different categories 
							  values. If empty, COUNT(*) is used as default aggregation.
 			biserial / boxplot / pearson / kendall / pearson / spearman
 				x: list of the vcolumns used to draw the Chart
 			bubble / scatter
 				x: numerical vcolumn
 				y: numerical vcolumn
 				z: numerical vcolumn (bubble size in case of bubble plot, third 
 					dimension in case of scatter plot)
 				c: [OPTIONAL] vcolumn used to compute the different categories
 			candlestick
 				x: date type like vcolumn
				y: Can be => numerical vcolumn 
						  => list of 5 expressions [last quantile, maximum, minimum, 
						  							first quantile, volume]
 			negative_bar
 				x: binary vcolumn used to compute the first category
 				y: vcolumn used to compute the second category
 				z: [OPTIONAL] numerical expression representing the categories values.
 							  If empty, COUNT(*) is used as default aggregation.
 			spider
 				x: vcolumn used to compute the different categories
 				y: [OPTIONAL] Can be => list of the expressions used to draw the Plot
 						  			 => single expression. 
 						  	  If empty, COUNT(*) is used as default aggregation.
 	aggregate: bool, optional
 		If set to True, the input vcolumns will be aggregated.
 	kind: str, optional
 		Chart Type.
 			area         : Area Chart
 			area_range   : Area Range Chart
 			area_ts      : Area Chart with Time Series Design
 			bar          : Bar Chart
 			biserial     : Biserial Point Matrix (Correlation between binary
 							variables and numerical)
			boxplot      : Box Plot
			bubble       : Bubble Plot
			candlestick  : Candlestick and Volumes (Time Series Special Plot)
			cramer       : Cramer's V Matrix (Correlation between categories)
			donut        : Donut Chart
			donut3d      : 3D Donut Chart
			heatmap      : Heatmap
			hist         : Histogram
			kendall      : Kendall Correlation Matrix
			line         : Line Plot
			negative_bar : Multi Bar Chart for binary classes
			pearson      : Pearson Correlation Matrix
			pie          : Pie Chart
			pie_half     : Half Pie Chart
			pie3d        : 3D Pie Chart
			scatter      : Scatter Plot
			spider       : Spider Chart
			spline       : Spline Plot
			stacked_bar  : Stacked Bar Chart
			stacked_hist : Stacked Histogram
			spearman     : Spearman Correlation Matrix
 	width: int, optional
 		Chart Width.
 	height: int, optional
 		Chart Height.
 	options: dict, optional
 		High Chart Dictionary used to customize the Chart. Look at the API 
 		documentation to know the different options.
 	h: float, optional
 		Interval width of the bar. If empty, an optimized h will be computed.
 	max_cardinality: int, optional
		Maximum number of the vcolumn distinct elements.
 	limit: int, optional
 		Maximum number of elements to draw.
 	drilldown: bool, optional
 		Drilldown Chart: Only possible for Bars, Histograms, donuts and pies.
 						 Instead of drawing 2D charts, this option allows you
 						 to add a drilldown effect to 1D Charts.
 	stock: bool, optional
 		Stock Chart: Only possible for Time Series. The design of the Time
 					 Series is dragable and have multiple options.
 	alpha: float, optional
 		Float used to determine the position of the upper and lower quantile
 		(Used when kind is set to 'candlestick')

 	Returns
 	-------
 	Highchart
		Chart Object

			   h: float = -1,
		"""
		check_types([("kind", kind, [str], False)])
		kind = kind.lower()
		check_types([
			("aggregate", aggregate, [bool], False),
			("kind", kind, ["area", "area_range", "area_ts", 
							"bar", "boxplot", "bubble", "candlestick",
							"donut", "donut3d", "heatmap", 
							"hist", "line", "negative_bar", 
							"pie", "pie_half", "pie3d", 
							"scatter", "spider", "spline", 
							"stacked_bar", "stacked_hist", "pearson", 
							"kendall", "cramer", "biserial", "spearman"], True),
			("options", options, [dict], False),
			("width", width, [int, float], False),
			("height", height, [int, float], False),
			("drilldown", drilldown, [bool], False),
			("stock", stock, [bool], False),
			("limit", limit, [int, float], False),
			("max_cardinality", max_cardinality, [int, float], False),
			("h", h, [int, float], False),
			("alpha", alpha, [float], False)])
		from verticapy.hchart import hchart_from_vdf
		return hchart_from_vdf(self, x, y, z, c, aggregate, kind, width, height, options, h, max_cardinality, limit, drilldown, stock, alpha)
	#---# 
	def head(self, limit: int = 5):
		"""
	---------------------------------------------------------------------------
	Returns the vDataFrame head.

	Parameters
 	----------
 	limit: int, optional
 		Number of elements to display.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.

	See Also
	--------
	vDataFrame.tail : Returns the vDataFrame tail.
		"""
		return (self.tail(limit = limit))
	#---# 
	def hexbin(self,
		   	   columns: list,
		   	   method: str = "count",
		       of: str = "",
		       cmap: str = '',
		       gridsize: int = 20,
		       color: str = "white",
		       bbox: list = [],
		   	   img: str = ""):
		"""
	---------------------------------------------------------------------------
	Draws the Hexbin of the input vcolumns based on an aggregation.

	Parameters
 	----------
 	columns: list
 		List of the vcolumns names. The list must have two elements.
 	method: str, optional
 		The method used to aggregate the data.
 			count   : Number of elements.
 			density : Percentage of the distribution.
 			mean    : Average of the vcolumn 'of'.
 			min     : Minimum of the vcolumn 'of'.
 			max     : Maximum of the vcolumn 'of'.
 			sum     : Sum of the vcolumn 'of'.
 	of: str, optional
 		The vcolumn used to compute the aggregation.
 	cmap: str, optional
 		Color Map.
 	gridsize: int, optional
 		Hexbin grid size.
 	color: str, optional
 		Color of the Hexbin borders.
 	bbox: list, optional
 		List of 4 elements to delimit the boundaries of the final Plot. 
 		It must be similar the following list: [xmin, xmax, ymin, ymax]
 	img: str, optional
 		Path to the image to display as background.

 	Returns
 	-------
 	vDataFrame
		self

 	See Also
 	--------
 	vDataFrame.pivot_table : Draws the Pivot Table of vcolumns based on an aggregation.
		"""
		check_types([
			("columns", columns, [list], False), 
			("method", method, ["density", "count", "avg", "min", "max", "sum"], True), 
			("of", of, [str], False), 
			("cmap", cmap, [str], False), 
			("gridsize", gridsize, [int, float], False), 
			("color", color, [str], False),
			("bbox", bbox, [list], False),
			("img", img, [str], False)])
		method = method.lower()
		columns_check(columns, self, [2])
		columns = vdf_columns_names(columns, self)
		if (of):
			columns_check([of], self)
			of = vdf_columns_names([of], self)[0]
		if not(cmap):
			from verticapy.plot import gen_cmap
			cmap = gen_cmap()[0]
		from verticapy.plot import hexbin
		hexbin(self, columns, method, of, cmap, gridsize, color, bbox, img)
		return (self)
	#---# 
	def hist(self,
			 columns: list,
			 method: str = "density",
			 of: str = "",
			 max_cardinality: tuple = (6, 6),
			 h: tuple = (None, None),
			 hist_type: str = "auto"):
		"""
	---------------------------------------------------------------------------
	Draws the Histogram of the input vcolumns based on an aggregation.

	Parameters
 	----------
 	columns: list
 		List of the vcolumns names. The list must have less than 5 elements.
 	method: str, optional
 		The method used to aggregate the data.
 			count   : Number of elements.
 			density : Percentage of the distribution.
 			mean    : Average of the vcolumn 'of'.
 			min     : Minimum of the vcolumn 'of'.
 			max     : Maximum of the vcolumn 'of'.
 			sum     : Sum of the vcolumn 'of'.
 			q%      : q Quantile of the vcolumn 'of' (ex: 50% to get the median).
 	of: str, optional
 		The vcolumn used to compute the aggregation.
 	h: tuple, optional
 		Interval width of the vcolumns 1 and 2 bars. It is only valid if the 
 		vcolumns are numerical. Optimized h will be computed if the parameter 
 		is empty or invalid.
 	max_cardinality: tuple, optional
 		Maximum number of distinct elements for vcolumns 1 and 2 to be used as 
 		categorical (No h will be picked or computed)
 	hist_type: str, optional
 		The Histogram Type.
 			auto    : Regular Histogram based on 1 or 2 vcolumns.
 			multi   : Multiple Regular Histograms based on 1 to 5 vcolumns.
			stacked : Stacked Histogram based on 2 vcolumns.

 	Returns
 	-------
 	vDataFrame
		self

 	See Also
 	--------
 	vDataFrame.bar         : Draws the Bar Chart of the input vcolumns based on an aggregation.
 	vDataFrame.boxplot     : Draws the Box Plot of the input vcolumns.
 	vDataFrame.pivot_table : Draws the Pivot Table of vcolumns based on an aggregation.
		"""
		check_types([
			("columns", columns, [list], False), 
			("method", method, [str], False), 
			("of", of, [str], False), 
			("max_cardinality", max_cardinality, [tuple], False), 
			("h", h, [tuple], False), 
			("hist_type", hist_type, ["auto", "multi", "stacked"], True)])
		method = method.lower()
		columns_check(columns, self, [1, 2, 3, 4, 5])
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
				from verticapy.plot import multiple_hist
				h_0 = h[0] if (h[0]) else 0
				multiple_hist(self, columns, method, of, h_0)
			else:
				from verticapy.plot import hist2D
				hist2D(self, columns, method, of, max_cardinality, h, stacked)
		return (self)
	#---#
	def info(self):
		"""
	---------------------------------------------------------------------------
	Displays information about the different vDataFrame transformations.

 	Returns
 	-------
 	vDataFrame
		self
		"""
		if (len(self._VERTICAPY_VARIABLES_["history"]) == 0):
			print("The vDataFrame was never modified.")
		elif (len(self._VERTICAPY_VARIABLES_["history"]) == 1):
			print("The vDataFrame was modified with only one action: ")
			print(" * " + self._VERTICAPY_VARIABLES_["history"][0])
		else:
			print("The vDataFrame was modified many times: ")
			for modif in self._VERTICAPY_VARIABLES_["history"]:
				print(" * " + modif)
		return (self)
	#---#
	def isin(self, 
			 val: dict):
		"""
	---------------------------------------------------------------------------
	Looks if some specific records are in the vDataFrame.

	Parameters
 	----------
 	val: dict
 		Dictionary of the different records. Each key of the dictionary must 
 		represent a vcolumn. For example, to check if Badr Ouali and 
 		Fouad Teban are in the vDataFrame. You can write the following dict:
 		{"name": ["Teban", "Ouali"], "surname": ["Fouad", "Badr"]}

 	Returns
 	-------
 	list
 		List containing the bools of the different searches.
		"""
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
			query = "SELECT * FROM {} WHERE {} LIMIT 1".format(self.__genSQL__(), " AND ".join(tmp_query))
			self._VERTICAPY_VARIABLES_["cursor"].execute(query)
			isin += [self._VERTICAPY_VARIABLES_["cursor"].fetchone() != None] 
		return (isin)
	#---#
	def join(self, 
			 input_relation, 
			 on: dict = {},
			 how: str = 'natural',
			 expr1: list = ['*'],
			 expr2: list = ['*']):
		"""
	---------------------------------------------------------------------------
	Joins the vDataFrame with another one or an input relation.

	Parameters
 	----------
 	input_relation: str/vDataFrame
 		Relation used to do the merging.
 	on: dict, optional
 		Dictionary of all the different keys. The dict must be similar to the following:
 		{"relationA_key1": "relationB_key1" ..., "relationA_keyk": "relationB_keyk"}
 		where relationA is the current vDataFrame and relationB is the input relation
 		or the input vDataFrame.
 	how: str, optional
 		Join Type.
 			left    : Left Join.
 			right   : Right Join.
 			cross   : Cross Join.
 			full    : Full Outer Join.
 			natural : Natural Join.
 			inner   : Inner Join.
 	expr1: list, optional
 		List of the different columns to select from the current vDataFrame. 
 		Pure SQL must be written. Aliases can also be given. 'column' or 
 		'column AS my_new_alias' are correct. Aliases are recommended to keep 
 		the track of the different features and not have ambiguous names. 
 	expr2: list, optional
 		List of the different columns to select from the input relation. 
 		Pure SQL must be written. Aliases can also be given. 'column' or 
 		'column AS my_new_alias' are correct. Aliases are recommended to keep 
 		the track of the different features and not have ambiguous names. 

 	Returns
 	-------
 	vDataFrame
 		object result of the join.

	See Also
	--------
	vDataFrame.append  : Merges the vDataFrame with another relation.
	vDataFrame.groupby : Aggregates the vDataFrame.
	vDataFrame.sort    : Sorts the vDataFrame.
		"""
		check_types([
			("on", on, [dict], False), 
			("how", how.lower(), ["left", "right", "cross", "full", "natural", "self", "inner", ""], True), 
			("expr1", expr1, [list], False), 
			("expr2", expr2, [list], False)])
		how = how.lower()
		columns_check([elem for elem in on], self)
		if (type(input_relation) == type(self)):
			columns_check([on[elem] for elem in on], input_relation)
			vdf_cols = []
			for elem in on:
				vdf_cols += [on[elem]]
			columns_check(vdf_cols, input_relation)
			second_relation = "(SELECT * FROM {}) AS y".format(input_relation.__genSQL__())
		elif (type(input_relation) == str):
			second_relation = "(SELECT * FROM {}) AS y".format(input_relation)
		else:
			raise TypeError("Parameter 'input_relation' type must be one of the following [{}, {}], found {}".format(str, type(self), type(input_relation)))
		on_join = " AND ".join(['x."' + elem.replace('"', '') + '" = y."' + on[elem].replace('"', '') + '"' for elem in on])
		on_join = " ON {}".format(on_join) if (on_join) else ""
		first_relation = "(SELECT * FROM {}) AS x".format(self.__genSQL__()) 
		expr1, expr2 = ['x.{}'.format(elem) for elem in expr1], ['y.{}'.format(elem) for elem in expr2]
		expr = expr1 + expr2
		expr = "*" if not(expr) else ", ".join(expr)
		table = "SELECT {} FROM {} {} JOIN {} {}".format(expr, first_relation, how.upper(), second_relation, on_join)
		return (self.__vdf_from_relation__("({}) VERTICAPY_JOIN".format(table), "join", "[Join]: Two relations were joined together".format(table)))
	#---#
	def kurt(self, 
			 columns: list = []):
		"""
	---------------------------------------------------------------------------
	Aggregates the vDataFrame using 'kurtosis'.

	Parameters
 	----------
 	columns: list, optional
 		List of the vcolumns names. If empty, all the numerical vcolumns will be 
 		used.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
		return self.kurtosis(columns = columns)
	#---#
	def kurtosis(self, 
				 columns: list = []):
		"""
	---------------------------------------------------------------------------
	Aggregates the vDataFrame using 'kurtosis'.

	Parameters
 	----------
 	columns: list, optional
 		List of the vcolumns names. If empty, all the numerical vcolumns will be 
 		used.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
		return (self.aggregate(func = ["kurtosis"], columns = columns))
	#---#
	def last(self, 
			 ts: str, 
			 offset: str):
		"""
	---------------------------------------------------------------------------
	Filters the vDataFrame by only keeping the last records.

	Parameters
 	----------
 	ts: str
 		TS (Time Series) vcolumn used to filter the data. The vcolumn type must be
 		date like (date, datetime, timestamp...)
 	offset: str
 		Interval offset. For example, to filter and keep only the last 6 months of
 		records, offset should be set to '6 months'.

 	Returns
 	-------
 	vDataFrame
		self

	See Also
	--------
	vDataFrame.at_time      : Filters the data at the input time.
	vDataFrame.between_time : Filters the data between two time ranges.
	vDataFrame.first        : Filters the data by only keeping the first records.
	vDataFrame.filter       : Filters the data using the input expression.
		"""
		check_types([
			("ts", ts, [str], False), 
			("offset", offset, [str], False)])
		ts = vdf_columns_names([ts], self)[0]
		query = "SELECT (MAX({}) - '{}'::interval)::varchar FROM {}".format(ts, offset, self.__genSQL__())
		self._VERTICAPY_VARIABLES_["cursor"].execute(query)
		last_date = self._VERTICAPY_VARIABLES_["cursor"].fetchone()[0]
		self.filter("{} >= '{}'".format(ts, last_date))
		return (self)
	#---#
	def load(self, 
			 offset: int = -1):
		"""
	---------------------------------------------------------------------------
	Loads a previous structure of the vDataFrame. 

	Parameters
	----------
	offset: int, optional
		offset of the saving. Example: -1 to load the last saving.

 	Returns
 	-------
 	vDataFrame
		vDataFrame of the loading.

	See Also
	--------
	vDataFrame.save : Saves the current vDataFrame structure.
		"""
		check_types([("offset", offset, [int, float], False)])
		save =  self._VERTICAPY_VARIABLES_["saving"][offset]
		vdf = {}
		exec(save, globals(), vdf)
		vdf = vdf["vdf_save"]
		vdf._VERTICAPY_VARIABLES_["cursor"] = self._VERTICAPY_VARIABLES_["cursor"]
		return (vdf)
	#---#
	def mad(self, 
			columns: list = []):
		"""
	---------------------------------------------------------------------------
	Aggregates the vDataFrame using 'mad' (Median Absolute Deviation).

	Parameters
 	----------
 	columns: list, optional
 		List of the vcolumns names. If empty, all the numerical vcolumns will be 
 		used.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
		return (self.aggregate(func = ["mad"], columns = columns))
	#---#
	def mae(self, 
			columns: list = []):
		"""
	---------------------------------------------------------------------------
	Aggregates the vDataFrame using 'mae' (Mean Absolute Error).

	Parameters
 	----------
 	columns: list, optional
 		List of the vcolumns names. If empty, all the numerical vcolumns will be 
 		used.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
		return (self.aggregate(func = ["mae"], columns = columns))
	#---#
	def max(self, 
			columns: list = []):
		"""
	---------------------------------------------------------------------------
	Aggregates the vDataFrame using 'max' (Maximum).

	Parameters
 	----------
 	columns: list, optional
 		List of the vcolumns names. If empty, all the numerical vcolumns will be 
 		used.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
		return (self.aggregate(func = ["max"], columns = columns))
	#---#
	def mean(self, 
			 columns: list = []):
		"""
	---------------------------------------------------------------------------
	Aggregates the vDataFrame using 'avg' (Average).

	Parameters
 	----------
 	columns: list, optional
 		List of the vcolumns names. If empty, all the numerical vcolumns will be 
 		used.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
		return (self.aggregate(func = ["avg"], columns = columns))
	#---#
	def median(self, 
			   columns: list = []):
		"""
	---------------------------------------------------------------------------
	Aggregates the vDataFrame using 'median'.

	Parameters
 	----------
 	columns: list, optional
 		List of the vcolumns names. If empty, all the numerical vcolumns will be 
 		used.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
		return (self.aggregate(func = ["median"], columns = columns))
	#---#
	def memory_usage(self):
		"""
	---------------------------------------------------------------------------
	Returns the vDataFrame memory usage. 

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.

	See Also
	--------
	vDataFrame.expected_store_usage : Returns the expected store usage.
		"""
		import sys
		total =  sum([sys.getsizeof(elem) for elem in self._VERTICAPY_VARIABLES_]) + sys.getsizeof(self) 
		values = {"index": ["object"], "value": [total]}
		for column in self._VERTICAPY_VARIABLES_["columns"]:
			values["index"] += [column] 
			values["value"] += [self[column].memory_usage()]
			total += self[column].memory_usage()
		values["index"] += ["total"]
		values["value"] += [total]
		return (tablesample(values = values, table_info = False))
	#---#
	def min(self, 
			columns: list = []):
		"""
	---------------------------------------------------------------------------
	Aggregates the vDataFrame using 'min' (Minimum).

	Parameters
 	----------
 	columns: list, optional
 		List of the vcolumns names. If empty, all the numerical vcolumns will be 
 		used.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
		return (self.aggregate(func = ["min"], columns = columns))
	#---#
	def normalize(self, 
				  columns: list = [], 
				  method: str = "zscore"):
		"""
	---------------------------------------------------------------------------
	Normalizes the input vcolumns using the input method.

	Parameters
 	----------
 	columns: list, optional
 		List of the vcolumns names. If empty, all the numerical vcolumns will be 
 		used.
 	method: str, optional
 		Method used to normalize.
 			zscore        : Normalization using the Z-Score (avg and std).
				(x - avg) / std
 			robust_zscore : Normalization using the Robust Z-Score (median and mad).
				(x - median) / (1.4826 * mad)
 			minmax        : Normalization using the MinMax (min and max).
				(x - min) / (max - min)

 	Returns
 	-------
 	vDataFrame
		self

	See Also
	--------
	vDataFrame.outliers    : Computes the vDataFrame Global Outliers.
	vDataFrame[].normalize : Normalizes the vcolumn. This method is more complete 
		than the vDataFrame.normalize method by allowing more parameters.
		"""
		check_types([
			("columns", columns, [list], False), 
			("method", method, ["zscore", "robust_zscore", "minmax"], True)])
		method = method.lower()
		columns_check(columns, self)
		no_cols = True if not(columns) else False
		columns = self.numcol() if not(columns) else vdf_columns_names(columns, self)
		for column in columns:
			if (self[column].isnum() and (self[column].ctype() != "boolean")):
				self[column].normalize(method = method)
			elif ((no_cols) and (self[column].ctype() == "boolean")):
				pass
			else:
				print("\u26A0 Warning: The vcolumn {} was skipped\nNormalize only accept numerical data types".format(column))
		return (self)
	#---#
	def numcol(self, 
			   exclude_columns: list = []):
		"""
	---------------------------------------------------------------------------
	Returns the vDataFrame numerical vcolumns.

	Parameters
 	----------
 	exclude_columns: list, optional
 		List of the vcolumns names to exclude from the final list. 

 	Returns
 	-------
 	List
 		List of the numerical vcolumns names. 
	
	See Also
	--------
	vDataFrame.catcol      : Returns all the vDataFrame categorical vcolumns.
	vDataFrame.get_columns : Returns all the vDataFrame vcolumns.
		"""
		columns, cols = [], self.get_columns(exclude_columns = exclude_columns)
		for column in cols:
			if self[column].isnum():
				columns += [column]
		return (columns)
	#---#
	def nunique(self, 
				columns: list = []):
		"""
	---------------------------------------------------------------------------
	Aggregates the vDataFrame using 'unique' (cardinality).

	Parameters
 	----------
 	columns: list, optional
 		List of the vcolumns names. If empty, all the vcolumns will be used.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
		return (self.aggregate(func = ["unique"], columns = columns))
	#---# 
	def outliers(self,
				 columns: list = [],
				 name: str = "distribution_outliers",
				 threshold: float = 3.0,
				 robust: bool = False):
		"""
	---------------------------------------------------------------------------
	Adds a new vcolumn labeled with 0 and 1. 1 means that the record is a global 
	outlier.

	Parameters
 	----------
 	columns: list, optional
 		List of the vcolumns names. If empty, all the numerical vcolumns will be 
 		used.
 	name: str, optional
 		Name of the new vcolumn.
 	threshold: float, optional
 		Threshold equals to the critical score.
 	robust: bool
 		If set to True, the score used will be the Robust Z-Score instead of 
 		the Z-Score.

 	Returns
 	-------
 	vDataFrame
		self

	See Also
	--------
	vDataFrame.normalize : Normalizes the input vcolumns.
		"""
		check_types([
			("columns", columns, [list], False), 
			("name", name, [str], False), 
			("threshold", threshold, [int, float], False)])
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
	#---#
	def pivot_table(self,
					columns: list,
					method: str = "count",
					of: str = "",
					max_cardinality: tuple = (20, 20),
					h: tuple = (None, None),
					show: bool = True,
					cmap: str = '',
					with_numbers: bool = True):
		"""
	---------------------------------------------------------------------------
	Draws the Pivot Table of one or two columns based on an aggregation.

	Parameters
 	----------
 	columns: list
 		List of the vcolumns names. The list must have one or two elements.
 	method: str, optional
 		The method used to aggregate the data.
 			count   : Number of elements.
 			density : Percentage of the distribution.
 			mean    : Average of the vcolumn 'of'.
 			min     : Minimum of the vcolumn 'of'.
 			max     : Maximum of the vcolumn 'of'.
 			sum     : Sum of the vcolumn 'of'.
 			q%      : q Quantile of the vcolumn 'of (ex: 50% to get the median).
 	of: str, optional
 		The vcolumn used to compute the aggregation.
 	max_cardinality: tuple, optional
 		Maximum number of distinct elements for vcolumns 1 and 2 to be used as 
 		categorical (No h will be picked or computed)
 	h: tuple, optional
 		Interval width of the vcolumns 1 and 2 bars. It is only valid if the 
 		vcolumns are numerical. Optimized h will be computed if the parameter 
 		is empty or invalid.
 	show: bool, optional
 		If set to True, the result will be drawn using Matplotlib.
 	cmap: str, optional
 		Color Map.
 	with_numbers: bool, optional
 		If set to True, no number will be displayed in the final drawing.

 	Returns
 	-------
 	vDataFrame
		self

  	See Also
 	--------
 	vDataFrame.hexbin : Draws the Hexbin Plot of 2 vcolumns based on an aggregation.
		"""
		check_types([
			("columns", columns, [list], False), 
			("method", method, [str], False), 
			("of", of, [str], False), 
			("max_cardinality", max_cardinality, [tuple], False), 
			("h", h, [tuple], False), 
			("cmap", cmap, [str], False), 
			("show", show, [bool], False), 
			("with_numbers", with_numbers, [bool], False)])
		method = method.lower()
		columns_check(columns, self, [1, 2])
		columns = vdf_columns_names(columns, self)
		if (of):
			columns_check([of], self)
			of = vdf_columns_names([of], self)[0]
		if not(cmap):
			from verticapy.plot import gen_cmap
			cmap = gen_cmap()[0]
		from verticapy.plot import pivot_table
		return (pivot_table(self, columns, method, of, h, max_cardinality, show, cmap, with_numbers))
	#---#
	def plot(self, 
		     ts: str,
			 columns: list = [], 
			 start_date: str = "",
		     end_date: str = ""):
		"""
	---------------------------------------------------------------------------
	Draws the Time Series.

	Parameters
 	----------
 	ts: str
 		TS (Time Series) vcolumn used to order the data. The vcolumn type must be
 		date like (date, datetime, timestamp...) or numerical.
 	columns: list, optional
 		List of the vcolumns names. If empty, all the numerical vcolumns will be 
 		used.
 	start_date: str, optional
 		Input Start Date. For example, time = '03-11-1993' will filter the data when 
 		'ts' is lesser than November 1993 the 3rd.
 	end_date: str, optional
 		Input End Date. For example, time = '03-11-1993' will filter the data when 
 		'ts' is greater than November 1993 the 3rd.

 	Returns
 	-------
 	vDataFrame
		self

	See Also
	--------
	vDataFrame[].plot : Draws the Time Series of one vcolumn.
		"""
		check_types([
			("columns", columns, [list], False), 
			("ts", ts, [str], False), 
			("start_date", start_date, [str], False), 
			("end_date", end_date, [str], False)])
		columns_check(columns + [ts], self)
		columns = vdf_columns_names(columns, self)
		ts = vdf_columns_names([ts], self)[0]
		from verticapy.plot import multi_ts_plot
		multi_ts_plot(self, ts, columns, start_date, end_date)	
		return (self)
	#---#
	def prod(self, 
			 columns: list = []):
		"""
	---------------------------------------------------------------------------
	Aggregates the vDataFrame using 'product'.

	Parameters
 	----------
 	columns: list, optional
 		List of the vcolumns names. If empty, all the numerical vcolumns will be 
 		used.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
		return (self.product(columns = columns))
	#---#
	def product(self, 
				columns = []):
		"""
	---------------------------------------------------------------------------
	Aggregates the vDataFrame using 'product'.

	Parameters
 	----------
 	columns: list, optional
 		List of the vcolumns names. If empty, all the numerical vcolumns will be 
 		used.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
		return (self.aggregate(func = ["prod"], columns = columns))
	#---#
	def quantile(self, 
				 q: list, 
				 columns: list = []):
		"""
	---------------------------------------------------------------------------
	Aggregates the vDataFrame using a list of 'quantiles'.

	Parameters
 	----------
 	q: list
 		List of the different quantiles. They must be numbers between 0 and 1.
 		For example [0.25, 0.75] will return Q1 and Q3.
 	columns: list, optional
 		List of the vcolumns names. If empty, all the numerical vcolumns will be 
 		used.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
		return (self.aggregate(func = ["{}%".format(float(item) * 100) for item in q], columns = columns))
	#---#
	def regexp(self, 
			   column: str,
			   pattern: str,
			   method: str = "substr", 
			   position: int = 1,
			   occurrence: int = 1,
			   replacement: str = '',
			   return_position : int = 0,
			   name: str = ""):
		"""
	---------------------------------------------------------------------------
	Computes a new vcolumn based on regular expressions. 

	Parameters
 	----------
 	column: str
 		Input vcolumn used to compute the regular expression.
 	pattern: str
 		The regular expression.
 	method: str, optional
 		Method used to compute the regular expressions.
 			count     : Returns the number times a regular expression matches 
 				each element of the input vcolumn. 
 			ilike     : Returns True if the vcolumn element contains a match 
 				for the regular expression.
 			instr     : Returns the starting or ending position in a vcolumn 
 				element where a regular expression matches. 
 			like      : Returns True if the vcolumn element matches the regular 
 				expression.
 			not_ilike : Returns True if the vcolumn element does not match the 
 				case-insensitive regular expression.
 			not_like  : Returns True if the vcolumn element does not contain a 
 				match for the regular expression.
 			replace   : Replaces all occurrences of a substring that match a 
 				regular expression with another substring.
 			substr    : Returns the substring that matches a regular expression 
 				within a vcolumn.
 	position: int, optional
 		The number of characters from the start of the string where the function 
 		should start searching for matches.
 	occurrence: int, optional
 		Controls which occurrence of a pattern match in the string to return.
 	replacement: str, optional
 		The string to replace matched substrings.
 	return_position: int, optional
 		Sets the position within the string to return.
 	name: str, optional
 		New feature name. If empty, a name will be generated.

 	Returns
 	-------
 	vDataFrame
 		self

	See Also
	--------
	vDataFrame.eval : Evaluates a customized expression.
	"""
		check_types([
			("column", column, [str], False),
			("pattern", pattern, [str], False),
			("method", method, ["count", "ilike", "instr", "like", "not_ilike", "not_like", "replace", "substr"], True), 
			("position", position, [int], False), 
			("occurrence", occurrence, [int], False),
			("replacement", replacement, [str], False),
			("return_position", return_position, [int], False)])
		columns_check([column], self)
		column = vdf_columns_names([column], self)[0]
		expr = "REGEXP_{}({}, '{}'".format(method.upper(), column, pattern.replace("'", "''"))
		if method in ("replace"):
			expr += ", '{}'".format(replacement.replace("'", "''"))
		if method in ("count", "instr", "replace", "substr"):
			expr += ", {}".format(position)
		if method in ("instr", "replace", "substr"):
			expr += ", {}".format(occurrence)
		if method in ("instr"):
			expr += ", {}".format(return_position)
		expr += ")"
		gen_name([method, column])
		return (self.eval(name = name, expr = expr))
	#---#
	def regr(self, 
			 columns: list = [],
			 method: str = "r2", 
			 cmap: str = "",
			 show: bool = True):
		"""
	---------------------------------------------------------------------------
	Computes the Regression Matrix of the vDataFrame.

	Parameters
 	----------
 	columns: list, optional
 		List of the vcolumns names. If empty, all the numerical vcolumns will be 
 		used.
 	method: str, optional
 		Method used to compute the regression matrix.
 			avgx  : Average of the independent expression in an expression pair.
 			avgy  : Average of the dependent expression in an expression pair.
 			count : Count of all rows in an expression pair.
 			alpha : Intercept of the regression line determined by a set of 
 				expression pairs.
 			r2    : Square of the correlation coefficient of a set of expression 
 				pairs.
 			beta  : Slope of the regression line, determined by a set of expression 
 				pairs.
 			sxx   : Sum of squares of the independent expression in an expression 
 				pair.
 			sxy   : Sum of products of the independent expression multiplied by the 
 				dependent expression in an expression pair.
 			syy   : Returns the sum of squares of the dependent expression in an 
 				expression pair.
 	cmap: str, optional
 		Color Map.
 	show: bool, optional
 		If set to True, the Regression Matrix will be drawn using Matplotlib.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.

	See Also
	--------
	vDataFrame.autocorr : Computes the Correlations between a vcolumn and its lags.
	vDataFrame.cov      : Computes the Covariance Matrix of the vDataFrame.
	vDataFrame.corr     : Computes the Correlation Matrix of the vDataFrame.
		"""
		check_types([
			("columns", columns, [list], False), 
			("method", method, ["avgx", "avgy", "count", "intercept", "r2", "slope", "sxx", "sxy", "syy", "beta", "alpha"], True), 
			("cmap", cmap, [str], False), 
			("show", show, [bool], False)])
		if (method == "beta"):
			method = "slope"
		elif (method == "alpha"):
			method = "intercept"
		method = "regr_{}".format(method)
		if not(columns): columns = self.numcol()
		columns_check(columns, self)
		columns = vdf_columns_names(columns, self)
		columns = vdf_columns_names(columns, self)
		n = len(columns)
		all_list, nb_precomputed = [], 0
		for i in range(0, n):
			for j in range(0, n):
				cast_i = "::int" if (self[columns[i]].ctype() == "boolean") else ""
				cast_j = "::int" if (self[columns[j]].ctype() == "boolean") else ""
				pre_comp_val = self.__get_catalog_value__(method = method, columns = [columns[i], columns[j]])
				if (pre_comp_val == None or pre_comp_val != pre_comp_val): pre_comp_val = 'NULL'
				if (pre_comp_val != "VERTICAPY_NOT_PRECOMPUTED"):
					all_list += [str(pre_comp_val)]
					nb_precomputed += 1
				else:
					all_list += ["{}({}{}, {}{})".format(method.upper(), columns[i], cast_i, columns[j], cast_j)]
		try:
			if (nb_precomputed == n * n):
				self._VERTICAPY_VARIABLES_["cursor"].execute("SELECT {}".format(", ".join(all_list)))
			else:
				self.__executeSQL__(query = "SELECT {} FROM {}".format(", ".join(all_list), self.__genSQL__()), title = "Computes the {} Matrix.".format(method.upper()))
			result = self._VERTICAPY_VARIABLES_["cursor"].fetchone()
			if (n == 1):
				return result[0]
		except:
			n = len(columns)
			result = []
			for i in range(0, n):
				for j in range(0, n):
					self.__executeSQL__(query = "SELECT {}({}{}, {}{}) FROM {}".format(method.upper(), columns[i], cast_i, columns[j], cast_j, self.__genSQL__()), title = "Computes the {} aggregation, one at a time.".format(method.upper()))
					result += [self._VERTICAPY_VARIABLES_["cursor"].fetchone()[0]]
		matrix = [[1 for i in range(0, n + 1)] for i in range(0, n + 1)]
		matrix[0] = [""] + columns
		for i in range(0, n + 1):
			matrix[i][0] = columns[i - 1]
		k = 0
		for i in range(0, n):
			for j in range(0, n):
				current = result[k]
				k += 1
				if (current == None):
					current = float('nan')
				matrix[i + 1][j + 1] = current
		if (show):
			from verticapy.plot import cmatrix
			if not(cmap):
				from verticapy.plot import gen_cmap
				cmap = gen_cmap()[0]
			if (method == "slope"):
				method_title = "Beta"
			elif (method == "intercept"):
				method_title = "Alpha"
			else:
				method_title = method
			cmatrix(matrix, columns, columns, n, n, vmax = None, vmin = None, cmap = cmap, title = "{} Matrix".format(method_title))
		values = {"index" : matrix[0][1:len(matrix[0])]}
		del(matrix[0])
		for column in matrix:
			values[column[0]] = column[1:len(column)]
		for column1 in values:
			if (column1 != "index"):
				val = {}
				for idx, column2 in enumerate(values["index"]):
					val[column2] = values[column1][idx]
				self.__update_catalog__(values = val, matrix = method, column = column1)
		return tablesample(values = values, table_info = False)
	#---#
	def rolling(self, 
				func: str, 
				column: str, 
				preceding, 
				following, 
				column2: str = "",
				name: str = "",
				by: list = [], 
				order_by = [], 
				method: str = "rows",
				rule: str = "auto"):
		"""
	---------------------------------------------------------------------------
	Adds a new vcolumn to the vDataFrame by using an advanced analytical window 
	function on one or two specific vcolumns.

	Parameters
 	----------
 	func: str
 		Function to use.
 			beta        : Beta Coefficient between 2 vcolumns
 			count       : number of non-missing elements
 			corr        : Pearson correlation between 2 vcolumns
 			cov         : covariance between 2 vcolumns
			kurtosis    : kurtosis
			jb          : Jarque Bera index 
			mae         : mean absolute error (deviation)
			max         : maximum
			mean        : average
			min         : minimum
			prod        : product
			range       : difference between the max and the min
			sem         : standard error of the mean
			skewness    : skewness
			sum         : sum
			std         : standard deviation
			var         : variance
				Other window functions could work if it is part of 
				the DB version you are using.
 	column: str
 		Input vcolumn.
 	preceding: int/str
 		First part of the moving window. With which lag/lead the window 
 		should begin. It can be an integer or an interval.
 	following: int/str
 		Second part of the moving window. With which lag/lead the window 
 		should end. It can be an integer or an interval.
 	column2: str, optional
		Second input vcolumn in case of functions using 2 parameters.
 	name: str, optional
 		Name of the new vcolumn. If empty a default name based on the other
 		parameters will be generated.
 	by: list, optional
 		vcolumns used in the partition.
 	order_by: dict / list, optional
 		List of the vcolumns used to sort the data using asc order or
 		dictionary of all the sorting methods. For example, to sort by "column1"
 		ASC and "column2" DESC, write {"column1": "asc", "column2": "desc"}
 	method: str, optional
 		Method used to compute the window.
 			rows : Uses number of leads/lags instead of time intervals
 			range: Uses time intervals instead of number of leads/lags
 	rule: str, optional
 		Rule used to compute the window.
 			auto   : The 'preceding' parameter will correspond to a past event and 
				the parameter 'following' to a future event.
 			past   : Both parameters 'preceding' and following will consider
				past events.
 			future : Both parameters 'preceding' and following will consider
				future events.

 	Returns
 	-------
 	vDataFrame
		self

	See Also
	--------
	vDataFrame.eval     : Evaluates a customized expression.
	vDataFrame.analytic : Adds a new vcolumn to the vDataFrame by using an advanced 
		analytical function on a specific vcolumn.
		"""
		check_types([
			("name", name, [str], False), 
			("func", func, [str], False), 
			("column", column, [str], False), 
			("column2", column2, [str], False), 
			("preceding", preceding, [str, int, float], False), 
			("following", following, [str, int, float], False), 
			("by", by, [list], False), 
			("order_by", order_by, [list, dict], False), 
			("method", method, ["rows", "range"], True), 
			("rule", rule.lower(), ["auto", "past", "future"], True)])
		method = method.lower()
		columns_check([column] + by + [elem for elem in order_by], self)
		rule = rule.lower()
		if not(name): name = "moving_{}_{}".format(gen_name([func, column, column2, preceding, following]), rule)
		if (rule == "past"):
			rule_p, rule_f = "PRECEDING", "PRECEDING"
		elif (rule == "future"):
			rule_p, rule_f = "FOLLOWING", "FOLLOWING"
		else:
			rule_p, rule_f = "PRECEDING", "FOLLOWING"
		column = vdf_columns_names([column], self)[0]
		by = "" if not(by) else "PARTITION BY " + ", ".join(vdf_columns_names(by, self))
		order_by = " ORDER BY {}".format(column) if not(order_by) else sort_str(order_by, self)
		func = str_function(func.lower())
		if (method == "rows"):
			preceding = "{}".format(preceding) if (str(preceding).upper() != "UNBOUNDED") else "UNBOUNDED"
			following = "{}".format(following) if (str(following).upper() != "UNBOUNDED") else "UNBOUNDED"
		else:
			preceding = "'{}'".format(preceding) if (str(preceding).upper() != "UNBOUNDED") else "UNBOUNDED"
			following = "'{}'".format(following) if (str(following).upper() != "UNBOUNDED") else "UNBOUNDED"
		preceding, following = "{} {}".format(preceding, rule_p), "{} {}".format(following, rule_f)
		windows_frame = " OVER ({}{} {} BETWEEN {} AND {})".format(by, order_by, method.upper(), preceding, following)
		all_cols = [elem.replace('"', '').lower() for elem in self.get_columns()]
		if (func in ("kurtosis", "skewness", "mae", "prod", "jb")):
			if (func in ("skewness", "kurtosis", "mae", "jb")):
				mean_name = "{}_mean_{}".format(column.replace('"', ''), name).lower()
				std_name = "{}_std_{}".format(column.replace('"', ''), name).lower()
				count_name = "{}_count_{}".format(column.replace('"', ''), name).lower()
				if (mean_name not in all_cols):
					self.eval(mean_name, "AVG({}){}".format(column, windows_frame))
				if (std_name not in all_cols) and (func != "mae"):
					self.eval(std_name, "STDDEV({}){}".format(column, windows_frame))
				if (count_name not in all_cols) and (func != "mae"):
					self.eval(count_name, "COUNT({}){}".format(column, windows_frame))
				if (func == "kurtosis"):
					expr = "AVG(POWER(({} - {}) / NULLIFZERO({}), 4))# * POWER({}, 2) * ({} + 1) / NULLIFZERO(({} - 1) * ({} - 2) * ({} - 3)) - 3 * POWER({} - 1, 2) / NULLIFZERO(({} - 2) * ({} - 3))".format(
						column, mean_name, std_name, count_name, count_name, count_name, count_name, count_name, count_name, count_name, count_name)
				elif (func == "skewness"):
					expr = "AVG(POWER(({} - {}) / NULLIFZERO({}), 3))# * POWER({}, 2) / NULLIFZERO(({} - 1) * ({} - 2))".format(
						column, mean_name, std_name, count_name, count_name, count_name)
				elif (func == "jb"):
					expr = "{} / 6 * (POWER(AVG(POWER(({} - {}) / NULLIFZERO({}), 3))# * POWER({}, 2) / NULLIFZERO(({} - 1) * ({} - 2)), 2) + POWER(AVG(POWER(({} - {}) / NULLIFZERO({}), 4))# * POWER({}, 2) * ({} + 1) / NULLIFZERO(({} - 1) * ({} - 2) * ({} - 3)) - 3 * POWER({} - 1, 2) / NULLIFZERO(({} - 2) * ({} - 3)), 2) / 4)".format(
						count_name, column, mean_name, std_name, count_name, count_name, count_name, column, mean_name, std_name, count_name, count_name, count_name, count_name, count_name, count_name, count_name, count_name)
				elif (func == "mae"):
					expr = "AVG(ABS({} - {}))#".format(column, mean_name)
			else:
				expr = "DECODE(ABS(MOD(SUM(CASE WHEN {} < 0 THEN 1 ELSE 0 END)#, 2)), 0, 1, -1) * POWER(10, SUM(LOG(ABS({})))#)".format(column, column)
		elif func in ("corr", "cov", "beta"):
			columns_check([column2], self)
			column2 = vdf_columns_names([column2], self)[0]
			mean_name  = "{}_mean_{}".format(column.replace('"', ''), name).lower()
			mean_name2 = "{}_mean_{}".format(column2.replace('"', ''), name).lower()
			if (mean_name not in all_cols):
				self.eval(mean_name, "AVG({}){}".format(column, windows_frame))
			if (mean_name2 not in all_cols):
				self.eval(mean_name2, "AVG({}){}".format(column2, windows_frame))
			if (func == "corr"):
				den = " / (STDDEV({})# * STDDEV({})#)".format(column, column2)
			elif (func == "beta"):
				den = " / (VARIANCE({})#)".format(column2)
			else:
				den = ""
			expr = "AVG(({} - {}) * ({} - {}))#{}".format(column, mean_name, column2, mean_name2, den)
		elif (func == "range"):
			expr = "MAX({})# - MIN({})#".format(column, column) 
		elif (func == "sem"):
			expr = "STDDEV({})# / SQRT(COUNT({})#)".format(column, column) 
		else:
			expr = "{}({})#".format(func.upper(), column) 
		expr = expr.replace('#', windows_frame)
		return (self.eval(name = name, expr = expr))
	#---#
	def sample(self, 
			   x: float):
		"""
	---------------------------------------------------------------------------
	Downsamples the vDataFrame by filtering using a random vcolumn.
	Be Careful: the result will change for each SQL code generation as the 
	random numbers are not memorized. You can use this method to downsample 
	before saving the final relation to the DataBase.

	Parameters
 	----------
 	x: float
 		The sample size. For example it has to be equal to 0.33 to downsample to 
 		approximatively 33% of the relation.

 	Returns
 	-------
 	vDataFrame
		self
		"""
		check_types([("x", x, [int, float], False)])
		if ((x <= 0) or (x >= 1)):
			raise ValueError("Parameter 'x' must be between 0 and 1")
		name = "__verticapy_random_{}__".format(random.randint(0, 10000000))
		self.eval(name, "RANDOM()")
		self.filter("{} < {}".format(name, x), print_info = False)
		self[name].drop()
		return (self)
	#---#
	def save(self):
		"""
	---------------------------------------------------------------------------
	Saves the current structure of the vDataFrame. 
	This function is useful to load a previous transformation.

 	Returns
 	-------
 	vDataFrame
		self

	See Also
	--------
	vDataFrame.load : Loads a saving.
		"""
		save = 'vdf_save = vDataFrame("", empty = True)'
		save += '\nvdf_save._VERTICAPY_VARIABLES_["dsn"] = \'{}\''.format(self._VERTICAPY_VARIABLES_["dsn"].replace("'", "\\'"))
		save += '\nvdf_save._VERTICAPY_VARIABLES_["input_relation"] = \'{}\''.format(self._VERTICAPY_VARIABLES_["input_relation"].replace("'", "\\'"))
		save += '\nvdf_save._VERTICAPY_VARIABLES_["main_relation"] = \'{}\''.format(self._VERTICAPY_VARIABLES_["main_relation"].replace("'", "\\'"))
		save += '\nvdf_save._VERTICAPY_VARIABLES_["schema"] = \'{}\''.format(self._VERTICAPY_VARIABLES_["schema"].replace("'", "\\'"))
		save += '\nvdf_save._VERTICAPY_VARIABLES_["columns"] = {}'.format(self._VERTICAPY_VARIABLES_["columns"])
		save += '\nvdf_save._VERTICAPY_VARIABLES_["exclude_columns"] = {}'.format(self._VERTICAPY_VARIABLES_["exclude_columns"])
		save += '\nvdf_save._VERTICAPY_VARIABLES_["where"] = {}'.format(self._VERTICAPY_VARIABLES_["where"])
		save += '\nvdf_save._VERTICAPY_VARIABLES_["query_on"] = {}'.format(self._VERTICAPY_VARIABLES_["query_on"])
		save += '\nvdf_save._VERTICAPY_VARIABLES_["time_on"] = {}'.format(self._VERTICAPY_VARIABLES_["time_on"])
		save += '\nvdf_save._VERTICAPY_VARIABLES_["order_by"] = {}'.format(self._VERTICAPY_VARIABLES_["order_by"])
		save += '\nvdf_save._VERTICAPY_VARIABLES_["history"] = {}'.format(self._VERTICAPY_VARIABLES_["history"])
		save += '\nvdf_save._VERTICAPY_VARIABLES_["saving"] = {}'.format(self._VERTICAPY_VARIABLES_["saving"])
		save += '\nvdf_save._VERTICAPY_VARIABLES_["schema_writing"] = \'{}\''.format(self._VERTICAPY_VARIABLES_["schema_writing"].replace("'", "\\'"))
		for column in self._VERTICAPY_VARIABLES_["columns"]:
			save += '\nsave_vColumn = vColumn(\'{}\', parent = vdf_save, transformations = {}, catalog = {})'.format(column.replace("'", "\\'"), self[column].transformations, self[column].catalog)
			save += '\nsetattr(vdf_save, \'{}\', save_vColumn)'.format(column.replace("'", "\\'"))
			save += '\nsetattr(vdf_save, \'{}\', save_vColumn)'.format(column[1:-1].replace("'", "\\'"))
		self._VERTICAPY_VARIABLES_["saving"] += [save]
		return (self)
	#---#
	def scatter(self,
			  	columns: list,
			  	catcol: str = "",
			  	max_cardinality: int = 6,
			  	cat_priority: list = [],
			  	with_others: bool = True,
			  	max_nb_points: int = 20000,
			  	bbox: list = [],
			 	img: str = ""):
		"""
	---------------------------------------------------------------------------
	Draws the Scatter Plot of the input vcolumns.

	Parameters
 	----------
 	columns: list
 		List of the vcolumns names. The list must have two or three elements.
 	catcol: str, optional
 		Categorical vcolumn used to label the data.
 	max_cardinality: int, optional
 		Maximum number of distinct elements for 'catcol' to be used as 
 		categorical. The less frequent elements will be gathered together to 
 		create a new category: 'Others'.
 	cat_priority: list, optional
 		List of the different categories to consider when labeling the data using
 		the vcolumn 'catcol'. The other categories will be filtered.
 	with_others: bool, optional
 		If set to false and the cardinality of the vcolumn 'catcol' is too big then 
 		the less frequent element will not be merged to another category and they 
 		will not be drawn.
 	max_nb_points: int, optional
 		Maximum number of points to display.
 	bbox: list, optional
 		List of 4 elements to delimit the boundaries of the final Plot. 
 		It must be similar the following list: [xmin, xmax, ymin, ymax]
 	img: str, optional
 		Path to the image to display as background.

 	Returns
 	-------
 	vDataFrame
		self

 	See Also
 	--------
 	vDataFrame.bubble      : Draws the Bubble Plot of the input vcolumns.
 	vDataFrame.pivot_table : Draws the Pivot Table of vcolumns based on an aggregation.
		"""
		check_types([
			("columns", columns, [list], False), 
			("catcol", catcol, [str], False), 
			("max_cardinality", max_cardinality, [int, float], False), 
			("cat_priority", cat_priority, [list], False), 
			("with_others", with_others, [bool], False), 
			("max_nb_points", max_nb_points, [int, float], False),
			("img", img, [str], False),
			("bbox", bbox, [list], False)])
		columns_check(columns, self, [2, 3])
		columns = vdf_columns_names(columns, self)
		if (catcol):
			columns_check([catcol], self)
			catcol = vdf_columns_names([catcol], self) 
		else:
			catcol = []
		if (len(columns) == 2):
			from verticapy.plot import scatter2D
			scatter2D(self, columns + catcol, max_cardinality, cat_priority, with_others, max_nb_points, bbox, img)	
		else:
			from verticapy.plot import scatter3D
			scatter3D(self, columns + catcol, max_cardinality, cat_priority, with_others, max_nb_points)
		return (self)
	#---#
	def scatter_matrix(self, 
					   columns: list = []):
		"""
	---------------------------------------------------------------------------
	Draws the Scatter Matrix of the vDataFrame.

	Parameters
 	----------
 	columns: list, optional
 		List of the vcolumns names. If empty, all the numerical vcolumns will be 
 		used.

 	Returns
 	-------
 	vDataFrame
		self

 	See Also
 	--------
 	vDataFrame.scatter : Draws the Scatter Plot of the input vcolumns.
		"""
		check_types([("columns", columns, [list], False)])
		columns_check(columns, self)
		columns = vdf_columns_names(columns, self)
		from verticapy.plot import scatter_matrix
		scatter_matrix(self, columns)
		return (self)
	#---#
	def search(self,  
			   conditions = "",
			   usecols: list = [],
			   expr: list = [],
			   order_by = []):
		"""
	---------------------------------------------------------------------------
	Searches the elements which matches with the input conditions.
	
	Parameters
	----------
	conditions: str / list, optional
 		Filters of the search. It can be a list of conditions or an expression.
 	usecols: list, optional
 		vcolumns to select from the final vDataFrame relation. If empty, all the
 		vcolumns will be selected.
 	expr: list, optional
 		List of customized expressions. It must be pure SQL. For example, it is
 		possible to write 'column1 * column2 AS my_name'.
	order_by: dict / list, optional
 		List of the vcolumns used to sort the data using asc order or
 		dictionary of all the sorting methods. For example, to sort by "column1"
 		ASC and "column2" DESC, write {"column1": "asc", "column2": "desc"}

 	Returns
 	-------
 	vDataFrame
 		vDataFrame of the search

	See Also
	--------
	vDataFrame.filter : Filters the vDataFrame using the input expressions.
	vDataFrame.select : Returns a copy of the vDataFrame with only the selected vcolumns.
		"""
		check_types([
			("conditions", conditions, [str, list], False), 
			("usecols", usecols, [list], False),
			("expr", expr, [list], False),
			("order_by", order_by, [dict, list], False)])
		columns_check(usecols, self)
		usecols = self.get_columns() if not(usecols) else vdf_columns_names(usecols, self)
		if (type(conditions) == list): conditions = " AND ".join(['({})'.format(elem) for elem in conditions])
		conditions = " WHERE {}".format(conditions) if conditions else ""
		order_by = sort_str(order_by, self)
		all_cols = ", ".join(usecols + expr)
		conditions +=  order_by
		table = "(SELECT {} FROM {}{}) x".format(all_cols, self.__genSQL__(), conditions)
		return (self.__vdf_from_relation__(table, "search", ""))
	#---#
	def select(self, 
			   columns: list,
			   check: bool = True):
		"""
	---------------------------------------------------------------------------
	Returns a copy of the vDataFrame with only the selected vcolumns.

	Parameters
 	----------
 	columns: list
 		List of the vcolumns to select.
 	check: bool, optional
 		Do a columns checking before the computation. Set it to False to do
 		customized selection.

 	Returns
 	-------
 	vDataFrame
 		object with only the selected columns.

 	See Also
	--------
	vDataFrame.search : Searches the elements which matches with the input conditions.
		"""
		check_types([("columns", columns, [list], False)])
		if (check):
			columns_check(columns, self)
			columns = vdf_columns_names(columns, self)
		table = "(SELECT {} FROM {}) VERTICAPY_SUBTABLE".format(", ".join(columns), self.__genSQL__())
		return (self.__vdf_from_relation__(table, self._VERTICAPY_VARIABLES_["input_relation"], ""))
	#---#
	def sem(self, 
			columns: list = []):
		"""
	---------------------------------------------------------------------------
	Aggregates the vDataFrame using 'sem' (Standard Error of the Mean).

	Parameters
 	----------
 	columns: list, optional
 		List of the vcolumns names. If empty, all the numerical vcolumns will be 
 		used.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
		return (self.aggregate(func = ["sem"], columns = columns))
	#---#
	def sessionize(self,
				   ts: str,
				   by: list = [],
				   session_threshold: str = "30 minutes",
				   name: str = "session_id"):
		"""
	---------------------------------------------------------------------------
	Adds a new vcolumn to the vDataFrame which will correspond to sessions 
	(user activity during a specific time). A session ends when ts - lag(ts) 
	is greater than a specific threshold.

	Parameters
 	----------
 	ts: str
 		vcolumn used as timeline. It will be used to order the data. It can be
 		a numerical or type date like (date, datetime, timestamp...) vcolumn.
 	by: list, optional
 		vcolumns used in the partition.
 	session_threshold: str, optional
 		This parameter is the threshold which will determine the end of the 
 		session. For example, if it is set to '10 minutes' the session ends
 		after 10 minutes of inactivity.
 	name: str, optional
 		The session name.

 	Returns
 	-------
 	vDataFrame
		self

	See Also
 	--------
 	vDataFrame.analytic : Adds a new vcolumn to the vDataFrame by using an advanced 
		analytical function on a specific vcolumn.
		"""
		check_types([
			("ts", ts, [str], False), 
			("by", by, [list], False), 
			("session_threshold", session_threshold, [str], False), 
			("name", name, [str], False)])
		columns_check(by + [ts], self)
		by = vdf_columns_names(by, self)
		ts = vdf_columns_names([ts], self)[0]
		partition = "PARTITION BY {}".format(", ".join(by)) if (by) else ""
		expr = "CONDITIONAL_TRUE_EVENT({} - LAG({}) > '{}') OVER ({} ORDER BY {})".format(ts, ts, session_threshold, partition, ts)
		return (self.eval(name = name, expr = expr))
	#---#
	def set_cursor(self, 
				   cursor):
		"""
	---------------------------------------------------------------------------
	Sets a new DB cursor. It can be very usefull if the connection to the DB is 
	lost.

	Parameters
	----------
	cursor: DBcursor
		New cursor.

 	Returns
 	-------
 	vDataFrame
		self

 	See Also
 	--------
 	vDataFrame.set_dsn: Sets a new DSN.
 	vDataFrame.dsn_restart: Restarts a new connection using the vDataFrame DSN.
		"""
		check_cursor(cursor)
		cursor.execute("SELECT 1;")
		self._VERTICAPY_VARIABLES_["cursor"] = cursor
		return (self)
	#---#
	def set_dsn(self, 
				dsn: str):
		"""
	---------------------------------------------------------------------------
	Sets a new DSN. This DSN can be used to restart a DB connection.

	Parameters
	----------
	dsn: str
		New DSN name.

 	Returns
 	-------
 	vDataFrame
		self

 	See Also
 	--------
 	vDataFrame.set_cursor  : Sets a new DB cursor.
 	vDataFrame.dsn_restart : Restarts a new connection using the vDataFrame DSN.
		"""
		check_types([("dsn", dsn, [str], False)])
		self._VERTICAPY_VARIABLES_["dsn"] = dsn
		return (self)
	#---#
	def set_schema_writing(self, 
						   schema_writing: str):
		"""
	---------------------------------------------------------------------------
	Sets a new writing schema, this schema will be used to create temporary table
	if it is necessary.

	Parameters
	----------
	schema_writing: str
		New schema writing name.

 	Returns
 	-------
 	vDataFrame
		self

 	See Also
 	--------
 	vDataFrame.set_cursor : Sets a new DB cursor.
 	vDataFrame.set_dsn    : Sets a new DB DSN.
		"""
		check_types([("schema_writing", schema_writing, [str], False)])
		self._VERTICAPY_VARIABLES_["cursor"].execute("SELECT table_schema FROM columns WHERE table_schema = '{}'".format(schema_writing.replace("'", "''")))
		if (self._VERTICAPY_VARIABLES_["cursor"].fetchone()) or (schema_writing.lower() not in ['"v_temp_schema"', 'v_temp_schema', '"public"', 'public']):
			self._VERTICAPY_VARIABLES_["schema_writing"] = schema_writing
		else:
			raise ValueError("The schema '{}' doesn't exist or is not accessible.\nThe attribute of the vDataFrame 'schema_writing' did not change.".format(schema_writing))
		return (self)
	#---#
	def score(self, 
			  y_true: str, 
			  y_score: str,
			  method: str):
		"""
	---------------------------------------------------------------------------
	Computes the score using the input columns and the input method.

	Parameters
	----------
	y_true: str
		Response column.
	y_score: str
		Prediction.
	method: str
		The method used to compute the score.
			--- For Classification ---
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
			--- For Regression ---
			max    : Max Error
			mae    : Mean Absolute Error
			median : Median Absolute Error
			mse    : Mean Squared Error
			msle   : Mean Squared Log Error
			r2     : R squared coefficient
			var    : Explained Variance  
			--- Plots ---
			roc  : ROC Curve
			prc  : PRC Curve
			lift : Lift Chart

	Returns
	-------
	float / tablesample
 		score / tablesample of the curve

 	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
		from verticapy.learn.plot import roc_curve, prc_curve, lift_chart
		check_types([
			("y_true", y_true, [str], False), 
			("y_score", y_score, [str], False),
			("method", method, [str], False)])
		columns_check([y_true, y_score], self)
		if (method in ("r2", "rsquared")):
			from verticapy.learn.metrics import r2_score
			return (r2_score(y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]))
		elif (method in ("mae", "mean_absolute_error")):
			from verticapy.learn.metrics import mean_absolute_error
			return (mean_absolute_error(y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]))
		elif (method in ("mse", "mean_squared_error")):
			from verticapy.learn.metrics import mean_squared_error
			return (mean_squared_error(y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]))
		elif (method in ("msle", "mean_squared_log_error")):
			from verticapy.learn.metrics import mean_squared_log_error
			return (mean_squared_log_error(y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]))
		elif (method in ("max", "max_error")):
			from verticapy.learn.metrics import max_error
			return (max_error(y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]))
		elif (method in ("median", "median_absolute_error")):
			from verticapy.learn.metrics import median_absolute_error
			return (median_absolute_error(y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]))
		elif (method in ("var", "explained_variance")):
			from verticapy.learn.metrics import explained_variance
			return (explained_variance(y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]))
		elif (method in ("accuracy", "acc")):
			from verticapy.learn.metrics import accuracy_score
			return (accuracy_score(y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"], pos_label = None))
		elif (method == "auc"):
			from verticapy.learn.metrics import auc
			return auc(y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"])
		elif (method == "prc_auc"):
			from verticapy.learn.metrics import prc_auc
			return prc_auc(y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"])
		elif (method in ("best_cutoff", "best_threshold")):
			from verticapy.learn.metrics import roc_curve
			return (roc_curve(y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"], best_threshold = True))
		elif (method in ("recall", "tpr")):
			from verticapy.learn.metrics import recall_score
			return (recall_score(y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]))
		elif (method in ("precision", "ppv")):
			from verticapy.learn.metrics import precision_score
			return (precision_score(y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]))
		elif (method in ("specificity", "tnr")):
			from verticapy.learn.metrics import specificity_score
			return (specificity_score(y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]))
		elif (method in ("negative_predictive_value", "npv")):
			from verticapy.learn.metrics import precision_score
			return (precision_score(y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]))
		elif (method in ("log_loss", "logloss")):
			from verticapy.learn.metrics import log_loss
			return (log_loss(y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]))
		elif (method == "f1"):
			from verticapy.learn.metrics import f1_score
			return (f1_score(y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]))
		elif (method == "mcc"):
			from verticapy.learn.metrics import matthews_corrcoef
			return (matthews_corrcoef(y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]))
		elif (method in ("bm", "informedness")):
			from verticapy.learn.metrics import informedness
			return (informedness(y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]))
		elif (method in ("mk", "markedness")):
			from verticapy.learn.metrics import markedness
			return (markedness(y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]))
		elif (method in ("csi", "critical_success_index")):
			from verticapy.learn.metrics import critical_success_index
			return (critical_success_index(y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]))
		elif (method in ("roc_curve", "roc")):
			from verticapy.learn.plot import roc_curve
			return (roc_curve(y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]))
		elif (method in ("prc_curve", "prc")):
			from verticapy.learn.plot import prc_curve
			return (prc_curve(y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]))
		elif (method in ("lift_chart", "lift")):
			from verticapy.learn.plot import lift_chart
			return (lift_chart(y_true, y_score, self.__genSQL__(), self._VERTICAPY_VARIABLES_["cursor"]))
		else:
			raise ValueError("The parameter 'method' must be in roc|prc|lift|accuracy|auc|prc_auc|best_cutoff|recall|precision|log_loss|negative_predictive_value|specificity|mcc|informedness|markedness|critical_success_index|r2|mae|mse|msle|max|median|var")
	#---#
	def shape(self):
		"""
	---------------------------------------------------------------------------
	Returns the number of rows and columns of the vDataFrame.

 	Returns
 	-------
 	tuple
 		(number of lines, number of columns)
		"""
		m = len(self.get_columns())
		pre_comp = self.__get_catalog_value__("VERTICAPY_COUNT")
		if (pre_comp != "VERTICAPY_NOT_PRECOMPUTED"):
			return (pre_comp, m)
		query = "SELECT COUNT(*) FROM {} LIMIT 1".format(self.__genSQL__())
		self.__executeSQL__(query, title = "Computes the total number of elements (COUNT(*))")
		self._VERTICAPY_VARIABLES_["count"] = self._VERTICAPY_VARIABLES_["cursor"].fetchone()[0]
		return (self._VERTICAPY_VARIABLES_["count"], m)
	#---#
	def skew(self, 
			 columns: list = []):
		"""
	---------------------------------------------------------------------------
	Aggregates the vDataFrame using 'skewness'.

	Parameters
 	----------
 	columns: list, optional
 		List of the vcolumns names. If empty, all the numerical vcolumns will be 
 		used.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
		return self.skewness(columns = columns)
	#---#
	def skewness(self, 
				 columns: list = []):
		"""
	---------------------------------------------------------------------------
	Aggregates the vDataFrame using 'skewness'.

	Parameters
 	----------
 	columns: list, optional
 		List of the vcolumns names. If empty, all the numerical vcolumns will be 
 		used.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
		return (self.aggregate(func = ["skewness"], columns = columns))
	#---#
	def sort(self, 
			 columns):
		"""
	---------------------------------------------------------------------------
	Sorts the vDataFrame using the input vcolumns.

	Parameters
 	----------
 	columns: dict / list
 		List of the vcolumns used to sort the data using asc order or
 		dictionary of all the sorting methods. For example, to sort by "column1"
 		ASC and "column2" DESC, write {"column1": "asc", "column2": "desc"}

 	Returns
 	-------
 	vDataFrame
		self

	See Also
	--------
	vDataFrame.append  : Merges the vDataFrame with another relation.
	vDataFrame.groupby : Aggregates the vDataFrame.
	vDataFrame.join    : Joins the vDataFrame with another relation.
		"""
		check_types([("columns", columns, [dict, list], False)])
		columns_check([elem for elem in columns], self)
		max_pos = 0
		for column in self._VERTICAPY_VARIABLES_["columns"]:
			max_pos = max(max_pos, len(self[column].transformations) - 1)
		self._VERTICAPY_VARIABLES_["order_by"][max_pos] = sort_str(columns, self)
		return (self)
	#---#
	def sql_on_off(self):
		"""
	---------------------------------------------------------------------------
	Displays each query generated by the vDataFrame.

 	Returns
 	-------
 	vDataFrame
		self
	
	See Also
	--------
	vDataFrame.time_on_off : Displays the time took by each query.
		"""
		self._VERTICAPY_VARIABLES_["query_on"] = not(self._VERTICAPY_VARIABLES_["query_on"])
		return (self)
	#---#
	def statistics(self, 
				   columns: list = []):
		"""
	---------------------------------------------------------------------------
	Aggregates the vDataFrame using multiple statistical aggregations: 
	kurtosis, skewness, min, max...

	Parameters
 	----------
 	columns: list, optional
 		List of the vcolumns names. If empty, all the numerical vcolumns will be 
 		used.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
		return (self.aggregate(func = ["skewness", "kurtosis", "count", "avg", "stddev", "min", "10%", "25%", "median", "75%", "90%", "max"], columns = columns))
	#---#
	def std(self, 
			columns = []):
		"""
	---------------------------------------------------------------------------
	Aggregates the vDataFrame using 'std' (Standard Deviation).

	Parameters
 	----------
 	columns: list, optional
 		List of the vcolumns names. If empty, all the numerical vcolumns will be 
 		used.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
		return (self.aggregate(func = ["stddev"], columns = columns))
	#---#
	def sum(self, 
			columns = []):
		"""
	---------------------------------------------------------------------------
	Aggregates the vDataFrame using 'sum'.

	Parameters
 	----------
 	columns: list, optional
 		List of the vcolumns names. If empty, all the numerical vcolumns will be 
 		used.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
		return (self.aggregate(func = ["sum"], columns = columns))
	#---#
	def tail(self, 
			 limit: int = 5, 
			 offset: int = 0):
		"""
	---------------------------------------------------------------------------
	Returns a part of the vDataFrame tail (delimited by an offset and a limit).

	Parameters
 	----------
 	limit: int, optional
 		Number of elements to display.
 	offset: int, optional
 		Number of elements to skip.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.

	See Also
	--------
	vDataFrame.head : Returns the vDataFrame head.
		"""
		check_types([
			("limit", limit, [int, float], False), 
			("offset", offset, [int, float], False)])
		columns = self.get_columns()
		all_columns = []
		for column in columns:
			all_columns += ["{} AS {}".format(convert_special_type(self[column].category(), True, column), column)]
		query_on, time_on, title = self._VERTICAPY_VARIABLES_["query_on"], self._VERTICAPY_VARIABLES_["time_on"], "Reads the final relation using a limit of {} and an offset of {}.".format(limit, offset)
		tail = to_tablesample("SELECT {} FROM {} LIMIT {} OFFSET {}".format(", ".join(all_columns), self.__genSQL__(), limit, offset), self._VERTICAPY_VARIABLES_["cursor"], query_on = query_on, time_on = time_on, title = title)
		tail.count = self.shape()[0]
		tail.offset = offset
		tail.name = self._VERTICAPY_VARIABLES_["input_relation"]
		for column in tail.values:
			tail.dtype[column] = self[column].ctype()
		return (tail)
	#---#
	def testdf(self, 
			   ts: str, 
			   column: str, 
			   p: int = 1, 
			   alpha: float = 0.05,
			   by: list = [],
			   print_info: bool = True):
		"""
	---------------------------------------------------------------------------
	Dickey Fuller test (Time Series stationarity).

	Parameters
 	----------
 	ts: str
 		vcolumn used as timeline. It will be used to order the data. It can be
 		a numerical or type date like (date, datetime, timestamp...) vcolumn.
 	column: str
 		Input vcolumn to test.
 	p: int, optional
 		Number of lags to consider in the test.
 	alpha: float, optional
 		Significance Level. Probability to accept H0.
 	by: list, optional
 		vcolumns used in the partition.
 	print_info: bool, optional
 		If set to True, displays all the test information.

 	Returns
 	-------
 	tuple
 		[0] result of the test : True if H0 was rejected
 		[1] gamma + 1          : gamma is the the Linear Regression lag1 coefficient.
 		[2] p_value            : p value of the coefficient.

	See Also
	--------
	vDataFrame.testdw : Durbin Watson test.
	vDataFrame.testjb : Jarque Bera test.
	vDataFrame.testmk : Mann Kendall test.
		"""
		check_types([
			("ts", ts, [str], False), 
			("column", column, [str], False),
			("p", p, [int, float], False),
			("by", by, [list], False),
			("alpha", alpha, [int, float], False),
			("print_info", print_info, [bool], False)])
		columns_check([ts, column] + by, self)
		ts = vdf_columns_names([ts], self)[0]
		column = vdf_columns_names([column], self)[0]
		by = vdf_columns_names(by, self)
		from verticapy.learn.linear_model import LinearRegression
		schema = self._VERTICAPY_VARIABLES_["schema_writing"]
		if not(schema): schema = "public" 
		name = "{}.VERTICAPY_TEMP_MODEL_LINEAR_REGRESSION_{}".format(schema, gen_name([column]).upper())
		relation_name = "{}.VERTICAPY_TEMP_MODEL_LINEAR_REGRESSION_VIEW_{}".format(schema, gen_name([column]).upper())
		try:
			self._VERTICAPY_VARIABLES_["cursor"].execute("DROP MODEL IF EXISTS {}".format(name))
			self._VERTICAPY_VARIABLES_["cursor"].execute("DROP VIEW IF EXISTS {}".format(relation_name))
		except:
			pass
		lag = ["LAG({}, {}) OVER ({}ORDER BY {})".format(column, i, "PARTITION BY {}".format(", ".join(by)) if (by) else "", ts) for i in range(0, p + 1)]
		lag[0] = "{} - {} AS delta".format(lag[0], lag[1])
		for i in range(1, len(lag)):
			lag[i] += " AS lag{}".format(i)
		query = "CREATE VIEW {} AS SELECT {}, {} AS ts FROM {}".format(relation_name, ", ".join(lag), "TIMESTAMPDIFF(SECOND, {}, MIN({}) OVER ())".format(ts, ts) if self[ts].isdate() else ts, self.__genSQL__())
		self._VERTICAPY_VARIABLES_["cursor"].execute(query)
		model = LinearRegression(name, self._VERTICAPY_VARIABLES_["cursor"])
		model.fit(relation_name, ["lag{}".format(i) for i in range(1, p + 1)] + ["ts"], "delta")
		coef = model.coef.transpose()
		gamma = coef.values["lag1"][0]
		p_value = coef.values["lag1"][3]
		self._VERTICAPY_VARIABLES_["cursor"].execute("DROP MODEL IF EXISTS {}".format(name))
		self._VERTICAPY_VARIABLES_["cursor"].execute("DROP VIEW IF EXISTS {}".format(relation_name))
		if (print_info):
			print("We test the following hypothesis:\n(H0) The time series {} = f({}) is non-stationary\n(H1) The time series {} = f({}) is stationary".format(column, ts, column, ts))
		if (p_value < alpha) and (-1 < gamma + 1 < 1):
			if (print_info):
				print("\U0001F44D - The Process might be stationary (time independent structure)")
			result = True
		else:
			if (print_info):
				print("\U0001F44E - The Process might be non-stationary (time dependent structure)")
			result = False
		if (print_info):
			print("gamma + 1 = {}\np_value = {}".format(gamma + 1, p_value))
		return (result, gamma + 1, p_value)
	#---#
	def testdw(self, 
			   ts: str,
			   response: str, 
			   predictors: list, 
			   by: list = [],
			   print_info: bool = True):
		"""
	---------------------------------------------------------------------------
	Durbin Watson test (residuals autocorrelation).

	Parameters
 	----------
 	ts: str
 		vcolumn used as timeline. It will be used to order the data. It can be
 		a numerical or type date like (date, datetime, timestamp...) vcolumn.
 	response: str
 		Input vcolumn used as response.
 	predictors: list
 		Input vcolumns used as predictors.
 	by: list, optional
 		vcolumns used in the partition.
 	print_info: bool, optional
 		If set to true, displays all the test information.

 	Returns
 	-------
 	tuple
 		[0] result of the test : True if H0 was rejected
 		[1]                  d : Durbin Watson index (a float between 0 and 4).

	See Also
	--------
	vDataFrame.testdf : Dickey Fuller test.
	vDataFrame.testjb : Jarque Bera test.
	vDataFrame.testmk : Mann Kendall test.
		"""
		check_types([
			("ts", ts, [str], False), 
			("response", response, [str], False),
			("predictors", predictors, [list], False),
			("by", by, [list], False),
			("print_info", print_info, [bool], False)])
		columns_check(predictors + [response] + [ts] + by, self)
		response = vdf_columns_names([response], self)[0]
		ts = vdf_columns_names([ts], self)[0]
		predictors = vdf_columns_names(predictors, self)
		by = vdf_columns_names(by, self)
		from verticapy.learn.linear_model import LinearRegression
		schema = self._VERTICAPY_VARIABLES_["schema_writing"]
		if not(schema): schema = "public"
		name = "{}.VERTICAPY_TEMP_MODEL_LINEAR_REGRESSION_{}".format(schema, gen_name([response]).upper())
		relation_name = "{}.VERTICAPY_TEMP_MODEL_LINEAR_REGRESSION_VIEW_{}".format(schema, gen_name([response]).upper())
		try:
			self._VERTICAPY_VARIABLES_["cursor"].execute("DROP MODEL IF EXISTS {}".format(name))
			self._VERTICAPY_VARIABLES_["cursor"].execute("DROP VIEW IF EXISTS {}".format(relation_name))
		except:
			pass
		query = "CREATE VIEW {} AS SELECT {}, {}, {}{} FROM {}".format(relation_name, ", ".join(predictors), response, ts, ", {}".format(", ".join(by)) if by else "", self.__genSQL__())
		self._VERTICAPY_VARIABLES_["cursor"].execute(query)
		model = LinearRegression(name, self._VERTICAPY_VARIABLES_["cursor"])
		model.fit(relation_name, predictors, response)
		query = "(SELECT et, LAG(et) OVER({}ORDER BY {}) AS lag_et FROM (SELECT {}{}, {} - PREDICT_LINEAR_REG({} USING PARAMETERS model_name = '{}') AS et FROM {}) x) y".format("PARTITION BY {} ".format(", ".join(by)) if (by) else "", ts, "{}, ".format(", ".join(by)) if by else "", ts, response, ", ".join(predictors), name, relation_name)
		self.__executeSQL__("SELECT SUM(POWER(et - lag_et, 2)) / SUM(POWER(et, 2)) FROM {}".format(query), title = "Computes the Durbin Watson d.")
		d = self._VERTICAPY_VARIABLES_["cursor"].fetchone()[0]
		self._VERTICAPY_VARIABLES_["cursor"].execute("DROP MODEL IF EXISTS {}".format(name))
		self._VERTICAPY_VARIABLES_["cursor"].execute("DROP VIEW IF EXISTS {}".format(relation_name))
		if (print_info):
			print("We test the following hypothesis:\n(H0) The residuals from regression {} = LinearRegression({}) are stationary\n(H1) There is a First Order Auto Correlation in residuals from regression {} = LinearRegression({})".format(response, ", ".join(predictors), response, ", ".join(predictors)))
		if (d < 1.5):
			if (print_info):
				print("\U0001F44E - The residuals might be positively correlated")
			result = False
		elif (d > 2.5):
			if (print_info):
				print("\U0001F44E - The residuals might be negatively correlated")
			result = False
		else:
			if (print_info):
				print("\U0001F44D - The residuals might be stationary")
			result = True
		if (print_info):
			print("d = {}".format(d))
		return (result, d)
	#---#
	def testjb(self, 
			   column: str, 
			   alpha: float = 0.05, 
			   print_info: bool = True):
		"""
	---------------------------------------------------------------------------
	Jarque Bera test (Distribution Normality).

	Parameters
 	----------
 	column: str
 		Input vcolumn to test.
 	alpha: float, optional
 		Significance Level. Probability to accept H0.
 	print_info: bool, optional
 		If set to True, displays all the test information.

 	Returns
 	-------
 	tuple
 		[0] result of the test : True if H0 was rejected
 		[1]                 jb : Jarque Bera Index.
 		[2]            p_value : p value of the coefficient.

	See Also
	--------
	vDataFrame.testdf : Dickey Fuller test.
	vDataFrame.testdw : Durbin Watson test.
	vDataFrame.testmk : Mann Kendall test.
		"""
		check_types([
			("column", column, [str], False), 
			("alpha", alpha, [int, float], False), 
			("print_info", print_info, [bool], False)])
		columns_check([column], self)
		column = vdf_columns_names([column], self)[0]
		if (print_info):
			print("We test the following hypothesis:\n(H0) The distribution of {} is not normal\n(H1) The distribution of {} is normal".format(column, column))
		jb, n = self[column].agg(["jb", "count"]).values[column]
		from scipy.stats import chi2
		pvalue = chi2.cdf(jb, n)
		if (pvalue < alpha):
			if (print_info):
				print("\U0001F44D - The distribution of {} might be normal".format(column))
			result = True
		else:
			if (print_info):
				print("\U0001F44E - The distribution of {} might not be normal".format(column))
			result = False
		if (print_info):
			print("jb = {}\np_value = {}".format(jb, pvalue))
		return (result, jb, pvalue)
	#---#
	def testmk(self, 
			   ts: str,
			   column: str, 
			   k = 1000,
			   alpha: float = 0.05, 
			   print_info: bool = True):
		"""
	---------------------------------------------------------------------------
	Mann Kendall test (Time Series trend).

	Parameters
 	----------
 	ts: str
 		vcolumn used as timeline. It will be used to order the data. It can be
 		a numerical or type date like (date, datetime, timestamp...) vcolumn.
 	column: str
 		Input vcolumn to test.
 	k: int/str, optional
 		Number of elements to take into account to see the trend. It can be 
 		also a time interval. For example: if k = 1000, only the last 1000
 		elements will be taken into account whereas if k = '1 day', the test
 		will identify the last day trend compared to the past.
 	alpha: float, optional
 		Significance Level. Probability to accept H0.
 	print_info: bool, optional
 		If set to True, displays all the test information.

 	Returns
 	-------
 	tuple
 		[0] result of the test : True if H0 was rejected
 		[1]                ZMK : Mann Kendall ZMK Index.
 		[2]            p_value : p value of the coefficient.

	See Also
	--------
	vDataFrame.testdf : Dickey Fuller test.
	vDataFrame.testdw : Durbin Watson test.
	vDataFrame.testjb : Jarque Bera test.
		"""
		check_types([
			("ts", ts, [str], False), 
			("column", column, [str], False),
			("k", k, [int, float, str], False), 
			("alpha", alpha, [int, float], False), 
			("print_info", print_info, [bool], False)])
		columns_check([column, ts], self)
		column = vdf_columns_names([column], self)[0]
		ts = vdf_columns_names([ts], self)[0]
		table_x = "(SELECT {}, {} FROM {}) x".format(column, ts, self.__genSQL__())
		if (type(k) == str):
			table_y = "(SELECT {}, {} FROM {} WHERE {} > '{}'::date - '{}'::interval) y".format(column, ts, self.__genSQL__(), ts, self[ts].max(), k)
		else:
			table_y = "(SELECT {}, {} FROM {} ORDER BY {} DESC LIMIT {}) y".format(column, ts, self.__genSQL__(), ts, int(k))
		query = "SELECT SUM(SIGN(y.{} - x.{})) FROM {} CROSS JOIN {} WHERE y.{} > x.{}".format(column, column, table_x, table_y, ts, ts)
		self.__executeSQL__(query, title = "Computes the Mann Kendall S.")
		S = self._VERTICAPY_VARIABLES_["cursor"].fetchone()[0]
		try:
			S = float(S)
		except:
			S = None
		query = "SELECT SQRT((SUM(row) * (SUM(row) - 1) * (2 * SUM(row) + 5) - SUM(row * (row - 1) * (2 * row + 5))) / 18) FROM (SELECT ROW_NUMBER() OVER (PARTITION BY {}) AS row FROM {}) x GROUP BY row".format(column, self.__genSQL__())
		self.__executeSQL__(query, title = "Computes the Mann Kendall S standard deviation.")
		STDS = self._VERTICAPY_VARIABLES_["cursor"].fetchone()[0]
		try:
			STDS = float(STDS)
		except:
			STDS = None
		if (print_info):
			print("We test the following hypothesis:\n(H0) The variable {} has no monotonic trend\n(H1) The variable {} has a monotonic trend".format(column, column))
		if (STDS in [0, None] or S == None):
			if (print_info):
				print("\u26A0 - The test is inconclusive")
			return None
		if (S > 0):
			ZMK = (S - 1) / STDS
		elif (S < 0):
			ZMK = (S + 1) / STDS
		else:
			ZMK = 0
		pvalue = (1.0 - math.erf(ZMK / math.sqrt(2.0))) / 2.0
		if (ZMK >= 0 and pvalue < alpha):
			if (print_info):
				print("\U0001F44D - The variable {} might have an upward monotonic trend".format(column))
			result = True
		elif (ZMK <= 0 and pvalue < alpha):
			if (print_info):
				print("\U0001F44D - The variable {} might have a downward monotonic trend".format(column))
			result = True
		else:
			if (print_info):
				print("\U0001F44E - The variable {} might have no monotonic trend".format(column))
			result = False
		if (print_info):
			print("ZMK = {}\np_value = {}\nS = {}\nstdS = {}".format(ZMK, pvalue, S, STDS))
		return (result, ZMK, pvalue)
	#---#
	def time_on_off(self):
		"""
	---------------------------------------------------------------------------
	Displays the time took by each query.

 	Returns
 	-------
 	vDataFrame
		self
	
	See Also
	--------
	vDataFrame.sql_on_off : Displays each query generated by the vDataFrame.
		"""
		self._VERTICAPY_VARIABLES_["time_on"] = not(self._VERTICAPY_VARIABLES_["time_on"])
		return (self)
	#---#
	def to_csv(self, 
			   name: str,
			   path: str = '', 
			   sep: str = ',',
			   na_rep: str = '',
			   quotechar: str = '"',
			   usecols: list = [],
			   header: bool = True,
			   new_header: list = [],
			   order_by = [],
			   limit: int = 0):
		"""
	---------------------------------------------------------------------------
	Creates a CSV file of the current vDataFrame relation.

	Parameters
 	----------
 	name: str
 		Name of the CSV file. Be careful: if a CSV file with the same name exists, 
 		it will over-write it.
 	path: str, optional
 		Absolute path where the CSV file will be created.
 	sep: str, optional
 		Column separator.
 	na_rep: str, optional
 		Missing values representation.
 	quotechar: str, optional
 		Char which will enclose the str values.
 	usecols: list, optional
 		vcolumns to select from the final vDataFrame relation. If empty, all the
 		vcolumns will be selected.
 	header: bool, optional
 		If set to False, no header will be written in the CSV file.
 	new_header: list, optional
 		List of columns used to replace vcolumns name in the CSV.
 	order_by: dict / list, optional
 		List of the vcolumns used to sort the data using asc order or
 		dictionary of all the sorting methods. For example, to sort by "column1"
 		ASC and "column2" DESC, write {"column1": "asc", "column2": "desc"}
 	limit: int, optional
 		If greater than 0, the maximum number of elements to write at the same time 
 		in the CSV file. It can be used to minimize memory impacts. Be sure to keep
 		the same order to avoid unexpected results.

 	Returns
 	-------
 	vDataFrame
		self

	See Also
	--------
	vDataFrame.to_db   : Saves the vDataFrame current relation to the Vertica Database.
	vDataFrame.to_json : Creates a JSON file of the current vDataFrame relation.
		"""
		check_types([
			("name", name, [str], False), 
			("path", path, [str], False), 
			("sep", sep, [str], False), 
			("na_rep", na_rep, [str], False), 
			("quotechar", quotechar, [str], False), 
			("usecols", usecols, [list], False), 
			("header", header, [bool], False), 
			("new_header", new_header, [list], False), 
			("order_by", order_by, [list, dict], False), 
			("limit", limit, [int, float], False)])
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
		if (limit <= 0): limit = total 
		order_by = sort_str(order_by, self)
		while (current_nb_rows_written < total):
			self._VERTICAPY_VARIABLES_["cursor"].execute("SELECT {} FROM {}{} LIMIT {} OFFSET {}".format(", ".join(columns), self.__genSQL__(), order_by, limit, current_nb_rows_written))
			result = self._VERTICAPY_VARIABLES_["cursor"].fetchall()
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
	#---#
	def to_db(self,
			  name: str,
			  usecols: list = [],
			  relation_type: str = "view",
			  inplace: bool = False,
			  db_filter = "",
			  nb_split: int = 0):
		"""
	---------------------------------------------------------------------------
	Saves the vDataFrame current relation to the Vertica Database.

	Parameters
 	----------
 	name: str
 		Name of the relation. To save the relation in a specific schema you can
 		write '"my_schema"."my_relation"'. Use double quotes '"' to avoid errors
 		due to special characters.
 	usecols: list, optional
 		vcolumns to select from the final vDataFrame relation. If empty, all the
 		columns will be selected.
 	relation_type: str, optional
 		Type of the relation.
 			view      : View
 			table     : Table
 			temporary : Temporary Table
 	inplace: bool, optional
 		If set to True, the vDataFrame will be replaced using the new relation.
 	db_filter: str / list, optional
 		Filter used before creating the relation in the DB. It can be a list of
 		conditions or an expression. This parameter is very useful to create train 
 		and test sets on TS.
 	nb_split: int, optional
		If this parameter is greater than 0, it will add to the final relation a
		new column '_verticapy_split_' which will contain values in 
		[0;nb_split - 1] where each category will have represent 1 / nb_split
		of the entire distribution. 

 	Returns
 	-------
 	vDataFrame
		self

	See Also
	--------
	vDataFrame.to_csv : Creates a csv file of the current vDataFrame relation.
		"""
		check_types([
			("name", name, [str], False), 
			("usecols", usecols, [list], False), 
			("relation_type", relation_type, ["view", "temporary", "table"], True), 
			("inplace", inplace, [bool], False),
			("db_filter", db_filter, [str, list], False),
			("nb_split", nb_split, [int, float], False)])
		relation_type = relation_type.lower()
		columns_check(usecols, self)
		usecols = vdf_columns_names(usecols, self)
		commit = " ON COMMIT PRESERVE ROWS" if (relation_type == "temporary") else ""
		if (relation_type == "temporary"):
			relation_type += " table" 
		usecols = "*" if not(usecols) else ", ".join([str_column(column) for column in usecols])
		nb_split = ", RANDOMINT({}) AS _verticapy_split_".format(nb_split) if (nb_split > 0) else ""
		if (type(db_filter) == list): db_filter = " AND ".join(["({})".format(elem) for elem in db_filter])
		db_filter = " WHERE {}".format(db_filter) if (db_filter) else ""
		query = "CREATE {} {}{} AS SELECT {}{} FROM {}{}".format(relation_type.upper(), name, commit, usecols, nb_split, self.__genSQL__(), db_filter)
		self.__executeSQL__(query = query, title = "Creates a new {} to save the vDataFrame.".format(relation_type))
		self.__add_to_history__("[Save]: The vDataFrame was saved into a {} named '{}'.".format(relation_type, name))
		if (inplace):
			query_on, time_on, history, saving = self._VERTICAPY_VARIABLES_["query_on"], self._VERTICAPY_VARIABLES_["time_on"], self._VERTICAPY_VARIABLES_["history"], self._VERTICAPY_VARIABLES_["saving"]
			catalog_vars, columns = {}, self.get_columns()
			for column in columns:
				catalog_vars[column] = self[column].catalog
			self.__init__(name, self._VERTICAPY_VARIABLES_["cursor"])
			self._VERTICAPY_VARIABLES_["history"] = history
			self._VERTICAPY_VARIABLES_["query_on"] = query_on
			self._VERTICAPY_VARIABLES_["time_on"] = time_on
			for column in columns:
				self[column].catalog = catalog_vars[column]
		return (self)
	#---#
	def to_json(self, 
			   	name: str,
			   	path: str = '', 
			   	usecols: list = [],
			   	order_by = [],
			   	limit: int = 0):
		"""
	---------------------------------------------------------------------------
	Creates a JSON file of the current vDataFrame relation.

	Parameters
 	----------
 	name: str
 		Name of the JSON file. Be careful: if a JSON file with the same name exists, 
 		it will over-write it.
 	path: str, optional
 		Absolute path where the JSON file will be created.
 	usecols: list, optional
 		vcolumns to select from the final vDataFrame relation. If empty, all the
 		vcolumns will be selected.
 	order_by: dict / list, optional
 		List of the vcolumns used to sort the data using asc order or
 		dictionary of all the sorting methods. For example, to sort by "column1"
 		ASC and "column2" DESC, write {"column1": "asc", "column2": "desc"}
 	limit: int, optional
 		If greater than 0, the maximum number of elements to write at the same time 
 		in the JSON file. It can be used to minimize memory impacts. Be sure to keep
 		the same order to avoid unexpected results.

 	Returns
 	-------
 	vDataFrame
		self

	See Also
	--------
	vDataFrame.to_csv : Creates a CSV file of the current vDataFrame relation.
	vDataFrame.to_db  : Saves the vDataFrame current relation to the Vertica Database.
		"""
		check_types([
			("name", name, [str], False), 
			("path", path, [str], False), 
			("usecols", usecols, [list], False),
			("order_by", order_by, [list, dict], False), 
			("limit", limit, [int, float], False)])
		file = open("{}{}.json".format(path, name), "w+") 
		columns = self.get_columns() if not(usecols) else [str_column(column) for column in usecols]
		total = self.shape()[0]
		current_nb_rows_written = 0
		if (limit <= 0): limit = total 
		file.write("[\n")
		order_by = sort_str(order_by, self)
		while (current_nb_rows_written < total):
			self._VERTICAPY_VARIABLES_["cursor"].execute("SELECT {} FROM {}{} LIMIT {} OFFSET {}".format(", ".join(columns), self.__genSQL__(), order_by, limit, current_nb_rows_written))
			result = self._VERTICAPY_VARIABLES_["cursor"].fetchall()
			for row in result:
				tmp_row = []
				for i, item in enumerate(row):
					if (type(item) == str):
						tmp_row += ['{}: "{}"'.format(str_column(columns[i]), item)]
					elif (item != None):
						tmp_row += ['{}: {}'.format(str_column(columns[i]), item)]
				file.write("{" + ", ".join(tmp_row) + "},\n")
			current_nb_rows_written += limit
		file.write("]")
		file.close()
		return (self)
	#---#
	def to_pandas(self):
		"""
	---------------------------------------------------------------------------
	Converts the vDataFrame to a pandas DataFrame. Be careful as you'll move data 
	in memory.

 	Returns
 	-------
 	pandas.DataFrame
 		The pandas.DataFrame of the current vDataFrame relation.
		"""
		import pandas as pd
		query = "SELECT * FROM {}".format(self.__genSQL__())
		self._VERTICAPY_VARIABLES_["cursor"].execute(query)
		column_names = [column[0] for column in self._VERTICAPY_VARIABLES_["cursor"].description]
		query_result = self._VERTICAPY_VARIABLES_["cursor"].fetchall()
		data = [list(item) for item in query_result]
		df = pd.DataFrame(data)
		df.columns = column_names
		return (df)
	#---#
	def to_vdf(self, 
			   name: str):
		"""
	---------------------------------------------------------------------------
	Saves the vDataFrame to a .vdf text file.
	The saving can be loaded using the 'read_vdf' function.

	Parameters
 	----------
 	name: str
 		Name of the file. Be careful: if a VDF file with the same name exists, it 
 		will over-write it.

 	Returns
 	-------
 	vDataFrame
		self

	See Also
	--------
	read_vdf : Loads a .vdf text file and returns a vDataFrame.
		"""
		check_types([("name", name, [str], False)])
		self.save()
		file = open("{}.vdf".format(name), "w+") 
		file.write(self._VERTICAPY_VARIABLES_["saving"][-1])
		file.close()
		return (self)
	#---#
	def var(self, 
			columns: list = []):
		"""
	---------------------------------------------------------------------------
	Aggregates the vDataFrame using 'variance'.

	Parameters
 	----------
 	columns: list, optional
 		List of the vcolumns names. If empty, all the numerical vcolumns will be 
 		used.

 	Returns
 	-------
 	tablesample
 		An object containing the result. For more information, check out
 		utilities.tablesample.

	See Also
	--------
	vDataFrame.aggregate : Computes the vDataFrame input aggregations.
		"""
		return (self.aggregate(func = ["variance"], columns = columns))
	#---#
	def version(self):
		"""
	---------------------------------------------------------------------------
	Returns the Vertica Version.

 	Returns
 	-------
 	str
 		Vertica Version
		"""
		self._VERTICAPY_VARIABLES_["cursor"].execute("SELECT version()")
		version = self._VERTICAPY_VARIABLES_["cursor"].fetchone()[0]
		try:
			return version.split("Vertica Analytic Database v")[1]
		except:
			return version
#---#
def vDataframe(input_relation: str,
			   cursor = None,
			   dsn: str = "",
			   usecols: list = [],
			   schema: str = ""):
	"""
Alias of vDataFrame. Check out vDataFrame to get all the information.
	"""
	return vDataFrame(input_relation, cursor, dsn, usecols, schema)