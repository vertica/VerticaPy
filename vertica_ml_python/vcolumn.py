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
import math
from vertica_ml_python.utilities import tablesample, to_tablesample, category_from_type, str_column, column_check_ambiguous, columns_check, vdf_columns_names, check_types, convert_special_type
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
			query = "(SELECT data_type FROM columns WHERE table_name = '{}' AND column_name = '{}')".format(self.parent.VERTICA_ML_PYTHON_VARIABLES["input_relation"].replace("'", "''"), self.alias.replace("'", "''"))
			query += " UNION (SELECT data_type FROM view_columns WHERE table_name = '{}' AND column_name = '{}')".format(self.parent.VERTICA_ML_PYTHON_VARIABLES["input_relation"].replace("'", "''"), self.alias.replace("'", "''"))
			self.parent.VERTICA_ML_PYTHON_VARIABLES["cursor"].execute(query)
			result = self.parent.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchone()
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
	def __setattr__(self, attr, val):
		self.__dict__[attr] = val
	#
	#
	#
	# METHODS
	#  
	def abs(self):
		return (self.apply(func = "ABS({})"))
	# 
	def add(self, x):
		check_types([("x", x, [int, float, str], False)])
		if (self.isdate()):
			return (self.apply(func = "TIMESTAMPADD(SECOND, {}, {})".format(x, "{}")))
		else:
			return (self.apply(func = "{} + ({})".format("{}", x)))
	#
	def add_copy(self, name: str):
		check_types([("name", name, [str], False)])
		name = str_column(name.replace('"', '_'))
		if not(name.replace('"', '')):
			raise ValueError("The parameter 'name' must not be empty")
		elif column_check_ambiguous(name, self.parent.get_columns()):
			raise ValueError("A Virtual Column has already the alias {}.\nBy changing the parameter 'name', you'll be able to solve this issue.".format(name))
		new_vColumn = vColumn(name, parent = self.parent, transformations = [item for item in self.transformations])
		setattr(self.parent, name, new_vColumn)
		setattr(self.parent, name[1:-1], new_vColumn)
		self.parent.VERTICA_ML_PYTHON_VARIABLES["columns"] += [name]
		self.parent.add_to_history("[Add Copy]: A copy of the Virtual Column {} named {} was added to the vDataframe.".format(self.alias, name))
		return (self.parent)
	# 
	def add_prefix(self, prefix: str):
		return (self.apply(func = "'{}' || {}".format(prefix.replace("'", "''"), "{}")))
	#
	def add_suffix(self, suffix: str):
		return (self.apply(func = "{} || '{}'".format("{}", suffix.replace("'", "''"))))
	#
	def agg(self, func: list):
		return (self.aggregate(func = func))
	def aggregate(self, func: list):
		return (self.parent.aggregate(func = func, columns = [self.alias]).transpose())
	#
	def apply(self, func: str, copy: bool = False, copy_name: str = ""):
		check_types([("func", func, [str], False), ("copy", copy, [bool], False), ("copy_name", copy_name, [str], False)])
		try:
			self.executeSQL(query = "DROP TABLE IF EXISTS v_temp_schema.VERTICA_ML_PYTHON_TEST;", title = "Drop the Existing Temp Table")
			self.executeSQL(query = "CREATE LOCAL TEMPORARY TABLE VERTICA_ML_PYTHON_TEST ON COMMIT PRESERVE ROWS AS SELECT {} FROM {} WHERE {} IS NOT NULL LIMIT 10;".format(func.replace("{}", self.alias), self.parent.genSQL(), self.alias), title = "TEST FUNC {}".format(func))
			self.executeSQL(query = "SELECT data_type FROM columns WHERE table_name = 'VERTICA_ML_PYTHON_TEST' AND table_schema = 'v_temp_schema'", title = "SELECT NEW DATA TYPE")
			ctype = self.parent.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchone()[0]
			self.executeSQL(query = "DROP TABLE IF EXISTS v_temp_schema.VERTICA_ML_PYTHON_TEST;", title = "Drop the Temp Table")
			category = category_from_type(ctype = ctype)
			if (copy):
				self.add_copy(name = copy_name)
				self.parent[copy_name].transformations += [(func, ctype, category)]
				self.parent.add_to_history("[{}]: The Virtual Column '{}' was transformed with the func 'x -> {}'.".format(func.replace("{}", ""), copy_name.replace('"', ''), func.replace("{}", "x")))
			else:
				self.transformations += [(func, ctype, category)]
				self.parent.add_to_history("[{}]: The Virtual Column '{}' was transformed with the func 'x -> {}'.".format(func.replace("{}", ""), self.alias.replace('"', ''), func.replace("{}", "x")))
			return (self.parent)
		except Exception as e:
			raise Exception("{}\nError when applying the func 'x -> {}' to '{}'".format(e, func.replace("{}", "x"), self.alias.replace('"', '')))
	# 
	def astype(self, dtype: str):
		check_types([("dtype", dtype, [str], False)])
		try:
			query = "SELECT {}::{} AS {} FROM {} WHERE {} IS NOT NULL LIMIT 20".format(self.alias, dtype, self.alias, self.parent.genSQL(), self.alias)
			self.executeSQL(query, title = "Data Type Conversion - TEST")
			self.transformations += [("{}::{}".format('{}', dtype), dtype, category_from_type(ctype = dtype))]
			self.parent.add_to_history("[AsType]: The Virtual Column {} was converted to {}.".format(self.alias, dtype))
			return (self.parent)
		except:
			raise Exception("The Virtual Column {} can not be converted to {}".format(self.alias, dtype))
	# 
	def avg(self):
		return (self.mean())
	# 
	def bar(self,
			method: str = "density",
			of: str = "",
			max_cardinality: int = 6,
			bins: int = 0,
			h: float = 0,
			color: str = '#214579'):
		check_types([("method", method, ["density", "count", "avg", "min", "max", "sum"], True), ("of", of, [str], False), ("max_cardinality", max_cardinality, [int, float], False), ("bins", bins, [int, float], False), ("h", h, [int, float], False), ("color", color, [str], False)])
		if (of):	
			columns_check([of], self.parent)
			of = vdf_columns_names([of], self.parent)[0]
		from vertica_ml_python.plot import bar
		bar(self, method, of, max_cardinality, bins, h, color)	
		return (self.parent)
	# 
	def boxplot(self, 
				by: str = "", 
				h: float = 0, 
				max_cardinality: int = 8, 
				cat_priority: list = []):
		check_types([("by", by, [str], False), ("max_cardinality", max_cardinality, [int, float], False), ("h", h, [int, float], False), ("cat_priority", cat_priority, [list], False)])
		if (by):
			columns_check([by], self.parent)
			by = vdf_columns_names([by], self.parent)[0]
		from vertica_ml_python.plot import boxplot
		boxplot(self, by, h, max_cardinality, cat_priority)
		return (self.parent)
	# 
	def category(self):
		return (self.transformations[-1][2])
	#
	def clip(self, lower = None, upper = None):
		check_types([("lower", lower, [float, int, type(None)], False), ("upper", upper, [float, int, type(None)], False)])
		if ((lower == None) and (upper == None)):
			raise ValueError("At least 'lower' or 'upper' must have a numerical value")
		lower_when = "WHEN {} < {} THEN {} ".format("{}", lower, lower) if (type(lower) in (float, int)) else ""
		upper_when = "WHEN {} > {} THEN {} ".format("{}", upper, upper) if (type(upper) in (float, int)) else "" 
		func = "(CASE {}{}ELSE {} END)".format(lower_when, upper_when, "{}")
		self.apply(func = func)
		return (self.parent)
	# 
	def count(self):
		return (self.aggregate(["count"]).values[self.alias][0])
	#
	def ctype(self):
		return (self.transformations[-1][1].lower())
	# 
	def date_part(self, field: str):
		return (self.apply(func = "DATE_PART('{}', {})".format(field, "{}")))
	# 
	def decode(self, values: dict, others = None):
		check_types([("values", values, [dict], False)])
		new_dict = {}
		for elem in values:
			if (type(values[elem]) == str):
				val = "'{}'".format(values[elem].replace("'", "''"))
			elif (values[elem] == None):
				val = "NULL"
			else:
				val = values[elem]
			if str(elem).upper() in ('NULL', 'NONE'):
				new_dict["NULL"] = val
			else:
				new_dict["'" + elem + "'"] = val
		others = "'{}'".format(others.replace("'", "''")) if (type(others) == str) else others
		others = "NULL" if (others == None) else others
		fun = "DECODE({}, " + ", ".join(["{}, {}".format(item, new_dict[item]) for item in new_dict]) + ", {})".format(others)
		return (self.apply(func = fun))
	# 
	def density(self,
				a = None,
				kernel: str = "gaussian",
				smooth: int = 200,
				color: str = '#214579'):
		check_types([("kernel", kernel, ["gaussian", "logistic", "sigmoid", "silverman"], True), ("smooth", smooth, [int, float], False), ("color", color, [str], False), ("a", a, [type(None), float, int], False)])
		from vertica_ml_python.plot import density
		density(self, a, kernel, smooth, color)
		return (self.parent)
	# 
	def describe(self, 
				 method: str = "auto", 
				 max_cardinality: int = 6,
				 numcol: str = ""):
		check_types([("method", method, ["auto", "numerical", "categorical", "cat_stats"], True), ("max_cardinality", max_cardinality, [int, float], False), ("numcol", numcol, [str], False)])
		if (method not in ["auto", "numerical", "categorical", "cat_stats"]):
			raise ValueError("The parameter 'method' must be in auto|categorical|numerical|cat_stats")
		elif (method == "cat_stats") and not(numcol):
			raise ValueError("The parameter 'numcol' must be a vDataframe column if the method is 'cat_stats'")
		distinct_count, is_numeric, is_date = self.nunique(), self.isnum(), self.isdate()
		if ((is_date) and not(method == "categorical")):
			query = "SELECT COUNT({}) AS count, MIN({}) AS min, MAX({}) AS max FROM {};".format(self.alias,self.alias,self.alias,self.parent.genSQL())
			self.executeSQL(query = query, title = "Compute the descriptive statistics of "+self.alias)
			result = self.parent.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchall()
			result = [item for sublist in result for item in sublist]
			index = ['count', 'min', 'max']
		elif ((method == "cat_stats") and (numcol != "")):
			numcol = vdf_columns_names([numcol], self.parent)[0]
			if (self.parent[numcol].category() not in ("float", "int")):
				raise TypeError("The column 'numcol' must be numerical")
			cast = "::int" if (self.parent[numcol].ctype() == "boolean") else ""
			query, cat = [], self.distinct()
			if (len(cat) == 1):
				lp, rp = "(", ")"
			else:
				lp, rp = "", ""
			for category in cat:
				tmp_query = "SELECT '{}' AS 'index', COUNT({}) AS count, 100 * COUNT({}) / {} AS percent, AVG({}{}) AS mean, STDDEV({}{}) AS std, MIN({}{}) AS min, APPROXIMATE_PERCENTILE ({}{} USING PARAMETERS percentile = 0.1) AS '10%', APPROXIMATE_PERCENTILE ({}{} USING PARAMETERS percentile = 0.25) AS '25%', APPROXIMATE_PERCENTILE ({}{} USING PARAMETERS percentile = 0.5) AS '50%', APPROXIMATE_PERCENTILE ({}{} USING PARAMETERS percentile = 0.75) AS '75%', APPROXIMATE_PERCENTILE ({}{} USING PARAMETERS percentile = 0.9) AS '90%', MAX({}{}) AS max FROM vdf_table"
				tmp_query = tmp_query.format(category, self.alias, self.alias, self.parent.shape()[0], numcol, cast, numcol, cast, numcol, cast, numcol, cast, numcol, cast, numcol, cast, numcol, cast, numcol, cast, numcol, cast)
				tmp_query += " WHERE {} IS NULL".format(self.alias) if (category in ('None', None)) else " WHERE {} = '{}'".format(convert_special_type(self.category(), False, self.alias), category)
				query += [lp + tmp_query + rp]
			query = "WITH vdf_table AS ({}) {}".format(self.parent.genSQL(return_without_alias = True), " UNION ALL ".join(query))
			result = to_tablesample(query, self.parent.VERTICA_ML_PYTHON_VARIABLES["cursor"])
			result.table_info = False
			return (result)
		elif (((distinct_count < max_cardinality + 1) and (method != "numerical")) or not(is_numeric) or (method == "categorical")):
			query = "(SELECT {} || '', COUNT(*) FROM vdf_table GROUP BY {} ORDER BY COUNT(*) DESC LIMIT {})".format(self.alias, self.alias, max_cardinality)
			if (distinct_count > max_cardinality):
				query += ("UNION ALL (SELECT 'Others', SUM(count) FROM (SELECT COUNT(*) AS count FROM vdf_table WHERE {} IS NOT NULL GROUP BY {} ORDER BY COUNT(*) DESC OFFSET {}) x) ORDER BY count DESC").format(self.alias, self.alias, max_cardinality + 1)
			query = "WITH vdf_table AS ({}) {}".format(self.parent.genSQL(return_without_alias = True), query)
			self.executeSQL(query = query, title = "Compute the descriptive statistics of {}".format(self.alias))
			query_result = self.parent.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchall()
			result = [distinct_count] + [item[1] for item in query_result]
			index = ['unique'] + [item[0] for item in query_result]
		else:
			result = self.summarize_numcol().values["value"]
			result = [distinct_count] + result
			index = ['unique', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
		values = {"index" : ["name", "dtype"] + index, "value" : [self.alias, self.ctype()] + result}
		return (tablesample(values, table_info = False))
	# 
	def discretize(self, 
				   method: str = "auto",
				   h: float = 0, 
				   bins: int = -1,
				   k: int = 6,
				   new_category: str = 'Others',
				   response: str = "",
				   temp_information: tuple = ("public.vml_temp_view", "public.vml_temp_model"),
				   min_bin_size: int = 20,
				   return_enum_trans: bool = False):
		check_types([("temp_information", temp_information, [tuple], False),("min_bin_size", min_bin_size, [int, float], False), ("return_enum_trans", return_enum_trans, [bool], False), ("h", h, [int, float], False), ("response", response, [str], False), ("bins", bins, [int, float], False), ("method", method, ["auto", "smart", "same_width", "same_freq", "topk"], True), ("return_enum_trans", return_enum_trans, [bool], False)])
		if (self.isnum() and method == "smart"):
			if (bins < 2):
				raise ValueError("Parameter 'bins' must be greater or equals to 2 in case of discretization using the method 'smart'")
			columns_check([response], self.parent)
			response = vdf_columns_names([response], self.parent)[0]
			self.parent.VERTICA_ML_PYTHON_VARIABLES["cursor"].execute("DROP VIEW IF EXISTS {}".format(temp_information[0]))
			self.parent.VERTICA_ML_PYTHON_VARIABLES["cursor"].execute("DROP MODEL IF EXISTS {}".format(temp_information[1]))
			self.parent.to_db(temp_information[0])
			from vertica_ml_python.learn.ensemble import RandomForestClassifier
			model = RandomForestClassifier(temp_information[1], self.parent.VERTICA_ML_PYTHON_VARIABLES["cursor"], n_estimators = 20, max_depth = 3, nbins = 100, min_samples_leaf = min_bin_size)
			model.fit(temp_information[0], [self.alias], response)
			query = ["(SELECT READ_TREE(USING PARAMETERS model_name = '{}', tree_id = {}, format = 'tabular'))".format(temp_information[1], i) for i in range(20)]
			query = "SELECT split_value FROM (SELECT split_value, COUNT(*) FROM ({}) x WHERE split_value IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT {}) y ORDER BY split_value::float".format(" UNION ALL ".join(query), bins - 1)
			self.executeSQL(query = query, title = "Compute the Optimized Histogram separations using RF")
			result = self.parent.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchall()
			result = [elem[0] for elem in result]
			self.parent.VERTICA_ML_PYTHON_VARIABLES["cursor"].execute("DROP VIEW IF EXISTS {}".format(temp_information[0]))
			self.parent.VERTICA_ML_PYTHON_VARIABLES["cursor"].execute("DROP MODEL IF EXISTS {}".format(temp_information[1]))
			result = [self.min()] + result + [self.max()]
		elif (method == "topk"):
			if (k < 2):
				raise ValueError("Parameter 'k' must be greater or equals to 2 in case of discretization using the method 'topk'")
			distinct = self.topk(k).values["index"]
			trans = ("(CASE WHEN {} IN ({}) THEN {} || '' ELSE '{}' END)".format(convert_special_type(self.category(), False), ', '.join(["'{}'".format(str(elem).replace("'", "''")) for elem in distinct]), convert_special_type(self.category(), False), new_category.replace("'", "''")), "varchar", "text")
		elif (self.isnum() and method == "same_freq"):
			if (bins < 2):
				raise ValueError("Parameter 'bins' must be greater or equals to 2 in case of discretization using the method 'same_freq'")
			count = self.count()
			nb = int(float(count / int(bins)))
			if (nb == 0):
				raise Exception("Not enough values to compute the Equal Frequency discretization")
			total, query, nth_elems = nb, [], []
			while (total < count - 1):
				nth_elems += [str(total)]
				total += nb
			where = "WHERE row_nb IN ({})".format(", ".join(['1'] + nth_elems + [str(count)]))
			query = "SELECT {} FROM (SELECT {}, ROW_NUMBER() OVER (ORDER BY {}) AS row_nb FROM {} WHERE {} IS NOT NULL) x {}".format(self.alias, self.alias, self.alias, self.parent.genSQL(), self.alias, where)
			self.executeSQL(query = query, title = "Compute the Equal Frequency Histogram separations")
			result = self.parent.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchall()
			result = [elem[0] for elem in result]
		elif (self.isnum() and method in ("same_width", "auto")):
			h = round(self.numh(), 2) if (h <= 0) else h
			h = int(max(math.floor(h), 1)) if (self.category() == "int") else h
			floor_end = - 1 if (self.category() == "int") else ""
			if (h > 1) or (self.category() == "float"):
				trans = ("'[' || FLOOR({} / {}) * {} || ';' || (FLOOR({} / {}) * {} + {}{}) || ']'".format(
					"{}", h, h, "{}", h, h, h, floor_end), "varchar", "text")
			else:
				trans = ("FLOOR({}) || ''", "varchar", "text")
		else:
			trans = ("{} || ''", "varchar", "text")
		if ((self.isnum() and method == "same_freq") or (self.isnum() and method == "smart")):
			n = len(result)
			trans = "(CASE "
			for i in range(1, n):
				trans += "WHEN {} BETWEEN {} AND {} THEN '[{}-{}]' ".format("{}", result[i - 1], result[i], result[i - 1], result[i])
			trans += " ELSE NULL END)"
			trans = (trans, "varchar", "text")
		if (return_enum_trans):
			return(trans)
		else:
			self.transformations += [trans]
			self.parent.add_to_history("[Discretize]: The Virtual Column {} was discretized.".format(self.alias))
		return (self.parent)
	# 
	def distinct(self):
		query = "SELECT {} AS {} FROM {} WHERE {} IS NOT NULL GROUP BY {} ORDER BY {}".format(convert_special_type(self.category(), True, self.alias), self.alias, self.parent.genSQL(), self.alias, self.alias, self.alias)
		self.executeSQL(query = query, title = "Compute the distinct categories of {}".format(self.alias))
		query_result = self.parent.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchall()
		return ([item for sublist in query_result for item in sublist])
	# 
	def div(self, x: float):
		return (self.divide(x = x))
	def divide(self, x: float):
		check_types([("x", x, [int, float], False)])
		if (x == 0):
			raise ZeroDivisionError("Division by 0 is forbidden")
		return (self.apply(func = "{} / ({})".format("{}", x)))
	# 
	def donut(self,
			  method: str = "density",
			  of: str = "",
			  max_cardinality: int = 6,
			  h: float = 0):
		check_types([("method", method, ["density", "count", "avg", "min", "max", "sum"], True), ("of", of, [str], False), ("max_cardinality", max_cardinality, [int, float], False), ("h", h, [int, float], False)])
		if (of):
			columns_check([of], self.parent)
			of = vdf_columns_names([of], self.parent)[0]
		from vertica_ml_python.plot import pie
		pie(self, method, of, max_cardinality, h, True)	
		return (self.parent)
	# 
	def drop(self, add_history: bool = True):
		check_types([("add_history", add_history, [bool], False)])
		try:
			parent = self.parent
			force_columns = [column for column in self.parent.VERTICA_ML_PYTHON_VARIABLES["columns"]]
			force_columns.remove(self.alias)
			self.parent.VERTICA_ML_PYTHON_VARIABLES["cursor"].execute("SELECT * FROM {} LIMIT 10".format(self.parent.genSQL(force_columns = force_columns)))
			self.parent.VERTICA_ML_PYTHON_VARIABLES["columns"].remove(self.alias)
			delattr(self.parent, self.alias)
		except:
			self.parent.VERTICA_ML_PYTHON_VARIABLES["exclude_columns"] += [self.alias]
		if (add_history):
			self.parent.add_to_history("[Drop]: Virtual Column {} was deleted from the vDataframe.".format(self.alias))
		return (parent)
	# 
	def dropna(self, print_info: bool = True):
		check_types([("print_info", print_info, [bool], False)])
		count = self.parent.shape()[0]
		self.parent.VERTICA_ML_PYTHON_VARIABLES["where"] += [("{} IS NOT NULL".format(self.alias), len(self.transformations) - 1)]
		total = abs(count - self.parent.shape()[0])
		if (total > 1):
			if (print_info):
				print("{} elements were dropped".format(total))
			self.parent.add_to_history("[Dropna]: The {} missing elements of Virtual Column {} were dropped from the vDataframe.".format(total, self.alias))
		elif (total == 1):
			if (print_info):
				print("1 element was dropped")
			self.parent.add_to_history("[Dropna]: The only missing element of the Virtual Column {} was dropped from the vDataframe.".format(self.alias))
		else:
			del self.parent.VERTICA_ML_PYTHON_VARIABLES["where"][-1]
			if (print_info):
				print("\u26A0 Warning: Nothing was dropped")
		return (self.parent) 
	# 
	def drop_outliers(self, alpha: float = 0.05, use_threshold: bool = True, threshold: float = 4.0):
		check_types([("alpha", alpha, [int, float], False), ("use_threshold", use_threshold, [bool], False), ("threshold", threshold, [int, float], False)])
		if (use_threshold):
			result = self.aggregate(func = ["std", "avg"]).transpose().values
			self.parent.filter(expr = "ABS({} - {}) / {} < {}".format(self.alias, result["avg"][0], result["std"][0], threshold))
		else:
			query = "SELECT PERCENTILE_CONT({}) WITHIN GROUP (ORDER BY {}) OVER (), PERCENTILE_CONT(1 - {}) WITHIN GROUP (ORDER BY {}) OVER () FROM {} LIMIT 1".format(alpha, self.alias, alpha, self.alias, self.parent.genSQL())
			self.executeSQL(query = query, title = "Compute the PERCENTILE_CONT of " + self.alias)
			p_alpha, p_1_alpha = self.parent.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchone()
			self.parent.filter(expr = "({} BETWEEN {} AND {})".format(self.alias, p_alpha, p_1_alpha))
		return (self.parent)
	#
	def dtype(self):
		print("col".ljust(6) + self.ctype().rjust(12))
		print("dtype: object")
		return (self.ctype())
	#
	def ema(self, ts: str, by: list = [], alpha: float = 0.5):
		check_types([("ts", ts, [str], False), ("by", by, [list], False), ("alpha", alpha, [int, float], False)])
		columns_check(by, self)
		by = "PARTITION BY {}".format(", ".join(vdf_columns_names(by, self.parent))) if (by) else ""
		return (self.apply(func = "EXPONENTIAL_MOVING_AVERAGE({}, {}) OVER ({} ORDER BY {})".format("{}", alpha, by, str_column(ts))))
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
		check_types([("method", method, ["auto", "mode", "0ifnull", 'mean', 'avg', 'median', 'ffill', 'pad', 'bfill', 'backfill', ], True), ("by", by, [list], False), ("order_by", order_by, [list], False)])
		columns_check(order_by + by, self.parent)
		by, order_by = vdf_columns_names(by, self.parent), vdf_columns_names(order_by, self.parent)
		if (method == "auto"):
			method = "mean" if (self.isnum() and self.nunique(True) > 6) else "mode"
		total = self.count()
		if ((method == "mode") and (val == None)):
			val = self.mode()
			if (val == None):
				print('\u26A0 Warning: The Virtual Column {} has no mode (only missing values)\nNothing was filled.'.format(self.alias))
				return (self.parent)
		if (type(val) == str):
			val = val.replace("'", "''")
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
			elif (len(by) == 1):
				try:
					fun = "APPROXIMATE_MEDIAN" if (fun == "MEDIAN") else fun
					query = "SELECT {}, {}({}) FROM {} GROUP BY {};".format(by[0], fun, self.alias, self.parent.genSQL(), by[0])
					self.executeSQL(query, title = "Compute the different agg")
					result = self.parent.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchall()
					for idx, elem in enumerate(result):
						result[idx][0] = 'NULL' if (elem[0] == None) else "'{}'".format(str(elem[0]).replace("'", "''"))
						result[idx][1] = 'NULL' if (elem[1] == None) else str(elem[1])
					new_column = "COALESCE({}, DECODE({}, {}, NULL))".format("{}", by[0], ", ".join(["{}, {}".format(elem[0], elem[1]) for elem in result]))
					self.executeSQL("SELECT {} FROM {} LIMIT 1".format(new_column.format(self.alias), self.parent.genSQL()), title = "Test the query")
				except:
					new_column = "COALESCE({}, {}({}) OVER (PARTITION BY {}))".format("{}", fun, "{}", ", ".join(by))
			else:
				new_column = "COALESCE({}, {}({}) OVER (PARTITION BY {}))".format("{}", fun, "{}", ", ".join(by))
		elif (method in ('ffill', 'pad', 'bfill', 'backfill')):
			if not(order_by):
				raise ValueError("If the method is in ffill|pad|bfill|backfill then 'order_by' must be a list of at least one element used to order the data")
			desc = " DESC" if (method in ("ffill", "pad")) else ""
			by = "PARTITION BY {}".format(", ".join([str_column(column) for column in by])) if (by) else ""
			order_by = ", ".join([str_column(column) for column in order_by])
			new_column = "COALESCE({}, LAST_VALUE({} IGNORE NULLS) OVER ({} ORDER BY {}{}))".format("{}", "{}", by, order_by, desc)
		else:
			raise ValueError("The method '{}' does not exist or is not available\nPlease use a method in auto|mean|median|mode|ffill|bfill|0ifnull".format(method))
		if (method in ("mean", "median") or (type(val) == float)):
			category, ctype = "float", "float"
		elif (method == "0ifnull"):
			category, ctype = "int", "bool"
		else:
			category, ctype = self.category(), self.ctype()
		self.transformations += [(new_column, ctype, category)]
		try:
			total = abs(self.count() - total)
		except Exception as e:
			del self.transformations[-1]
			raise Exception("{}\nAn Error happened during the filling.".format(e))
		if (total > 1):
			if (print_info):
				print("{} elements were filled".format(int(total)))
			self.parent.add_to_history("[Fillna]: {} missing values of the Virtual Column {} were filled.".format(total, self.alias))
		elif (total == 0):
			if (print_info):
				print("Nothing was filled")
			del self.transformations[-1]
		else:
			if (print_info):
				print("1 element was filled")
			self.parent.add_to_history("[Fillna]: 1 missing value of the Virtual Column {} was filled.".format(self.alias))
		return (self.parent)
	#
	def fill_outliers(self, method: str = "winsorize", alpha: float = 0.05, use_threshold: bool = True, threshold: float = 4.0):
		check_types([("method", method, ["winsorize", "null", "mean"], True), ("alpha", alpha, [int, float], False), ("use_threshold", use_threshold, [bool], False), ("threshold", threshold, [int, float], False)])
		if (method not in ("winsorize", "null", "mean")):
			raise ValueError("The parameter 'method' must be in winsorize|null|mean")
		else:
			if (use_threshold):
				result = self.aggregate(func = ["std", "avg"]).transpose().values
				p_alpha, p_1_alpha = - threshold * result["std"][0] + result["avg"][0], threshold * result["std"][0] + result["avg"][0]
			else:
				query = "SELECT PERCENTILE_CONT({}) WITHIN GROUP (ORDER BY {}) OVER (), PERCENTILE_CONT(1 - {}) WITHIN GROUP (ORDER BY {}) OVER () FROM {} LIMIT 1".format(alpha, self.alias, alpha, self.alias, self.parent.genSQL())
				self.executeSQL(query = query, title = "Compute the PERCENTILE_CONT of " + self.alias)
				p_alpha, p_1_alpha = self.parent.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchone()
			if (method == "winsorize"):
				self.clip(lower = p_alpha, upper = p_1_alpha)
			elif (method == "null"):
				self.apply(func = "(CASE WHEN ({} BETWEEN {} AND {}) THEN {} ELSE NULL END)".format('{}', p_alpha, p_1_alpha, '{}'))
			elif (method == "mean"):
				query = "WITH vdf_table AS ({}) (SELECT AVG({}) FROM vdf_table WHERE {} < {}) UNION ALL (SELECT AVG({}) FROM vdf_table WHERE {} > {})".format(
					self.parent.genSQL(return_without_alias = True), self.alias, self.alias, p_alpha, self.alias, self.alias, p_1_alpha)
				self.executeSQL(query = query, title = "Compute the MEAN of the {}'s lower and upper outliers".format(self.alias))
				mean_alpha, mean_1_alpha = [item[0] for item in self.parent.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchall()]
				self.apply(func = "(CASE WHEN {} < {} THEN {} WHEN {} > {} THEN {} ELSE {} END)".format('{}', p_alpha, mean_alpha, '{}', p_1_alpha, mean_1_alpha, '{}'))
		return (self.parent)
	#
	def ge(self, x: float):
		check_types([("x", x, [int, float], False)])
		return (self.apply(func = "{} >= ({})".format("{}", x)))
	# 
	def get_dummies(self, 
					prefix: str = "", 
					prefix_sep: str = "_", 
					drop_first: bool = True, 
					use_numbers_as_suffix: bool = False):
		check_types([("prefix", prefix, [str], False), ("prefix_sep", prefix_sep, [str], False), ("drop_first", drop_first, [bool], False), ("use_numbers_as_suffix", use_numbers_as_suffix, [bool], False)])
		distinct_elements = self.distinct()
		if (distinct_elements not in ([0, 1], [1, 0]) or self.ctype() == "boolean"):
			all_new_features = []
			prefix = self.alias.replace('"', '') + prefix_sep.replace('"', '_') if not(prefix) else prefix.replace('"', '_') + prefix_sep.replace('"', '_')
			n = 1 if drop_first else 0
			columns = self.parent.get_columns()
			for k in range(len(distinct_elements) - n):
				name = '"{}{}"'.format(prefix, k) if (use_numbers_as_suffix) else '"{}{}"'.format(prefix, str(distinct_elements[k]).replace('"', '_'))
				if (column_check_ambiguous(name, columns)):
					raise ValueError("A Virtual Column has already the alias of one of the dummies ({}).\nIt can be the result of using previously the method on the Virtual Column or simply because of ambiguous columns naming.\nBy changing one of the parameters ('prefix', 'prefix_sep'), you'll be able to solve this issue.".format(name))
			for k in range(len(distinct_elements) - n):
				name = '"{}{}"'.format(prefix, k) if (use_numbers_as_suffix) else '"{}{}"'.format(prefix, str(distinct_elements[k]).replace('"', '_'))
				expr = "DECODE({}, '{}', 1, 0)".format("{}", str(distinct_elements[k]).replace("'", "''"))
				transformations = self.transformations + [(expr, "bool", "int")]
				new_vColumn = vColumn(name, parent = self.parent, transformations = transformations)
				setattr(self.parent, name, new_vColumn)
				setattr(self.parent, name.replace('"', ''), new_vColumn)
				self.parent.VERTICA_ML_PYTHON_VARIABLES["columns"] += [name]
				all_new_features += [name]
			self.parent.add_to_history("[Get Dummies]: One hot encoder was applied to the Virtual Column {}\n{} features were created: {}".format(self.alias, len(all_new_features), ", ".join(all_new_features)) + ".")
		return (self.parent)
	#
	def gt(self, x: float):
		check_types([("x", x, [int, float], False)])
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
		check_types([("method", method, ["density", "count", "avg", "min", "max", "sum"], True), ("of", of, [str], False), ("max_cardinality", max_cardinality, [int, float], False), ("h", h, [int, float], False), ("bins", bins, [int, float], False), ("color", color, [str], False)])
		if (of):	
			columns_check([of], self.parent)
			of = vdf_columns_names([of], self.parent)[0]
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
		if (self.category() in ["date", "float"]):
			print("\u26A0 Warning: label_encode is only available for categorical variables.")
		else:
			distinct_elements = self.distinct()
			expr = ["DECODE({}"]
			text_info = "\n"
			for k in range(len(distinct_elements)):
				expr += ["'{}', {}".format(str(distinct_elements[k]).replace("'", "''"), k)]
				text_info += "\t{} => {}".format(distinct_elements[k], k)
			expr = ", ".join(expr) + ", {})".format(len(distinct_elements))
			self.transformations += [(expr, 'int', 'int')]
			self.parent.add_to_history("[Label Encoding]: Label Encoding was applied to the Virtual Column {} using the following mapping:{}".format(self.alias, text_info))
		return (self.parent)
	#
	def le(self, x: float):
		check_types([("x", x, [int, float], False)])
		return (self.apply(func = "{} <= ({})".format("{}", x)))
	#
	def lt(self, x: float):
		check_types([("x", x, [int, float], False)])
		return (self.apply(func = "{} < ({})".format("{}", x)))
	# 
	def mad(self):
		return (self.aggregate(["mad"]).values[self.alias][0])
	# 
	def mae(self):
		return (self.aggregate(["mae"]).values[self.alias][0])
	# 
	def max(self):
		return (self.aggregate(["max"]).values[self.alias][0])
	# 
	def mean(self):
		return (self.aggregate(["avg"]).values[self.alias][0])
	# 
	def mean_encode(self, response_column: str):
		check_types([("response_column", response_column, [str], False)])
		columns_check([response_column], self.parent)
		response_column = vdf_columns_names([response_column], self.parent)[0]
		if not(self.parent[response_column].isnum()):
			raise TypeError("The response column must be numerical to use a mean encoding")
		else:
			self.transformations += [("AVG({}) OVER (PARTITION BY {})".format(response_column,"{}"), "int", "float")]
			self.parent.add_to_history("[Mean Encode]: The Virtual Column {} was transformed using a mean encoding with {} as Response Column.".format(self.alias, response_column))
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
		check_types([("n", n, [int, float], False)])
		return (self.apply(func = "MOD({}, {})".format("{}", n)))
	# 
	def mode(self, dropna: bool = True, n: int = 1):
		check_types([("dropna", dropna, [bool], False), ("n", n, [int, float], False)])
		if (n < 1):
			raise ValueError("Parameter 'n' must be greater or equal to 1")
		self.executeSQL("SELECT {} FROM (SELECT {}, COUNT(*) AS cnt FROM {} WHERE {} IS NOT NULL GROUP BY {} ORDER BY cnt DESC LIMIT {}) x ORDER BY cnt ASC LIMIT 1".format(self.alias, self.alias, self.parent.genSQL(), self.alias, self.alias, n))
		try:
			return (self.parent.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchone()[0])
		except:
			return None
	# 
	def mul(self, x: float):
		check_types([("x", x, [int, float], False)])
		return (self.apply(func = "{} * ({})".format("{}", x)))
	# 
	def neq(self, x):
		return (self.apply(func = "{} != {}".format("{}", x)))
	#
	def next(self, order_by: list, by: list = []):
		check_types([("order_by", order_by, [list], False), ("by", by, [list], False)])
		columns_check(order_by + by, self.parent)
		order_by, by = vdf_columns_names(order_by, self.parent), vdf_columns_names(by, self.parent)
		by = "PARTITION BY {}".format(", ".join(by)) if (by) else ""
		return (self.apply(func = "LEAD({}) OVER ({} ORDER BY {})".format("{}", by, ", ".join(order_by))))
	# 
	def nlargest(self, n: int = 10):
		check_types([("n", n, [int, float], False)])
		query = "SELECT * FROM {} WHERE {} IS NOT NULL ORDER BY {} DESC LIMIT {}".format(self.parent.genSQL(), self.alias, self.alias, n)
		return (to_tablesample(query, self.parent.VERTICA_ML_PYTHON_VARIABLES["cursor"], name = "nlargest"))
	# 
	def normalize(self, method: str = "zscore", by: list = [], return_trans: bool = False):
		check_types([("method", method, ["zscore", "robust_zscore", "minmax"], True), ("by", by, [list], False)])
		columns_check(by, self.parent)
		by = vdf_columns_names(by, self.parent)
		n = len(by)
		nullifzero = 1
		if (self.ctype() == "boolean"):
			print("\u26A0 Warning: Normalize doesn't work on booleans".format(self.alias))
		elif (self.isnum()):
			if (method == "zscore"):
				if (n == 0):
					nullifzero = 0
					avg, stddev = self.aggregate(["avg", "std"]).values[self.alias]
					if (stddev == 0):
						print("\u26A0 Warning: Can not normalize {} using a Z-Score - The Standard Deviation is null !".format(self.alias))
						return (self)
				elif (n == 1):
					try:
						self.executeSQL("SELECT {}, AVG({}), STDDEV({}) FROM {} GROUP BY {}".format(by[0], self.alias, self.alias, self.parent.genSQL(), by[0]), title = "Computing the different categories to normalize")
						result = self.parent.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchall()
						for i in range(len(result)):
							if (result[i][2] == None):
								pass
							elif (math.isnan(result[i][2])):
								result[i][2] = None
						avg    = "DECODE({}, {}, NULL)".format(by[0], ", ".join(["{}, {}".format("'{}'".format(str(elem[0]).replace("'", "''")) if elem[0] != None else 'NULL', elem[1] if elem[1] != None else 'NULL') for elem in result if elem[1] != None]))
						stddev = "DECODE({}, {}, NULL)".format(by[0], ", ".join(["{}, {}".format("'{}'".format(str(elem[0]).replace("'", "''")) if elem[0] != None else 'NULL', elem[2] if elem[2] != None else 'NULL') for elem in result if elem[2] != None]))
						self.executeSQL("SELECT {}, {} FROM {} LIMIT 1".format(avg, stddev, self.parent.genSQL()), title = "Test Normalize")
					except:
						avg, stddev = "AVG({}) OVER (PARTITION BY {})".format(self.alias, ", ".join(by)), "STDDEV({}) OVER (PARTITION BY {})".format(self.alias, ", ".join(by))
				else:
					avg, stddev = "AVG({}) OVER (PARTITION BY {})".format(self.alias, ", ".join(by)), "STDDEV({}) OVER (PARTITION BY {})".format(self.alias, ", ".join(by))
				if (return_trans):
					return "({} - {}) / {}({})".format(self.alias, avg, "NULLIFZERO" if (nullifzero) else "", stddev)
				else:
					self.transformations += [("({} - {}) / {}({})".format("{}", avg, "NULLIFZERO" if (nullifzero) else "", stddev), "float", "float")]
			elif (method == "robust_zscore"):
				if (n > 0):
					print("\u26A0 Warning: the method 'robust_zscore' is only available only if the parameter 'by' is empty\nIf you want to normalize by grouping by elements, please use a method in zscore|minmax")
					return(self)
				mad, med = self.aggregate(["mad", "median"]).values[self.alias]
				mad *= 1.4826
				if (mad != 0):
					if (return_trans):
						return "({} - {}) / ({})".format(self.alias, med, mad)
					else:
						self.transformations += [("({} - {}) / ({})".format("{}", med, mad), "float", "float")]
				else:
					print("\u26A0 Warning: Can not normalize {} using a Robust Z-Score - The MAD is null !".format(self.alias))
					return (self)
			elif (method == "minmax"):
				if (n == 0):
					nullifzero = 0
					cmin, cmax = self.aggregate(["min", "max"]).values[self.alias]
					if (cmax - cmin == 0):
						print("\u26A0 Warning: Can not normalize {} using the MIN and the MAX. MAX = MIN !".format(self.alias))
						return(self)
				elif (n == 1):
					try:
						self.executeSQL("SELECT {}, MIN({}), MAX({}) FROM {} GROUP BY {}".format(by[0], self.alias, self.alias, self.parent.genSQL(), by[0]), title = "Computing the different categories to normalize")
						result = self.parent.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchall()
						cmin = "DECODE({}, {}, NULL)".format(by[0], ", ".join(["{}, {}".format("'{}'".format(str(elem[0]).replace("'", "''")) if elem[0] != None else 'NULL', elem[1] if elem[1] != None else 'NULL') for elem in result if elem[1] != None]))
						cmax = "DECODE({}, {}, NULL)".format(by[0], ", ".join(["{}, {}".format("'{}'".format(str(elem[0]).replace("'", "''")) if elem[0] != None else 'NULL', elem[2] if elem[2] != None else 'NULL') for elem in result if elem[2] != None]))
						self.executeSQL("SELECT {}, {} FROM {} LIMIT 1".format(cmax, cmin, self.parent.genSQL()), title = "Test Normalize")
					except:
						cmax, cmin = "MAX({}) OVER (PARTITION BY {})".format(self.alias, ", ".join(by)), "MIN({}) OVER (PARTITION BY {})".format(self.alias, ", ".join(by))
				else:
					cmax, cmin = "MAX({}) OVER (PARTITION BY {})".format(self.alias, ", ".join(by)), "MIN({}) OVER (PARTITION BY {})".format(self.alias, ", ".join(by))
				if (return_trans):
					return "({} - {}) / {}({} - {})".format(self.alias, cmin, "NULLIFZERO" if (nullifzero) else "", cmax, cmin)
				else:
					self.transformations += [("({} - {}) / {}({} - {})".format("{}", cmin, "NULLIFZERO" if (nullifzero) else "", cmax, cmin), "float", "float")]
			self.parent.add_to_history("[Normalize]: The vColumn '{}' was normalized with the method '{}'.".format(self.alias,method))
		else:
			raise TypeError("The Virtual Column must be numerical for Normalization")
		return (self.parent)
	# 
	def nsmallest(self, n: int = 10):
		check_types([("n", n, [int, float], False)])
		query = "SELECT * FROM {} WHERE {} IS NOT NULL ORDER BY {} ASC LIMIT {}".format(self.parent.genSQL(), self.alias, self.alias, n)
		return (to_tablesample(query, self.parent.VERTICA_ML_PYTHON_VARIABLES["cursor"], name = "nsmallest"))
	# 
	def numh(self, method: str = "auto"):
		check_types([("method", method, ["sturges", "freedman_diaconis", "fd", "auto"], True)])
		if (self.isnum()):
			result = self.summarize_numcol().values["value"]
			count, vColumn_min, vColumn_025, vColumn_075, vColumn_max = result[0], result[3], result[4], result[6], result[7]
		elif (self.isdate()):
			min_date = self.min()
			table = "(SELECT DATEDIFF('second', '{}'::timestamp, {}) AS {} FROM {}) best_h_date_table".format(min_date, self.alias, self.alias, self.parent.genSQL())
			query = "SELECT COUNT({}) AS NAs, MIN({}) AS min, APPROXIMATE_PERCENTILE({} USING PARAMETERS percentile = 0.25) AS Q1, APPROXIMATE_PERCENTILE({} USING PARAMETERS percentile = 0.75) AS Q3, MAX({}) AS max FROM {}".format(self.alias, self.alias, self.alias, self.alias, self.alias, table)
			self.executeSQL(query, title = "AGGR to compute h")
			result = self.parent.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchone()
			count, vColumn_min, vColumn_025, vColumn_075, vColumn_max  = result
		else:
			raise TypeError("numh is only available on type numeric|date")
		sturges = max(float(vColumn_max - vColumn_min) / int(math.floor(math.log(count, 2) + 1)), 1e-99)
		fd = max(2.0 * (vColumn_075 - vColumn_025) / (count) ** (1.0 / 3.0), 1e-99)
		if (method.lower() == "sturges"):
			best_h = (sturges)
		elif (method.lower() in ("freedman_diaconis", "fd")):
			best_h = (fd) 
		else:
			best_h = (max(sturges, fd))
		best_h = max(math.floor(best_h), 1) if (self.category() == "int") else best_h
		return (best_h)
	# 
	def nunique(self, approx: bool = False):
		check_types([("approx", approx, [bool], False)])
		if (approx):
			return (self.aggregate(func = ["approx_unique"]).values[self.alias][0])
		else:
			return (self.aggregate(func = ["unique"]).values[self.alias][0])
	#
	def pct_change(self, order_by: list, by: list = []):
		check_types([("order_by", order_by, [list], False), ("by", by, [list], False)])
		columns_check(order_by + by, self.parent)
		order_by, by = vdf_columns_names(order_by, self.parent), vdf_columns_names(by, self.parent)
		by = "PARTITION BY {}".format(", ".join(by)) if (by) else ""
		return (self.apply(func = "LEAD({}) OVER ({} ORDER BY {}) / {}".format("{}", by, ", ".join(order_by), "{}")))
	# 
	def pie(self,
			method: str = "density",
			of: str = "",
			max_cardinality: int = 6,
			h: float = 0):
		check_types([("method", method, ["density", "count", "avg", "min", "max", "sum"], True), ("of", of, [str], False), ("max_cardinality", max_cardinality, [int, float], False), ("h", h, [int, float], False)])
		if (of):
			columns_check([of], self.parent)
			of = vdf_columns_names([of], self.parent)[0]
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
		check_types([("ts", ts, [str], False), ("by", by, [str], False), ("start_date", start_date, [str], False), ("end_date", end_date, [str], False), ("color", color, [str], False), ("area", area, [bool], False)])
		if (by):
			columns_check([by], self.parent)
			by = vdf_columns_names([by], self.parent)[0]
		from vertica_ml_python.plot import ts_plot
		ts_plot(self, ts, by, start_date, end_date, color, area)
		return (self.parent)	
	# 
	def pow(self, x: float):
		check_types([("x", x, [int, float], False)])
		return (self.apply(func = "POWER({}, {})".format("{}", x)))
	#
	def prev(self, order_by: list, by: list = []):
		check_types([("order_by", order_by, [list], False), ("by", by, [list], False)])
		columns_check(order_by + by, self.parent)
		order_by, by = vdf_columns_names(order_by, self.parent), vdf_columns_names(by, self.parent)
		by = "PARTITION BY {}".format(", ".join(by)) if (by) else ""
		return (self.apply(func = "LAG({}) OVER ({} ORDER BY {})".format("{}", by, ", ".join(order_by))))
	#
	def prod(self):
		return (self.product())
	def product(self):
		return (self.aggregate(func = ["prod"]).values[self.alias][0])
	# 
	def quantile(self, x: float):
		check_types([("x", x, [int, float], False)])
		return (self.aggregate(func = ["{}%".format(x * 100)]).values[self.alias][0])
	#
	def rename(self, new_name: str):
		check_types([("new_name", new_name, [str], False)])
		old_name = str_column(self.alias)
		new_name = str_column(new_name)
		if (column_check_ambiguous(new_name, self.parent.get_columns())):
			raise ValueError("A Virtual Column has already the alias {}.\nBy changing the parameter 'new_name', you'll be able to solve this issue.".format(new_name))
		self.add_copy(new_name)
		parent = self.drop(add_history = False)
		parent.add_to_history("[Rename]: The Virtual Column {} was renamed '{}'.".format(old_name, new_name))
		return (parent)
	# 
	def round(self, n: int):
		check_types([("n", n, [int, float], False)])
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
		check_types([("length", length, [int, float], False), ("unit", unit, [str], False), ("start", start, [bool], False)])
		start_or_end = "START" if (start) else "END"
		return (self.apply(func = "TIME_SLICE({}, {}, '{}', '{}')".format("{}", length, unit.upper(), start_or_end)))
	#
	def store_usage(self):
		self.parent.VERTICA_ML_PYTHON_VARIABLES["cursor"].execute("SELECT ZEROIFNULL(SUM(LENGTH({}::varchar))) FROM {}".format(convert_special_type(self.category(), False, self.alias), self.parent.genSQL()))
		return (self.parent.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchone()[0])
	#
	def summarize_numcol(self):
		cast = "::int" if (self.ctype() == "boolean") else ""
		# For Vertica 9.0 and higher
		try:
			query = "SELECT SUMMARIZE_NUMCOL({}{}) OVER () FROM {}".format(self.alias, cast, self.parent.genSQL())
			self.executeSQL(query = query, title = "Compute a direct SUMMARIZE_NUMCOL({})".format(self.alias))
			result = self.parent.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchone()
			val = [float(result[i]) for i in range(1, len(result))]
		# For all versions of Vertica
		except:
			try:
				query = "SELECT COUNT({}) AS count, AVG({}{}) AS mean, STDDEV({}{}) AS std, MIN({}{}) AS min, APPROXIMATE_PERCENTILE ({}{} USING PARAMETERS percentile = 0.25) AS Q1, APPROXIMATE_PERCENTILE ({}{} USING PARAMETERS percentile = 0.5) AS Median, APPROXIMATE_PERCENTILE ({}{} USING PARAMETERS percentile = 0.75) AS Q3, MAX({}{}) AS max FROM {}"
				query = query.format(self.alias, self.alias, cast, self.alias, cast, self.alias, cast, self.alias, cast, self.alias, cast, self.alias, cast, self.alias, cast, self.parent.genSQL())
				self.executeSQL(query = query, title = "Compute a manual SUMMARIZE_NUMCOL({})".format(self.alias))
				result = self.parent.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchone()
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
		check_types([("pat", pat, [str], False)])
		return (self.apply(func = "REGEXP_COUNT({}, '{}') > 0".format("{}", pat.replace("'", "''")))) 
	# 
	def str_count(self, pat: str):
		check_types([("pat", pat, [str], False)])
		return (self.apply(func = "REGEXP_COUNT({}, '{}')".format("{}", pat.replace("'", "''")))) 
	# 
	def str_extract(self, pat: str):
		check_types([("pat", pat, [str], False)])
		return (self.apply(func = "REGEXP_SUBSTR({}, '{}')".format("{}", pat.replace("'", "''")))) 
	#
	def str_replace(self, to_replace: str, value: str = ""):
		check_types([("to_replace", to_replace, [str], False), ("value", value, [str], False)])
		return (self.apply(func = "REGEXP_REPLACE({}, '{}', '{}')".format("{}", to_replace.replace("'", "''"), value.replace("'", "''")))) 
	# 
	def str_slice(self, start: int, step: int):
		check_types([("start", start, [int, float], False), ("step", step, [int, float], False)])
		return (self.apply(func = "SUBSTR({}, {}, {})".format("{}", start, step))) 
	# 
	def sub(self, x: float):
		check_types([("x", x, [int, float, str], False)])
		if (self.isdate()):
			return (self.apply(func = "TIMESTAMPADD(SECOND, -({}), {})".format(x, "{}")))
		else:
			return (self.apply(func = "{} - ({})".format("{}", x)))
	# 
	def sum(self):
		return (self.aggregate(["sum"]).values[self.alias][0])
	# 
	def tail(self, limit: int = 5, offset: int = 0):
		check_types([("limit", limit, [int, float], False), ("offset", offset, [int, float], False)])
		tail = to_tablesample("SELECT {} AS {} FROM {} LIMIT {} OFFSET {}".format(convert_special_type(self.category(), False, self.alias), self.alias, self.parent.genSQL(), limit, offset), self.parent.VERTICA_ML_PYTHON_VARIABLES["cursor"])
		tail.count = self.parent.shape()[0]
		tail.offset = offset
		tail.dtype[self.alias.replace('"', '')] = self.ctype()
		tail.name = self.alias.replace('"', '')
		return (tail)
	# 
	def topk(self, k: int = -1, dropna: bool = True):
		check_types([("k", k, [int, float], False), ("dropna", dropna, [bool], False)])
		try:
			topk = "" if (k < 1) else "TOPK = {},".format(k)
			query = "SELECT SUMMARIZE_CATCOL({}::varchar USING PARAMETERS {} WITH_TOTALCOUNT = False) OVER () FROM {}".format(self.alias, topk, self.parent.genSQL())
			if (dropna):
				query += " WHERE {} IS NOT NULL".format(self.alias)
			self.executeSQL(query, title = "Compute the TOPK categories of {}".format(self.alias))
		except:
			topk = "" if (k < 1) else "LIMIT {}".format(k)
			query = "SELECT {} AS {}, COUNT(*) AS cnt, 100 * COUNT(*) / {} AS percent FROM {} GROUP BY {} ORDER BY cnt DESC {}".format(convert_special_type(self.category(), True, self.alias), self.alias, self.parent.shape()[0], self.parent.genSQL(), self.alias, topk)
			self.executeSQL(query, title = "Compute the TOPK categories of {}".format(self.alias))
		result = self.parent.VERTICA_ML_PYTHON_VARIABLES["cursor"].fetchall()
		values = {"index" : [item[0] for item in result], "count" : [item[1] for item in result], "percent" : [round(item[2], 3) for item in result]}
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