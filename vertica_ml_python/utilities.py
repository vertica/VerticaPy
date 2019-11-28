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

#
def category_from_type(ctype: str = ""):
	if not(ctype == ""):
		if (ctype[0:4] == "date") or (ctype[0:4] == "time") or (ctype[0:8] == "interval"):
			category = "date"
		elif ((ctype[0:3] == "int") or (ctype[0:4] == "bool")):
			category = "int"
		elif ((ctype[0:7] == "numeric") or (ctype[0:5] == "float")):
			category = "float"
		else:
			category = "text"
	else:
		category = "undefined"
	return category
#
def drop_model(name: str, 
			   cursor, 
			   print_info: bool = True):
	cursor.execute("SELECT 1;")
	try:
		query = "DROP MODEL {};".format(name)
		cursor.execute(query)
		if (print_info):
			print("The model {} was successfully dropped.".format(name))
	except:
		print("/!\\ Warning: The model {} doesn't exist !".format(name))
# 
def drop_table(name: str, 
			   cursor, 
			   print_info: bool = True):
	cursor.execute("SELECT 1;")
	try:
		query="DROP TABLE {};".format(name)
		cursor.execute(query)
		if (print_info):
			print("The table {} was successfully dropped.".format(name))
	except:
		print("/!\\ Warning: The table {} doesn't exist !".format(name))
# 
def drop_text_index(name: str, 
			   		cursor, 
			   		print_info: bool = True):
	cursor.execute("SELECT 1;")
	try:
		query="DROP TEXT INDEX {};".format(name)
		cursor.execute(query)
		if (print_info):
			print("The index {} was successfully dropped.".format(name))
	except:
		print("/!\\ Warning: The table {} doesn't exist !".format(name))
# 
def drop_view(name: str,
			  cursor,
			  print_info: bool = True):
	cursor.execute("SELECT 1;")
	try:
		query="DROP VIEW {};".format(name)
		cursor.execute(query)
		if (print_info):
			print("The view {} was successfully dropped.".format(name))
	except:
		print("/!\\ Warning: The view {} doesn't exist !".format(name))

# 
def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False # Terminal running IPython
        else:
            return False # Other type (?)
    except NameError:
        return False # Probably standard Python interpreter
#
def load_model(name: str, cursor, test_relation: str = ""):
	try:
		info = cursor.execute("SELECT GET_MODEL_ATTRIBUTE (USING PARAMETERS model_name = '" + name + "', attr_name = 'call_string')").fetchone()[0].replace('\n', ' ')
	except:
		try:
			info = cursor.execute("SELECT GET_MODEL_SUMMARY (USING PARAMETERS model_name = '" + name + "')").fetchone()[0].replace('\n', ' ')
			info = "kmeans(" + info.split("kmeans(")[1]
		except:
			from vertica_ml_python.learn.preprocessing import Normalizer
			model = Normalizer(name, cursor)
			model.param = to_tablesample(query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'details')".format(name), cursor = cursor)
			model.param.table_info = False
			model.X = ['"' + item + '"' for item in model.param.values["column_name"]]
			if ("avg" in model.param.values):
				model.method = "zscore" 
			elif ("max" in model.param.values):
				model.method = "minmax" 
			else:
				model.method = "robust_zscore"
			return model
	try:
		info = info.split("SELECT ")[1].split("(")
	except:
		info = info.split("(")
	model_type = info[0].lower()
	info = info[1].split(")")[0].replace(" ", '').split("USINGPARAMETERS")
	if (model_type == "svm_classifier"):
		parameters = "".join(info[1].split("class_weights=")[1].split("'"))
		parameters = parameters[3:len(parameters)].split(",")
		del parameters[0]
		parameters += ["class_weights=" + info[1].split("class_weights=")[1].split("'")[1]]
	elif (model_type != "svd"):
		parameters = info[1].split(",")
	if (model_type != "svd"):
		parameters = [item.split("=") for item in parameters]
		parameters_dict = {}
		for item in parameters:
			parameters_dict[item[0]] = item[1]
	info = info[0]
	if (model_type == "rf_regressor"):
		from vertica_ml_python.learn.ensemble import RandomForestRegressor
		model = RandomForestRegressor(name, cursor, int(parameters_dict['ntree']), int(parameters_dict['mtry']), int(parameters_dict['max_breadth']), float(parameters_dict['sampling_size']), int(parameters_dict['max_depth']), int(parameters_dict['min_leaf_size']), float(parameters_dict['min_info_gain']), int(parameters_dict['nbins']))
	elif (model_type == "rf_classifier"):
		from vertica_ml_python.learn.ensemble import RandomForestClassifier
		model = RandomForestClassifier(name, cursor, int(parameters_dict['ntree']), int(parameters_dict['mtry']), int(parameters_dict['max_breadth']), float(parameters_dict['sampling_size']), int(parameters_dict['max_depth']), int(parameters_dict['min_leaf_size']), float(parameters_dict['min_info_gain']), int(parameters_dict['nbins']))
	elif (model_type == "logistic_reg"):
		from vertica_ml_python.learn.linear_model import LogisticRegression
		model = LogisticRegression(name, cursor, parameters_dict['regularization'], float(parameters_dict['epsilon']), float(parameters_dict['lambda']), int(parameters_dict['max_iterations']), parameters_dict['optimizer'], float(parameters_dict['alpha']))
	elif (model_type == "linear_reg"):
		from vertica_ml_python.learn.linear_model import ElasticNet
		model = ElasticNet(name, cursor, parameters_dict['regularization'], float(parameters_dict['epsilon']), float(parameters_dict['lambda']), int(parameters_dict['max_iterations']), parameters_dict['optimizer'], float(parameters_dict['alpha']))
	elif (model_type == "naive_bayes"):
		from vertica_ml_python.learn.naive_bayes import MultinomialNB
		model = MultinomialNB(name, cursor, float(parameters_dict['alpha']))
	elif (model_type == "svm_regressor"):
		from vertica_ml_python.learn.svm import LinearSVR
		model = LinearSVR(name, cursor, float(parameters_dict['epsilon']), float(parameters_dict['C']), True, float(parameters_dict['intercept_scaling']), parameters_dict['intercept_mode'], float(parameters_dict['error_tolerance']), int(parameters_dict['max_iterations']))
	elif (model_type == "svm_classifier"):
		from vertica_ml_python.learn.svm import LinearSVC
		model = LinearSVC(name, cursor, float(parameters_dict['epsilon']), float(parameters_dict['C']), True, float(parameters_dict['intercept_scaling']), parameters_dict['intercept_mode'], [float(item) for item in parameters_dict['class_weights'].split(",")], int(parameters_dict['max_iterations']))
	elif (model_type == "kmeans"):
		from vertica_ml_python.learn.cluster import KMeans
		model = KMeans(name, cursor, -1, parameters_dict['init_method'], int(parameters_dict['max_iterations']), float(parameters_dict['epsilon']))
	elif (model_type == "pca"):
		from vertica_ml_python.learn.decomposition import PCA
		model = PCA(name, cursor, 0, bool(parameters_dict['scale']))
	elif (model_type == "svd"):
		from vertica_ml_python.learn.decomposition import SVD
		model = SVD(name, cursor)
	elif (model_type == "one_hot_encoder_fit"):
		from vertica_ml_python.learn.preprocessing import OneHotEncoder
		model = OneHotEncoder(name, cursor)
	model.input_relation = info.split(",")[1].replace("'", '').replace('\\', '')
	model.test_relation = test_relation if (test_relation) else model.input_relation
	if (model_type not in ("kmeans", "pca", "svd", "one_hot_encoder_fit")):
		model.X = info.split(",")[3:len(info.split(","))]
		model.X = [item.replace("'", '').replace('\\', '') for item in model.X]
		model.y = info.split(",")[2].replace("'", '').replace('\\', '')
	elif (model_type in ("pca")):
		model.X = info.split(",")[2:len(info.split(","))]
		model.X = [item.replace("'", '').replace('\\', '') for item in model.X]
		model.components = to_tablesample(query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'principal_components')".format(name), cursor = cursor)
		model.components.table_info = False
		model.explained_variance = to_tablesample(query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'singular_values')".format(name), cursor = cursor)
		model.explained_variance.table_info = False
		model.mean = to_tablesample(query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'columns')".format(name), cursor = cursor)
		model.mean.table_info = False
	elif (model_type in ("svd")):
		model.X = info.split(",")[2:len(info.split(","))]
		model.X = [item.replace("'", '').replace('\\', '') for item in model.X]
		model.singular_values = to_tablesample(query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'right_singular_vectors')".format(name), cursor = cursor)
		model.singular_values.table_info = False
		model.explained_variance = to_tablesample(query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'singular_values')".format(name), cursor = cursor)
		model.explained_variance.table_info = False
	elif (model_type in ("one_hot_encoder_fit")):
		model.X = info.split(",")[2:len(info.split(","))]
		model.X = [item.replace("'", '').replace('\\', '') for item in model.X]
		model.param = to_tablesample(query = "SELECT category_name, category_level::varchar, category_level_index FROM (SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'integer_categories')) x UNION ALL SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'varchar_categories')".format(name, name), cursor = cursor)
		model.param.table_info = False
	else:
		model.X = info.split(",")[2:len(info.split(",")) - 1]
		model.X = [item.replace("'", '').replace('\\', '') for item in model.X]
		model.n_cluster = int(info.split(",")[-1])
		model.cluster_centers = to_tablesample(query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'centers')".format(name), cursor = cursor)
		model.cluster_centers.table_info = False
		query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'metrics')".format(name)
		cursor.execute(query)
		result = cursor.fetchone()[0]
		values = {"index": ["Between-Cluster Sum of Squares", "Total Sum of Squares", "Total Within-Cluster Sum of Squares", "Between-Cluster SS / Total SS", "converged"]}
		values["value"] = [float(result.split("Between-Cluster Sum of Squares: ")[1].split("\n")[0]), float(result.split("Total Sum of Squares: ")[1].split("\n")[0]), float(result.split("Total Within-Cluster Sum of Squares: ")[1].split("\n")[0]), float(result.split("Between-Cluster Sum of Squares: ")[1].split("\n")[0]) / float(result.split("Total Sum of Squares: ")[1].split("\n")[0]), result.split("Converged: ")[1].split("\n")[0] == "True"] 
		model.metrics = tablesample(values, table_info = False)
	if (model.type == "classifier"):
		classes = cursor.execute("SELECT DISTINCT {} FROM {} WHERE {} IS NOT NULL ORDER BY 1".format(model.y, model.input_relation, model.y)).fetchall()
		model.classes = [item[0] for item in classes]
	return (model)
# 
def pandas_to_vertica(df, cur, name: str):
	path = "{}.csv".format(name)
	df.to_csv(path, index = False)
	from vertica_ml_python import read_csv
	read_csv(path, cur)
	os.remove(path)
# 
def print_table(data_columns, is_finished = True, offset = 0, repeat_first_column = False, first_element = ""):
	data_columns_rep = [] + data_columns
	if (repeat_first_column):
		del data_columns_rep[0]
		columns_ljust_val = min(len(max([str(item) for item in data_columns[0]], key = len)) + 4, 40)
	else:
		columns_ljust_val = len(str(len(data_columns[0]))) + 2
	try:
		import shutil
		screen_columns = shutil.get_terminal_size().columns
	except:
		screen_rows, screen_columns = os.popen('stty size', 'r').read().split()
	formatted_text = ""
	rjust_val = []
	for idx in range(0,len(data_columns_rep)):
		rjust_val += [min(len(max([str(item) for item in data_columns_rep[idx]], key = len)) + 2, 40)]
	total_column_len = len(data_columns_rep[0])
	while (rjust_val != []):
		columns_to_print = [data_columns_rep[0]]
		columns_rjust_val = [rjust_val[0]]
		max_screen_size = int(screen_columns) - 14 - int(rjust_val[0])
		del data_columns_rep[0]
		del rjust_val[0]
		while ((max_screen_size > 0) and (rjust_val != [])):
			columns_to_print += [data_columns_rep[0]]
			columns_rjust_val += [rjust_val[0]]
			max_screen_size = max_screen_size-7-int(rjust_val[0])
			del data_columns_rep[0]
			del rjust_val[0]
		if (repeat_first_column):
			columns_to_print = [data_columns[0]] + columns_to_print
		else:
			columns_to_print=[[i - 1 + offset for i in range(0,total_column_len)]] + columns_to_print
		columns_to_print[0][0] = first_element
		columns_rjust_val = [columns_ljust_val]+columns_rjust_val
		column_count = len(columns_to_print)
		for i in range(0,total_column_len):
			for k in range(0,column_count):
				val = columns_to_print[k][i]
				if len(str(val)) > 40:
					val = str(val)[0:37] + "..."
				if (k == 0):
					formatted_text += str(val).ljust(columns_rjust_val[k])
				else:
					formatted_text += str(val).rjust(columns_rjust_val[k])+"  "
			if ((rjust_val != [])):
				formatted_text += " \\\\"
			formatted_text += "\n"	
		if (not(is_finished) and (i == total_column_len-1)):
			for k in range(0,column_count):
				if (k==0):
					formatted_text += "...".ljust(columns_rjust_val[k])
				else:
					formatted_text += "...".rjust(columns_rjust_val[k])+"  "
			if (rjust_val != []):
				formatted_text += " \\\\"
			formatted_text += "\n"
	try:	
		if (isnotebook()):
			from IPython.core.display import HTML, display
			if not(repeat_first_column):
				data_columns=[[""] + list(range(0 + offset, len(data_columns[0]) - 1 + offset))] + data_columns
			m = len(data_columns)
			n = len(data_columns[0])
			html_table = "<table style=\"border-collapse: collapse; border: 2px solid white\">"
			for i in range(n):
				html_table += "<tr style=\"{border: 1px solid white;}\">"
				for j in range(m):
					if (j == 0):
						html_table += "<td style=\"border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white\"><b>" + str(data_columns[j][i]) + "</b></td>"
					elif (i == 0):
						html_table += "<td style=\"font-size:1.02em;background-color:#214579;color:white\"><b>" + str(data_columns[j][i]) + "</b></td>"
					else:
						html_table += "<td style=\"border: 1px solid white;\">" + str(data_columns[j][i]) + "</td>"
				html_table += "</tr>"
			if not(is_finished):
				html_table += "<tr>"
				for j in range(m):
					if (j == 0):
						html_table += "<td style=\"border-top: 1px solid white;background-color:#214579;color:white\"></td>"
					else:
						html_table += "<td style=\"border: 1px solid white;\">...</td>"
				html_table += "</tr>"
			html_table += "</table>"
			display(HTML(html_table))
			return "<object>  "
		else:
			return formatted_text
	except:
		return formatted_text
def read_csv(path: str, 
			 cursor, 
			 schema: str = 'public', 
			 table_name: str = '', 
			 delimiter: str = ',', 
			 header_names: list = [],
			 dtype: dict = {}, 
			 null: str = '', 
			 enclosed_by: str = '"', 
			 escape: str = '\\', 
			 skip: int = 1, 
			 genSQL: bool = False,
			 return_dlist: bool = False,
			 parse_n_lines: int = -1):
	table_name = table_name if (table_name) else path.split("/")[-1].split(".csv")[0]
	query = "SELECT column_name FROM columns WHERE table_name='{}' AND table_schema='{}'".format(table_name, schema)
	result = cursor.execute(query).fetchall()
	if (result != []):
		raise Exception("The table {} already exists !".format(table_name))
	else:
		input_relation = '{}.{}'.format(schema, table_name)
		if (len(header_names) == 0):
			f = open(path,'r')
			header_names  = f.readline().replace('\n', '').replace('"', '').split(delimiter)
			f.close()
			if (parse_n_lines > 0):
				f = open(path,'r')
				f2 = open(path[0:-4] + "_vpython_copy.csv",'w')
				for i in range(parse_n_lines + 1):
					line = f.readline()
					f2.write(line)
				f.close()
				f2.close()
				path_test = path[0:-4] + "_vpython_copy.csv"
			else:
				path_test = path
			flex_name = "_vpython" + str(np.random.randint(10000000)) + "_flex_"
			cursor.execute("CREATE FLEX LOCAL TEMP TABLE {}(x int) ON COMMIT PRESERVE ROWS".format(flex_name))
			query = "COPY {} FROM LOCAL '{}' parser fcsvparser(delimiter = '{}', enclosed_by = '{}', escape = '{}') NULL '{}'"
			cursor.execute(query.format(flex_name, path_test, delimiter, enclosed_by, escape, null))
			query = "SELECT compute_flextable_keys('{}');".format(flex_name)
			cursor.execute(query)
			query = "SELECT key_name, data_type_guess FROM {}_keys".format(flex_name)
			cursor.execute(query)
			result = cursor.fetchall()
			if (return_dlist):
				return result
			for column_dtype in result:
				if column_dtype[0] not in dtype:
					try:
						if ("Varchar" not in column_dtype[1]):
							query='SELECT (CASE WHEN "{}"=\'{}\' THEN NULL ELSE "{}" END)::{} AS "{}" FROM {} WHERE "{}" IS NOT NULL LIMIT 1000'.format(column_dtype[0], null, column_dtype[0], column_dtype[1], column_dtype[0], flex_name, column_dtype[0])
							cursor.execute(query)
						dtype[column_dtype[0]] = column_dtype[1]
					except:
						dtype[column_dtype[0]] = "Varchar(100)"
			cursor.execute("DROP TABLE " + flex_name)
		if (parse_n_lines > 0):
			os.remove(path[0:-4] + "_vpython_copy.csv")
		query = "CREATE TABLE {}({})".format(input_relation, ", ".join(['"{}" {}'.format(column, dtype[column]) for column in header_names]))
		if (genSQL):
			print(query)
		cursor.execute(query)
		query="COPY {}({}) FROM LOCAL '{}' DELIMITER '{}' NULL '{}' ENCLOSED BY '{}' ESCAPE AS '{}' SKIP {};".format(
			input_relation,", ".join(['"' + column + '"' for column in header_names]), path, delimiter, null, enclosed_by, escape, skip)
		if (genSQL):
			print(query)
		cursor.execute(query)
		print("The table {} has been successfully created.".format(input_relation))
		from vertica_ml_python import vDataframe
		return vDataframe(input_relation, cursor)
# 
def read_vdf(path: str, cursor):
	file = open(path, "r")
	save =  "from vertica_ml_python import vDataframe\nfrom vertica_ml_python.vcolumn import vColumn\n" + "".join(file.readlines())
	file.close()
	vdf = {}
	exec(save, globals(), vdf)
	vdf = vdf["vdf_save"]
	vdf.cursor = cursor
	return (vdf)
#
class tablesample:
	# Initialization
	def  __init__(self, 
				  values: dict = {}, 
				  dtype: dict = {}, 
				  name: str = "Sample", 
				  count: int = 0, 
				  offset: int = 0, 
				  table_info: bool = True):
		self.values = values
		self.dtype = dtype
		self.count = count
		self.offset = offset
		self.table_info = table_info
		self.name = name
		for column in values:
			if column not in dtype:
				self.dtype[column] = "undefined"
	# Representation
	def __repr__(self):
		if (len(self.values) == 0):
			return ""
		data_columns = [[column] + self.values[column] for column in self.values]
		formatted_text = print_table(data_columns, is_finished = (self.count <= len(data_columns[0]) + self.offset), offset = self.offset, repeat_first_column = ("index" in self.values))
		if (self.table_info):
			if (len(self.values) == 1):
				column = list(self.values.keys())[0]
				formatted_text += "Name: {}, Number of rows: {}, dtype: {}".format(column, max(len(data_columns[0]) - 1, self.count), self.dtype[column]) 
			else:
				formatted_text += "Name: {}, Number of rows: {}, Number of columns: {}".format(self.name, max(len(data_columns[0]) - 1, self.count), len(data_columns)) 
		else:
			formatted_text = formatted_text[0:-2]
		return formatted_text
	# 
	def transpose(self):
		index = [column for column in self.values]
		first_item = list(self.values.keys())[0]
		columns =[[] for i in range(len(self.values[first_item]))]
		for column in self.values:
			for idx, item in enumerate(self.values[column]):
				columns[idx] += [item]
		columns = [index] + columns
		values = {}
		for item in columns:
			values[item[0]] = item[1:len(item)]
		self.values = values
		return (self)
	#
	def to_pandas(self):
		import pandas as pd
		if ("index" in self.values):
			df = pd.DataFrame(data = self.values, index = self.values["index"])
			return df.drop(columns = ['index'])
		else:
			return pd.DataFrame(data = self.values)
	def to_sql(self):
		sql = []
		n = len(self.values[list(self.values.keys())[0]])
		for i in range(n):
			row = [] 
			for column in self.values:
				val = self.values[column][i]
				if (type(val) == str):
					val = "'" + val.replace("'", "''") + "'"
				elif (val == None):
					val = "NULL"
				elif (math.isnan(val)):
					val = "NULL"
				row += ["{} AS {}".format(val, '"' + column.replace('"', '') + '"')]
			sql += ["(SELECT {})".format(", ".join(row))]
		sql = " UNION ALL ".join(sql)
		return (sql)
	def to_vdf(self, cursor = None, dsn: str = ""):
		from vertica_ml_python import vdf_from_relation
		relation = "({}) sql_relation".format(self.to_sql())
		return (vdf_from_relation(relation, cursor = cursor, dsn = dsn)) 
#
def to_tablesample(query: str, cursor, name = "Sample"):
	cursor.execute(query)
	result = cursor.fetchall()
	columns = [column[0] for column in cursor.description]
	data_columns = [[item] for item in columns]
	data = [item for item in result]
	for row in data:
		for idx, val in enumerate(row):
			data_columns[idx] += [val]
	values = {}
	for column in data_columns:
		values[column[0]] = column[1:len(column)]
	return tablesample(values = values, name = name)
#
def vertica_cursor(dsn: str):
	try:
		import vertica_python
		cursor = vertica_python.connect(** to_vertica_python_format(dsn)).cursor()
	except:
		print("Failed to connect with vertica_python, try a connection with pyodbc")
		import pyodbc
		cursor = pyodbc.connect("DSN=" + dsn).cursor()
	return (cursor)
def read_dsn(dsn: str):
	f = open(os.environ['ODBCINI'], "r")
	odbc = f.read()
	f.close()
	if ("[{}]".format(dsn) not in odbc):
		raise ValueError("The DSN '{}' doesn't exist".format(dsn))
	odbc = odbc.split("[{}]\n".format(dsn))[1].split("\n\n")[0].split("\n")
	dsn = {}
	for elem in odbc:
		info = elem.replace(' ','').split('=')
		dsn[info[0].lower()] = info[1]
	return (dsn)
def to_vertica_python_format(dsn: str):
	dsn = read_dsn(dsn)
	conn_info = {'host': dsn["servername"], 'port': 5433, 'user': dsn["uid"], 'password': dsn["pwd"], 'database': dsn["database"]}
	return (conn_info)
#
def vdf_from_relation(relation: str, name: str = "VDF", cursor = None, dsn: str = ""):
	from vertica_ml_python import vDataframe
	vdf = vDataframe("", empty = True)
	vdf.dsn = dsn
	if (cursor == None):
		from vertica_ml_python import vertica_cursor
		cursor = vertica_cursor(dsn)
	vdf.input_relation = name
	vdf.main_relation = relation
	vdf.schema = ""
	vdf.cursor = cursor
	vdf.query_on = False
	vdf.time_on = False
	cursor.execute("DROP TABLE IF EXISTS _vpython_{}_test_; CREATE TEMPORARY TABLE _vpython_{}_test_ AS SELECT * FROM {} LIMIT 10;".format(name, name, relation))
	cursor.execute("SELECT column_name, data_type FROM columns where table_name = '_vpython_{}_test_'".format(name))
	result = cursor.fetchall()
	cursor.execute("DROP TABLE IF EXISTS _vpython_{}_test_;".format(name))
	vdf.columns = ['"' + item[0] + '"' for item in result]
	vdf.where = []
	vdf.order_by = []
	vdf.exclude_columns = []
	vdf.history = []
	vdf.saving = []
	for column, ctype in result:
		column = '"' + column + '"'
		from vertica_ml_python.vcolumn import vColumn
		new_vColumn = vColumn(column, parent = vdf, transformations = [(column, ctype, category_from_type(ctype = ctype))])
		setattr(vdf, column, new_vColumn)
		setattr(vdf, column[1:-1], new_vColumn)
	return (vdf)
	