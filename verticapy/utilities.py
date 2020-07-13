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
# Standard Python Modules
import random, os, math, shutil, re, time
# VerticaPy Modules
from verticapy.toolbox import *
from verticapy.connections.connect import read_auto_connect
#
#---#
def drop_model(name: str, 
			   cursor = None, 
			   print_info: bool = True,
			   raise_error: bool = False):
	"""
---------------------------------------------------------------------------
Drops the input model.

Parameters
----------
name: str
	Model name.
cursor: DBcursor, optional
	Vertica DB cursor.
print_info: bool, optional
	If set to True, displays the result of the query.
raise_error: bool, optional
	If the model couldn't be dropped, raises the entire error instead of
	displaying a warning.
	"""
	check_types([
		("name", name, [str], False),
		("print_info", print_info, [bool], False),
		("raise_error", raise_error, [bool], False)])
	if not(cursor):
		conn = read_auto_connect()
		cursor = conn.cursor()
	else:
		conn = False
		check_cursor(cursor)
	try:
		query = "DROP MODEL {};".format(name)
		cursor.execute(query)
		if (conn):
			conn.close()
		if (print_info):
			print("The model {} was successfully dropped.".format(name))
	except:
		if (conn):
			conn.close()
		if (raise_error):
			raise
		else:
			print("\u26A0 Warning: The model {} doesn't exist or can not be dropped !\nUse parameter: raise_error = True to get more information.".format(name))
#---#
def drop_table(name: str, 
			   cursor = None, 
			   print_info: bool = True,
			   raise_error: bool = False):
	"""
---------------------------------------------------------------------------
Drops the input table.

Parameters
----------
name: str
	Table name.
cursor: DBcursor, optional
	Vertica DB cursor. 
print_info: bool, optional
	If set to True, displays the result of the query.
raise_error: bool, optional
	If the table couldn't be dropped, raises the entire error instead of
	displaying a warning.
	"""
	check_types([
		("name", name, [str], False),
		("print_info", print_info, [bool], False),
		("raise_error", raise_error, [bool], False)])
	if not(cursor):
		conn = read_auto_connect()
		cursor = conn.cursor()
	else:
		conn = False
		check_cursor(cursor)
	try:
		query="DROP TABLE {};".format(name)
		cursor.execute(query)
		if (conn):
			conn.close()
		if (print_info):
			print("The table {} was successfully dropped.".format(name))
	except:
		if (conn):
			conn.close()
		if (raise_error):
			raise
		else:
			print("\u26A0 Warning: The table {} doesn't exist or can not be dropped !\nUse parameter: raise_error = True to get more information.".format(name))
#---#
def drop_text_index(name: str, 
			   		cursor = None, 
			   		print_info: bool = True,
			   		raise_error: bool = False):
	"""
---------------------------------------------------------------------------
Drops the input text index.

Parameters
----------
name: str
	Text index name.
cursor: DBcursor, optional
	Vertica DB cursor. 
print_info: bool, optional
	If set to true, displays the result of the query.
raise_error: bool, optional
	If the text index couldn't be dropped, raises the entire error instead 
	of displaying a warning.
	"""
	check_types([
		("name", name, [str], False),
		("print_info", print_info, [bool], False),
		("raise_error", raise_error, [bool], False)])
	if not(cursor):
		conn = read_auto_connect()
		cursor = conn.cursor()
	else:
		conn = False
		check_cursor(cursor)
	try:
		query="DROP TEXT INDEX {};".format(name)
		cursor.execute(query)
		if (conn):
			conn.close()
		if (print_info):
			print("The text index {} was successfully dropped.".format(name))
	except:
		if (conn):
			conn.close()
		if (raise_error):
			raise
		else:
			print("\u26A0 Warning: The text index {} doesn't exist or can not be dropped !\nUse parameter: raise_error = True to get more information.".format(name))
#---#
def drop_view(name: str,
			  cursor = None,
			  print_info: bool = True,
			  raise_error: bool = False):
	"""
---------------------------------------------------------------------------
Drops the input view.

Parameters
----------
name: str
	View name.
cursor: DBcursor, optional
	Vertica DB cursor. 
print_info: bool, optional
	If set to true, display the result of the query.
raise_error: bool, optional
	If the view couldn't be dropped, raises the entire error instead of 
	displaying a warning.
	"""
	check_types([
		("name", name, [str], False),
		("print_info", print_info, [bool], False),
		("raise_error", raise_error, [bool], False)])
	if not(cursor):
		conn = read_auto_connect()
		cursor = conn.cursor()
	else:
		conn = False
		check_cursor(cursor)
	try:
		query="DROP VIEW {};".format(name)
		cursor.execute(query)
		if (conn):
			conn.close()
		if (print_info):
			print("The view {} was successfully dropped.".format(name))
	except:
		if (conn):
			conn.close()
		if (raise_error):
			raise
		else:
			print("\u26A0 Warning: The view {} doesn't exist or can not be dropped !\nUse parameter: raise_error = True to get more information.".format(name))
#---#
def readSQL(query: str,
			cursor = None,
			dsn: str = "",
			time_on: bool = False,
			limit: int = 10):
	"""
	---------------------------------------------------------------------------
	Returns the Result of a SQL query as a tablesample object.

	Parameters
	----------
	query: str, optional
		SQL Query. 
	cursor: DBcursor, optional
		Vertica DB cursor.
	dsn: str, optional
		Vertica DB DSN.
	time_on: bool, optional
		If set to True, displays the query elapsed time.
	limit: int, optional
		Number maximum of elements to display.

 	Returns
 	-------
 	tablesample
 		Result of the query.
	"""
	check_types([
		("query", query, [str], False),
		("dsn", dsn, [str], False),
		("time_on", time_on, [bool], False),
		("limit", limit, [int, float], False)])
	conn = False
	if not(cursor) and not(dsn):
		conn = read_auto_connect()
		cursor = conn.cursor()
	elif not(cursor):
		cursor = vertica_cursor(dsn)
	cursor.execute("SELECT COUNT(*) FROM ({}) x".format(query))
	count = cursor.fetchone()[0]
	result = to_tablesample("SELECT * FROM ({}) x LIMIT {}".format(query, limit), cursor, "readSQL", False, time_on)
	result.count = count
	return (result)
#---#
def get_data_types(table: str, 
				   cursor = None, 
				   column_name: str = "", 
				   schema_writing: str = ""):
	"""
---------------------------------------------------------------------------
Returns a customized relation columns and the respective data types. It will
create a temporary table during the process. 

Parameters
----------
table: str
	Relation. It must be pure SQL.
cursor: DBcursor, optional
	Vertica DB cursor.
column_name: str, optional
	If not empty, it will return only the data type of the input column if it
	is in the relation.
schema_writing: str, optional
	Schema used to create the temporary table. If empty, the function will 
	create a local temporary table.

Returns
-------
list of tuples
	The list of the different columns and their respective type.
	"""
	if not(cursor):
		conn = read_auto_connect()
		cursor = conn.cursor()
	else:
		conn = False
		check_cursor(cursor)
	tmp_name = "_VERTICAPY_TEMPORARY_TABLE_{}".format(random.randint(0, 10000000))
	schema = "v_temp_schema" if not(schema_writing) else schema_writing
	try:
		cursor.execute("DROP TABLE IF EXISTS {}.{}".format(schema, tmp_name))
	except:
		pass
	try:
		if (schema == "v_temp_schema"):
			cursor.execute("CREATE LOCAL TEMPORARY TABLE {} ON COMMIT PRESERVE ROWS AS {}".format(tmp_name, table))
		else:
			cursor.execute("CREATE TEMPORARY TABLE {}.{} ON COMMIT PRESERVE ROWS AS {}".format(schema, tmp_name, table))
	except:
		cursor.execute("DROP TABLE IF EXISTS {}.{}".format(schema, tmp_name))
		raise
	query = "SELECT column_name, data_type FROM columns WHERE {}table_name = '{}' AND table_schema = '{}'".format(
				"column_name = '{}' AND ".format(column_name) if (column_name) else "", tmp_name, schema)
	cursor.execute(query)
	if (column_name):
		ctype = cursor.fetchone()[1]
	else:
		ctype = cursor.fetchall()
	cursor.execute("DROP TABLE IF EXISTS {}.{}".format(schema, tmp_name))
	if (conn):
		conn.close()
	return (ctype)
#---#
def load_model(name: str, 
			   cursor = None, 
			   test_relation: str = ""):
	"""
---------------------------------------------------------------------------
Loads a Vertica model and returns the associated object.

Parameters
----------
name: str
	Model Name.
cursor: DBcursor, optional
	Vertica DB cursor.
test_relation: str, optional
	Relation used to do the testing. All the methods will use this relation 
	for the scoring. If empty, the training relation will be used as testing.

Returns
-------
model
	The model.
	"""
	check_types([
		("name", name, [str], False), 
		("test_relation", test_relation, [str], False)])
	if not(cursor):
		cursor = read_auto_connect().cursor()
	else:
		check_cursor(cursor)
	try:
		cursor.execute("SELECT GET_MODEL_ATTRIBUTE (USING PARAMETERS model_name = '" + name + "', attr_name = 'call_string')")
		info = cursor.fetchone()[0].replace('\n', ' ')
	except:
		try:
			cursor.execute("SELECT GET_MODEL_SUMMARY (USING PARAMETERS model_name = '" + name + "')")
			info = cursor.fetchone()[0].replace('\n', ' ')
			info = "kmeans(" + info.split("kmeans(")[1]
		except:
			from verticapy.learn.preprocessing import Normalizer
			model = Normalizer(name, cursor)
			model.param = to_tablesample(query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'details')".format(name.replace("'", "''")), cursor = cursor)
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
	for elem in parameters_dict:
		if type(parameters_dict[elem]) == str:
			parameters_dict[elem] = parameters_dict[elem].replace("'", "")
	if (model_type == "rf_regressor"):
		from verticapy.learn.ensemble import RandomForestRegressor
		model = RandomForestRegressor(name, cursor, int(parameters_dict['ntree']), int(parameters_dict['mtry']), int(parameters_dict['max_breadth']), float(parameters_dict['sampling_size']), int(parameters_dict['max_depth']), int(parameters_dict['min_leaf_size']), float(parameters_dict['min_info_gain']), int(parameters_dict['nbins']))
	elif (model_type == "rf_classifier"):
		from verticapy.learn.ensemble import RandomForestClassifier
		model = RandomForestClassifier(name, cursor, int(parameters_dict['ntree']), int(parameters_dict['mtry']), int(parameters_dict['max_breadth']), float(parameters_dict['sampling_size']), int(parameters_dict['max_depth']), int(parameters_dict['min_leaf_size']), float(parameters_dict['min_info_gain']), int(parameters_dict['nbins']))
	elif (model_type == "logistic_reg"):
		from verticapy.learn.linear_model import LogisticRegression
		model = LogisticRegression(name, cursor, parameters_dict['regularization'], float(parameters_dict['epsilon']), float(parameters_dict['lambda']), int(parameters_dict['max_iterations']), parameters_dict['optimizer'], float(parameters_dict['alpha']))
	elif (model_type == "linear_reg"):
		from verticapy.learn.linear_model import ElasticNet
		model = ElasticNet(name, cursor, parameters_dict['regularization'], float(parameters_dict['epsilon']), float(parameters_dict['lambda']), int(parameters_dict['max_iterations']), parameters_dict['optimizer'], float(parameters_dict['alpha']))
	elif (model_type == "naive_bayes"):
		from verticapy.learn.naive_bayes import MultinomialNB
		model = MultinomialNB(name, cursor, float(parameters_dict['alpha']))
	elif (model_type == "svm_regressor"):
		from verticapy.learn.svm import LinearSVR
		model = LinearSVR(name, cursor, float(parameters_dict['epsilon']), float(parameters_dict['C']), True, float(parameters_dict['intercept_scaling']), parameters_dict['intercept_mode'], float(parameters_dict['error_tolerance']), int(parameters_dict['max_iterations']))
	elif (model_type == "svm_classifier"):
		from verticapy.learn.svm import LinearSVC
		model = LinearSVC(name, cursor, float(parameters_dict['epsilon']), float(parameters_dict['C']), True, float(parameters_dict['intercept_scaling']), parameters_dict['intercept_mode'], [float(item) for item in parameters_dict['class_weights'].split(",")], int(parameters_dict['max_iterations']))
	elif (model_type == "kmeans"):
		from verticapy.learn.cluster import KMeans
		model = KMeans(name, cursor, -1, parameters_dict['init_method'], int(parameters_dict['max_iterations']), float(parameters_dict['epsilon']))
	elif (model_type == "pca"):
		from verticapy.learn.decomposition import PCA
		model = PCA(name, cursor, 0, bool(parameters_dict['scale']))
	elif (model_type == "svd"):
		from verticapy.learn.decomposition import SVD
		model = SVD(name, cursor)
	elif (model_type == "one_hot_encoder_fit"):
		from verticapy.learn.preprocessing import OneHotEncoder
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
		model.components = to_tablesample(query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'principal_components')".format(name.replace("'", "''")), cursor = cursor)
		model.components.table_info = False
		model.explained_variance = to_tablesample(query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'singular_values')".format(name.replace("'", "''")), cursor = cursor)
		model.explained_variance.table_info = False
		model.mean = to_tablesample(query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'columns')".format(name.replace("'", "''")), cursor = cursor)
		model.mean.table_info = False
	elif (model_type in ("svd")):
		model.X = info.split(",")[2:len(info.split(","))]
		model.X = [item.replace("'", '').replace('\\', '') for item in model.X]
		model.singular_values = to_tablesample(query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'right_singular_vectors')".format(name.replace("'", "''")), cursor = cursor)
		model.singular_values.table_info = False
		model.explained_variance = to_tablesample(query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'singular_values')".format(name.replace("'", "''")), cursor = cursor)
		model.explained_variance.table_info = False
	elif (model_type in ("one_hot_encoder_fit")):
		model.X = info.split(",")[2:len(info.split(","))]
		model.X = [item.replace("'", '').replace('\\', '') for item in model.X]
		model.param = to_tablesample(query = "SELECT category_name, category_level::varchar, category_level_index FROM (SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'integer_categories')) x UNION ALL SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'varchar_categories')".format(name.replace("'", "''"), name.replace("'", "''")), cursor = cursor)
		model.param.table_info = False
	else:
		model.X = info.split(",")[2:len(info.split(",")) - 1]
		model.X = [item.replace("'", '').replace('\\', '') for item in model.X]
		model.n_cluster = int(info.split(",")[-1])
		model.cluster_centers = to_tablesample(query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'centers')".format(name.replace("'", "''")), cursor = cursor)
		model.cluster_centers.table_info = False
		query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'metrics')".format(name.replace("'", "''"))
		cursor.execute(query)
		result = cursor.fetchone()[0]
		values = {"index": ["Between-Cluster Sum of Squares", "Total Sum of Squares", "Total Within-Cluster Sum of Squares", "Between-Cluster SS / Total SS", "converged"]}
		values["value"] = [float(result.split("Between-Cluster Sum of Squares: ")[1].split("\n")[0]), float(result.split("Total Sum of Squares: ")[1].split("\n")[0]), float(result.split("Total Within-Cluster Sum of Squares: ")[1].split("\n")[0]), float(result.split("Between-Cluster Sum of Squares: ")[1].split("\n")[0]) / float(result.split("Total Sum of Squares: ")[1].split("\n")[0]), result.split("Converged: ")[1].split("\n")[0] == "True"] 
		model.metrics = tablesample(values, table_info = False)
	if (model.type == "classifier"):
		cursor.execute("SELECT DISTINCT {} FROM {} WHERE {} IS NOT NULL ORDER BY 1".format(model.y, model.input_relation, model.y))
		classes = cursor.fetchall()
		model.classes = [item[0] for item in classes]
	return (model)
#---#
def pandas_to_vertica(df, 
					  name: str,
					  cursor = None, 
					  schema: str = "public", 
					  insert: bool = False):
	"""
---------------------------------------------------------------------------
Ingests a pandas DataFrame to Vertica DB by creating a CSV file first and 
then using flex tables.

Parameters
----------
df: pandas.DataFrame
	The pandas.DataFrame to ingest.
name: str
	Name of the new relation.
cursor: DBcursor, optional
	Vertica DB cursor. 
schema: str, optional
	Schema of the new relation. The default schema is public.
insert: bool, optional
	If set to True, the data will be ingested to the input relation. Be sure
	that your file has a header corresponding to the name of the relation
	columns otherwise the ingestion will not work.
	

Returns
-------
vDataFrame
	vDataFrame of the new relation.

See Also
--------
read_csv  : Ingests a CSV file in the Vertica DB.
read_json : Ingests a JSON file in the Vertica DB.
	"""
	check_types([
		("name", name, [str], False), 
		("schema", schema, [str], False), 
		("insert", insert, [bool], False)])
	if not(cursor):
		conn = read_auto_connect()
		cursor = conn.cursor()
	else:
		conn = False
		check_cursor(cursor)
	path = "verticapy_{}.csv".format(gen_name([name]))
	try:
		df.to_csv(path, index = False)
		read_csv(path, cursor, table_name = name, schema = schema, insert = insert)
		os.remove(path)
	except:
		os.remove(path)
		raise
	from verticapy import vDataFrame
	return vDataFrame(input_relation = name, schema = schema, cursor = cursor)
#---#
def pcsv(path: str, 
		 cursor = None, 
		 sep: str = ',', 
		 header: bool = True,
		 header_names: list = [],
		 na_rep: str = '', 
		 quotechar: str = '"',
		 escape: str = '\\'):
	"""
---------------------------------------------------------------------------
Parses a CSV file using flex tables. It will identify the columns and their
respective types.

Parameters
----------
path: str
	Absolute path where the CSV file is located.
cursor: DBcursor, optional
	Vertica DB cursor. 
sep: str, optional
	Column separator.
header: bool, optional
	If set to False, the parameter 'header_names' will be used to name the 
	different columns.
header_names: list, optional
	List of the columns names.
na_rep: str, optional
	Missing values representation.
quotechar: str, optional
	Char which is enclosing the str values.
escape: str, optional
	Separator between each record.

Returns
-------
dict
	dictionary containing for each column its type.

See Also
--------
read_csv  : Ingests a CSV file in the Vertica DB.
read_json : Ingests a JSON file in the Vertica DB.
	"""
	if not(cursor):
		conn = read_auto_connect()
		cursor = conn.cursor()
	else:
		conn = False
		check_cursor(cursor)
	flex_name = "VERTICAPY_{}_FLEX".format(random.randint(0, 10000000))
	cursor.execute("CREATE FLEX LOCAL TEMP TABLE {}(x int) ON COMMIT PRESERVE ROWS;".format(flex_name))
	header_names = '' if not(header_names) else "header_names = '{}',".format(sep.join(header_names))
	try:
		with open(path, "r") as fs:
			cursor.copy("COPY {} FROM STDIN PARSER FCSVPARSER(type = 'traditional', delimiter = '{}', header = {}, {} enclosed_by = '{}', escape = '{}') NULL '{}';".format(flex_name, sep, header, header_names, quotechar, escape, na_rep), fs)
	except:
		cursor.execute("COPY {} FROM LOCAL '{}' PARSER FCSVPARSER(type = 'traditional', delimiter = '{}', header = {}, {} enclosed_by = '{}', escape = '{}') NULL '{}';".format(flex_name, path, sep, header, header_names, quotechar, escape, na_rep))
	cursor.execute("SELECT compute_flextable_keys('{}');".format(flex_name))
	cursor.execute("SELECT key_name, data_type_guess FROM {}_keys".format(flex_name))
	result = cursor.fetchall()
	dtype = {}
	for column_dtype in result:
		try:
			query = 'SELECT (CASE WHEN "{}"=\'{}\' THEN NULL ELSE "{}" END)::{} AS "{}" FROM {} WHERE "{}" IS NOT NULL LIMIT 1000'.format(column_dtype[0], na_rep, column_dtype[0], column_dtype[1], column_dtype[0], flex_name, column_dtype[0])
			cursor.execute(query)
			dtype[column_dtype[0]] = column_dtype[1]
		except:
			dtype[column_dtype[0]] = "Varchar(100)"
	cursor.execute("DROP TABLE {}".format(flex_name))
	if (conn):
		conn.close()
	return (dtype)
#---#
def pjson(path: str, 
		  cursor = None):
	"""
---------------------------------------------------------------------------
Parses a JSON file using flex tables. It will identify the columns and their
respective types.

Parameters
----------
path: str
	Absolute path where the JSON file is located.
cursor: DBcursor, optional
	Vertica DB cursor. 

Returns
-------
dict
	dictionary containing for each column its type.

See Also
--------
read_csv  : Ingests a CSV file in the Vertica DB.
read_json : Ingests a JSON file in the Vertica DB.
	"""
	if not(cursor):
		conn = read_auto_connect()
		cursor = conn.cursor()
	else:
		conn = False
		check_cursor(cursor)
	flex_name = "VERTICAPY_{}_FLEX".format(random.randint(0, 10000000))
	cursor.execute("CREATE FLEX LOCAL TEMP TABLE {}(x int) ON COMMIT PRESERVE ROWS;".format(flex_name))
	if ("vertica_python" in str(type(cursor))):
		with open(path, "r") as fs:
			cursor.copy("COPY {} FROM STDIN PARSER FJSONPARSER();".format(flex_name), fs)
	else:
		cursor.execute("COPY {} FROM LOCAL '{}' PARSER FJSONPARSER();".format(flex_name, path.replace("'", "''")))
	cursor.execute("SELECT compute_flextable_keys('{}');".format(flex_name))
	cursor.execute("SELECT key_name, data_type_guess FROM {}_keys".format(flex_name))
	result = cursor.fetchall()
	dtype = {}
	for column_dtype in result:
		dtype[column_dtype[0]] = column_dtype[1]
	cursor.execute("DROP TABLE " + flex_name)
	if (conn):
		conn.close()
	return (dtype)
#---#
def read_csv(path: str, 
			 cursor = None, 
			 schema: str = 'public', 
			 table_name: str = '', 
			 sep: str = ',', 
			 header: bool = True,
			 header_names: list = [],
			 na_rep: str = '', 
			 quotechar: str = '"', 
			 escape: str = '\\', 
			 genSQL: bool = False,
			 parse_n_lines: int = -1,
			 insert: bool = False):
	"""
---------------------------------------------------------------------------
Ingests a CSV file using flex tables.

Parameters
----------
path: str
	Absolute path where the CSV file is located.
cursor: DBcursor, optional
	Vertica DB cursor.
schema: str, optional
	Schema where the CSV file will be ingested.
table_name: str, optional
	Final relation name.
sep: str, optional
	Column separator.
header: bool, optional
	If set to False, the parameter 'header_names' will be used to name the 
	different columns.
header_names: list, optional
	List of the columns names.
na_rep: str, optional
	Missing values representation.
quotechar: str, optional
	Char which is enclosing the str values.
escape: str, optional
	Separator between each record.
genSQL: bool, optional
	If set to True, the SQL code used to create the final table will be 
	generated but not executed. It is a good way to change the final
	relation types or to customize the data ingestion.
parse_n_lines: int, optional
	If this parameter is greater than 0. A new file of 'parse_n_lines' lines
	will be created and ingested first to identify the data types. It will be
	then dropped and the entire file will be ingested. The data types identification
	will be less precise but this parameter can make the process faster if the
	file is heavy.
insert: bool, optional
	If set to True, the data will be ingested to the input relation. Be sure
	that your file has a header corresponding to the name of the relation
	columns otherwise the ingestion will not work.

Returns
-------
vDataFrame
	The vDataFrame of the relation.

See Also
--------
read_json : Ingests a JSON file in the Vertica DB.
	"""
	check_types([
		("schema", schema, [str], False), 
		("table_name", table_name, [str], False), 
		("sep", sep, [str], False), 
		("header", header, [bool], False), 
		("header_names", header_names, [list], False), 
		("na_rep", na_rep, [str], False), 
		("quotechar", quotechar, [str], False), 
		("escape", escape, [str], False), 
		("genSQL", genSQL, [bool], False), 
		("parse_n_lines", parse_n_lines, [int, float], False), 
		("insert", insert, [bool], False)])
	if not(cursor):
		cursor = read_auto_connect().cursor()
	else:
		check_cursor(cursor)
	path, sep, header_names, na_rep, quotechar, escape = path.replace("'", "''"), sep.replace("'", "''"), [str(elem).replace("'", "''") for elem in header_names], na_rep.replace("'", "''"), quotechar.replace("'", "''"), escape.replace("'", "''")
	file = path.split("/")[-1]
	file_extension = file[-3:len(file)]
	if (file_extension != 'csv'):
		raise ValueError("The file extension is incorrect !")
	table_name = table_name if (table_name) else path.split("/")[-1].split(".csv")[0]
	query = "SELECT column_name FROM columns WHERE table_name = '{}' AND table_schema = '{}'".format(table_name.replace("'", "''"), schema.replace("'", "''"))
	cursor.execute(query)
	result = cursor.fetchall()
	if ((result != []) and not(insert)):
		raise Exception("The table \"{}\".\"{}\" already exists !".format(schema, table_name))
	elif ((result == []) and (insert)):
		raise Exception("The table \"{}\".\"{}\" doesn't exist !".format(schema, table_name))
	else:
		input_relation = '"{}"."{}"'.format(schema, table_name)
		f = open(path,'r')
		file_header = f.readline().replace('\n', '').replace('"', '').split(sep)
		f.close()
		if ((header_names == []) and (header)):
			header_names = file_header
		elif (len(file_header) > len(header_names)):
			header_names += ["ucol{}".format(i + len(header_names)) for i in range(len(file_header) - len(header_names))]
		if ((parse_n_lines > 0) and not(insert)):
			f = open(path,'r')
			f2 = open(path[0:-4] + "VERTICAPY_COPY.csv",'w')
			for i in range(parse_n_lines + int(header)):
				line = f.readline()
				f2.write(line)
			f.close()
			f2.close()
			path_test = path[0:-4] + "VERTICAPY_COPY.csv"
		else:
			path_test = path
		query1 = ""
		if not(insert):
			dtype = pcsv(path_test, cursor, sep, header, header_names, na_rep, quotechar, escape)
			if (parse_n_lines > 0):
				os.remove(path[0:-4] + "VERTICAPY_COPY.csv")
			query1  = "CREATE TABLE {}({});".format(input_relation, ", ".join(['"{}" {}'.format(column, dtype[column]) for column in header_names]))
		skip   = " SKIP 1" if (header) else ""
		query2 = "COPY {}({}) FROM {} DELIMITER '{}' NULL '{}' ENCLOSED BY '{}' ESCAPE AS '{}'{};".format(input_relation, ", ".join(['"' + column + '"' for column in header_names]), "{}", sep, na_rep, quotechar, escape, skip)
		if (genSQL):
			print(query1 + "\n" + query2)
		else:
			if (query1):
				cursor.execute(query1)
			if ("vertica_python" in str(type(cursor))):
				with open(path, "r") as fs:
					cursor.copy(query2.format('STDIN'), fs)
			else:
				cursor.execute(query2.format("LOCAL '{}'".format(path)))
			if (query1):
				print("The table {} has been successfully created.".format(input_relation))
			from verticapy import vDataFrame
			return vDataFrame(table_name, cursor, schema = schema)
#---#
def read_json(path: str, 
			  cursor = None, 
			  schema: str = 'public', 
			  table_name: str = '',
			  usecols: list = [],
			  new_name: dict = {},
			  insert: bool = False):
	"""
---------------------------------------------------------------------------
Ingests a JSON file using flex tables.

Parameters
----------
path: str
	Absolute path where the JSON file is located.
cursor: DBcursor, optional
	Vertica DB cursor.
schema: str, optional
	Schema where the JSON file will be ingested.
table_name: str, optional
	Final relation name.
usecols: list, optional
	List of the JSON parameters to ingest. The other ones will be ignored. If
	empty all the JSON parameters will be ingested.
new_name: dict, optional
	Dictionary of the new columns name. If the JSON file is nested, it is advised
	to change the final names as special characters will be included.
	For example, {"param": {"age": 3, "name": Badr}, "date": 1993-03-11} will 
	create 3 columns: "param.age", "param.name" and "date". You can rename these 
	columns using the 'new_name' parameter with the following dictionary:
	{"param.age": "age", "param.name": "name"}
insert: bool, optional
	If set to True, the data will be ingested to the input relation. The JSON
	parameters must be the same than the input relation otherwise they will
	not be ingested.

Returns
-------
vDataFrame
	The vDataFrame of the relation.

See Also
--------
read_csv : Ingests a CSV file in the Vertica DB.
	"""
	check_types([
		("schema", schema, [str], False), 
		("table_name", table_name, [str], False), 
		("usecols", usecols, [list], False), 
		("new_name", new_name, [dict], False), 
		("insert", insert, [bool], False)])
	if not(cursor):
		cursor = read_auto_connect().cursor()
	else:
		check_cursor(cursor)
	file = path.split("/")[-1]
	file_extension = file[-4:len(file)]
	if (file_extension != 'json'):
		raise ValueError("The file extension is incorrect !")
	if not(table_name): table_name = path.split("/")[-1].split(".json")[0]
	query = "SELECT column_name, data_type FROM columns WHERE table_name = '{}' AND table_schema = '{}'".format(table_name.replace("'", "''"), schema.replace("'", "''"))
	cursor.execute(query)
	column_name = cursor.fetchall()
	if ((column_name != []) and not(insert)):
		raise Exception('The table "{}"."{}" already exists !'.format(schema, table_name))
	elif ((column_name == []) and (insert)):
		raise Exception('The table "{}"."{}" doesn\'t exist !'.format(schema, table_name))
	else:
		input_relation, flex_name = '"{}"."{}"'.format(schema, table_name), "VERTICAPY_" + str(random.randint(0, 10000000)) + "_FLEX"
		cursor.execute("CREATE FLEX LOCAL TEMP TABLE {}(x int) ON COMMIT PRESERVE ROWS;".format(flex_name))
		if ("vertica_python" in str(type(cursor))):
			with open(path, "r") as fs:
				cursor.copy("COPY {} FROM STDIN PARSER FJSONPARSER();".format(flex_name), fs)
		else:
			cursor.execute("COPY {} FROM LOCAL '{}' PARSER FJSONPARSER();".format(flex_name, path.replace("'", "''")))
		cursor.execute("SELECT compute_flextable_keys('{}');".format(flex_name))
		cursor.execute("SELECT key_name, data_type_guess FROM {}_keys".format(flex_name))
		result = cursor.fetchall()
		dtype = {}
		for column_dtype in result:
			try:
				cursor.execute('SELECT "{}"::{} FROM {} LIMIT 1000'.format(column_dtype[0], column_dtype[1], flex_name))
				dtype[column_dtype[0]] = column_dtype[1]
			except:
				dtype[column_dtype[0]] = "Varchar(100)"
		if not(insert):
			cols = [column for column in dtype] if not(usecols) else [column for column in usecols]
			for i, column in enumerate(cols):
				cols[i] = '"{}"::{} AS "{}"'.format(column.replace('"', ''), dtype[column], new_name[column]) if (column in new_name) else '"{}"::{}'.format(column.replace('"', ''), dtype[column])
			cursor.execute("CREATE TABLE {} AS SELECT {} FROM {}".format(input_relation, ", ".join(cols), flex_name))
			print("The table {} has been successfully created.".format(input_relation))
		else:
			column_name_dtype = {}
			for elem in column_name:
				column_name_dtype[elem[0]] = elem[1]
			final_cols = {}
			for column in column_name_dtype:
				final_cols[column] = None
			for column in column_name_dtype:
				if column in dtype:
					final_cols[column] = column
				else:
					for col in new_name:
						if (new_name[col] == column):
							final_cols[column] = col
			final_transformation = []
			for column in final_cols:
				final_transformation += ['NULL AS "{}"'.format(column)] if (final_cols[column] == None) else ['"{}"::{} AS "{}"'.format(final_cols[column], column_name_dtype[column], column)]
			cursor.execute("INSERT INTO {} SELECT {} FROM {}".format(input_relation, ", ".join(final_transformation), flex_name))
		cursor.execute("DROP TABLE {}".format(flex_name))
		from verticapy import vDataFrame
		return vDataFrame(table_name, cursor, schema = schema)
#---#
def read_vdf(path: str, 
			 cursor = None):
	"""
---------------------------------------------------------------------------
Reads a VDF file and create the associated vDataFrame.

Parameters
----------
path: str
	Absolute path where the VDF file is located.
cursor: DBcursor, optional
	Vertica DB cursor.

Returns
-------
vDataFrame
	The vDataFrame associated to the vdf file.

See Also
--------
vDataFrame.to_vdf : Saves the vDataFrame to a .vdf text file.
vdf_from_relation : Creates a vDataFrame based on a customized relation.
	"""
	check_types([("path", path, [str], False)])
	if not(cursor):
		cursor = read_auto_connect().cursor()
	else:
		check_cursor(cursor)
	file = open(path, "r")
	save =  "from verticapy import vDataFrame\nfrom verticapy.vcolumn import vColumn\n" + "".join(file.readlines())
	file.close()
	vdf = {}
	exec(save, globals(), vdf)
	vdf = vdf["vdf_save"]
	vdf._VERTICAPY_VARIABLES_["cursor"] = cursor
	return (vdf)
#---#
class tablesample:
	"""
---------------------------------------------------------------------------
The tablesample is the transition from 'Big Data' to 'Small Data'. 
This object was created to have a nice way of displaying the results and to 
not have any dependency to any other module. It stores the aggregated result 
in memory and has some useful method to transform it to pandas.DataFrame or 
vDataFrame.

Parameters
----------
values: dict, optional
	Dictionary of columns (keys) and their values. The dictionary must be
	similar to the following one:
	{"column1": [val1, ..., valm], ... "columnk": [val1, ..., valm]}
dtype: dict, optional
	Columns data types. 
name: str, optional
	Name of the object. It is used only for rendering purposes.
count: int, optional
	Number of elements if we had to load the entire dataset. It is used 
	only for rendering purposes.
offset: int, optional
	Number of elements which had been skipped if we had to load the entire 
	dataset. It is used only for rendering purposes.
table_info: bool, optional
	If set to True, the tablesample informations will be displayed.

Attributes
----------
The tablesample attributes are the same than the parameters.
	"""
	#
	# Special Methods
	#
	#---#
	def  __init__(self, 
				  values: dict = {}, 
				  dtype: dict = {}, 
				  name: str = "Sample", 
				  count: int = 0, 
				  offset: int = 0, 
				  table_info: bool = True):
		check_types([
			("values", values, [dict], False), 
			("dtype", dtype, [dict], False), 
			("name", name, [str], False), 
			("count", count, [int], False), 
			("offset", offset, [int], False), 
			("table_info", table_info, [bool], False)])
		self.values = values
		self.dtype = dtype
		self.count = count
		self.offset = offset
		self.table_info = table_info
		self.name = name
		for column in values:
			if column not in dtype:
				self.dtype[column] = "undefined"
	#---#
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
	# Methods
	#
	#---# 
	def transpose(self):
		"""
	---------------------------------------------------------------------------
	Transposes the tablesample.

 	Returns
 	-------
 	tablesample
 		transposed tablesample
		"""
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
	#---#
	def to_pandas(self):
		"""
	---------------------------------------------------------------------------
	Converts the tablesample to a pandas DataFrame.

 	Returns
 	-------
 	pandas.DataFrame
 		pandas DataFrame of the tablesample.

	See Also
	--------
	tablesample.to_sql : Generates the SQL query associated to the tablesample.
	tablesample.to_vdf : Converts the tablesample to vDataFrame.
		"""
		import pandas as pd
		if ("index" in self.values):
			df = pd.DataFrame(data = self.values, index = self.values["index"])
			return df.drop(columns = ['index'])
		else:
			return pd.DataFrame(data = self.values)
	#---#
	def to_sql(self):
		"""
	---------------------------------------------------------------------------
	Generates the SQL query associated to the tablesample.

 	Returns
 	-------
 	str
 		SQL query associated to the tablesample.

	See Also
	--------
	tablesample.to_pandas : Converts the tablesample to a pandas DataFrame.
	tablesample.to_sql    : Generates the SQL query associated to the tablesample.
		"""
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
	#---#
	def to_vdf(self, 
			   cursor = None, 
			   dsn: str = ""):
		"""
	---------------------------------------------------------------------------
	Converts the tablesample to a vDataFrame.

	Parameters
	----------
	cursor: DBcursor, optional
		Vertica DB cursor. 
	dsn: str, optional
		Data Base DSN.

 	Returns
 	-------
 	vDataFrame
 		vDataFrame of the tablesample.

	See Also
	--------
	tablesample.to_pandas : Converts the tablesample to a pandas DataFrame.
	tablesample.to_sql    : Generates the SQL query associated to the tablesample.
		"""
		check_types([("dsn", dsn, [str], False)])
		if not(cursor) and not(dsn):
			cursor = read_auto_connect().cursor()
		elif not(cursor):
			from verticapy import vertica_cursor
			cursor = vertica_cursor(dsn)
		else:
			check_cursor(cursor)
		relation = "({}) sql_relation".format(self.to_sql())
		return (vdf_from_relation(relation, cursor = cursor, dsn = dsn)) 
#---#
def to_tablesample(query: str, 
				   cursor = None, 
				   name: str = "Sample",
				   query_on: bool = False,
				   time_on: bool = False,
				   title: str = ""):
	"""
	---------------------------------------------------------------------------
	Returns the Result of a SQL query as a tablesample object.

	Parameters
	----------
	query: str, optional
		SQL Query. 
	cursor: DBcursor, optional
		Vertica DB cursor. 
	name: str, optional
		Name of the object. It is used only for rendering purposes.
	query_on: bool, optional
		If set to True, display the query.
	time_on: bool, optional
		If set to True, display the query elapsed time.
	title: str, optional
		Query title when the query is displayed.

 	Returns
 	-------
 	tablesample
 		Result of the query.

	See Also
	--------
	tablesample : Object in memory created for rendering purposes.
	"""
	check_types([
		("query", query, [str], False), 
		("name", name, [str], False)])
	if not(cursor):
		conn = read_auto_connect()
		cursor = conn.cursor()
	else:
		conn = False
		check_cursor(cursor)
	if (query_on):
		print_query(query, title)
	start_time = time.time()
	cursor.execute(query)
	elapsed_time = time.time() - start_time
	if (time_on):
		print_time(elapsed_time)
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
	if (conn):
		conn.close()
	return tablesample(values = values, name = name)
#---#
def vdf_from_relation(relation: str, 
					  name: str = "VDF", 
					  cursor = None, 
					  dsn: str = "", 
					  schema: str = "public",
					  schema_writing: str = "",
					  history: list = [],
					  saving: list = [],
					  query_on: bool = False,
					  time_on: bool = False):
	"""
---------------------------------------------------------------------------
Creates a vDataFrame based on a customized relation.

Parameters
----------
relation: str
	Relation. It can be a customized relation but you need to englobe it using
	an alias. For example "(SELECT 1) x" is correct whereas "(SELECT 1)" or
	"SELECT 1" are incorrect.
name: str, optional
	Name of the vDataFrame. It is used only when displaying the vDataFrame.
cursor: DBcursor, optional
	Vertica DB cursor. 
	For a cursor designed by Vertica, look at vertica_python
	For ODBC, look at pyodbc.
	For JDBC, look at jaydebeapi.
	Check out utilities.vHelp, it may help you.
dsn: str, optional
	Data Base DSN. OS File including the DB credentials.
	VERTICAPY will try to create a vertica_python cursor first.
	If it didn't find the library, it will try to create a pyodbc cursor.
	Check out utilities.vHelp, it may help you.
schema: str, optional
	Relation schema. It can be used to be less ambiguous and allow to create schema 
	and relation name with dots '.' inside.
schema_writing: str, optional
	Schema used to create the temporary table. If empty, the function will create 
	a local temporary table.
history: list, optional
	vDataFrame history (user modifications). Used to keep the previous vDataFrame
	history.
saving: list, optional
	List used to reconstruct the vDataFrame from previous transformations. 
query_on: bool, optional
	If set to True, all the query will be printed.
time_on: bool, optional
	If set to True, all the query elapsed time will be printed.

Returns
-------
vDataFrame
	The vDataFrame associated to the input relation.
	"""
	check_types([
		("relation", relation, [str], False), 
		("name", name, [str], False), 
		("dsn", dsn, [str], False), 
		("schema", schema, [str], False),
		("history", history, [list], False),
		("saving", saving, [list], False),
		("query_on", query_on, [bool], False),
		("time_on", time_on, [bool], False)])
	name = gen_name([name])
	from verticapy import vDataFrame
	vdf = vDataFrame("", empty = True)
	vdf._VERTICAPY_VARIABLES_["dsn"] = dsn
	if not(cursor) and not(dsn):
		cursor = read_auto_connect().cursor()
	elif not(cursor):
		from verticapy import vertica_cursor
		cursor = vertica_cursor(dsn)
	else:
		check_cursor(cursor)
	vdf._VERTICAPY_VARIABLES_["input_relation"] = name
	vdf._VERTICAPY_VARIABLES_["main_relation"] = relation
	vdf._VERTICAPY_VARIABLES_["schema"] = schema
	vdf._VERTICAPY_VARIABLES_["schema_writing"] = schema_writing
	vdf._VERTICAPY_VARIABLES_["cursor"] = cursor
	vdf._VERTICAPY_VARIABLES_["query_on"] = query_on
	vdf._VERTICAPY_VARIABLES_["time_on"] = time_on
	vdf._VERTICAPY_VARIABLES_["where"] = []
	vdf._VERTICAPY_VARIABLES_["order_by"] = {}
	vdf._VERTICAPY_VARIABLES_["exclude_columns"] = []
	vdf._VERTICAPY_VARIABLES_["history"] = history
	vdf._VERTICAPY_VARIABLES_["saving"] = saving
	try:
		cursor.execute("DROP TABLE IF EXISTS v_temp_schema.VERTICAPY_{}_TEST;".format(name))
	except:
		pass
	cursor.execute("CREATE LOCAL TEMPORARY TABLE VERTICAPY_{}_TEST ON COMMIT PRESERVE ROWS AS SELECT * FROM {} LIMIT 10;".format(name, relation))
	cursor.execute("SELECT column_name, data_type FROM columns WHERE table_name = 'VERTICAPY_{}_TEST' AND table_schema = 'v_temp_schema'".format(name))
	result = cursor.fetchall()
	cursor.execute("DROP TABLE IF EXISTS v_temp_schema.VERTICAPY_{}_TEST;".format(name))
	vdf._VERTICAPY_VARIABLES_["columns"] = ['"' + item[0] + '"' for item in result]
	for column, ctype in result:
		if ('"' in column):
			print("\u26A0 Warning: A double quote \" was found in the column {}, its alias was changed using underscores '_' to {}".format(column, column.replace('"', '_')))
		from verticapy.vcolumn import vColumn
		new_vColumn = vColumn('"{}"'.format(column.replace('"', '_')), parent = vdf, transformations = [('"{}"'.format(column.replace('"', '""')), ctype, category_from_type(ctype))])
		setattr(vdf, '"{}"'.format(column.replace('"', '_')), new_vColumn)
		setattr(vdf, column.replace('"', '_'), new_vColumn)
	return (vdf)
#---#
def vHelp():
	"""
---------------------------------------------------------------------------
VERTICAPY Interactive Help (FAQ).
	"""
	import verticapy
	try:
		from IPython.core.display import HTML, display, Markdown
	except:
		pass
	path  = os.path.dirname(verticapy.__file__)
	img1  = "<center><img src='https://raw.githubusercontent.com/vertica/VerticaPy/master/img/logo.png' width=\"180px\"></center>"
	img2  = "              ____________       ______\n"
	img2 += "             / __          `\\     /     /\n"
	img2 += "            |  \\/         /    /     /\n"
	img2 += "            |______      /    /     /\n"
	img2 += "                   |____/    /     /\n"
	img2 += "          _____________     /     /\n"
	img2 += "          \\           /    /     /\n"
	img2 += "           \\         /    /     /\n"
	img2 += "            \\_______/    /     /\n"
	img2 += "             ______     /     /\n"
	img2 += "             \\    /    /     /\n"
	img2 += "              \\  /    /     /\n"
	img2 += "               \\/    /     /\n"
	img2 += "                    /     /\n"
	img2 += "                   /     /\n"
	img2 += "                   \\    /\n"
	img2 += "                    \\  /\n"
	img2 += "                     \\/\n"
	message  = img1 if (isnotebook()) else img2
	message += "\n\n&#128226; Welcome to the <b>VERTICAPY</b> help Module. You are about to use a new fantastic way to analyze your data !\n\nYou can learn quickly how to set up a connection, how to create a Virtual DataFrame and much more.\n\nWhat do you want to know?\n - <b>[Enter  0]</b> Do you want to know why you should use this library ?\n - <b>[Enter  1]</b> Do you want to know how to connect to your Vertica Database using Python and to Create a Virtual DataFrame ?\n - <b>[Enter  2]</b> Do you want to know if your Vertica Version is compatible with the API ?\n - <b>[Enter  3]</b> You don't have data to play with and you want to load an available dataset ?\n - <b>[Enter  4]</b> Do you want to know other modules which can make your Data Science experience more complete ?\n - <b>[Enter  5]</b> Do you want to look at a quick example ?\n - <b>[Enter  6]</b> Do you want to look at the different functions available ?\n - <b>[Enter  7]</b> Do you want to get a link to the VERTICAPY github ?\n - <b>[Enter  8]</b> Do you want to know how to display the Virtual DataFrame SQL code generation and the time elapsed to run the query ?\n - <b>[Enter  9]</b> Do you want to know how to load your own dataset inside Vertica ?\n - <b>[Enter 10]</b> Do you want to know how you writing direct SQL queries in Jupyter ?\n - <b>[Enter 11]</b> Do you want to know how you could read and write using specific schemas ?\n - <b>[Enter -1]</b> Exit"
	if not(isnotebook()):
		message = message.replace("<b>", "").replace("</b>", "").replace("&#128226;", "\u26A0")
	display(Markdown(message)) if (isnotebook()) else print(message)
	try:
		response = int(input())
	except:
		print("The choice is incorrect.\nPlease enter a number between 0 and 11.")
		try:
			response = int(input())
		except:
			print("The choice is still incorrect.\nRerun the help function when you need help.")
			return
	if (response == 0):
		message = "# VerticaPy\nNowadays, The 'Big Data' (Tb of data) is one of the main topics in the Data Science World. Data Scientists are now very important for any organisation. Becoming Data-Driven is mandatory to survive. Vertica is the first real analytic columnar Database and is still the fastest in the market. However, SQL is not enough flexible to be very popular for Data Scientists. Python flexibility is priceless and provides to any user a very nice experience. The level of abstraction is so high that it is enough to think about a function to notice that it already exists. Many Data Science APIs were created during the last 15 years and were directly adopted by the Data Science community (examples: pandas and scikit-learn). However, Python is only working in-memory for a single node process. Even if some famous highly distributed programming languages exist to face this challenge, they are still in-memory and most of the time they can not process on all the data. Besides, moving the data can become very expensive. Data Scientists must also find a way to deploy their data preparation and their models. We are far away from easiness and the entire process can become time expensive. \nThe idea behind VERTICAPY is simple: Combining the Scalability of VERTICA with the Flexibility of Python to give to the community what they need *Bringing the logic to the data and not the opposite*. This version 1.0 is the work of 3 years of new ideas and improvement.\nMain Advantages:\n - easy Data Exploration.\n - easy Data Preparation.\n - easy Data Modeling.\n - easy Model Evaluation.\n - easy Model Deployment.\n - most of what pandas.DataFrame can do, verticapy.vdataframe can do (and even much more)\n - easy ML model creation and evaluation.\n - many scikit functions and algorithms are available (and scalable!).\n\n&#9888; Please read the VERTICAPY Documentation. If you do not have time just read below.\n\n&#9888; The previous API is really nothing compare to the new version and many methods and functions totally changed. Consider this API as a totally new one.\nIf you have any feedback about the library, please contact me: <a href=\"mailto:badr.ouali@microfocus.com\">badr.ouali@microfocus.com</a>"
	elif (response == 1):
		message = "## Connection to the Database\nThis step is useless if <b>vertica-python</b> or <b>pyodbc</b> is already installed and you have a DSN in your machine. With this configuration, you do not need to manually create a cursor. It is possible to create a vDataFrame using directly the DSN (<b>dsn</b> parameter of the vDataFrame).\n### ODBC\nTo connect to the database, the user can use an ODBC connection to the Vertica database. <b>vertica-python</b> and <b>pyodbc</b> provide a cursor that will point to the database. It will be used by the <b>VerticaPy</b> to create all the different objects.\n```python\n#\n# vertica_python\n#\nimport vertica_python\n# Connection using all the DSN information\nconn_info = {'host': \"10.211.55.14\", 'port': 5433, 'user': \"dbadmin\", 'password': \"XxX\", 'database': \"testdb\"}\ncur = vertica_python.connect(** conn_info).cursor()\n# Connection using directly the DSN\nfrom verticapy.utilities import to_vertica_python_format # This function will parse the odbc.ini file\ndsn = \"VerticaDSN\"\ncur = vertica_python.connect(** to_vertica_python_format(dsn)).cursor()\n#\n# pyodbc\n#\nimport pyodbc\n# Connection using all the DSN information\ndriver = \"/Library/Vertica/ODBC/lib/libverticaodbc.dylib\"\nserver = \"10.211.55.14\"\ndatabase = \"testdb\"\nport = \"5433\"\nuid = \"dbadmin\"\npwd = \"XxX\"\ndsn = (\"DRIVER={}; SERVER={}; DATABASE={}; PORT={}; UID={}; PWD={};\").format(driver, server, database, port, uid, pwd)\ncur = pyodbc.connect(dsn).cursor()\n# Connection using directly the DSN\ndsn = (\"DSN=VerticaDSN\")\ncur = pyodbc.connect(dsn).cursor()\n```\n### JDBC\nThe user can also use a JDBC connection to the Vertica Database. \n```python\nimport jaydebeapi\n# Vertica Server Details\ndatabase = \"testdb\"\nhostname = \"10.211.55.14\"\nport = \"5433\"\nuid = \"dbadmin\"\npwd = \"XxX\"\n# Vertica JDBC class name\njdbc_driver_name = \"com.vertica.jdbc.Driver\"\n# Vertica JDBC driver path\njdbc_driver_loc = \"/Library/Vertica/JDBC/vertica-jdbc-9.3.1-0.jar\"\n# JDBC connection string\nconnection_string = 'jdbc:vertica://' + hostname + ':' + port + '/' + database\nurl = '{}:user={};password={}'.format(connection_string, uid, pwd)\nconn = jaydebeapi.connect(jdbc_driver_name, connection_string, {'user': uid, 'password': pwd}, jars = jdbc_driver_loc)\ncur = conn.cursor()\n```\nHappy Playing ! &#128540;\n"
	elif (response == 2):
		message = "## Vertica Version\n - If your Vertica version is greater or equal to 9.1, everything is well adapted.\n - If your Vertica version is greater or equal to 8.0, some algorithms may not work.\n - If your Vertica version is greater or equal to 7.0, only some algorithms will be available.\n - For other Vertica versions, the Virtual DataFrame may work but no ML algorithms will be available."
	elif (response == 3):
		message = "In VERTICAPY many datasets (titanic, iris, smart_meters, amazon, winequality) are already available to be ingested in your Vertica Database.\n\nTo ingest a dataset you can use the associated load function.\n\n<b>Example:</b>\n\n```python\nfrom vertica_python.learn.datasets import load_titanic\nvdf = load_titanic(db_cursor)\n```"
	elif (response == 4):
		message = "Some module will help VERTICAPY to get more rendering capabilities:\n - <b>matplotlib</b> will help you to get rendering capabilities\n - <b>numpy</b> to enjoy 3D plot\n - <b>anytree</b> to be able to plot trees"
	elif (response == 5):
		message = "## Quick Start\nInstall the library using the <b>pip</b> command:\n```\nroot@ubuntu:~$ pip3 install verticapy\n```\nInstall <b>vertica_python</b> or <b>pyodbc</b> to build a DB cursor:\n```shell\nroot@ubuntu:~$ pip3 install vertica_python\n```\nCreate a vertica cursor\n```python\nfrom verticapy.utilities import vertica_cursor\ncur = vertica_cursor(\"VerticaDSN\")\n```\nCreate the Virtual DataFrame of your relation:\n```python\nfrom verticapy import vDataFrame\nvdf = vDataFrame(\"my_relation\", cursor = cur)\n```\nIf you don't have data to play, you can easily load well known datasets\n```python\nfrom verticapy.learn.datasets import load_titanic\nvdf = load_titanic(cursor = cur)\n```\nYou can now play with the data...\n```python\nvdf.describe()\n# Output\n               min       25%        50%        75%   \nage           0.33      21.0       28.0       39.0   \nbody           1.0     79.25      160.5      257.5   \nfare           0.0    7.8958    14.4542    31.3875   \nparch          0.0       0.0        0.0        0.0   \npclass         1.0       1.0        3.0        3.0   \nsibsp          0.0       0.0        0.0        1.0   \nsurvived       0.0       0.0        0.0        1.0   \n                   max    unique  \nage               80.0        96  \nbody             328.0       118  \nfare          512.3292       277  \nparch              9.0         8  \npclass             3.0         3  \nsibsp              8.0         7  \nsurvived           1.0         2 \n```\nYou can also print the SQL code generation using the <b>sql_on_off</b> method.\n```python\nvdf.sql_on_off()\nvdf.describe()\n# Output\n## Compute the descriptive statistics of all the numerical columns ##\nSELECT\n\tSUMMARIZE_NUMCOL(\"age\",\"body\",\"survived\",\"pclass\",\"parch\",\"fare\",\"sibsp\") OVER ()\nFROM public.titanic\n```\nWith VERTICAPY, it is now possible to solve a ML problem with four lines of code (two if we don't consider the libraries loading).\n```python\nfrom verticapy.learn.model_selection import cross_validate\nfrom verticapy.learn.ensemble import RandomForestClassifier\n# Data Preparation\nvdf[\"sex\"].label_encode()[\"boat\"].fillna(method = \"0ifnull\")[\"name\"].str_extract(' ([A-Za-z]+)\.').eval(\"family_size\", expr = \"parch + sibsp + 1\").drop(columns = [\"cabin\", \"body\", \"ticket\", \"home.dest\"])[\"fare\"].fill_outliers().fillna().to_db(\"titanic_clean\")\n# Model Evaluation\ncross_validate(RandomForestClassifier(\"rf_titanic\", cur, max_leaf_nodes = 100, n_estimators = 30), \"titanic_clean\", [\"age\", \"family_size\", \"sex\", \"pclass\", \"fare\", \"boat\"], \"survived\", cutoff = 0.35)\n# Output\n                           auc               prc_auc   \n1-fold      0.9877114427860691    0.9530465915039339   \n2-fold      0.9965555014605642    0.7676485351425721   \n3-fold      0.9927239216549301    0.6419135521132449   \navg             0.992330288634        0.787536226253   \nstd           0.00362128464093         0.12779562393   \n                     accuracy              log_loss   \n1-fold      0.971291866028708    0.0502052541223871   \n2-fold      0.983253588516746    0.0298167751798457   \n3-fold      0.964824120603015    0.0392745694400433   \navg            0.973123191716       0.0397655329141   \nstd           0.0076344236729      0.00833079837099   \n                     precision                recall   \n1-fold                    0.96                  0.96   \n2-fold      0.9556962025316456                   1.0   \n3-fold      0.9647887323943662    0.9383561643835616   \navg             0.960161644975        0.966118721461   \nstd           0.00371376912311        0.025535200301   \n                      f1-score                   mcc   \n1-fold      0.9687259282082884    0.9376119402985075   \n2-fold      0.9867172675521821    0.9646971010878469   \n3-fold      0.9588020287309097    0.9240569687684576   \navg              0.97141507483        0.942122003385   \nstd            0.0115538960753       0.0168949813163   \n                  informedness            markedness   \n1-fold      0.9376119402985075    0.9376119402985075   \n2-fold      0.9737827715355807    0.9556962025316456   \n3-fold      0.9185148945422918    0.9296324823943662   \navg             0.943303202125        0.940980208408   \nstd            0.0229190954261       0.0109037699717   \n                           csi  \n1-fold      0.9230769230769231  \n2-fold      0.9556962025316456  \n3-fold      0.9072847682119205  \navg             0.928685964607  \nstd            0.0201579224026\n```\nHappy Playing ! &#128540;"
	elif (response == 6):
		if not(isnotebook()):
			message = "Please go to https://github.com/vertica/VerticaPy/blob/master/FEATURES.md"
		else:
			message = "Please go to <a href='https://github.com/vertica/VerticaPy/blob/master/FEATURES.md'>https://github.com/vertica/VerticaPy/blob/master/FEATURES.md</a>"
	elif (response == 7):
		if not(isnotebook()):
			message = "Please go to https://github.com/vertica/VerticaPy/"
		else:
			message = "Please go to <a href='https://github.com/vertica/VerticaPy/wiki'>https://github.com/vertica/VerticaPy/wiki</a>"
	elif (response == 8):
		message = "You can Display the SQL Code generation of the Virtual DataFrame using the <b>sql_on_off</b> method. You can also Display the query elapsed time using the <b>time_on_off</b> method.\nIt is also possible to print the current Virtual DataFrame relation using the <b>current_relation</b> method.\n"
	elif (response == 9):
		message = "VERTICAPY allows you many ways to ingest data file. It is using Vertica Flex Tables to identify the columns types and store the data inside Vertica. These functions will also return the associated Virtual DataFrame.\n\nLet's load the data from the 'data.csv' file.\n\n\n```python\nfrom verticapy import read_csv\nvdf = read_csv('data.csv', db_cursor)\n```\n\nThe same applies to json. Let's consider the file 'data.json'.\n\n\n```python\nfrom verticapy import read_json\nvdf = read_json('data.json', db_cursor)\n```\n\n"
	elif (response == 10):
		message = "SQL Alchemy and SQL Magic offer you a nice way to interact with Vertica. To install the modules, run the following commands in your Terminal: \n```\npip install pyodbc\npip install sqlalchemy-vertica[pyodbc,vertica-python]\npip install ipython-sql\n```\n\nWhen these modules are installed, you have a nice way to interact with Jupyter.\n```python\n# Creating a Connection\n%load_ext sql\n%sql vertica+pyodbc://VerticaDSN\n# You can run your sql code using the %sql or %%sql command\n%sql SELECT * FROM my_relation;\n```"
	elif (response == 11):
		message = "VERTICAPY pushes all the heavy computation to Vertica. Database users may not have rights to write or read specific tables in specific schemas. That's why all the functions and objects have the possibility to change the schema. Some of them propose a parameter 'schema' or 'schema_writing' to fulfill the task. Some have not this parameter and the schema must be written preceding the table name with a dot '.' in the middle.\n\n To handle special characters, you must use double quote '\"' in your parameter.\n\nFor example: 'my_schema.my_table' is correct but 'my-schema.my table' is incorrect, you should write '\"my-schema\".\"my table\"'\n\nModels are using the same rule for naming. Vertica has its own model management system so models belong to a specific schema. Some models are not built-in models and do not have a name but they still have the possibility to export the prediction to the DB using specific functions.\n\nIf a specific schema couldn't be retrieved, VERTICAPY will work on the 'public' schema."
	elif (response == -1):
		message = "Thank you for using the VERTICAPY help."
	elif (response == 666):
		message = "Thank you so much for using this library. My only purpose is to solve real Big Data problems in the context of Data Science. I worked years to be able to create this API and give you a real way to analyse your data.\n\nYour devoted Data Scientist: <i>Badr Ouali</i>"
	else:
		message = "The choice is incorrect.\nPlease enter a number between 0 and 11."
	if not(isnotebook()):
		message = message.replace("<b>", "").replace("</b>", "").replace("&#128226;", "\u26A0").replace("&#128540;", ";p").replace("&#9888;", "\u26A0")
		message = message.replace('<a href="mailto:badr.ouali@microfocus.com">badr.ouali@microfocus.com</a>', "badr.ouali@microfocus.com")
	display(Markdown(message)) if (isnotebook()) else print(message)