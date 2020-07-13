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
import os
# VerticaPy Modules
import verticapy
from verticapy import vDataFrame
from verticapy.utilities import *
from verticapy.toolbox import *
from verticapy.connections.connect import read_auto_connect
#---#
def load_amazon(cursor = None, 
				schema: str = 'public', 
				name: str = 'amazon'):
	"""
---------------------------------------------------------------------------
Ingests the amazon dataset in the Vertica DB (Dataset ideal for TS and
Regression). If a table with the same name and schema already exists, 
this function will create a vDataFrame from the input relation.

Parameters
----------
cursor: DBcursor, optional
	Vertica DB cursor. 
schema: str, optional
	Schema of the new relation. The default schema is public.
name: str, optional
	Name of the new relation.

Returns
-------
vDataFrame
	the amazon vDataFrame.

See Also
--------
load_iris         : Ingests the iris dataset in the Vertica DB.
	(Clustering / Classification).
load_market       : Ingests the market dataset in the Vertica DB.
	(Basic Data Exploration).
load_smart_meters : Ingests the smart meters dataset in the Vertica DB.
	(Time Series / Regression).
load_titanic      : Ingests the titanic dataset in the Vertica DB.
	(Classification).
load_winequality  : Ingests the winequality dataset in the Vertica DB.
	(Regression / Classification).
	"""
	check_types([
			("schema", schema, [str], False),
			("name", name, [str], False)])
	if not(cursor):
		cursor = read_auto_connect().cursor()
	else:
		check_cursor(cursor)
	try:
		vdf = vDataFrame(name, cursor, schema = schema)
	except:
		cursor.execute("CREATE TABLE {}.{}(\"number\" Integer, \"date\" Date, \"state\" Varchar(32));".format(str_column(schema), str_column(name)))
		try:
			path = os.path.dirname(verticapy.__file__) + "/learn/data/amazon.csv"
			query = "COPY {}.{}(\"number\", \"date\", \"state\") FROM {} DELIMITER ',' NULL '' ENCLOSED BY '\"' ESCAPE AS '\\' SKIP 1;".format(str_column(schema), str_column(name), "{}")
			if ("vertica_python" in str(type(cursor))):
				with open(path, "r") as fs:
	   				cursor.copy(query.format('STDIN'), fs)
			else:
				cursor.execute(query.format("LOCAL '{}'".format(path)))
			vdf = vDataFrame(name, cursor, schema = schema)
		except:
			cursor.execute("DROP TABLE {}.{}".format(str_column(schema), str_column(name)))
			raise
	return (vdf)
#---#
def load_iris(cursor = None, 
			  schema: str = 'public', 
			  name: str = 'iris'):
	"""
---------------------------------------------------------------------------
Ingests the iris dataset in the Vertica DB (Dataset ideal for Classification
and Clustering). If a table with the same name and schema already exists, 
this function will create a vDataFrame from the input relation.

Parameters
----------
cursor: DBcursor, optional
	Vertica DB cursor. 
schema: str, optional
	Schema of the new relation. The default schema is public.
name: str, optional
	Name of the new relation.

Returns
-------
vDataFrame
	the iris vDataFrame.

See Also
--------
load_amazon       : Ingests the amazon dataset in the Vertica DB.
	(Time Series / Regression).
load_market       : Ingests the market dataset in the Vertica DB.
	(Basic Data Exploration).
load_smart_meters : Ingests the smart meters dataset in the Vertica DB.
	(Time Series / Regression).
load_titanic      : Ingests the titanic dataset in the Vertica DB.
	(Classification).
load_winequality  : Ingests the winequality dataset in the Vertica DB.
	(Regression / Classification).
	"""
	check_types([
			("schema", schema, [str], False),
			("name", name, [str], False)])
	if not(cursor):
		cursor = read_auto_connect().cursor()
	else:
		check_cursor(cursor)
	try:
		vdf = vDataFrame(name, cursor, schema = schema)
	except:
		cursor.execute("CREATE TABLE {}.{}(\"SepalLengthCm\" Numeric(5,2), \"SepalWidthCm\" Numeric(5,2), \"PetalLengthCm\" Numeric(5,2), \"PetalWidthCm\" Numeric(5,2), \"Species\" Varchar(30));".format(str_column(schema), str_column(name)))
		try:
			path = os.path.dirname(verticapy.__file__) + "/learn/data/iris.csv"
			query = "COPY {}.{}(\"Id\" FILLER Integer, \"SepalLengthCm\", \"SepalWidthCm\", \"PetalLengthCm\", \"PetalWidthCm\", \"Species\") FROM {} DELIMITER ',' NULL '' ENCLOSED BY '\"' ESCAPE AS '\\' SKIP 1;".format(str_column(schema), str_column(name), "{}")
			if ("vertica_python" in str(type(cursor))):
				with open(path, "r") as fs:
	   				cursor.copy(query.format('STDIN'), fs)
			else:
				cursor.execute(query.format("LOCAL '{}'".format(path)))
			vdf = vDataFrame(name, cursor, schema = schema)
		except:
			cursor.execute("DROP TABLE {}.{}".format(str_column(schema), str_column(name)))
			raise
	return (vdf)
#---#
def load_market(cursor = None, 
				schema: str = 'public', 
				name: str = 'market'):
	"""
---------------------------------------------------------------------------
Ingests the market dataset in the Vertica DB (Dataset ideal for easy 
exploration). If a table with the same name and schema already exists, 
this function will create a vDataFrame from the input relation.

Parameters
----------
cursor: DBcursor, optional
	Vertica DB cursor. 
schema: str, optional
	Schema of the new relation. The default schema is public.
name: str, optional
	Name of the new relation.

Returns
-------
vDataFrame
	the market vDataFrame.

See Also
--------
load_amazon       : Ingests the amazon dataset in the Vertica DB.
	(Time Series / Regression).
load_iris         : Ingests the iris dataset in the Vertica DB.
	(Clustering / Classification).
load_smart_meters : Ingests the smart meters dataset in the Vertica DB.
	(Time Series / Regression).
load_titanic      : Ingests the titanic dataset in the Vertica DB.
	(Classification).
load_winequality  : Ingests the winequality dataset in the Vertica DB.
	(Regression / Classification).
	"""
	check_types([
			("schema", schema, [str], False),
			("name", name, [str], False)])
	if not(cursor):
		cursor = read_auto_connect().cursor()
	else:
		check_cursor(cursor)
	try:
		vdf = vDataFrame(name, cursor, schema = schema)
	except:
		cursor.execute("CREATE TABLE {}.{}(\"Name\" Varchar(32), \"Form\" Varchar(32), \"Price\" Float);".format(str_column(schema), str_column(name)))
		try:
			path = os.path.dirname(verticapy.__file__) + "/learn/data/market.csv"
			query = "COPY {}.{}(\"Form\", \"Name\", \"Price\") FROM {} DELIMITER ',' NULL '' ENCLOSED BY '\"' ESCAPE AS '\\' SKIP 1;".format(str_column(schema), str_column(name), "{}")
			if ("vertica_python" in str(type(cursor))):
				with open(path, "r") as fs:
	   				cursor.copy(query.format('STDIN'), fs)
			else:
				cursor.execute(query.format("LOCAL '{}'".format(path)))
			vdf = vDataFrame(name, cursor, schema = schema)
		except:
			cursor.execute("DROP TABLE {}.{}".format(str_column(schema), str_column(name)))
			raise
	return (vdf)
#---#
def load_smart_meters(cursor = None, 
					  schema: str = 'public', 
					  name: str = 'smart_meters'):
	"""
---------------------------------------------------------------------------
Ingests the smart meters dataset in the Vertica DB (Dataset ideal for TS
and Regression). If a table with the same name and schema already exists, 
this function will create a vDataFrame from the input relation.

Parameters
----------
cursor: DBcursor, optional
	Vertica DB cursor. 
schema: str, optional
	Schema of the new relation. The default schema is public.
name: str, optional
	Name of the new relation.

Returns
-------
vDataFrame
	the smart meters vDataFrame.

See Also
--------
load_amazon       : Ingests the amazon dataset in the Vertica DB.
	(Time Series / Regression).
load_iris         : Ingests the iris dataset in the Vertica DB.
	(Clustering / Classification).
load_market       : Ingests the market dataset in the Vertica DB.
	(Basic Data Exploration).
load_titanic      : Ingests the titanic dataset in the Vertica DB.
	(Classification).
load_winequality  : Ingests the winequality dataset in the Vertica DB.
	(Regression / Classification).
	"""
	check_types([
			("schema", schema, [str], False),
			("name", name, [str], False)])
	if not(cursor):
		cursor = read_auto_connect().cursor()
	else:
		check_cursor(cursor)
	try:
		vdf = vDataFrame(name, cursor, schema = schema)
	except:
		cursor.execute("CREATE TABLE {}.{}(\"time\" Timestamp, \"val\" Numeric(11,7), \"id\" Integer);".format(str_column(schema), str_column(name)))
		try:
			path = os.path.dirname(verticapy.__file__) + "/learn/data/smart_meters.csv"
			query = "COPY {}.{}(\"time\", \"val\", \"id\") FROM {} DELIMITER ',' NULL '' ENCLOSED BY '\"' ESCAPE AS '\\' SKIP 1;".format(str_column(schema), str_column(name), "{}")
			if ("vertica_python" in str(type(cursor))):
				with open(path, "r") as fs:
	   				cursor.copy(query.format('STDIN'), fs)
			else:
				cursor.execute(query.format("LOCAL '{}'".format(path)))
			vdf = vDataFrame(name, cursor, schema = schema)
		except:
			cursor.execute("DROP TABLE {}.{}".format(str_column(schema), str_column(name)))
			raise
	return (vdf)
#---#
def load_titanic(cursor = None, 
				 schema: str = 'public', 
				 name: str = 'titanic'):
	"""
---------------------------------------------------------------------------
Ingests the titanic dataset in the Vertica DB (Dataset ideal for 
Classification). If a table with the same name and schema already exists, 
this function will create a vDataFrame from the input relation.

Parameters
----------
cursor: DBcursor, optional
	Vertica DB cursor. 
schema: str, optional
	Schema of the new relation. The default schema is public.
name: str, optional
	Name of the new relation.

Returns
-------
vDataFrame
	the titanic vDataFrame.

See Also
--------
load_amazon       : Ingests the amazon dataset in the Vertica DB.
	(Time Series / Regression).
load_iris         : Ingests the iris dataset in the Vertica DB.
	(Clustering / Classification).
load_market       : Ingests the market dataset in the Vertica DB.
	(Basic Data Exploration).
load_smart_meters : Ingests the smart meters dataset in the Vertica DB.
	(Time Series / Regression).
load_winequality  : Ingests the winequality dataset in the Vertica DB.
	(Regression / Classification).
	"""
	check_types([
			("schema", schema, [str], False),
			("name", name, [str], False)])
	if not(cursor):
		cursor = read_auto_connect().cursor()
	else:
		check_cursor(cursor)
	try:
		vdf = vDataFrame(name, cursor, schema = schema)
	except:
		cursor.execute("CREATE TABLE {}.{}(\"pclass\" Integer, \"survived\" Integer, \"name\" Varchar(164), \"sex\" Varchar(20), \"age\" Numeric(6,3), \"sibsp\" Integer, \"parch\" Integer, \"ticket\" Varchar(36), \"fare\" Numeric(10,5), \"cabin\" Varchar(30), \"embarked\" Varchar(20), \"boat\" Varchar(100), \"body\" Integer, \"home.dest\" Varchar(100));".format(str_column(schema), str_column(name)))
		try:
			path = os.path.dirname(verticapy.__file__) + "/learn/data/titanic.csv"
			query = "COPY {}.{}(\"pclass\", \"survived\", \"name\", \"sex\", \"age\", \"sibsp\", \"parch\", \"ticket\", \"fare\", \"cabin\", \"embarked\", \"boat\", \"body\", \"home.dest\") FROM {} DELIMITER ',' NULL '' ENCLOSED BY '\"' ESCAPE AS '\\' SKIP 1;".format(str_column(schema), str_column(name), "{}")
			if ("vertica_python" in str(type(cursor))):
				with open(path, "r") as fs:
	   				cursor.copy(query.format('STDIN'), fs)
			else:
				cursor.execute(query.format("LOCAL '{}'".format(path)))
			vdf = vDataFrame(name, cursor, schema = schema)
		except:
			cursor.execute("DROP TABLE {}.{}".format(str_column(schema), str_column(name)))
			raise
	return (vdf)
#---#
def load_winequality(cursor = None, 
					 schema: str = 'public', 
					 name: str = 'winequality'):
	"""
---------------------------------------------------------------------------
Ingests the winequality dataset in the Vertica DB (Dataset ideal for Regression
and Classification). If a table with the same name and schema already exists, 
this function will create a vDataFrame from the input relation.

Parameters
----------
cursor: DBcursor, optional
	Vertica DB cursor. 
schema: str, optional
	Schema of the new relation. The default schema is public.
name: str, optional
	Name of the new relation.

Returns
-------
vDataFrame
	the winequality vDataFrame.

See Also
--------
load_amazon       : Ingests the amazon dataset in the Vertica DB.
	(Time Series / Regression).
load_iris         : Ingests the iris dataset in the Vertica DB.
	(Clustering / Classification).
load_market       : Ingests the market dataset in the Vertica DB.
	(Basic Data Exploration).
load_smart_meters : Ingests the smart meters dataset in the Vertica DB.
	(Time Series / Regression).
load_titanic      : Ingests the titanic dataset in the Vertica DB.
	(Classification).
	"""
	check_types([
			("schema", schema, [str], False),
			("name", name, [str], False)])
	if not(cursor):
		cursor = read_auto_connect().cursor()
	else:
		check_cursor(cursor)
	try:
		vdf = vDataFrame(name, cursor, schema = schema)
	except:
		cursor.execute("CREATE TABLE {}.{}(\"fixed_acidity\" Numeric(6,3), \"volatile_acidity\" Numeric(7,4), \"citric_acid\" Numeric(6,3), \"residual_sugar\" Numeric(7,3), \"chlorides\" Float, \"free_sulfur_dioxide\" Numeric(7,2), \"total_sulfur_dioxide\" Numeric(7,2), \"density\" Float, \"pH\" Numeric(6,3), \"sulphates\" Numeric(6,3), \"alcohol\" Float, \"quality\" Integer, \"good\" Integer, \"color\" Varchar(20));".format(str_column(schema), str_column(name)))
		try:
			path = os.path.dirname(verticapy.__file__) + "/learn/data/winequality.csv"
			query = "COPY {}.{}(\"fixed_acidity\", \"volatile_acidity\", \"citric_acid\", \"residual_sugar\", \"chlorides\", \"free_sulfur_dioxide\", \"total_sulfur_dioxide\", \"density\", \"pH\", \"sulphates\", \"alcohol\", \"quality\", \"good\", \"color\") FROM {} DELIMITER ',' NULL '' ENCLOSED BY '\"' ESCAPE AS '\\' SKIP 1;".format(str_column(schema), str_column(name), "{}")
			if ("vertica_python" in str(type(cursor))):
				with open(path, "r") as fs:
	   				cursor.copy(query.format('STDIN'), fs)
			else:
				cursor.execute(query.format("LOCAL '{}'".format(path)))
			vdf = vDataFrame(name, cursor, schema = schema)
		except:
			cursor.execute("DROP TABLE {}.{}".format(str_column(schema), str_column(name)))
			raise
	return (vdf)