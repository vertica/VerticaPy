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
from vertica_ml_python import vDataframe
import os
import vertica_ml_python
from vertica_ml_python.utilities import str_column

#
def load_amazon(cursor, schema: str = 'public', name = 'amazon'):
	try:
		vdf = vDataframe(name, cursor, schema = schema)
	except:
		cursor.execute("CREATE TABLE {}.{}(\"number\" Integer, \"date\" Date, \"state\" Varchar(32));".format(str_column(schema), str_column(name)))
		try:
			path = os.path.dirname(vertica_ml_python.__file__) + "/learn/data/amazon.csv"
			query = "COPY {}.{}(\"number\", \"date\", \"state\") FROM {} DELIMITER ',' NULL '' ENCLOSED BY '\"' ESCAPE AS '\\' SKIP 1;".format(str_column(schema), str_column(name), "{}")
			if ("vertica_python" in str(type(cursor))):
				with open(path, "r") as fs:
	   				cursor.copy(query.format('STDIN'), fs)
			else:
				cursor.execute(query.format("LOCAL '{}'".format(path)))
			vdf = vDataframe(name, cursor, schema = schema)
		except:
			cursor.execute("DROP TABLE {}.{}".format(str_column(schema), str_column(name)))
			raise
	return (vdf)
#
def load_iris(cursor, schema: str = 'public', name = 'iris'):
	try:
		vdf = vDataframe(name, cursor, schema = schema)
	except:
		cursor.execute("CREATE TABLE {}.{}(\"SepalLengthCm\" Numeric(5,2), \"SepalWidthCm\" Numeric(5,2), \"PetalLengthCm\" Numeric(5,2), \"PetalWidthCm\" Numeric(5,2), \"Species\" Varchar(30));".format(str_column(schema), str_column(name)))
		try:
			path = os.path.dirname(vertica_ml_python.__file__) + "/learn/data/iris.csv"
			query = "COPY {}.{}(\"Id\" FILLER Integer, \"SepalLengthCm\", \"SepalWidthCm\", \"PetalLengthCm\", \"PetalWidthCm\", \"Species\") FROM {} DELIMITER ',' NULL '' ENCLOSED BY '\"' ESCAPE AS '\\' SKIP 1;".format(str_column(schema), str_column(name), "{}")
			if ("vertica_python" in str(type(cursor))):
				with open(path, "r") as fs:
	   				cursor.copy(query.format('STDIN'), fs)
			else:
				cursor.execute(query.format("LOCAL '{}'".format(path)))
			vdf = vDataframe(name, cursor, schema = schema)
		except:
			cursor.execute("DROP TABLE {}.{}".format(str_column(schema), str_column(name)))
			raise
	return (vdf)
#
def load_smart_meters(cursor, schema: str = 'public', name = 'smart_meters'):
	try:
		vdf = vDataframe(name, cursor, schema = schema)
	except:
		cursor.execute("CREATE TABLE {}.{}(\"time\" Timestamp, \"val\" Numeric(11,7), \"id\" Integer);".format(str_column(schema), str_column(name)))
		try:
			path = os.path.dirname(vertica_ml_python.__file__) + "/learn/data/smart_meters.csv"
			query = "COPY {}.{}(\"time\", \"val\", \"id\") FROM {} DELIMITER ',' NULL '' ENCLOSED BY '\"' ESCAPE AS '\\' SKIP 1;".format(str_column(schema), str_column(name), "{}")
			if ("vertica_python" in str(type(cursor))):
				with open(path, "r") as fs:
	   				cursor.copy(query.format('STDIN'), fs)
			else:
				cursor.execute(query.format("LOCAL '{}'".format(path)))
			vdf = vDataframe(name, cursor, schema = schema)
		except:
			cursor.execute("DROP TABLE {}.{}".format(str_column(schema), str_column(name)))
			raise
	return (vdf)
#
def load_titanic(cursor, schema: str = 'public', name = 'titanic'):
	try:
		vdf = vDataframe(name, cursor, schema = schema)
	except:
		cursor.execute("CREATE TABLE {}.{}(\"pclass\" Integer, \"survived\" Integer, \"name\" Varchar(164), \"sex\" Varchar(20), \"age\" Numeric(6,3), \"sibsp\" Integer, \"parch\" Integer, \"ticket\" Varchar(36), \"fare\" Numeric(10,5), \"cabin\" Varchar(30), \"embarked\" Varchar(20), \"boat\" Varchar(100), \"body\" Integer, \"home.dest\" Varchar(100));".format(str_column(schema), str_column(name)))
		try:
			path = os.path.dirname(vertica_ml_python.__file__) + "/learn/data/titanic.csv"
			query = "COPY {}.{}(\"pclass\", \"survived\", \"name\", \"sex\", \"age\", \"sibsp\", \"parch\", \"ticket\", \"fare\", \"cabin\", \"embarked\", \"boat\", \"body\", \"home.dest\") FROM {} DELIMITER ',' NULL '' ENCLOSED BY '\"' ESCAPE AS '\\' SKIP 1;".format(str_column(schema), str_column(name), "{}")
			if ("vertica_python" in str(type(cursor))):
				with open(path, "r") as fs:
	   				cursor.copy(query.format('STDIN'), fs)
			else:
				cursor.execute(query.format("LOCAL '{}'".format(path)))
			vdf = vDataframe(name, cursor, schema = schema)
		except:
			cursor.execute("DROP TABLE {}.{}".format(str_column(schema), str_column(name)))
			raise
	return (vdf)
#
def load_winequality(cursor, schema: str = 'public', name = 'winequality'):
	try:
		vdf = vDataframe(name, cursor, schema = schema)
	except:
		cursor.execute("CREATE TABLE {}.{}(\"fixed_acidity\" Numeric(6,3), \"volatile_acidity\" Numeric(7,4), \"citric_acid\" Numeric(6,3), \"residual_sugar\" Numeric(7,3), \"chlorides\" Float, \"free_sulfur_dioxide\" Numeric(7,2), \"total_sulfur_dioxide\" Numeric(7,2), \"density\" Float, \"pH\" Numeric(6,3), \"sulphates\" Numeric(6,3), \"alcohol\" Float, \"quality\" Integer, \"good\" Integer, \"color\" Varchar(20));".format(str_column(schema), str_column(name)))
		try:
			path = os.path.dirname(vertica_ml_python.__file__) + "/learn/data/winequality.csv"
			query = "COPY {}.{}(\"fixed_acidity\", \"volatile_acidity\", \"citric_acid\", \"residual_sugar\", \"chlorides\", \"free_sulfur_dioxide\", \"total_sulfur_dioxide\", \"density\", \"pH\", \"sulphates\", \"alcohol\", \"quality\", \"good\", \"color\") FROM {} DELIMITER ',' NULL '' ENCLOSED BY '\"' ESCAPE AS '\\' SKIP 1;".format(str_column(schema), str_column(name), "{}")
			if ("vertica_python" in str(type(cursor))):
				with open(path, "r") as fs:
	   				cursor.copy(query.format('STDIN'), fs)
			else:
				cursor.execute(query.format("LOCAL '{}'".format(path)))
			vdf = vDataframe(name, cursor, schema = schema)
		except:
			cursor.execute("DROP TABLE {}.{}".format(str_column(schema), str_column(name)))
			raise
	return (vdf)
