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
from vertica_ml_python import read_csv
from vertica_ml_python import vDataframe
import os
import vertica_ml_python

#
def load_iris(cursor, schema: str = 'public', name = 'iris'):
	try:
		vdf = read_csv(os.path.dirname(vertica_ml_python.__file__) + "/learn/data/iris.csv", cursor, schema, name)
	except:
		vdf = vDataframe(schema + "." + name, cursor)
	return (vdf)
#
def load_smart_meters(cursor, schema: str = 'public', name = 'smart_meters'):
	try:
		vdf = read_csv(os.path.dirname(vertica_ml_python.__file__) + "/learn/data/smart_meters.csv", cursor, schema, name)
	except:
		vdf = vDataframe(schema + "." + name, cursor)
	return (vdf)
#
def load_titanic(cursor, schema: str = 'public', name = 'titanic'):
	try:
		vdf = read_csv(os.path.dirname(vertica_ml_python.__file__) + "/learn/data/titanic.csv", cursor, schema, name)
	except:
		vdf = vDataframe(schema + "." + name, cursor)
	return (vdf)
#
def load_winequality(cursor, schema: str = 'public', name = 'winequality'):
	try:
		vdf = read_csv(os.path.dirname(vertica_ml_python.__file__) + "/learn/data/winequality.csv", cursor, schema, name)
	except:
		vdf = vDataframe(schema + "." + name, cursor)
	return (vdf)
