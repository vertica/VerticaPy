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
from vertica_ml_python.learn.ensemble import RandomForestClassifier, RandomForestRegressor
#
def DecisionTreeClassifier(name: str,
						   cursor,
						   max_features = "auto",
						   max_leaf_nodes: int = 1e9, 
						   max_depth: int = 100,
						   min_samples_leaf: int = 1,
						   min_info_gain: float = 0.0,
						   nbins: int = 32):
	return RandomForestClassifier(name = name, 
								  cursor = cursor, 
								  n_estimators = 1, 
								  max_features = max_features, 
								  max_leaf_nodes = max_leaf_nodes,
								  sample = 1.0,
								  max_depth = max_depth,
								  min_samples_leaf = min_samples_leaf,
								  min_info_gain = min_info_gain,
								  nbins = nbins)
#
def DecisionTreeRegressor(name: str,
						  cursor,
						  max_features = "auto",
						  max_leaf_nodes: int = 1e9, 
						  max_depth: int = 100,
						  min_samples_leaf: int = 1,
						  min_info_gain: float = 0.0,
						  nbins: int = 32):
	return RandomForestRegressor(name = name, 
								 cursor = cursor, 
								 n_estimators = 1, 
								 max_features = max_features, 
								 max_leaf_nodes = max_leaf_nodes,
								 sample = 1.0,
								 max_depth = max_depth,
								 min_samples_leaf = min_samples_leaf,
								 min_info_gain = min_info_gain,
								 nbins = nbins)
#
def DummyTreeClassifier(name: str, cursor):
	return RandomForestClassifier(name = name, 
								  cursor = cursor, 
								  n_estimators = 1, 
								  max_features = "max", 
								  max_leaf_nodes = 1e9,
								  sample = 1.0,
								  max_depth = 100,
								  min_samples_leaf = 1,
								  min_info_gain = 0.0,
								  nbins = 1000)
#
def DummyTreeRegressor(name: str, cursor):
	return RandomForestRegressor(name = name, 
								 cursor = cursor, 
								 n_estimators = 1, 
								 max_features = "max", 
								 max_leaf_nodes = 1e9,
								 sample = 1.0,
								 max_depth = 100,
								 min_samples_leaf = 1,
								 min_info_gain = 0.0,
								 nbins = 1000)