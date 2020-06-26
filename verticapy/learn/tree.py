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
# VerticaPy Modules
from verticapy.learn.ensemble import RandomForestClassifier, RandomForestRegressor
from verticapy.connections.connect import read_auto_connect
#---#
def DecisionTreeClassifier(name: str,
						   cursor = None,
						   max_features = "auto",
						   max_leaf_nodes: int = 1e9, 
						   max_depth: int = 100,
						   min_samples_leaf: int = 1,
						   min_info_gain: float = 0.0,
						   nbins: int = 32):
	"""
---------------------------------------------------------------------------
Single Decision Tree Classifier.
 => RandomForestClassifier of one tree using all the data.

Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
cursor: DBcursor, optional
	Vertica DB cursor.
max_features: str, optional
	The number of randomly chosen features from which to pick the best feature 
	to split on a given tree node. It can be an integer or one of the two following
	methods.
		auto : square root of the total number of predictors.
		max  : number of predictors.
max_leaf_nodes: int, optional
	The maximum number of leaf nodes a tree in the forest can have, an integer 
	between 1 and 1e9, inclusive. 
max_depth: int, optional
	The maximum depth for growing each tree, an integer between 1 and 100, inclusive.
min_samples_leaf: int, optional
	The minimum number of samples each branch must have after splitting a node, an 
	integer between 1 and 1e6, inclusive. A split that causes fewer remaining samples 
	is discarded. 
min_info_gain: float, optional
	The minimum threshold for including a split, a float between 0.0 and 1.0, inclusive. 
	A split with information gain less than this threshold is discarded.
nbins: int, optional 
	The number of bins to use for continuous features, an integer between 2 and 1000, 
	inclusive.
	"""
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
#---#
def DecisionTreeRegressor(name: str,
						  cursor = None,
						  max_features = "auto",
						  max_leaf_nodes: int = 1e9, 
						  max_depth: int = 100,
						  min_samples_leaf: int = 1,
						  min_info_gain: float = 0.0,
						  nbins: int = 32):
	"""
---------------------------------------------------------------------------
Single Decision Tree Regressor.
 => RandomForestRegressor of one tree using all the data.

Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
cursor: DBcursor, optional
	Vertica DB cursor.
max_features: str, optional
	The number of randomly chosen features from which to pick the best feature 
	to split on a given tree node. It can be an integer or one of the two following
	methods.
		auto : square root of the total number of predictors.
		max  : number of predictors.
max_leaf_nodes: int, optional
	The maximum number of leaf nodes a tree in the forest can have, an integer 
	between 1 and 1e9, inclusive. 
max_depth: int, optional
	The maximum depth for growing each tree, an integer between 1 and 100, inclusive.
min_samples_leaf: int, optional
	The minimum number of samples each branch must have after splitting a node, an 
	integer between 1 and 1e6, inclusive. A split that causes fewer remaining samples 
	is discarded. 
min_info_gain: float, optional
	The minimum threshold for including a split, a float between 0.0 and 1.0, inclusive. 
	A split with information gain less than this threshold is discarded.
nbins: int, optional 
	The number of bins to use for continuous features, an integer between 2 and 1000, 
	inclusive.
	"""
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
#---#
def DummyTreeClassifier(name: str, 
						cursor = None):
	"""
---------------------------------------------------------------------------
Dummy Tree Classifier. This classifier learns by heart the training data. 
 => very depth RandomForestClassifier of one tree using all the data.

Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
cursor: DBcursor, optional
	Vertica DB cursor. 
	"""
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
#---#
def DummyTreeRegressor(name: str, 
					   cursor = None):
	"""
---------------------------------------------------------------------------
Dummy Tree Regressor. This regressor learns by heart the training data. 
 => very depth RandomForestRegressor of one tree using all the data.

Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
cursor: DBcursor, optional
	Vertica DB cursor. 
	"""
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