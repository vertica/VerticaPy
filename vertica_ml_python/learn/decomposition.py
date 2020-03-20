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
from vertica_ml_python import drop_model
from vertica_ml_python import tablesample
from vertica_ml_python import to_tablesample

#
class PCA:
	#
	def  __init__(self,
				  name: str,
				  cursor,
				  n_components: int = 0,
				  scale: bool = False, 
				  method: str = "Lapack"):
		self.type = "decomposition"
		self.cursor = cursor
		self.name = name
		self.n_components = n_components
		self.scale = scale
		self.method = method
	# 
	def __repr__(self):
		try:
			self.cursor.execute("SELECT GET_MODEL_SUMMARY(USING PARAMETERS model_name = '" + self.name + "')")
			return (self.cursor.fetchone()[0])
		except:
			return "<PCA>"
	#
	#
	#
	# METHODS
	# 
	#
	def deploySQL(self, n_components: int = 0, cutoff: float = 1, key_columns: list = []):
		sql = "APPLY_PCA({} USING PARAMETERS model_name = '{}', match_by_pos = 'true'"
		if (key_columns):
			sql += ", key_columns = '{}'".format(", ".join(['"' + item.replace('"', '') + '"' for item in key_columns]))
		if (n_components):
			sql += ", num_components = {}".format(n_components)
		else:
			sql += ", cutoff = {}".format(cutoff)
		sql += ")"
		return (sql.format(", ".join(self.X), self.name))
	#
	def deployInverseSQL(self, key_columns: list = []):
		sql = "APPLY_INVERSE_PCA({} USING PARAMETERS model_name = '{}', match_by_pos = 'true'"
		if (key_columns):
			sql += ", key_columns = '{}'".format(", ".join(['"' + item.replace('"', '') + '"' for item in key_columns]))
		sql += ")"
		return (sql.format(", ".join(self.X), self.name))
	#
	def drop(self):
		drop_model(self.name, self.cursor, print_info = False)
	#
	def fit(self,
			input_relation: str, 
			X: list):
		self.input_relation = input_relation
		self.X = ['"' + column.replace('"', '') + '"' for column in X]
		query = "SELECT PCA('{}', '{}', '{}' USING PARAMETERS scale = {}, method = '{}'"
		query = query.format(self.name, input_relation, ", ".join(self.X), self.scale, self.method)
		if (self.n_components):
			query += ", num_components = {}".format(self.n_components)
		query += ")"
		self.cursor.execute(query)
		self.components = to_tablesample(query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'principal_components')".format(self.name), cursor = self.cursor)
		self.components.table_info = False
		self.explained_variance = to_tablesample(query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'singular_values')".format(self.name), cursor = self.cursor)
		self.explained_variance.table_info = False
		self.mean = to_tablesample(query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'columns')".format(self.name), cursor = self.cursor)
		self.mean.table_info = False
		return (self)
	#
	def to_vdf(self, n_components: int = 0,  cutoff: float = 1, key_columns: list = [], func: str = 'pca', inverse: bool = False):
		from vertica_ml_python.utilities import vdf_from_relation
		input_relation = "pca_table_" + self.input_relation 
		if (inverse):
			main_relation = "(SELECT {} FROM {}) inverse_pca_table_{}".format(self.deployInverseSQL(key_columns), self.input_relation, self.input_relation)
		else:
			main_relation = "(SELECT {} FROM {}) pca_table_{}".format(self.deploySQL(n_components, cutoff, key_columns), self.input_relation, self.input_relation)
		return (vdf_from_relation(main_relation, input_relation, self.cursor))
#
class SVD:
	#
	def  __init__(self,
				  name: str,
				  cursor,
				  n_components: int = 0, 
				  method: str = "Lapack"):
		self.type = "decomposition"
		self.cursor = cursor
		self.name = name
		self.n_components = n_components
		self.method = method
	# 
	def __repr__(self):
		try:
			self.cursor.execute("SELECT GET_MODEL_SUMMARY(USING PARAMETERS model_name = '" + self.name + "')")
			return (self.cursor.fetchone()[0])
		except:
			return "<SVD>"
	#
	#
	#
	# METHODS
	# 
	#
	def deploySQL(self, n_components: int = 0, cutoff: float = 1, key_columns: list = []):
		sql = "APPLY_SVD({} USING PARAMETERS model_name = '{}', match_by_pos = 'true'"
		if (key_columns):
			sql += ", key_columns = '{}'".format(", ".join(['"' + item.replace('"', '') + '"' for item in key_columns]))
		if (n_components):
			sql += ", num_components = {}".format(n_components)
		else:
			sql += ", cutoff = {}".format(cutoff)
		sql += ")"
		return (sql.format(", ".join(self.X), self.name))
	#
	def deployInverseSQL(self, key_columns: list = []):
		sql = "APPLY_INVERSE_SVD({} USING PARAMETERS model_name = '{}', match_by_pos = 'true'"
		if (key_columns):
			sql += ", key_columns = '{}'".format(", ".join(['"' + item.replace('"', '') + '"' for item in key_columns]))
		sql += ")"
		return (sql.format(", ".join(self.X), self.name))
	#
	def drop(self):
		drop_model(self.name, self.cursor, print_info = False)
	#
	def fit(self,
			input_relation: str, 
			X: list):
		self.input_relation = input_relation
		self.X = ['"' + column.replace('"', '') + '"' for column in X]
		query = "SELECT SVD('{}', '{}', '{}' USING PARAMETERS method = '{}'"
		query = query.format(self.name, input_relation, ", ".join(self.X), self.method)
		if (self.n_components):
			query += ", num_components = {}".format(self.n_components)
		query += ")"
		self.cursor.execute(query)
		self.singular_values = to_tablesample(query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'right_singular_vectors')".format(self.name), cursor = self.cursor)
		self.singular_values.table_info = False
		self.explained_variance = to_tablesample(query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'singular_values')".format(self.name), cursor = self.cursor)
		self.explained_variance.table_info = False
		return (self)
	#
	def to_vdf(self, n_components: int = 0,  cutoff: float = 1, key_columns: list = [], func: str = 'svd', inverse: bool = False):
		from vertica_ml_python.utilities import vdf_from_relation
		input_relation = "svd_table_" + self.input_relation
		if (inverse):
			main_relation = "(SELECT {} FROM {}) inverse_svd_table_{}".format(self.deployInverseSQL(key_columns), self.input_relation, self.input_relation)
		else:
			main_relation = "(SELECT {} FROM {}) svd_table_{}".format(self.deploySQL(n_components, cutoff, key_columns), self.input_relation, self.input_relation)
		return (vdf_from_relation(main_relation, input_relation, self.cursor))