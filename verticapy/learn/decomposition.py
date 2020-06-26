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
from verticapy.utilities import *
from verticapy.toolbox import *
from verticapy.connections.connect import read_auto_connect
#---#
class PCA:
	"""
---------------------------------------------------------------------------
Creates a PCA (Principal Component Analysis) object by using the Vertica 
Highly Distributed and Scalable PCA on the data.
 
Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
cursor: DBcursor, optional
	Vertica DB cursor.
n_components: int, optional
	The number of components to keep in the model. If this value is not provided, 
	all components are kept. The maximum number of components is the number of 
	non-zero singular values returned by the internal call to SVD. This number is 
	less than or equal to SVD (number of columns, number of rows). 
scale: bool, optional
	A Boolean value that specifies whether to standardize the columns during the 
	preparation step.
method: str, optional
	The method used to calculate PCA.
		lapack: Lapack definition.

Attributes
----------
After the object creation, all the parameters become attributes. 
The model will also create extra attributes when fitting the model:

components: tablesample
	The principal components.
explained_variance: tablesample
	The singular values explained variance.
mean: tablesample
	The information about columns from the input relation used for creating 
	the PCA model.
input_relation: str
	Train relation.
X: list
	List of the predictors.
	"""
	#
	# Special Methods
	#
	#---#
	def  __init__(self,
				  name: str,
				  cursor = None,
				  n_components: int = 0,
				  scale: bool = False, 
				  method: str = "lapack"):
		check_types([
			("name", name, [str], False),
			("n_components", n_components, [int, float], False),
			("scale", scale, [bool], False),
			("method", method, ["lapack"], True)])
		if not(cursor):
			cursor = read_auto_connect().cursor()
		else:
			check_cursor(cursor)
		self.type = "decomposition"
		self.cursor = cursor
		self.name = name
		self.n_components = n_components
		self.scale = scale
		self.method = method.lower()
	#---#
	def __repr__(self):
		try:
			self.cursor.execute("SELECT GET_MODEL_SUMMARY(USING PARAMETERS model_name = '{}')".format(self.name))
			return (self.cursor.fetchone()[0])
		except:
			return "<PCA>"
	#
	# Methods
	#
	#---# 
	def deploySQL(self, 
				  n_components: int = 0, 
				  cutoff: float = 1, 
				  key_columns: list = []):
		"""
	---------------------------------------------------------------------------
	Returns the SQL code needed to deploy the model. 

	Parameters
	----------
	n_components: int, optional
		Number of components to return. If set to 0, all the components will be
		deployed.
	cutoff: float, optional
		Specifies the minimum accumulated explained variance. Components are taken 
		until the accumulated explained variance reaches this value.
	key_columns: list, optional
		Predictors used during the algorithm computation which will be deployed
		with the principal components.

	Returns
	-------
	str/list
 		the SQL code needed to deploy the model.
		"""
		check_types([
			("n_components", n_components, [int, float], False),
			("cutoff", cutoff, [int, float], False),
			("key_columns", key_columns, [list], False)])
		sql = "APPLY_PCA({} USING PARAMETERS model_name = '{}', match_by_pos = 'true'"
		if (key_columns):
			sql += ", key_columns = '{}'".format(", ".join([str_column(item) for item in key_columns]))
		if (n_components):
			sql += ", num_components = {}".format(n_components)
		else:
			sql += ", cutoff = {}".format(cutoff)
		sql += ")"
		return (sql.format(", ".join(self.X), self.name))
	#---#
	def deployInverseSQL(self, 
						 key_columns: list = []):
		"""
	---------------------------------------------------------------------------
	Returns the SQL code needed to deploy the inverse model (PCA ** -1). 

	Parameters
	----------
	key_columns: list, optional
		Predictors used during the algorithm computation which will be deployed
		with the principal components.

	Returns
	-------
	str/list
 		the SQL code needed to deploy the inverse model (PCA ** -1).
		"""
		check_types([("key_columns", key_columns, [list], False)])
		sql = "APPLY_INVERSE_PCA({} USING PARAMETERS model_name = '{}', match_by_pos = 'true'"
		if (key_columns):
			sql += ", key_columns = '{}'".format(", ".join([str_column(item) for item in key_columns]))
		sql += ")"
		return (sql.format(", ".join(self.X), self.name))
	#---#
	def drop(self):
		"""
	---------------------------------------------------------------------------
	Drops the model from the Vertica DB.
		"""
		drop_model(self.name, self.cursor, print_info = False)
	#---#
	def fit(self,
			input_relation: str, 
			X: list):
		"""
	---------------------------------------------------------------------------
	Trains the model.

	Parameters
	----------
	input_relation: str
		Train relation.
	X: list
		List of the predictors.

	Returns
	-------
	object
 		self
		"""
		check_types([
			("input_relation", input_relation, [str], False),
			("X", X, [list], False)])
		self.input_relation = input_relation
		self.X = [str_column(column) for column in X]
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
	#---#
	def to_vdf(self, 
			   n_components: int = 0,  
			   cutoff: float = 1, 
			   key_columns: list = [], 
			   inverse: bool = False):
		"""
	---------------------------------------------------------------------------
	Creates a vDataFrame of the model.

	Parameters
	----------
	n_components: int, optional
		Number of components to return. If set to 0, all the components will be
		deployed.
	cutoff: float, optional
		Specifies the minimum accumulated explained variance. Components are 
		taken until the accumulated explained variance reaches this value.
	key_columns: list, optional
		Predictors used during the algorithm computation which will be deployed
		with the principal components.
	inverse: bool, optional
		If set to true, the inverse model will be deployed.

	Returns
	-------
	vDataFrame
 		model vDataFrame
		"""
		check_types([
			("n_components", n_components, [int, float], False),
			("cutoff", cutoff, [int, float], False),
			("key_columns", key_columns, [list], False),
			("inverse", inverse, [bool], False)])
		if (inverse):
			main_relation = "(SELECT {} FROM {}) x".format(self.deployInverseSQL(key_columns), self.input_relation)
		else:
			main_relation = "(SELECT {} FROM {}) x".format(self.deploySQL(n_components, cutoff, key_columns), self.input_relation)
		return (vdf_from_relation(main_relation, "pca_" + ''.join(ch for ch in self.input_relation if ch.isalnum()), self.cursor))
#---#
class SVD:
	"""
---------------------------------------------------------------------------
Creates a SVD (Singular Value Decomposition) object by using the Vertica 
Highly Distributed and Scalable SVD on the data.
 
Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
cursor: DBcursor, optional
	Vertica DB cursor.
n_components: int, optional
	The number of components to keep in the model. If this value is not provided, 
	all components are kept. The maximum number of components is the number of 
	non-zero singular values returned by the internal call to SVD. This number is 
	less than or equal to SVD (number of columns, number of rows).
method: str, optional
	The method used to calculate SVD.
		lapack: Lapack definition.

Attributes
----------
After the object creation, all the parameters become attributes. 
The model will also create extra attributes when fitting the model:

singular_values: tablesample
	The singular values.
explained_variance: tablesample
	The singular values explained variance.
input_relation: str
	Train relation.
X: list
	List of the predictors.
	"""
	#
	# Special Methods
	#
	#---#
	def  __init__(self,
				  name: str,
				  cursor = None,
				  n_components: int = 0, 
				  method: str = "lapack"):
		check_types([
			("name", name, [str], False),
			("n_components", n_components, [int, float], False),
			("method", method, ["lapack"], True)])
		if not(cursor):
			cursor = read_auto_connect().cursor()
		else:
			check_cursor(cursor)
		self.type = "decomposition"
		self.cursor = cursor
		self.name = name
		self.n_components = n_components
		self.method = method.lower()
	#---#
	def __repr__(self):
		try:
			self.cursor.execute("SELECT GET_MODEL_SUMMARY(USING PARAMETERS model_name = '" + self.name + "')")
			return (self.cursor.fetchone()[0])
		except:
			return "<SVD>"
	#
	# Methods
	#
	#---# 
	def deploySQL(self, 
				  n_components: int = 0, 
				  cutoff: float = 1, 
				  key_columns: list = []):
		"""
	---------------------------------------------------------------------------
	Returns the SQL code needed to deploy the model. 

	Parameters
	----------
	n_components: int, optional
		Number of components to return. If set to 0, all the singular values will 
		be deployed.
	cutoff: float, optional
		Specifies the minimum accumulated explained variance. Singular Value are 
		taken until the accumulated explained variance reaches this value.
	key_columns: list, optional
		Predictors used during the algorithm computation which will be deployed with 
		the singular values.

	Returns
	-------
	str/list
 		the SQL code needed to deploy the model.
		"""
		check_types([
			("n_components", n_components, [int, float], False),
			("cutoff", cutoff, [int, float], False),
			("key_columns", key_columns, [list], False)])
		sql = "APPLY_SVD({} USING PARAMETERS model_name = '{}', match_by_pos = 'true'"
		if (key_columns):
			sql += ", key_columns = '{}'".format(", ".join([str_column(item) for item in key_columns]))
		if (n_components):
			sql += ", num_components = {}".format(n_components)
		else:
			sql += ", cutoff = {}".format(cutoff)
		sql += ")"
		return (sql.format(", ".join(self.X), self.name))
	#---#
	def deployInverseSQL(self, 
						 key_columns: list = []):
		"""
	---------------------------------------------------------------------------
	Returns the SQL code needed to deploy the inverse model (SVD ** -1). 

	Parameters
	----------
	key_columns: list, optional
		Predictors used during the algorithm computation which will be deployed
		with the principal components.

	Returns
	-------
	str/list
 		the SQL code needed to deploy the inverse model (SVD ** -1).
		"""
		check_types([("key_columns", key_columns, [list], False)])
		sql = "APPLY_INVERSE_SVD({} USING PARAMETERS model_name = '{}', match_by_pos = 'true'"
		if (key_columns):
			sql += ", key_columns = '{}'".format(", ".join([str_column(item) for item in key_columns]))
		sql += ")"
		return (sql.format(", ".join(self.X), self.name))
	#---#
	def drop(self):
		"""
	---------------------------------------------------------------------------
	Drops the model from the Vertica DB.
		"""
		drop_model(self.name, self.cursor, print_info = False)
	#---#
	def fit(self,
			input_relation: str, 
			X: list):
		"""
	---------------------------------------------------------------------------
	Trains the model.

	Parameters
	----------
	input_relation: str
		Train relation.
	X: list
		List of the predictors.

	Returns
	-------
	object
 		self
		"""
		check_types([
			("input_relation", input_relation, [str], False),
			("X", X, [list], False)])
		self.input_relation = input_relation
		self.X = [str_column(column) for column in X]
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
	#---#
	def to_vdf(self, 
			   n_components: int = 0,  
			   cutoff: float = 1, 
			   key_columns: list = [], 
			   inverse: bool = False):
		"""
	---------------------------------------------------------------------------
	Creates a vDataFrame of the model.

	Parameters
	----------
	n_components: int, optional
		Number of singular value to return. If set to 0, all the components will 
		be deployed.
	cutoff: float, optional
		Specifies the minimum accumulated explained variance. Components are 
		taken until the accumulated explained variance reaches this value.
	key_columns: list, optional
		Predictors used during the algorithm computation which will be deployed
		with the singular values.
	inverse: bool, optional
		If set to True, the inverse model will be deployed.

	Returns
	-------
	vDataFrame
 		model vDataFrame
		"""
		check_types([
			("n_components", n_components, [int, float], False),
			("cutoff", cutoff, [int, float], False),
			("key_columns", key_columns, [list], False),
			("inverse", inverse, [bool], False)])
		input_relation = "svd_table_" + self.input_relation
		if (inverse):
			main_relation = "(SELECT {} FROM {}) x".format(self.deployInverseSQL(key_columns), self.input_relation)
		else:
			main_relation = "(SELECT {} FROM {}) x".format(self.deploySQL(n_components, cutoff, key_columns), self.input_relation)
		return (vdf_from_relation(main_relation, "svd_" + ''.join(ch for ch in self.input_relation if ch.isalnum()), self.cursor))