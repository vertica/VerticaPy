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
from vertica_ml_python.utilities import schema_relation, str_column, drop_text_index, to_tablesample
#
def Balance(name: str, input_relation: str, cursor, y: str, method: str, ratio = 0.5):
	if (method not in ("hybrid", "over", "under")):
		raise ValueError("'method' must be in hybrid|over|under")
	elif (ratio >= 1 or ratio <= 0):
		raise ValueError("'ratio' must live in ]0,1[")
	sql = "SELECT BALANCE('{}', '{}', '{}', '{}_sampling' USING PARAMETERS sampling_ratio = {})".format(name, input_relation, y, method, ratio)
	cursor.execute(sql)
#
class CountVectorizer:
	#
	def  __init__(self,
				  name: str,
				  cursor,
				  lowercase: bool = True,
				  max_df: float = 1.0,
				  min_df: float = 0.0,
				  max_features: int = -1,
				  ignore_special: bool = True,
				  max_text_size: int = 2000):
		self.type = "preprocessing"
		self.name = name
		self.cursor = cursor
		self.lowercase = lowercase
		self.max_df = max_df
		self.min_df = min_df
		self.max_features = max_features
		self.ignore_special = ignore_special
		self.max_text_size = max_text_size
	# 
	def __repr__(self):
		return "<CountVectorizer>"
	#
	#
	#
	# METHODS
	# 
	#
	def deploySQL(self):
		sql = "SELECT * FROM (SELECT token, cnt / SUM(cnt) OVER () AS df, cnt, rnk FROM (SELECT token, COUNT(*) AS cnt, RANK() OVER (ORDER BY COUNT(*) DESC) AS rnk FROM {} GROUP BY 1) x) y WHERE (df BETWEEN {} AND {})".format(self.name, self.min_df, self.max_df)
		if (self.max_features > 0):
			sql += " AND (rnk <= {})".format(self.max_features)
		return (sql.format(", ".join(self.X), self.name))
	#
	def drop(self):
		drop_text_index(self.name, self.cursor, print_info = False)
	#
	def fit(self, input_relation: str, X: list):
		self.input_relation = input_relation
		self.X = [str_column(elem) for elem in X]
		schema, relation = schema_relation(input_relation)
		schema = str_column(schema)
		relation_alpha = ''.join(ch for ch in relation if ch.isalnum())
		self.cursor.execute("DROP TABLE IF EXISTS {}.VERTICA_ML_PYTHON_COUNT_VECTORIZER_{} CASCADE".format(schema, relation_alpha))
		sql = "CREATE TABLE {}.VERTICA_ML_PYTHON_COUNT_VECTORIZER_{}(id identity(2000) primary key, text varchar({})) ORDER BY id SEGMENTED BY HASH(id) ALL NODES KSAFE;"
		self.cursor.execute(sql.format(schema, relation_alpha, self.max_text_size))
		text = " || ".join(self.X) if not (self.lowercase) else "LOWER({})".format(" || ".join(self.X))
		if (self.ignore_special):
			text = "REGEXP_REPLACE({}, '[^a-zA-Z0-9\\s]+', '')".format(text)
		sql = "INSERT INTO {}.VERTICA_ML_PYTHON_COUNT_VECTORIZER_{}(text) SELECT {} FROM {}".format(schema, relation_alpha, text, input_relation)
		self.cursor.execute(sql)
		sql = "CREATE TEXT INDEX {} ON {}.VERTICA_ML_PYTHON_COUNT_VECTORIZER_{}(id, text) stemmer NONE;".format(self.name, schema, relation_alpha)
		self.cursor.execute(sql)
		stop_words = "SELECT token FROM (SELECT token, cnt / SUM(cnt) OVER () AS df, rnk FROM (SELECT token, COUNT(*) AS cnt, RANK() OVER (ORDER BY COUNT(*) DESC) AS rnk FROM {} GROUP BY 1) x) y WHERE not(df BETWEEN {} AND {})".format(self.name, self.min_df, self.max_df)
		if (self.max_features > 0):
			stop_words += " OR (rnk > {})".format(self.max_features)
		self.cursor.execute(stop_words)
		self.stop_words = [item[0] for item in self.cursor.fetchall()]
		self.cursor.execute(self.deploySQL())
		self.vocabulary = [item[0] for item in self.cursor.fetchall()]
		return (self)
	#
	def to_vdf(self):
		from vertica_ml_python.utilities import vdf_from_relation
		return (vdf_from_relation("({}) x".format(self.deploySQL()), self.name, self.cursor))
#
class Normalizer:
	#
	def  __init__(self,
				  name: str,
				  cursor,
				  method: str = "zscore"):
		self.type = "preprocessing"
		self.name = name
		self.cursor = cursor
		self.method = method
	# 
	def __repr__(self):
		try:
			self.cursor.execute("SELECT GET_MODEL_SUMMARY(USING PARAMETERS model_name = '" + self.name + "')")
			return (self.cursor.fetchone()[0])
		except:
			return "<Normalizer>"
	#
	#
	#
	# METHODS
	# 
	#
	def deploySQL(self):
		sql = "APPLY_NORMALIZE({} USING PARAMETERS model_name = '{}')"
		return (sql.format(", ".join(self.X), self.name))
	#
	def deployInverseSQL(self):
		sql = "REVERSE_NORMALIZE({} USING PARAMETERS model_name = '{}')"
		return (sql.format(", ".join(self.X), self.name))
	#
	def drop(self):
		drop_model(self.name, self.cursor, print_info = False)
	#
	def fit(self, input_relation: str, X: list):
		self.input_relation = input_relation
		self.X = [str_column(column) for column in X]
		query = "SELECT NORMALIZE_FIT('{}', '{}', '{}', '{}')".format(self.name, input_relation, ", ".join(self.X), self.method)
		self.cursor.execute(query)
		self.param = to_tablesample(query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'details')".format(self.name), cursor = self.cursor)
		self.param.table_info = False
		return (self)
	#
	def to_vdf(self, reverse = False):
		func = self.deploySQL() if not(reverse) else self.deployInverseSQL()
		from vertica_ml_python.utilities import vdf_from_relation
		return (vdf_from_relation("(SELECT {} FROM {}) {}".format(func, self.input_relation, self.name), self.name, self.cursor))
#
class OneHotEncoder:
	#
	def  __init__(self,
				  name: str,
				  cursor, 
				  extra_levels: dict = {},
				  drop_first: bool = True,
				  ignore_null: bool = True):
		self.type = "preprocessing"
		self.name = name
		self.cursor = cursor
		self.drop_first = drop_first
		self.ignore_null = ignore_null
		self.extra_levels = extra_levels
	# 
	def __repr__(self):
		try:
			self.cursor.execute("SELECT GET_MODEL_SUMMARY(USING PARAMETERS model_name = '" + self.name + "')")
			return (self.cursor.fetchone()[0])
		except:
			return "<OneHotEncoder>"
	#
	#
	#
	# METHODS
	# 
	#
	def deploySQL(self):
		sql = "APPLY_ONE_HOT_ENCODER({} USING PARAMETERS model_name = '{}', column_naming = 'values_relaxed', drop_first = {}, ignore_null = {})"
		return (sql.format(", ".join(self.X), self.name, self.drop_first, self.ignore_null))
	#
	def drop(self):
		drop_model(self.name, self.cursor, print_info = False)
	#
	def fit(self, input_relation: str, X: list):
		self.input_relation = input_relation
		self.X = [str_column(column) for column in X]
		query = "SELECT ONE_HOT_ENCODER_FIT('{}', '{}', '{}' USING PARAMETERS extra_levels = '{}')".format(self.name, input_relation, ", ".join(self.X), self.extra_levels)
		self.cursor.execute(query)
		try:
			self.param = to_tablesample(query = "SELECT category_name, category_level::varchar, category_level_index FROM (SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'integer_categories')) x UNION ALL SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'varchar_categories')".format(self.name, self.name), cursor = self.cursor)
		except:
			try:
				self.param = to_tablesample(query = "SELECT category_name, category_level::varchar, category_level_index FROM (SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'integer_categories')) x".format(self.name), cursor = self.cursor)
			except:
				self.param = to_tablesample(query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'varchar_categories')".format(self.name), cursor = self.cursor)
		self.param.table_info = False
		return (self)
	#
	def to_vdf(self, reverse = False):
		from vertica_ml_python.utilities import vdf_from_relation
		return (vdf_from_relation("(SELECT {} FROM {}) {}".format(self.deploySQL(), self.input_relation, self.name), self.name, self.cursor))