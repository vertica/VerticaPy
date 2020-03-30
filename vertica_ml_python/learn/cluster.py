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
import os

from vertica_ml_python import drop_model
from vertica_ml_python import tablesample
from vertica_ml_python import to_tablesample
from vertica_ml_python.utilities import str_column
from vertica_ml_python.utilities import schema_relation

#
class DBSCAN:
	#
	def  __init__(self,
				  name: str,
				  cursor,
				  eps: float = 0.5,
				  min_samples: int = 5,
				  p: int = 2):
		self.type = "clustering"
		self.name = name
		self.cursor = cursor
		self.eps = eps
		self.min_samples = min_samples
		self.p = p 
	# 
	def __repr__(self):
		try:
			rep = "<DBSCAN>\nNumber of Clusters: {}\nNumber of Outliers: {}".format(self.n_cluster, self.n_noise)
			return (rep)
		except:
			return "<DBSCAN>"
	#
	#
	#
	# METHODS
	# 
	#
	def fit(self, input_relation: str, X: list, key_columns: list = [], index: str = ""):
		X = [str_column(column) for column in X]
		self.X = X
		self.key_columns = [str_column(column) for column in key_columns]
		self.input_relation = input_relation
		schema, relation = schema_relation(input_relation)
		relation_alpha = ''.join(ch for ch in relation if ch.isalnum())
		cursor = self.cursor
		if not(index):
			index = "id"
			main_table = "main_{}_vpython_".format(relation_alpha)
			cursor.execute("DROP TABLE IF EXISTS v_temp_schema.{}".format(main_table))
			sql = "CREATE LOCAL TEMPORARY TABLE {} ON COMMIT PRESERVE ROWS AS SELECT ROW_NUMBER() OVER() AS id, {} FROM {} WHERE {}".format(main_table, ", ".join(X + key_columns), input_relation, " AND ".join(["{} IS NOT NULL".format(item) for item in X]))
			cursor.execute(sql)
		else:
			main_table = input_relation
		sql = ["POWER(ABS(x.{} - y.{}), {})".format(X[i], X[i], self.p) for i in range(len(X))] 
		distance = "POWER({}, 1 / {})".format(" + ".join(sql), self.p)
		sql = "SELECT x.{} AS node_id, y.{} AS nn_id, {} AS distance FROM {} AS x CROSS JOIN {} AS y".format(index, index, distance, main_table, main_table)
		sql = "SELECT node_id, nn_id, SUM(CASE WHEN distance <= {} THEN 1 ELSE 0 END) OVER (PARTITION BY node_id) AS density, distance FROM ({}) distance_table".format(self.eps, sql)
		sql = "SELECT node_id, nn_id FROM ({}) x WHERE density > {} AND distance < {} AND node_id != nn_id".format(sql, self.min_samples, self.eps)
		cursor.execute(sql)
		graph = cursor.fetchall()
		main_nodes = list(dict.fromkeys([elem[0] for elem in graph] + [elem[1] for elem in graph]))
		clusters = {}
		for elem in main_nodes:
			clusters[elem] = None
		i = 0
		while (graph):
			node = graph[0][0]
			node_neighbor = graph[0][1]
			if (clusters[node] == None) and (clusters[node_neighbor] == None):
				clusters[node] = i 
				clusters[node_neighbor] = i
				i = i + 1
			else:
				if (clusters[node] != None):
					clusters[node_neighbor] = clusters[node]
				else:
					clusters[node] = clusters[node_neighbor]
			del(graph[0])
		try:
			f = open("dbscan_id_cluster_vpython.csv", 'w')
			for elem in clusters:
				f.write("{}, {}\n".format(elem, clusters[elem]))
			f.close()
			cursor.execute("DROP TABLE IF EXISTS v_temp_schema.dbscan_clusters")
			cursor.execute("CREATE LOCAL TEMPORARY TABLE dbscan_clusters(node_id int, cluster int) ON COMMIT PRESERVE ROWS")
			if ("vertica_python" in str(type(cursor))):
				with open('./dbscan_id_cluster_vpython.csv', "r") as fs:
					cursor.copy("COPY v_temp_schema.dbscan_clusters(node_id, cluster) FROM STDIN DELIMITER ',' ESCAPE AS '\\';", fs)
			else:
				cursor.execute("COPY v_temp_schema.dbscan_clusters(node_id, cluster) FROM LOCAL './dbscan_id_cluster_vpython.csv' DELIMITER ',' ESCAPE AS '\\';")
			cursor.execute("COMMIT")
			os.remove("dbscan_id_cluster_vpython.csv")
		except:
			os.remove("dbscan_id_cluster_vpython.csv")
			raise
		self.n_cluster = i
		cursor.execute("CREATE TABLE {} AS SELECT {}, COALESCE(cluster, -1) AS dbscan_cluster FROM v_temp_schema.{} AS x LEFT JOIN v_temp_schema.dbscan_clusters AS y ON x.{} = y.node_id".format(self.name, ", ".join(self.X + self.key_columns), main_table, index))
		cursor.execute("SELECT COUNT(*) FROM {} WHERE dbscan_cluster = -1".format(self.name))
		self.n_noise = cursor.fetchone()[0]
		cursor.execute("DROP TABLE IF EXISTS v_temp_schema.main_{}_vpython_".format(relation_alpha))
		cursor.execute("DROP TABLE IF EXISTS v_temp_schema.dbscan_clusters")
		return (self)
	#
	def info(self):
		try:
			print("DBSCAN was successfully achieved by building {} cluster(s) and by identifying {} elements as noise.\nIf you are not happy with the result, do not forget to normalise the data before applying DBSCAN. As this algorithm is using the p-distance, it is really sensible to the data distribution.".format(self.n_cluster, self.n_noise))
		except:
			print("Please use the 'fit' method to start the algorithm.")
	#
	def plot(self):
		from vertica_ml_python import vDataframe
		if (len(self.X) <= 3):
			vDataframe(self.name, self.cursor).scatter(columns = self.X, catcol = "dbscan_cluster", max_cardinality = 100, max_nb_points = 10000)
		else:
			raise ValueError("Clustering Plots are only available in 2D or 3D")
	#
	def to_vdf(self):
		from vertica_ml_python import vDataframe
		return (vDataframe(self.name, self.cursor))
#
class KMeans:
	#
	def  __init__(self,
				  name: str,
				  cursor,
				  n_cluster: int = 8,
				  init = "kmeanspp",
				  max_iter: int = 300,
				  tol: float = 1e-4):
		self.type = "clustering"
		self.name = name
		self.cursor = cursor
		self.n_cluster = n_cluster
		self.init = init
		self.max_iter = max_iter 
		self.tol = tol 
	# 
	def __repr__(self):
		try:
			self.cursor.execute("SELECT GET_MODEL_SUMMARY(USING PARAMETERS model_name = '" + self.name + "')")
			return (self.cursor.fetchone()[0])
		except:
			return "<KMeans>"
	#
	#
	#
	# METHODS
	# 
	#
	def add_to_vdf(self,
				   vdf,
				   name: str = ""):
		name = "KMeans_" + self.name if not (name) else name
		return (vdf.eval(name, self.deploySQL()))
	#
	def deploySQL(self):
		sql = "APPLY_KMEANS({} USING PARAMETERS model_name = '{}', match_by_pos = 'true')"
		return (sql.format(", ".join(self.X), self.name))
	#
	def drop(self):
		drop_model(self.name, self.cursor, print_info = False)
	#
	def fit(self, input_relation: str, X: list):
		self.input_relation = input_relation
		self.X = [str_column(column) for column in X]
		query = "SELECT KMEANS('{}', '{}', '{}', {} USING PARAMETERS max_iterations = {}, epsilon = {}".format(self.name, input_relation, ", ".join(self.X), self.n_cluster, self.max_iter, self.tol)
		name = "_vpython_kmeans_initial_centers_table_" 
		if (type(self.init) != str):
			self.cursor.execute("DROP TABLE IF EXISTS v_temp_schema.{}".format(name))
			if (len(self.init) != self.n_cluster):
				raise ValueError("'init' must be a list of 'n_cluster' = {} points".format(self.n_cluster))
			else:
				for item in self.init:
					if (len(X) != len(item)):
						raise ValueError("Each points of 'init' must be of size len(X) = {}".format(len(self.X)))
				temp_initial_centers = [item for item in self.init]
				for item in temp_initial_centers:
					del temp_initial_centers[0]
					if (item in temp_initial_centers):
						raise ValueError("All the points of 'init' must be different")
				query0 = []
				for i in range(len(self.init)):
					line = []
					for j in range(len(self.init[0])):
						line += [str(self.init[i][j]) + " AS " + X[j]]
					line = ",".join(line)
					query0 += ["SELECT " + line]
				query0 = " UNION ".join(query0)
				query0 = "CREATE LOCAL TEMPORARY TABLE {} ON COMMIT PRESERVE ROWS AS {}".format(name, query0)
				self.cursor.execute(query0)
				query += ", initial_centers_table = 'v_temp_schema.{}'".format(name)
		else:
			query += ", init_method = '" + self.init + "'"
		query += ")"
		self.cursor.execute(query)
		self.cursor.execute("DROP TABLE IF EXISTS v_temp_schema.{}".format(name))
		self.cluster_centers = to_tablesample(query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'centers')".format(self.name), cursor = self.cursor)
		self.cluster_centers.table_info = False
		query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'metrics')".format(self.name)
		self.cursor.execute(query)
		result = self.cursor.fetchone()[0]
		values = {"index": ["Between-Cluster Sum of Squares", "Total Sum of Squares", "Total Within-Cluster Sum of Squares", "Between-Cluster SS / Total SS", "converged"]}
		values["value"] = [float(result.split("Between-Cluster Sum of Squares: ")[1].split("\n")[0]), float(result.split("Total Sum of Squares: ")[1].split("\n")[0]), float(result.split("Total Within-Cluster Sum of Squares: ")[1].split("\n")[0]), float(result.split("Between-Cluster Sum of Squares: ")[1].split("\n")[0]) / float(result.split("Total Sum of Squares: ")[1].split("\n")[0]), result.split("Converged: ")[1].split("\n")[0] == "True"] 
		self.metrics = tablesample(values, table_info = False)
		return (self)
	#
	def plot(self):
		from vertica_ml_python import vDataframe
		vdf = vDataframe(self.input_relation, self.cursor)
		self.add_to_vdf(vdf, "kmeans_cluster")
		if (len(self.X) <= 3):
			vdf.scatter(columns = self.X, catcol = "kmeans_cluster", max_cardinality = 100, max_nb_points = 10000)
		else:
			raise ValueError("Clustering Plots are only available in 2D or 3D")