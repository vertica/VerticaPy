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
from verticapy.utilities import *
from verticapy.toolbox import *
from verticapy import vDataFrame
from verticapy.connections.connect import read_auto_connect
#---#
class DBSCAN:
	"""
---------------------------------------------------------------------------
[Beta Version]
Creates a DBSCAN object by using the DBSCAN algorithm as defined by Martin 
Ester, Hans-Peter Kriegel, Jörg Sander and Xiaowei Xu. This object is using 
pure SQL to compute all the distances and neighbors. It is also using Python 
to compute the cluster propagation (non scalable phase). This model is using 
CROSS JOIN and may be really expensive in some cases. It will index all the 
elements of the table in order to be optimal (the CROSS JOIN will happen only 
with IDs which are integers). As DBSCAN is using the p-distance, it is highly 
sensible to un-normalized data. However, DBSCAN is really robust to outliers 
and can find non-linear clusters. It is a very powerful algorithm for outliers 
detection and clustering. 

Parameters
----------
name: str
	Name of the the model. As it is not a built in model, this name will be used
	to build the final table.
cursor: DBcursor, optional
	Vertica DB cursor.
eps: float, optional
	The radius of a neighborhood with respect to some point.
min_samples: int, optional
	Minimum number of points required to form a dense region.
p: int, optional
	The p of the p-distance (distance metric used during the model computation).

Attributes
----------
After the object creation, all the parameters become attributes. 
The model will also create extra attributes when fitting the model:

n_cluster: int
	Number of clusters created during the process.
n_noise: int
	Number of points with no clusters.
input_relation: str
	Train relation.
X: list
	List of the predictors.
key_columns: list
	Columns not used during the algorithm computation but which will be used
	to create the final relation.
	"""
	#
	# Special Methods
	#
	#---#
	def  __init__(self,
				  name: str,
				  cursor = None,
				  eps: float = 0.5,
				  min_samples: int = 5,
				  p: int = 2):
		check_types([
			("name", name, [str], False),
			("eps", eps, [int, float], False),
			("min_samples", min_samples, [int, float], False),
			("p", p, [int, float], False)])
		if not(cursor):
			cursor = read_auto_connect().cursor()
		else:
			check_cursor(cursor)
		self.type = "clustering"
		self.name = name
		self.cursor = cursor
		self.eps = eps
		self.min_samples = min_samples
		self.p = p 
	#---#
	def __repr__(self):
		try:
			rep = "<DBSCAN>\nNumber of Clusters: {}\nNumber of Outliers: {}".format(self.n_cluster, self.n_noise)
			return (rep)
		except:
			return "<DBSCAN>"
	#
	# Methods
	#
	#---#
	def fit(self, 
			input_relation: str, 
			X: list, 
			key_columns: list = [], 
			index: str = ""):
		"""
	---------------------------------------------------------------------------
	Trains the model.

	Parameters
	----------
	input_relation: str
		Train relation.
	X: list
		List of the predictors.
	key_columns: list, optional
		Columns not used during the algorithm computation but which will be used
		to create the final relation.
	index: str, optional
		Index used to identify each row separately. It is highly recommanded to
		have one already in the main table to avoid creation of temporary tables.

	Returns
	-------
	object
 		self
		"""
		check_types([
			("input_relation", input_relation, [str], False),
			("X", X, [list], False),
			("key_columns", key_columns, [list], False),
			("index", index, [str], False)])
		X = [str_column(column) for column in X]
		self.X = X
		self.key_columns = [str_column(column) for column in key_columns]
		self.input_relation = input_relation
		schema, relation = schema_relation(input_relation)
		relation_alpha = ''.join(ch for ch in relation if ch.isalnum())
		cursor = self.cursor
		if not(index):
			index = "id"
			main_table = "VERTICAPY_MAIN_{}".format(relation_alpha)
			try:
				cursor.execute("DROP TABLE IF EXISTS v_temp_schema.{}".format(main_table))
			except:
				pass
			sql = "CREATE LOCAL TEMPORARY TABLE {} ON COMMIT PRESERVE ROWS AS SELECT ROW_NUMBER() OVER() AS id, {} FROM {} WHERE {}".format(main_table, ", ".join(X + key_columns), input_relation, " AND ".join(["{} IS NOT NULL".format(item) for item in X]))
			cursor.execute(sql)
		else:
			cursor.execute("SELECT {} FROM {} LIMIT 10".format(", ".join(X + key_columns + [index]), input_relation))
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
				if (clusters[node] != None and clusters[node_neighbor] == None):
					clusters[node_neighbor] = clusters[node]
				elif (clusters[node_neighbor] != None and clusters[node] == None):
					clusters[node] = clusters[node_neighbor]
			del(graph[0])
		try:
			f = open("VERTICAPY_DBSCAN_CLUSTERS_ID.csv", 'w')
			for elem in clusters:
				f.write("{}, {}\n".format(elem, clusters[elem]))
			f.close()
			try:
				cursor.execute("DROP TABLE IF EXISTS v_temp_schema.VERTICAPY_DBSCAN_CLUSTERS")
			except:
				pass
			cursor.execute("CREATE LOCAL TEMPORARY TABLE VERTICAPY_DBSCAN_CLUSTERS(node_id int, cluster int) ON COMMIT PRESERVE ROWS")
			if ("vertica_python" in str(type(cursor))):
				with open('./VERTICAPY_DBSCAN_CLUSTERS_ID.csv', "r") as fs:
					cursor.copy("COPY v_temp_schema.VERTICAPY_DBSCAN_CLUSTERS(node_id, cluster) FROM STDIN DELIMITER ',' ESCAPE AS '\\';", fs)
			else:
				cursor.execute("COPY v_temp_schema.VERTICAPY_DBSCAN_CLUSTERS(node_id, cluster) FROM LOCAL './VERTICAPY_DBSCAN_CLUSTERS_ID.csv' DELIMITER ',' ESCAPE AS '\\';")
			try:
				cursor.execute("COMMIT")
			except:
				pass
			os.remove("VERTICAPY_DBSCAN_CLUSTERS_ID.csv")
		except:
			os.remove("VERTICAPY_DBSCAN_CLUSTERS_ID.csv")
			raise
		self.n_cluster = i
		cursor.execute("CREATE TABLE {} AS SELECT {}, COALESCE(cluster, -1) AS dbscan_cluster FROM v_temp_schema.{} AS x LEFT JOIN v_temp_schema.VERTICAPY_DBSCAN_CLUSTERS AS y ON x.{} = y.node_id".format(self.name, ", ".join(self.X + self.key_columns), main_table, index))
		cursor.execute("SELECT COUNT(*) FROM {} WHERE dbscan_cluster = -1".format(self.name))
		self.n_noise = cursor.fetchone()[0]
		cursor.execute("DROP TABLE IF EXISTS v_temp_schema.VERTICAPY_MAIN_{}".format(relation_alpha))
		cursor.execute("DROP TABLE IF EXISTS v_temp_schema.VERTICAPY_DBSCAN_CLUSTERS")
		return (self)
	#---#
	def info(self):
		"""
	---------------------------------------------------------------------------
	Displays some information about the model.
		"""
		try:
			print("DBSCAN was successfully achieved by building {} cluster(s) and by identifying {} elements as noise.\nIf you are not happy with the result, do not forget to normalise the data before applying DBSCAN. As this algorithm is using the p-distance, it is really sensible to the data distribution.".format(self.n_cluster, self.n_noise))
		except:
			print("Please use the 'fit' method to start the algorithm.")
	#---#
	def plot(self):
		"""
	---------------------------------------------------------------------------
	Draws the model is the number of predictors is 2 or 3.
		"""
		if (2 <= len(self.X) <= 3):
			vDataFrame(self.name, self.cursor).scatter(columns = self.X, catcol = "dbscan_cluster", max_cardinality = 100, max_nb_points = 10000)
		else:
			raise ValueError("Clustering Plots are only available in 2D or 3D")
	#---#
	def to_vdf(self):
		"""
	---------------------------------------------------------------------------
	Creates a vDataFrame of the model.

	Returns
	-------
	vDataFrame
 		model vDataFrame
		"""
		return (vDataFrame(self.name, self.cursor))
#---#
class KMeans:
	"""
---------------------------------------------------------------------------
Creates a KMeans object by using the Vertica Highly Distributed and Scalable 
KMeans on the data. K-means clustering is a method of vector quantization, 
originally from signal processing, that aims to partition n observations into 
k clusters in which each observation belongs to the cluster with the nearest 
mean (cluster centers or cluster centroid), serving as a prototype of the 
cluster. This results in a partitioning of the data space into Voronoi cells. 

Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
cursor: DBcursor, optional
	Vertica DB cursor.
n_cluster: int, optional
	Number of clusters
init: str/list, optional
	The method used to find the initial cluster centers.
		kmeanspp : Uses the KMeans++ method to initialize the centers.
		random   : The initial centers.
	It can be also a list with the initial cluster centers to use.
max_iter: int, optional
	The maximum number of iterations the algorithm performs.
tol: float, optional
	Determines whether the algorithm has converged. The algorithm is considered 
	converged after no center has moved more than a distance of 'tol' from the 
	previous iteration. 

Attributes
----------
After the object creation, all the parameters become attributes. 
The model will also create extra attributes when fitting the model:

cluster_centers: tablesample
	Clusters result of the algorithm.
metrics: tablesample
	Different metrics to evaluate the model.
input_relation: str
	Train relation.
X: list
	List of the predictors.
	"""
	def  __init__(self,
				  name: str,
				  cursor = None,
				  n_cluster: int = 8,
				  init: str = "kmeanspp",
				  max_iter: int = 300,
				  tol: float = 1e-4):
		check_types([
			("name", name, [str], False),
			("n_cluster", n_cluster, [int, float], False),
			("max_iter", max_iter, [int, float], False),
			("tol", tol, [int, float], False)])
		if not(cursor):
			cursor = read_auto_connect().cursor()
		else:
			check_cursor(cursor)
		self.type = "clustering"
		self.name = name
		self.cursor = cursor
		self.n_cluster = n_cluster
		if (type(init) == str):
			self.init = init.lower()
		else:
			self.init = init
		self.max_iter = max_iter 
		self.tol = tol 
	#---#
	def __repr__(self):
		try:
			self.cursor.execute("SELECT GET_MODEL_SUMMARY(USING PARAMETERS model_name = '" + self.name + "')")
			return (self.cursor.fetchone()[0])
		except:
			return "<KMeans>"
	#
	# Methods
	#
	#---#
	def deploySQL(self):
		"""
	---------------------------------------------------------------------------
	Returns the SQL code needed to deploy the model. 

	Returns
	-------
	str
 		the SQL code needed to deploy the model.
		"""
		sql = "APPLY_KMEANS({} USING PARAMETERS model_name = '{}', match_by_pos = 'true')"
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
		query = "SELECT KMEANS('{}', '{}', '{}', {} USING PARAMETERS max_iterations = {}, epsilon = {}".format(self.name, input_relation, ", ".join(self.X), self.n_cluster, self.max_iter, self.tol)
		schema = schema_relation(self.name)[0]
		name = "VERTICAPY_KMEANS_INITIAL"
		if (type(self.init) != str):
			try:
				self.cursor.execute("DROP TABLE IF EXISTS {}.{}".format(schema, name))
			except:
				pass
			if (len(self.init) != self.n_cluster):
				raise ValueError("'init' must be a list of 'n_cluster' = {} points".format(self.n_cluster))
			else:
				for item in self.init:
					if (len(X) != len(item)):
						raise ValueError("Each points of 'init' must be of size len(X) = {}".format(len(self.X)))
				query0 = []
				for i in range(len(self.init)):
					line = []
					for j in range(len(self.init[0])):
						line += [str(self.init[i][j]) + " AS " + X[j]]
					line = ",".join(line)
					query0 += ["SELECT " + line]
				query0 = " UNION ".join(query0)
				query0 = "CREATE TABLE {}.{} AS {}".format(schema, name, query0)
				self.cursor.execute(query0)
				query += ", initial_centers_table = '{}.{}'".format(schema, name)
		else:
			query += ", init_method = '{}'".format(self.init)
		query += ")"
		self.cursor.execute(query)
		try:
			self.cursor.execute("DROP TABLE IF EXISTS {}.{}".format(schema, name))
		except:
			pass
		self.cluster_centers = to_tablesample(query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'centers')".format(self.name), cursor = self.cursor)
		self.cluster_centers.table_info = False
		query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'metrics')".format(self.name)
		self.cursor.execute(query)
		result = self.cursor.fetchone()[0]
		values = {"index": ["Between-Cluster Sum of Squares", "Total Sum of Squares", "Total Within-Cluster Sum of Squares", "Between-Cluster SS / Total SS", "converged"]}
		values["value"] = [float(result.split("Between-Cluster Sum of Squares: ")[1].split("\n")[0]), float(result.split("Total Sum of Squares: ")[1].split("\n")[0]), float(result.split("Total Within-Cluster Sum of Squares: ")[1].split("\n")[0]), float(result.split("Between-Cluster Sum of Squares: ")[1].split("\n")[0]) / float(result.split("Total Sum of Squares: ")[1].split("\n")[0]), result.split("Converged: ")[1].split("\n")[0] == "True"] 
		self.metrics = tablesample(values, table_info = False)
		return (self)
	#---#
	def plot(self, 
			 voronoi: bool = False):
		"""
	---------------------------------------------------------------------------
	Draws the KMeans clusters.

	Parameters
	----------
	voronoi: bool, optional
		If set to true, a voronoi plot will be drawn. It is only available for
		KMeans using 2 predictors.
		"""
		if (voronoi):
			if (len(self.X) == 2):
				from verticapy.learn.plot import voronoi_plot
				query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'centers')".format(self.name)
				self.cursor.execute(query)
				clusters = self.cursor.fetchall()
				voronoi_plot(clusters = clusters, columns = self.X)
			else:
				raise ValueError("Voronoi Plots are only available in 2D")
		else:
			vdf = vDataFrame(self.input_relation, self.cursor)
			self.predict(vdf, "kmeans_cluster")
			if (len(self.X) <= 3):
				vdf.scatter(columns = self.X, catcol = "kmeans_cluster", max_cardinality = 100, max_nb_points = 10000)
			else:
				raise ValueError("Clustering Plots are only available in 2D or 3D")
	#---#
	def predict(self, 
				vdf, 
				name: str = ""):
		"""
	---------------------------------------------------------------------------
	Adds the prediction in a vDataFrame.

	Parameters
	----------
	vdf: vDataFrame
		Object used to insert the prediction as a vcolumn.
	name: str, optional
		Name of the added vcolumn. If empty, a name will be generated.

	Returns
	-------
	vDataFrame
		the input object.
		"""
		check_types([
			("name", name, [str], False)],
			vdf = ["vdf", vdf])
		name = "KMeans_" + ''.join(ch for ch in self.name if ch.isalnum()) if not (name) else name
		return (vdf.eval(name, self.deploySQL()))