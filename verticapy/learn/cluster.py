# (c) Copyright [2018-2022] Micro Focus or one of its affiliates.
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
# VerticaPy is a Python library with scikit-like functionality for conducting
# data science projects on data stored in Vertica, taking advantage Vertica’s
# speed and built-in analytics and machine learning features. It supports the
# entire data science life cycle, uses a ‘pipeline’ mechanism to sequentialize
# data transformation operations, and offers beautiful graphical options.
#
# VerticaPy aims to do all of the above. The idea is simple: instead of moving
# data around for processing, VerticaPy brings the logic to the data.
#
#
# Modules
#
# Standard Python Modules
import os

# VerticaPy Modules
import vertica_python, verticapy
from verticapy import vDataFrame
from verticapy.connect import current_cursor
from verticapy.utilities import *
from verticapy.toolbox import *
from verticapy.errors import *
from verticapy.learn.vmodel import *
from verticapy.learn.tools import *

# ---#
class BisectingKMeans(Clustering, Tree):
    """
---------------------------------------------------------------------------
Creates a BisectingKMeans object using the Vertica bisecting k-means 
algorithm on the data. k-means clustering is a method of vector quantization, 
originally from signal processing, that aims to partition n observations into 
k clusters. Each observation belongs to the cluster with the nearest 
mean (cluster centers or cluster centroid), which serves as a prototype of 
the cluster. This results in a partitioning of the data space into Voronoi
cells. Bisecting k-means combines k-means and hierarchical clustering.

Parameters
----------
name: str
    Name of the the model. The model will be stored in the DB.
n_cluster: int, optional
    Number of clusters
bisection_iterations: int, optional
    The number of iterations the bisecting KMeans algorithm performs for each 
    bisection step. This corresponds to how many times a standalone KMeans 
    algorithm runs in each bisection step. Setting to more than 1 allows 
    the algorithm to run and choose the best KMeans run within each bisection 
    step. Note that if you are using kmeanspp the bisection_iterations value is 
    always 1, because kmeanspp is more costly to run but also better than the 
    alternatives, so it does not require multiple runs.
split_method: str, optional
    The method used to choose a cluster to bisect/split.
        size        : Choose the largest cluster to bisect.
        sum_squares : Choose the cluster with the largest withInSS to bisect.
min_divisible_cluster_size: int, optional
    The minimum number of points of a divisible cluster. Must be greater than or 
    equal to 2.
distance_method: str, optional
    The measure for distance between two data points. Only Euclidean distance 
    is supported at this time.
init: str/list, optional
    The method to use to find the initial KMeans cluster centers.
        kmeanspp : Uses the KMeans++ method to initialize the centers.
        pseudo   : Uses "pseudo center" approach used by Spark, bisects given 
            center without iterating over points.
    It can be also a list with the initial cluster centers to use.
max_iter: int, optional
    The maximum number of iterations the KMeans algorithm performs.
tol: float, optional
    Determines whether the KMeans algorithm has converged. The algorithm is 
    considered converged after no center has moved more than a distance of 
    'tol' from the previous iteration.
    """

    def __init__(
        self,
        name: str,
        n_cluster: int = 8,
        bisection_iterations: int = 1,
        split_method: str = "sum_squares",
        min_divisible_cluster_size: int = 2,
        distance_method: str = "euclidean",
        init: str = "kmeanspp",
        max_iter: int = 300,
        tol: float = 1e-4,
    ):
        check_types([("name", name, [str])])
        self.type, self.name = "BisectingKMeans", name
        self.set_params(
            {
                "n_cluster": n_cluster,
                "bisection_iterations": bisection_iterations,
                "split_method": split_method,
                "min_divisible_cluster_size": min_divisible_cluster_size,
                "distance_method": distance_method,
                "init": init,
                "max_iter": max_iter,
                "tol": tol,
            }
        )
        version(condition=[9, 3, 1])

    # ---#
    def get_tree(self):
        """
    ---------------------------------------------------------------------------
    Returns a table containing information about the BK-tree.
        """
        return self.cluster_centers_


# ---#
class DBSCAN(vModel):
    """
---------------------------------------------------------------------------
[Beta Version]
Creates a DBSCAN object by using the DBSCAN algorithm as defined by Martin 
Ester, Hans-Peter Kriegel, Jörg Sander, and Xiaowei Xu. This object uses 
pure SQL to compute the distances and neighbors and uses Python to compute 
the cluster propagation (non-scalable phase).

\u26A0 Warning : This algorithm uses a CROSS JOIN during computation and
                 is therefore computationally expensive at O(n * n), where
                 n is the total number of elements.
                 This algorithm indexes elements of the table in order to be optimal 
                 (the CROSS JOIN will happen only with IDs which are integers). 
                 Since DBSCAN is uses the p-distance, it is highly sensitive to 
                 unnormalized data. However, DBSCAN is robust to outliers and can 
                 find non-linear clusters. It is a very powerful algorithm for 
                 outliers detection and clustering. A table will be created 
                 at the end of the learning phase.

Parameters
----------
name: str
	Name of the the model. This is not a built-in model, so this name will be used
    to build the final table.
eps: float, optional
	The radius of a neighborhood with respect to some point.
min_samples: int, optional
	Minimum number of points required to form a dense region.
p: int, optional
	The p of the p-distance (distance metric used during the model computation).
	"""

    def __init__(self, name: str, eps: float = 0.5, min_samples: int = 5, p: int = 2):
        check_types([("name", name, [str])])
        self.type, self.name = "DBSCAN", name
        self.set_params({"eps": eps, "min_samples": min_samples, "p": p})

    # ---#
    def fit(
        self,
        input_relation: (str, vDataFrame),
        X: list = [],
        key_columns: list = [],
        index: str = "",
    ):
        """
	---------------------------------------------------------------------------
	Trains the model.

	Parameters
	----------
	input_relation: str/vDataFrame
		Training relation.
	X: list, optional
		List of the predictors. If empty, all the numerical vcolumns will be used.
	key_columns: list, optional
		Columns not used during the algorithm computation but which will be used
		to create the final relation.
	index: str, optional
		Index used to identify each row separately. It is highly recommanded to
        have one already in the main table to avoid creating temporary tables.

	Returns
	-------
	object
 		self
		"""
        if isinstance(key_columns, str):
            key_columns = [key_columns]
        if isinstance(X, str):
            X = [X]
        check_types(
            [
                ("input_relation", input_relation, [str, vDataFrame]),
                ("X", X, [list]),
                ("key_columns", key_columns, [list]),
                ("index", index, [str]),
            ]
        )
        if verticapy.options["overwrite_model"]:
            self.drop()
        else:
            does_model_exist(name=self.name, raise_error=True)
        if isinstance(input_relation, vDataFrame):
            if not (X):
                X = input_relation.numcol()
            input_relation = input_relation.__genSQL__()
        else:
            if not (X):
                X = vDataFrame(input_relation).numcol()
        X = [quote_ident(column) for column in X]
        self.X = X
        self.key_columns = [quote_ident(column) for column in key_columns]
        self.input_relation = input_relation
        schema, relation = schema_relation(input_relation)
        name_main, name_dbscan_clusters = (
            gen_tmp_name(name="main"),
            gen_tmp_name(name="clusters"),
        )
        try:
            if not (index):
                index = "id"
                drop(f"v_temp_schema.{name_main}", method="table")
                sql = """CREATE LOCAL TEMPORARY TABLE {0} 
                         ON COMMIT PRESERVE ROWS AS 
                         SELECT 
                            ROW_NUMBER() OVER() AS id, 
                            {1} 
                         FROM {2} 
                         WHERE {3}""".format(
                    name_main,
                    ", ".join(X + key_columns),
                    self.input_relation,
                    " AND ".join([f"{item} IS NOT NULL" for item in X]),
                )
                executeSQL(sql, title="Computing the DBSCAN Table [Step 0]")
            else:
                executeSQL(
                    "SELECT {0} FROM {1} LIMIT 10".format(
                        ", ".join(X + key_columns + [index]), self.input_relation
                    ),
                    print_time_sql=False,
                )
                name_main = self.input_relation
            sql = [
                "POWER(ABS(x.{0} - y.{0}), {1})".format(X[i], self.parameters["p"])
                for i in range(len(X))
            ]
            distance = "POWER({0}, 1 / {1})".format(
                " + ".join(sql), self.parameters["p"]
            )
            sql = """SELECT 
                        x.{0} AS node_id, 
                        y.{0} AS nn_id, 
                        {1} AS distance 
                     FROM {2} AS x 
                     CROSS JOIN {2} AS y""".format(
                index, distance, name_main
            )
            sql = """SELECT 
                        node_id, 
                        nn_id, 
                        SUM(CASE WHEN distance <= {0} THEN 1 ELSE 0 END) 
                            OVER (PARTITION BY node_id) AS density, 
                        distance 
                     FROM ({1}) distance_table""".format(
                self.parameters["eps"], sql
            )
            if isinstance(verticapy.options["random_state"], int):
                order_by = "ORDER BY node_id, nn_id"
            else:
                order_by = ""
            sql = """SELECT 
                        node_id, 
                        nn_id 
                     FROM ({0}) VERTICAPY_SUBTABLE 
                     WHERE density > {1} 
                        AND distance < {2} 
                        AND node_id != nn_id {3}""".format(
                sql, self.parameters["min_samples"], self.parameters["eps"], order_by,
            )
            graph = executeSQL(
                sql, title="Computing the DBSCAN Table [Step 1]", method="fetchall"
            )
            main_nodes = list(
                dict.fromkeys([elem[0] for elem in graph] + [elem[1] for elem in graph])
            )
            clusters = {}
            for elem in main_nodes:
                clusters[elem] = None
            i = 0
            while graph:
                node = graph[0][0]
                node_neighbor = graph[0][1]
                if (clusters[node] == None) and (clusters[node_neighbor] == None):
                    clusters[node] = i
                    clusters[node_neighbor] = i
                    i = i + 1
                else:
                    if clusters[node] != None and clusters[node_neighbor] == None:
                        clusters[node_neighbor] = clusters[node]
                    elif clusters[node_neighbor] != None and clusters[node] == None:
                        clusters[node] = clusters[node_neighbor]
                del graph[0]
            try:
                f = open("{}.csv".format(name_dbscan_clusters), "w")
                for elem in clusters:
                    f.write("{}, {}\n".format(elem, clusters[elem]))
                f.close()
                drop("v_temp_schema.{}".format(name_dbscan_clusters), method="table")
                executeSQL(
                    (
                        f"CREATE LOCAL TEMPORARY TABLE {name_dbscan_clusters}"
                        "(node_id int, cluster int) ON COMMIT PRESERVE ROWS"
                    ),
                    print_time_sql=False,
                )
                if isinstance(current_cursor(), vertica_python.vertica.cursor.Cursor):
                    executeSQL(
                        (
                            f"COPY v_temp_schema.{name_dbscan_clusters}(node_id, cluster)"
                            " FROM STDIN DELIMITER ',' ESCAPE AS '\\';"
                        ),
                        method="copy",
                        print_time_sql=False,
                        path=f"./{name_dbscan_clusters}.csv",
                    )
                else:
                    executeSQL(
                        """COPY v_temp_schema.{0}(node_id, cluster) 
                           FROM LOCAL './{0}.csv' DELIMITER ',' ESCAPE AS '\\';""".format(
                            name_dbscan_clusters
                        ),
                        print_time_sql=False,
                    )
                executeSQL("COMMIT;", print_time_sql=False)
                os.remove(f"{name_dbscan_clusters}.csv")
            except:
                os.remove(f"{name_dbscan_clusters}.csv")
                raise
            self.n_cluster_ = i
            executeSQL(
                """CREATE TABLE {0} AS 
                   SELECT 
                        {1}, 
                        COALESCE(cluster, -1) AS dbscan_cluster 
                   FROM v_temp_schema.{2} AS x 
                   LEFT JOIN v_temp_schema.{3} AS y 
                   ON x.{4} = y.node_id""".format(
                    self.name,
                    ", ".join(self.X + self.key_columns),
                    name_main,
                    name_dbscan_clusters,
                    index,
                ),
                title="Computing the DBSCAN Table [Step 2]",
            )
            self.n_noise_ = executeSQL(
                "SELECT COUNT(*) FROM {0} WHERE dbscan_cluster = -1".format(self.name),
                method="fetchfirstelem",
                print_time_sql=False,
            )
        except:
            drop(f"v_temp_schema.{name_main}", method="table")
            drop(f"v_temp_schema.{name_dbscan_clusters}", method="table")
            raise
        drop(f"v_temp_schema.{name_main}", method="table")
        drop(f"v_temp_schema.{name_dbscan_clusters}", method="table")
        model_save = {
            "type": "DBSCAN",
            "input_relation": self.input_relation,
            "key_columns": self.key_columns,
            "X": self.X,
            "p": self.parameters["p"],
            "eps": self.parameters["eps"],
            "min_samples": self.parameters["min_samples"],
            "n_cluster": self.n_cluster_,
            "n_noise": self.n_noise_,
        }
        insert_verticapy_schema(
            model_name=self.name, model_type="DBSCAN", model_save=model_save,
        )
        return self

    # ---#
    def predict(self):
        """
	---------------------------------------------------------------------------
	Creates a vDataFrame of the model.

	Returns
	-------
	vDataFrame
 		the vDataFrame including the prediction.
		"""
        return vDataFrame(self.name)


# ---#
class KMeans(Clustering):
    """
---------------------------------------------------------------------------
Creates a KMeans object using the Vertica k-means algorithm on the data. 
k-means clustering is a method of vector quantization, originally from signal 
processing, that aims to partition n observations into k clusters in which 
each observation belongs to the cluster with the nearest mean (cluster centers 
or cluster centroid), serving as a prototype of the cluster. This results in 
a partitioning of the data space into Voronoi cells.

Parameters
----------
name: str
	Name of the the model. The model will be stored in the database.
n_cluster: int, optional
	Number of clusters
init: str/list, optional
	The method to use to find the initial cluster centers.
		kmeanspp : Uses the KMeans++ method to initialize the centers.
		random   : The initial centers.
	It can be also a list with the initial cluster centers to use.
max_iter: int, optional
	The maximum number of iterations the algorithm performs.
tol: float, optional
	Determines whether the algorithm has converged. The algorithm is considered 
	converged after no center has moved more than a distance of 'tol' from the 
	previous iteration.
	"""

    def __init__(
        self,
        name: str,
        n_cluster: int = 8,
        init: str = "kmeanspp",
        max_iter: int = 300,
        tol: float = 1e-4,
    ):
        check_types([("name", name, [str])])
        self.type, self.name = "KMeans", name
        self.set_params(
            {
                "n_cluster": n_cluster,
                "init": init.lower() if isinstance(init, str) else init,
                "max_iter": max_iter,
                "tol": tol,
            }
        )
        version(condition=[8, 0, 0])

    # ---#
    def plot_voronoi(
        self, max_nb_points: int = 50, plot_crosses: bool = True, ax=None, **style_kwds
    ):
        """
    ---------------------------------------------------------------------------
    Draws the Voronoi Graph of the model.

    Parameters
    ----------
    max_nb_points: int, optional
        Maximum number of points to display.
    plot_crosses: bool, optional
        If set to True, the centers are represented by white crosses.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    Figure
        Matplotlib Figure
        """
        if len(self.X) == 2:
            from verticapy.learn.mlplot import voronoi_plot

            query = "SELECT GET_MODEL_ATTRIBUTE(USING PARAMETERS model_name = '{}', attr_name = 'centers')".format(
                self.name
            )
            clusters = executeSQL(query, print_time_sql=False, method="fetchall")
            return voronoi_plot(
                clusters=clusters,
                columns=self.X,
                input_relation=self.input_relation,
                plot_crosses=plot_crosses,
                ax=ax,
                max_nb_points=max_nb_points,
                **style_kwds,
            )
        else:
            raise Exception("Voronoi Plots are only available in 2D")
