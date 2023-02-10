"""
(c)  Copyright  [2018-2023]  OpenText  or one of its
affiliates.  Licensed  under  the   Apache  License,
Version 2.0 (the  "License"); You  may  not use this
file except in compliance with the License.

You may obtain a copy of the License at:
http://www.apache.org/licenses/LICENSE-2.0

Unless  required  by applicable  law or  agreed to in
writing, software  distributed  under the  License is
distributed on an  "AS IS" BASIS,  WITHOUT WARRANTIES
OR CONDITIONS OF ANY KIND, either express or implied.
See the  License for the specific  language governing
permissions and limitations under the License.
"""

#
#
# Modules
#
# Standard Python Modules
import os
from typing import Literal

# VerticaPy Modules
import vertica_python, verticapy
from verticapy.utils._decorators import (
    save_verticapy_logs,
    check_minimum_version,
)
from verticapy import vDataFrame
from verticapy.connect import current_cursor
from verticapy.utilities import *
from verticapy.utils._toolbox import *
from verticapy.utils._gen import gen_tmp_name
from verticapy.sql.read import _executeSQL
from verticapy.errors import *
from verticapy.learn.vmodel import *
from verticapy.learn.tools import *
from verticapy.sql._utils._format import quote_ident, schema_relation


class BisectingKMeans(Clustering, Tree):
    """
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
init: str / list, optional
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

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str,
        n_cluster: int = 8,
        bisection_iterations: int = 1,
        split_method: Literal["size", "sum_squares"] = "sum_squares",
        min_divisible_cluster_size: int = 2,
        distance_method: Literal["euclidean"] = "euclidean",
        init: Union[Literal["kmeanspp", "pseudo", "random"], list] = "kmeanspp",
        max_iter: int = 300,
        tol: float = 1e-4,
    ):
        self.type, self.name = "BisectingKMeans", name
        self.VERTICA_FIT_FUNCTION_SQL = "BISECTING_KMEANS"
        self.VERTICA_PREDICT_FUNCTION_SQL = "APPLY_BISECTING_KMEANS"
        self.MODEL_TYPE = "UNSUPERVISED"
        self.MODEL_SUBTYPE = "CLUSTERING"
        self.parameters = {
            "n_cluster": n_cluster,
            "bisection_iterations": bisection_iterations,
            "split_method": str(split_method).lower(),
            "min_divisible_cluster_size": min_divisible_cluster_size,
            "distance_method": str(distance_method).lower(),
            "init": init,
            "max_iter": max_iter,
            "tol": tol,
        }

    def get_tree(self):
        """
    Returns a table containing information about the BK-tree.
        """
        return self.cluster_centers_


class DBSCAN(vModel):
    """
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

    @save_verticapy_logs
    def __init__(self, name: str, eps: float = 0.5, min_samples: int = 5, p: int = 2):
        self.type, self.name = "DBSCAN", name
        self.VERTICA_FIT_FUNCTION_SQL = ""
        self.VERTICA_PREDICT_FUNCTION_SQL = ""
        self.MODEL_TYPE = "UNSUPERVISED"
        self.MODEL_SUBTYPE = "CLUSTERING"
        self.parameters = {"eps": eps, "min_samples": min_samples, "p": p}

    def fit(
        self,
        input_relation: Union[str, vDataFrame],
        X: Union[str, list] = [],
        key_columns: Union[str, list] = [],
        index: str = "",
    ):
        """
	Trains the model.

	Parameters
	----------
	input_relation: str / vDataFrame
		Training relation.
	X: str / list, optional
		List of the predictors. If empty, all the numerical vcolumns will be used.
	key_columns: str / list, optional
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
        if verticapy.OPTIONS["overwrite_model"]:
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
        name_main = gen_tmp_name(name="main")
        name_dbscan_clusters = gen_tmp_name(name="clusters")
        try:
            if not (index):
                index = "id"
                drop(f"v_temp_schema.{name_main}", method="table")
                _executeSQL(
                    query=f"""
                    CREATE LOCAL TEMPORARY TABLE {name_main} 
                    ON COMMIT PRESERVE ROWS AS 
                    SELECT /*+LABEL('learn.cluster.DBSCAN.fit')*/
                        ROW_NUMBER() OVER() AS id, 
                        {', '.join(X + key_columns)} 
                    FROM {self.input_relation} 
                    WHERE {' AND '.join([f"{item} IS NOT NULL" for item in X])}""",
                    title="Computing the DBSCAN Table [Step 0]",
                )
            else:
                _executeSQL(
                    query=f"""
                        SELECT 
                            /*+LABEL('learn.cluster.DBSCAN.fit')*/ 
                            {', '.join(X + key_columns + [index])} 
                        FROM {self.input_relation} 
                        LIMIT 10""",
                    print_time_sql=False,
                )
                name_main = self.input_relation
            distance = [
                f"POWER(ABS(x.{X[i]} - y.{X[i]}), {self.parameters['p']})"
                for i in range(len(X))
            ]
            distance = f"POWER({' + '.join(distance)}, 1 / {self.parameters['p']})"
            table = f"""
                SELECT 
                    node_id, 
                    nn_id, 
                    SUM(CASE 
                          WHEN distance <= {self.parameters['eps']} 
                            THEN 1 
                          ELSE 0 
                        END) 
                        OVER (PARTITION BY node_id) AS density, 
                    distance 
                FROM (SELECT 
                          x.{index} AS node_id, 
                          y.{index} AS nn_id, 
                          {distance} AS distance 
                      FROM 
                      {name_main} AS x 
                      CROSS JOIN 
                      {name_main} AS y) distance_table"""
            if isinstance(verticapy.OPTIONS["random_state"], int):
                order_by = "ORDER BY node_id, nn_id"
            else:
                order_by = ""
            graph = _executeSQL(
                query=f"""
                    SELECT /*+LABEL('learn.cluster.DBSCAN.fit')*/
                        node_id, 
                        nn_id 
                    FROM ({table}) VERTICAPY_SUBTABLE 
                    WHERE density > {self.parameters['min_samples']} 
                      AND distance < {self.parameters['eps']} 
                      AND node_id != nn_id {order_by}""",
                title="Computing the DBSCAN Table [Step 1]",
                method="fetchall",
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
                f = open(f"{name_dbscan_clusters}.csv", "w")
                for c in clusters:
                    f.write(f"{c}, {clusters[c]}\n")
                f.close()
                drop(f"v_temp_schema.{name_dbscan_clusters}", method="table")
                _executeSQL(
                    query=f"""
                    CREATE LOCAL TEMPORARY TABLE {name_dbscan_clusters}
                    (node_id int, cluster int) 
                    ON COMMIT PRESERVE ROWS""",
                    print_time_sql=False,
                )
                query = f"""
                    COPY v_temp_schema.{name_dbscan_clusters}(node_id, cluster) 
                    FROM {{}} 
                         DELIMITER ',' 
                         ESCAPE AS '\\';"""
                if isinstance(current_cursor(), vertica_python.vertica.cursor.Cursor):
                    _executeSQL(
                        query=query.format("STDIN"),
                        method="copy",
                        print_time_sql=False,
                        path=f"./{name_dbscan_clusters}.csv",
                    )
                else:
                    _executeSQL(
                        query=query.format(f"LOCAL './{name_dbscan_clusters}.csv'"),
                        print_time_sql=False,
                    )
                _executeSQL("COMMIT;", print_time_sql=False)
            finally:
                os.remove(f"{name_dbscan_clusters}.csv")
            self.n_cluster_ = i
            _executeSQL(
                query=f"""
                    CREATE TABLE {self.name} AS 
                       SELECT /*+LABEL('learn.cluster.DBSCAN.fit')*/
                            {', '.join(self.X + self.key_columns)}, 
                            COALESCE(cluster, -1) AS dbscan_cluster 
                       FROM v_temp_schema.{name_main} AS x 
                       LEFT JOIN v_temp_schema.{name_dbscan_clusters} AS y 
                       ON x.{index} = y.node_id""",
                title="Computing the DBSCAN Table [Step 2]",
            )
            self.n_noise_ = _executeSQL(
                query=f"""
                    SELECT 
                        /*+LABEL('learn.cluster.DBSCAN.fit')*/ 
                        COUNT(*) 
                    FROM {self.name} 
                    WHERE dbscan_cluster = -1""",
                method="fetchfirstelem",
                print_time_sql=False,
            )
        finally:
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

    def predict(self):
        """
	Creates a vDataFrame of the model.

	Returns
	-------
	vDataFrame
 		the vDataFrame including the prediction.
		"""
        return vDataFrame(self.name)


class KMeans(Clustering):
    """
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
init: str / list, optional
	The method to use to find the initial cluster centers.
		kmeanspp : Uses the KMeans++ method to initialize the centers.
		random   : The centers are initialized randomly.
	It can be also a list with the initial cluster centers to use.
max_iter: int, optional
	The maximum number of iterations the algorithm performs.
tol: float, optional
	Determines whether the algorithm has converged. The algorithm is considered 
	converged after no center has moved more than a distance of 'tol' from the 
	previous iteration.
	"""

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str,
        n_cluster: int = 8,
        init: Union[Literal["kmeanspp", "random"], list] = "kmeanspp",
        max_iter: int = 300,
        tol: float = 1e-4,
    ):
        self.type, self.name = "KMeans", name
        self.VERTICA_FIT_FUNCTION_SQL = "KMEANS"
        self.VERTICA_PREDICT_FUNCTION_SQL = "APPLY_KMEANS"
        self.MODEL_TYPE = "UNSUPERVISED"
        self.MODEL_SUBTYPE = "CLUSTERING"
        self.parameters = {
            "n_cluster": n_cluster,
            "init": init,
            "max_iter": max_iter,
            "tol": tol,
        }

    def plot_voronoi(
        self, max_nb_points: int = 50, plot_crosses: bool = True, ax=None, **style_kwds,
    ):
        """
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
            from verticapy.plotting._matplotlib import voronoi_plot

            clusters = _executeSQL(
                query=f"""
                SELECT 
                    /*+LABEL('learn.cluster.KMeans.plot_voronoi')*/ 
                    GET_MODEL_ATTRIBUTE(USING PARAMETERS 
                                        model_name = '{self.name}', 
                                        attr_name = 'centers')""",
                print_time_sql=False,
                method="fetchall",
            )
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


class KPrototypes(Clustering):
    """
Creates a KPrototypes object by using the Vertica k-prototypes algorithm on 
the data. The algorithm combines the k-means and k-modes algorithms in order
to handle both numerical and categorical data.

Parameters
----------
name: str
    Name of the the model. The model is stored in the database.
n_cluster: int, optional
    Number of clusters.
init: str / list, optional
    The method used to find the initial cluster centers.
        random   : The centers are initialized randomly.
    You can also provide a list of initial cluster centers.
max_iter: int, optional
    The maximum number of iterations the algorithm performs.
tol: float, optional
    Determines whether the algorithm has converged. The algorithm is considered 
    converged when no center moves more than a distance of 'tol' from the 
    previous iteration.
gamma: float, optional
    Weighting factor for categorical columns. It determines the relative 
    importance of numerical and categorical attributes.
    """

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str,
        n_cluster: int = 8,
        init: Union[Literal["random"], list] = "random",
        max_iter: int = 300,
        tol: float = 1e-4,
        gamma: float = 1.0,
    ):
        self.type, self.name = "KPrototypes", name
        self.VERTICA_FIT_FUNCTION_SQL = "KPROTOTYPES"
        self.VERTICA_PREDICT_FUNCTION_SQL = "APPLY_KPROTOTYPES"
        self.MODEL_TYPE = "UNSUPERVISED"
        self.MODEL_SUBTYPE = "CLUSTERING"
        self.parameters = {
            "n_cluster": n_cluster,
            "init": init,
            "max_iter": max_iter,
            "tol": tol,
            "gamma": gamma,
        }
