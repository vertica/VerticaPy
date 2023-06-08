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
import copy
import os
import vertica_python
from abc import abstractmethod
from typing import Literal, Optional, Union

import numpy as np

from vertica_python.errors import QueryError

import verticapy._config.config as conf
from verticapy._typing import (
    NoneType,
    PlottingObject,
    PythonScalar,
    SQLColumns,
    SQLRelation,
)
from verticapy._utils._gen import gen_tmp_name
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import format_type, quote_ident, schema_relation
from verticapy._utils._sql._sys import _executeSQL
from verticapy._utils._sql._vertica_version import check_minimum_version
from verticapy.connection import current_cursor

from verticapy.core.tablesample.base import TableSample
from verticapy.core.vdataframe.base import vDataFrame

import verticapy.machine_learning.memmodel as mm
from verticapy.machine_learning.vertica.base import (
    MulticlassClassifier,
    Tree,
    Unsupervised,
    VerticaModel,
)

from verticapy.sql.drop import drop

if conf.get_import_success("graphviz"):
    from graphviz import Source

"""
General Class.
"""


class Clustering(Unsupervised):
    # Properties.

    @property
    @abstractmethod
    def _vertica_predict_sql(self) -> str:
        """Must be overridden in child class"""
        raise NotImplementedError

    # System & Special Methods.

    @abstractmethod
    def __init__(self) -> None:
        """Must be overridden in the child class"""
        super().__init__()

    # Prediction / Transformation Methods.

    def predict(
        self,
        vdf: SQLRelation,
        X: Optional[SQLColumns] = None,
        name: Optional[str] = None,
        inplace: bool = True,
    ) -> vDataFrame:
        """
        Makes predictions using the input relation.

        Parameters
        ----------
        vdf: SQLRelation
            Object  used to run the prediction.  You can
            also  specify a  customized relation,  but you
            must  enclose  it with an alias. For  example:
            "(SELECT 1) x" is valid whereas "(SELECT 1)"
            and "SELECT 1" are invalid.
        X: SQLColumns, optional
            List of the columns used to deploy the models.
            If empty, the model predictors are used.
        name: str, optional
            Name  of  the added  vDataColumn. If empty,  a
            name is generated.
        inplace: bool, optional
            If  set to True, the prediction is  added
            to the vDataFrame.

        Returns
        -------
        vDataFrame
            the input object.
        """
        if isinstance(X, NoneType):
            X = self.X
        X = format_type(X, dtype=list)
        if isinstance(vdf, str):
            vdf = vDataFrame(vdf)
        X = quote_ident(X)
        if not name:
            name = (
                self._model_type
                + "_"
                + "".join(ch for ch in self.model_name if ch.isalnum())
            )
        if inplace:
            return vdf.eval(name, self.deploySQL(X=X))
        else:
            return vdf.copy().eval(name, self.deploySQL(X=X))

    def _get_plot_kwargs(
        self,
        nbins: int = 30,
        chart: Optional[PlottingObject] = None,
        method: Optional[str] = None,
    ) -> dict:
        """
        Returns the kwargs used by plotting methods.
        """
        res = {"nbins": nbins, "chart": chart}
        if method == "contour":
            res["func_name"] = "cluster"
        else:
            raise NotImplementedError
        return res

    def plot(
        self,
        max_nb_points: int = 100,
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the model.

        Parameters
        ----------
        max_nb_points: int
            Maximum number  of points to display.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any optional parameter to pass to the
            Plotting functions.

        Returns
        -------
        obj
            Plotting Object.
        """
        vdf = vDataFrame(self.input_relation)
        kwargs = {
            "columns": self.X,
            "max_nb_points": max_nb_points,
            "chart": chart,
            **style_kwargs,
        }
        if self._model_subcategory == "ANOMALY_DETECTION":
            fun = vdf.bubble
            name = "anomaly_score"
            kwargs["cmap_col"] = name
        else:
            fun = vdf.scatter
            name = "cluster"
            kwargs["by"] = name
            kwargs["max_cardinality"] = 100
        self.predict(vdf, name=name)
        return fun(**kwargs)


"""
KMeans Algorithms & Extensions.
"""


class KMeans(Clustering):
    """
    Creates  a  KMeans object using the  Vertica  k-means
    algorithm.  k-means  clustering is a method of vector
    quantization, originally from signal processing, that
    aims to partition  n  observations into k clusters in
    which each observation belongs to the cluster with the
    nearest mean (cluster centers or cluster centroid),
    serving as a prototype of  the cluster.  This results
    in a partitioning of the data space into Voronoi cells.

    Parameters
    ----------
    name: str
        Name  of  the model. The model is stored in the
        database.
    n_cluster: int, optional
        Number of clusters
    init: str / list, optional
        The  method  used to find the initial  cluster
        centers.
                kmeanspp : Uses   the  KMeans++  method   to
                       initialize the centers.
                random   : The   centers   are   initialized
                       randomly.
        You can also provide a list with the initial
        cluster centers.
    max_iter: int, optional
        The  maximum number of iterations  the algorithm
        performs.
    tol: float, optional
        Determines whether the algorithm has  converged.
        The  algorithm is considered converged after  no
        center  has moved more than a distance of  'tol'
        from the previous iteration.
    """

    # Properties.

    @property
    def _vertica_fit_sql(self) -> Literal["KMEANS"]:
        return "KMEANS"

    @property
    def _vertica_predict_sql(self) -> Literal["APPLY_KMEANS"]:
        return "APPLY_KMEANS"

    @property
    def _model_category(self) -> Literal["UNSUPERVISED"]:
        return "UNSUPERVISED"

    @property
    def _model_subcategory(self) -> Literal["CLUSTERING"]:
        return "CLUSTERING"

    @property
    def _model_type(self) -> Literal["KMeans"]:
        return "KMeans"

    @property
    def _attributes(self) -> list[str]:
        return [
            "clusters_",
            "p_",
            "between_cluster_ss_",
            "total_ss_",
            "total_within_cluster_ss_",
            "elbow_score_",
            "converged_",
        ]

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str,
        n_cluster: int = 8,
        init: Union[Literal["kmeanspp", "random"], list] = "kmeanspp",
        max_iter: int = 300,
        tol: float = 1e-4,
    ) -> None:
        super().__init__()
        self.model_name = name
        self.parameters = {
            "n_cluster": n_cluster,
            "init": init,
            "max_iter": max_iter,
            "tol": tol,
        }

    # Attributes Methods.

    def _compute_attributes(self) -> None:
        """
        Computes the model's attributes.
        """
        centers = self.get_vertica_attributes("centers")
        self.clusters_ = centers.to_numpy()
        self.p_ = 2
        self._compute_metrics()

    def _compute_metrics(self) -> None:
        """
        Computes the KMeans metrics.
        """
        metrics_str = self.get_vertica_attributes("metrics").values["metrics"][0]
        metrics = np.array(
            [
                float(
                    metrics_str.split("Between-Cluster Sum of Squares: ")[1].split(
                        "\n"
                    )[0]
                ),
                float(metrics_str.split("Total Sum of Squares: ")[1].split("\n")[0]),
                float(
                    metrics_str.split("Total Within-Cluster Sum of Squares: ")[1].split(
                        "\n"
                    )[0]
                ),
                float(
                    metrics_str.split("Between-Cluster Sum of Squares: ")[1].split(
                        "\n"
                    )[0]
                )
                / float(metrics_str.split("Total Sum of Squares: ")[1].split("\n")[0]),
                metrics_str.split("Converged: ")[1].split("\n")[0] == "True",
            ]
        )
        self.between_cluster_ss_ = metrics[0]
        self.total_ss_ = metrics[1]
        self.total_within_cluster_ss_ = metrics[2]
        self.elbow_score_ = metrics[3]
        self.converged_ = metrics[4]

    # I/O Methods.

    def to_memmodel(self) -> mm.KMeans:
        """
        Converts the  model to an InMemory  object  that
        can be used for different types of predictions.
        """
        return mm.KMeans(
            self.clusters_,
            self.p_,
        )

    # Plotting Methods.

    def plot_voronoi(
        self,
        max_nb_points: int = 50,
        plot_crosses: bool = True,
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the Voronoi Graph of the model.

        Parameters
        ----------
        max_nb_points: int, optional
            Maximum  number  of   points  to   display.
        plot_crosses: bool, optional
            If set to True, the centers are represented
            by white crosses.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any  optional  parameter  to  pass  to  the
            Plotting functions.

        Returns
        -------
        obj
            Plotting Object.
        """
        if len(self.X) == 2:
            vpy_plt, kwargs = self.get_plotting_lib(
                class_name="VoronoiPlot",
                chart=chart,
                matplotlib_kwargs={"plot_crosses": plot_crosses},
                style_kwargs=style_kwargs,
            )
            return vpy_plt.VoronoiPlot(
                vdf=vDataFrame(self.input_relation),
                columns=self.X,
                max_nb_points=max_nb_points,
                misc_data={"clusters": self.clusters_},
            ).draw(**kwargs)
        else:
            raise Exception("Voronoi Plots are only available in 2D")


class KPrototypes(KMeans):
    """
    Creates a KPrototypes object by using the Vertica
    k-prototypes algorithm. The algorithm combines
    the k-means and k-modes  algorithms  in order to
    handle both numerical and  categorical data.

    Parameters
    ----------
    name: str
        Name  of the model. The model is stored in  the
        database.
    n_cluster: int, optional
        Number of clusters.
    init: str / list, optional
        The  method  used  to  find the  initial  cluster
        centers.
            random: The centers are initialized randomly.
        You  can  also provide a list of initial  cluster
        centers.
    max_iter: int, optional
        The  maximum number of iterations the  algorithm
        performs.
    tol: float, optional
        Determines whether  the algorithm has converged.
        The  algorithm is  considered converged when  no
        center  moves more than a distance of 'tol' from
        the previous iteration.
    gamma: float, optional
        Weighting  factor  for  categorical columns.  It
        determines the relative importance of  numerical
        and categorical attributes.
    """

    # Properties.

    @property
    def _vertica_fit_sql(self) -> Literal["KPROTOTYPES"]:
        return "KPROTOTYPES"

    @property
    def _vertica_predict_sql(self) -> Literal["APPLY_KPROTOTYPES"]:
        return "APPLY_KPROTOTYPES"

    @property
    def _model_type(self) -> Literal["KPrototypes"]:
        return "KPrototypes"

    @property
    def _attributes(self) -> list[str]:
        return [
            "clusters_",
            "p_",
            "gamma_",
            "is_categorical_",
            "between_cluster_ss_",
            "total_ss_",
            "total_within_cluster_ss_",
            "elbow_score_",
            "converged_",
        ]

    # System & Special Methods.

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
    ) -> None:
        super().__init__(name)
        self.parameters = {
            "n_cluster": n_cluster,
            "init": init,
            "max_iter": max_iter,
            "tol": tol,
            "gamma": gamma,
        }

    # Attributes Methods.

    def _compute_attributes(self) -> None:
        """
        Computes the model's attributes.
        """
        centers = self.get_vertica_attributes("centers")
        self.clusters_ = centers.to_numpy()
        self.p_ = 2
        self.gamma_ = self.parameters["gamma"]
        dtypes = centers.dtype
        self.is_categorical_ = np.array(
            [("char" in dtypes[key].lower()) for key in dtypes]
        )
        self._compute_metrics()

    # I/O Methods.

    def to_memmodel(self) -> mm.KPrototypes:
        """
        Converts the  model to an InMemory  object that
        can be used for different types of predictions.
        """
        return mm.KPrototypes(
            self.clusters_, self.p_, self.gamma_, self.is_categorical_
        )


class BisectingKMeans(KMeans, Tree):
    """
    Creates   a  BisectingKMeans  object  using   the
    Vertica   bisecting  k-means  algorithm.  k-means
    clustering is a method of vector quantization,
    originally  from  signal processing, that aims to
    partition n observations into k  clusters. Each
    observation  belongs  to the  cluster  with  the
    nearest  mean   (cluster centers  or  cluster
    centroid),  which serves  as a  prototype  of the
    cluster.  This  results  in a partitioning of the
    data space into  Voronoi cells. Bisecting k-means
    combines k-means  and hierarchical clustering.

    Parameters
    ----------
    name: str
        Name of the model.  The  model is stored in
        the database.
    n_cluster: int, optional
        Number of clusters
    bisection_iterations: int, optional
        The number of iterations the bisecting KMeans
        algorithm  performs for each bisection  step.
        This   corresponds  to  how   many  times  a
        standalone  KMeans  algorithm runs  in  each
        bisection step. Setting to a value greater
        than 1 allows the algorithm to run and choose
        the best KMeans run within each bisection
        step.  If you are using  kmeanspp,   the
        bisection_iterations value is always 1
        because kmeanspp is more costly  to run  but
        also  better  than  the alternatives,  so it
        does  not   require multiple runs.
    split_method: str, optional
        The method used to choose a cluster to
        bisect/split.
            size        : Choose the largest cluster
                          to bisect.
            sum_squares : Choose the cluster with the
                          largest withInSS to bisect.
    min_divisible_cluster_size: int, optional
        The minimum number of points of a divisible
        cluster. Must be greater than or equal to 2.
    distance_method: str, optional
        The  distance measure between two  data
        points. Only Euclidean distance is supported
        at this time.
    init: str / list, optional
        The method used to find the initial KMeans
        cluster centers.
            kmeanspp : Uses  the KMeans++ method  to
                       initialize the centers.
            pseudo   : Uses "pseudo center" approach
                       used by Spark,  bisects given
                       center without iterating over
                       points.
        You can also provide a list with the initial
        cluster centers.
    max_iter: int, optional
        The maximum number of iterations the KMeans
        algorithm performs.
    tol: float, optional
        Determines whether the KMeans algorithm has
        converged.  The   algorithm  is  considered
        converged  after  no center has moved  more
        than a distance of  'tol' from the previous
        iteration.
    """

    # Properties.

    @property
    def _vertica_fit_sql(self) -> Literal["BISECTING_KMEANS"]:
        return "BISECTING_KMEANS"

    @property
    def _vertica_predict_sql(self) -> Literal["APPLY_BISECTING_KMEANS"]:
        return "APPLY_BISECTING_KMEANS"

    @property
    def _model_category(self) -> Literal["UNSUPERVISED"]:
        return "UNSUPERVISED"

    @property
    def _model_subcategory(self) -> Literal["CLUSTERING"]:
        return "CLUSTERING"

    @property
    def _model_type(self) -> Literal["BisectingKMeans"]:
        return "BisectingKMeans"

    @property
    def _attributes(self) -> list[str]:
        return [
            "tree_",
            "clusters_",
            "children_left_",
            "children_right_",
            "cluster_size_",
            "cluster_score_",
            "p_",
            "between_cluster_ss_",
            "total_ss_",
            "total_within_cluster_ss_",
            "elbow_score_",
            "cluster_i_ss_",
        ]

    # System & Special Methods.

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
    ) -> None:
        super().__init__(name)
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

    # Attributes Methods.

    def _compute_attributes(self) -> None:
        """
        Computes the model's attributes.
        """
        centers = self.get_vertica_attributes("BKTree")
        self.tree_ = copy.deepcopy(centers)
        self.clusters_ = centers.to_numpy()[:, 1 : len(self.X) + 1]
        self.children_left_ = np.array(centers["left_child"])
        self.children_right_ = np.array(centers["right_child"])
        self.cluster_size_ = np.array(centers["cluster_size"])
        wt, tot = centers["withinss"], centers["totWithinss"]
        n = len(wt)
        self.cluster_score_ = np.array([wt[i] / tot[i] for i in range(n)])
        self.p_ = 2
        metrics = self.get_vertica_attributes("Metrics")["Value"]
        self.total_ss_ = metrics[0]
        self.total_within_cluster_ss_ = metrics[1]
        self.between_cluster_ss_ = metrics[2]
        self.elbow_score_ = metrics[3]
        self.cluster_i_ss_ = np.array(metrics[4:])

    # Parameters Methods.

    @staticmethod
    def _map_to_vertica_param_dict() -> dict:
        return {
            "tol": "kmeans_epsilon",
            "max_iter": "kmeans_max_iterations",
            "init": "kmeans_center_init_method",
        }

    # I/O Methods.

    def to_memmodel(self) -> mm.BisectingKMeans:
        """
        Converts  the model to an InMemory object  that
        can be used for different types of predictions.
        """
        return mm.BisectingKMeans(
            self.clusters_,
            self.children_left_,
            self.children_right_,
            self.cluster_size_,
            self.cluster_score_,
            self.p_,
        )

    # Trees Representation Methods.

    def get_tree(self) -> TableSample:
        """
        Returns a table containing information about the
        BK-tree.
        """
        return self.tree_

    def to_graphviz(
        self,
        round_score: int = 2,
        percent: bool = False,
        vertical: bool = True,
        node_style: dict = {"shape": "none"},
        arrow_style: Optional[dict] = None,
        leaf_style: Optional[dict] = None,
    ) -> str:
        """
        Returns the code for a Graphviz tree.

        Parameters
        ----------
        round_score: int, optional
            The number of  decimals to round the  node's
            score to. Zero rounds to an integer.
        percent: bool, optional
            If set to True, the scores are  returned  as
            a percent.
        vertical: bool, optional
            If set to True,  the  function  generates  a
            vertical tree.
        node_style: dict, optional
            Dictionary of options to customize each  node
            of the tree.
            For a list of options,  see the Graphviz API:
            https://graphviz.org/doc/info/attrs.html
        arrow_style: dict, optional
            Dictionary of options to customize each arrow
            of the tree.
            For  a list of options, see the Graphviz API:
            https://graphviz.org/doc/info/attrs.html
        leaf_style: dict, optional
            Dictionary of options to customize each  leaf
            of the tree.
            For a list of options,  see the Graphviz API:
            https://graphviz.org/doc/info/attrs.html

        Returns
        -------
        str
            Graphviz code.
        """
        return self.to_memmodel().to_graphviz(
            round_score=round_score,
            percent=percent,
            vertical=vertical,
            node_style=node_style,
            arrow_style=arrow_style,
            leaf_style=leaf_style,
        )

    def plot_tree(
        self,
        pic_path: Optional[str] = None,
        *args,
        **kwargs,
    ) -> "Source":
        """
        Draws the input tree. Requires the graphviz module.

        Parameters
        ----------
        pic_path: str, optional
            Absolute  path to save the image of the  tree.
        *args, **kwargs: Any, optional
            Arguments to pass to the 'to_graphviz' method.

        Returns
        -------
        graphviz.Source
            graphviz object.
        """
        return self.to_memmodel().plot_tree(
            pic_path=pic_path,
            *args,
            **kwargs,
        )


"""
Algorithms used for clustering.
"""


class DBSCAN(VerticaModel):
    """
    [Beta Version]
    Creates a DBSCAN object by  using the DBSCAN algorithm
    as defined by Martin  Ester,  Hans-Peter Kriegel, Jörg
    Sander, and Xiaowei Xu.  This  object uses pure SQL to
    compute the distances and neighbors, and uses Python to
    compute the cluster propagation (non-scalable phase).

    \u26A0 Warning : This   algorithm   uses  a   CROSS  JOIN
                     during   computation  and  is  therefore
                     computationally  expensive at  O(n * n),
                     where n is the total number of elements.
                     This  algorithm indexes elements of  the
                     table in order to be optimal  (the CROSS
                     JOIN will happen only with IDs which are
                     integers).
                     Since  DBSCAN uses  the  p-distance, it
                     is highly sensitive  to unnormalized data.
                     However,  DBSCAN is robust to outliers and
                     can find non-linear clusters. It is a very
                     powerful algorithm for outlier  detection
                     and clustering. A table is created at
                     the end of the learning phase.

    Parameters
    ----------
    name: str
        Name  of the model.  This is  not a  built-in
        model, so this name is  used to build the
        final table.
    eps: float, optional
        The radius of a  neighborhood with respect to
        some point.
    min_samples: int, optional
        Minimum number  of points required to  form a
        dense region.
    p: int, optional
        The p of the p-distance (distance metric used
        during the model computation).
    """

    # Properties.

    @property
    def _is_native(self) -> Literal[False]:
        return False

    @property
    def _vertica_fit_sql(self) -> Literal[""]:
        return ""

    @property
    def _vertica_predict_sql(self) -> Literal[""]:
        return ""

    @property
    def _model_category(self) -> Literal["UNSUPERVISED"]:
        return "UNSUPERVISED"

    @property
    def _model_subcategory(self) -> Literal["CLUSTERING"]:
        return "CLUSTERING"

    @property
    def _model_type(self) -> Literal["DBSCAN"]:
        return "DBSCAN"

    @property
    def _attributes(self) -> list[str]:
        return ["n_cluster_", "n_noise_", "p_"]

    # System & Special Methods.

    @save_verticapy_logs
    def __init__(
        self, name: str, eps: float = 0.5, min_samples: int = 5, p: int = 2
    ) -> None:
        super().__init__()
        self.model_name = name
        self.parameters = {"eps": eps, "min_samples": min_samples, "p": p}

    def drop(self) -> bool:
        """
        Drops the model from the Vertica database.
        """
        try:
            _executeSQL(
                query=f"SELECT dbscan_cluster FROM {self.model_name} LIMIT 0;",
                title="Looking if the DBSCAN table exists.",
            )
            return drop(self.model_name, method="table")
        except QueryError:
            return False

    # Model Fitting Method.

    def fit(
        self,
        input_relation: SQLRelation,
        X: Optional[SQLColumns] = None,
        key_columns: Optional[SQLColumns] = None,
        index: Optional[str] = None,
    ) -> None:
        """
        Trains the model.

        Parameters
        ----------
        input_relation: SQLRelation
            Training relation.
        X: SQLColumns, optional
            List of the predictors. If empty, all the
            numerical vDataColumns are used.
        key_columns: SQLColumns, optional
            Columns  not  used  during  the  algorithm
            computation  but  which  are  used  to
            create the final relation.
        index: str, optional
            Index  used to identify each row  separately.
            It is highly  recommanded to have one already
            in the main table to avoid creating temporary
            tables.
        """
        if conf.get_option("overwrite_model"):
            self.drop()
        else:
            self._is_already_stored(raise_error=True)
        if isinstance(input_relation, vDataFrame):
            if isinstance(X, NoneType):
                X = input_relation.numcol()
            input_relation = input_relation.current_relation()
        else:
            if isinstance(X, NoneType):
                X = vDataFrame(input_relation).numcol()
        X, key_columns = format_type(X, key_columns, dtype=list)
        X = quote_ident(X)
        self.X = X
        self.key_columns = quote_ident(key_columns)
        self.input_relation = input_relation
        name_main = gen_tmp_name(name="main")
        name_dbscan_clusters = gen_tmp_name(name="clusters")
        try:
            if not index:
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
            if isinstance(conf.get_option("random_state"), int):
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
                if isinstance(clusters[node], NoneType) and isinstance(
                    clusters[node_neighbor], NoneType
                ):
                    clusters[node] = i
                    clusters[node_neighbor] = i
                    i = i + 1
                else:
                    if not isinstance(clusters[node], NoneType) and isinstance(
                        clusters[node_neighbor], NoneType
                    ):
                        clusters[node_neighbor] = clusters[node]
                    elif not (
                        isinstance(clusters[node_neighbor], NoneType)
                    ) and isinstance(clusters[node], NoneType):
                        clusters[node] = clusters[node_neighbor]
                del graph[0]
            try:
                with open(f"{name_dbscan_clusters}.csv", "w", encoding="utf-8") as f:
                    for c in clusters:
                        f.write(f"{c}, {clusters[c]}\n")
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
                    CREATE TABLE {self.model_name} AS 
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
                    FROM {self.model_name} 
                    WHERE dbscan_cluster = -1""",
                method="fetchfirstelem",
                print_time_sql=False,
            )
            self.p_ = self.parameters["p"]
        finally:
            drop(f"v_temp_schema.{name_main}", method="table")
            drop(f"v_temp_schema.{name_dbscan_clusters}", method="table")

    # Prediction / Transformation Methods.

    def predict(self) -> vDataFrame:
        """
        Creates a vDataFrame of the model.

        Returns
        -------
        vDataFrame
            the vDataFrame including the prediction.
        """
        return vDataFrame(self.model_name)

    # Plotting Methods.

    def plot(
        self,
        max_nb_points: int = 100,
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the model.

        Parameters
        ----------
        max_nb_points: int
            Maximum number of points to display.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any optional parameter to pass to the
            Plotting functions.

        Returns
        -------
        obj
            Plotting Object.
        """
        return vDataFrame(self.model_name).scatter(
            columns=self.X,
            by="dbscan_cluster",
            max_cardinality=100,
            max_nb_points=max_nb_points,
            chart=chart,
            **style_kwargs,
        )


"""
Algorithms used for classification.
"""


class NearestCentroid(MulticlassClassifier):
    """
    Creates  a NearestCentroid object using the  k-nearest
    centroid algorithm.
    This object uses pure SQL to compute the distances and
    final score.

    Parameters
    ----------
    p: int, optional
        The p corresponding to the one of the p-distances
        (distance metric used to compute the model).
    """

    # Properties.

    @property
    def _is_native(self) -> Literal[False]:
        return False

    @property
    def _vertica_fit_sql(self) -> Literal[""]:
        return ""

    @property
    def _vertica_predict_sql(self) -> Literal[""]:
        return ""

    @property
    def _model_category(self) -> Literal["SUPERVISED"]:
        return "SUPERVISED"

    @property
    def _model_subcategory(self) -> Literal["CLASSIFIER"]:
        return "CLASSIFIER"

    @property
    def _model_type(self) -> Literal["NearestCentroid"]:
        return "NearestCentroid"

    @property
    def _attributes(self) -> list[str]:
        return ["clusters_", "classes_", "p_"]

    # System & Special Methods.

    @save_verticapy_logs
    def __init__(self, name: str, p: int = 2) -> None:
        super().__init__()
        self.model_name = name
        self.parameters = {"p": p}

    def drop(self) -> bool:
        """
        NearestCentroid models are not stored in the Vertica DB.
        """
        return False

    # Attributes Methods.

    def _compute_attributes(self) -> None:
        """
        Computes the model's attributes.
        """
        func = "APPROXIMATE_MEDIAN" if (self.parameters["p"] == 1) else "AVG"
        X_str = ", ".join([f"{func}({column}) AS {column}" for column in self.X])
        centroids = TableSample.read_sql(
            query=f"""
            SELECT 
                {X_str}, 
                {self.y} 
            FROM {self.input_relation} 
            WHERE {self.y} IS NOT NULL 
            GROUP BY {self.y} 
            ORDER BY {self.y} ASC""",
            title="Getting Model Centroids.",
        )
        self.clusters_ = centroids.to_numpy()[:, 0:-1]
        self.classes_ = self._array_to_int(centroids.to_numpy()[:, -1])
        self.p_ = self.parameters["p"]

    # Prediction / Transformation Methods.

    def _get_y_proba(
        self,
        pos_label: Optional[PythonScalar] = None,
    ) -> str:
        """
        Returns the input that represents the model's
        probabilities.
        """
        idx = self.get_match_index(pos_label, self.classes_, False)
        return self.deploySQL(allSQL=True)[idx]

    # I/O Methods.

    def to_memmodel(self) -> mm.NearestCentroid:
        """
        Converts  the  model to  an InMemory object that
        can be used for different types of predictions.
        """
        return mm.NearestCentroid(
            self.clusters_,
            self.classes_,
            self.p_,
        )
