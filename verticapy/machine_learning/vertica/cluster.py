"""
Copyright  (c)  2018-2024 Open Text  or  one  of its
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
    def __init__(self, name: str, overwrite_model: bool = False) -> None:
        """Must be overridden in the child class"""
        super().__init__(name, overwrite_model)

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
            fun = vdf.scatter
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
    name: str, optional
        Name  of  the model. The model is stored in the
        database.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
    n_cluster: int, optional
        Number of clusters
    init: str / list, optional
        The  method  used to find the initial  cluster
        centers.

        - kmeanspp:
            Uses   the  KMeans++  method   to initialize
            the centers.
        - random:
            The   centers   are   initialized randomly.

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

    Attributes
    ----------
    Many attributes are created during the fitting phase.

    clusters_: numpy.array
        Cluster centers.
    p_: int
        The p of the p-distances.
    between_cluster_ss_: float
        The between-cluster sum of squares (BSS) measures
        the dispersion between different clusters and is
        an important metric in evaluating the effectiveness
        of a clustering algorithm.
    total_ss_: float
        The total sum of squares (TSS) is used to assess
        the total dispersion of data points from the overall
        mean, providing a basis for evaluating the clustering
        algorithm's performance.
    total_within_cluster_ss_: float
        The within-cluster sum of squares (WSS) gauges the
        dispersion of data points within individual clusters
        in a clustering analysis. It reflects the compactness
        of clusters and is instrumental in evaluating the
        homogeneity of the clusters produced by the algorithm.
    elbow_score_: float
        The elbow score. It helps identify the optimal number
        of clusters by observing the point where the rate of
        WSS reduction slows down, resembling the bend or
        'elbow' in the plot, indicative of an optimal clustering
        solution. The bigger the better.
    converged_: boolean
        True if the model converged.

    .. note::

        All attributes can be accessed using the
        :py:meth:`verticapy.machine_learning.vertica.cluster.Clustering.get_attributes``
        method.

    .. note::

        Several other attributes can be accessed by using the
        :py:meth:`verticapy.machine_learning.vertica.cluster.Clustering.get_vertica_attributes``
        method.

    Examples
    ---------

    The following examples provide a basic understanding of usage.
    For more detailed examples, please refer to the
    :ref:`user_guide.machine_learning` or the
    `Examples <https://www.vertica.com/python/examples/>`_
    section on the website.

    Load data for machine learning
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    We import :py:mod:`verticapy`:

    .. ipython:: python

        import verticapy as vp

    .. hint::

        By assigning an alias to :py:mod:`verticapy`, we mitigate the risk of code
        collisions with other libraries. This precaution is necessary
        because verticapy uses commonly known function names like "average"
        and "median", which can potentially lead to naming conflicts.
        The use of an alias ensures that the functions from verticapy are
        used as intended without interfering with functions from other
        libraries.

    For this example, we will use the winequality dataset.

    .. code-block:: python

        import verticapy.datasets as vpd

        data = vpd.load_winequality()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_winequality.html

    .. note::

        VerticaPy offers a wide range of sample datasets that are
        ideal for training and testing purposes. You can explore
        the full list of available datasets in the :ref:`api.datasets`,
        which provides detailed information on each dataset
        and how to use them effectively. These datasets are invaluable
        resources for honing your data analysis and machine learning
        skills within the VerticaPy environment.

    .. ipython:: python
        :suppress:

        import verticapy.datasets as vpd
        data = vpd.load_winequality()

    Model Initialization
    ^^^^^^^^^^^^^^^^^^^^^

    First we import the ``KMeans`` model:

    .. code-block::

        from verticapy.machine_learning.vertica import KMeans

    .. ipython:: python
        :suppress:

        from verticapy.machine_learning.vertica import KMeans

    Then we can create the model:

    .. ipython:: python
        :okwarning:

        model = KMeans(
               n_cluster = 8,
               init = "kmeanspp",
               max_iter = 300,
               tol = 1e-4,
        )

    .. hint::

        In :py:mod:`verticapy` 1.0.x and higher, you do not need to specify the
        model name, as the name is automatically assigned. If you need to
        re-use the model, you can fetch the model name from the model's
        attributes.

    .. important::

        The model name is crucial for the model management system and
        versioning. It's highly recommended to provide a name if you
        plan to reuse the model later.

    Model Training
    ^^^^^^^^^^^^^^^

    We can now fit the model:

    .. ipython:: python
        :okwarning:

        model.fit(data, X = ["density", "sulphates"])

    .. important::

        To train a model, you can directly use the :py:class:`vDataFrame` or the
        name of the relation stored in the database. The test set is optional
        and is only used to compute the test metrics. In :py:mod:`verticapy`, we
        don't work using ``X`` matrices and ``y`` vectors. Instead, we work
        directly with lists of predictors and the response name.

    .. hint::

        For clustering and anomaly detection, the use of predictors is
        optional. In such cases, all available predictors are considered,
        which can include solely numerical variables or a combination of
        numerical and categorical variables, depending on the model's
        capabilities.

    Metrics
    ^^^^^^^^

    You can also find the cluster positions by:

    .. ipython:: python

        model.clusters_

    To evaluate the model, various attributes are computed, such as
    the between sum of squares, the total within clusters sum of
    squares, and the total sum of squares.

    .. ipython:: python

        model.between_cluster_ss_
        model.total_within_cluster_ss_
        model.total_ss_

    Some other useful attributes can be used to evaluate the model,
    like the Elbow Score (the bigger it is, the better it is).

    .. ipython:: python

        model.elbow_score_

    Prediction
    ^^^^^^^^^^^

    Predicting or ranking the dataset is straight-forward:

    .. ipython:: python
        :suppress:

        result = model.predict(data, ["density", "sulphates"], name = "Cluster IDs")
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_kmeans_prediction.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.predict(data, ["density", "sulphates"], name = "Cluster IDs")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_kmeans_prediction.html

    As shown above, a new column has been created, containing
    the clusters.

    .. hint::
        The name of the new column is optional. If not provided,
        it is randomly assigned.

    Plots - Cluster Plot
    ^^^^^^^^^^^^^^^^^^^^^

    Plots highlighting the different clusters can be easily drawn using:

    .. code-block:: python

        model.plot()

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "plotly")
        fig = model.plot(width = 600)
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_kmeans_plot.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_kmeans_plot.html

    Plots - Voronoi
    ^^^^^^^^^^^^^^^^

    ``KMeans`` models can be visualized by drawing their voronoi plots.
    For more examples, check out :ref:`chart_gallery.voronoi_plot`.

    .. code-block:: python

        model.plot_voronoi()

    .. ipython:: python
        :suppress:

        fig = model.plot_voronoi(width = 600)
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_kmeans_plot_voronoi.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_kmeans_plot_voronoi.html

    Plots - Contour
    ^^^^^^^^^^^^^^^^

    In order to understand the parameter space, we can also look
    at the contour plots:

    .. code-block:: python

        model.contour()

    .. ipython:: python
        :suppress:

        fig = model.contour(width = 600)
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_kmeans_contour.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_kmeans_contour.html

    .. note::

        Machine learning models with two predictors can usually benefit
        from their own contour plot. This visual representation aids in
        exploring predictions and gaining a deeper understanding of how
        these models perform in different scenarios. Please refer to
        :ref:`chart_gallery.contour_plot` for more examples.

    Parameter Modification
    ^^^^^^^^^^^^^^^^^^^^^^^

    In order to see the parameters:

    .. ipython:: python

        model.get_params()

    And to manually change some of the parameters:

    .. ipython:: python

        model.set_params({'n_cluster': 5})

    Model Register
    ^^^^^^^^^^^^^^

    In order to register the model for tracking and versioning:

    .. code-block:: python

        model.register("model_v1")

    Please refer to :ref:`notebooks/ml/model_tracking_versioning/index.html`
    for more details on model tracking and versioning.

    Model Exporting
    ^^^^^^^^^^^^^^^^

    **To Memmodel**

    .. code-block:: python

        model.to_memmodel()

    .. note::

        ``MemModel`` objects serve as in-memory representations of machine
        learning models. They can be used for both in-database and in-memory
        prediction tasks. These objects can be pickled in the same way that
        you would pickle a ``scikit-learn`` model.

    The preceding methods for exporting the model use ``MemModel``, and it
    is recommended to use ``MemModel`` directly.

    **To SQL**

    You can get the SQL query equivalent of the ``KMeans`` model by:

    .. ipython:: python

        model.to_sql()

    .. note:: This SQL query can be directly used in any database.

    **Deploy SQL**

    To get the SQL query which uses Vertica functions use below:

    .. ipython:: python

        model.deploySQL()

    **To Python**

    To obtain the prediction function in Python syntax, use the following code:

    .. ipython:: python

        X = [[0.9, 0.5]]
        model.to_python()(X)

    .. hint::

        The
        :py:meth:`verticapy.machine_learning.vertica.tree.KMeans.to_python`
        method is used to retrieve the anomaly score.
        For specific details on how to
        use this method for different model types, refer to the relevant
        documentation for each model.
    """

    # Properties.

    @property
    def _vertica_fit_sql(self) -> Literal["KMEANS"]:
        return "KMEANS"

    @property
    def _vertica_predict_sql(self) -> Literal["APPLY_KMEANS"]:
        return "APPLY_KMEANS"

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
        name: str = None,
        overwrite_model: bool = False,
        n_cluster: int = 8,
        init: Union[Literal["kmeanspp", "random"], list] = "kmeanspp",
        max_iter: int = 300,
        tol: float = 1e-4,
    ) -> None:
        super().__init__(name, overwrite_model)
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
    name: str, optional
        Name  of the model. The model is stored in  the
        database.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
    n_cluster: int, optional
        Number of clusters.
    init: str / list, optional
        The  method  used  to  find the  initial  cluster
        centers.

        - random:
            The centers are initialized randomly.

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

    Attributes
    ----------
    Many attributes are created during the fitting phase.

    clusters_: numpy.array
        Cluster centers.
    p_: int
        The p of the p-distances.
    between_cluster_ss_: float
        The between-cluster sum of squares (BSS) measures
        the dispersion between different clusters and is
        an important metric in evaluating the effectiveness
        of a clustering algorithm.
    total_ss_: float
        The total sum of squares (TSS) is used to assess
        the total dispersion of data points from the overall
        mean, providing a basis for evaluating the clustering
        algorithm's performance.
    total_within_cluster_ss_: float
        The within-cluster sum of squares (WSS) gauges the
        dispersion of data points within individual clusters
        in a clustering analysis. It reflects the compactness
        of clusters and is instrumental in evaluating the
        homogeneity of the clusters produced by the algorithm.
    elbow_score_: float
        The elbow score. It helps identify the optimal number
        of clusters by observing the point where the rate of
        WSS reduction slows down, resembling the bend or
        'elbow' in the plot, indicative of an optimal clustering
        solution. The bigger the better.
    converged_: boolean
        True if the model converged.

    .. note::

        All attributes can be accessed using the
        :py:meth:`verticapy.machine_learning.vertica.cluster.Clustering.get_attributes``
        method.

    .. note::

        Several other attributes can be accessed by using the
        :py:meth:`verticapy.machine_learning.vertica.cluster.Clustering.get_vertica_attributes``
        method.

    Examples
    ---------

    The following examples provide a basic understanding of usage.
    For more detailed examples, please refer to the
    :ref:`user_guide.machine_learning` or the
    `Examples <https://www.vertica.com/python/examples/>`_
    section on the website.

    Load data for machine learning
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    We import :py:mod:`verticapy`:

    .. ipython:: python

        import verticapy as vp

    .. hint::

        By assigning an alias to :py:mod:`verticapy`, we mitigate the risk of code
        collisions with other libraries. This precaution is necessary
        because verticapy uses commonly known function names like "average"
        and "median", which can potentially lead to naming conflicts.
        The use of an alias ensures that the functions from verticapy are
        used as intended without interfering with functions from other
        libraries.

    For this example, we will use the winequality dataset.

    .. code-block:: python

        import verticapy.datasets as vpd

        data = vpd.load_winequality()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_winequality.html

    .. note::

        VerticaPy offers a wide range of sample datasets that are
        ideal for training and testing purposes. You can explore
        the full list of available datasets in the :ref:`api.datasets`,
        which provides detailed information on each dataset
        and how to use them effectively. These datasets are invaluable
        resources for honing your data analysis and machine learning
        skills within the VerticaPy environment.

    .. ipython:: python
        :suppress:

        import verticapy.datasets as vpd
        data = vpd.load_winequality()

    Model Initialization
    ^^^^^^^^^^^^^^^^^^^^^

    First we import the ``KPrototypes`` model:

    .. code-block::

        from verticapy.machine_learning.vertica import KPrototypes

    .. ipython:: python
        :suppress:

        from verticapy.machine_learning.vertica import KPrototypes

    Then we can create the model:

    .. ipython:: python
        :okwarning:

        model = KPrototypes(
            n_cluster = 8,
            init = "random",
            max_iter = 300,
            tol = 1e-4,
            gamma = 0.2,
        )

    .. hint::

        In :py:mod:`verticapy` 1.0.x and higher, you do not need to specify the
        model name, as the name is automatically assigned. If you need to
        re-use the model, you can fetch the model name from the model's
        attributes.

    .. important::

        The model name is crucial for the model management system and
        versioning. It's highly recommended to provide a name if you
        plan to reuse the model later.

    Model Training
    ^^^^^^^^^^^^^^^

    We can now fit the model:

    .. ipython:: python
        :okwarning:

        model.fit(data, X = ["color", "sulphates"])

    .. important::

        To train a model, you can directly use the :py:class:`vDataFrame` or the
        name of the relation stored in the database.

    .. note::

        In the above example we have use one categorical ("color")
        and one numeric ("sulphates") feature. ``KProtoTypes``
        can handle both types of features.

    .. hint::

        For clustering and anomaly detection, the use of predictors is
        optional. In such cases, all available predictors are considered,
        which can include solely numerical variables or a combination of
        numerical and categorical variables, depending on the model's
        capabilities.

    Metrics
    ^^^^^^^^

    You can also find the cluster positions by:

    .. ipython:: python

        model.clusters_

    To evaluate the model, various attributes are computed, such as
    the between sum of squares, the total within clusters sum of
    squares, and the total sum of squares.

    .. ipython:: python

        model.between_cluster_ss_
        model.total_within_cluster_ss_
        model.total_ss_

    Some other useful attributes can be used to evaluate the model,
    like the Elbow Score (the bigger it is, the better it is).

    .. ipython:: python

        model.elbow_score_

    Prediction
    ^^^^^^^^^^^

    Predicting or ranking the dataset is straight-forward:

    .. ipython:: python
        :suppress:

        result = model.predict(data, ["density", "sulphates"], name = "Cluster IDs")
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_kprototypes_prediction.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.predict(data, ["density", "sulphates"], name = "Cluster IDs")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_kprototypes_prediction.html

    As shown above, a new column has been created, containing
    the clusters.

    .. hint::
        The name of the new column is optional. If not provided,
        it is randomly assigned.

    Plots - Cluster Plot
    ^^^^^^^^^^^^^^^^^^^^^

    Plots highlighting the different clusters can be easily drawn using:

    .. code-block:: python

        model.plot()

    .. note::

        As we are using a categorical feature in this example,
        this plot cannot be drawn. It is only for numeric features.
        Please have a look at :py:meth:`verticapy.machine_learning.vertica.KMeans`
        for plotting examples.

    Plots - Voronoi
    ^^^^^^^^^^^^^^^^

    ``KPrototypes`` models can be visualized by drawing their voronoi plots.
    For more examples, check out :ref:`chart_gallery.voronoi_plot`.

    .. code-block:: python

        model.plot_voronoi()

    .. note::

        As we are using a categorical feature in this example,
        this plot cannot be drawn. It is only for numeric features.
        Please have a look at :py:meth:`verticapy.machine_learning.vertica.KMeans`
        for plotting examples.


    Plots - Contour
    ^^^^^^^^^^^^^^^^

    In order to understand the parameter space, we can also look
    at the contour plots:

    .. code-block:: python

        model.contour()

    .. note::

        As we are using a categorical feature in this example,
        this plot cannot be drawn. It is only for numeric features.
        Please have a look at :py:meth:`verticapy.machine_learning.vertica.KMeans`
        for plotting examples.

    .. note::

        Machine learning models with two predictors can usually benefit
        from their own contour plot. This visual representation aids in
        exploring predictions and gaining a deeper understanding of how
        these models perform in different scenarios. Please refer to
        :ref:`chart_gallery.contour_plot` for more examples.

    Parameter Modification
    ^^^^^^^^^^^^^^^^^^^^^^^

    In order to see the parameters:

    .. ipython:: python

        model.get_params()

    And to manually change some of the parameters:

    .. ipython:: python

        model.set_params({'n_cluster': 5})

    Model Register
    ^^^^^^^^^^^^^^

    In order to register the model for tracking and versioning:

    .. code-block:: python

        model.register("model_v1")

    Please refer to :ref:`notebooks/ml/model_tracking_versioning/index.html`
    for more details on model tracking and versioning.

    Model Exporting
    ^^^^^^^^^^^^^^^^

    **To Memmodel**

    .. code-block:: python

        model.to_memmodel()

    .. note::

        ``MemModel`` objects serve as in-memory representations of machine
        learning models. They can be used for both in-database and in-memory
        prediction tasks. These objects can be pickled in the same way that
        you would pickle a ``scikit-learn`` model.

    The preceding methods for exporting the model use ``MemModel``, and it
    is recommended to use ``MemModel`` directly.

    **To SQL**

    You can get the SQL query equivalent of the ``KPrototypes`` model by:

    .. ipython:: python

        model.to_sql()

    .. note:: This SQL query can be directly used in any database.

    **Deploy SQL**

    To get the SQL query which uses Vertica functions use below:

    .. ipython:: python

        model.deploySQL()

    **To Python**

    To obtain the prediction function in Python syntax, use the following code:

    .. ipython:: python

        X = [[0.9, 0.5]]
        model.to_python()(X)

    .. hint::

        The
        :py:meth:`verticapy.machine_learning.vertica.tree.KPrototypes.to_python`
        method is used to retrieve the anomaly score.
        For specific details on how to
        use this method for different model types, refer to the relevant
        documentation for each model.
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
        name: str = None,
        overwrite_model: bool = False,
        n_cluster: int = 8,
        init: Union[Literal["random"], list] = "random",
        max_iter: int = 300,
        tol: float = 1e-4,
        gamma: float = 1.0,
    ) -> None:
        super().__init__(name, overwrite_model)
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
    name: str, optional
        Name of the model.  The  model is stored in
        the database.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
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

        - size:
            Choose the largest cluster to bisect.
        - sum_squares:
            Choose the cluster with the largest
            withInSS to bisect.

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

        - kmeanspp:
            Uses  the KMeans++ method  to initialize
            the centers.
        - pseudo:
            Uses "pseudo center" approach used by
            Spark,  bisects given center without iterating
            over points.

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

    Attributes
    ----------
    Many attributes are created during the fitting phase.

            "children_left_",
            "children_right_",


    tree_:


    clusters_: numpy.array
        Cluster centers.
    p_: int
        The p of the p-distances.
    children_left_: numpy.array
        A list  of node IDs, where  children_left[i] is
        the node ID of the left child of node i.
    children_right_: numpy.array
        A list of node IDs, where  children_right[i] is
        the node ID of the right child of node i.
    cluster_score_: numpy.array
        The array containing the sizes for each cluster
        in a clustering analysis.
    cluster_score_: numpy.array
        The array containing the cluster scores for each
        cluster in a clustering analysis.
    between_cluster_ss_: float
        The between-cluster sum of squares (BSS) measures
        the dispersion between different clusters and is
        an important metric in evaluating the effectiveness
        of a clustering algorithm.
    total_ss_: float
        The total sum of squares (TSS) is used to assess
        the total dispersion of data points from the overall
        mean, providing a basis for evaluating the clustering
        algorithm's performance.
    total_within_cluster_ss_: float
        The within-cluster sum of squares (WSS) gauges the
        dispersion of data points within individual clusters
        in a clustering analysis. It reflects the compactness
        of clusters and is instrumental in evaluating the
        homogeneity of the clusters produced by the algorithm.
    elbow_score_: float
        The elbow score. It helps identify the optimal number
        of clusters by observing the point where the rate of
        WSS reduction slows down, resembling the bend or
        'elbow' in the plot, indicative of an optimal clustering
        solution. The bigger the better.
    cluster_i_ss_: numpy.array
        The array containing the sum of squares (SS) for each
        cluster in a clustering analysis.

    .. note::

        All attributes can be accessed using the
        :py:meth:`verticapy.machine_learning.vertica.cluster.Clustering.get_attributes``
        method.

    .. note::

        Several other attributes can be accessed by using the
        :py:meth:`verticapy.machine_learning.vertica.cluster.Clustering.get_vertica_attributes``
        method.

    Examples
    ---------

    The following examples provide a basic understanding of usage.
    For more detailed examples, please refer to the
    :ref:`user_guide.machine_learning` or the
    `Examples <https://www.vertica.com/python/examples/>`_
    section on the website.

    Load data for machine learning
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    We import :py:mod:`verticapy`:

    .. ipython:: python

        import verticapy as vp

    .. hint::

        By assigning an alias to :py:mod:`verticapy`, we mitigate the risk of code
        collisions with other libraries. This precaution is necessary
        because verticapy uses commonly known function names like "average"
        and "median", which can potentially lead to naming conflicts.
        The use of an alias ensures that the functions from verticapy are
        used as intended without interfering with functions from other
        libraries.

    For this example, we will use the winequality dataset.

    .. code-block:: python

        import verticapy.datasets as vpd

        data = vpd.load_winequality()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_winequality.html

    .. note::

        VerticaPy offers a wide range of sample datasets that are
        ideal for training and testing purposes. You can explore
        the full list of available datasets in the :ref:`api.datasets`,
        which provides detailed information on each dataset
        and how to use them effectively. These datasets are invaluable
        resources for honing your data analysis and machine learning
        skills within the VerticaPy environment.

    .. ipython:: python
        :suppress:

        import verticapy.datasets as vpd
        data = vpd.load_winequality()

    Model Initialization
    ^^^^^^^^^^^^^^^^^^^^^

    First we import the ``BisectingKMeans`` model:

    .. code-block::

        from verticapy.machine_learning.vertica import BisectingKMeans

    .. ipython:: python
        :suppress:

        from verticapy.machine_learning.vertica import BisectingKMeans

    Then we can create the model:

    .. ipython:: python
        :okwarning:

        model = BisectingKMeans(
            n_cluster = 8,
            bisection_iterations = 1,
            split_method = 'sum_squares',
            min_divisible_cluster_size = 2,
            distance_method = "euclidean",
            init = "kmeanspp",
            max_iter = 300,
            tol = 1e-4,
        )

    .. hint::

        In :py:mod:`verticapy` 1.0.x and higher, you do not need to specify the
        model name, as the name is automatically assigned. If you need to
        re-use the model, you can fetch the model name from the model's
        attributes.

    .. important::

        The model name is crucial for the model management system and
        versioning. It's highly recommended to provide a name if you
        plan to reuse the model later.

    Model Training
    ^^^^^^^^^^^^^^^

    We can now fit the model:

    .. ipython:: python
        :okwarning:

        model.fit(data, X = ["density", "sulphates"])

    .. important::

        To train a model, you can directly use the :py:class:`vDataFrame` or the
        name of the relation stored in the database. The test set is optional
        and is only used to compute the test metrics. In :py:mod:`verticapy`, we
        don't work using ``X`` matrices and ``y`` vectors. Instead, we work
        directly with lists of predictors and the response name.

    .. hint::

        For clustering and anomaly detection, the use of predictors is
        optional. In such cases, all available predictors are considered,
        which can include solely numerical variables or a combination of
        numerical and categorical variables, depending on the model's
        capabilities.

    Metrics
    ^^^^^^^^

    You can also find the cluster positions by:

    .. ipython:: python

        model.clusters_

    In order to get the size of each cluster, you can use:

    .. ipython:: python

        model.cluster_size_

    To evaluate the model, various attributes are computed, such as
    the between sum of squares, the total within clusters sum of
    squares, and the total sum of squares.

    .. ipython:: python

        model.between_cluster_ss_
        model.total_within_cluster_ss_
        model.total_ss_

    You also have access to the sum of squares of each cluster.

    .. ipython:: python

        model.cluster_i_ss_

    Some other useful attributes can be used to evaluate the model,
    like the Elbow Score (the bigger it is, the better it is).

    .. ipython:: python

        model.elbow_score_

    Prediction
    ^^^^^^^^^^^

    Predicting or ranking the dataset is straight-forward:

    .. ipython:: python
        :suppress:

        result = model.predict(data, ["density", "sulphates"])
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_bisect_km_prediction.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.predict(data, ["density", "sulphates"])

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_bisect_km_prediction.html

    As shown above, a new column has been created, containing
    the bisected clusters.

    Plots - Cluster Plot
    ^^^^^^^^^^^^^^^^^^^^^

    Plots highlighting the different clusters can be easily drawn using:

    .. code-block:: python

        model.plot()

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "plotly")
        fig = model.plot(width = 600)
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_bisect_km_plot.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_bisect_km_plot.html

    Plots - Tree
    ^^^^^^^^^^^^^

    Tree models can be visualized by drawing their tree plots.
    For more examples, check out :ref:`chart_gallery.tree`.

    .. code-block:: python

        model.plot_tree()

    .. ipython:: python
        :suppress:

        res = model.plot_tree()
        res.render(filename='figures/machine_learning_vertica_tree_bisect_km_', format='png')

    .. image:: /../figures/machine_learning_vertica_tree_bisect_km_.png

    .. note::

        The above example may not render properly in the doc because
        of the huge size of the tree. But it should render nicely
        in jupyter environment.

    In order to plot graph using `graphviz <https://graphviz.org/>`_
    separately, you can extract the graphviz DOT file code as follows:

    .. ipython:: python

        model.to_graphviz()

    This string can then be copied into a DOT file which can be
    parsed by graphviz.

    Plots - Contour
    ^^^^^^^^^^^^^^^^

    In order to understand the parameter space, we can also look
    at the contour plots:

    .. code-block:: python

        model.contour()

    .. ipython:: python
        :suppress:

        fig = model.contour(width = 600)
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_bisect_km_contour.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_bisect_km_contour.html

    .. note::

        Machine learning models with two predictors can usually benefit
        from their own contour plot. This visual representation aids in
        exploring predictions and gaining a deeper understanding of how
        these models perform in different scenarios. Please refer to
        :ref:`chart_gallery.contour_plot` for more examples.

    Parameter Modification
    ^^^^^^^^^^^^^^^^^^^^^^^

    In order to see the parameters:

    .. ipython:: python

        model.get_params()

    And to manually change some of the parameters:

    .. ipython:: python

        model.set_params({'n_cluster': 5})

    Model Register
    ^^^^^^^^^^^^^^

    In order to register the model for tracking and versioning:

    .. code-block:: python

        model.register("model_v1")

    Please refer to :ref:`notebooks/ml/model_tracking_versioning/index.html`
    for more details on model tracking and versioning.

    Model Exporting
    ^^^^^^^^^^^^^^^^

    **To Memmodel**

    .. code-block:: python

        model.to_memmodel()

    .. note::

        ``MemModel`` objects serve as in-memory representations of machine
        learning models. They can be used for both in-database and in-memory
        prediction tasks. These objects can be pickled in the same way that
        you would pickle a ``scikit-learn`` model.

    The preceding methods for exporting the model use ``MemModel``, and it
    is recommended to use ``MemModel`` directly.

    **To SQL**

    You can get the SQL query equivalent of the XGB model by:

    .. ipython:: python

        model.to_sql()

    .. note:: This SQL query can be directly used in any database.

    **Deploy SQL**

    To get the SQL query which uses Vertica functions use below:

    .. ipython:: python

        model.deploySQL()

    **To Python**

    To obtain the prediction function in Python syntax, use the following code:

    .. ipython:: python

        X = [[0.9, 0.5]]
        model.to_python()(X)

    .. hint::

        The
        :py:meth:`verticapy.machine_learning.vertica.tree.BisectingKMeans.to_python`
        method is used to retrieve the anomaly score.
        For specific details on how to
        use this method for different model types, refer to the relevant
        documentation for each model.
    """

    # Properties.

    @property
    def _vertica_fit_sql(self) -> Literal["BISECTING_KMEANS"]:
        return "BISECTING_KMEANS"

    @property
    def _vertica_predict_sql(self) -> Literal["APPLY_BISECTING_KMEANS"]:
        return "APPLY_BISECTING_KMEANS"

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
        name: str = None,
        overwrite_model: bool = False,
        n_cluster: int = 8,
        bisection_iterations: int = 1,
        split_method: Literal["size", "sum_squares"] = "sum_squares",
        min_divisible_cluster_size: int = 2,
        distance_method: Literal["euclidean"] = "euclidean",
        init: Union[Literal["kmeanspp", "pseudo", "random"], list] = "kmeanspp",
        max_iter: int = 300,
        tol: float = 1e-4,
    ) -> None:
        super().__init__(name, overwrite_model)
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
        self.clusters_ = centers.to_numpy()[:, 1 : len(self.X) + 1].astype(float)
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

    .. warning :

        This   algorithm   uses  a   CROSS  JOIN
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

    .. important::

        This algorithm is not Vertica Native and relies solely
        on SQL for attribute computation. While this model does
        not take advantage of the benefits provided by a model
        management system, including versioning and tracking,
        the SQL code it generates can still be used to create a
        pipeline.

    Parameters
    ----------
    name: str, optional
        Name  of the model.  This is  not a  built-in
        model, so this name is  used to build the
        final table.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
    eps: float, optional
        The radius of a  neighborhood with respect to
        some point.
    min_samples: int, optional
        Minimum number  of points required to  form a
        dense region.
    p: int, optional
        The p of the p-distance (distance metric used
        during the model computation).

    Attributes
    ----------
    Many attributes are created during the fitting phase.

    n_cluster_: int
        Number of clusters.
    p_: int
        The p of the p-distances.
    n_noise_: int
        Number of outliers.

    .. note::

        All attributes can be accessed using the
        :py:meth:`verticapy.machine_learning.vertica.base.VerticaModel.get_attributes``
        method.

    Examples
    ---------

    The following examples provide a basic understanding of usage.
    For more detailed examples, please refer to the
    :ref:`user_guide.machine_learning` or the
    `Examples <https://www.vertica.com/python/examples/>`_
    section on the website.

    Load data for machine learning
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    We import :py:mod:`verticapy`:

    .. ipython:: python

        import verticapy as vp

    .. hint::

        By assigning an alias to :py:mod:`verticapy`, we mitigate the risk of code
        collisions with other libraries. This precaution is necessary
        because verticapy uses commonly known function names like "average"
        and "median", which can potentially lead to naming conflicts.
        The use of an alias ensures that the functions from verticapy are
        used as intended without interfering with functions from other
        libraries.

    For this example, we will create a small dataset.

    .. ipython:: python

        data = vp.vDataFrame({"col":[1.2, 1.1, 1.3, 1.5, 2, 2.2, 1.09, 0.9, 100, 102]})

    .. note::

        VerticaPy offers a wide range of sample datasets that are
        ideal for training and testing purposes. You can explore
        the full list of available datasets in the :ref:`api.datasets`,
        which provides detailed information on each dataset
        and how to use them effectively. These datasets are invaluable
        resources for honing your data analysis and machine learning
        skills within the VerticaPy environment.

    Model Initialization
    ^^^^^^^^^^^^^^^^^^^^^

    First we import the ``DBSCAN`` model:

    .. code-block::

        from verticapy.machine_learning.vertica import DBSCAN

    .. ipython:: python
        :suppress:

        from verticapy.machine_learning.vertica import DBSCAN

    Then we can create the model:

    .. ipython:: python
        :okwarning:

        model = DBSCAN(
            eps = 0.5,
            min_samples = 2,
            p = 2,
        )

    .. important::

        As this model is not native, it solely relies on SQL statements to
        compute various attributes, storing them within the object. No data
        is saved in the database.

    Model Training
    ^^^^^^^^^^^^^^^

    We can now fit the model:

    .. ipython:: python
        :okwarning:

        model.fit(data, X = ["col"])

    .. important::

        To train a model, you can directly use the :py:class:`vDataFrame` or the
        name of the relation stored in the database.

    .. hint::

        For clustering and anomaly detection, the use of predictors is
        optional. In such cases, all available predictors are considered,
        which can include solely numerical variables or a combination of
        numerical and categorical variables, depending on the model's
        capabilities.

    .. important::

        As this model is not native, it solely relies on SQL statements to
        compute various attributes, storing them within the object. No data
        is saved in the database.

    Prediction
    ^^^^^^^^^^^

    Predicting or ranking the dataset is straight-forward:

    .. ipython:: python
        :suppress:

        result = model.predict()
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_dbscan_prediction.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.predict()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_dbscan_prediction.html

    As shown above, a new column has been created, containing
    the clusters.

    .. hint::
        The name of the new column is optional. If not provided,
        it is randomly assigned.

    Parameter Modification
    ^^^^^^^^^^^^^^^^^^^^^^^

    In order to see the parameters:

    .. ipython:: python

        model.get_params()

    And to manually change some of the parameters:

    .. ipython:: python

        model.set_params({'min_samples': 5})

    Model Register
    ^^^^^^^^^^^^^^

    As this model is not native, it does not support model management and
    versioning. However, it is possible to use the SQL code it generates
    for deployment.
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
        self,
        name: str = None,
        overwrite_model: bool = False,
        eps: float = 0.5,
        min_samples: int = 5,
        p: int = 2,
    ) -> None:
        super().__init__(name, overwrite_model)
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
        return_report: bool = False,
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
        if self.overwrite_model:
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

    .. important::

        This algorithm is not Vertica Native and relies solely
        on SQL for attribute computation. While this model does
        not take advantage of the benefits provided by a model
        management system, including versioning and tracking,
        the SQL code it generates can still be used to create a
        pipeline.

    Parameters
    ----------
    p: int, optional
        The p corresponding to the one of the p-distances
        (distance metric used to compute the model).

    Attributes
    ----------
    Many attributes are created during the fitting phase.

    clusters_: numpy.array
        Cluster centers.
    p_: int
        The p of the p-distances.
    classes_: numpy.array
        The classes labels.

    .. note::

        All attributes can be accessed using the
        :py:meth:`verticapy.machine_learning.vertica.base.VerticaModel.get_attributes``
        method.

    Examples
    ---------

    The following examples provide a basic understanding of usage.
    For more detailed examples, please refer to the
    :ref:`user_guide.machine_learning` or the
    `Examples <https://www.vertica.com/python/examples/>`_
    section on the website.

    Load data for machine learning
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    We import :py:mod:`verticapy`:

    .. code-block:: python

        import verticapy as vp

    .. hint::

        By assigning an alias to :py:mod:`verticapy`, we mitigate the risk of code
        collisions with other libraries. This precaution is necessary
        because verticapy uses commonly known function names like "average"
        and "median", which can potentially lead to naming conflicts.
        The use of an alias ensures that the functions from verticapy are
        used as intended without interfering with functions from other
        libraries.

    For this example, we will use the iris dataset.

    .. code-block:: python

        import verticapy.datasets as vpd

        data = vpd.load_iris()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_iris.html

    .. note::

        VerticaPy offers a wide range of sample datasets that are
        ideal for training and testing purposes. You can explore
        the full list of available datasets in the :ref:`api.datasets`,
        which provides detailed information on each dataset
        and how to use them effectively. These datasets are invaluable
        resources for honing your data analysis and machine learning
        skills within the VerticaPy environment.

    You can easily divide your dataset into training and testing subsets
    using the :py:meth:`vDataFrame.train_test_split` method. This is a
    crucial step when preparing your data for machine learning, as it
    allows you to evaluate the performance of your models accurately.

    .. code-block:: python

        data = vpd.load_iris()
        train, test = data.train_test_split(test_size = 0.2)

    .. warning::

        In this case, VerticaPy utilizes seeded randomization to guarantee
        the reproducibility of your data split. However, please be aware
        that this approach may lead to reduced performance. For a more
        efficient data split, you can use the :py:meth:`vDataFrame.to_db`
        method to save your results into ``tables`` or ``temporary tables``.
        This will help enhance the overall performance of the process.

    .. ipython:: python
        :suppress:

        import verticapy as vp
        import verticapy.datasets as vpd
        data = vpd.load_iris()
        train, test = data.train_test_split(test_size = 0.2)

    Balancing the Dataset
    ^^^^^^^^^^^^^^^^^^^^^^

    In VerticaPy, balancing a dataset to address class imbalances
    is made straightforward through the
    :py:meth:`verticapy.machine_learning.vertica.preprocessing.balance`
    function within the ``preprocessing`` module. This function
    enables users to rectify skewed class distributions efficiently.
    By specifying the target variable and setting parameters like
    the method for balancing, users can effortlessly achieve a more
    equitable representation of classes in their dataset.
    Whether opting for over-sampling, under-sampling, or a combination
    of both, VerticaPy's
    :py:meth:`verticapy.machine_learning.vertica.preprocessing.balance`
    function streamlines the process, empowering users to enhance the
    performance and fairness of their machine learning models trained
    on imbalanced data.

    To balance the dataset, use the following syntax.

    .. code-block:: python

        from verticapy.machine_learning.vertica.preprocessing import balance

        balanced_train = balance(
            name = "my_schema.train_balanced",
            input_relation = train,
            y = "good",
            method = "hybrid",
        )

    .. note::

        With this code, a table named `train_balanced` is created in the
        `my_schema` schema. It can then be used to train the model. In the
        rest of the example, we will work with the full dataset.

    .. hint::

        Balancing the dataset is a crucial step in improving the accuracy
        of machine learning models, particularly when faced with imbalanced
        class distributions. By addressing disparities in the number of
        instances across different classes, the model becomes more adept at
        learning patterns from all classes rather than being biased towards
        the majority class. This, in turn, enhances the model's ability to
        make accurate predictions for under-represented classes. The balanced
        dataset ensures that the model is not dominated by the majority class
        and, as a result, leads to more robust and unbiased model performance.
        Therefore, by employing techniques such as over-sampling, under-sampling,
        or a combination of both during dataset preparation, practitioners can
        significantly contribute to achieving higher accuracy and better
        generalization of their machine learning models.

    Model Initialization
    ^^^^^^^^^^^^^^^^^^^^^

    First we import the ``NearestCentroid`` model:

    .. ipython:: python

        from verticapy.machine_learning.vertica import NearestCentroid

    Then we can create the model:

    .. ipython:: python

        model = NearestCentroid(p = 2)

    Model Training
    ^^^^^^^^^^^^^^^

    We can now fit the model:

    .. ipython:: python

        model.fit(
            train,
            [
                "SepalLengthCm",
                "SepalWidthCm",
                "PetalLengthCm",
                "PetalWidthCm",
            ],
            "Species",
            test,
        )

    .. important::

        To train a model, you can directly use the :py:class:`vDataFrame` or the
        name of the relation stored in the database. The test set is optional
        and is only used to compute the test metrics. In :py:mod:`verticapy`, we
        don't work using ``X`` matrices and ``y`` vectors. Instead, we work
        directly with lists of predictors and the response name.

    .. important::

        As this model is not native, it solely relies on SQL statements to
        compute various attributes, storing them within the object. No data
        is saved in the database.

    Metrics
    ^^^^^^^^

    We can get the entire report using:

    .. ipython:: python
        :suppress:

        result = model.report()
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_cluster_nearest_centroid_report.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.report()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_cluster_nearest_centroid_report.html

    .. important::

        Most metrics are computed using a single SQL query, but some of them might
        require multiple SQL queries. Selecting only the necessary metrics in the
        report can help optimize performance.
        E.g. ``model.report(metrics = ["auc", "accuracy"])``.

    For classification models, we can easily modify the ``cutoff`` to observe
    the effect on different metrics:

    .. ipython:: python
        :suppress:

        result = model.report(cutoff = 0.2)
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_cluster_nearest_centroid_report_cutoff.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.report(cutoff = 0.2)

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_cluster_nearest_centroid_report_cutoff.html


    You can also use the ``NearestCentroid.score`` function to compute any
    classification metric. The default metric is the accuracy:

    .. ipython:: python

        model.score(metric = "f1", average = "macro")

    .. note::

        For multi-class scoring, :py:mod:`verticapy` allows the
        flexibility to use three averaging techniques:
        micro, macro and weighted. Please refer to
        `this link <https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f>`_
        for more details on how they are calculated.

    Prediction
    ^^^^^^^^^^^

    Prediction is straight-forward:

    .. ipython:: python
        :suppress:

        result = model.predict(
            test,
            [
                "SepalLengthCm",
                "SepalWidthCm",
                "PetalLengthCm",
                "PetalWidthCm",
            ],
            "prediction",
        )
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_cluster_nearest_centroid_prediction.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.predict(
            test,
            [
                "SepalLengthCm",
                "SepalWidthCm",
                "PetalLengthCm",
                "PetalWidthCm",
            ],
            "prediction",
        )

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_cluster_nearest_centroid_prediction.html

    .. note::

        Predictions can be made automatically using the test set, in which
        case you don't need to specify the predictors. Alternatively, you
        can pass only the :py:class:`vDataFrame` to the
        :py:meth:`verticapy.machine_learning.vertica.cluster.NearestCentroid.predict`
        function, but in this case, it's essential that the column names of
        the :py:class:`vDataFrame` match the predictors and response name in the
        model.

    Probabilities
    ^^^^^^^^^^^^^^

    It is also easy to get the model's probabilities:

    .. ipython:: python
        :suppress:

        result = model.predict_proba(
            test,
            [
                "SepalLengthCm",
                "SepalWidthCm",
                "PetalLengthCm",
                "PetalWidthCm",
            ],
            "prediction",
        )
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_cluster_nearest_centroid_proba.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. code-block:: python

        model.predict_proba(
            test,
            [
                "SepalLengthCm",
                "SepalWidthCm",
                "PetalLengthCm",
                "PetalWidthCm",
            ],
            "prediction",
        )

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_cluster_nearest_centroid_proba.html

    .. note::

        Probabilities are added to the vDataFrame, and VerticaPy uses the
        corresponding probability function in SQL behind the scenes. You
        can use the ``pos_label`` parameter to add only the probability
        of the selected category.

    Confusion Matrix
    ^^^^^^^^^^^^^^^^^

    You can obtain the confusion matrix.

    .. ipython:: python

        model.confusion_matrix()

    .. hint::

        In the context of multi-class classification, you typically work
        with an overall confusion matrix that summarizes the classification
        efficiency across all classes. However, you have the flexibility to
        specify a ``pos_label`` and adjust the cutoff threshold. In this case,
        a binary confusion matrix is computed, where the chosen class is treated
        as the positive class, allowing you to evaluate its efficiency as if it
        were a binary classification problem.

        **Specific confusion matrix:**

        .. ipython:: python

            model.confusion_matrix(pos_label = "Iris-setosa", cutoff = 0.6)

    .. note::

        In classification, the ``cutoff`` is a threshold value used to
        determine class assignment based on predicted probabilities or
        scores from a classification model. In binary classification,
        if the predicted probability for a specific class is greater
        than or equal to the cutoff, the instance is assigned to the
        positive class; otherwise, it is assigned to the negative class.
        Adjusting the cutoff allows for trade-offs between true positives
        and false positives, enabling the model to be optimized for
        specific objectives or to consider the relative costs of different
        classification errors. The choice of cutoff is critical for
        tailoring the model's performance to meet specific needs.

    Main Plots (Classification Curves)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Classification models allow for the creation of various plots that
    are very helpful in understanding the model, such as the ROC Curve,
    PRC Curve, Cutoff Curve, Gain Curve, and more.

    Most of the classification curves can be found in the
    :ref:`chart_gallery.classification_curve`.

    For example, let's draw the model's ROC curve.

    .. code-block:: python

        model.roc_curve(pos_label = "Iris-setosa")

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "plotly")
        fig = model.roc_curve(pos_label = "Iris-setosa")
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_cluster_nearest_centroid_roc.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_cluster_nearest_centroid_roc.html

    .. important::

        Most of the curves have a parameter called ``nbins``, which is essential
        for estimating metrics. The larger the ``nbins``, the more precise the
        estimation, but it can significantly impact performance. Exercise caution
        when increasing this parameter excessively.

    .. hint::

        In binary classification, various curves can be easily plotted. However,
        in multi-class classification, it's important to select the ``pos_label``
        , representing the class to be treated as positive when drawing the curve.

    Other Plots
    ^^^^^^^^^^^^

    **Contour plot** is another useful plot that can be produced
    for models with two predictors.

    .. code-block:: python

        model.contour(pos_label = "Iris-setosa")

    .. important::

        Machine learning models with two predictors can usually
        benefit from their own contour plot. This visual representation
        aids in exploring predictions and gaining a deeper understanding
        of how these models perform in different scenarios.
        Please refer to  :ref:`chart_gallery.contour` for more examples.

    Parameter Modification
    ^^^^^^^^^^^^^^^^^^^^^^^

    In order to see the parameters:

    .. ipython:: python

        model.get_params()

    And to manually change some of the parameters:

    .. ipython:: python

        model.set_params({'p': 3})

    Model Register
    ^^^^^^^^^^^^^^

    As this model is not native, it does not support model management and
    versioning. However, it is possible to use the SQL code it generates
    for deployment.

    Model Exporting
    ^^^^^^^^^^^^^^^^

    **To Memmodel**

    .. code-block:: python

        model.to_memmodel()

    .. note::

        ``MemModel`` objects serve as in-memory representations of machine
        learning models. They can be used for both in-database and in-memory
        prediction tasks. These objects can be pickled in the same way that
        you would pickle a ``scikit-learn`` model.

    The following methods for exporting the model use ``MemModel``, and it
    is recommended to use ``MemModel`` directly.

    **To SQL**

    You can get the SQL code by:

    .. ipython:: python

        model.to_sql()

    **To Python**

    To obtain the prediction function in Python syntax, use the following code:

    .. ipython:: python

        X = [[5, 2, 3, 1]]
        model.to_python()(X)

    .. hint::

        The
        :py:meth:`verticapy.machine_learning.vertica.cluster.NearestCentroid.to_python`
        method is used to retrieve predictions,
        probabilities, or cluster distances. For specific details on how to
        use this method for different model types, refer to the relevant
        documentation for each model.
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
    def __init__(
        self, name: str = None, overwrite_model: bool = False, p: int = 2
    ) -> None:
        super().__init__(name, overwrite_model)
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
        self.clusters_ = centroids.to_numpy()[:, 0:-1].astype(float)
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
