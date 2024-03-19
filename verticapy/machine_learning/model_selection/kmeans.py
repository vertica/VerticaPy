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

from typing import Literal, Optional, Union

import numpy as np

from tqdm.auto import tqdm

import verticapy._config.config as conf
from verticapy._typing import PlottingObject, SQLColumns, SQLRelation
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._gen import gen_tmp_name
from verticapy._utils._sql._format import format_type, quote_ident, schema_relation

from verticapy.core.tablesample.base import TableSample

from verticapy.machine_learning.vertica.cluster import KMeans, KPrototypes

from verticapy.plotting._utils import PlottingUtils


@save_verticapy_logs
def best_k(
    input_relation: SQLRelation,
    X: Optional[SQLColumns] = None,
    n_cluster: Union[tuple, list] = (1, 100),
    init: Literal["kmeanspp", "random", None] = None,
    max_iter: int = 50,
    tol: float = 1e-4,
    use_kprototype: bool = False,
    gamma: float = 1.0,
    elbow_score_stop: float = 0.8,
    **kwargs,
) -> int:
    """
    Finds the :py:class:`~verticapy.machine_learning.vertica.KMeans`
    / :py:class:`~verticapy.machine_learning.vertica.KPrototypes`
    ``k`` based on a score.

    Parameters
    ----------
    input_relation: SQLRelation
        Relation used to train the
        model.
    X: SQLColumns, optional
        ``list`` of the predictor
        columns. If empty, all
        numerical columns are used.
    n_cluster: tuple | list, optional
        Tuple representing the number
        of clusters to start and end
        with. This can also be a
        customized list with various
        ``k`` values to test.
    init: str | list, optional
        The method used to  find the
        initial cluster centers.

        - kmeanspp:
            Only available when
            ``use_kprototype = False``
            Use the ``k-means++`` method
            to initialize the centers.
        - random:
            Randomly subsamples the data
            to find initial centers.

        Default  value  is  ``kmeanspp`` if
        ``use_kprototype = False``; otherwise,
        ``random``.
    max_iter: int, optional
        The maximum number of iterations
        for the algorithm.
    tol: float, optional
        Determines whether the algorithm
        has converged. The algorithm is
        considered  converged after no
        center has moved more than a
        distance of ``tol`` from the
        previous iteration.
    use_kprototype: bool, optional
        If set to ``True``, the function
        uses the
        :py:class:`~verticapy.machine_learning.vertica.KPrototypes`
        algorithm instead of
        :py:class:`~verticapy.machine_learning.vertica.KMeans`.
        :py:class:`~verticapy.machine_learning.vertica.KPrototypes`
        can handle categorical features.
    gamma: float, optional
        Only if ``use_kprototype = True``.
        Weighting factor for categorical
        columns. It determines the relative
        importance of numerical and
        categorical attributes.
    elbow_score_stop: float, optional
        Stops searching for parameters when
        the specified elbow score is reached.

    Returns
    -------
    int
        the k-means / k-prototypes ``k``.

    Examples
    --------
    The following examples provide a
    basic understanding of usage.
    For more detailed examples, please
    refer to the: :ref:`chart_gallery.elbow`
    page.

    Load data for machine learning
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    We import :py:mod:`verticapy`:

    .. ipython:: python

        import verticapy as vp

    .. hint::

        By assigning an alias to :py:mod:`verticapy`,
        we mitigate the risk of code collisions with
        other libraries. This precaution is necessary
        because verticapy uses commonly known function
        names like "average" and "median", which can
        potentially lead to naming conflicts. The use
        of an alias ensures that the functions from
        :py:mod:`verticapy` are used as intended
        without interfering with functions from other
        libraries.

    For this example, we will
    use the iris dataset.

    .. code-block:: python

        import verticapy.datasets as vpd

        data = vpd.load_iris()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_iris.html

    .. note::

        VerticaPy offers a wide range of sample
        datasets that are ideal for training
        and testing purposes. You can explore
        the full list of available datasets in
        the :ref:`api.datasets`, which provides
        detailed information on each dataset and
        how to use them effectively. These datasets
        are invaluable resources for honing your
        data analysis and machine learning skills
        within the VerticaPy environment.

    .. ipython:: python
        :suppress:

        import verticapy.datasets as vpd
        data = vpd.load_iris()

    Data Exploration
    ^^^^^^^^^^^^^^^^^

    Through a quick scatter plot, we can
    observe that the data has three main
    clusters.

    .. code-block:: python

        data.scatter(
            columns = ["PetalLengthCm", "SepalLengthCm"],
            by = "Species",
        )

    .. ipython:: python
        :suppress:
        :okwarning:

        vp.set_option("plotting_lib", "plotly")
        fig = data.scatter(
            columns = ["PetalLengthCm", "SepalLengthCm"],
            by = "Species",
        )
        fig.write_html("SPHINX_DIRECTORY/figures/core_machine_learning_best_k.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/core_machine_learning_best_k.html

    Elbow Score
    ^^^^^^^^^^^^

    Let's compute the optimal ``k`` for our
    :py:class:`~verticapy.machine_learning.vertica.KMeans`
    algorithm and check if it aligns with
    the three clusters we observed earlier.

    .. ipython:: python
        :okwarning:

        from verticapy.machine_learning.model_selection import best_k

        best_k(
            input_relation = data,
            X = data.get_columns(exclude_columns= "Species"), # All columns except Species
            n_cluster = (1, 100),
            init = "kmeanspp",
            elbow_score_stop = 0.9,
        )

    .. note::

        You can experiment with the Elbow score to
        determine the optimal number of clusters.
        The score is based on the ratio of Between
        -Cluster Sum of Squares to Total Sum of
        Squares, providing a way to assess the
        clustering accuracy. A score of 1 indicates
        a perfect clustering.

    .. seealso::

        | :py:func:`~verticapy.machine_learning.model_selection.kmeans.elbow` :
            Draws an Elbow curve.
    """
    X = format_type(X, dtype=list)
    if X == []:
        X = None
    if not init and (use_kprototype):
        init = "random"
    elif not init:
        init = "kmeanspp"
    if isinstance(n_cluster, tuple):
        L = range(n_cluster[0], n_cluster[1])
    else:
        L = n_cluster
        L.sort()
    schema = schema_relation(input_relation)[0]
    if not schema:
        schema = conf.get_option("temp_schema")
    schema = quote_ident(schema)
    if conf.get_option("tqdm") and (
        "tqdm" not in kwargs or ("tqdm" in kwargs and kwargs["tqdm"])
    ):
        loop = tqdm(L)
    else:
        loop = L
    i = None
    for i in loop:
        if use_kprototype:
            if init == "kmeanspp":
                init = "random"
            model = KPrototypes(
                n_cluster=i,
                init=init,
                max_iter=max_iter,
                tol=tol,
                gama=gamma,
            )
        else:
            model = KMeans(
                n_cluster=i,
                init=init,
                max_iter=max_iter,
                tol=tol,
            )
        model.fit(
            input_relation,
            X,
            return_report=True,
        )
        score = model.elbow_score_
        if score > elbow_score_stop:
            return i
        model.drop()
    print(
        f"\u26A0 The K was not found. The last K (= {i}) "
        f"is returned with an elbow score of {score}"
    )
    return i


@save_verticapy_logs
def elbow(
    input_relation: SQLRelation,
    X: Optional[SQLColumns] = None,
    n_cluster: Union[tuple, list] = (1, 15),
    init: Literal["kmeanspp", "random", None] = None,
    max_iter: int = 50,
    tol: float = 1e-4,
    use_kprototype: bool = False,
    gamma: float = 1.0,
    show: bool = True,
    chart: Optional[PlottingObject] = None,
    **style_kwargs,
) -> TableSample:
    """
    Draws an Elbow curve.

    Parameters
    ----------
    input_relation: SQLRelation
        Relation used to train the
        model.
    X: SQLColumns, optional
        ``list`` of the predictor
        columns. If empty, all
        numerical columns are used.
    n_cluster: tuple | list, optional
        Tuple representing the number
        of clusters to start and end
        with. This can also be a
        customized list with various
        ``k`` values to test.
    init: str | list, optional
        The method used to  find the
        initial cluster centers.

        - kmeanspp:
            Only available when
            ``use_kprototype = False``
            Use the ``k-means++`` method
            to initialize the centers.
        - random:
            Randomly subsamples the data
            to find initial centers.

        Default  value  is  ``kmeanspp`` if
        ``use_kprototype = False``; otherwise,
        ``random``.
    max_iter: int, optional
        The maximum number of iterations
        for the algorithm.
    tol: float, optional
        Determines whether the algorithm
        has converged. The algorithm is
        considered  converged after no
        center has moved more than a
        distance of ``tol`` from the
        previous iteration.
    use_kprototype: bool, optional
        If set to ``True``, the function
        uses the
        :py:class:`~verticapy.machine_learning.vertica.KPrototypes`
        algorithm instead of
        :py:class:`~verticapy.machine_learning.vertica.KMeans`.
        :py:class:`~verticapy.machine_learning.vertica.KPrototypes`
        can handle categorical features.
    gamma: float, optional
        Only if ``use_kprototype = True``.
        Weighting factor for categorical
        columns. It determines the relative
        importance of numerical and
        categorical attributes.
    show: bool, optional
        If set to ``True``, the  Plotting
        object is returned.
    chart: PlottingObject, optional
        The chart object to plot on.
    **style_kwargs
        Any optional parameter to pass to
        the Plotting functions.

    Returns
    -------
    TableSample
        ``nb_clusters,total_within_cluster_ss,between_cluster_ss,total_ss, elbow_score``

    Examples
    --------
    The following examples provide a
    basic understanding of usage.
    For more detailed examples, please
    refer to the: :ref:`chart_gallery.elbow`
    page.

    Load data for machine learning
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    We import :py:mod:`verticapy`:

    .. ipython:: python

        import verticapy as vp

    .. hint::

        By assigning an alias to :py:mod:`verticapy`,
        we mitigate the risk of code collisions with
        other libraries. This precaution is necessary
        because verticapy uses commonly known function
        names like "average" and "median", which can
        potentially lead to naming conflicts. The use
        of an alias ensures that the functions from
        :py:mod:`verticapy` are used as intended
        without interfering with functions from other
        libraries.

    For this example, we will
    use the iris dataset.

    .. code-block:: python

        import verticapy.datasets as vpd

        data = vpd.load_iris()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_iris.html

    .. note::

        VerticaPy offers a wide range of sample
        datasets that are ideal for training
        and testing purposes. You can explore
        the full list of available datasets in
        the :ref:`api.datasets`, which provides
        detailed information on each dataset and
        how to use them effectively. These datasets
        are invaluable resources for honing your
        data analysis and machine learning skills
        within the VerticaPy environment.

    .. ipython:: python
        :suppress:

        import verticapy.datasets as vpd
        data = vpd.load_iris()

    Data Exploration
    ^^^^^^^^^^^^^^^^^

    Through a quick scatter plot, we can
    observe that the data has three main
    clusters.

    .. code-block:: python

        data.scatter(
            columns = ["PetalLengthCm", "SepalLengthCm"],
            by = "Species",
        )

    .. ipython:: python
        :suppress:
        :okwarning:

        vp.set_option("plotting_lib", "plotly")
        fig = data.scatter(
            columns = ["PetalLengthCm", "SepalLengthCm"],
            by = "Species",
        )
        fig.write_html("SPHINX_DIRECTORY/figures/core_machine_learning_best_k.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/core_machine_learning_best_k.html

    Elbow Curve
    ^^^^^^^^^^^^

    Let's compute the optimal ``k`` for our
    :py:class:`~verticapy.machine_learning.vertica.KMeans`
    algorithm and check if it aligns with the three
    clusters we observed earlier.

    To achieve this, let's create the Elbow curve.

    .. code-block:: python

        from verticapy.machine_learning.model_selection import elbow

        elbow(
            input_relation = data,
            X = data.get_columns(exclude_columns= "Species"), # All columns except Species
            n_cluster = (1, 100),
            init = "kmeanspp",
        )

    .. ipython:: python
        :suppress:
        :okwarning:

        vp.set_option("plotting_lib", "plotly")
        from verticapy.machine_learning.model_selection import elbow
        fig = elbow(
            input_relation = data,
            X = data.get_columns(exclude_columns= "Species"), # All columns except Species
            n_cluster = (1, 100),
            init = "kmeanspp",
        )
        fig.write_html("SPHINX_DIRECTORY/figures/core_machine_learning_elbow.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/core_machine_learning_elbow.html

    .. note::

        You can experiment with the Elbow score to
        determine the optimal number of clusters.
        The score is based on the ratio of Between
        -Cluster Sum of Squares to Total Sum of
        Squares, providing a way to assess the
        clustering accuracy. A score of 1 indicates
        a perfect clustering.

    .. note::

        It's evident from the Elbow curve that
        ``k=3`` is a suitable choice, indicating
        the optimal number of clusters for the
        KMeans algorithm.

    .. seealso::

        | :py:func:`~verticapy.machine_learning.model_selection.kmeans.best_k` :
            Finds the :py:class:`~verticapy.machine_learning.vertica.KMeans` /
            :py:class:`~verticapy.machine_learning.vertica.KPrototypes` ``k``
            based on a score.
    """
    X = format_type(X, dtype=list)
    if X == []:
        X = None
    if not init and (use_kprototype):
        init = "random"
    elif not init:
        init = "kmeanspp"
    if isinstance(n_cluster, tuple):
        L = range(n_cluster[0], n_cluster[1])
    else:
        L = n_cluster
        L.sort()
    schema = schema_relation(input_relation)[0]
    elbow_score = []
    between_cluster_ss = []
    total_ss = []
    total_within_cluster_ss = []
    if isinstance(n_cluster, tuple):
        L = [i for i in range(n_cluster[0], n_cluster[1])]
    else:
        L = n_cluster
        L.sort()
    if conf.get_option("tqdm"):
        loop = tqdm(L)
    else:
        loop = L
    for i in loop:
        if use_kprototype:
            if init == "kmeanspp":
                init = "random"
            model = KPrototypes(
                n_cluster=i,
                init=init,
                max_iter=max_iter,
                tol=tol,
                gamma=gamma,
            )
        else:
            model = KMeans(
                n_cluster=i,
                init=init,
                max_iter=max_iter,
                tol=tol,
            )
        model.fit(
            input_relation,
            X,
            return_report=True,
        )
        elbow_score += [float(model.elbow_score_)]
        between_cluster_ss += [float(model.between_cluster_ss_)]
        total_ss += [float(model.total_ss_)]
        total_within_cluster_ss += [float(model.total_within_cluster_ss_)]
        model.drop()
    if show:
        vpy_plt, kwargs = PlottingUtils().get_plotting_lib(
            class_name="ElbowCurve",
            chart=chart,
            style_kwargs=style_kwargs,
        )
        data = {
            "x": np.array(L),
            "y": np.array(elbow_score),
            "z0": np.array(total_within_cluster_ss),
            "z1": np.array(between_cluster_ss),
            "z2": np.array(total_ss),
        }
        layout = {
            "title": "Elbow Curve",
            "x_label": "Number of Clusters",
            "y_label": "Elbow Score (Between-Cluster SS / Total SS)",
            "z0_label": "Total Within Cluster SS",
            "z1_label": "Between Cluster SS",
            "z2_label": "Total SS",
        }
        return vpy_plt.ElbowCurve(data=data, layout=layout).draw(**kwargs)
    values = {
        "index": L,
        "total_within_cluster_ss": total_within_cluster_ss,
        "between_cluster_ss": between_cluster_ss,
        "total_ss": total_ss,
        "elbow_score": elbow_score,
    }
    return TableSample(values=values)
