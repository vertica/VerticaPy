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
from typing import Literal, Optional, Union

import numpy as np

from verticapy._typing import ArrayLike, NoneType
from verticapy._utils._sql._format import clean_query, format_magic, format_type

from verticapy.machine_learning.memmodel.base import InMemoryModel
from verticapy.machine_learning.memmodel.tree import Tree


class Clustering(InMemoryModel):
    """
    :py:meth:`verticapy.machine_learning.memmodel.base.InMemoryModel`
    implementation of clustering algorithms.

    .. note::

        This is base class for all in-memory implementations of
        clustering algorithms viz.
        :py:meth:`verticapy.machine_learning.memmodel.cluster.KMeans`,
        :py:meth:`verticapy.machine_learning.memmodel.cluster.NearestCentroid`
        , :py:meth:`verticapy.machine_learning.memmodel.cluster.BisectingKMeans`
        and :py:meth:`verticapy.machine_learning.memmodel.cluster.KPrototypes`

    Parameters
    ----------
    clusters: ArrayLike
        ArrayLike   of   the   model's  cluster   centers.
    p: int, optional
        The p corresponding to one of the p-distances.
    clusters_names: ArrayLike, optional
        Names of the clusters.
    """

    # Properties.

    @property
    def object_type(self) -> Literal["Clustering"]:
        return "Clustering"

    @property
    def _attributes(self) -> list[str]:
        return ["clusters_", "p_"]

    # System & Special Methods.

    def __init__(
        self,
        clusters: ArrayLike,
        p: int = 2,
        clusters_names: Optional[ArrayLike] = None,
    ) -> None:
        clusters_names = format_type(clusters_names, dtype=list)
        self.clusters_ = np.array(clusters).astype(float)
        self.classes_ = np.array(clusters_names)
        self.p_ = p

    # Prediction / Transformation Methods - IN MEMORY.

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predicts  clusters  using  the input  matrix.

        Parameters
        ----------
        X: ArrayLike
            The data on which to make the prediction.

        Returns
        -------
        numpy.array
            Predicted values.
        """
        distances = self.transform(X)
        clusters_pred_id = np.argmin(distances, axis=1).astype(object)
        if hasattr(self, "classes_") and len(self.classes_) > 0:
            for idx, c in enumerate(self.classes_):
                clusters_pred_id[clusters_pred_id == idx] = c
        return clusters_pred_id

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """
        Predicts the probability of each input to belong
        to the model clusters.

        Parameters
        ----------
        X: ArrayLike
            The data on which to make the prediction.

        Returns
        -------
        numpy.array
            Probabilities.
        """
        distances = self.transform(X)
        return (
            1 / (distances + 1e-99) / np.sum(1 / (distances + 1e-99), axis=1)[:, None]
        )

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Transforms and returns the distance to each cluster.

        Parameters
        ----------
        X: ArrayLike
            The data on which to make the transformation.

        Returns
        -------
        numpy.array
            Transformed values.
        """
        result = []
        for centroid in self.clusters_:
            result += [
                np.sum((np.array(centroid) - X) ** self.p_, axis=1) ** (1 / self.p_)
            ]
        return np.column_stack(result)

    # Prediction / Transformation Methods - IN DATABASE.

    def predict_sql(self, X: ArrayLike) -> str:
        """
        Returns the SQL code needed to deploy the model using
        its attributes.

        Parameters
        ----------
        X: ArrayLike
            The names or values of the input predictors.

        Returns
        -------
        str
            SQL code.
        """
        if hasattr(self, "classes_"):
            n = len(self.classes_)
        else:
            n = 0
        clusters_distance = self.transform_sql(X)
        sql = []
        k = len(clusters_distance)
        for i in range(k):
            list_tmp = []
            for j in range(i):
                list_tmp += [f"{clusters_distance[i]} <= {clusters_distance[j]}"]
            sql += [" AND ".join(list_tmp)]
        sql = sql[1:]
        sql.reverse()
        is_null_x = " OR ".join([f"{x} IS NULL" for x in X])
        sql_final = f"CASE WHEN {is_null_x} THEN NULL"
        for i in range(k - 1):
            if n == 0:
                c = k - i - 1
            else:
                c = format_magic(self.classes_[k - i - 1])
            sql_final += f" WHEN {sql[i]} THEN {c}"
        if n == 0:
            c = 0
        else:
            c = format_magic(self.classes_[0])
        sql_final += f" ELSE {c} END"
        return sql_final

    def predict_proba_sql(self, X: ArrayLike) -> list[str]:
        """
        Returns  the SQL code needed to deploy the model
        probabilities.

        Parameters
        ----------
        X: ArrayLike
            The names or values of the input predictors.

        Returns
        -------
        list
            SQL code.
        """
        clusters_distance = self.transform_sql(X)
        sum_distance = " + ".join([f"1 / ({d})" for d in clusters_distance])
        proba = [
            f"""
            (CASE 
                WHEN {clusters_distance[i]} = 0 
                    THEN 1.0 
                ELSE 1 / ({clusters_distance[i]}) 
                      / ({sum_distance})
            END)"""
            for i in range(len(clusters_distance))
        ]
        return [clean_query(p) for p in proba]

    def transform_sql(self, X: ArrayLike) -> list[str]:
        """
        Transforms  and returns the SQL distance to each
        cluster.

        Parameters
        ----------
        X: ArrayLike
            The names or values of the input predictors.

        Returns
        -------
        list
            SQL code.
        """
        for c in self.clusters_:
            if len(X) != len(c):
                raise ValueError(
                    "The length of parameter 'X' must be the same as "
                    "the length of each cluster."
                )
        clusters_distance = []
        for c in self.clusters_:
            list_tmp = []
            for idx in range(len(X)):
                list_tmp += [f"POWER({X[idx]} - {c[idx]}, {self.p_})"]
            clusters_distance += ["POWER(" + " + ".join(list_tmp) + f", 1 / {self.p_})"]
        return clusters_distance


class KMeans(Clustering):
    """
    :py:meth:`verticapy.machine_learning.memmodel.base.InMemoryModel`
    implementation of KMeans.

    Parameters
    ----------
    clusters: ArrayLike
        List of the model's cluster centers.
    p: int, optional
        The p corresponding to one of the p-distances.

    .. note::

        :py:meth:`verticapy.machine_learning.memmodel` are
        defined entirely by their attributes. For example,
        'cluster centroids' and 'p value' define a KMeans model.

    Attributes
    ----------
    Attributes are identical to the input parameters, followed by an
    underscore ('_').

    Examples
    --------

    **Initalization**

    Import the required module.

    .. ipython:: python

        from verticapy.machine_learning.memmodel.cluster import KMeans

    A KMeans model is defined by its cluster centroids and the p value.
    In this example, we will use the following:

    .. ipython:: python

        clusters = [[0.5, 0.6], [1, 2], [100, 200]]
        p = 2

    Let's create a
    :py:meth:`verticapy.machine_learning.memmodel.cluster.KMeans` model.

    .. ipython:: python

        model_km = KMeans(clusters, p)

    Create a dataset.

    .. ipython:: python

        data = [[2, 3]]

    **Making In-Memory Predictions**

    Use :py:meth:`verticapy.machine_learning.memmodel.cluster.KMeans.predict`
    method to do predictions

    .. ipython:: python

        model_km.predict(data)[0]

    .. note::

        :py:meth:`verticapy.machine_learning.memmodel.cluster.KMeans`
        assigns a cluster id to identify each cluster.
        In this example, cluster with centroid [0.5, 0.6] will have id = 0,
        with centroid [1,2] will have id = 1 and so on.
        :py:meth:`verticapy.machine_learning.memmodel.cluster.KMeans.predict`
        method returns the id of the predicted cluster.

    Use :py:meth:`verticapy.machine_learning.memmodel.cluster.KMeans.predict_proba`
    method to compute the predicted probabilities for each cluster

    .. ipython:: python

        model_km.predict_proba(data)

    Use :py:meth:`verticapy.machine_learning.memmodel.cluster.KMeans.transform`
    method to compute the distance from each cluster

    .. ipython:: python

        model_km.transform(data)

    **Deploy SQL Code**

    Let's use the following column names:

    .. ipython:: python

        cnames = ['col1', 'col2']

    Use
    :py:meth:`verticapy.machine_learning.memmodel.cluster.KMeans.predict_sql`
    method to get the SQL code needed to deploy the model
    using its attributes.

    .. ipython:: python

        model_km.predict_sql(cnames)

    Use
    :py:meth:`verticapy.machine_learning.memmodel.cluster.KMeans.predict_proba_sql`
    method to get the SQL code needed to deploy the model that
    computes predicted probabilities.

    .. ipython:: python

        model_km.predict_proba_sql(cnames)

    Use
    :py:meth:`verticapy.machine_learning.memmodel.cluster.KMeans.transform_sql`
    method to get the SQL code needed to deploy the model that
    computes distance from each cluster.

    .. ipython:: python

        model_km.transform_sql(cnames)

    .. hint::

        This object can be pickled and used in any in-memory
        environment, just like `SKLEARN <https://scikit-learn.org/>`_ models.
    """

    # Properties.

    @property
    def object_type(self) -> Literal["KMeans"]:
        return "KMeans"

    # System & Special Methods.

    def __init__(self, clusters: ArrayLike, p: int = 2) -> None:
        self.clusters_ = np.array(clusters).astype(float)
        self.p_ = p


class NearestCentroid(Clustering):
    """
    :py:meth:`verticapy.machine_learning.memmodel.base.InMemoryModel`
    implementation  of NearestCentroid algorithm.

    Parameters
    ----------
    clusters: ArrayLike
        List of the model's cluster centers.
    classes: ArrayLike
        Names of the classes.
    p: int, optional
        The p corresponding to  one of the p-distances.

    Attributes
    ----------
    Attributes are identical to the input parameters, followed by an
    underscore ('_').

    Examples
    --------

    **Initalization**

    Import the required module.

    .. ipython:: python

        from verticapy.machine_learning.memmodel.cluster import NearestCentroid

    A NearestCentroid model is defined by its cluster centroids,
    classes and the p value. In this example, we will use the
    following:

    .. ipython:: python

        clusters = [[0.5, 0.6], [1, 2], [100, 200]]
        p = 2
        classes = ['class_a', 'class_b', 'class_c']

    Let's create a
    :py:meth:`verticapy.machine_learning.memmodel.cluster.NearestCentroid`
    model.

    .. ipython:: python

        model_nc = NearestCentroid(clusters, classes, p)

    Create a dataset.

    .. ipython:: python

        data = [[2, 3]]

    **Making In-Memory Predictions**

    Use
    :py:meth:`verticapy.machine_learning.memmodel.cluster.NearestCentroid.predict`
    method to do predictions.

    .. ipython:: python

        model_nc.predict(data)[0]

    Use
    :py:meth:`verticapy.machine_learning.memmodel.cluster.NearestCentroid.predict_proba`
    method to compute the predicted probabilities for
    each cluster.

    .. ipython:: python

        model_nc.predict_proba(data)

    Use
    :py:meth:`verticapy.machine_learning.memmodel.cluster.NearestCentroid.transform`
    method to compute the distance from each cluster.

    .. ipython:: python

        model_nc.transform(data)

    **Deploy SQL Code**

    Let's use the following column names:

    .. ipython:: python

        cnames = ['col1', 'col2']

    Use
    :py:meth:`verticapy.machine_learning.memmodel.cluster.NearestCentroid.predict_sql`
    method to get the SQL code needed to deploy the model
    using its attributes.

    .. ipython:: python

        model_nc.predict_sql(cnames)

    Use
    :py:meth:`verticapy.machine_learning.memmodel.cluster.NearestCentroid.predict_proba_sql`
    method to get the SQL code needed to deploy the model
    that computes predicted probabilities.

    .. ipython:: python

        model_nc.predict_proba_sql(cnames)

    Use
    :py:meth:`verticapy.machine_learning.memmodel.cluster.NearestCentroid.transform_sql`
    method to get the SQL code needed to deploy the model
    that computes distance from each cluster.

    .. ipython:: python

        model_nc.transform_sql(cnames)

    .. hint::

        This object can be pickled and used in any in-memory environment,
        just like `SKLEARN <https://scikit-learn.org/>`_ models.
    """

    # Properties.

    @property
    def object_type(self) -> Literal["NearestCentroid"]:
        return "NearestCentroid"

    @property
    def _attributes(self) -> list[str]:
        return ["clusters_", "classes_", "p_"]

    # System & Special Methods.

    def __init__(
        self,
        clusters: ArrayLike,
        classes: ArrayLike,
        p: int = 2,
    ) -> None:
        self.clusters_ = np.array(clusters).astype(float)
        self.classes_ = np.array(classes)
        self.p_ = p


class BisectingKMeans(Clustering, Tree):
    """
    :py:meth:`verticapy.machine_learning.memmodel.base.InMemoryModel`
    implementation of BisectingKMeans.

    Parameters
    ----------
    clusters: ArrayLike
        List of the model's cluster centers.
    children_left: ArrayLike
        A list  of node IDs, where  children_left[i] is
        the node ID of the left child of node i.
    children_right: ArrayLike
        A list of node IDs, where  children_right[i] is
        the node ID of the right child of node i.
    cluster_size: ArrayLike
        A list of sizes,  where  cluster_size[i] is the
        number of elements in node i.
    cluster_score: ArrayLike
        A list of scores, where cluster_score[i] is the
        score  for internal  node i.  The score is  the
        ratio between the within-cluster sum of squares
        of the node and the total within-cluster sum of
        squares.
    p: int, optional
        The p corresponding to one of the p-distances.

    Attributes
    ----------
    Attributes are identical to the input parameters, followed by an
    underscore ('_').

    Examples
    --------

    **Initalization**

    Import the required module.

    .. ipython:: python

        from verticapy.machine_learning.memmodel.cluster import BisectingKMeans

    A BisectingKMeans model is defined by its cluster centroids,
    left and right child node id's of given node. In this example,
    we will use the following:

    .. ipython:: python

        clusters = [[0.5, 0.6], [1, 2], [100, 200], [10, 700], [-100, -200]]
        children_left = [1, 3, None, None, None]
        children_right = [2, 4, None, None, None]

    Let's create a
    :py:meth:`verticapy.machine_learning.memmodel.cluster.BisectingKMeans`
    model.

    .. ipython:: python

        model_bkm = BisectingKMeans(clusters, children_left, children_right)

    Create a dataset.

    .. ipython:: python

        data = [[2, 3]]

    **Making In-Memory Predictions**

    Use
    :py:meth:`verticapy.machine_learning.memmodel.cluster.BisectingKMeans.predict`
    method to do predictions.

    .. ipython:: python

        model_bkm.predict(data)[0]

    Use
    :py:meth:`verticapy.machine_learning.memmodel.cluster.BisectingKMeans.predict_proba`
    method to compute the predicted probabilities for
    each cluster.

    .. ipython:: python

        model_bkm.predict_proba(data)

    Use
    :py:meth:`verticapy.machine_learning.memmodel.cluster.BisectingKMeans.transform`
    method to compute the distance from each cluster.

    .. ipython:: python

        model_bkm.transform(data)

    Use
    :py:meth:`verticapy.machine_learning.memmodel.cluster.BisectingKMeans.to_graphviz`
    method to generate code for a `Graphviz <https://graphviz.org/>`_
    tree.

    .. ipython:: python

        model_bkm.to_graphviz()

    Use
    :py:meth:`verticapy.machine_learning.memmodel.cluster.BisectingKMeans.plot_tree`
    method to draw the input tree.

    .. code-block:: python

        model_bkm.plot_tree()

    .. ipython:: python
        :suppress:

        res = model_bkm.plot_tree()
        res.render(filename='figures/machine_learning_cluster_bisecting_kmeans', format='png')

    .. image:: /../figures/machine_learning_cluster_bisecting_kmeans.png

    .. note::

        :py:meth:`verticapy.machine_learning.memmodel.cluster.BisectingKMeans.plot_tree`
        requires the `Graphviz <https://graphviz.org/download/>`_
        module.

    **Deploy SQL Code**

    Let's use the following column names:

    .. ipython:: python

        cnames = ['col1', 'col2']

    Use
    :py:meth:`verticapy.machine_learning.memmodel.cluster.BisectingKMeans.predict_sql`
    method to get the SQL code needed to deploy the model
    using its attributes.

    .. ipython:: python

        model_bkm.predict_sql(cnames)

    Use
    :py:meth:`verticapy.machine_learning.memmodel.cluster.BisectingKMeans.predict_proba_sql`
    method to get the SQL code needed to deploy the model
    that computes predicted probabilities.

    .. ipython:: python

        model_bkm.predict_proba_sql(cnames)

    Use
    :py:meth:`verticapy.machine_learning.memmodel.cluster.BisectingKMeans.transform_sql`
    method to get the SQL code needed to deploy the model
    that computes distance from each cluster.

    .. ipython:: python

        model_bkm.transform_sql(cnames)

    .. hint::

        This object can be pickled and used in any in-memory environment,
        just like `SKLEARN <https://scikit-learn.org/>`_ models.
    """

    # Properties.

    @property
    def object_type(self) -> Literal["BisectingKMeans"]:
        return "BisectingKMeans"

    @property
    def _attributes(
        self,
    ) -> Literal[
        "clusters_",
        "children_left_",
        "children_left_",
        "children_right_",
        "cluster_size_",
        "cluster_score_",
        "p_",
    ]:
        return [
            "clusters_",
            "children_left_",
            "children_left_",
            "children_right_",
            "cluster_size_",
            "cluster_score_",
            "p_",
        ]

    # System & Special Methods.

    def __init__(
        self,
        clusters: ArrayLike,
        children_left: ArrayLike,
        children_right: ArrayLike,
        cluster_size: Optional[ArrayLike] = None,
        cluster_score: Optional[ArrayLike] = None,
        p: int = 2,
    ) -> None:
        cluster_size, cluster_score = format_type(
            cluster_size, cluster_score, dtype=list
        )
        self.clusters_ = np.array(clusters).astype(float)
        self.children_left_ = np.array(children_left)
        self.children_right_ = np.array(children_right)
        self.cluster_size_ = np.array(cluster_size)
        self.cluster_score_ = np.array(cluster_score)
        self.p_ = p

    # Prediction / Transformation Methods - IN MEMORY.

    def _predict_tree(
        self,
        X: ArrayLike,
        node_id: int,
    ) -> int:
        """
        Function used recursively to get the tree prediction
        starting at the input node.
        """
        if isinstance(self.children_left_[node_id], NoneType) and isinstance(
            self.children_right_[node_id], NoneType
        ):
            return int(node_id)
        else:
            right_node = int(self.children_right_[node_id])
            left_node = int(self.children_left_[node_id])
            if np.sum((X - self.clusters_[left_node]) ** self.p_) < np.sum(
                (X - self.clusters_[right_node]) ** self.p_
            ):
                return self._predict_tree(X, left_node)
            else:
                return self._predict_tree(X, right_node)

    def _predict_row(self, X: ArrayLike) -> int:
        """
        Function used recursively to get the tree prediction.
        """
        return self._predict_tree(X, 0)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predicts using the bisecting k-means model.

        Parameters
        ----------
        X: ArrayLike
            The data on which to make the prediction.

        Returns
        -------
        numpy.array
            Predicted values.
        """
        return np.apply_along_axis(self._predict_row, 1, X)

    # Prediction / Transformation Methods - IN DATABASE.

    def _predict_tree_sql(
        self,
        children_right: ArrayLike,
        children_left: ArrayLike,
        node_id: int,
        clusters_distance: ArrayLike,
    ) -> Union[int, str]:
        """
        Function used recursively to do the final SQL code
        generation.
        """
        if isinstance(children_left[node_id], NoneType) and isinstance(
            children_right[node_id], NoneType
        ):
            return int(node_id)
        else:
            right_node = int(children_right[node_id])
            left_node = int(children_left[node_id])
            x = clusters_distance[left_node]
            th = clusters_distance[right_node]
            y0 = self._predict_tree_sql(
                children_right, children_left, left_node, clusters_distance
            )
            y1 = self._predict_tree_sql(
                children_right, children_left, right_node, clusters_distance
            )
            return f"(CASE WHEN {x} < {th} THEN {y0} ELSE {y1} END)"

    def predict_sql(self, X: ArrayLike) -> str:
        """
        Returns the SQL code needed to deploy the bisecting
        k-means model using its attributes.

        Parameters
        ----------
        X: ArrayLike
            The names or values of the input predictors.

        Returns
        -------
        str
            SQL code.
        """
        for c in self.clusters_:
            if len(X) != len(c):
                ValueError(
                    "The length of parameter 'X' must be the same as "
                    "the length of each cluster."
                )
        clusters_distance = []
        for c in self.clusters_:
            list_tmp = []
            for idx in range(len(X)):
                list_tmp += [f"POWER({X[idx]} - {c[idx]}, {self.p_})"]
            clusters_distance += [f"POWER({' + '.join(list_tmp)}, 1/{self.p_})"]
        is_null_x = " OR ".join([f"{x} IS NULL" for x in X])
        res = self._predict_tree_sql(
            self.children_right_, self.children_left_, 0, clusters_distance
        )
        sql_final = f"""
            (CASE 
                WHEN {is_null_x} 
                    THEN NULL 
                ELSE {res} 
            END)"""
        return clean_query(sql_final)

    # Trees Representation Methods.

    def to_graphviz(
        self,
        round_score: int = 2,
        percent: bool = False,
        vertical: bool = True,
        node_style: Optional[dict] = None,
        arrow_style: Optional[dict] = None,
        leaf_style: Optional[dict] = None,
    ) -> str:
        """
        Returns the code for a Graphviz tree.

        Parameters
        ----------
        round_score: int, optional
            The number of decimals to round the node's score to 0
            rounds to an integer.
        percent: bool, optional
            If set to True, the scores are returned as a percent.
        vertical: bool, optional
            If  set to True,  the function  generates a  vertical
            tree.
        node_style: dict, optional
            Dictionary  of options to customize each node of  the
            tree. For a list of options, see the Graphviz API:
            https://graphviz.org/doc/info/attrs.html
        arrow_style: dict, optional
            Dictionary  of options to customize each arrow of the
            tree. For a list of options, see the Graphviz API:
            https://graphviz.org/doc/info/attrs.html
        leaf_style: dict, optional
            Dictionary  of options to customize each leaf of  the
            tree. For a list of options, see the Graphviz API:
            https://graphviz.org/doc/info/attrs.html

        Returns
        -------
        str
            Graphviz code.
        """
        node_style, leaf_style = format_type(
            node_style, leaf_style, dtype=dict, na_out={"shape": "none"}
        )
        arrow_style = format_type(arrow_style, dtype=dict)
        n = len(self.children_left_)
        vertical = ""
        if not vertical:
            position = '\ngraph [rankdir = "LR"];'
        res = "digraph Tree{" + position
        for i in range(n):
            if (len(self.cluster_size_) == n) and (len(self.cluster_score_) == n):
                if "bgcolor" in node_style and (
                    self.children_left_[i] != self.children_right_[i]
                ):
                    color = node_style["bgcolor"]
                elif "color" in node_style and (
                    self.children_left_[i] != self.children_right_[i]
                ):
                    color = node_style["color"]
                elif self.children_left_[i] != self.children_right_[i]:
                    color = "#87cefa"
                elif "bgcolor" in leaf_style:
                    color = node_style["bgcolor"]
                elif "color" in leaf_style:
                    color = node_style["color"]
                else:
                    color = "#efc5b5"
                label = (
                    '<<table border="0" cellspacing="0"> <tr><td port="port1" '
                    f'border="1" bgcolor="{color}"><b> cluster_id: {i} </b></td></tr>'
                )
                if len(self.cluster_size_) == n:
                    label += '<tr><td port="port2" border="1" align="left">'
                    label += f" size: {self.cluster_size_[i]} </td></tr>"
                if len(self.cluster_score_) == n:
                    val = (
                        round(self.cluster_score_[i] * 100, round_score)
                        if percent
                        else round(self.cluster_score_[i], round_score)
                    )
                    if percent:
                        val = str(val) + "%"
                    label += '<tr><td port="port3" border="1" align="left"> '
                    label += f"score: {val} </td></tr>"
                label += "</table>>"
            else:
                label = f'"{i}"'
            if self.children_left_[i] != self.children_right_[i]:
                flat_dict_str = self._flat_dict(node_style)
            else:
                flat_dict_str = self._flat_dict(leaf_style)
            res += f"\n{i} [label={label}{flat_dict_str}]"
            if self.children_left_[i] != self.children_right_[i]:
                res += f'\n{i} -> {self.children_left_[i]} [label=""{self._flat_dict(arrow_style)}]'
                res += f'\n{i} -> {self.children_right_[i]} [label=""{self._flat_dict(arrow_style)}]'
        return res + "\n}"


class KPrototypes(Clustering):
    """
    :py:meth:`verticapy.machine_learning.memmodel.base.InMemoryModel`
    implementation of KPrototypes.

    Parameters
    ----------
    clusters: ArrayLike
        List of the model's cluster centers.
    p: int, optional
        The p corresponding to  one of the p-distances.
    gamma: float, optional
        Weighting  factor  for  categorical columns.  This
        determines  relative  importance of numerical  and
        categorical attributes.
    is_categorical: list / numpy.array, optional
        ArrayLike  of booleans to indicate whether  X[idx]
        is  a categorical  variable, where True  indicates
        categorical  and  False numerical.  If empty,  all
        the variables are considered categorical.

    .. note::

        :py:meth:`verticapy.machine_learning.memmodel.cluster.KPrototypes`
        algorithm allows you to use categorical variables directly
        without the need to encode them.

    Attributes
    ----------
    Attributes are identical to the input parameters, followed by an
    underscore ('_').

    Examples
    --------

    **Initalization**

    Import the required module.

    .. ipython:: python

        from verticapy.machine_learning.memmodel.cluster import KPrototypes

    A KPrototypes model is defined by its cluster centroids. Optionally
    you can also provide p value, gamma and provide information about
    categorical variables. In this example, we will use the following:

    .. ipython:: python

        clusters = [[0.5, 'high'], [1, 'low'], [100, 'high']]
        p = 2
        gamma = 1.0
        is_categorical = [0, 1]

    Let's create a
    :py:meth:`verticapy.machine_learning.memmodel.cluster.KPrototypes`
    model.

    .. ipython:: python

        model_kp = KPrototypes(clusters, p, gamma, is_categorical)

    Create a dataset.

    .. ipython:: python

        data = [[2, 'low']]

    **Making In-Memory Predictions**

    Use
    :py:meth:`verticapy.machine_learning.memmodel.cluster.KPrototypes.predict`
    method to do predictions.

    .. ipython:: python

        model_kp.predict(data)[0]

    .. note::

        :py:meth:`verticapy.machine_learning.memmodel.cluster.KPrototypes`
        assigns a cluster id to identify each cluster.
        In this example, cluster with centroid [0.5, 'high'] will have
        id = 0, with centroid [1,'low'] will have id = 1 and so on.
        :py:meth:`verticapy.machine_learning.memmodel.cluster.KPrototypes.predict`
        method returns the id of the predicted cluster.

    Use
    :py:meth:`verticapy.machine_learning.memmodel.cluster.KPrototypes.predict_proba`
    method to compute the predicted probabilities
    for each cluster.

    .. ipython:: python

        model_kp.predict_proba(data)

    Use
    :py:meth:`verticapy.machine_learning.memmodel.cluster.KPrototypes.transform`
    method to compute the distance from each cluster.

    .. ipython:: python

        model_kp.transform(data)

    **Deploy SQL Code**

    Let's use the following column names:

    .. ipython:: python

        cnames = ['col1', 'col2']

    Use
    :py:meth:`verticapy.machine_learning.memmodel.cluster.KPrototypes.predict_sql`
    method to get the SQL code needed to deploy the
    model using its attributes.

    .. ipython:: python

        model_kp.predict_sql(cnames)

    Use
    :py:meth:`verticapy.machine_learning.memmodel.cluster.KPrototypes.predict_proba_sql`
    method to get the SQL code needed to deploy the
    model that computes predicted probabilities.

    .. ipython:: python

        model_kp.predict_proba_sql(cnames)

    Use
    :py:meth:`verticapy.machine_learning.memmodel.cluster.KPrototypes.transform_sql`
    method to get the SQL code needed to deploy the
    model that computes distance from each cluster.

    .. ipython:: python

        model_kp.transform_sql(cnames)

    .. hint::

        This object can be pickled and used in any in-memory environment,
        just like `SKLEARN <https://scikit-learn.org/>`_ models.
    """

    # Properties.

    @property
    def object_type(self) -> Literal["KPrototypes"]:
        return "KPrototypes"

    @property
    def _attributes(self) -> list[str]:
        return ["clusters_", "p_", "gamma_", "is_categorical_"]

    # System & Special Methods.

    def __init__(
        self,
        clusters: ArrayLike,
        p: int = 2,
        gamma: float = 1.0,
        is_categorical: Optional[ArrayLike] = None,
    ) -> None:
        is_categorical = format_type(is_categorical, dtype=list)
        self.clusters_ = np.array(clusters)
        self.p_ = p
        self.gamma_ = gamma
        self.is_categorical_ = np.array(is_categorical)

    # Prediction / Transformation Methods - IN MEMORY.

    def _transform_row(self, X: ArrayLike) -> list:
        """
        Transforms and returns the distance to each cluster
        for one row.
        """
        distance = []
        for centroid in self.clusters_:
            distance_num, distance_cat = 0, 0
            for idx in range(len(X)):
                val, centroid_val = X[idx], centroid[idx]
                try:
                    val = float(val)
                    centroid_val = float(centroid_val)
                except (TypeError, ValueError):
                    pass
                if isinstance(centroid_val, str) or isinstance(centroid_val, NoneType):
                    distance_cat += abs(int(val == centroid_val) - 1)
                else:
                    distance_num += (val - centroid_val) ** self.p_
            distance_final = distance_num + self.gamma_ * distance_cat
            distance += [distance_final]
        return distance

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Transforms and returns the distance to each cluster.

        Parameters
        ----------
        X: ArrayLike
            The data on which to make the transformation.

        Returns
        -------
        numpy.array
            Transformed values.
        """
        return np.apply_along_axis(self._transform_row, 1, X)

    # Prediction / Transformation Methods - IN DATABASE.

    def transform_sql(self, X: ArrayLike) -> list[str]:
        """
        Transforms and returns the SQL distance to each cluster.

        Parameters
        ----------
        X: ArrayLike
            The names or values of the input predictors.

        Returns
        -------
        list
            SQL code.
        """
        if len(self.is_categorical_) == 0:
            is_categorical = np.array([True for i in range(len(X))])
        else:
            is_categorical = copy.deepcopy(self.is_categorical_)

        for c in self.clusters_:
            if not len(X) == len(c) == len(is_categorical):
                raise ValueError(
                    "The length of parameter 'X' must be the same as "
                    "the length of each cluster AND the categorical vector."
                )
        clusters_distance = []
        for c in self.clusters_:
            clusters_distance_num, clusters_distance_cat = [], []
            for idx in range(len(X)):
                if is_categorical[idx]:
                    c_i = str(c[idx]).replace("'", "''")
                    clusters_distance_cat += [f"ABS(({X[idx]} = '{c_i}')::int - 1)"]
                else:
                    clusters_distance_num += [f"POWER({X[idx]} - {c[idx]}, {self.p_})"]
            final_cluster_distance = ""
            if clusters_distance_num:
                final_cluster_distance += (
                    f"POWER({' + '.join(clusters_distance_num)}, 1 / {self.p_})"
                )
            if clusters_distance_cat:
                if clusters_distance_num:
                    final_cluster_distance += " + "
                final_cluster_distance += (
                    f"{self.gamma_} * ({' + '.join(clusters_distance_cat)})"
                )
            clusters_distance += [final_cluster_distance]
        return clusters_distance
