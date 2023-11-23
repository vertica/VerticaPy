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
import math
from collections.abc import Iterable
from typing import Literal, Optional, Union

import numpy as np

import verticapy._config.config as conf
from verticapy._typing import ArrayLike, NoneType
from verticapy._utils._sql._format import clean_query, format_magic, format_type

from verticapy.machine_learning.memmodel.base import InMemoryModel

if conf.get_import_success("graphviz"):
    import graphviz
    from graphviz import Source


class Tree(InMemoryModel):
    """
    Base Class for tree representation.
    """

    # System & Special Methods.

    ## Math

    @staticmethod
    def _heuristic_length(i: int) -> float:
        """
        Returns the heuristic length of the input integer.
        """
        GAMMA = 0.5772156649
        if i == 2:
            return 1.0
        elif i > 2:
            return 2 * (math.log(i - 1) + GAMMA) - 2 * (i - 1) / i
        else:
            return 0.0

    ## Tree Decision

    def _go_left(self, X: ArrayLike, node_id: int) -> bool:
        """
        Function used to decide either to go left
        or not.
        """
        th = self.threshold_[node_id]
        c = self.feature_[node_id]
        if isinstance(th, str):
            if str(X[c]) == th:
                return True
        else:
            if float(X[c]) < float(th):
                return True
        return False

    def _scoring_function(self, node_id: int = 0) -> float:
        """
        Must be implemented in the child class.
        """
        raise NotImplementedError

    def _scoring_function_proba(self, node_id: int = 0) -> float:
        """
        Must be implemented in the child class.
        """
        raise NotImplementedError

    # Prediction / Transformation Methods - IN MEMORY.

    def _predict_row(
        self, X: ArrayLike, node_id: int = 0, return_proba: bool = False
    ) -> float:
        """
        Function used recursively to get the tree
        prediction.
        """
        if self.children_left_[node_id] == self.children_right_[node_id]:
            if return_proba:
                return self._scoring_function_proba(node_id)
            else:
                return self._scoring_function(node_id)
        else:
            if self._go_left(X, node_id):
                children = self.children_left_[node_id]
            else:
                children = self.children_right_[node_id]
            return self._predict_row(X, children, return_proba)

    def _predict_row_proba(self, X: ArrayLike, node_id: int = 0) -> ArrayLike:
        """
        Function used  recursively to get the Tree
        prediction.
        """
        return self._predict_row(X, node_id, True)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predicts using the Binary Tree model.

        Parameters
        ----------
        X: ArrayLike
            The data on which to make the prediction.

        Returns
        -------
        numpy.array
            Predicted values.
        """
        return np.apply_along_axis(self._predict_row, 1, np.array(X))

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """
        Returns the model probabilities.

        Parameters
        ----------
        X: ArrayLike
            The data on which to make the prediction.

        Returns
        -------
        numpy.array
            Probabilities.
        """
        return np.apply_along_axis(self._predict_row_proba, 1, np.array(X))

    # Prediction / Transformation Methods - IN DATABASE.

    def _predict_tree_sql(
        self,
        X: ArrayLike,
        node_id: int = 0,
        return_proba: bool = False,
        class_id: int = 0,
    ) -> str:
        """
        Function used recursively to do the final SQL code generation.
        """
        if self.children_left_[node_id] == self.children_right_[node_id]:
            if return_proba:
                return self._scoring_function_proba(node_id)[class_id]
            else:
                return format_magic(self._scoring_function(node_id))
        else:
            if isinstance(self.threshold_[node_id], str):
                op = "="
                q = "'"
            else:
                op = "<"
                q = ""
            y0 = self._predict_tree_sql(
                X, self.children_left_[node_id], return_proba, class_id
            )
            y1 = self._predict_tree_sql(
                X, self.children_right_[node_id], return_proba, class_id
            )
            query = f"""
                (CASE 
                    WHEN {X[self.feature_[node_id]]} {op} {q}{self.threshold_[node_id]}{q} 
                    THEN {y0} ELSE {y1} 
                END)"""
            return clean_query(query)

    def predict_sql(self, X: ArrayLike) -> str:
        """
        Returns the SQL code needed to deploy the model.

        Parameters
        ----------
        X: ArrayLike
            The names or values of the input predictors.

        Returns
        -------
        str
            SQL code.
        """
        return self._predict_tree_sql(X)

    def predict_proba_sql(self, X: ArrayLike) -> list[str]:
        """
        Returns the SQL code needed to deploy the model
        probabilities.

        Parameters
        ----------
        X: ArrayLike
            The names or values of the input predictors.

        Returns
        -------
        str
            SQL code.
        """
        n = max(len(val) if not isinstance(val, NoneType) else 0 for val in self.value_)
        return [self._predict_tree_sql(X, 0, True, i) for i in range(n)]

    # Trees Representation Methods.

    @staticmethod
    def _default_colors() -> list[str]:
        """
        Default colors for the tree representation.
        """
        return [
            "#87cefa",
            "#efc5b5",
            "#d4ede3",
            "#f0ead2",
            "#d2cbaf",
            "#fcf0e5",
            "#f1ece2",
            "#98f6b0",
            "#d7d3a6",
            "#f8f8ff",
            "#d7cec5",
            "#f7d560",
            "#e5e7e9",
            "#ffa180",
            "#efc0fe",
            "#ffc5cb",
            "#eeeeaa",
            "#e7feff",
        ]

    @staticmethod
    def _flat_dict(d: dict) -> str:
        """
        Converts dictionary to string with a specific
        format used  during the Graphviz  convertion.
        """
        res = []
        for key in d:
            q = '"' if isinstance(d[key], str) else ""
            res += [f"{key}={q}{d[key]}{q}"]
        res = ", ".join(res)
        if res:
            res = f", {res}"
        return res

    @property
    def _get_output_kind(self) -> Literal["pred", "prob", "logodds", "contamination"]:
        """
        Returns the tree's output kind.
        """
        output_kind = "pred"
        if self.object_type == "BinaryTreeAnomaly":
            output_kind = "contamination"
        elif self.object_type == "BinaryTreeClassifier":
            output_kind = "prob"
            for val in self.value_:
                if isinstance(val, list) and not 0.99 < sum(val) <= 1.0:
                    output_kind = "logodds"
                    break
        return output_kind

    def to_graphviz(
        self,
        feature_names: Optional[ArrayLike] = None,
        classes_color: Optional[ArrayLike] = None,
        round_pred: int = 2,
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
        feature_names: ArrayLike, optional
            List of the names of each feature.
        classes_color: ArrayLike, optional
            Colors that represent the different classes.
        round_pred: int, optional
            The number of decimals to round the prediction to.
            0 rounds to an integer.
        percent: bool, optional
            If set to True, the probabilities are returned as
            percents.
        vertical: bool, optional
            If set to True, the function generates a vertical
            tree.
        node_style: dict, optional
            Dictionary  of options to customize each node  of
            the tree. For a list of options, see the Graphviz
            API: https://graphviz.org/doc/info/attrs.html
        arrow_style: dict, optional
            Dictionary  of options to customize each arrow of
            the tree. For a list of options, see the Graphviz
            API: https://graphviz.org/doc/info/attrs.html
        leaf_style: dict, optional
            Dictionary  of options to customize each leaf  of
            the tree. For a list of options, see the Graphviz
            API: https://graphviz.org/doc/info/attrs.html

        Returns
        -------
        str
            Graphviz code.
        """
        feature_names, classes_color = format_type(
            feature_names, classes_color, dtype=list
        )
        node_style = format_type(
            node_style, dtype=dict, na_out={"shape": "box", "style": "filled"}
        )
        arrow_style, leaf_style = format_type(arrow_style, leaf_style, dtype=dict)
        empty_color = False
        if len(classes_color) == 0:
            empty_color = True
            classes_color = self._default_colors()
        if not vertical:
            position = '\ngraph [rankdir = "LR"];'
        else:
            position = ""
        n, res = len(self.children_left_), "digraph Tree{" + position
        for i in range(n):
            if self.children_left_[i] != self.children_right_[i]:
                if feature_names:
                    name = feature_names[self.feature_[i]].replace('"', '\\"')
                else:
                    name = f"X{self.feature_[i]}"
                if isinstance(self.threshold_[i], str):
                    q, not_q = "=", "!="
                else:
                    q, not_q = "<=", ">"
                res += f'\n{i} [label="{name}"{self._flat_dict(node_style)}]'
                res += f'\n{i} -> {self.children_left_[i]} [label="{q} {self.threshold_[i]}"'
                res += f"{self._flat_dict(arrow_style)}]\n{i} -> {self.children_right_[i]} "
                res += f'[label="{not_q} {self.threshold_[i]}"{self._flat_dict(arrow_style)}]'
            else:
                color = ""
                if isinstance(self.value_[i], (int, float)):
                    label = f'"{self.value_[i]}"'
                elif hasattr(self, "psy"):
                    if not leaf_style:
                        leaf_style = {"shape": "none"}
                    if not empty_color:
                        color = classes_color[0]
                    else:
                        color = "#eeeeee"
                    anomaly_score = self.value_[i][0] + self._heuristic_length(
                        self.value_[i][1]
                    )
                    anomaly_score = -(anomaly_score) / self._heuristic_length(self.psy)
                    anomaly_score = float(2**anomaly_score)
                    if anomaly_score < 0.5:
                        color_anomaly = "#ffffff"
                    else:
                        color_anomaly = "#"
                        rgb = [255, 0, 0]
                        for idx in range(3):
                            rgb[idx] = int(
                                255 - 2 * (anomaly_score - 0.5) * (255 - rgb[idx])
                            )
                            color_anomaly += str(hex(rgb[idx]))[2:]
                    label = (
                        '<<table border="0" cellspacing="0"> <tr>'
                        f'<td port="port1" border="1" bgcolor="{color}"'
                        '><b> leaf </b></td></tr><tr><td port="port0" '
                        'border="1" align="left"> leaf_path_length: '
                        f'{self.value_[i][0]} </td></tr><tr><td port="port1"'
                        ' border="1" align="left"> training_row_count: '
                        f'{self.value_[i][1]} </td></tr><tr><td port="port2" '
                        f'border="1" align="left" bgcolor="{color_anomaly}">'
                        f" anomaly_score: {anomaly_score} </td></tr></table>>"
                    )
                else:
                    if not leaf_style:
                        leaf_style = {"shape": "none"}
                    if len(self.classes_) == 0:
                        classes_ = [k for k in range(len(self.value_[i]))]
                    else:
                        classes_ = copy.deepcopy(self.classes_)
                    color = classes_color[
                        (np.argmax(self.value_[i])) % len(classes_color)
                    ]
                    label = (
                        '<<table border="0" cellspacing="0"> <tr>'
                        f'<td port="port1" border="1" bgcolor="{color}">'
                        f"<b> prediction: {classes_[np.argmax(self.value_[i])]} "
                        "</b></td></tr>"
                    )
                    for j in range(len(self.value_[i])):
                        if percent:
                            val = str(round(self.value_[i][j] * 100, round_pred)) + "%"
                        else:
                            val = round(self.value_[i][j], round_pred)
                        label += f'<tr><td port="port{j}" border="1" align="left">'
                        label += (
                            f" {self._get_output_kind}({classes_[j]}): {val} </td></tr>"
                        )
                    label += "</table>>"
                res += f"\n{i} [label={label}{self._flat_dict(leaf_style)}]"
        return res + "\n}"

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
            Absolute  path to  save the image of the  tree.
        *args, **kwargs: Any, optional
            Arguments to pass to  the 'to_graphviz' method.

        Returns
        -------
        graphviz.Source
            graphviz object.
        """
        if not conf.get_import_success("graphviz"):
            raise ImportError(
                "The graphviz module doesn't seem to be "
                "installed in your environment.\nTo be "
                "able to use this method, you'll have to "
                "install it.\n[Tips] Run: 'pip3 install "
                "graphviz' in your terminal to install "
                "the module."
            )
        res = graphviz.Source(self.to_graphviz(*args, **kwargs))
        if pic_path:
            res.render(filename=pic_path)
        return res


class BinaryTreeRegressor(Tree):
    """
    :py:meth:`verticapy.machine_learning.memmodel.base.InMemoryModel`
    implementation  of  binary  trees  for regression.

    Parameters
    ----------
    children_left: ArrayLike
        A list of node IDs, where children_left[i] is the
        node id of the left child of node i.
    children_right: ArrayLike
        A list of node IDs, children_right[i] is the node
        id of the right child of node i.
    feature: ArrayLike
        A  list  of features,  where  feature[i]  is  the
        feature to split on for the internal node i.
    threshold: ArrayLike
        A  list of thresholds, where threshold[i] is  the
        threshold for the internal node i.
    value: ArrayLike
        Contains  the  constant  prediction value of each
        node.   If  used  for   classification   and
        return_proba is set to True,  each element of the
        list must be a sublist  with the probabilities of
        each class.

    Attributes
    ----------
    Attributes are identical to the input parameters, followed by an
    underscore ('_').

    Examples
    --------

    **Initalization**

    Import the required module.

    .. ipython:: python

        from verticapy.machine_learning.memmodel.tree import BinaryTreeRegressor

    A BinaryTreeRegressor model is defined by its left and right
    child node id's, feature and threshold value to split a node.
    Final values at leaf nodes are also required.

    Let's create a
    :py:meth:`verticapy.machine_learning.memmodel.tree.BinaryTreeRegressor`
    model:

    .. ipython:: python

        from verticapy.machine_learning.memmodel.tree import BinaryTreeRegressor

        # Different Attributes
        children_left = [1, 3, None, None, None]
        children_right = [2, 4, None, None, None]
        feature = [0, 1, None, None, None]
        threshold = ["female", 30, None, None, None]
        value = [None, None, 3, 11, 1993]

        # Building the Model
        model_btr = BinaryTreeRegressor(
            children_left = children_left,
            children_right = children_right,
            feature = feature,
            threshold = threshold,
            value = value,
        )

    Create a dataset.

    .. ipython:: python

        data = [["male", 100], ["female", 20], ["female", 50]]

    **Making In-Memory Predictions**

    Use
    :py:meth:`verticapy.machine_learning.memmodel.tree.BinaryTreeRegressor.predict`
    method to do predictions.

    .. ipython:: python

        model_btr.predict(data)

    **Deploy SQL Code**

    Let's use the following column names:

    .. ipython:: python

        cnames = ["sex", "fare"]

    Use
    :py:meth:`verticapy.machine_learning.memmodel.tree.BinaryTreeRegressor.predict_sql`
    method to get the SQL code needed to deploy the model using its attributes.

    .. ipython:: python

        model_btr.predict_sql(cnames)

    .. hint::

        This object can be pickled and used in any in-memory
        environment, just like `SKLEARN <https://scikit-learn.org/>`_ models.

    **Drawing Tree**

    Use
    :py:meth:`verticapy.machine_learning.memmodel.tree.BinaryTreeRegressor.to_graphviz`
    method to generate code for a
    `Graphviz <https://graphviz.org/>`_ tree.

    .. ipython:: python

        model_btr.to_graphviz()

    Use
    :py:meth:`verticapy.machine_learning.memmodel.tree.BinaryTreeRegressor.plot_tree`
    method to draw the input tree.

    .. code-block:: python

        model_btr.plot_tree()

    .. ipython:: python
        :suppress:

        res = model_btr.plot_tree()
        res.render(filename='figures/machine_learning_memmodel_tree_binarytreereg', format='png')

    .. image:: /../figures/machine_learning_memmodel_tree_binarytreereg.png

    .. important::

        :py:meth:`verticapy.machine_learning.memmodel.tree.BinaryTreeRegressor.plot_tree`
        requires the `Graphviz <https://graphviz.org/download/>`_ module.

    .. note::

        The above example is a very basic one. For
        other more detailed examples and customization
        options, please see :ref:`chart_gallery.tree`_
    """

    # Properties.

    @property
    def object_type(self) -> Literal["BinaryTreeRegressor"]:
        return "BinaryTreeRegressor"

    @property
    def _attributes(self) -> list[str]:
        return ["children_left_", "children_right_", "feature_", "threshold_", "value_"]

    # System & Special Methods.

    def __init__(
        self,
        children_left: ArrayLike,
        children_right: ArrayLike,
        feature: ArrayLike,
        threshold: ArrayLike,
        value: ArrayLike,
    ) -> None:
        self.children_left_ = np.array(children_left)
        self.children_right_ = np.array(children_right)
        self.feature_ = np.array(feature)
        self.threshold_ = np.array(threshold)
        self.value_ = np.array(value, dtype=object)

    # Prediction / Transformation Methods - IN MEMORY.

    def _scoring_function(self, node_id: int = 0) -> float:
        """
        Function used to take the final decision.
        """
        return self.value_[node_id]


class BinaryTreeAnomaly(Tree):
    """
    :py:meth:`verticapy.machine_learning.memmodel.base.InMemoryModel`
    implementation  of  binary  trees  for anomaly detection.

    Parameters
    ----------
    children_left: ArrayLike
        A list of node IDs, where children_left[i] is the
        node id of the left child of node i.
    children_right: ArrayLike
        A list of node IDs, children_right[i] is the node
        id of the right child of node i.
    feature: ArrayLike
        A  list  of features,  where  feature[i]  is  the
        feature to split on for the internal node i.
    threshold: ArrayLike
        A  list of thresholds, where threshold[i] is  the
        threshold for the internal node i.
    value: ArrayLike
        List of elements,  which are  null except for the
        leaves,  where each  leaf contains a list of  two
        elements   representing  the  number  of   points
        classified as outliers and those that are not.
    psy: int, optional
        Sampling  size used to  compute the final  score.

    Attributes
    ----------
    Attributes are identical to the input parameters, followed by an
    underscore ('_').

    Examples
    --------

    **Initalization**

    Import the required module.

    .. ipython:: python

        from verticapy.machine_learning.memmodel.tree import BinaryTreeAnomaly

    A BinaryTreeAnomaly model is defined by its left and right
    child node id's, feature and threshold value to split a node.
    Final values at leaf nodes are also required.
    Let's create a
    :py:meth:`verticapy.machine_learning.memmodel.tree.BinaryTreeAnomaly`
    model:

    .. ipython:: python

        from verticapy.machine_learning.memmodel.tree import BinaryTreeAnomaly

        # Different Attributes
        children_left = [1, 3, None, None, None]
        children_right = [2, 4, None, None, None]
        feature = [0, 1, None, None, None]
        threshold = ["female", 30, None, None, None]
        value = [None, None, [2, 10], [3, 4], [7, 8]]

        # Building the Model
        model_bta = BinaryTreeAnomaly(
            children_left = children_left,
            children_right = children_right,
            feature = feature,
            threshold = threshold,
            value = value,
            psy = 100,
        )

    .. important::

        The parameter ``psy`` corresponds to the sampling
        size used to compute the final  score. This parameter
        is needed to compute the final score. A wrong parameter
        can lead to a wrong computation.

    .. note::

        For ``BinaryTreeAnomaly``, the parameter ``value``
        represent the number of points classified as outliers
        and those that are not. Leaves are then a list of
        two elements.

    Create a dataset.

    .. ipython:: python

        data = [["male", 100], ["female", 20], ["female", 50]]

    **Making In-Memory Predictions**

    Use
    :py:meth:`verticapy.machine_learning.memmodel.tree.BinaryTreeAnomaly.predict`
    method to do predictions.

    .. ipython:: python

        model_bta.predict(data)

    **Deploy SQL Code**

    Let's use the following column names:

    .. ipython:: python

        cnames = ["sex", "fare"]

    Use
    :py:meth:`verticapy.machine_learning.memmodel.tree.BinaryTreeAnomaly.predict_sql`
    method to get the SQL code needed to deploy the model
    using its attributes.

    .. ipython:: python

        model_bta.predict_sql(cnames)

    .. hint::

        This object can be pickled and used in any in-memory
        environment, just like `SKLEARN <https://scikit-learn.org/>`_ models.

    **Drawing Tree**

    Use
    :py:meth:`verticapy.machine_learning.memmodel.tree.BinaryTreeAnomaly.to_graphviz`
    method to generate code for a
    `Graphviz <https://graphviz.org/>`_ tree.

    .. ipython:: python

        model_bta.to_graphviz()

    Use
    :py:meth:`verticapy.machine_learning.memmodel.tree.BinaryTreeAnomaly.plot_tree`
    method to draw the input tree.

    .. code-block:: python

        model_bta.plot_tree()

    .. ipython:: python
        :suppress:

        res = model_bta.plot_tree()
        res.render(filename='figures/machine_learning_memmodel_tree_binarytreeanomaly', format='png')

    .. image:: /../figures/machine_learning_memmodel_tree_binarytreeanomaly.png

    .. important::

        :py:meth:`verticapy.machine_learning.memmodel.tree.BinaryTreeAnomaly.plot_tree`
        requires the `Graphviz <https://graphviz.org/download/>`_ module.

    .. note::

        The above example is a very basic one. For
        other more detailed examples and customization
        options, please see :ref:`chart_gallery.tree`_
    """

    # Properties.

    @property
    def object_type(self) -> Literal["BinaryTreeAnomaly"]:
        return "BinaryTreeAnomaly"

    @property
    def _attributes(self) -> list[str]:
        return [
            "children_left_",
            "children_right_",
            "feature_",
            "threshold_",
            "value_",
            "psy_",
        ]

    # System & Special Methods.

    def __init__(
        self,
        children_left: ArrayLike,
        children_right: ArrayLike,
        feature: ArrayLike,
        threshold: ArrayLike,
        value: ArrayLike,
        psy: int = 1,
    ) -> None:
        self.children_left_ = np.array(children_left)
        self.children_right_ = np.array(children_right)
        self.feature_ = np.array(feature)
        self.threshold_ = np.array(threshold)
        self.value_ = np.array(value, dtype=object)
        self.psy = psy

    # Prediction / Transformation Methods - IN MEMORY.

    def _scoring_function(self, node_id: int = 0) -> float:
        """
        Function used to take the final decision.
        """
        res = self.value_[node_id][0] + self._heuristic_length(self.value_[node_id][1])
        return res / self._heuristic_length(self.psy)


class BinaryTreeClassifier(Tree):
    """
    :py:meth:`verticapy.machine_learning.memmodel.base.InMemoryModel`
    implementation  of  binary  trees  for classification.

    Parameters
    ----------
    children_left: ArrayLike
        A list of node IDs, where children_left[i] is the
        node id of the left child of node i.
    children_right: ArrayLike
        A list of node IDs, children_right[i] is the node
        id of the right child of node i.
    feature: ArrayLike
        A  list  of features,  where  feature[i]  is  the
        feature to split on for the internal node i.
    threshold: ArrayLike
        A  list of thresholds, where threshold[i] is  the
        threshold for the internal node i.
    value: ArrayLike
        Contains  the  constant  prediction value of each
        node.   If  used  for   classification   and
        return_proba is set to True,  each element of the
        list must be a sublist  with the probabilities of
        each class.
    classes: ArrayLike, optional
        The classes for the binary tree model.

    Attributes
    ----------
    Attributes are identical to the input parameters, followed by an
    underscore ('_').

    Examples
    --------

    **Initalization**

    Import the required module.

    .. ipython:: python

        from verticapy.machine_learning.memmodel.tree import BinaryTreeClassifier

    A BinaryTreeClassifier tree model is defined by its left
    and right child node id's, feature and threshold value to
    split a node. Final values at leaf nodes and name of classes
    are also required. Let's create a
    :py:meth:`verticapy.machine_learning.memmodel.tree.BinaryTreeClassifier`
    model.

    .. ipython:: python

        # Different Attributes
        children_left = [1, 3, None, None, None]
        children_right = [2, 4, None, None, None]
        feature = [0, 1, None, None, None]
        threshold = ["female", 30, None, None, None]
        value = [None, None, [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]]
        classes = ["a", "b", "c"]

        # Building the Model
        model_btc = BinaryTreeClassifier(
            children_left = children_left,
            children_right = children_right,
            feature = feature,
            threshold = threshold,
            value = value,
            classes = classes,
        )

    Create a dataset.

    .. ipython:: python

        data = [["male", 100], ["female", 20], ["female", 50]]


    **Making In-Memory Predictions**

    Use
    :py:meth:`verticapy.machine_learning.memmodel.tree.BinaryTreeClassifier.predict`
    method to do predictions.

    .. ipython:: python

        model_btc.predict(data)

    Use
    :py:meth:`verticapy.machine_learning.memmodel.tree.BinaryTreeClassifier.predict_proba`
    method to compute the predicted probabilities for each class.

    .. ipython:: python

        model_btc.predict_proba(data)

    **Deploy SQL Code**

    Let's use the following column names:

    .. ipython:: python

        cnames = ["sex", "fare"]

    Use
    :py:meth:`verticapy.machine_learning.memmodel.tree.BinaryTreeClassifier.predict_sql`
    method to get the SQL code needed to deploy the model
    using its attributes.

    .. ipython:: python

        model_btc.predict_sql(cnames)

    Use
    :py:meth:`verticapy.machine_learning.memmodel.tree.BinaryTreeClassifier.predict_proba_sql`
    method to get the SQL code needed to deploy the model that
    computes predicted probabilities.

    .. ipython:: python

        model_btc.predict_proba_sql(cnames)

    .. hint::

        This object can be pickled and used in any in-memory
        environment, just like `SKLEARN <https://scikit-learn.org/>`_ models.

    **Drawing Tree**

    Use
    :py:meth:`verticapy.machine_learning.memmodel.tree.BinaryTreeClassifier.to_graphviz`
    method to generate code for a
    `Graphviz <https://graphviz.org/>`_ tree.

    .. ipython:: python

        model_btc.to_graphviz()

    Use
    :py:meth:`verticapy.machine_learning.memmodel.tree.BinaryTreeClassifier.plot_tree`
    method to draw the input tree.

    .. code-block:: python

        model_btc.plot_tree()

    .. ipython:: python
        :suppress:

        res = model_btc.plot_tree()
        res.render(filename='figures/machine_learning_memmodel_tree_binarytreeclassifier', format='png')

    .. image:: /../figures/machine_learning_memmodel_tree_binarytreeclassifier.png

    .. important::

        :py:meth:`verticapy.machine_learning.memmodel.tree.BinaryTreeClassifier.plot_tree`
        requires the `Graphviz <https://graphviz.org/download/>`_ module.

    .. note::

        The above example is a very basic one. For
        other more detailed examples and customization
        options, please see :ref:`chart_gallery.tree`_
    """

    # Properties.

    @property
    def object_type(self) -> Literal["BinaryTreeClassifier"]:
        return "BinaryTreeClassifier"

    @property
    def _attributes(self) -> list[str]:
        return [
            "children_left_",
            "children_right_",
            "feature_",
            "threshold_",
            "value_",
            "classes_",
        ]

    # System & Special Methods.

    def __init__(
        self,
        children_left: ArrayLike,
        children_right: ArrayLike,
        feature: ArrayLike,
        threshold: ArrayLike,
        value: ArrayLike,
        classes: Optional[ArrayLike] = None,
    ) -> None:
        classes = format_type(classes, dtype=list)
        self.children_left_ = np.array(children_left)
        self.children_right_ = np.array(children_right)
        self.feature_ = np.array(feature)
        self.threshold_ = np.array(threshold)
        self.value_ = copy.deepcopy(value)
        self.classes_ = np.array(classes)

    # Prediction / Transformation Methods - IN MEMORY.

    def _scoring_function(self, node_id: int = 0) -> float:
        """
        Function used to take the final decision.
        """
        if isinstance(self.classes_, Iterable) and len(self.classes_) > 0:
            return self.classes_[np.argmax(self.value_[node_id])]
        else:
            return np.argmax(self.value_[node_id])

    def _scoring_function_proba(self, node_id: int = 0) -> float:
        """
        Function used to take the final decision.
        """
        return self.value_[node_id]


class NonBinaryTree(Tree):
    """
    :py:meth:`verticapy.machine_learning.memmodel.base.InMemoryModel`
    implementation of non-binary trees.

    Parameters
    ----------
    tree: dict
        A  NonBinaryTree  tree.  NonBinaryTrees  can
        be generated with the vDataFrame.chaid method.
    classes: ArrayLike, optional
        The classes for the non-binary tree model.

    Attributes
    ----------
    Attributes are identical to the input parameters, followed by an
    underscore ('_').

    Examples
    --------

    **Initalization**

    Import the required module.

    .. ipython:: python

        from verticapy.machine_learning.memmodel.tree import NonBinaryTree

    A NonBinaryTree tree model is defined by the non-binary
    decision tree and name of classes.

    We will first generate a non-binary tree using
    :py:meth:`verticapy.vDataFrame.chaid` method.
    For this example, we will use the Titanic dataset.

    .. code-block:: python

        import verticapy.datasets as vpd

        data = vpd.load_titanic()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_titanic.html

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
        data = vpd.load_titanic()

    Lets create a non-binary tree using
    :py:meth:`verticapy.vDataFrame.chaid` method.

    .. ipython:: python

        tree = data.chaid("survived", ["sex", "fare"]).tree_

    Our non-binary tree is ready, we will now provide
    information about classes and create a
    :py:meth:`verticapy.machine_learning.memmodel.tree.NonBinaryTree`
    model.

    .. ipython:: python

        classes = ["a", "b"]
        model_nbt = NonBinaryTree(tree, classes)

    Create a dataset.

    .. ipython:: python

        data = [["male", 100], ["female", 20], ["female", 50]]

    **Making In-Memory Predictions**

    Use
    :py:meth:`verticapy.machine_learning.memmodel.tree.NonBinaryTree.predict`
    method to do predictions.

    .. ipython:: python

        model_nbt.predict(data)

    Use
    :py:meth:`verticapy.machine_learning.memmodel.tree.NonBinaryTree.predict_proba`
    method to compute the predicted probabilities for each class.

    .. ipython:: python

        model_nbt.predict_proba(data)

    **Deploy SQL Code**

    Let's use the following column names:

    .. ipython:: python

        cnames = ["sex", "fare"]

    Use
    :py:meth:`verticapy.machine_learning.memmodel.tree.NonBinaryTree.predict_sql`
    method to get the SQL code needed to deploy the model
    using its attributes.

    .. ipython:: python

        model_nbt.predict_sql(cnames)

    Use
    :py:meth:`verticapy.machine_learning.memmodel.tree.NonBinaryTree.predict_proba_sql`
    method to get the SQL code needed to deploy the
    model that computes predicted probabilities.

    .. ipython:: python

        model_nbt.predict_proba_sql(cnames)

    .. hint::

        This object can be pickled and used in any in-memory
        environment, just like `SKLEARN <https://scikit-learn.org/>`_ models.

    **Drawing Tree**

    Use
    :py:meth:`verticapy.machine_learning.memmodel.tree.NonBinaryTree.to_graphviz`
    method to generate code for a
    `Graphviz <https://graphviz.org/>`_ tree.

    .. ipython:: python

        model_nbt.to_graphviz()

    Use
    :py:meth:`verticapy.machine_learning.memmodel.tree.NonBinaryTree.plot_tree`
    method to draw the input tree.

    .. code-block:: python

        model_nbt.plot_tree()

    .. ipython:: python
        :suppress:

        res = model_nbt.plot_tree()
        res.render(filename='figures/machine_learning_memmodel_tree_NonBinaryTree', format='png')

    .. image:: /../figures/machine_learning_memmodel_tree_NonBinaryTree.png

    .. important::

        :py:meth:`verticapy.machine_learning.memmodel.tree.NonBinaryTree.plot_tree`
        requires the `Graphviz <https://graphviz.org/download/>`_ module.

    .. note::

        The above example is a very basic one. For
        other more detailed examples and customization
        options, please see :ref:`chart_gallery.tree`_
    """

    # Properties.

    @property
    def object_type(self) -> Literal["NonBinaryTree"]:
        return "NonBinaryTree"

    @property
    def _attributes(self) -> list[str]:
        return ["tree_", "classes_"]

    # System & Special Methods.

    def __init__(self, tree: dict, classes: Optional[ArrayLike] = None) -> None:
        classes = format_type(classes, dtype=list)
        self.tree_ = copy.deepcopy(tree)
        self.classes_ = np.array(classes)

    # Prediction / Transformation Methods - IN MEMORY.

    def _predict_tree(
        self, X: ArrayLike, tree: dict, return_proba: bool = False
    ) -> Union[ArrayLike, str, int]:
        """
        Function used recursively to get the tree
        prediction.
        """
        if tree["is_leaf"]:
            if return_proba:
                return tree["prediction"]
            elif isinstance(self.classes_, Iterable) and len(self.classes_) > 0:
                return self.classes_[np.argmax(tree["prediction"])]
            else:
                return np.argmax(tree["prediction"])
        else:
            for c in tree["children"]:
                if (
                    tree["split_is_numerical"]
                    and (float(X[tree["split_predictor_idx"]]) <= float(c))
                ) or (
                    not tree["split_is_numerical"]
                    and (X[tree["split_predictor_idx"]] == c)
                ):
                    return self._predict_tree(X, tree["children"][c], return_proba)

    def _predict_row(self, X: ArrayLike) -> Union[ArrayLike, str, int]:
        """
        Function used recursively to get the Tree
        prediction for one row.
        """
        return self._predict_tree(X, self.tree_, False)

    def _predict_proba_row(self, X: ArrayLike) -> ArrayLike:
        """
        Function used recursively to get the Tree
        probabilities for one row.
        """
        return self._predict_tree(X, self.tree_, True)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predicts using the CHAID model.

        Parameters
        ----------
        X: ArrayLike
            The data on which to make the prediction.

        Returns
        -------
        numpy.array
            Predicted values.
        """
        return np.apply_along_axis(self._predict_row, 1, np.array(X))

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """
        Returns probabilities  using the CHAID model.

        Parameters
        ----------
        X: ArrayLike
            The data on which to make the prediction.

        Returns
        -------
        numpy.array
            Probabilities.
        """
        return np.apply_along_axis(self._predict_proba_row, 1, np.array(X))

    # Prediction / Transformation Methods - IN DATABASE.

    def _predict_tree_sql(
        self, X: ArrayLike, tree: dict, class_id: int = 0, return_proba: bool = False
    ) -> str:
        """
        Function used recursively to do the final SQL
        code generation.
        """
        if tree["is_leaf"]:
            if return_proba:
                return tree["prediction"][class_id]
            elif isinstance(self.classes_, Iterable) and len(self.classes_) > 0:
                res = self.classes_[np.argmax(tree["prediction"])]
                return format_magic(res)
            else:
                return np.argmax(tree["prediction"])
        else:
            res = "(CASE "
            for c in tree["children"]:
                x = X[tree["split_predictor_idx"]]
                y = self._predict_tree_sql(
                    X, tree["children"][c], class_id, return_proba
                )
                if tree["split_is_numerical"]:
                    res += f"WHEN {x} <= {float(c)} THEN {y} "
                else:
                    res += f"WHEN {x} = '{c}' THEN {y} "
            return res + "ELSE NULL END)"

    def predict_sql(self, X: ArrayLike) -> str:
        """
        Returns the SQL code needed to deploy the model
        using its attributes.

        Parameters
        ----------
        X: list / numpy.array
            The names or values of the input predictors.

        Returns
        -------
        str
            SQL code.
        """
        return self._predict_tree_sql(X, self.tree_)

    def predict_proba_sql(self, X: ArrayLike) -> list[str]:
        """
        Returns the SQL code needed to deploy the model
        probabilities.

        Parameters
        ----------
        X: list / numpy.array
            The names or values of the input predictors.

        Returns
        -------
        list
            SQL code.
        """
        n = len(self.classes_)
        return [self._predict_tree_sql(X, self.tree_, i, True) for i in range(n)]

    # Trees Representation Methods.

    def _to_graphviz_tree(
        self,
        tree: dict,
        classes_color: Optional[list] = None,
        round_pred: int = 2,
        percent: bool = False,
        vertical: bool = True,
        node_style: Optional[dict] = None,
        arrow_style: Optional[dict] = None,
        leaf_style: Optional[dict] = None,
        process: bool = True,
    ) -> str:
        """
        Returns the code for a Graphviz tree.
        """
        node_style, arrow_style, leaf_style = format_type(
            node_style, arrow_style, leaf_style, dtype=dict
        )
        classes_color = format_type(classes_color, dtype=list)
        if process and len(classes_color) == 0:
            classes_color = self._default_colors()
        if tree["is_leaf"]:
            color = ""
            if isinstance(tree["prediction"], float):
                label = f'"{tree["prediction"]}"'
            else:
                if not leaf_style:
                    leaf_style = {"shape": "none"}
                if len(self.classes_) == 0:
                    classes_ = [k for k in range(len(tree["prediction"]))]
                else:
                    classes_ = copy.deepcopy(self.classes_)
                color = classes_color[
                    (np.argmax(tree["prediction"])) % len(classes_color)
                ]
                label = (
                    '<<table border="0" cellspacing="0"><tr>'
                    f'<td port="port1" border="1" bgcolor="{color}">'
                    f"<b> prediction: {classes_[np.argmax(tree['prediction'])]}"
                    " </b></td></tr>"
                )
                for j in range(len(tree["prediction"])):
                    if percent:
                        val = round(tree["prediction"][j] * 100, round_pred)
                        val = str(val) + "%"
                    else:
                        val = round(tree["prediction"][j], round_pred)
                    label += f'<tr><td port="port{j}" border="1" align="left"> '
                    label += f"prob({classes_[j]}): {val} </td></tr>"
                label += "</table>>"
            return f"{tree['node_id']} [label={label}{self._flat_dict(leaf_style)}]"
        else:
            res = ""
            for c in tree["children"]:
                q = "=" if isinstance(c, str) else "<="
                split_predictor = tree["split_predictor"].replace('"', '\\"')
                res += f'\n{tree["node_id"]} [label="{split_predictor}"{self._flat_dict(node_style)}]'
                if tree["children"][c]["is_leaf"] or tree["children"][c]["children"]:
                    res += f'\n{tree["node_id"]} -> {tree["children"][c]["node_id"]}'
                    res += f'[label="{q} {c}"{self._flat_dict(arrow_style)}]'
                res += self._to_graphviz_tree(
                    tree=tree["children"][c],
                    classes_color=classes_color,
                    round_pred=round_pred,
                    percent=percent,
                    vertical=vertical,
                    node_style=node_style,
                    arrow_style=arrow_style,
                    leaf_style=leaf_style,
                    process=False,
                )
            if process:
                if not vertical:
                    position = '\ngraph [rankdir = "LR"];'
                else:
                    position = ""
                res = "digraph Tree{" + position + res + "\n}"
            return res

    def to_graphviz(
        self,
        classes_color: Optional[ArrayLike] = None,
        round_pred: int = 2,
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
        classes_color: ArrayLike, optional
            Colors that represent the different classes.
        round_pred: int, optional
            The number of decimals to round the prediction to.
            0 rounds to an integer.
        percent: bool, optional
            If set to True, the probabilities are returned as
            percents.
        vertical: bool, optional
            If set to True, the function generates a vertical
            tree.
        node_style: dict, optional
            Dictionary  of options to customize each node  of
            the tree. For a list of options, see the Graphviz
            API: https://graphviz.org/doc/info/attrs.html
        arrow_style: dict, optional
            Dictionary  of options to customize each arrow of
            the tree. For a list of options, see the Graphviz
            API: https://graphviz.org/doc/info/attrs.html
        leaf_style: dict, optional
            Dictionary  of options to customize each leaf  of
            the tree. For a list of options, see the Graphviz
            API: https://graphviz.org/doc/info/attrs.html

        Returns
        -------
        str
          Graphviz code.
        """
        return self._to_graphviz_tree(
            self.tree_,
            classes_color,
            round_pred,
            percent,
            vertical,
            node_style,
            arrow_style,
            leaf_style,
        )
