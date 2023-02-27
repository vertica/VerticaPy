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
from collections.abc import Iterable
from typing import Literal, Union
import numpy as np

import verticapy._config.config as conf
from verticapy._typing import ArrayLike
from verticapy._utils._sql._format import clean_query, format_magic
from verticapy._utils.math import heuristic_length

from verticapy.machine_learning.memmodel.base import InMemoryModel

if conf._get_import_success("graphviz"):
    import graphviz


class Tree(InMemoryModel):
    @staticmethod
    def _default_colors() -> list[str]:
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
        Converts dictionary to string with a specific format
        used during the Graphviz convertion.
        """
        res = []
        for key in d:
            q = '"' if isinstance(d[key], str) else ""
            res += [f"{key}={q}{d[key]}{q}"]
        res = ", ".join(res)
        if res:
            res = f", {res}"
        return res

    def _go_left(self, X: ArrayLike, node_id: int):
        """
        Function used to decide either to go left or not.
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

    def _predict_row(
        self, X: ArrayLike, node_id: int = 0, return_proba: bool = False
    ) -> float:
        """
        Function used recursively to get the Tree prediction.
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
        Function used recursively to get the Tree prediction.
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

    def _predict_tree_sql(
        self,
        X: ArrayLike,
        node_id: int = 0,
        return_proba: bool = False,
        class_id: int = 0,
    ):
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
        n = max([len(val) if val != None else 0 for val in self.value_])
        return [self._predict_tree_sql(X, 0, True, i) for i in range(n)]

    def to_graphviz(
        self,
        feature_names: ArrayLike = [],
        classes_color: ArrayLike = [],
        prefix_pred: str = "prob",
        round_pred: int = 2,
        percent: bool = False,
        vertical: bool = True,
        node_style: dict = {},
        arrow_style: dict = {},
        leaf_style: dict = {},
    ):
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
            Dictionary of options to customize each node of 
            the tree. For a list of options, see the Graphviz 
            API: https://graphviz.org/doc/info/attrs.html
        arrow_style: dict, optional
            Dictionary of options to customize each arrow of 
            the tree. For a list of options, see the Graphviz 
            API: https://graphviz.org/doc/info/attrs.html
        leaf_style: dict, optional
            Dictionary of options to customize each leaf of 
            the tree. For a list of options, see the Graphviz 
            API: https://graphviz.org/doc/info/attrs.html

        Returns
        -------
        str
            Graphviz code.
        """
        empty_color = False
        if len(classes_color) == 0:
            empty_color = True
            classes_color = self._default_colors()
        if not (vertical):
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
                if isinstance(self.value_[i], float):
                    label = f'"{self.value_[i]}"'
                elif hasattr(self, "psy"):
                    if not (leaf_style):
                        leaf_style = {"shape": "none"}
                    if not (empty_color):
                        color = classes_color[0]
                    else:
                        color = "#eeeeee"
                    anomaly_score = value[i][0] + heuristic_length(value[i][1])
                    anomaly_score = -(anomaly_score) / heuristic_length(self.psy)
                    anomaly_score = float(2 ** anomaly_score)
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
                    if not (leaf_style):
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
                        label += f" {prefix_pred}({classes_[j]}): {val} </td></tr>"
                    label += "</table>>"
                res += f"\n{i} [label={label}{self._flat_dict(leaf_style)}]"
        return res + "\n}"

    def plot_tree(
        self, pic_path: str = "", *argv, **kwds,
    ):
        """
        Draws the input tree. Requires the graphviz module.

        Parameters
        ----------
        pic_path: str, optional
            Absolute path to save the image of the tree.
        *argv, **kwds: Any, optional
            Arguments to pass to the 'to_graphviz' method.

        Returns
        -------
        graphviz.Source
            graphviz object.
        """
        if not (conf._get_import_success("graphviz")):
            raise ImportError(
                "The graphviz module doesn't seem to be "
                "installed in your environment.\nTo be "
                "able to use this method, you'll have to "
                "install it.\n[Tips] Run: 'pip3 install "
                "graphviz' in your terminal to install "
                "the module."
            )
        res = graphviz.Source(self.to_graphviz(*argv, **kwds))
        if pic_path:
            res.view(pic_path)
        return res


class BinaryTreeRegressor(Tree):
    """
    InMemoryModel Implementation of Binary Trees for Regression.

    Parameters
    ----------
    children_left: ArrayLike
        A list of node IDs, where children_left[i] is the 
        node id of the left child of node i.
    children_right: ArrayLike
        A list of node IDs, children_right[i] is the node 
        id of the right child of node i.
    feature: ArrayLike
        A list of features, where feature[i] is the feature 
        to split on for the internal node i.
    threshold: ArrayLike
        A list of thresholds, where threshold[i] is the 
        threshold for the internal node i.
    value: ArrayLike
        Contains the constant prediction value of each node. 
        If used for classification and if return_proba is 
        set to True, each element of the list must be a sublist 
        with the probabilities of each class.
    """

    @property
    def _object_type(self) -> Literal["BinaryTreeRegressor"]:
        return "BinaryTreeRegressor"

    @property
    def _attributes(
        self,
    ) -> Literal[
        "children_left_", "children_right_", "feature_", "threshold_", "value_"
    ]:
        return ["children_left_", "children_right_", "feature_", "threshold_", "value_"]

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
        return None

    def _scoring_function(self, node_id: int = 0) -> float:
        """
        Function used to take the final decision.
        """
        return self.value_[node_id]


class BinaryTreeAnomaly(Tree):
    """
    InMemoryModel Implementation of Binary Trees for Anomaly
    Detection.

    Parameters
    ----------
    children_left: ArrayLike
        A list of node IDs, where children_left[i] is the 
        node id of the left child of node i.
    children_right: ArrayLike
        A list of node IDs, children_right[i] is the node 
        id of the right child of node i.
    feature: ArrayLike
        A list of features, where feature[i] is the feature 
        to split on for the internal node i.
    threshold: ArrayLike
        A list of thresholds, where threshold[i] is the 
        threshold for the internal node i.
    value: ArrayLike
        Contains the constant prediction value of each node. 
        If used for classification and if return_proba is 
        set to True, each element of the list must be a sublist 
        with the probabilities of each class.
    psy: int, optional
        Sampling size used to compute the final score.
    """

    @property
    def _object_type(self) -> Literal["BinaryTreeAnomaly"]:
        return "BinaryTreeAnomaly"

    @property
    def _attributes(
        self,
    ) -> Literal[
        "children_left_", "children_right_", "feature_", "threshold_", "value_", "psy_"
    ]:
        return [
            "children_left_",
            "children_right_",
            "feature_",
            "threshold_",
            "value_",
            "psy_",
        ]

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
        return None

    def _scoring_function(self, node_id: int = 0) -> float:
        """
        Function used to take the final decision.
        """
        res = self.value_[node_id][0] + heuristic_length(self.value_[node_id][1])
        return res / heuristic_length(self.psy)


class BinaryTreeClassifier(Tree):
    """
    InMemoryModel Implementation of Binary Trees for Classification.

    Parameters
    ----------
    children_left: ArrayLike
        A list of node IDs, where children_left[i] is the 
        node id of the left child of node i.
    children_right: ArrayLike
        A list of node IDs, children_right[i] is the node 
        id of the right child of node i.
    feature: ArrayLike
        A list of features, where feature[i] is the feature 
        to split on for the internal node i.
    threshold: ArrayLike
        A list of thresholds, where threshold[i] is the 
        threshold for the internal node i.
    value: ArrayLike
        Contains the constant prediction value of each node. 
        If used for classification and if return_proba is 
        set to True, each element of the list must be a sublist 
        with the probabilities of each class.
    classes: ArrayLike, optional
        The classes for the binary tree model.
    """

    @property
    def _object_type(self) -> Literal["BinaryTreeClassifier"]:
        return "BinaryTreeClassifier"

    @property
    def _attributes(
        self,
    ) -> Literal[
        "children_left_",
        "children_right_",
        "feature_",
        "threshold_",
        "value_",
        "classes_",
    ]:
        return [
            "children_left_",
            "children_right_",
            "feature_",
            "threshold_",
            "value_",
            "classes_",
        ]

    def __init__(
        self,
        children_left: ArrayLike,
        children_right: ArrayLike,
        feature: ArrayLike,
        threshold: ArrayLike,
        value: ArrayLike,
        classes: ArrayLike = [],
    ) -> None:
        self.children_left_ = np.array(children_left)
        self.children_right_ = np.array(children_right)
        self.feature_ = np.array(feature)
        self.threshold_ = np.array(threshold)
        self.value_ = copy.deepcopy(value)
        self.classes_ = np.array(classes)
        return None

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
    InMemoryModel Implementation of Non Binary Trees.

    Parameters
    ----------
    tree: dict
        A NonBinaryTree tree. Non Binary Trees can 
        be generated with the vDataFrame.chaid method.
    classes: ArrayLike, optional
        The p corresponding to the one of the p-distances.
    """

    @property
    def _object_type(self) -> Literal["NonBinaryTree"]:
        return "NonBinaryTree"

    @property
    def _attributes(self) -> Literal["tree_", "classes_"]:
        return ["tree_", "classes_"]

    def __init__(self, tree: dict, classes: ArrayLike = []) -> None:
        self.tree_ = copy.deepcopy(tree)
        self.classes_ = np.array(classes)
        return None

    def _predict_tree(
        self, X: ArrayLike, tree: dict, return_proba: bool = False
    ) -> Union[ArrayLike, str, int]:
        """
        Function used recursively to get the Tree prediction.
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
                    not (tree["split_is_numerical"])
                    and (X[tree["split_predictor_idx"]] == c)
                ):
                    return self._predict_tree(X, tree["children"][c], return_proba)
            return None

    def _predict_row(self, X: ArrayLike) -> Union[ArrayLike, str, int]:
        """
        Function used recursively to get the Tree prediction 
        for one row.
        """
        return self._predict_tree(X, self.tree_, False)

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

    def _predict_proba_row(self, X: ArrayLike) -> ArrayLike:
        """
        Function used recursively to get the Tree probabilities
        for one row.
        """
        return self._predict_tree(X, self.tree_, True)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """
        Returns probabilities using the CHAID model.

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

    def _predict_tree_sql(
        self, X: ArrayLike, tree: dict, class_id: int = 0, return_proba: bool = False
    ):
        """
        Function used recursively to do the final SQL code generation.
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
        Returns the SQL code needed to deploy the model using its 
        attributes.

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
        Returns the SQL code needed to deploy the model probabilities.

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

    def _to_graphviz_tree(
        self,
        tree: dict,
        classes_color: list = [],
        round_pred: int = 2,
        percent: bool = False,
        vertical: bool = True,
        node_style: dict = {},
        arrow_style: dict = {},
        leaf_style: dict = {},
        process: bool = True,
    ) -> str:
        """
        Returns the code for a Graphviz tree.
        """
        if process and len(classes_color) == 0:
            classes_color = self._default_colors()
        if tree["is_leaf"]:
            color = ""
            if isinstance(tree["prediction"], float):
                label = f'"{tree["prediction"]}"'
            else:
                if not (leaf_style):
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
                if isinstance(c, str):
                    q, not_q = "=", "!="
                else:
                    q, not_q = "<=", ">"
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
                if not (vertical):
                    position = '\ngraph [rankdir = "LR"];'
                else:
                    position = ""
                res = "digraph Tree{" + position + res + "\n}"
            return res

    def to_graphviz(
        self,
        classes_color: ArrayLike = [],
        round_pred: int = 2,
        percent: bool = False,
        vertical: bool = True,
        node_style: dict = {},
        arrow_style: dict = {},
        leaf_style: dict = {},
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
            Dictionary of options to customize each node of 
            the tree. For a list of options, see the Graphviz 
            API: https://graphviz.org/doc/info/attrs.html
        arrow_style: dict, optional
            Dictionary of options to customize each arrow of 
            the tree. For a list of options, see the Graphviz 
            API: https://graphviz.org/doc/info/attrs.html
        leaf_style: dict, optional
            Dictionary of options to customize each leaf of 
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
