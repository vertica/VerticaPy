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
import numpy as np
from collections.abc import Iterable
from typing import Union

# VerticaPy Modules
from verticapy._utils._sql._format import clean_query
from verticapy._utils.math import heuristic_length


def flat_dict(d: dict) -> str:
    # converts dictionary to string with a specific format
    res = []
    for key in d:
        q = '"' if isinstance(d[key], str) else ""
        res += [f"{key}={q}{d[key]}{q}"]
    res = ", ".join(res)
    if res:
        res = f", {res}"
    return res


def predict_from_chaid_tree(
    X: Union[list, np.ndarray],
    tree: dict,
    classes: Union[list, np.ndarray] = [],
    return_proba: bool = False,
) -> np.ndarray:
    """
    Predicts using a CHAID model and the input attributes.

    Parameters
    ----------
    X: list / numpy.array
      Data on which to make the prediction.
    tree: dict
      A CHAID tree. CHAID trees can be generated with the vDataFrame.chaid 
      method.
    classes: list / numpy.array, optional
      The classes in the CHAID model.
    return_proba: bool, optional
      If set to True, the probability of each class is returned.

    Returns
    -------
    numpy.array
      Predicted values
    """

    def predict_tree(X, tree, classes):
        if tree["is_leaf"]:
            if return_proba:
                return tree["prediction"]
            elif isinstance(classes, Iterable) and len(classes) > 0:
                return classes[np.argmax(tree["prediction"])]
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
                    return predict_tree(X, tree["children"][c], classes)
            return None

    def predict_tree_final(X):
        return predict_tree(X, tree, classes)

    return np.apply_along_axis(predict_tree_final, 1, np.array(X))


def sql_from_chaid_tree(
    X: Union[list, np.ndarray],
    tree: dict,
    classes: Union[list, np.ndarray] = [],
    return_proba: bool = False,
) -> np.ndarray:
    """
    Returns the SQL code needed to deploy the CHAID model.

    Parameters
    ----------
    X: list / numpy.array
      Data on which to make the prediction.
    tree: dict
      A CHAID tree. Chaid trees can be generated with the vDataFrame.chaid 
      method.
    classes: list / numpy.array, optional
      The classes in the CHAID model.
    return_proba: bool, optional
      If set to True, the probability of each class is returned.

    Returns
    -------
    str / list
      SQL code
    """

    def predict_tree(X, tree, classes, prob_ID: int = 0):
        if tree["is_leaf"]:
            if return_proba:
                return tree["prediction"][prob_ID]
            elif isinstance(classes, Iterable) and len(classes) > 0:
                res = classes[np.argmax(tree["prediction"])]
                if isinstance(res, str):
                    res = f"'{res}'"
                return res
            else:
                return np.argmax(tree["prediction"])
        else:
            res = "(CASE "
            for c in tree["children"]:
                x = X[tree["split_predictor_idx"]]
                y = predict_tree(X, tree["children"][c], classes, prob_ID)
                if tree["split_is_numerical"]:
                    th = float(c)
                    res += f"WHEN {x} <= {th} THEN {y} "
                else:
                    th = c
                    res += f"WHEN {x} = '{th}' THEN {y} "
            return res + "ELSE NULL END)"

    if return_proba:
        n = len(classes)
        return [predict_tree(X, tree, classes, i) for i in range(n)]
    else:
        return predict_tree(X, tree, classes)


def chaid_to_graphviz(
    tree: dict,
    classes: Union[list, np.ndarray] = [],
    classes_color: list = [],
    round_pred: int = 2,
    percent: bool = False,
    vertical: bool = True,
    node_style: dict = {},
    arrow_style: dict = {},
    leaf_style: dict = {},
    **kwds,
):
    """
    Returns the code for a Graphviz tree.

    Parameters
    ----------
    tree: dict
        CHAID tree. You can generate this tree with the vDataFrame.chaid 
        method.
    classes: list / numpy.array, optional
        The classes in the CHAID model.
    classes_color: list, optional
        Colors that represent the different classes.
    round_pred: int, optional
        The number of decimals to round the prediction to. 0 rounds to 
        an integer.
    percent: bool, optional
        If set to True, the probabilities are returned as percents.
    vertical: bool, optional
        If set to True, the function generates a vertical tree.
    node_style: dict, optional
        Dictionary of options to customize each node of the tree. 
        For a list of options, see the Graphviz API: 
        https://graphviz.org/doc/info/attrs.html
    arrow_style: dict, optional
        Dictionary of options to customize each arrow of the tree. 
        For a list of options, see the Graphviz API: 
        https://graphviz.org/doc/info/attrs.html
    leaf_style: dict, optional
        Dictionary of options to customize each leaf of the tree. 
        For a list of options, see the Graphviz API: 
        https://graphviz.org/doc/info/attrs.html

    Returns
    -------
    str
      Graphviz code.
    """
    if "process" not in kwds or kwds["process"]:
        if len(classes_color) == 0:
            classes_color = [
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
    if tree["is_leaf"]:
        color = ""
        if isinstance(tree["prediction"], float):
            label = f'"{tree["prediction"]}"'
        else:
            if not (leaf_style):
                leaf_style = {"shape": "none"}
            classes_ = (
                [k for k in range(len(tree["prediction"]))]
                if (len(classes) == 0)
                else classes.copy()
            )
            color = classes_color[(np.argmax(tree["prediction"])) % len(classes_color)]
            label = (
                '<<table border="0" cellspacing="0"> '
                f'<tr><td port="port1" border="1" bgcolor="{color}">'
                f"<b> prediction: {classes_[np.argmax(tree['prediction'])]}"
                " </b></td></tr>"
            )
            for j in range(len(tree["prediction"])):
                val = (
                    round(tree["prediction"][j] * 100, round_pred)
                    if percent
                    else round(tree["prediction"][j], round_pred)
                )
                if percent:
                    val = str(val) + "%"
                label += f'<tr><td port="port{j}" border="1" align="left"> '
                label += f"prob({classes_[j]}): {val} </td></tr>"
            label += "</table>>"
        return f"{tree['node_id']} [label={label}{flat_dict(leaf_style)}]"
    else:
        res = ""
        for c in tree["children"]:
            q = "=" if isinstance(c, str) else "<="
            not_q = "!=" if isinstance(c, str) else ">"
            split_predictor = tree["split_predictor"].replace('"', '\\"')
            res += f'\n{tree["node_id"]} [label="{split_predictor}"{flat_dict(node_style)}]'
            if tree["children"][c]["is_leaf"] or tree["children"][c]["children"]:
                res += f'\n{tree["node_id"]} -> {tree["children"][c]["node_id"]}'
                res += f'[label="{q} {c}"{flat_dict(arrow_style)}]'
            res += chaid_to_graphviz(
                tree=tree["children"][c],
                classes=classes,
                classes_color=classes_color,
                round_pred=round_pred,
                percent=percent,
                vertical=vertical,
                node_style=node_style,
                arrow_style=arrow_style,
                leaf_style=leaf_style,
                process=False,
            )
        if "process" not in kwds or kwds["process"]:
            position = '\ngraph [rankdir = "LR"];' if not (vertical) else ""
            res = "digraph Tree{" + position + res + "\n}"
        return res


def predict_from_binary_tree(
    X: Union[list, np.ndarray],
    children_left: Union[list, np.ndarray],
    children_right: Union[list, np.ndarray],
    feature: Union[list, np.ndarray],
    threshold: Union[list, np.ndarray],
    value: Union[list, np.ndarray],
    classes: Union[list, np.ndarray] = [],
    return_proba: bool = False,
    is_regressor: bool = True,
    is_anomaly: bool = False,
    psy: int = -1,
) -> np.ndarray:
    """
    Predicts using a binary tree model and the input attributes.

    Parameters
    ----------
    X: list / numpy.array
        Data on which to make the prediction.
    children_left: list / numpy.array
        A list of node IDs, where children_left[i] is the node id of the 
        left child of node i.
    children_right: list / numpy.array
        A list of node IDs, children_right[i] is the node id of the right 
        child of node i.
    feature: list / numpy.array
         A list of features, where feature[i] is the feature to split on 
         for the internal node i.
    threshold: list / numpy.array
        A list of thresholds, where threshold[i] is the threshold for the 
        internal node i.
    value: list / numpy.array
        Contains the constant prediction value of each node. If used for 
        classification and if return_proba is set to True, each element 
        of the list must be a sublist with the probabilities of each class.
    classes: list / numpy.array, optional
        The classes for the binary tree model.
    return_proba: bool, optional
        If set to True, the probability of each class is returned.
    is_regressor: bool, optional
        If set to True, the parameter 'value' corresponds to the result of
        a regression.
    is_anomaly: bool, optional
        If set to True, the parameter 'value' corresponds to the result of
        an Isolation Forest (a tuple that includes leaf path length and 
        training row count).
    psy: int, optional
        Sampling size used to compute the Isolation Forest Score.

    Returns
    -------
    numpy.array
        Predicted values
    """

    def predict_tree(
        children_left, children_right, feature, threshold, value, node_id, X
    ):
        if children_left[node_id] == children_right[node_id]:
            if is_anomaly:
                return (
                    value[node_id][0] + heuristic_length(value[node_id][1])
                ) / heuristic_length(psy)
            elif (
                not (is_regressor)
                and not (return_proba)
                and isinstance(value, Iterable)
            ):
                if isinstance(classes, Iterable) and len(classes) > 0:
                    return classes[np.argmax(value[node_id])]
                else:
                    return np.argmax(value[node_id])
            else:
                return value[node_id]
        else:
            if (
                isinstance(threshold[node_id], str)
                and str(X[feature[node_id]]) == threshold[node_id]
            ) or (
                not (isinstance(threshold[node_id], str))
                and float(X[feature[node_id]]) < float(threshold[node_id])
            ):
                return predict_tree(
                    children_left,
                    children_right,
                    feature,
                    threshold,
                    value,
                    children_left[node_id],
                    X,
                )
            else:
                return predict_tree(
                    children_left,
                    children_right,
                    feature,
                    threshold,
                    value,
                    children_right[node_id],
                    X,
                )

    def predict_tree_final(X):
        return predict_tree(
            children_left, children_right, feature, threshold, value, 0, X
        )

    return np.apply_along_axis(predict_tree_final, 1, np.array(X))


def sql_from_binary_tree(
    X: Union[list, np.ndarray],
    children_left: Union[list, np.ndarray],
    children_right: Union[list, np.ndarray],
    feature: Union[list, np.ndarray],
    threshold: Union[list, np.ndarray],
    value: Union[list, np.ndarray],
    classes: Union[list, np.ndarray] = [],
    return_proba: bool = False,
    is_regressor: bool = True,
    is_anomaly: bool = False,
    psy: int = -1,
) -> Union[list, str]:
    """
    Returns the SQL code needed to deploy a binary tree model using 
    its attributes.

    Parameters
    ----------
    X: list / numpy.array
        Data on which to make the prediction.
    children_left: list / numpy.array
        A list of node IDs, where children_left[i] is the node id of the 
        left child of node i.
    children_right: list / numpy.array
        A list of node IDs, children_right[i] is the node id of the right 
        child of node i.
    feature: list / numpy.array
        A list of features, where feature[i] is the feature to split on 
        for the internal node i.
    threshold: list / numpy.array
        A list of thresholds, where threshold[i] is the threshold for the 
        internal node i.
    value: list / numpy.array
        Contains the constant prediction value of each node. If used for 
        classification and if return_proba is set to True, each element 
        of the list must be a sublist with the probabilities of each class.
    classes: list / numpy.array, optional
        The classes for the binary tree model.
    return_proba: bool, optional
        If set to True, the probability of each class is returned.
    is_regressor: bool, optional
        If set to True, the parameter 'value' corresponds to the result of
        a regression.
    is_anomaly: bool, optional
        If set to True, the parameter 'value' corresponds to the result of
        an Isolation Forest (a tuple that includes leaf path length and 
        training row count).
    psy: int, optional
        Sampling size used to compute the Isolation Forest Score.

    Returns
    -------
    str / list
        SQL code
    """

    def predict_tree(
        children_left, children_right, feature, threshold, value, node_id, X, prob_ID=0,
    ):
        if children_left[node_id] == children_right[node_id]:
            if return_proba:
                return value[node_id][prob_ID]
            else:
                if is_anomaly:
                    return (
                        value[node_id][0] + heuristic_length(value[node_id][1])
                    ) / heuristic_length(psy)
                elif (
                    not (is_regressor)
                    and isinstance(classes, Iterable)
                    and len(classes) > 0
                ):
                    result = classes[np.argmax(value[node_id])]
                    if isinstance(result, str):
                        return "'" + result + "'"
                    else:
                        return result
                else:
                    return value[node_id]
        else:
            if isinstance(threshold[node_id], str):
                op = "="
                q = "'"
            else:
                op = "<"
                q = ""
            y0 = predict_tree(
                children_left,
                children_right,
                feature,
                threshold,
                value,
                children_left[node_id],
                X,
                prob_ID,
            )
            y1 = predict_tree(
                children_left,
                children_right,
                feature,
                threshold,
                value,
                children_right[node_id],
                X,
                prob_ID,
            )
            query = f"""
                (CASE 
                    WHEN {X[feature[node_id]]} {op} {q}{threshold[node_id]}{q} 
                    THEN {y0} ELSE {y1} 
                END)"""
            return clean_query(query)

    if return_proba:
        n = max([len(l) if l != None else 0 for l in value])
        return [
            predict_tree(
                children_left, children_right, feature, threshold, value, 0, X, i
            )
            for i in range(n)
        ]
    else:
        return predict_tree(
            children_left, children_right, feature, threshold, value, 0, X
        )


def binary_tree_to_graphviz(
    children_left: Union[list, np.ndarray],
    children_right: Union[list, np.ndarray],
    feature: Union[list, np.ndarray],
    threshold: Union[list, np.ndarray],
    value: Union[list, np.ndarray],
    feature_names: Union[list, np.ndarray] = [],
    classes: Union[list, np.ndarray] = [],
    classes_color: list = [],
    prefix_pred: str = "prob",
    round_pred: int = 2,
    percent: bool = False,
    vertical: bool = True,
    node_style: dict = {},
    arrow_style: dict = {},
    leaf_style: dict = {},
    psy: int = -1,
):
    """
    Returns the code for a Graphviz tree.

    Parameters
    ----------
    children_left: list / numpy.array
        A list of node IDs, where children_left[i] is the node ID 
        of the left child of node i.
    children_right: list / numpy.array
        A list of node IDs, where children_right[i] is the node ID 
        of the right child of node i.
    feature: list / numpy.array
        A list of features, where feature[i] is the feature to split 
        on for internal node i.
    threshold: list / numpy.array
        A list of thresholds, where threshold[i] is the threshold for 
        internal node i.
    value: list / numpy.array
        A list of constant prediction values of each node. If used for 
        classification and return_proba is set to True, each element of 
        the list must be a sublist with the probabilities of each class.
    feature_names: list / numpy.array, optional
        List of the names of each feature.
    classes: list / numpy.array, optional
        The classes for the binary tree model.
    classes_color: list, optional
        Colors that represent the different classes.
    prefix_pred: str, optional
        The prefix for the name of each prediction.
    round_pred: int, optional
        The number of decimals to round the prediction to. 0 rounds to 
        an integer.
    percent: bool, optional
        If set to True, the probabilities are returned as percents.
    vertical: bool, optional
        If set to True, the function generates a vertical tree.
    node_style: dict, optional
        Dictionary of options to customize each node of the tree. 
        For a list of options, see the Graphviz API: 
        https://graphviz.org/doc/info/attrs.html
    arrow_style: dict, optional
        Dictionary of options to customize each arrow of the tree. 
        For a list of options, see the Graphviz API: 
        https://graphviz.org/doc/info/attrs.html
    leaf_style: dict, optional
        Dictionary of options to customize each leaf of the tree. 
        For a list of options, see the Graphviz API: 
        https://graphviz.org/doc/info/attrs.html
    psy: int, optional
        Sampling size used to compute the Isolation Forest Score.

    Returns
    -------
    str
        Graphviz code.
    """
    empty_color = False
    if len(classes_color) == 0:
        empty_color = True
        classes_color = [
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
    position = '\ngraph [rankdir = "LR"];' if not (vertical) else ""
    n, res = len(children_left), "digraph Tree{" + position
    for i in range(n):
        if children_left[i] != children_right[i]:
            if feature_names:
                name = feature_names[feature[i]].replace('"', '\\"')
            else:
                name = f"X{feature[i]}"
            q = "=" if isinstance(threshold[i], str) else "<="
            not_q = "!=" if isinstance(threshold[i], str) else ">"
            res += f'\n{i} [label="{name}"{flat_dict(node_style)}]'
            res += f'\n{i} -> {children_left[i]} [label="{q} {threshold[i]}"'
            res += f"{flat_dict(arrow_style)}]\n{i} -> {children_right[i]} "
            res += f'[label="{not_q} {threshold[i]}"{flat_dict(arrow_style)}]'
        else:
            color = ""
            if isinstance(value[i], float):
                label = f'"{value[i]}"'
            elif (
                isinstance(value[i], list)
                and (len(value[i]) == 2)
                and (isinstance(value[i][0], int))
                and (isinstance(value[i][1], int))
            ):
                if not (leaf_style):
                    leaf_style = {"shape": "none"}
                color = classes_color[0] if not (empty_color) else "#eeeeee"
                anomaly_score = float(
                    2
                    ** (
                        -(value[i][0] + heuristic_length(value[i][1]))
                        / heuristic_length(psy)
                    )
                )
                if anomaly_score < 0.5:
                    color_anomaly = "#ffffff"
                else:
                    rgb = [255, 0, 0]
                    for idx in range(3):
                        rgb[idx] = int(
                            255 - 2 * (anomaly_score - 0.5) * (255 - rgb[idx])
                        )
                    color_anomaly = (
                        "#"
                        + str(hex(rgb[0]))[2:]
                        + str(hex(rgb[1]))[2:]
                        + str(hex(rgb[2]))[2:]
                    )
                label = (
                    '<<table border="0" cellspacing="0"> <tr><td port="port1"'
                    f' border="1" bgcolor="{color}"><b> leaf </b></td></tr><tr><td '
                    f'port="port0" border="1" align="left"> leaf_path_length: '
                    f'{value[i][0]} </td></tr><tr><td port="port1" border="1" '
                    f'align="left"> training_row_count: {value[i][1]} </td></tr>'
                    f'<tr><td port="port2" border="1" align="left" bgcolor="{color_anomaly}">'
                    f" anomaly_score: {anomaly_score} </td></tr></table>>"
                )
            else:
                if not (leaf_style):
                    leaf_style = {"shape": "none"}
                classes_ = (
                    [k for k in range(len(value[i]))]
                    if (len(classes) == 0)
                    else classes.copy()
                )
                color = classes_color[(np.argmax(value[i])) % len(classes_color)]
                label = (
                    '<<table border="0" cellspacing="0"> <tr><td port="port1" border="1" '
                    f'bgcolor="{color}"><b> prediction: {classes_[np.argmax(value[i])]} '
                    "</b></td></tr>"
                )
                for j in range(len(value[i])):
                    if percent:
                        val = str(round(value[i][j] * 100, round_pred)) + "%"
                    else:
                        val = round(value[i][j], round_pred)
                    label += f'<tr><td port="port{j}" border="1" align="left">'
                    label += f" {prefix_pred}({classes_[j]}): {val} </td></tr>"
                label += "</table>>"
            res += f"\n{i} [label={label}{flat_dict(leaf_style)}]"
    return res + "\n}"
