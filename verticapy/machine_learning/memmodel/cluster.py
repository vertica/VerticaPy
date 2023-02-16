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
from typing import Union
import numpy as np

# VerticaPy Modules
from verticapy.errors import ParameterError
from verticapy.sql._utils import clean_query


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


def bisecting_kmeans_to_graphviz(
    children_left: Union[list, np.ndarray],
    children_right: Union[list, np.ndarray],
    cluster_size: Union[list, np.ndarray] = [],
    cluster_score: Union[list, np.ndarray] = [],
    round_score: int = 2,
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
    children_left: list / numpy.array
        A list of node IDs, where children_left[i] is the node ID of the left
        child of node i.
    children_right: list / numpy.array
        A list of node IDs, where children_right[i] is the node ID of the right child
        of node i.
    cluster_size: list / numpy.array
        A list of sizes, where cluster_size[i] is the number of elements in node i.
    cluster_score: list / numpy.array
        A list of scores, where cluster_score[i] is the score for internal node i.
        The score is the ratio between the within-cluster sum of squares of the node 
        and the total within-cluster sum of squares.
    round_score: int, optional
        The number of decimals to round the node's score to. 0 rounds to an integer.
    percent: bool, optional
        If set to True, the scores are returned as a percent.
    vertical: bool, optional
        If set to True, the function generates a vertical tree.
    node_style: dict, optional
        Dictionary of options to customize each node of the tree. For a list of options, see
        the Graphviz API: https://graphviz.org/doc/info/attrs.html
    arrow_style: dict, optional
        Dictionary of options to customize each arrow of the tree. For a list of options, see
        the Graphviz API: https://graphviz.org/doc/info/attrs.html
    leaf_style: dict, optional
        Dictionary of options to customize each leaf of the tree. For a list of options, see
        the Graphviz API: https://graphviz.org/doc/info/attrs.html

    Returns
    -------
    str
        Graphviz code.
    """
    if len(leaf_style) == 0:
        leaf_style = {"shape": "none"}
    n, position = (
        len(children_left),
        '\ngraph [rankdir = "LR"];' if not (vertical) else "",
    )
    res = "digraph Tree{" + position
    for i in range(n):
        if (len(cluster_size) == n) and (len(cluster_score) == n):
            if "bgcolor" in node_style and (children_left[i] != children_right[i]):
                color = node_style["bgcolor"]
            elif "color" in node_style and (children_left[i] != children_right[i]):
                color = node_style["color"]
            elif children_left[i] != children_right[i]:
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
            if len(cluster_size) == n:
                label += '<tr><td port="port2" border="1" align="left">'
                label += f" size: {cluster_size[i]} </td></tr>"
            if len(cluster_score) == n:
                val = (
                    round(cluster_score[i] * 100, round_score)
                    if percent
                    else round(cluster_score[i], round_score)
                )
                if percent:
                    val = str(val) + "%"
                label += '<tr><td port="port3" border="1" align="left"> '
                label += f"score: {val} </td></tr>"
            label += "</table>>"
        else:
            label = f'"{i}"'
        if children_left[i] != children_right[i]:
            flat_dict_str = flat_dict(node_style)
        else:
            flat_dict_str = flat_dict(leaf_style)
        res += f"\n{i} [label={label}{flat_dict_str}]"
        if children_left[i] != children_right[i]:
            res += f'\n{i} -> {children_left[i]} [label=""{flat_dict(arrow_style)}]'
            res += f'\n{i} -> {children_right[i]} [label=""{flat_dict(arrow_style)}]'
    return res + "\n}"


def predict_from_bisecting_kmeans(
    X: Union[list, np.ndarray],
    clusters: Union[list, np.ndarray],
    left_child: Union[list, np.ndarray],
    right_child: Union[list, np.ndarray],
    p: int = 2,
) -> np.ndarray:
    """
    Predicts using a bisecting k-means model and the input attributes.

    Parameters
    ----------
    X: list / numpy.array
        The data on which to make the prediction.
    clusters: list / numpy.array
        List of the model's cluster centers.
    left_child: list / numpy.array
        List of the model's left children IDs. ID i corresponds to the left 
        child ID of node i.
    right_child: list / numpy.array
        List of the model's right children IDs. ID i corresponds to the right 
        child ID of node i.
    p: int, optional
        The p corresponding to the one of the p-distances.

    Returns
    -------
    numpy.array
        Predicted values
    """
    centroids = np.array(clusters)

    def predict_tree(right_child, left_child, row, node_id, centroids):
        if left_child[node_id] == right_child[node_id] == None:
            return int(node_id)
        else:
            right_node = int(right_child[node_id])
            left_node = int(left_child[node_id])
            if np.sum((row - centroids[left_node]) ** p) < np.sum(
                (row - centroids[right_node]) ** p
            ):
                return predict_tree(right_child, left_child, row, left_node, centroids)
            else:
                return predict_tree(right_child, left_child, row, right_node, centroids)

    def predict_tree_final(row):
        return predict_tree(right_child, left_child, row, 0, centroids)

    return np.apply_along_axis(predict_tree_final, 1, X)


def sql_from_bisecting_kmeans(
    X: Union[list, np.ndarray],
    clusters: Union[list, np.ndarray],
    left_child: Union[list, np.ndarray],
    right_child: Union[list, np.ndarray],
    return_distance_clusters: bool = False,
    p: int = 2,
) -> Union[list, str]:
    """
    Returns the SQL code needed to deploy a bisecting k-means model using its 
    attributes.

    Parameters
    ----------
    X: list / numpy.array
        The names or values of the input predictors.
    clusters: list / numpy.array
        List of the model's cluster centers.
    left_child: list / numpy.array
        List of the model's left children IDs. ID i corresponds to the left 
        child ID of node i.
    right_child: list / numpy.array
        List of the model's right children IDs. ID i corresponds to the right 
        child ID of node i.
    return_distance_clusters: bool, optional
        If set to True, the distance to the clusters is returned.
    p: int, optional
        The p corresponding to the one of the p-distances.

    Returns
    -------
    str / list
        SQL code
    """
    for c in clusters:
        assert len(X) == len(c), ParameterError(
            "The length of parameter 'X' must be the same as the length of each cluster."
        )
    clusters_distance = []
    for c in clusters:
        list_tmp = []
        for idx, col in enumerate(X):
            list_tmp += [f"POWER({X[idx]} - {c[idx]}, {p})"]
        clusters_distance += [f"POWER({' + '.join(list_tmp)}, 1/{p})"]
    if return_distance_clusters:
        return clusters_distance

    def predict_tree(
        right_child: list, left_child: list, node_id: int, clusters_distance: list
    ):
        if left_child[node_id] == right_child[node_id] == None:
            return int(node_id)
        else:
            right_node = int(right_child[node_id])
            left_node = int(left_child[node_id])
            x = clusters_distance[left_node]
            th = clusters_distance[right_node]
            y0 = predict_tree(right_child, left_child, left_node, clusters_distance)
            y1 = predict_tree(right_child, left_child, right_node, clusters_distance)
            return f"(CASE WHEN {x} < {th} THEN {y0} ELSE {y1} END)"

    is_null_x = " OR ".join([f"{x} IS NULL" for x in X])
    sql_final = f"""
        (CASE 
            WHEN {is_null_x} 
                THEN NULL 
            ELSE {predict_tree(right_child, left_child, 0, clusters_distance)} 
        END)"""
    return clean_query(sql_final)


def predict_from_clusters(
    X: Union[list, np.ndarray],
    clusters: Union[list, np.ndarray],
    return_distance_clusters: bool = False,
    return_proba: bool = False,
    classes: Union[list, np.ndarray] = [],
    p: int = 2,
) -> np.ndarray:
    """
    Predicts using a k-means or nearest centroid model and the input attributes.

    Parameters
    ----------
    X: list / numpy.array
        The data on which to make the prediction.
    clusters: list / numpy.array
        List of the model's cluster centers.
    return_distance_clusters: bool, optional
        If set to True, the distance to the clusters is returned.
    return_proba: bool, optional
        If set to True, the probability to belong to the clusters is returned.
    classes: list / numpy.array, optional
        The classes for the nearest centroids model.
    p: int, optional
        The p corresponding to the one of the p-distances.

    Returns
    -------
    numpy.array
        Predicted values
    """
    assert not (return_distance_clusters) or not (return_proba), ParameterError(
        "Parameters 'return_distance_clusters' and 'return_proba' cannot both be set to True."
    )
    centroids = np.array(clusters)
    result = []
    for centroid in centroids:
        result += [np.sum((np.array(centroid) - X) ** p, axis=1) ** (1 / p)]
    result = np.column_stack(result)
    if return_proba:
        result = 1 / (result + 1e-99) / np.sum(1 / (result + 1e-99), axis=1)[:, None]
    elif not (return_distance_clusters):
        result = np.argmin(result, axis=1)
        if classes:
            class_is_str = isinstance(classes[0], str)
            for idx, c in enumerate(classes):
                tmp_idx = str(idx) if class_is_str and idx > 0 else idx
                result = np.where(result == tmp_idx, c, result)
    return result


def sql_from_clusters(
    X: Union[list, np.ndarray],
    clusters: Union[list, np.ndarray],
    return_distance_clusters: bool = False,
    return_proba: bool = False,
    classes: Union[list, np.ndarray] = [],
    p: int = 2,
) -> Union[list, str]:
    """
    Returns the SQL code needed to deploy a k-means or nearest centroids model 
    using its attributes.

    Parameters
    ----------
    X: list / numpy.array
        The names or values of the input predictors.
    clusters: list / numpy.array
        List of the model's cluster centers.
    return_distance_clusters: bool, optional
        If set to True, the distance to the clusters is returned.
    return_proba: bool, optional
        If set to True, the probability to belong to the clusters is returned.
    classes: list / numpy.array, optional
        The classes for the nearest centroids model.
    p: int, optional
        The p corresponding to the one of the p-distances.

    Returns
    -------
    str / list
        SQL code
    """
    for c in clusters:
        assert len(X) == len(c), ParameterError(
            "The length of parameter 'X' must be the same as the length of each cluster."
        )
    assert not (return_distance_clusters) or not (return_proba), ParameterError(
        "Parameters 'return_distance_clusters' and 'return_proba' cannot be set to True."
    )
    classes_tmp = []
    for i in range(len(classes)):
        val = classes[i]
        if isinstance(val, str):
            val = f"'{classes[i]}'"
        elif val == None:
            val = "NULL"
        classes_tmp += [val]
    clusters_distance = []
    for c in clusters:
        list_tmp = []
        for idx, col in enumerate(X):
            list_tmp += [f"POWER({X[idx]} - {c[idx]}, {p})"]
        clusters_distance += ["POWER(" + " + ".join(list_tmp) + f", 1 / {p})"]

    if return_distance_clusters:
        return clusters_distance

    if return_proba:
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
        if not classes:
            c = k - i - 1
        else:
            c = classes_tmp[k - i - 1]
        sql_final += f" WHEN {sql[i]} THEN {c}"
    if not classes:
        c = 0
    else:
        c = classes_tmp[0]
    sql_final += f" ELSE {c} END"

    return sql_final


def predict_from_clusters_kprotypes(
    X: Union[list, np.ndarray],
    clusters: Union[list, np.ndarray],
    return_distance_clusters: bool = False,
    return_proba: bool = False,
    p: int = 2,
    gamma: float = 1.0,
) -> np.ndarray:
    """
    Predicts using a k-prototypes model and the input attributes.

    Parameters
    ----------
    X: list / numpy.array
        The data on which to make the prediction.
    clusters: list / numpy.array
        List of the model's cluster centers.
    return_distance_clusters: bool, optional
        If set to True, the dissimilarity function output to the clusters 
        is returned.
    return_proba: bool, optional
        If set to True, the probability of belonging to the clusters is 
        returned.
    p: int, optional
        The p corresponding to one of the p-distances.
    gamma: float, optional
        Weighting factor for categorical columns. This determines relative 
        importance of numerical and categorical attributes.

    Returns
    -------
    numpy.array
        Predicted values
    """

    assert not (return_distance_clusters) or not (return_proba), ParameterError(
        "Parameters 'return_distance_clusters' and 'return_proba' cannot both be set to True."
    )

    centroids = np.array(clusters)

    def compute_distance_row(X):
        result = []
        for centroid in centroids:
            distance_num, distance_cat = 0, 0
            for idx in range(len(X)):
                val, centroid_val = X[idx], centroid[idx]
                try:
                    val = float(val)
                    centroid_val = float(centroid_val)
                except:
                    pass
                if isinstance(centroid_val, str) or centroid_val == None:
                    distance_cat += abs(int(val == centroid_val) - 1)
                else:
                    distance_num += (val - centroid_val) ** p
            distance_final = distance_num + gamma * distance_cat
            result += [distance_final]
        return result

    result = np.apply_along_axis(compute_distance_row, 1, X)

    if return_proba:
        result = 1 / (result + 1e-99) / np.sum(1 / (result + 1e-99), axis=1)[:, None]
    elif not (return_distance_clusters):
        result = np.argmin(result, axis=1)

    return result


def sql_from_clusters_kprotypes(
    X: Union[list, np.ndarray],
    clusters: Union[list, np.ndarray],
    return_distance_clusters: bool = False,
    return_proba: bool = False,
    p: int = 2,
    gamma: float = 1.0,
    is_categorical: Union[list, np.ndarray] = [],
) -> Union[list, str]:
    """
    Returns the SQL code needed to deploy a k-prototypes or nearest centroids 
    model using its attributes.

    Parameters
    ----------
    X: list / numpy.array
        The names or values of the input predictors.
    clusters: list / numpy.array
        List of the model's cluster centers.
    return_distance_clusters: bool, optional
        If set to True, the distance to the clusters is returned.
    return_proba: bool, optional
        If set to True, the probability of belonging to the clusters is 
        returned.
    p: int, optional
        The p corresponding to one of the p-distances.
    gamma: float, optional
        Weighting factor for categorical columns. This determines relative 
        importance of numerical and categorical attributes.
    is_categorical: list / numpy.array, optional
        List of booleans to indicate whether X[idx] is a categorical variable,
        where True indicates categorical and False numerical. If empty, all
        the variables are considered categorical.

    Returns
    -------
    str / list
        SQL code
    """

    assert not (return_distance_clusters) or not (return_proba), ParameterError(
        "Parameters 'return_distance_clusters' and 'return_proba' cannot "
        "both be set to True."
    )

    if not (is_categorical):
        is_categorical = [True for i in range(len(X))]

    for c in clusters:
        assert len(X) == len(c) == len(is_categorical), ParameterError(
            "The length of parameter 'X' must be the same as the length "
            "of each cluster AND the categorical vector."
        )

    clusters_distance = []
    for c in clusters:
        clusters_distance_num, clusters_distance_cat = [], []
        for idx, col in enumerate(X):
            if is_categorical[idx]:
                c_i = str(c[idx]).replace("'", "''")
                clusters_distance_cat += [f"ABS(({X[idx]} = '{c_i}')::int - 1)"]
            else:
                clusters_distance_num += [f"POWER({X[idx]} - {c[idx]}, {p})"]
        final_cluster_distance = ""
        if clusters_distance_num:
            final_cluster_distance += (
                f"POWER({' + '.join(clusters_distance_num)}, 1 / {p})"
            )
        if clusters_distance_cat:
            if clusters_distance_num:
                final_cluster_distance += " + "
            final_cluster_distance += f"{gamma} * ({' + '.join(clusters_distance_cat)})"
        clusters_distance += [final_cluster_distance]

    if return_distance_clusters:
        return clusters_distance

    if return_proba:
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
        sql_final += f" WHEN {sql[i]} THEN {k - i - 1}"
    sql_final += " ELSE 0 END"
    return sql_final
