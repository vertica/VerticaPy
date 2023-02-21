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
from typing import Literal, Union

from verticapy._utils._collect import save_verticapy_logs
from verticapy._utils._sql._format import quote_ident
from verticapy._version import check_minimum_version

from verticapy.machine_learning.vertica.base import (
    MulticlassClassifier,
    Regressor,
    Tree,
)


def get_tree_list_of_arrays(
    tree, X: list, model_type: str, return_probability: bool = False
):
    """
    Takes as input a tree which is represented by a TableSample
    It returns a list of arrays. Each index of the arrays represents
    a node value.
    """

    def map_idx(x):
        for idx, elem in enumerate(X):
            if quote_ident(x).lower() == quote_ident(elem).lower():
                return idx

    tree_list = []
    for idx in range(len(tree["tree_id"])):
        tree.values["left_child_id"] = [
            idx if elem == tree.values["node_id"][idx] else elem
            for elem in tree.values["left_child_id"]
        ]
        tree.values["right_child_id"] = [
            idx if elem == tree.values["node_id"][idx] else elem
            for elem in tree.values["right_child_id"]
        ]
        tree.values["node_id"][idx] = idx
        tree.values["split_predictor"][idx] = map_idx(tree["split_predictor"][idx])
        if model_type == "XGBoostClassifier" and isinstance(tree["log_odds"][idx], str):
            val, all_val = tree["log_odds"][idx].split(","), {}
            for elem in val:
                all_val[elem.split(":")[0]] = float(elem.split(":")[1])
            tree.values["log_odds"][idx] = all_val
    if model_type == "IsolationForest":
        tree.values["prediction"], n = [], len(tree.values["leaf_path_length"])
        for idx in range(n):
            if tree.values["leaf_path_length"][idx] != None:
                tree.values["prediction"] += [
                    [
                        int(float(tree.values["leaf_path_length"][idx])),
                        int(float(tree.values["training_row_count"][idx])),
                    ]
                ]
            else:
                tree.values["prediction"] += [None]
    tree_list = [
        tree["left_child_id"],
        tree["right_child_id"],
        tree["split_predictor"],
        tree["split_value"],
        tree["prediction"],
        tree["is_categorical_split"],
    ]
    if model_type == "XGBoostClassifier":
        tree_list += [tree["log_odds"]]
    if return_probability:
        tree_list += [tree["probability/variance"]]
    return tree_list


#
# Tree Algorithms
#


class DecisionTreeClassifier(MulticlassClassifier, Tree):
    """
    A DecisionTreeClassifier made of a single tree.

    Parameters
    ----------
    name: str
        Name of the the model. The model will be stored in the DB.
    max_features: str / int, optional
        The number of randomly chosen features from which to pick the best
        feature to split on a given tree node. It can be an integer or one
        of the two following methods.
            auto : square root of the total number of predictors.
            max  : number of predictors.
    max_leaf_nodes: int / float, optional
        The maximum number of leaf nodes a tree in the forest can have, an
        integer between 1 and 1e9, inclusive.
    max_depth: int, optional
        The maximum depth for growing each tree, an integer between 1 and 100,
        inclusive.
    min_samples_leaf: int, optional
        The minimum number of samples each branch must have after splitting a
        node, an integer between 1 and 1e6, inclusive. A split that causes
        fewer remaining samples is discarded.
    min_info_gain: int / float, optional
        The minimum threshold for including a split, a float between 0.0 and
        1.0, inclusive. A split with information gain less than this threshold
        is discarded.
    nbins: int, optional
        The number of bins to use for continuous features, an integer between 2
        and 1000, inclusive.
    """

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str,
        max_features: Union[Literal["auto", "max"], int] = "auto",
        max_leaf_nodes: Union[int, float] = 1e9,
        max_depth: int = 100,
        min_samples_leaf: int = 1,
        min_info_gain: Union[int, float] = 0.0,
        nbins: int = 32,
    ):
        self.type, self.name = "RandomForestClassifier", name
        self.VERTICA_FIT_FUNCTION_SQL = "RF_CLASSIFIER"
        self.VERTICA_PREDICT_FUNCTION_SQL = "PREDICT_RF_CLASSIFIER"
        self.MODEL_TYPE = "SUPERVISED"
        self.MODEL_SUBTYPE = "CLASSIFIER"
        self.parameters = {
            "n_estimators": 1,
            "max_features": max_features,
            "max_leaf_nodes": max_leaf_nodes,
            "sample": 1.0,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "min_info_gain": min_info_gain,
            "nbins": nbins,
        }


class DecisionTreeRegressor(Regressor, Tree):
    """
    A DecisionTreeRegressor made of a single tree.

    Parameters
    ----------
    name: str
        Name of the the model. The model will be stored in the DB.
    max_features: str / int, optional
        The number of randomly chosen features from which to pick the best
        feature to split on a given tree node. It can be an integer or one
        of the two following methods.
            auto : square root of the total number of predictors.
            max  : number of predictors.
    max_leaf_nodes: int / float, optional
        The maximum number of leaf nodes a tree in the forest can have, an
        integer between 1 and 1e9, inclusive.
    max_depth: int, optional
        The maximum depth for growing each tree, an integer between 1 and 100,
        inclusive.
    min_samples_leaf: int, optional
        The minimum number of samples each branch must have after splitting
        a node, an integer between 1 and 1e6, inclusive. A split that causes
        fewer remaining samples is discarded.
    min_info_gain: int / float, optional
        The minimum threshold for including a split, a float between 0.0 and
        1.0, inclusive. A split with information gain less than this threshold
        is discarded.
    nbins: int, optional
        The number of bins to use for continuous features, an integer between 2
        and 1000, inclusive.
    """

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str,
        max_features: Union[Literal["auto", "max"], int] = "auto",
        max_leaf_nodes: Union[int, float] = 1e9,
        max_depth: int = 100,
        min_samples_leaf: int = 1,
        min_info_gain: Union[int, float] = 0.0,
        nbins: int = 32,
    ):
        self.type, self.name = "RandomForestRegressor", name
        self.VERTICA_FIT_FUNCTION_SQL = "RF_REGRESSOR"
        self.VERTICA_PREDICT_FUNCTION_SQL = "PREDICT_RF_REGRESSOR"
        self.MODEL_TYPE = "SUPERVISED"
        self.MODEL_SUBTYPE = "REGRESSOR"
        self.parameters = {
            "n_estimators": 1,
            "max_features": max_features,
            "max_leaf_nodes": max_leaf_nodes,
            "sample": 1.0,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "min_info_gain": min_info_gain,
            "nbins": nbins,
        }


class DummyTreeClassifier(MulticlassClassifier, Tree):
    """
    A classifier that overfits the training data. These models are typically
    used as a control to compare with your other models.

    Parameters
    ----------
    name: str
        Name of the the model. The model will be stored in the DB.
    """

    @check_minimum_version
    @save_verticapy_logs
    def __init__(self, name: str):
        self.type, self.name = "RandomForestClassifier", name
        self.VERTICA_FIT_FUNCTION_SQL = "RF_CLASSIFIER"
        self.VERTICA_PREDICT_FUNCTION_SQL = "PREDICT_RF_CLASSIFIER"
        self.MODEL_TYPE = "SUPERVISED"
        self.MODEL_SUBTYPE = "CLASSIFIER"
        self.parameters = {
            "n_estimators": 1,
            "max_features": "max",
            "max_leaf_nodes": 1e9,
            "sample": 1.0,
            "max_depth": 100,
            "min_samples_leaf": 1,
            "min_info_gain": 0.0,
            "nbins": 1000,
        }


class DummyTreeRegressor(Regressor, Tree):
    """
    A regressor that overfits the training data. These models are typically
    used as a control to compare with your other models.

    Parameters
    ----------
    name: str
        Name of the the model. The model will be stored in the DB.
    """

    @check_minimum_version
    @save_verticapy_logs
    def __init__(self, name: str):
        self.type, self.name = "RandomForestRegressor", name
        self.VERTICA_FIT_FUNCTION_SQL = "RF_REGRESSOR"
        self.VERTICA_PREDICT_FUNCTION_SQL = "PREDICT_RF_REGRESSOR"
        self.MODEL_TYPE = "SUPERVISED"
        self.MODEL_SUBTYPE = "REGRESSOR"
        self.parameters = {
            "n_estimators": 1,
            "max_features": "max",
            "max_leaf_nodes": 1e9,
            "sample": 1.0,
            "max_depth": 100,
            "min_samples_leaf": 1,
            "min_info_gain": 0.0,
            "nbins": 1000,
        }
