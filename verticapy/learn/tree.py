# (c) Copyright [2018-2022] Micro Focus or one of its affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# |_     |~) _  _| _  /~\    _ |.
# |_)\/  |_)(_|(_||   \_/|_|(_|||
#    /
#              ____________       ______
#             / __        `\     /     /
#            |  \/         /    /     /
#            |______      /    /     /
#                   |____/    /     /
#          _____________     /     /
#          \           /    /     /
#           \         /    /     /
#            \_______/    /     /
#             ______     /     /
#             \    /    /     /
#              \  /    /     /
#               \/    /     /
#                    /     /
#                   /     /
#                   \    /
#                    \  /
#                     \/
#                    _
# \  / _  __|_. _ _ |_)
#  \/ (/_|  | |(_(_|| \/
#                     /
# VerticaPy is a Python library with scikit-like functionality for conducting
# data science projects on data stored in Vertica, taking advantage Vertica’s
# speed and built-in analytics and machine learning features. It supports the
# entire data science life cycle, uses a ‘pipeline’ mechanism to sequentialize
# data transformation operations, and offers beautiful graphical options.
#
# VerticaPy aims to do all of the above. The idea is simple: instead of moving
# data around for processing, VerticaPy brings the logic to the data.
#
#
# Modules
#
# VerticaPy Modules
from verticapy.learn.vmodel import *

# Standard Python Modules
from typing import Union

#
# Functions used to simplify the code
#
# ---#
def get_tree_list_of_arrays(
    tree, X: list, model_type: str, return_probability: bool = False
):
    """
    Takes as input a tree which is represented by a tablesample
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
# ---#
class DecisionTreeClassifier(MulticlassClassifier, Tree):
    """
    ---------------------------------------------------------------------------
    A DecisionTreeClassifier made of a single tree.

    Parameters
    ----------
    name: str
            Name of the the model. The model will be stored in the DB.
    max_features: str/int, optional
            The number of randomly chosen features from which to pick the best
        feature to split on a given tree node. It can be an integer or one
        of the two following methods.
                    auto : square root of the total number of predictors.
                    max  : number of predictors.
    max_leaf_nodes: int, optional
            The maximum number of leaf nodes a tree in the forest can have, an
        integer between 1 and 1e9, inclusive.
    max_depth: int, optional
            The maximum depth for growing each tree, an integer between 1 and 100,
        inclusive.
    min_samples_leaf: int, optional
            The minimum number of samples each branch must have after splitting a
        node, an integer between 1 and 1e6, inclusive. A split that causes
        fewer remaining samples is discarded.
    min_info_gain: float, optional
            The minimum threshold for including a split, a float between 0.0 and
        1.0, inclusive. A split with information gain less than this threshold
        is discarded.
    nbins: int, optional
            The number of bins to use for continuous features, an integer between 2
        and 1000, inclusive.
    """

    def __init__(
        self,
        name: str,
        max_features: Union[int, str] = "auto",
        max_leaf_nodes: int = 1e9,
        max_depth: int = 100,
        min_samples_leaf: int = 1,
        min_info_gain: float = 0.0,
        nbins: int = 32,
    ):
        version(condition=[8, 1, 1])
        check_types([("name", name, [str])])
        self.type, self.name = "RandomForestClassifier", name
        self.set_params(
            {
                "n_estimators": 1,
                "max_features": max_features,
                "max_leaf_nodes": max_leaf_nodes,
                "sample": 1.0,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "min_info_gain": min_info_gain,
                "nbins": nbins,
            }
        )


# ---#
class DecisionTreeRegressor(Regressor, Tree):
    """
    ---------------------------------------------------------------------------
    A DecisionTreeRegressor made of a single tree.

    Parameters
    ----------
    name: str
            Name of the the model. The model will be stored in the DB.
    max_features: str/int, optional
            The number of randomly chosen features from which to pick the best
        feature to split on a given tree node. It can be an integer or one
        of the two following methods.
                    auto : square root of the total number of predictors.
                    max  : number of predictors.
    max_leaf_nodes: int, optional
            The maximum number of leaf nodes a tree in the forest can have, an
        integer between 1 and 1e9, inclusive.
    max_depth: int, optional
            The maximum depth for growing each tree, an integer between 1 and 100,
        inclusive.
    min_samples_leaf: int, optional
            The minimum number of samples each branch must have after splitting
        a node, an integer between 1 and 1e6, inclusive. A split that causes
        fewer remaining samples is discarded.
    min_info_gain: float, optional
            The minimum threshold for including a split, a float between 0.0 and
        1.0, inclusive. A split with information gain less than this threshold
        is discarded.
    nbins: int, optional
            The number of bins to use for continuous features, an integer between 2
        and 1000, inclusive.
    """

    def __init__(
        self,
        name: str,
        max_features: Union[int, str] = "auto",
        max_leaf_nodes: int = 1e9,
        max_depth: int = 100,
        min_samples_leaf: int = 1,
        min_info_gain: float = 0.0,
        nbins: int = 32,
    ):
        version(condition=[9, 0, 1])
        check_types([("name", name, [str])])
        self.type, self.name = "RandomForestRegressor", name
        self.set_params(
            {
                "n_estimators": 1,
                "max_features": max_features,
                "max_leaf_nodes": max_leaf_nodes,
                "sample": 1.0,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "min_info_gain": min_info_gain,
                "nbins": nbins,
            }
        )


# ---#
class DummyTreeClassifier(MulticlassClassifier, Tree):
    """
    ---------------------------------------------------------------------------
    A classifier that overfits the training data. These models are typically
    used as a control to compare with your other models.

    Parameters
    ----------
    name: str
            Name of the the model. The model will be stored in the DB.
    """

    def __init__(self, name: str):
        version(condition=[8, 1, 1])
        check_types([("name", name, [str])])
        self.type, self.name = "RandomForestClassifier", name
        self.set_params(
            {
                "n_estimators": 1,
                "max_features": "max",
                "max_leaf_nodes": 1e9,
                "sample": 1.0,
                "max_depth": 100,
                "min_samples_leaf": 1,
                "min_info_gain": 0.0,
                "nbins": 1000,
            }
        )


# ---#
class DummyTreeRegressor(Regressor, Tree):
    """
    ---------------------------------------------------------------------------
    A regressor that overfits the training data. These models are typically
    used as a control to compare with your other models.

    Parameters
    ----------
    name: str
            Name of the the model. The model will be stored in the DB.
    """

    def __init__(self, name: str):
        version(condition=[9, 0, 1])
        check_types([("name", name, [str])])
        self.type, self.name = "RandomForestRegressor", name
        self.set_params(
            {
                "n_estimators": 1,
                "max_features": "max",
                "max_leaf_nodes": 1e9,
                "sample": 1.0,
                "max_depth": 100,
                "min_samples_leaf": 1,
                "min_info_gain": 0.0,
                "nbins": 1000,
            }
        )
