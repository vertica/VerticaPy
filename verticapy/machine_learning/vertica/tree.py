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

from verticapy._typing import PythonNumber
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._vertica_version import check_minimum_version

from verticapy.machine_learning.vertica.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)

"""
Algorithms used for regression.
"""


class DecisionTreeRegressor(RandomForestRegressor):
    """
    A DecisionTreeRegressor consisting of a single tree.

    Parameters
    ----------
    name: str, optional
        Name of the model. The model is stored in the
        database.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
    max_features: str / int, optional
        The number of randomly  chosen features from which
        to pick the best feature  to split on a given tree
        node.  It  can  be  an integer  or one of the  two
        following methods:
            auto : square root of the total number of
                   predictors.
            max  : number of predictors.
    max_leaf_nodes: PythonNumber, optional
        The maximum number of leaf nodes for a tree in the
        forest, an integer between 1 and 1e9, inclusive.
    max_depth: int, optional
        The maximum depth for growing each tree, an integer
        between 1 and 100, inclusive.
    min_samples_leaf: int, optional
        The minimum number of samples each branch must have
        after a node is split, an integer between 1 and 1e6,
        inclusive. Any split that results in fewer remaining
        samples is discarded.
    min_info_gain: PythonNumber, optional
        The  minimum  threshold  for including a  split,  a
        float between 0.0 and 1.0,  inclusive. A split with
        information  gain  less  than   this  threshold  is
        discarded.
    nbins: int, optional
        The number of bins to use for continuous  features,
        an integer between 2 and 1000, inclusive.
    """

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str = None,
        overwrite_model: bool = False,
        max_features: Union[Literal["auto", "max"], int] = "auto",
        max_leaf_nodes: PythonNumber = 1e9,
        max_depth: int = 100,
        min_samples_leaf: int = 1,
        min_info_gain: PythonNumber = 0.0,
        nbins: int = 32,
    ) -> None:
        super().__init__(name, overwrite_model)
        self.parameters = {
            "n_estimators": 1,
            "max_features": max_features,
            "max_leaf_nodes": int(max_leaf_nodes),
            "sample": 1.0,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "min_info_gain": min_info_gain,
            "nbins": nbins,
        }


class DummyTreeRegressor(RandomForestRegressor):
    """
    A regressor that overfits the training data.
    These models are typically used as a control
    to compare with your other models.

    Parameters
    ----------
    name: str, optional
        Name of the model. The model is stored
        in the database.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
    """

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(self, name: str = None, overwrite_model: bool = False) -> None:
        super().__init__(name, overwrite_model)
        self.parameters = {
            "n_estimators": 1,
            "max_features": "max",
            "max_leaf_nodes": int(1e9),
            "sample": 1.0,
            "max_depth": 100,
            "min_samples_leaf": 1,
            "min_info_gain": 0.0,
            "nbins": 1000,
        }


"""
Algorithms used for classification.
"""


class DecisionTreeClassifier(RandomForestClassifier):
    """
    A DecisionTreeClassifier consisting of a single tree.

    Parameters
    ----------
    name: str, optional
        Name of the model. The model is stored in the
        database.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
    max_features: str / int, optional
        The number of randomly  chosen features from which
        to pick the best feature  to split on a given tree
        node.  It  can  be  an integer  or one of the  two
        following methods.
            auto : square root of the total number of
                   predictors.
            max  : number of predictors.
    max_leaf_nodes: PythonNumber, optional
        The maximum number of leaf nodes for a tree in the
        forest, an integer between 1 and 1e9, inclusive.
    max_depth: int, optional
        The maximum depth for growing each tree, an integer
        between 1 and 100, inclusive.
    min_samples_leaf: int, optional
        The minimum number of samples each branch must have
        after a node is split, an integer between 1 and 1e6,
        inclusive. Any split that results in fewer remaining
        samples is discarded.
    min_info_gain: PythonNumber, optional
        The  minimum  threshold  for including a  split,  a
        float between 0.0 and 1.0,  inclusive. A split with
        information  gain  less  than   this  threshold  is
        discarded.
    nbins: int, optional
        The number of bins to use for continuous  features,
        an integer between 2 and 1000, inclusive.
    """

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str = None,
        overwrite_model: bool = False,
        max_features: Union[Literal["auto", "max"], int] = "auto",
        max_leaf_nodes: PythonNumber = 1e9,
        max_depth: int = 100,
        min_samples_leaf: int = 1,
        min_info_gain: PythonNumber = 0.0,
        nbins: int = 32,
    ) -> None:
        super().__init__(name, overwrite_model)
        self.parameters = {
            "n_estimators": 1,
            "max_features": max_features,
            "max_leaf_nodes": int(max_leaf_nodes),
            "sample": 1.0,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "min_info_gain": min_info_gain,
            "nbins": nbins,
        }


class DummyTreeClassifier(RandomForestClassifier):
    """
    A classifier that overfits the training data.
    These models are  typically used as a control
    to compare with your other models.

    Parameters
    ----------
    name: str, optional
        Name of  the  model. The model is stored
        in the database.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
    """

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(self, name: str = None, overwrite_model: bool = False) -> None:
        super().__init__(name, overwrite_model)
        self.parameters = {
            "n_estimators": 1,
            "max_features": "max",
            "max_leaf_nodes": int(1e9),
            "sample": 1.0,
            "max_depth": 100,
            "min_samples_leaf": 1,
            "min_info_gain": 0.0,
            "nbins": 1000,
        }
