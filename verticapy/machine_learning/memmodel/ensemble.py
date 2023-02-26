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
from typing import Literal

import verticapy._config.config as conf
from verticapy._utils._sql._format import format_magic
from verticapy._typing import ArrayLike

from verticapy.machine_learning.memmodel.base import InMemoryModel, MulticlassClassifier
from verticapy.machine_learning.memmodel.tree import (
    BinaryTreeAnomaly,
    BinaryTreeClassifier,
    BinaryTreeRegressor,
)

if conf._get_import_success("graphviz"):
    import graphviz


class Ensemble(InMemoryModel):
    """
    InMemoryModel Implementation of Ensemble Algorithms.
    """

    def _predict_trees(self, X: ArrayLike) -> np.ndarray:
        """
        Predicts using all the model trees.
        """
        return np.column_stack([tree.predict(X) for tree in self.trees_])

    def _predict_trees_sql(self, X: ArrayLike) -> list[str]:
        """
        Predicts using all the model trees.
        """
        return [str(tree.predict_sql(X)) for tree in self.trees_]

    def plot_tree(
        self, pic_path: str = "", tree_id: int = 0, *argv, **kwds,
    ):
        """
        Draws the input tree. Requires the graphviz module.

        Parameters
        ----------
        pic_path: str, optional
            Absolute path to save the image of the tree.
        tree_id: int, optional
            Unique tree identifier, an integer in the 
            range [0, n_estimators - 1].
        *argv, **kwds: Any, optional
            Arguments to pass to the 'to_graphviz' method.

        Returns
        -------
        graphviz.Source
            graphviz object.
        """
        return self.trees_[tree_id].plot_tree(pic_path, *argv, **kwds)


class RandomForestRegressor(Ensemble):
    """
    InMemoryModel Implementation of the Random Forest Regressor Algorithm.

    Parameters
    ----------
    trees: list[BinaryTreeRegressor]
        list of BinaryTrees for Regression.
    """

    @property
    def _object_type(self) -> Literal["RandomForestRegressor"]:
        return "RandomForestRegressor"

    @property
    def _attributes(self) -> Literal["trees_"]:
        return ["trees_"]

    def __init__(self, trees: list[BinaryTreeRegressor]) -> None:
        self.trees_ = copy.deepcopy(trees)
        return None

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predicts using the Random Forest Regressor model.

        Parameters
        ----------
        X: ArrayLike
            The data on which to make the prediction.

        Returns
        -------
        numpy.array
            Predicted values.
        """
        return np.average(self._predict_trees(X), axis=1)

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
        trees_pred = self._predict_trees_sql(X)
        return f"({' + '.join(trees_pred)}) / {len(trees_pred)}"


class RandomForestClassifier(Ensemble, MulticlassClassifier):
    """
    InMemoryModel Implementation of the Random Forest Classifier Algorithm.

    Parameters
    ----------
    trees: list[BinaryTreeClassifier]
        List of BinaryTrees for Classification.
    classes: ArrayLike, optional
    	The model's classes.
    """

    @property
    def _object_type(self) -> Literal["RandomForestClassifier"]:
        return "RandomForestClassifier"

    @property
    def _attributes(self) -> Literal["trees_", "classes_"]:
        return ["trees_", "classes_"]

    def __init__(
        self, trees: list[BinaryTreeClassifier], classes: ArrayLike = []
    ) -> None:
        self.trees_ = copy.deepcopy(trees)
        self.classes_ = np.array(classes)
        return None

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """
        Computes the model's probabilites using the input Matrix.

        Parameters
        ----------
        X: list / numpy.array
            The data on which to make the prediction.

        Returns
        -------
        numpy.array
            Probabilities.
        """
        trees_prob_sum, n = 0, len(self.trees_)
        for i in range(n):
            tree_prob_i = self.trees_[i].predict_proba(X)
            tree_prob_i_arg = np.zeros_like(tree_prob_i)
            tree_prob_i_arg[np.arange(len(tree_prob_i)), tree_prob_i.argmax(1)] = 1
            trees_prob_sum += tree_prob_i_arg
        return trees_prob_sum / n


class XGBoostRegressor(Ensemble):
    """
    InMemoryModel Implementation of the XGBoost Regressor Algorithm.

    Parameters
    ----------
    trees: list[BinaryTreeRegressor]
        List of BinaryTrees for Regression.
    mean: float, optional
        Average of the response column.
    eta: float, optional
        Learning rate.
    """

    @property
    def _object_type(self) -> Literal["XGBoostRegressor"]:
        return "XGBoostRegressor"

    @property
    def _attributes(self) -> Literal["trees_", "mean_", "eta_"]:
        return ["trees_", "mean_", "eta_"]

    def __init__(
        self, trees: list[BinaryTreeRegressor], mean: float = 0.0, eta: float = 1.0,
    ) -> None:
        self.trees_ = copy.deepcopy(trees)
        self.mean_ = mean
        self.eta_ = learning_rate
        return None

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predicts using the Random Forest Regressor model.

        Parameters
        ----------
        X: ArrayLike
            The data on which to make the prediction.

        Returns
        -------
        numpy.array
            Predicted values.
        """
        trees_pred_sum = np.sum(self._predict_trees(X), axis=1)
        return trees_pred_sum * self.eta_ + self.mean_

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
        trees_pred = self._predict_trees_sql(X)
        return f"({' + '.join(trees_pred)}) * {self.eta_} + {self.mean_}"


class XGBoostClassifier(Ensemble, MulticlassClassifier):
    """
    InMemoryModel Implementation of the XGBoost Classifier Algorithm.

    Parameters
    ----------
    trees: list[BinaryTreeRegressor]
        List of BinaryTrees for Regression.
    logodds: ArrayLike[float], optional
        ArrayLike of the logodds of the response classes.
    classes: ArrayLike, optional
    	The model's classes.
    learning_rate: float, optional
        Learning rate.
    """

    @property
    def _object_type(self) -> Literal["XGBoostClassifier"]:
        return "XGBoostClassifier"

    @property
    def _attributes(self) -> Literal["trees_", "logodds_", "classes_", "eta_"]:
        return ["trees_", "logodds_", "classes_", "eta_"]

    def __init__(
        self,
        trees: list[BinaryTreeRegressor],
        logodds: ArrayLike,
        classes: ArrayLike = [],
        learning_rate: float = 1.0,
    ) -> None:
        self.trees_ = copy.deepcopy(trees)
        self.logodds_ = np.array(logodds)
        self.classes_ = np.array(classes)
        self.eta_ = learning_rate
        return None

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """
        Computes the model's probabilites using the input Matrix.

        Parameters
        ----------
        X: list / numpy.array
            The data on which to make the prediction.

        Returns
        -------
        numpy.array
            Probabilities.
        """
        trees_prob = 0
        for tree in self.trees_:
            trees_prob += tree.predict_proba(X)
        trees_prob = self.logodds_ + self.eta_ * trees_prob
        logit = 1 / (1 + np.exp(-trees_prob))
        softmax = logit / np.sum(logit, axis=1)[:, None]
        return softmax


class IsolationForest(Ensemble):
    """
    InMemoryModel Implementation of the Isolation Forest Algorithm.

    Parameters
    ----------
    trees: list[BinaryTreeAnomaly]
        list of BinaryTrees for Anomaly Detection.
    """

    @property
    def _object_type(self) -> Literal["IsolationForest"]:
        return "IsolationForest"

    @property
    def _attributes(self) -> Literal["trees_"]:
        return ["trees_"]

    def __init__(self, trees: list[BinaryTreeAnomaly]) -> None:
        self.trees_ = copy.deepcopy(trees)
        return None

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predicts using the Random Forest Regressor model.

        Parameters
        ----------
        X: ArrayLike
            The data on which to make the prediction.

        Returns
        -------
        numpy.array
            Predicted values.
        """
        return 2 ** (-np.average(self._predict_trees(X), axis=1))

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
        trees_pred = self._predict_trees_sql(X)
        return f"POWER(2, - (({' + '.join(trees_pred)}) / {len(trees_pred)}))"
