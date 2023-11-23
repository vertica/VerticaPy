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
from typing import Literal, Optional, TYPE_CHECKING

import numpy as np

import verticapy._config.config as conf
from verticapy._typing import ArrayLike
from verticapy._utils._sql._format import clean_query, format_type

from verticapy.machine_learning.memmodel.base import InMemoryModel, MulticlassClassifier
from verticapy.machine_learning.memmodel.tree import (
    BinaryTreeAnomaly,
    BinaryTreeClassifier,
    BinaryTreeRegressor,
)

if TYPE_CHECKING and conf.get_import_success("graphviz"):
    from graphviz import Source


class Ensemble(InMemoryModel):
    """
    :py:meth:`verticapy.machine_learning.memmodel.base.InMemoryModel`
    implementation of ensemble algorithms.
    """

    # Prediction / Transformation Methods - IN MEMORY.

    def _predict_trees(self, X: ArrayLike) -> np.ndarray:
        """
        Predicts using all the model trees.
        """
        return np.column_stack([tree.predict(X) for tree in self.trees_])

    # Prediction / Transformation Methods - IN DATABASE.

    def _predict_trees_sql(self, X: ArrayLike) -> list[str]:
        """
        Predicts using all the model trees.
        """
        return [str(tree.predict_sql(X)) for tree in self.trees_]

    # Trees Representation Methods.

    def plot_tree(
        self,
        pic_path: Optional[str] = None,
        tree_id: int = 0,
        *args,
        **kwargs,
    ) -> "Source":
        """
        Draws the input tree. Requires the Graphviz module.

        Parameters
        ----------
        pic_path: str, optional
            Absolute  path to  save the image of the  tree.
        tree_id: int, optional
            Unique  tree identifier,   an  integer  in  the
            range [0, n_estimators - 1].
        *args, **kwargs: Any, optional
            Arguments to pass to the 'to_graphviz'  method.

        Returns
        -------
        graphviz.Source
            Graphviz object.
        """
        return self.trees_[tree_id].plot_tree(pic_path, *args, **kwargs)


class RandomForestRegressor(Ensemble):
    """
    :py:meth:`verticapy.machine_learning.memmodel.base.InMemoryModel`
    implementation of the random forest regressor algorithm.

    Parameters
    ----------
    trees: list[BinaryTreeRegressor]
        list of BinaryTrees for regression.

    Attributes
    ----------
    Attributes are identical to the input parameters, followed by an
    underscore ('_').

    Examples
    --------

    **Initalization**

    A Random Forest Regressor model is an ensemble of multiple binary
    tree regressor models. In this example, we will create three
    :py:meth:`verticapy.machine_learning.memmodel.tree.BinaryTreeRegressor`
    models:

    .. ipython:: python

        from verticapy.machine_learning.memmodel.tree import BinaryTreeRegressor

        model1 = BinaryTreeRegressor(
            children_left = [1, 3, None, None, None],
            children_right = [2, 4, None, None, None],
            feature = [0, 1, None, None, None],
            threshold = ["female", 30, None, None, None],
            value = [None, None, 3.0, 11.0, 23.5],
        )
        model2 = BinaryTreeRegressor(
            children_left = [1, 3, None, None, None],
            children_right = [2, 4, None, None, None],
            feature = [0, 1, None, None, None],
            threshold = ["female", 30, None, None, None],
            value = [None, None, -3, 12, 56],
        )
        model3 = BinaryTreeRegressor(
            children_left = [1, 3, None, None, None],
            children_right = [2, 4, None, None, None],
            feature = [0, 1, None, None, None],
            threshold = ["female", 30, None, None, None],
            value = [None, None, 1, 3, 6],
        )

    Now we will use above models to create
    :py:meth:`verticapy.machine_learning.memmodel.ensemble.RandomForestRegressor`
    model.

    .. ipython:: python

        from verticapy.machine_learning.memmodel.ensemble import RandomForestRegressor

        model_rfr = RandomForestRegressor(trees = [model1, model2, model3])

    Create a dataset.

    .. ipython:: python

        data = [["male", 100], ["female", 20], ["female", 50]]

    **Making In-Memory Predictions**

    Use
    :py:meth:`verticapy.machine_learning.memmodel.ensemble.RandomForestRegressor.predict`
    method to do predictions.

    .. ipython:: python

        model_rfr.predict(data)

    **Deploy SQL Code**

    Let's use the following column names:

    .. ipython:: python

        cnames = ["sex", "fare"]

    Use
    :py:meth:`verticapy.machine_learning.memmodel.ensemble.RandomForestRegressor.predict_sql`
    method to get the SQL code needed to deploy the model using
    its attributes.

    .. ipython:: python

        model_rfr.predict_sql(cnames)

    .. hint::

        This object can be pickled and used in any in-memory
        environment, just like `SKLEARN <https://scikit-learn.org/>`_
        models.

    **Drawing Trees**

    Use
    :py:meth:`verticapy.machine_learning.memmodel.ensemble.RandomForestRegressor.plot_tree`
    method to draw the input tree.

    .. code-block:: python

        model_rfr.plot_tree(tree_id = 0)

    .. ipython:: python
        :suppress:

        res = model_rfr.plot_tree(tree_id = 0)
        res.render(filename='figures/machine_learning_memmodel_tree_rndforestreg', format='png')

    .. image:: /../figures/machine_learning_memmodel_tree_rndforestreg.png

    .. important::

        :py:meth:`verticapy.machine_learning.memmodel.ensemble.RandomForestRegressor.plot_tree`
        requires the `Graphviz <https://graphviz.org/download/>`_ module.

    .. note::

        The above example is a very basic one. For
        other more detailed examples and customization
        options, please see :ref:`chart_gallery.tree`_
    """

    # Properties.

    @property
    def object_type(self) -> Literal["RandomForestRegressor"]:
        return "RandomForestRegressor"

    @property
    def _attributes(self) -> list[str]:
        return ["trees_"]

    # System & Special Methods.

    def __init__(self, trees: list[BinaryTreeRegressor]) -> None:
        self.trees_ = copy.deepcopy(trees)

    # Prediction / Transformation Methods - IN MEMORY.

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predicts using the Random Forest regressor model.

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

    # Prediction / Transformation Methods - IN DATABASE.

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
    :py:meth:`verticapy.machine_learning.memmodel.base.InMemoryModel`
    implementation of the random forest classifier algorithm.

    Parameters
    ----------
    trees: list[BinaryTreeClassifier]
        List of BinaryTrees for classification.
    classes: ArrayLike, optional
        The model's classes.

    Attributes
    ----------
    Attributes are identical to the input parameters, followed by an
    underscore ('_').

    Examples
    --------

    **Initalization**

    A Random Forest Classifier model is an ensemble of multiple binary
    tree classifier models. In this example, we will create three
    :py:meth:`verticapy.machine_learning.memmodel.tree.BinaryTreeClassifier`
    models:

    .. ipython:: python

        from verticapy.machine_learning.memmodel.tree import BinaryTreeClassifier

        model1 = BinaryTreeClassifier(
            children_left = [1, 3, None, None, None],
            children_right = [2, 4, None, None, None],
            feature = [0, 1, None, None, None],
            threshold = ["female", 30, None, None, None],
            value = [None, None, [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]],
            classes = ["a", "b", "c"],
        )
        model2 = BinaryTreeClassifier(
            children_left = [1, 3, None, None, None],
            children_right = [2, 4, None, None, None],
            feature = [0, 1, None, None, None],
            threshold = ["female", 30, None, None, None],
            value = [None, None, [0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.2, 0.2, 0.6]],
            classes = ["a", "b", "c"],
        )
        model3 = BinaryTreeClassifier(
            children_left = [1, 3, None, None, None],
            children_right = [2, 4, None, None, None],
            feature = [0, 1, None, None, None],
            threshold = ["female", 30, None, None, None],
            value = [None, None, [0.4, 0.4, 0.2], [0.2, 0.2, 0.6], [0.2, 0.5, 0.3]],
            classes = ["a", "b", "c"],
        )

    Now we will use above models to create
    :py:meth:`verticapy.machine_learning.memmodel.ensemble.RandomForestClassifier`
    model.

    .. ipython:: python

        from verticapy.machine_learning.memmodel.ensemble import RandomForestClassifier

        model_rfc = RandomForestClassifier(
            trees = [model1, model2, model3],
            classes = ["a", "b", "c"],
        )

    Create a dataset.

    .. ipython:: python

        data = [["male", 100], ["female", 20], ["female", 50]]

    **Making In-Memory Predictions**

    Use
    :py:meth:`verticapy.machine_learning.memmodel.ensemble.RandomForestClassifier.predict`
    method to do predictions.

    .. ipython:: python

        model_rfc.predict(data)

    Use
    :py:meth:`verticapy.machine_learning.memmodel.ensemble.RandomForestClassifier.predict_proba`
    method to compute the predicted probabilities for each class.

    .. ipython:: python

        model_rfc.predict_proba(data)

    **Deploy SQL Code**

    Let's use the following column names:

    .. ipython:: python

        cnames = ["sex", "fare"]

    Use
    :py:meth:`verticapy.machine_learning.memmodel.ensemble.RandomForestClassifier.predict_sql`
    method to get the SQL code needed to deploy the model using
    its attributes.

    .. ipython:: python

        model_rfc.predict_sql(cnames)

    Use
    :py:meth:`verticapy.machine_learning.memmodel.ensemble.RandomForestClassifier.predict_proba_sql`
    method to get the SQL code needed to deploy the model probabilities
    using its attributes.

    .. ipython:: python

        model_rfc.predict_proba_sql(cnames)

    .. hint::

        This object can be pickled and used in any in-memory
        environment, just like `SKLEARN <https://scikit-learn.org/>`_
        models.

    **Drawing Trees**

    Use
    :py:meth:`verticapy.machine_learning.memmodel.ensemble.RandomForestClassifier.plot_tree`
    method to draw the input tree.

    .. code-block:: python

        model_rfc.plot_tree(tree_id = 0)

    .. ipython:: python
        :suppress:

        res = model_rfc.plot_tree(tree_id = 0)
        res.render(filename='figures/machine_learning_memmodel_ensemble_rfclassifier', format='png')

    .. image:: /../figures/machine_learning_memmodel_ensemble_rfclassifier.png

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
    def object_type(self) -> Literal["RandomForestClassifier"]:
        return "RandomForestClassifier"

    @property
    def _attributes(self) -> list[str]:
        return ["trees_", "classes_"]

    # System & Special Methods.

    def __init__(
        self, trees: list[BinaryTreeClassifier], classes: Optional[ArrayLike] = None
    ) -> None:
        classes = format_type(classes, dtype=list)
        self.trees_ = copy.deepcopy(trees)
        if len(classes) == 0:
            self.classes_ = copy.deepcopy(trees[0].classes_)
        else:
            self.classes_ = np.array(classes)

    # Prediction / Transformation Methods - IN MEMORY.

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """
        Computes  the model's probabilites using  the
        input matrix.

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

    # Prediction / Transformation Methods - IN DATABASE.

    def predict_proba_sql(self, X: ArrayLike) -> list[str]:
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
        n = len(self.trees_)
        m = len(self.classes_)
        trees = []
        for i in range(n):
            value = []
            for v in self.trees_[i].value_:
                if v is None:
                    value += [v]
                else:
                    val_class_1 = np.zeros_like([v])
                    val_class_1[np.arange(1), np.array([v]).argmax(1)] = 1
                    value += [list(val_class_1[0])]
            tree = BinaryTreeClassifier(
                children_left=self.trees_[i].children_left_,
                children_right=self.trees_[i].children_right_,
                feature=self.trees_[i].feature_,
                threshold=self.trees_[i].threshold_,
                value=value,
                classes=self.trees_[i].classes_,
            )
            trees += [tree]
        trees_pred = [trees[i].predict_proba_sql(X) for i in range(n)]
        res = []
        for i in range(m):
            res += [f"({' + '.join([val[i] for val in trees_pred])}) / {n}"]
        return clean_query(res)


class XGBRegressor(Ensemble):
    """
    :py:meth:`verticapy.machine_learning.memmodel.base.InMemoryModel`
    implementation of the XGBoost regressor algorithm.

    Parameters
    ----------
    trees: list[BinaryTreeRegressor]
        List  of  BinaryTrees  for  regression.
    mean: float, optional
        Average   of   the   response   column.
    eta: float, optional
        Learning rate.

    Attributes
    ----------
    Attributes are identical to the input parameters, followed by an
    underscore ('_').

    Examples
    --------

    **Initalization**

    A  model is an ensemble of multiple binary tree regressors.
    In this example, we will create three
    :py:meth:`verticapy.machine_learning.memmodel.tree.BinaryTreeRegressor`
    models.

    .. ipython:: python

        from verticapy.machine_learning.memmodel.tree import BinaryTreeRegressor

        model1 = BinaryTreeRegressor(
            children_left = [1, 3, None, None, None],
            children_right = [2, 4, None, None, None],
            feature = [0, 1, None, None, None],
            threshold = ["female", 30, None, None, None],
            value = [None, None, 3, 11, 23],
        )
        model2 = BinaryTreeRegressor(
            children_left = [1, 3, None, None, None],
            children_right = [2, 4, None, None, None],
            feature = [0, 1, None, None, None],
            threshold = ["female", 30, None, None, None],
            value = [None, None, -3, 12, 56],
        )
        model3 = BinaryTreeRegressor(
            children_left = [1, 3, None, None, None],
            children_right = [2, 4, None, None, None],
            feature = [0, 1, None, None, None],
            threshold = ["female", 30, None, None, None],
            value = [None, None, 1, 3, 6],
        )

    Now we will use above models to create
    :py:meth:`verticapy.machine_learning.memmodel.ensemble.XGBRegressor`
    model.

    .. ipython:: python

        from verticapy.machine_learning.memmodel.ensemble import XGBRegressor

        model_xgbr = XGBRegressor(
            trees = [model1, model2, model3],
            mean = 2.5,
            eta = 0.9,
        )

    .. note::

        We have used *mean* that represents average of the response column
        and *eta* that represents learning rate of XG Boost regressor model.
        Both are optional parameters.

    Create a dataset.

    .. ipython:: python

        data = [["male", 100], ["female", 20], ["female", 50]]

    **Making In-Memory Predictions**

    Use
    :py:meth:`verticapy.machine_learning.memmodel.ensemble.XGBRegressor.predict`
    method to do predictions.

    .. ipython:: python

        model_xgbr.predict(data)

    **Deploy SQL Code**

    Let's use the following column names:

    .. ipython:: python

        cnames = ["sex", "fare"]

    Use
    :py:meth:`verticapy.machine_learning.memmodel.ensemble.XGBRegressor.predict_sql`
    method to get the SQL code needed to deploy the model using
    its attributes.

    .. ipython:: python

        model_xgbr.predict_sql(cnames)

    .. hint::

        This object can be pickled and used in any in-memory
        environment, just like `SKLEARN <https://scikit-learn.org/>`_
        models.

    **Drawing Trees**

    Use
    :py:meth:`verticapy.machine_learning.memmodel.ensemble.XGBRegressor.plot_tree`
    method to draw the input tree.

    .. code-block:: python

        model_xgbr.plot_tree(tree_id = 0)

    .. ipython:: python
        :suppress:

        res = model_xgbr.plot_tree(tree_id = 0)
        res.render(filename='figures/machine_learning_memmodel_tree_xgbreg', format='png')

    .. image:: /../figures/machine_learning_memmodel_tree_xgbreg.png

    .. important::

        :py:meth:`verticapy.machine_learning.memmodel.ensemble.XGBRegressor.plot_tree`
        requires the `Graphviz <https://graphviz.org/download/>`_ module.

    .. note::

        The above example is a very basic one. For
        other more detailed examples and customization
        options, please see :ref:`chart_gallery.tree`_
    """

    # Properties.

    @property
    def object_type(self) -> Literal["XGBRegressor"]:
        return "XGBRegressor"

    @property
    def _attributes(self) -> list[str]:
        return ["trees_", "mean_", "eta_"]

    # System & Special Methods.

    def __init__(
        self,
        trees: list[BinaryTreeRegressor],
        mean: float = 0.0,
        eta: float = 1.0,
    ) -> None:
        self.trees_ = copy.deepcopy(trees)
        self.mean_ = mean
        self.eta_ = eta

    # Prediction / Transformation Methods - IN MEMORY.

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

    # Prediction / Transformation Methods - IN DATABASE.

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


class XGBClassifier(Ensemble, MulticlassClassifier):
    """
    :py:meth:`verticapy.machine_learning.memmodel.base.InMemoryModel`
    implementation of the XGBoost classifier algorithm.

    Parameters
    ----------
    trees: list[BinaryTreeRegressor]
        List  of   BinaryTrees  for  regression.
    logodds: ArrayLike[float], optional
        ArrayLike of the logodds of the response
        classes.
    classes: ArrayLike, optional
        The model's classes.
    learning_rate: float, optional
        Learning rate.

    Attributes
    ----------
    Attributes are identical to the input parameters, followed by an
    underscore ('_').

    Examples
    --------

    **Initalization**

    A XGBoost Classifier model is an ensemble of multiple binary
    tree classifier models. In this example, we will create three
    :py:meth:`verticapy.machine_learning.memmodel.tree.BinaryTreeClassifier`
    models:

    .. ipython:: python

        from verticapy.machine_learning.memmodel.tree import BinaryTreeClassifier

        model1 = BinaryTreeClassifier(
            children_left = [1, 3, None, None, None],
            children_right = [2, 4, None, None, None],
            feature = [0, 1, None, None, None],
            threshold = ["female", 30, None, None, None],
            value = [None, None, [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]],
            classes = ["a", "b", "c"]
        )
        model2 = BinaryTreeClassifier(
            children_left = [1, 3, None, None, None],
            children_right = [2, 4, None, None, None],
            feature = [0, 1, None, None, None],
            threshold = ["female", 30, None, None, None],
            value = [None, None, [0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.2, 0.2, 0.6]],
            classes = ["a", "b", "c"],
        )
        model3 = BinaryTreeClassifier(
            children_left = [1, 3, None, None, None],
            children_right = [2, 4, None, None, None],
            feature = [0, 1, None, None, None],
            threshold = ["female", 30, None, None, None],
            value = [None, None, [0.4, 0.4, 0.2], [0.2, 0.2, 0.6], [0.2, 0.5, 0.3]],
            classes = ["a", "b", "c"],
        )

    Now we will use above models to create
    :py:meth:`verticapy.machine_learning.memmodel.ensemble.XGBClassifier`
    model.

    .. ipython:: python

        from verticapy.machine_learning.memmodel.ensemble import XGBClassifier

        model_xgbc = XGBClassifier(
            trees = [model1, model2, model3],
            classes = ["a", "b", "c"],
            logodds = [0.1, 0.12, 0.15],
            learning_rate = 0.1,
        )

    .. note::

        We have used *logodds* that represents logodds of the response
        column and *learning_rate* that represents learning rate of
        XGBoost regressor model. Both are optional parameters.

    Create a dataset.

    .. ipython:: python

        data = [["male", 100], ["female", 20], ["female", 50]]

    **Making In-Memory Predictions**

    Use
    :py:meth:`verticapy.machine_learning.memmodel.ensemble.XGBClassifier.predict`
    method to do predictions.

    .. ipython:: python

        model_xgbc.predict(data)

    Use
    :py:meth:`verticapy.machine_learning.memmodel.ensemble.XGBClassifier.predict_proba`
    method to compute the predicted probabilities for each class.

    .. ipython:: python

        model_xgbc.predict_proba(data)

    **Deploy SQL Code**

    Let's use the following column names:

    .. ipython:: python

        cnames = ["sex", "fare"]

    Use
    :py:meth:`verticapy.machine_learning.memmodel.ensemble.XGBClassifier.predict_sql`
    method to get the SQL code needed to deploy the model using
    its attributes.

    .. ipython:: python

        model_xgbc.predict_sql(cnames)

    Use
    :py:meth:`verticapy.machine_learning.memmodel.ensemble.XGBClassifier.predict_proba_sql`
    method to get the SQL code needed to deploy the model probabilities
    using its attributes.

    .. ipython:: python

        model_xgbc.predict_proba_sql(cnames)

    .. hint::

        This object can be pickled and used in any in-memory
        environment, just like `SKLEARN <https://scikit-learn.org/>`_
        models.

    **Drawing Trees**

    Use
    :py:meth:`verticapy.machine_learning.memmodel.ensemble.XGBClassifier.plot_tree`
    method to draw the input tree.

    .. code-block:: python

        model_xgbc.plot_tree(tree_id = 0)

    .. ipython:: python
        :suppress:

        res = model_xgbc.plot_tree(tree_id = 0)
        res.render(filename='figures/machine_learning_memmodel_ensemble_xgbclassifier', format='png')

    .. image:: /../figures/machine_learning_memmodel_ensemble_xgbclassifier.png

    .. important::

        :py:meth:`verticapy.machine_learning.memmodel.ensemble.XGBClassifier.plot_tree`
        requires the `Graphviz <https://graphviz.org/download/>`_ module.

    .. note::

        The above example is a very basic one. For
        other more detailed examples and customization
        options, please see :ref:`chart_gallery.tree`_
    """

    # Properties.

    @property
    def object_type(self) -> Literal["XGBClassifier"]:
        return "XGBClassifier"

    @property
    def _attributes(self) -> list[str]:
        return ["trees_", "logodds_", "classes_", "eta_"]

    # System & Special Methods.

    def __init__(
        self,
        trees: list[BinaryTreeRegressor],
        logodds: ArrayLike,
        classes: Optional[ArrayLike] = None,
        learning_rate: float = 1.0,
    ) -> None:
        classes = format_type(classes, dtype=list)
        self.trees_ = copy.deepcopy(trees)
        self.logodds_ = np.array(logodds)
        if len(classes) == 0:
            self.classes_ = copy.deepcopy(trees[0].classes_)
        else:
            self.classes_ = np.array(classes)
        self.eta_ = learning_rate

    # Prediction / Transformation Methods - IN MEMORY.

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """
        Computes the model's probabilites using the input
        matrix.

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
        return logit / np.sum(logit, axis=1)[:, None]

    # Prediction / Transformation Methods - IN DATABASE.

    def predict_proba_sql(self, X: ArrayLike) -> list[str]:
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
        n = len(self.trees_)
        m = len(self.classes_)
        proba = []
        trees_pred = [self.trees_[i].predict_proba_sql(X) for i in range(n)]
        for i in range(m):
            proba += [
                f"""(1 / (1 + EXP(- ({self.logodds_[i]} + {self.eta_} 
                     * ({' + '.join([prob[i] for prob in trees_pred])})))))"""
            ]
        proba_sum = f"({' + '.join(proba)})"
        return clean_query([f"{p} / {proba_sum}" for p in proba])


class IsolationForest(Ensemble):
    """
    :py:meth:`verticapy.machine_learning.memmodel.base.InMemoryModel`
    implementation of the isolation forest algorithm.

    Parameters
    ----------
    trees: list[BinaryTreeAnomaly]
        list of BinaryTrees for anomaly detection.

    Attributes
    ----------
    Attributes are identical to the input parameters, followed by an
    underscore ('_').

    Examples
    --------

    **Initalization**

    An Isolation Forest model is an ensemble of multiple binary tree
    anomaly models. In this example, we will create three
    :py:meth:`verticapy.machine_learning.memmodel.tree.BinaryTreeAnomaly`
    models:

    .. ipython:: python

        from verticapy.machine_learning.memmodel.tree import BinaryTreeAnomaly

        model1 = BinaryTreeAnomaly(
            children_left = [1, 3, None, None, None],
            children_right = [2, 4, None, None, None],
            feature = [0, 1, None, None, None],
            threshold = ["female", 30, None, None, None],
            value = [None, None, [2, 10], [3, 4], [7, 8]],
            psy = 100,
        )
        model2 = BinaryTreeAnomaly(
            children_left = [1, 3, None, None, None],
            children_right = [2, 4, None, None, None],
            feature = [0, 1, None, None, None],
            threshold = ["female", 30, None, None, None],
            value = [None, None, [1, 11], [2, 5], [5, 10]],
            psy = 100,
        )
        model3 = BinaryTreeAnomaly(
            children_left = [1, 3, None, None, None],
            children_right = [2, 4, None, None, None],
            feature = [0, 1, None, None, None],
            threshold = ["female", 30, None, None, None],
            value = [None, None, [3, 9], [1, 6], [8, 7]],
            psy = 100,
        )

    Now we will use above models to create
    :py:meth:`verticapy.machine_learning.memmodel.ensemble.IsolationForest`
    model.

    .. ipython:: python

        from verticapy.machine_learning.memmodel.ensemble import IsolationForest

        model_isf = IsolationForest(trees = [model1, model2, model3])

    Create a dataset.

    .. ipython:: python

        data = [["male", 100], ["female", 20], ["female", 50]]

    **Making In-Memory Predictions**

    Use
    :py:meth:`verticapy.machine_learning.memmodel.ensemble.IsolationForest.predict`
    method to do predictions.

    .. ipython:: python

        model_isf.predict(data)

    **Deploy SQL Code**

    Let's use the following column names:

    .. ipython:: python

        cnames = ["sex", "fare"]

    Use
    :py:meth:`verticapy.machine_learning.memmodel.ensemble.IsolationForest.predict_sql`
    method to get the SQL code needed to deploy the model using
    its attributes.

    .. ipython:: python

        model_isf.predict_sql(cnames)

    .. hint::

        This object can be pickled and used in any in-memory
        environment, just like `SKLEARN <https://scikit-learn.org/>`_
        models.

    **Drawing Trees**

    Use
    :py:meth:`verticapy.machine_learning.memmodel.ensemble.IsolationForest.plot_tree`
    method to draw the input tree.

    .. code-block:: python

        model_isf.plot_tree(tree_id = 0)

    .. ipython:: python
        :suppress:

        res = model_isf.plot_tree(tree_id = 0)
        res.render(filename='figures/machine_learning_memmodel_ensemble_iforest', format='png')

    .. image:: /../figures/machine_learning_memmodel_ensemble_iforest.png

    .. important::

        :py:meth:`verticapy.machine_learning.memmodel.ensemble.IsolationForest.plot_tree`
        requires the `Graphviz <https://graphviz.org/download/>`_ module.

    .. note::

        The above example is a very basic one. For
        other more detailed examples and customization
        options, please see :ref:`chart_gallery.tree`_
    """

    # Properties.

    @property
    def object_type(self) -> Literal["IsolationForest"]:
        return "IsolationForest"

    @property
    def _attributes(self) -> list[str]:
        return ["trees_"]

    # System & Special Methods.

    def __init__(self, trees: list[BinaryTreeAnomaly]) -> None:
        self.trees_ = copy.deepcopy(trees)

    # Prediction / Transformation Methods - IN MEMORY.

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predicts using the isolation forest model.

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

    # Prediction / Transformation Methods - IN DATABASE.

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
