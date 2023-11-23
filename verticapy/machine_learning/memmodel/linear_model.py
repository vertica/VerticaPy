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
from typing import Literal

import numpy as np

from verticapy._typing import ArrayLike

from verticapy.machine_learning.memmodel.base import InMemoryModel


class LinearModel(InMemoryModel):
    """
    :py:meth:`verticapy.machine_learning.memmodel.base.InMemoryModel`
    implementation of linear algorithms.

    Parameters
    ----------
    coef: ArrayLike
        ArrayLike of the model's coefficients.
    intercept: float, optional
        The intercept or constant value.

    .. note::

        :py:meth:`verticapy.machine_learning.memmodel` are defined entirely
        by their attributes. For example, ``coefficients`` and ``intercept``
        define a linear regression model.

    Attributes
    ----------
    Attributes are identical to the input parameters, followed by an
    underscore ('_').

    Examples
    --------

    **Initalization**

    Import the required module.

    .. ipython:: python

        from verticapy.machine_learning.memmodel.linear_model import LinearModel

    A linear model is defined by its coefficients and an intercept
    value. In this example, we will use the following:

    .. ipython:: python

        coefficients = [0.5, 1.2]
        intercept = 2.0

    Let's create a :py:meth:`verticapy.machine_learning.memmodel.linear_model`.

    .. ipython:: python

        model_lm = LinearModel(coefficients, intercept)

    Create a dataset.

    .. ipython:: python

        data = [[1.0, 0.3], [2.0, -0.6]]

    **Making In-Memory Predictions**

    Use
    :py:meth:`verticapy.machine_learning.memmodel.linear_model.LinearModel.predict`
    method to do predictions.

    .. ipython:: python

        model_lm.predict(data)

    **Deploy SQL Code**

    Let's use the following column names:

    .. ipython:: python

        cnames = ['col1', 'col2']

    Use
    :py:meth:`verticapy.machine_learning.memmodel.linear_model.LinearModel.predict_sql`
    method to get the SQL code needed to deploy the model
    using its attributes.

    .. ipython:: python

        model_lm.predict_sql(cnames)

    .. hint::

        This object can be pickled and used in any in-memory environment,
        just like `SKLEARN <https://scikit-learn.org/>`_ models.
    """

    # Properties.

    @property
    def object_type(self) -> Literal["LinearModel"]:
        return "LinearModel"

    @property
    def _attributes(self) -> list[str]:
        return ["coef_", "intercept_"]

    # System & Special Methods.

    def __init__(self, coef: ArrayLike, intercept: float = 0.0) -> None:
        self.coef_ = np.array(coef)
        self.intercept_ = intercept

    # Prediction / Transformation Methods - IN MEMORY.

    def _predict_regression(self, X: ArrayLike) -> np.ndarray:
        """
        Computes the model's score using the input Matrix.
        """
        return self.intercept_ + np.sum(self.coef_ * np.array(X), axis=1)

    def _predict_logit(self, X: ArrayLike) -> np.ndarray:
        """
        Computes the model's logit score using the input Matrix.
        """
        return 1 / (1 + np.exp(-(self._predict_regression(X))))

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predicts using the input Matrix.

        Parameters
        ----------
        X: ArrayLike
            The data on which to make the prediction.

        Returns
        -------
        numpy.array
            Predicted values.
        """
        return self._predict_regression(X)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """
        Computes the model's probabilites using the input matrix.

        Parameters
        ----------
        X: ArrayLike
            The data on which to make the prediction.

        Returns
        -------
        numpy.array
            Probabilities.
        """
        probability_1 = self._predict_logit(X)
        return np.column_stack((1 - probability_1, probability_1))

    # Prediction / Transformation Methods - IN DATABASE.

    def _predict_regression_sql(self, X: ArrayLike) -> str:
        """
        Returns the model's SQL score using the input
        matrix.
        """
        if len(X) != len(self.coef_):
            raise ValueError(
                "The length of parameter 'X' must be equal to the number of coefficients."
            )
        sql = [str(self.intercept_)] + [
            f"{self.coef_[idx]} * {(X[idx])}" for idx in range(len(self.coef_))
        ]
        return " + ".join(sql)

    def _predict_logit_sql(self, X: ArrayLike) -> str:
        """
        Returns the model's SQL logit score using the
        input Matrix.
        """
        return f"1 / (1 + EXP(- ({self._predict_regression_sql(X)})))"

    def predict_sql(self, X: ArrayLike) -> str:
        """
        Returns the SQL code needed to deploy the model
        using its attributes.

        Parameters
        ----------
        X: ArrayLike
            The names or values of the input predictors.

        Returns
        -------
        str
            SQL code.
        """
        return self._predict_regression_sql(X)

    def predict_proba_sql(self, X: ArrayLike) -> list[str]:
        """
        Returns the SQL code needed to deploy the model
        probabilities using its attributes.

        Parameters
        ----------
        X: ArrayLike
            The names or values of the input predictors.

        Returns
        -------
        list
            SQL code.
        """
        probability_1 = self._predict_logit_sql(X)
        return [f"1 - ({probability_1})", probability_1]


class LinearModelClassifier(LinearModel):
    """
    :py:meth:`verticapy.machine_learning.memmodel.base.InMemoryModel`
    Implementation of linear algorithms for classification.

    Parameters
    ----------
    coefficients: ArrayLike
        ArrayLike of the model's coefficients.
    intercept: float, optional
        The intercept or constant value.

    Attributes
    ----------
    Attributes are identical to the input parameters, followed by an
    underscore ('_').

    Examples
    --------

    **Initalization**

    Import the required module.

    .. ipython:: python

        from verticapy.machine_learning.memmodel.linear_model import LinearModelClassifier

    A linear classifier model is defined by its coefficients and
    an intercept value. In this example, we will use the following:

    .. ipython:: python

        coefficients = [0.5, 1.2]
        intercept = 2.0

    Let's create a :py:meth:`verticapy.machine_learning.memmodel.linear_model`.

    .. ipython:: python

        model_lmc = LinearModelClassifier(coefficients, intercept)

    Create a dataset.

    .. ipython:: python

        data = [[1.0, 0.3], [-0.5, -0.8]]

    **Making In-Memory Predictions**

    Use
    :py:meth:`verticapy.machine_learning.memmodel.linear_model.LinearModelClassifier.predict`
    method to do predictions.

    .. ipython:: python

        model_lmc.predict(data)

    Use
    :py:meth:`verticapy.machine_learning.memmodel.linear_model.LinearModel.predict_proba`
    method to calculate the predicted probabilities for
    each class.

    .. ipython:: python

        model_lmc.predict_proba(data)

    **Deploy SQL Code**

    Let's use the following column names:

    .. ipython:: python

        cnames = ['col1', 'col2']

    Use
    :py:meth:`verticapy.machine_learning.memmodel.linear_model.LinearModelClassifier.predict_sql`
    method to get the SQL code needed to deploy the model
    using its attributes.

    .. ipython:: python

        model_lmc.predict_sql(cnames)

    Use
    :py:meth:`verticapy.machine_learning.memmodel.linear_model.LinearModel.predict_proba_sql`
    method to get the SQL code needed to deploy the model
    that computes predicted probabilities.

    .. ipython:: python

        model_lmc.predict_proba_sql(cnames)

    .. hint::

        This object can be pickled and used in any in-memory environment,
        just like `SKLEARN <https://scikit-learn.org/>`_ models.
    """

    # Properties.

    @property
    def object_type(self) -> Literal["LinearModelClassifier"]:
        return "LinearModelClassifier"

    # Prediction / Transformation Methods - IN MEMORY.

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predicts using the input matrix.

        Parameters
        ----------
        X: ArrayLike
            The data on which to make the prediction.

        Returns
        -------
        numpy.array
            Predicted values.
        """
        return np.where(self._predict_logit(X) > 0.5, 1, 0)

    # Prediction / Transformation Methods - IN DATABASE.

    def predict_sql(self, X: ArrayLike) -> str:
        """
        Returns the SQL code needed to deploy the model
        using its attributes.

        Parameters
        ----------
        X: ArrayLike
            The names or values of the input predictors.

        Returns
        -------
        str
            SQL code.
        """
        return f"(({self._predict_logit_sql(X)}) > 0.5)::int"
