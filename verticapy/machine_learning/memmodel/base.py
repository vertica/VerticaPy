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
from abc import abstractmethod
from typing import Literal

import numpy as np

from verticapy._typing import ArrayLike
from verticapy._utils._sql._format import clean_query, format_magic


class InMemoryModel:
    """
    Base Class for In-Memory models.
    They can be used to score in
    other DBs or in-memory.

    Examples
    --------
    This is a base class. To see a comprehensive
    example specific to your class of interest,
    please refer to that particular class.
    """

    # Properties.

    @property
    @abstractmethod
    def object_type(self) -> str:
        """Must be overridden in child class"""
        raise NotImplementedError

    @property
    @abstractmethod
    def _attributes(self) -> list[str]:
        """Must be overridden in child class"""
        raise NotImplementedError

    # System & Special Methods.

    def __repr__(self) -> str:
        """Returns the model Representation."""
        return f"<{self.object_type}>"

    # Attributes Methods.

    def get_attributes(self) -> dict:
        """
        Returns the model attributes.

        Examples
        --------
        Import the required module.

        .. ipython:: python

            from verticapy.machine_learning.memmodel.linear_model import LinearModel

        We will use the
        following attributes:

        .. ipython:: python

            coefficients = [0.5, 1.2]
            intercept = 2.0

        Let's create a model.

        .. ipython:: python

            model_lm = LinearModel(coefficients, intercept)

        Let's get the model attributes.

        .. ipython:: python

            model_lm.get_attributes()

        .. important::

            For this example, a specific model is
            utilized, and it may not correspond
            exactly to the model you are working
            with. To see a comprehensive example
            specific to your class of interest,
            please refer to that particular class.
        """
        attributes_ = {}
        for att in self._attributes:
            attributes_[att[:-1]] = copy.deepcopy(getattr(self, att))
        return attributes_

    def set_attributes(self, **kwargs) -> None:
        """
        Sets the model attributes.

        Examples
        --------
        Import the required module.

        .. ipython:: python

            from verticapy.machine_learning.memmodel.linear_model import LinearModel

        We will use the
        following attributes:

        .. ipython:: python

            coefficients = [0.5, 1.2]
            intercept = 2.0

        Let's create a model.

        .. ipython:: python

            model_lm = LinearModel(coefficients, intercept)

        Let's get the model attributes.

        .. ipython:: python

            model_lm.get_attributes()

        Change the model attributes.

        .. ipython:: python

            model_lm.set_attributes(intercept = 4.0)

        Get the model attributes again.

        .. ipython:: python

            model_lm.get_attributes()

        .. important::

            For this example, a specific model is
            utilized, and it may not correspond
            exactly to the model you are working
            with. To see a comprehensive example
            specific to your class of interest,
            please refer to that particular class.
        """
        attributes_ = {**self.get_attributes(), **kwargs}
        self.__init__(**attributes_)


class MulticlassClassifier(InMemoryModel):
    """
    Class to represent any in-memory multiclass classifier.
    """

    # Properties.

    @property
    def object_type(self) -> Literal["MulticlassClassifier"]:
        return "MulticlassClassifier"

    # Prediction | Transformation Methods - IN MEMORY.

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predicts using the input matrix.

        Parameters
        ----------
        X: ArrayLike
            The data on which to
            make the prediction.

        Returns
        -------
        numpy.array
            Predicted values.

        Examples
        --------
        Import the required module.

        .. ipython:: python

            from verticapy.machine_learning.memmodel.naive_bayes import NaiveBayes

        Let's define attributes representing
        each input feature:

        .. ipython:: python

            attributes = [
                {
                    "type": "gaussian",
                    "C": {"mu": 63.9878308300395, "sigma_sq": 7281.87598377196},
                    "Q": {"mu": 13.0217386792453, "sigma_sq": 211.626862330204},
                    "S": {"mu": 27.6928120412844, "sigma_sq": 1428.57067393938},
                },
                {
                    "type": "multinomial",
                    "C": 0.771666666666667,
                    "Q": 0.910714285714286,
                    "S": 0.878216123499142,
                },
                {
                    "type": "bernoulli",
                    "C": 0.771666666666667,
                    "Q": 0.910714285714286,
                    "S": 0.878216123499142,
                },
                {
                    "type": "categorical",
                    "C": {
                        "female": 0.407843137254902,
                        "male": 0.592156862745098,
                    },
                    "Q": {
                        "female": 0.416666666666667,
                        "male": 0.583333333333333,
                    },
                    "S": {
                        "female": 0.406666666666667,
                        "male": 0.593333333333333,
                    },
                },
            ]

        We also need to provide class names
        and their prior probabilities.

        .. ipython:: python

            prior = [0.8, 0.1, 0.1]
            classes = ["C", "Q", "S"]

        Let's create a model.

        .. ipython:: python

            model_nb = NaiveBayes(attributes, prior, classes)

        Create a dataset.

        .. ipython:: python

            data = [
                [40.0, 1, True, "male"],
                [60.0, 3, True, "male"],
                [15.0, 2, False, "female"],
            ]

        Compute the predictions.

        .. ipython:: python

            model_nb.predict(data)

        .. important::

            For this example, a specific model is
            utilized, and it may not correspond
            exactly to the model you are working
            with. To see a comprehensive example
            specific to your class of interest,
            please refer to that particular class.
        """
        res = np.argmax(self.predict_proba(X), axis=1)
        if len(self.classes_) > 0:
            res = np.array([self.classes_[i] for i in res])
        return res

    # Prediction | Transformation Methods - IN DATABASE.

    def predict_sql(self, X: ArrayLike) -> str:
        """
        Returns the SQL code needed to deploy the model.

        Parameters
        ----------
        X: ArrayLike
            The names or values of
            the input predictors.

        Returns
        -------
        str
            SQL code.

        Examples
        --------
        Import the required module.

        .. ipython:: python

            from verticapy.machine_learning.memmodel.naive_bayes import NaiveBayes

        Let's define attributes representing
        each input feature:

        .. ipython:: python

            attributes = [
                {
                    "type": "gaussian",
                    "C": {"mu": 63.9878308300395, "sigma_sq": 7281.87598377196},
                    "Q": {"mu": 13.0217386792453, "sigma_sq": 211.626862330204},
                    "S": {"mu": 27.6928120412844, "sigma_sq": 1428.57067393938},
                },
                {
                    "type": "multinomial",
                    "C": 0.771666666666667,
                    "Q": 0.910714285714286,
                    "S": 0.878216123499142,
                },
                {
                    "type": "bernoulli",
                    "C": 0.771666666666667,
                    "Q": 0.910714285714286,
                    "S": 0.878216123499142,
                },
                {
                    "type": "categorical",
                    "C": {
                        "female": 0.407843137254902,
                        "male": 0.592156862745098,
                    },
                    "Q": {
                        "female": 0.416666666666667,
                        "male": 0.583333333333333,
                    },
                    "S": {
                        "female": 0.406666666666667,
                        "male": 0.593333333333333,
                    },
                },
            ]

        We also need to provide class names
        and their prior probabilities.

        .. ipython:: python

            prior = [0.8, 0.1, 0.1]
            classes = ["C", "Q", "S"]

        Let's create a model.

        .. ipython:: python

            model_nb = NaiveBayes(attributes, prior, classes)

        Let's use the following column names:

        .. ipython:: python

            cnames = ["age", "pclass", "survived", "sex"]

        Get the SQL code needed
        to deploy the model.

        .. ipython:: python

            model_nb.predict_sql(cnames)

        .. important::

            For this example, a specific model is
            utilized, and it may not correspond
            exactly to the model you are working
            with. To see a comprehensive example
            specific to your class of interest,
            please refer to that particular class.
        """
        if hasattr(self, "_predict_score_sql"):
            trees_pred = self._predict_score_sql(X)
        else:
            trees_pred = self.predict_proba_sql(X)
        m = len(trees_pred)
        if len(self.classes_) > 0:
            classes_ = self.classes_
        else:
            classes_ = [i for i in range(m)]
        if m == 2:
            res = f"""
                (CASE 
                    WHEN {trees_pred[1]} > 0.5 
                        THEN {classes_[1]} 
                    ELSE {classes_[0]} 
                END)"""
        else:
            sql = []
            for i in range(m):
                max_sql = []
                for j in range(i):
                    max_sql += [f"{trees_pred[i]} >= {trees_pred[j]}"]
                sql += [" AND ".join(max_sql)]
            sql = sql[1:]
            sql.reverse()
            res = f"""
                CASE 
                    WHEN {' OR '.join([f"{x} IS NULL" for x in X])} 
                    THEN NULL"""
            for i in range(m - 1):
                class_i = format_magic(classes_[m - i - 1])
                res += f" WHEN {sql[i]} THEN {class_i}"
            classes_0 = format_magic(classes_[0])
            res += f" ELSE {classes_0} END"
        return clean_query(res)
