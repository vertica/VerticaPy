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
    Base Class for In-Memory models. They can be used
    to score in other DBs or in-memory.
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
        attributes_ = {}
        for att in self._attributes:
            attributes_[att[:-1]] = copy.deepcopy(getattr(self, att))
        return attributes_

    def set_attributes(self, **kwargs) -> None:
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
        res = np.argmax(self.predict_proba(X), axis=1)
        if len(self.classes_) > 0:
            res = np.array([self.classes_[i] for i in res])
        return res

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
