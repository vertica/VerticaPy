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
from typing import Literal

import numpy as np
from numpy.linalg import svd

from verticapy._typing import ArrayLike

from verticapy.machine_learning.memmodel.base import InMemoryModel


class PCA(InMemoryModel):
    """
    InMemoryModel implementation of the PCA algorithm.

    Parameters
    ----------
    principal_components: ArrayLike
        Matrix   of   the   principal   components.
    mean: ArrayLike
        List of the averages of each input feature.
    """

    # Properties.

    @property
    def object_type(self) -> Literal["PCA"]:
        return "PCA"

    @property
    def _attributes(self) -> list[str]:
        return ["principal_components_", "mean_"]

    # System & Special Methods.

    def __init__(self, principal_components: ArrayLike, mean: ArrayLike) -> None:
        self.principal_components_ = np.array(principal_components)
        self.mean_ = np.array(mean)

    # Prediction / Transformation Methods - IN MEMORY.

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Transforms and applies the PCA model to the input
        matrix.

        Parameters
        ----------
        X: ArrayLike
            The data on which to make the transformation.

        Returns
        -------
        numpy.array
            Transformed values.
        """
        X_trans = []
        n = self.principal_components_.shape[1]
        for i in range(n):
            X_trans += [
                np.sum((X - self.mean_) * self.principal_components_[:, i], axis=1)
            ]
        return np.column_stack(X_trans)

    # Prediction / Transformation Methods - IN DATABASE.

    def transform_sql(self, X: ArrayLike) -> list[str]:
        """
        Transforms and returns the SQL needed to deploy
        the PCA.

        Parameters
        ----------
        X: ArrayLike
            The names or values of the input predictors.

        Returns
        -------
        list
            SQL code.
        """
        if len(X) != len(self.mean_):
            raise ValueError(
                "The length of parameter 'X' must be equal to the length "
                "of the vector 'mean'."
            )
        sql = []
        for i in range(len(X)):
            sql_tmp = []
            for j in range(len(X)):
                sql_tmp += [
                    f"({X[j]} - {self.mean_[j]}) * {self.principal_components_[:, i][j]}"
                ]
            sql += [" + ".join(sql_tmp)]
        return sql

    # Special Methods - Matrix Rotation.

    @staticmethod
    def matrix_rotation(
        Phi: ArrayLike, gamma: float = 1.0, q: int = 20, tol: float = 1e-6
    ) -> None:
        """
        Performs an Oblimin  (Varimax, Quartimax) rotation on
        the input matrix.
        """
        # This piece of code was taken from
        # https://en.wikipedia.org/wiki/Talk:Varimax_rotation
        Phi = np.array(Phi)
        p, k = Phi.shape
        R = np.eye(k)
        d = 0
        for i in range(q):
            d_old = d
            Lambda = np.dot(Phi, R)
            u, s, vh = svd(
                np.dot(
                    Phi.T,
                    np.asarray(Lambda) ** 3
                    - (gamma / p)
                    * np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T, Lambda)))),
                )
            )
            R = np.dot(u, vh)
            d = np.sum(s)
            if d_old != 0 and d / d_old < 1 + tol:
                break
        return np.dot(Phi, R)

    def rotate(self, gamma: float = 1.0, q: int = 20, tol: float = 1e-6) -> None:
        """
        Performs an Oblimin (Varimax, Quartimax) rotation on the  PCA
        matrix.

        Parameters
        ----------
        gamma: float, optional
            Oblimin rotation factor, determines the type of rotation.
            It must be between 0.0 and 1.0.
                gamma = 0.0 results in a Quartimax rotation.
                gamma = 1.0 results in a Varimax rotation.
        q: int, optional
            Maximum number of iterations.
        tol: float, optional
            The  algorithm stops when the Frobenius norm of  gradient
            is less than tol.
        """
        res = self.matrix_rotation(self.principal_components_, gamma, q, tol)
        self.principal_components_ = res


class SVD(InMemoryModel):
    """
    InMemoryModel implementation of the SVD Algorithm.

    Parameters
    ----------
    vectors: ArrayLike
        Matrix of the right singular vectors.
    values: ArrayLike
        List of the singular values for each input
        feature.
    """

    # Properties.

    @property
    def object_type(self) -> Literal["SVD"]:
        return "SVD"

    @property
    def _attributes(self) -> list[str]:
        return ["vectors_", "values_"]

    # System & Special Methods.

    def __init__(self, vectors: ArrayLike, values: ArrayLike) -> None:
        self.vectors_ = np.array(vectors)
        self.values_ = np.array(values)

    # Prediction / Transformation Methods - IN MEMORY.

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Transforms and applies the SVD model to the input matrix.

        Parameters
        ----------
        X: ArrayLike
            The data on which to make the transformation.

        Returns
        -------
        numpy.array
            Transformed values.
        """
        X_trans = []
        n = self.vectors_.shape[1]
        for i in range(n):
            X_trans += [np.sum(X * self.vectors_[:, i] / self.values_[i], axis=1)]
        return np.column_stack(X_trans)

    # Prediction / Transformation Methods - IN DATABASE.

    def transform_sql(self, X: ArrayLike) -> list[str]:
        """
        Transforms and returns the SQL needed to deploy
        the PCA.

        Parameters
        ----------
        X: ArrayLike
            The names or values of the input predictors.

        Returns
        -------
        list
            SQL code.
        """
        if len(X) != len(self.values_):
            raise ValueError(
                "The length of parameter 'X' must be equal to the length "
                "of the vector 'values'."
            )
        sql = []
        for i in range(len(X)):
            sql_tmp = []
            for j in range(len(X)):
                sql_tmp += [f"{X[j]} * {self.vectors_[:, i][j]} / {self.values_[i]}"]
            sql += [" + ".join(sql_tmp)]
        return sql
