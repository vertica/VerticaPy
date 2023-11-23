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
from numpy.linalg import svd

from verticapy._typing import ArrayLike

from verticapy.machine_learning.memmodel.base import InMemoryModel


class PCA(InMemoryModel):
    """
    :py:meth:`verticapy.machine_learning.memmodel.base.InMemoryModel`
    implementation of the PCA algorithm.

    Parameters
    ----------
    principal_components: ArrayLike
        Matrix   of   the   principal   components.
    mean: ArrayLike
        List of the averages of each input feature.

    .. note::

        :py:meth:`verticapy.machine_learning.memmodel` are defined entirely
        by their attributes. For example, ``principal components`` and
        ``mean`` define a PCA model.

    Attributes
    ----------
    Attributes are identical to the input parameters, followed by an
    underscore ('_').

    Examples
    --------

    **Initalization**

    Import the required module.

    .. ipython:: python

        from verticapy.machine_learning.memmodel.decomposition import PCA

    A PCA model is defined by its principal components and mean value.
    In this example, we will use the following:

    .. ipython:: python

        principal_components = [[0.4, 0.5], [0.3, 0.2]]
        mean = [0.1, 0.3]

    Let's create a
    :py:meth:`verticapy.machine_learning.memmodel.decomposition.PCA`
    model.

    .. ipython:: python

        model_pca = PCA(principal_components, mean)

    Create a dataset.

    .. ipython:: python

        data = [[4, 5]]

    **Making In-Memory Transformation**

    Use
    :py:meth:`verticapy.machine_learning.memmodel.decomposition.PCA.transform`
    method to do transformation.

    .. ipython:: python

        model_pca.transform(data)

    **Deploy SQL Code**

    Let's use the following column names:

    .. ipython:: python

        cnames = ['col1', 'col2']

    Use
    :py:meth:`verticapy.machine_learning.memmodel.decomposition.PCA.transform_sql`
    method to get the SQL code needed to deploy the model
    using its attributes.

    .. ipython:: python

        model_pca.transform_sql(cnames)

    **Perform an Oblimin Rotation**

    Use
    :py:meth:`verticapy.machine_learning.memmodel.decomposition.PCA.rotate`
    method to perform Oblimin (Varimax, Quartimax) rotation on
    PCA matrix.

    .. ipython:: python

        model_pca.rotate()

    .. note::

        You can determine the type of rotation by adjusting value
        of gamma in
        :py:meth:`verticapy.machine_learning.memmodel.decomposition.PCA.rotate`
        method. It must be between 0.0 and 1.0.

    Use gamma = 0.0, for Quartimax rotation:

    .. ipython:: python

        gamma = 0.0
        model_pca.rotate(gamma)

    Use gamma = 1.0, for Varimax rotation:

    .. ipython:: python

        gamma = 1.0
        model_pca.rotate(gamma)

    Use
    :py:meth:`verticapy.machine_learning.memmodel.decomposition.PCA.get_attributes`
    method to check the attributes of the rotated model.

    .. ipython:: python

        model_pca.get_attributes()

    .. hint::
        This object can be pickled and used in any in-memory environment,
        just like `SKLEARN <https://scikit-learn.org/>`_ models.
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
        m, n = self.principal_components_.shape
        for i in range(n):
            sql_tmp = []
            for j in range(m):
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

            - gamma = 0.0 results in a Quartimax rotation.

            - gamma = 1.0 results in a Varimax rotation.

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
    :py:meth:`verticapy.machine_learning.memmodel.base.InMemoryModel`
    implementation of the SVD Algorithm.

    Parameters
    ----------
    vectors: ArrayLike
        Matrix of the right singular vectors.
    values: ArrayLike
        List of the singular values for each input
        feature.

    .. note::

        :py:meth:`verticapy.machine_learning.memmodel` are defined
        entirely by their attributes. For example, ``vectors``
        and 'values' define a SVD model.

    Attributes
    ----------
    Attributes are identical to the input parameters, followed by an
    underscore ('_').

    Examples
    --------

    **Initalization**

    Import the required module.

    .. ipython:: python

        from verticapy.machine_learning.memmodel.decomposition import SVD

    A SVD model is defined by its vectors and values.
    In this example, we will use the following:

    .. ipython:: python

        vectors = [[0.4, 0.5], [0.3, 0.2]]
        values = [0.1, 0.3]

    Let's create a
    :py:meth:`verticapy.machine_learning.memmodel.decomposition.SVD`
    model.

    .. ipython:: python

        model_svd = SVD(vectors, values)

    Create a dataset.

    .. ipython:: python

        data = [[0.3, 0.5]]

    **Making In-Memory Transformation**

    Use
    :py:meth:`verticapy.machine_learning.memmodel.decomposition.SVD.transform`
    method to do transformation.

    .. ipython:: python

        model_svd.transform(data)

    **Deploy SQL Code**

    Let's use the following column names:

    .. ipython:: python

        cnames = ['col1', 'col2']

    Use
    :py:meth:`verticapy.machine_learning.memmodel.decomposition.SVD.transform_sql`
    method to get the SQL code needed to deploy the model.
    using its attributes.

    .. ipython:: python

        model_svd.transform_sql(cnames)

    Use
    :py:meth:`verticapy.machine_learning.memmodel.decomposition.SVD.get_attributes`
    method to check the attributes of the rotated model.

    .. ipython:: python

        model_svd.get_attributes()

    .. hint::

        This object can be pickled and used in any in-memory environment,
        just like `SKLEARN <https://scikit-learn.org/>`_ models.
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
        if len(X) != len(self.vectors_):
            raise ValueError(
                "The length of parameter 'X' must be equal to the length "
                "of the vector 'values'."
            )
        sql = []
        m, n = self.vectors_.shape
        for i in range(n):
            sql_tmp = []
            for j in range(m):
                sql_tmp += [f"{X[j]} * {self.vectors_[:, i][j]} / {self.values_[i]}"]
            sql += [" + ".join(sql_tmp)]
        return sql
