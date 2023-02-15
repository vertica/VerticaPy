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
import numpy as np
from typing import Union

# VerticaPy Modules
from verticapy.errors import ParameterError

# This piece of code was taken from
# https://en.wikipedia.org/wiki/Talk:Varimax_rotation
def matrix_rotation(
    Phi: Union[list, np.ndarray],
    gamma: Union[int, float] = 1.0,
    q: int = 20,
    tol: float = 1e-6,
):
    """
Performs a Oblimin (Varimax, Quartimax) rotation on the the model's 
PCA matrix.

Parameters
----------
Phi: list / numpy.array
    input matrix.
gamma: float, optional
    Oblimin rotation factor, determines the type of rotation.
    It must be between 0.0 and 1.0.
        gamma = 0.0 results in a Quartimax rotation.
        gamma = 1.0 results in a Varimax rotation.
q: int, optional
    Maximum number of iterations.
tol: float, optional
    The algorithm stops when the Frobenius norm of gradient is less than tol.

Returns
-------
model
    The model.
    """
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


def transform_from_pca(
    X: Union[list, np.ndarray],
    principal_components: Union[list, np.ndarray],
    mean: Union[list, np.ndarray],
) -> np.ndarray:
    """
    Transforms the data with a PCA model using the input attributes.

    Parameters
    ----------
    X: list / numpy.array
        Data to transform.
    principal_components: list / numpy.array
        Matrix of the principal components.
    mean: list / numpy.array
        List of the averages of each input feature.

    Returns
    -------
    numpy.array
        Transformed data
    """
    pca_values = np.array(principal_components)
    result = X - np.array(mean)
    L, n = [], len(principal_components[0])
    for i in range(n):
        L += [np.sum(result * pca_values[:, i], axis=1)]
    return np.column_stack(L)


def sql_from_pca(
    X: Union[list, np.ndarray],
    principal_components: Union[list, np.ndarray],
    mean: Union[list, np.ndarray],
) -> list:
    """
    Returns the SQL code needed to deploy a PCA model using its attributes.

    Parameters
    ----------
    X: list / numpy.array
        Names or values of the input predictors.
    principal_components: list / numpy.array
        Matrix of the principal components.
    mean: list / numpy.array
        List of the averages of each input feature.

    Returns
    -------
    list
        SQL code
    """
    assert len(X) == len(mean), ParameterError(
        "The length of parameter 'X' must be equal to the length of the vector 'mean'."
    )
    sql = []
    for i in range(len(X)):
        sql_tmp = []
        for j in range(len(X)):
            sql_tmp += [
                f"({X[j]} - {mean[j]}) * {[pc[i] for pc in principal_components][j]}"
            ]
        sql += [" + ".join(sql_tmp)]
    return sql


def transform_from_svd(
    X: Union[list, np.ndarray],
    vectors: Union[list, np.ndarray],
    values: Union[list, np.ndarray],
) -> np.ndarray:
    """
    Transforms the data with an SVD model using the input attributes.

    Parameters
    ----------
    X: list / numpy.array
        Data to transform.
    vectors: list / numpy.array
        Matrix of the right singular vectors.
    values: list / numpy.array
        List of the singular values for each input feature.

    Returns
    -------
    numpy.array
        Transformed data
    """
    svd_vectors = np.array(vectors)
    L, n = [], len(svd_vectors[0])
    for i in range(n):
        L += [np.sum(X * svd_vectors[:, i] / values[i], axis=1)]
    return np.column_stack(L)


def sql_from_svd(
    X: Union[list, np.ndarray],
    vectors: Union[list, np.ndarray],
    values: Union[list, np.ndarray],
) -> list:
    """
    Returns the SQL code needed to deploy a SVD model using its attributes.

    Parameters
    ----------
    X: list / numpy.array
        input predictors name or values.
    vectors: list / numpy.array
        List of the model's right singular vectors.
    values: list / numpy.array
        List of the singular values for each input feature.

    Returns
    -------
    list
        SQL code
    """
    assert len(X) == len(values), ParameterError(
        "The length of parameter 'X' must be equal to the length of the vector 'values'."
    )
    sql = []
    for i in range(len(X)):
        sql_tmp = []
        for j in range(len(X)):
            sql_tmp += [f"{X[j]} * {[pc[i] for pc in vectors][j]} / {values[i]}"]
        sql += [" + ".join(sql_tmp)]
    return sql
