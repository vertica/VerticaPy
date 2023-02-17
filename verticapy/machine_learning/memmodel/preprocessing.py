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
import numpy as np

from verticapy.errors import ParameterError


def transform_from_normalizer(
    X: Union[list, np.ndarray],
    values: Union[list, np.ndarray],
    method: Literal["zscore", "robust_zscore", "minmax"] = "zscore",
) -> np.ndarray:
    """
    Transforms the data with a normalizer model using the input attributes.

    Parameters
    ----------
    X: list / numpy.array
        The data to transform.
    values: list / numpy.array
        List of tuples. These tuples depend on the specified method:
            'zscore': (mean, std)
            'robust_zscore': (median, mad)
            'minmax': (min, max)
    method: str, optional
        The model's category, one of the following: 'zscore', 'robust_zscore', or 'minmax'.

    Returns
    -------
    numpy.array
        Transformed data
    """
    a, b = (
        np.array([elem[0] for elem in values]),
        np.array([elem[1] for elem in values]),
    )
    if method == "minmax":
        b = b - a
    return (np.array(X) - a) / b


def sql_from_normalizer(
    X: Union[list, np.ndarray],
    values: Union[list, np.ndarray],
    method: Literal["zscore", "robust_zscore", "minmax"] = "zscore",
) -> list:
    """
    Returns the SQL code needed to deploy a normalizer model using its attributes.

    Parameters
    ----------
    X: list / numpy.array
        Names or values of the input predictors.
    values: list / numpy.array
        List of tuples, including the model's attributes. These required tuple  
        depends on the specified method:
            'zscore': (mean, std)
            'robust_zscore': (median, mad)
            'minmax': (min, max)
    method: str, optional
        The model's category, one of the following: 'zscore', 'robust_zscore', or 'minmax'.

    Returns
    -------
    list
        SQL code
    """
    assert len(X) == len(values), ParameterError(
        "The length of parameter 'X' must be equal to the length of the list 'values'."
    )
    sql = []
    for i in range(len(X)):
        den = values[i][1] - values[i][0] if method == "minmax" else values[i][1]
        sql += [f"({X[i]} - {values[i][0]}) / {den}"]
    return sql


def transform_from_one_hot_encoder(
    X: Union[list, np.ndarray],
    categories: Union[list, np.ndarray],
    drop_first: bool = False,
) -> np.ndarray:
    """
    Transforms the data with a one-hot encoder model using the input attributes.

    Parameters
    ----------
    X: list / numpy.array
        Data to transform.
    categories: list / numpy.array
        List of the categories of the different input columns.
    drop_first: bool, optional
        If set to False, the first dummy of each category will be dropped.

    Returns
    -------
    list
        SQL code
    """

    def ooe_row(X):
        result = []
        for idx, elem in enumerate(X):
            for idx2, item in enumerate(categories[idx]):
                if idx2 != 0 or not (drop_first):
                    if str(elem) == str(item):
                        result += [1]
                    else:
                        result += [0]
        return result

    return np.apply_along_axis(ooe_row, 1, X)


def sql_from_one_hot_encoder(
    X: Union[list, np.ndarray],
    categories: Union[list, np.ndarray],
    drop_first: bool = False,
    column_naming: Literal["indices", "values", "values_relaxed", None] = None,
) -> list:
    """
    Returns the SQL code needed to deploy a one-hot encoder model using its 
    attributes.

    Parameters
    ----------
    X: list / numpy.array
        The names or values of the input predictors.
    categories: list / numpy.array
        List of the categories of the different input columns.
    drop_first: bool, optional
        If set to False, the first dummy of each category will be dropped.
    column_naming: str, optional
        Appends categorical levels to column names according to the specified method:
            indices    : Uses integer indices to represent categorical 
                                     levels.
            values/values_relaxed  : Both methods use categorical-level names. If 
                                     duplicate column names occur, the function 
                                     attempts to disambiguate them by appending _n, 
                                     where n is a zero-based integer index (_0, _1,…).

    Returns
    -------
    list
        SQL code
    """
    assert len(X) == len(categories), ParameterError(
        "The length of parameter 'X' must be equal to the length of the list 'values'."
    )
    sql = []
    for i in range(len(X)):
        sql_tmp = []
        for j in range(len(categories[i])):
            if not (drop_first) or j > 0:
                val = categories[i][j]
                if isinstance(val, str):
                    val = f"'{val}'"
                elif val == None:
                    val = "NULL"
                sql_tmp_feature = f"(CASE WHEN {X[i]} = {val} THEN 1 ELSE 0 END)"
                X_i = str(X[i]).replace('"', "")
                if column_naming == "indices":
                    sql_tmp_feature += f' AS "{X_i}_{j}"'
                elif column_naming in ("values", "values_relaxed"):
                    if categories[i][j] != None:
                        categories_i_j = categories[i][j]
                    else:
                        categories_i_j = "NULL"
                    sql_tmp_feature += f' AS "{X_i}_{categories_i_j}"'
                sql_tmp += [sql_tmp_feature]
        sql += [sql_tmp]
    return sql
