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
from typing import Union, Literal
import numpy as np

from verticapy.errors import ParameterError


def predict_from_coef(
    X: Union[list, np.ndarray],
    coefficients: Union[list, np.ndarray],
    intercept: float,
    method: Literal[
        "LinearRegression", "LinearSVR", "LogisticRegression", "LinearSVC"
    ] = "LinearRegression",
    return_proba: bool = False,
) -> np.ndarray:
    """
    Predicts using a linear regression model and the input attributes.

    Parameters
    ----------
    X: list / numpy.array
        Data on which to make the prediction.
    coefficients: list / numpy.array
        List of the model's coefficients.
    intercept: float
        The intercept or constant value.
    method: str, optional
        The model category, one of the following: 'LinearRegression', 'LinearSVR', 
        'LogisticRegression', or 'LinearSVC'.
    return_proba: bool, optional
        If set to True and the method is set to 'LogisticRegression' or 'LinearSVC', 
        the probability is returned.

    Returns
    -------
    numpy.array
        Predicted values
    """
    result = intercept + np.sum(np.array(coefficients) * np.array(X), axis=1)
    if method in ("LogisticRegression", "LinearSVC"):
        result = 1 / (1 + np.exp(-(result)))
    else:
        return result
    if return_proba:
        return np.column_stack((1 - result, result))
    else:
        return np.where(result > 0.5, 1, 0)


def sql_from_coef(
    X: Union[list, np.ndarray],
    coefficients: Union[list, np.ndarray],
    intercept: float,
    method: Literal[
        "LinearRegression", "LinearSVR", "LogisticRegression", "LinearSVC"
    ] = "LinearRegression",
) -> str:
    """
    Returns the SQL code needed to deploy a linear model using its attributes.

    Parameters
    ----------
    X: list / numpy.array
        The name or values of the input predictors.
    coefficients: list / numpy.array
        List of the model's coefficients.
    intercept: float
        The intercept or constant value.
    method: str, optional
        The model category, one of the following: 'LinearRegression', 'LinearSVR', 
        'LogisticRegression', or 'LinearSVC'.

    Returns
    -------
    str
        SQL code
    """
    assert len(X) == len(coefficients), ParameterError(
        "The length of parameter 'X' must be equal to the number of coefficients."
    )
    sql = [str(intercept)] + [
        f"{coefficients[idx]} * {(X[idx])}" for idx in range(len(coefficients))
    ]
    sql = " + ".join(sql)
    if method in ("LogisticRegression", "LinearSVC"):
        return f"1 / (1 + EXP(- ({sql})))"
    return sql
