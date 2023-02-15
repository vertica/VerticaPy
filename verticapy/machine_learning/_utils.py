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
from typing import Union
from verticapy.core.vdataframe.vdataframe import vDataFrame
from verticapy.sql._utils._format import quote_ident
from verticapy._utils._sql import _executeSQL


def compute_area(X: list, Y: list):
    auc = 0
    for i in range(len(Y) - 1):
        if Y[i + 1] - Y[i] != 0.0:
            a = (X[i + 1] - X[i]) / (Y[i + 1] - Y[i])
            b = X[i + 1] - a * Y[i + 1]
            auc = (
                auc
                + a * (Y[i + 1] * Y[i + 1] - Y[i] * Y[i]) / 2
                + b * (Y[i + 1] - Y[i])
            )
    return min(-auc, 1.0)


def reverse_score(metric: str):
    if metric in [
        "logloss",
        "max",
        "mae",
        "median",
        "mse",
        "msle",
        "rmse",
        "aic",
        "bic",
        "auto",
    ]:
        return False
    return True


def get_match_index(x: str, col_list: list, str_check: bool = True):
    for idx, col in enumerate(col_list):
        if (str_check and quote_ident(x.lower()) == quote_ident(col.lower())) or (
            x == col
        ):
            return idx
    return None
