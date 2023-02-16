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
from verticapy.core.vdataframe.base import vDataFrame
from verticapy.sql._utils._format import quote_ident
from verticapy._utils._sql import _executeSQL


def _compute_tn_fn_fp_tp(
    y_true: str,
    y_score: str,
    input_relation: Union[str, vDataFrame],
    pos_label: Union[int, float, str] = 1,
):
    """
A helper function that computes the confusion matrix for the specified 
'pos_label' class and returns its values as a tuple of the following: 
true negatives, false negatives, false positives, and true positives.

Parameters
----------
y_true: str
    Response column.
y_score: str
    Prediction.
input_relation: str / vDataFrame
    Relation to use for scoring. This relation can be a view, table, or a 
    customized relation (if an alias is used at the end of the relation). 
    For example: (SELECT ... FROM ...) x
pos_label: int / float / str, optional
    To compute the Confusion Matrix, one of the response column classes must 
    be the positive one. The parameter 'pos_label' represents this class.

Returns
-------
tuple
    tn, fn, fp, tp
    """
    from verticapy.machine_learning.metrics.classification import confusion_matrix

    matrix = confusion_matrix(y_true, y_score, input_relation, pos_label)
    non_pos_label = 0 if (pos_label == 1) else f"Non-{pos_label}"
    tn, fn, fp, tp = (
        matrix.values[non_pos_label][0],
        matrix.values[non_pos_label][1],
        matrix.values[pos_label][0],
        matrix.values[pos_label][1],
    )
    return tn, fn, fp, tp


def _compute_metric_query(
    metric: str,
    y_true: str,
    y_score: str,
    input_relation: Union[str, vDataFrame],
    title: str = "",
    fetchfirstelem: bool = True,
):
    """
A helper function that uses a specified metric to generate and score a query.

Parameters
----------
metric: str
    The metric to use in the query.
y_true: str
    Response column.
y_score: str
    Prediction.
input_relation: str/vDataFrame
    Relation to use for scoring. This relation can be a view, table, or a 
    customized relation (if an alias is used at the end of the relation). 
    For example: (SELECT ... FROM ...) x
title: str, optional
    Relation to use to do the scoring. The relation can be a view or a table
    or even a customized relation. For example, you could write:
    "(SELECT ... FROM ...) x" as long as an alias is given at the end of the
Title of the query.
fetchfirstelem: bool, optional
    If set to True, this function returns one element. Otherwise, this 
    function returns a tuple.

Returns
-------
float or tuple of floats
    score(s)
    """
    if isinstance(input_relation, str):
        relation = input_relation
    else:
        relation = input_relation.__genSQL__()
    if fetchfirstelem:
        method = "fetchfirstelem"
    else:
        method = "fetchrow"
    return _executeSQL(
        query=f"""
            SELECT 
                /*+LABEL('learn.metrics._compute_metric_query')*/ 
                {metric.format(y_true, y_score)} 
            FROM {relation} 
            WHERE {y_true} IS NOT NULL 
              AND {y_score} IS NOT NULL;""",
        title=title,
        method=method,
    )


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
