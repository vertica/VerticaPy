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
from verticapy._typing import SQLRelation
from verticapy._utils._sql._sys import _executeSQL


def _compute_metric_query(
    metric: str,
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
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
input_relation: SQLRelation
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
        relation = input_relation._genSQL()
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
