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
import math
from typing import Union
import numpy as np
from scipy.stats import f

from verticapy._typing import PythonNumber, SQLRelation
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._sys import _executeSQL

from verticapy.core.tablesample.base import TableSample

"""
Special Functions.
"""


def _compute_metric_query(
    metric: str,
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    title: str = "",
    fetchfirstelem: bool = True,
) -> Union[float, tuple]:
    """
    A  helper  function  that uses  a  specified  metric  to 
    generate and score a query.

    Parameters
    ----------
    metric: str
        The metric to use in the query.
    y_true: str
        Response column.
    y_score: str
        Prediction.
    input_relation: SQLRelation
        Relation  to use for scoring.  This relation can  be 
        a view, table, or a customized relation (if an alias 
        is used at the end of the relation). 
        For example: (SELECT ... FROM ...) x
    title: str, optional
        Title of the query.
    fetchfirstelem: bool, optional
        If  set to True,  this function returns one  element. 
        Otherwise, this function returns a tuple.

    Returns
    -------
    float or tuple of floats.
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


"""
General Metrics.
"""


@save_verticapy_logs
def aic_bic(
    y_true: str, y_score: str, input_relation: SQLRelation, k: int = 1,
) -> tuple[float, float]:
    """
    Computes the AIC (Akaike’s Information Criterion) & BIC 
    (Bayesian Information Criterion).

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction.
    input_relation: SQLRelation
        Relation to use for scoring. This relation can be a 
        view, table,  or a customized relation (if an alias 
        is used at the end of the relation). 
        For example: (SELECT ... FROM ...) x
    k: int, optional
        Number of predictors.

    Returns
    -------
    tuple of floats
        (AIC, BIC)
    """
    rss, n = _compute_metric_query(
        "SUM(POWER({0} - {1}, 2)), COUNT(*)",
        y_true,
        y_score,
        input_relation,
        "Computing the RSS Score.",
        False,
    )
    if rss > 0:
        result = (
            n * math.log(rss / n)
            + 2 * (k + 1)
            + (2 * (k + 1) ** 2 + 2 * (k + 1)) / (n - k - 2),
            n * math.log(rss / n) + (k + 1) * math.log(n),
        )
    else:
        result = -float("inf"), -float("inf")
    return result


def aic_score(
    y_true: str, y_score: str, input_relation: SQLRelation, k: int = 1,
) -> float:
    """
    Returns the AIC score.
    """
    return aic_bic(y_true=y_true, y_score=y_score, input_relation=input_relation, k=k)[
        0
    ]


def bic_score(
    y_true: str, y_score: str, input_relation: SQLRelation, k: int = 1,
) -> float:
    """
    Returns the BIC score.
    """
    return aic_bic(y_true=y_true, y_score=y_score, input_relation=input_relation, k=k)[
        1
    ]


@save_verticapy_logs
def explained_variance(y_true: str, y_score: str, input_relation: SQLRelation) -> float:
    """
    Computes the Explained Variance.

    Parameters
    ----------
    y_true: str
    	Response column.
    y_score: str
    	Prediction.
    input_relation: SQLRelation
    	Relation to use for scoring. This relation can be a 
        view, table,  or a customized relation (if an alias 
        is used at the end of the relation). 
        For example: (SELECT ... FROM ...) x

    Returns
    -------
    float
    	score.
	"""
    return _compute_metric_query(
        "1 - VARIANCE({1} - {0}) / VARIANCE({0})",
        y_true,
        y_score,
        input_relation,
        "Computing the Explained Variance.",
    )


@save_verticapy_logs
def max_error(y_true: str, y_score: str, input_relation: SQLRelation) -> float:
    """
    Computes the Max Error.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction.
    input_relation: SQLRelation
        Relation to use for scoring. This relation can be a 
        view, table,  or a customized relation (if an alias 
        is used at the end of the relation). 
        For example: (SELECT ... FROM ...) x

    Returns
    -------
    float
        score.
    """
    return _compute_metric_query(
        "MAX(ABS({0} - {1}))::FLOAT",
        y_true,
        y_score,
        input_relation,
        "Computing the Max Error.",
    )


@save_verticapy_logs
def mean_absolute_error(
    y_true: str, y_score: str, input_relation: SQLRelation
) -> float:
    """
    Computes the Mean Absolute Error.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction.
    input_relation: SQLRelation
        Relation to use for scoring. This relation can be a 
        view, table,  or a customized relation (if an alias 
        is used at the end of the relation). 
        For example: (SELECT ... FROM ...) x

    Returns
    -------
    float
        score.
	"""
    return _compute_metric_query(
        "AVG(ABS({0} - {1}))",
        y_true,
        y_score,
        input_relation,
        "Computing the Mean Absolute Error.",
    )


@save_verticapy_logs
def mean_squared_error(
    y_true: str, y_score: str, input_relation: SQLRelation, root: bool = False,
) -> float:
    """
    Computes the Mean Squared Error.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction.
    input_relation: SQLRelation
        Relation to use for scoring. This relation can be a 
        view, table,  or a customized relation (if an alias 
        is used at the end of the relation). 
        For example: (SELECT ... FROM ...) x

    Returns
    -------
    float
        score.
    """
    res = _compute_metric_query(
        "MSE({0}, {1}) OVER ()", y_true, y_score, input_relation, "Computing the MSE.",
    )
    if root:
        return math.sqrt(res)
    return res


@save_verticapy_logs
def mean_squared_log_error(
    y_true: str, y_score: str, input_relation: SQLRelation
) -> float:
    """
    Computes the Mean Squared Log Error.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction.
    input_relation: SQLRelation
        Relation to use for scoring. This relation can be a 
        view, table,  or a customized relation (if an alias 
        is used at the end of the relation). 
        For example: (SELECT ... FROM ...) x

    Returns
    -------
    float
        score.
    """
    return _compute_metric_query(
        "AVG(POW(LOG({0} + 1) - LOG({1} + 1), 2))",
        y_true,
        y_score,
        input_relation,
        "Computing the Mean Squared Log Error.",
    )


@save_verticapy_logs
def median_absolute_error(
    y_true: str, y_score: str, input_relation: SQLRelation
) -> float:
    """
    Computes the Median Absolute Error.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction.
    input_relation: SQLRelation
        Relation to use for scoring. This relation can be a 
        view, table,  or a customized relation (if an alias 
        is used at the end of the relation). 
        For example: (SELECT ... FROM ...) x

    Returns
    -------
    float
        score.
    """
    return _compute_metric_query(
        "APPROXIMATE_MEDIAN(ABS({0} - {1}))",
        y_true,
        y_score,
        input_relation,
        "Computing the Median Absolute Error.",
    )


@save_verticapy_logs
def quantile_error(
    q: PythonNumber, y_true: str, y_score: str, input_relation: SQLRelation,
) -> float:
    """
    Computes the input Quantile of the Error.

    Parameters
    ----------
    q: PythonNumber
        Input Quantile.
    y_true: str
        Response column.
    y_score: str
        Prediction.
    input_relation: SQLRelation
        Relation to use for scoring. This relation can be a 
        view, table,  or a customized relation (if an alias 
        is used at the end of the relation). 
        For example: (SELECT ... FROM ...) x

    Returns
    -------
    float
        score.
    """
    metric = f"""APPROXIMATE_PERCENTILE(ABS({{0}} - {{1}}) 
                        USING PARAMETERS percentile = {q})"""
    return _compute_metric_query(
        metric, y_true, y_score, input_relation, "Computing the Quantile Error."
    )


@save_verticapy_logs
def r2_score(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    k: int = 0,
    adj: bool = True,
) -> float:
    """
    Computes the R2 Score.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction.
    input_relation: SQLRelation
        Relation to use for scoring. This relation can be a 
        view, table,  or a customized relation (if an alias 
        is used at the end of the relation). 
        For example: (SELECT ... FROM ...) x
    k: int, optional
        Number  of predictors. Only used to compute the  R2 
        adjusted.
    adj: bool, optional
        If set to True, computes the R2 adjusted.

    Returns
    -------
    float
    	score.
	"""
    result = _compute_metric_query(
        "RSQUARED({0}, {1}) OVER()",
        y_true,
        y_score,
        input_relation,
        "Computing the R2 Score.",
    )
    if adj and k > 0:
        n = _executeSQL(
            query=f"""
                SELECT /*+LABEL('learn.metrics.r2_score')*/ COUNT(*) 
                FROM {input_relation} 
                WHERE {y_true} IS NOT NULL 
                  AND {y_score} IS NOT NULL;""",
            title="Computing the table number of elements.",
            method="fetchfirstelem",
        )
        result = 1 - ((1 - result) * (n - 1) / (n - k - 1))
    return result


"""
Reports.
"""


@save_verticapy_logs
def anova_table(
    y_true: str, y_score: str, input_relation: SQLRelation, k: int = 1,
) -> TableSample:
    """
    Computes the Anova Table.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction.
    input_relation: SQLRelation
        Relation to use for scoring. This relation can be a 
        view, table,  or a customized relation (if an alias 
        is used at the end of the relation). 
        For example: (SELECT ... FROM ...) x
    k: int, optional
        Number of predictors.

    Returns
    -------
    TableSample
        anova table.
    """
    if isinstance(input_relation, str):
        relation = input_relation
    else:
        relation = input_relation._genSQL()
    n, avg = _executeSQL(
        query=f"""
        SELECT /*+LABEL('learn.metrics.anova_table')*/
            COUNT(*), 
            AVG({y_true}) 
        FROM {relation} 
        WHERE {y_true} IS NOT NULL 
          AND {y_score} IS NOT NULL;""",
        title="Computing n and the average of y.",
        method="fetchrow",
    )[0:2]
    SSR, SSE, SST = _executeSQL(
        query=f"""
            SELECT /*+LABEL('learn.metrics.anova_table')*/
                SUM(POWER({y_score} - {avg}, 2)), 
                SUM(POWER({y_true} - {y_score}, 2)), 
                SUM(POWER({y_true} - {avg}, 2)) 
            FROM {relation} 
            WHERE {y_score} IS NOT NULL 
              AND {y_true} IS NOT NULL;""",
        title="Computing SSR, SSE, SST.",
        method="fetchrow",
    )[0:3]
    dfr, dfe, dft = k, n - 1 - k, n - 1
    MSR, MSE = SSR / dfr, SSE / dfe
    if MSE == 0:
        F = float("inf")
    else:
        F = MSR / MSE
    pvalue = f.sf(F, k, n)
    return TableSample(
        {
            "index": ["Regression", "Residual", "Total"],
            "Df": [dfr, dfe, dft],
            "SS": [SSR, SSE, SST],
            "MS": [MSR, MSE, ""],
            "F": [F, "", ""],
            "p_value": [pvalue, "", ""],
        }
    )


@save_verticapy_logs
def regression_report(
    y_true: str, y_score: str, input_relation: SQLRelation, k: int = 1,
) -> TableSample:
    """
    Computes a regression report using multiple metrics (r2, 
    mse, max error...). 

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction.
    input_relation: SQLRelation
        Relation to use for scoring. This relation can be a 
        view, table,  or a customized relation (if an alias 
        is used at the end of the relation). 
        For example: (SELECT ... FROM ...) x
    k: int, optional
        Number  of predictors. Used  to compute the adjusted 
        R2.

    Returns
    -------
    TableSample
     	report.
	"""
    if isinstance(input_relation, str):
        relation = input_relation
    else:
        relation = input_relation._genSQL()
    query = f"""
        SELECT /*+LABEL('learn.metrics.regression_report')*/
            1 - VARIANCE({y_true} - {y_score}) / VARIANCE({y_true}), 
            MAX(ABS({y_true} - {y_score})),
            APPROXIMATE_MEDIAN(ABS({y_true} - {y_score})), 
            AVG(ABS({y_true} - {y_score})),
            AVG(POW({y_true} - {y_score}, 2)), 
            COUNT(*) 
        FROM {relation} 
        WHERE {y_true} IS NOT NULL 
          AND {y_score} IS NOT NULL;"""
    r2 = r2_score(y_true, y_score, input_relation)
    values = {
        "index": [
            "explained_variance",
            "max_error",
            "median_absolute_error",
            "mean_absolute_error",
            "mean_squared_error",
            "root_mean_squared_error",
            "r2",
            "r2_adj",
            "aic",
            "bic",
        ]
    }
    result = _executeSQL(
        query, title="Computing the Regression Report.", method="fetchrow"
    )
    n = result[5]
    if result[4] > 0:
        aic, bic = (
            n * math.log(result[4])
            + 2 * (k + 1)
            + (2 * (k + 1) ** 2 + 2 * (k + 1)) / (n - k - 2),
            n * math.log(result[4]) + (k + 1) * math.log(n),
        )
    else:
        aic, bic = -np.inf, -np.inf
    values["value"] = [
        result[0],
        result[1],
        result[2],
        result[3],
        result[4],
        math.sqrt(result[4]),
        r2,
        1 - ((1 - r2) * (n - 1) / (n - k - 1)),
        aic,
        bic,
    ]
    return TableSample(values)
