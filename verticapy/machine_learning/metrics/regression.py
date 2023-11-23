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
from typing import Union

import numpy as np
from scipy.stats import f

from verticapy._typing import NoneType, PythonNumber, SQLRelation
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._sys import _executeSQL

from verticapy.core.tablesample.base import TableSample

"""
SQL Metrics.
"""

FUNCTIONS_REGRESSION_SQL_DICTIONNARY = {
    "explained_variance": "1 - VARIANCE({y_true} - {y_score}) / VARIANCE({y_true})",
    "max_error": "MAX(ABS({y_true} - {y_score}))::float",
    "median": "APPROXIMATE_MEDIAN(ABS({y_true} - {y_score}))",
    "median_absolute_error": "APPROXIMATE_MEDIAN(ABS({y_true} - {y_score}))",
    "mae": "AVG(ABS({y_true} - {y_score}))",
    "mean_absolute_error": "AVG(ABS({y_true} - {y_score}))",
    "mse": "AVG(POW({y_true} - {y_score}, 2))",
    "mean_squared_error": "AVG(POW({y_true} - {y_score}, 2))",
    "rmse": "SQRT(AVG(POW({y_true} - {y_score}, 2)))",
    "root_mean_squared_error": "SQRT(AVG(POW({y_true} - {y_score}, 2)))",
    "mean_squared_log_error": "AVG(POW(LOG({y_true} + 1) - LOG({y_score} + 1), 2))",
    "msle": "AVG(POW(LOG({y_true} + 1) - LOG({y_score} + 1), 2))",
    "quantile_error": "APPROXIMATE_PERCENTILE(ABS({y_true} - {y_score}) USING PARAMETERS percentile = {q})",
    "qe": "APPROXIMATE_PERCENTILE(ABS({y_true} - {y_score}) USING PARAMETERS percentile = {q})",
    "r2": "1 - (SUM(POWER({y_true} - {y_score}, 2))) / (SUM(POWER({y_true} - _verticapy_avg_y_true, 2)))",
    "rsquared": "1 - (SUM(POWER({y_true} - {y_score}, 2))) / (SUM(POWER({y_true} - _verticapy_avg_y_true, 2)))",
    "r2_adj": "1 - ((1 - (1 - (SUM(POWER({y_true} - {y_score}, 2))) / (SUM(POWER({y_true} - _verticapy_avg_y_true, 2))))) * (MAX(_verticapy_cnt_y_true) - 1) / (MAX(_verticapy_cnt_y_true) - {k} - 1))",
    "rsquared_adj": "1 - ((1 - (1 - (SUM(POWER({y_true} - {y_score}, 2))) / (SUM(POWER({y_true} - _verticapy_avg_y_true, 2))))) * (MAX(_verticapy_cnt_y_true) - 1) / (MAX(_verticapy_cnt_y_true) - {k} - 1))",
    "r2adj": "1 - ((1 - (1 - (SUM(POWER({y_true} - {y_score}, 2))) / (SUM(POWER({y_true} - _verticapy_avg_y_true, 2))))) * (MAX(_verticapy_cnt_y_true) - 1) / (MAX(_verticapy_cnt_y_true) - {k} - 1))",
    "r2adjusted": "1 - ((1 - (1 - (SUM(POWER({y_true} - {y_score}, 2))) / (SUM(POWER({y_true} - _verticapy_avg_y_true, 2))))) * (MAX(_verticapy_cnt_y_true) - 1) / (MAX(_verticapy_cnt_y_true) - {k} - 1))",
    "aic": "MAX(_verticapy_cnt_y_true) * LN(MAX(_verticapy_mse)) + 2 * ({k} + 1) + (POWER(2 * ({k} + 1), 2) + 2 * ({k} + 1)) / (MAX(_verticapy_cnt_y_true) - {k} - 2)",
    "bic": "MAX(_verticapy_cnt_y_true) * LN(MAX(_verticapy_mse)) + ({k} + 1) * LN(MAX(_verticapy_cnt_y_true))",
}


"""
General Metrics.
"""


def aic_score(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    k: int = 1,
) -> float:
    """
    Returns the AIC score.

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

    Examples
    ---------

    We should first import verticapy.

    .. ipython:: python

        import verticapy as vp

    Let's create a small dataset that has:

    - true value
    - predicted value

    .. ipython:: python

        data = vp.vDataFrame(
            {
                "y_true": [1, 1.5, 3, 2, 5],
                "y_pred": [1.1, 1.55, 2.9, 2.01, 4.5],
            }
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import aic_score

    Now we can conveniently calculate the score:

    .. ipython:: python

        aic_score(
            y_true = "y_true",
            y_score = "y_pred",
            input_relation = data,
        )

    It is also possible to directly compute the score
    from the vDataFrame:

    .. ipython:: python

        data.score(
            y_true  = "y_true",
            y_score = "y_pred",
            metric  = "aic",
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """
    return regression_report(y_true, y_score, input_relation, metrics="aic", k=k)


def bic_score(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    k: int = 1,
) -> float:
    """
    Returns the BIC score.

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

    Examples
    ---------

    We should first import verticapy.

    .. ipython:: python

        import verticapy as vp

    Let's create a small dataset that has:

    - true value
    - predicted value

    .. ipython:: python

        data = vp.vDataFrame(
            {
                "y_true": [1, 1.5, 3, 2, 5],
                "y_pred": [1.1, 1.55, 2.9, 2.01, 4.5],
            }
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import bic_score

    Now we can conveniently calculate the score:

    .. ipython:: python

        bic_score(
            y_true = "y_true",
            y_score = "y_pred",
            input_relation = data,
        )

    It is also possible to directly compute the score
    from the vDataFrame:

    .. ipython:: python

        data.score(
            y_true  = "y_true",
            y_score = "y_pred",
            metric  = "bic",
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """
    return regression_report(y_true, y_score, input_relation, metrics="bic", k=k)


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

    Examples
    ---------

    We should first import verticapy.

    .. ipython:: python

        import verticapy as vp

    Let's create a small dataset that has:

    - true value
    - predicted value

    .. ipython:: python

        data = vp.vDataFrame(
            {
                "y_true": [1, 1.5, 3, 2, 5],
                "y_pred": [1.1, 1.55, 2.9, 2.01, 4.5],
            }
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import explained_variance

    Now we can conveniently calculate the score:

    .. ipython:: python

        explained_variance(
            y_true = "y_true",
            y_score = "y_pred",
            input_relation = data,
        )

    It is also possible to directly compute the score
    from the vDataFrame:

    .. ipython:: python

        data.score(
            y_true  = "y_true",
            y_score = "y_pred",
            metric  = "explained_variance",
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """
    return regression_report(
        y_true, y_score, input_relation, metrics="explained_variance"
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

    Examples
    ---------

    We should first import verticapy.

    .. ipython:: python

        import verticapy as vp

    Let's create a small dataset that has:

    - true value
    - predicted value

    .. ipython:: python

        data = vp.vDataFrame(
            {
                "y_true": [1, 1.5, 3, 2, 5],
                "y_pred": [1.1, 1.55, 2.9, 2.01, 4.5],
            }
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import max_error

    Now we can conveniently calculate the score:

    .. ipython:: python

        max_error(
            y_true = "y_true",
            y_score = "y_pred",
            input_relation = data,
        )

    It is also possible to directly compute the score
    from the vDataFrame:

    .. ipython:: python

        data.score(
            y_true  = "y_true",
            y_score = "y_pred",
            metric  = "max_error",
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """
    return regression_report(y_true, y_score, input_relation, metrics="max_error")


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

    Examples
    ---------

    We should first import verticapy.

    .. ipython:: python

        import verticapy as vp

    Let's create a small dataset that has:

    - true value
    - predicted value

    .. ipython:: python

        data = vp.vDataFrame(
            {
                "y_true": [1, 1.5, 3, 2, 5],
                "y_pred": [1.1, 1.55, 2.9, 2.01, 4.5],
            }
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import mean_absolute_error

    Now we can conveniently calculate the score:

    .. ipython:: python

        mean_absolute_error(
            y_true = "y_true",
            y_score = "y_pred",
            input_relation = data,
        )

    It is also possible to directly compute the score
    from the vDataFrame:

    .. ipython:: python

        data.score(
            y_true  = "y_true",
            y_score = "y_pred",
            metric  = "mean_absolute_error",
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """
    return regression_report(y_true, y_score, input_relation, metrics="mae")


@save_verticapy_logs
def mean_squared_error(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    root: bool = False,
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

    Examples
    ---------

    We should first import verticapy.

    .. ipython:: python

        import verticapy as vp

    Let's create a small dataset that has:

    - true value
    - predicted value

    .. ipython:: python

        data = vp.vDataFrame(
            {
                "y_true": [1, 1.5, 3, 2, 5],
                "y_pred": [1.1, 1.55, 2.9, 2.01, 4.5],
            }
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import mean_squared_error

    Now we can conveniently calculate the score:

    .. ipython:: python

        mean_squared_error(
            y_true = "y_true",
            y_score = "y_pred",
            input_relation = data,
        )

    It is also possible to directly compute the score
    from the vDataFrame:

    .. ipython:: python

        data.score(
            y_true  = "y_true",
            y_score = "y_pred",
            metric  = "mean_squared_error",
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """
    return regression_report(
        y_true, y_score, input_relation, metrics="rmse" if root else "mse"
    )


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

    Examples
    ---------

    We should first import verticapy.

    .. ipython:: python

        import verticapy as vp

    Let's create a small dataset that has:

    - true value
    - predicted value

    .. ipython:: python

        data = vp.vDataFrame(
            {
                "y_true": [1, 1.5, 3, 2, 5],
                "y_pred": [1.1, 1.55, 2.9, 2.01, 4.5],
            }
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import mean_squared_log_error

    Now we can conveniently calculate the score:

    .. ipython:: python

        mean_squared_log_error(
            y_true = "y_true",
            y_score = "y_pred",
            input_relation = data,
        )

    It is also possible to directly compute the score
    from the vDataFrame:

    .. ipython:: python

        data.score(
            y_true  = "y_true",
            y_score = "y_pred",
            metric  = "mean_squared_log_error",
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """
    return regression_report(y_true, y_score, input_relation, metrics="msle")


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

    Examples
    ---------

    We should first import verticapy.

    .. ipython:: python

        import verticapy as vp

    Let's create a small dataset that has:

    - true value
    - predicted value

    .. ipython:: python

        data = vp.vDataFrame(
            {
                "y_true": [1, 1.5, 3, 2, 5],
                "y_pred": [1.1, 1.55, 2.9, 2.01, 4.5],
            }
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import median_absolute_error

    Now we can conveniently calculate the score:

    .. ipython:: python

        median_absolute_error(
            y_true = "y_true",
            y_score = "y_pred",
            input_relation = data,
        )

    It is also possible to directly compute the score
    from the vDataFrame:

    .. ipython:: python

        data.score(
            y_true  = "y_true",
            y_score = "y_pred",
            metric  = "median_absolute_error",
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """
    return regression_report(
        y_true, y_score, input_relation, metrics="median_absolute_error"
    )


@save_verticapy_logs
def quantile_error(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    q: PythonNumber,
) -> float:
    """
    Computes the input quantile of the Error.

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
    q: PythonNumber
        Input quantile.

    Returns
    -------
    float
        score.

    Examples
    ---------

    We should first import verticapy.

    .. ipython:: python

        import verticapy as vp

    Let's create a small dataset that has:

    - true value
    - predicted value

    .. ipython:: python

        data = vp.vDataFrame(
            {
                "y_true": [1, 1.5, 3, 2, 5],
                "y_pred": [1.1, 1.55, 2.9, 2.01, 4.5],
            }
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import quantile_error

    Now we can conveniently calculate the score:

    .. ipython:: python

        quantile_error(
            y_true  = "y_true",
            y_score = "y_pred",
            input_relation = data,
            q = 0.25, # First Quartile
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """
    return regression_report(y_true, y_score, input_relation, metrics=f"qe{100 * q}%")


@save_verticapy_logs
def r2_score(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    k: int = 0,
    adj: bool = True,
) -> float:
    """
    Computes the R2 score.

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
        Number  of predictors. Only used to compute the
        adjusted R2.
    adj: bool, optional
        If set to True, computes the adjusted R2.

    Returns
    -------
    float
        score.

    Examples
    ---------

    We should first import verticapy.

    .. ipython:: python

        import verticapy as vp

    Let's create a small dataset that has:

    - true value
    - predicted value

    .. ipython:: python

        data = vp.vDataFrame(
            {
                "y_true": [1, 1.5, 3, 2, 5],
                "y_pred": [1.1, 1.55, 2.9, 2.01, 4.5],
            }
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import r2_score

    Now we can conveniently calculate the score:

    .. ipython:: python

        r2_score(
            y_true = "y_true",
            y_score = "y_pred",
            input_relation = data,
        )

    It is also possible to directly compute the score
    from the vDataFrame:

    .. ipython:: python

        data.score(
            y_true  = "y_true",
            y_score = "y_pred",
            metric  = "r2",
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """
    if adj:
        kwargs = {"metrics": "r2_adj", "k": k}
    else:
        kwargs = {"metrics": "r2"}
    return regression_report(y_true, y_score, input_relation, **kwargs)


"""
Reports.
"""


@save_verticapy_logs
def anova_table(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    k: int = 1,
) -> TableSample:
    """
    Computes the ANOVA table.

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
        ANOVA table.

    Examples
    ---------

    We should first import verticapy.

    .. ipython:: python

        import verticapy as vp

    Let's create a small dataset that has:

    - true value
    - predicted value

    .. ipython:: python

        data = vp.vDataFrame(
            {
                "y_true": [1, 1.5, 3, 2, 5],
                "y_pred": [1.1, 1.55, 2.9, 2.01, 4.5],
            }
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import anova_table

    Now we can conveniently compute the ANOVA table:

    .. ipython:: python

        anova_table(
            y_true  = "y_true",
            y_score = "y_pred",
            input_relation = data,
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """
    n, avg = _executeSQL(
        query=f"""
        SELECT /*+LABEL('learn.metrics.anova_table')*/
            COUNT(*), 
            AVG({y_true}) 
        FROM {input_relation} 
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
            FROM {input_relation} 
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
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    metrics: Union[None, str, list[str]] = None,
    k: int = 1,
) -> Union[float, TableSample]:
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
    metrics: list, optional
        List of the metrics used to compute the final
        report.

        - aic:
            Akaike's Information Criterion

        - bic:
            Bayesian Information Criterion

        - max:
            Max Error

        - mae:
            Mean Absolute Error

        - median:
            Median Absolute Error

        - mse:
            Mean Squared Error

        - msle:
            Mean Squared Log Error

        - r2:
            R squared coefficient

        - r2a:
            R2 adjusted

        - qe:
            quantile error, the quantile must be
            included in the name. Example:
            qe50.1% will  return the quantile
            error using q=0.501.

        - rmse   : Root Mean Squared Error
        - var    : Explained Variance

    k: int, optional
        Number  of predictors. Used  to compute the adjusted
        R2.


    Returns
    -------
    TableSample
        report.

    Examples
    ---------

    We should first import verticapy.

    .. ipython:: python

        import verticapy as vp

    Let's create a small dataset that has:

    - true value
    - predicted value

    .. ipython:: python

        data = vp.vDataFrame(
            {
                "y_true": [1, 1.5, 3, 2, 5],
                "y_pred": [1.1, 1.55, 2.9, 2.01, 4.5],
            }
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import regression_report

    Now we can conveniently compute the report:

    .. ipython:: python

        regression_report(
            y_true  = "y_true",
            y_score = "y_pred",
            input_relation = data,
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """
    return_scalar = False
    if isinstance(metrics, str):
        metrics = [metrics]
        return_scalar = True
    if isinstance(metrics, NoneType):
        selected_metrics = [
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
    else:
        selected_metrics = copy.deepcopy(metrics)
    q_subquery = []
    cnt_in, mse_in, avg_in = False, False, False
    for m in selected_metrics:
        if m in ("r2_adj", "aic", "bic") and not cnt_in:
            q_subquery += [f"COUNT({y_true}) OVER() AS _verticapy_cnt_y_true"]
            cnt_in = True
        if m in ("aic", "bic") and not mse_in:
            mse = mean_squared_error(y_true, y_score, input_relation)
            q_subquery += [f"{mse} AS _verticapy_mse"]
            mse_in = True
        if m in ("r2", "r2_adj") and not avg_in:
            q_subquery += [f"AVG({y_true}) OVER() AS _verticapy_avg_y_true"]
            avg_in = True
    if len(q_subquery) > 0:
        relation = f"""
            (SELECT
                *,
                {', '.join(q_subquery)}
            FROM {input_relation}) VERTICAPY_SUBTABLE"""
    else:
        relation = input_relation
    metrics_sql = []
    for m in selected_metrics:
        if m.startswith("q"):
            if m.startswith("qe"):
                q = float(m[2:-1]) / 100
            else:
                q = float(m[14:-1]) / 100
            metrics_sql += [
                FUNCTIONS_REGRESSION_SQL_DICTIONNARY["qe"].format(
                    y_true=y_true, y_score=y_score, q=q
                )
            ]
        elif m in ("mse", "mean_squared_error") and mse_in:
            metrics_sql += [str(mse)]
        else:
            metrics_sql += [
                FUNCTIONS_REGRESSION_SQL_DICTIONNARY[m].format(
                    y_true=y_true, y_score=y_score, k=k
                )
            ]
    query = f"""
        SELECT {', '.join(metrics_sql)} 
        FROM {relation} 
        WHERE {y_true} IS NOT NULL 
          AND {y_score} IS NOT NULL;"""
    result = _executeSQL(
        query, title="Computing the Regression Report.", method="fetchrow"
    )
    result = list(np.array(result).astype(float))
    if return_scalar:
        return result[0]
    else:
        return TableSample({"index": selected_metrics, "value": result})
