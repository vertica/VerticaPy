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

#
#
# Modules
#
# Standard Python Modules
import math
from collections.abc import Iterable
from typing import Union

# Other Modules
import numpy as np
from scipy.stats import f

# VerticaPy Modules
from verticapy import *
from verticapy.utils._decorators import (
    save_verticapy_logs,
    check_minimum_version,
)
from verticapy.core.vdataframe import vDataFrame
from verticapy.learn.model_selection import *
from verticapy.sql.read import to_tablesample
from verticapy.core.tablesample import tablesample
from verticapy.sql.read import _executeSQL

#
# Function used to simplify the code
#


def compute_metric_query(
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
                /*+LABEL('learn.metrics.compute_metric_query')*/ 
                {metric.format(y_true, y_score)} 
            FROM {relation} 
            WHERE {y_true} IS NOT NULL 
              AND {y_score} IS NOT NULL;""",
        title=title,
        method=method,
    )


def compute_tn_fn_fp_tp(
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

    matrix = confusion_matrix(y_true, y_score, input_relation, pos_label)
    non_pos_label = 0 if (pos_label == 1) else f"Non-{pos_label}"
    tn, fn, fp, tp = (
        matrix.values[non_pos_label][0],
        matrix.values[non_pos_label][1],
        matrix.values[pos_label][0],
        matrix.values[pos_label][1],
    )
    return tn, fn, fp, tp


#
# Regression
#


@save_verticapy_logs
def aic_bic(
    y_true: str, y_score: str, input_relation: Union[str, vDataFrame], k: int = 1,
):
    """
Computes the AIC (Akaike’s Information Criterion) & BIC (Bayesian Information 
Criterion).

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
k: int, optional
    Number of predictors.

Returns
-------
tuple of floats
    (AIC, BIC)
    """
    rss, n = compute_metric_query(
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


@save_verticapy_logs
def anova_table(
    y_true: str, y_score: str, input_relation: Union[str, vDataFrame], k: int = 1,
):
    """
Computes the Anova Table.

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
k: int, optional
    Number of predictors.

Returns
-------
tablesample
    An object containing the result. For more information, see
    utilities.tablesample.
    """
    if isinstance(input_relation, str):
        relation = input_relation
    else:
        relation = input_relation.__genSQL__()
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
    return tablesample(
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
def explained_variance(
    y_true: str, y_score: str, input_relation: Union[str, vDataFrame]
):
    """
Computes the Explained Variance.

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

Returns
-------
float
	score
	"""
    return compute_metric_query(
        "1 - VARIANCE({1} - {0}) / VARIANCE({0})",
        y_true,
        y_score,
        input_relation,
        "Computing the Explained Variance.",
    )


@save_verticapy_logs
def max_error(y_true: str, y_score: str, input_relation: Union[str, vDataFrame]):
    """
Computes the Max Error.

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

Returns
-------
float
	score
	"""
    return compute_metric_query(
        "MAX(ABS({0} - {1}))::FLOAT",
        y_true,
        y_score,
        input_relation,
        "Computing the Max Error.",
    )


@save_verticapy_logs
def mean_absolute_error(
    y_true: str, y_score: str, input_relation: Union[str, vDataFrame]
):
    """
Computes the Mean Absolute Error.

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

Returns
-------
float
	score
	"""
    return compute_metric_query(
        "AVG(ABS({0} - {1}))",
        y_true,
        y_score,
        input_relation,
        "Computing the Mean Absolute Error.",
    )


@save_verticapy_logs
def mean_squared_error(
    y_true: str,
    y_score: str,
    input_relation: Union[str, vDataFrame],
    root: bool = False,
):
    """
Computes the Mean Squared Error.

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
root: bool, optional
    If set to True, returns the RMSE (Root Mean Squared Error)

Returns
-------
float
	score
	"""
    result = compute_metric_query(
        "MSE({0}, {1}) OVER ()", y_true, y_score, input_relation, "Computing the MSE.",
    )
    if root:
        return math.sqrt(result)
    return result


@save_verticapy_logs
def mean_squared_log_error(
    y_true: str, y_score: str, input_relation: Union[str, vDataFrame]
):
    """
Computes the Mean Squared Log Error.

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

Returns
-------
float
	score
	"""
    return compute_metric_query(
        "AVG(POW(LOG({0} + 1) - LOG({1} + 1), 2))",
        y_true,
        y_score,
        input_relation,
        "Computing the Mean Squared Log Error.",
    )


@save_verticapy_logs
def median_absolute_error(
    y_true: str, y_score: str, input_relation: Union[str, vDataFrame]
):
    """
Computes the Median Absolute Error.

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

Returns
-------
float
	score
	"""
    return compute_metric_query(
        "APPROXIMATE_MEDIAN(ABS({0} - {1}))",
        y_true,
        y_score,
        input_relation,
        "Computing the Median Absolute Error.",
    )


@save_verticapy_logs
def quantile_error(
    q: Union[int, float],
    y_true: str,
    y_score: str,
    input_relation: Union[str, vDataFrame],
):
    """
Computes the input Quantile of the Error.

Parameters
----------
q: int / float
    Input Quantile
y_true: str
    Response column.
y_score: str
    Prediction.
input_relation: str / vDataFrame
    Relation to use for scoring. This relation can be a view, table, or a 
    customized relation (if an alias is used at the end of the relation). 
    For example: (SELECT ... FROM ...) x
    
Returns
-------
float
    score
    """
    metric = f"""APPROXIMATE_PERCENTILE(ABS({{0}} - {{1}}) 
                                        USING PARAMETERS percentile = {q})"""
    return compute_metric_query(
        metric, y_true, y_score, input_relation, "Computing the Quantile Error."
    )


@save_verticapy_logs
def r2_score(
    y_true: str,
    y_score: str,
    input_relation: Union[str, vDataFrame],
    k: int = 0,
    adj: bool = True,
):
    """
Computes the R2 Score.

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
k: int, optional
    Number of predictors. Only used to compute the R2 adjusted.
adj: bool, optional
    If set to True, computes the R2 adjusted.

Returns
-------
float
	score
	"""
    result = compute_metric_query(
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


@save_verticapy_logs
def regression_report(
    y_true: str, y_score: str, input_relation: Union[str, vDataFrame], k: int = 1,
):
    """
Computes a regression report using multiple metrics (r2, mse, max error...). 

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
k: int, optional
    Number of predictors. Used to compute the adjusted R2.

Returns
-------
tablesample
 	An object containing the result. For more information, see
 	utilities.tablesample.
	"""
    if isinstance(input_relation, str):
        relation = input_relation
    else:
        relation = input_relation.__genSQL__()
    query = f"""SELECT /*+LABEL('learn.metrics.regression_report')*/
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
    return tablesample(values)


#
# Classification
#


@save_verticapy_logs
def accuracy_score(
    y_true: str,
    y_score: str,
    input_relation: Union[str, vDataFrame],
    pos_label: Union[str, int, float] = None,
):
    """
Computes the Accuracy Score.

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
	Label to use to identify the positive class. If pos_label is NULL then the
	global accuracy will be computed.

Returns
-------
float
	score
	"""
    if pos_label != None:
        tn, fn, fp, tp = compute_tn_fn_fp_tp(y_true, y_score, input_relation, pos_label)
        acc = (tp + tn) / (tp + tn + fn + fp)
        return acc
    else:
        try:
            return compute_metric_query(
                "AVG(CASE WHEN {0} = {1} THEN 1 ELSE 0 END)",
                y_true,
                y_score,
                input_relation,
                "Computing the Accuracy Score.",
            )
        except:
            return compute_metric_query(
                "AVG(CASE WHEN {0}::varchar = {1}::varchar THEN 1 ELSE 0 END)",
                y_true,
                y_score,
                input_relation,
                "Computing the Accuracy Score.",
            )


@save_verticapy_logs
def auc(
    y_true: str,
    y_score: str,
    input_relation: Union[str, vDataFrame],
    pos_label: Union[int, float, str] = 1,
    nbins: int = 10000,
):
    """
Computes the ROC AUC (Area Under Curve).

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction Probability.
input_relation: str/vDataFrame
	Relation to use for scoring. This relation can be a view, table, or a 
    customized relation (if an alias is used at the end of the relation). 
    For example: (SELECT ... FROM ...) x
pos_label: int/float/str, optional
	To compute the ROC AUC, one of the response column classes must be the 
	positive one. The parameter 'pos_label' represents this class.
nbins: int, optional
    An integer value that determines the number of decision boundaries. 
    Decision boundaries are set at equally spaced intervals between 0 and 1, 
    inclusive. Greater values for nbins give more precise estimations of the AUC, 
    but can potentially decrease performance. The maximum value is 999,999. 
    If negative, the maximum value is used.

Returns
-------
float
	score
	"""
    return roc_curve(
        y_true, y_score, input_relation, pos_label, nbins=nbins, auc_roc=True
    )


@save_verticapy_logs
def classification_report(
    y_true: str = "",
    y_score: list = [],
    input_relation: Union[str, vDataFrame] = "",
    labels: list = [],
    cutoff: Union[int, float, list] = [],
    estimator=None,
    nbins: int = 10000,
):
    """
Computes a classification report using multiple metrics (AUC, accuracy, PRC 
AUC, F1...). It will consider each category as positive and switch to the 
next one during the computation.

Parameters
----------
y_true: str, optional
	Response column.
y_score: list, optional
	List containing the probability and the prediction.
input_relation: str / vDataFrame, optional
	Relation to use for scoring. This relation can be a view, table, or a 
    customized relation (if an alias is used at the end of the relation). 
    For example: (SELECT ... FROM ...) x
labels: list, optional
	List of the response column categories to use.
cutoff: int / float / list, optional
	Cutoff for which the tested category will be accepted as prediction. 
	For multiclass classification, the list will represent the the classes 
    threshold. If it is empty, the best cutoff will be used.
estimator: object, optional
	Estimator to use to compute the classification report.
nbins: int, optional
    [Used to compute ROC AUC, PRC AUC and the best cutoff]
    An integer value that determines the number of decision boundaries. 
    Decision boundaries are set at equally spaced intervals between 0 and 1, 
    inclusive. Greater values for nbins give more precise estimations of the 
    metrics, but can potentially decrease performance. The maximum value 
    is 999,999. If negative, the maximum value is used.

Returns
-------
tablesample
 	An object containing the result. For more information, see
 	utilities.tablesample.
	"""
    if estimator:
        num_classes = len(estimator.classes_)
        labels = labels if (num_classes != 2) else [estimator.classes_[1]]
    else:
        labels = [1] if not (labels) else labels
        num_classes = len(labels) + 1
    values = {
        "index": [
            "auc",
            "prc_auc",
            "accuracy",
            "log_loss",
            "precision",
            "recall",
            "f1_score",
            "mcc",
            "informedness",
            "markedness",
            "csi",
            "cutoff",
        ]
    }
    for idx, l in enumerate(labels):
        pos_label = l
        non_pos_label = 0 if (l == 1) else f"Non-{l}"
        if estimator:
            if not (cutoff):
                current_cutoff = estimator.score(
                    method="best_cutoff", pos_label=pos_label, nbins=nbins
                )
            elif isinstance(cutoff, Iterable):
                if len(cutoff) == 1:
                    current_cutoff = cutoff[0]
                else:
                    current_cutoff = cutoff[idx]
            else:
                current_cutoff = cutoff
            try:
                matrix = estimator.confusion_matrix(pos_label, current_cutoff)
            except:
                matrix = estimator.confusion_matrix(pos_label)
        else:
            y_s, y_p, y_t = (
                y_score[0].format(l),
                y_score[1],
                f"DECODE({y_true}, '{l}', 1, 0)",
            )
            matrix = confusion_matrix(y_true, y_p, input_relation, pos_label)
        if non_pos_label in matrix.values and pos_label in matrix.values:
            non_pos_label_, pos_label_ = non_pos_label, pos_label
        elif 0 in matrix.values and 1 in matrix.values:
            non_pos_label_, pos_label_ = 0, 1
        else:
            non_pos_label_, pos_label_ = matrix.values["index"]
        tn, fn, fp, tp = (
            matrix.values[non_pos_label_][0],
            matrix.values[non_pos_label_][1],
            matrix.values[pos_label_][0],
            matrix.values[pos_label_][1],
        )
        ppv = tp / (tp + fp) if (tp + fp != 0) else 0  # precision
        tpr = tp / (tp + fn) if (tp + fn != 0) else 0  # recall
        tnr = tn / (tn + fp) if (tn + fp != 0) else 0  # specificity
        npv = tn / (tn + fn) if (tn + fn != 0) else 0  # negative predictive score
        f1 = 2 * (tpr * ppv) / (tpr + ppv) if (tpr + ppv != 0) else 0  # f1
        csi = tp / (tp + fn + fp) if (tp + fn + fp != 0) else 0  # csi
        bm = tpr + tnr - 1  # informedness
        mk = ppv + npv - 1  # markedness
        mcc = (
            (tp * tn - fp * fn)
            / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            if (tp + fp != 0) and (tp + fn != 0) and (tn + fp != 0) and (tn + fn != 0)
            else 0
        )  # matthews corr coef
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        if estimator:
            auc_score, logloss, prc_auc_score = (
                estimator.score(pos_label=pos_label, method="auc", nbins=nbins),
                estimator.score(pos_label=pos_label, method="log_loss"),
                estimator.score(pos_label=pos_label, method="prc_auc", nbins=nbins),
            )
        else:
            auc_score = auc(y_t, y_s, input_relation, 1)
            prc_auc_score = prc_auc(y_t, y_s, input_relation, 1)
            y_p = f"DECODE({y_p}, '{l}', 1, 0)"
            logloss = log_loss(y_t, y_s, input_relation, 1)
            if not (cutoff):
                current_cutoff = roc_curve(
                    y_t, y_s, input_relation, best_threshold=True, nbins=nbins,
                )
            elif isinstance(cutoff, Iterable):
                if len(cutoff) == 1:
                    current_cutoff = cutoff[0]
                else:
                    current_cutoff = cutoff[idx]
            else:
                current_cutoff = cutoff
        if len(labels) == 1:
            l = "value"
        values[l] = [
            auc_score,
            prc_auc_score,
            accuracy,
            logloss,
            ppv,
            tpr,
            f1,
            mcc,
            bm,
            mk,
            csi,
            current_cutoff,
        ]
    return tablesample(values)


@check_minimum_version
@save_verticapy_logs
def confusion_matrix(
    y_true: str,
    y_score: str,
    input_relation: Union[str, vDataFrame],
    pos_label: Union[str, int, float] = 1,
):
    """
Computes the Confusion Matrix.

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
pos_label: str / int / float, optional
	To compute the one dimension Confusion Matrix, one of the response column 
	class must be the positive one. The parameter 'pos_label' represents 
	this class.

Returns
-------
tablesample
 	An object containing the result. For more information, see
 	utilities.tablesample.
	"""
    if isinstance(input_relation, str):
        relation = input_relation
    else:
        relation = input_relation.__genSQL__()
    result = to_tablesample(
        query=f"""
        SELECT 
            CONFUSION_MATRIX(obs, response 
            USING PARAMETERS num_classes = 2) OVER() 
        FROM 
            (SELECT 
                DECODE({y_true}, '{pos_label}', 
                       1, NULL, NULL, 0) AS obs, 
                DECODE({y_score}, '{pos_label}', 
                       1, NULL, NULL, 0) AS response 
             FROM {relation}) VERTICAPY_SUBTABLE;""",
        title="Computing Confusion matrix.",
    )
    if pos_label in [1, "1"]:
        labels = [0, 1]
    else:
        labels = [f"Non-{pos_label}", pos_label]
    del result.values["comment"]
    result = result.transpose()
    result.values["actual_class"] = labels
    result = result.transpose()
    matrix = {"index": labels}
    for elem in result.values:
        if elem != "actual_class":
            matrix[elem] = result.values[elem]
    result.values = matrix
    return result


@save_verticapy_logs
def critical_success_index(
    y_true: str,
    y_score: str,
    input_relation: Union[str, vDataFrame],
    pos_label: Union[int, float, str] = 1,
):
    """
Computes the Critical Success Index.

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction.
input_relation: str/vDataFrame
	Relation to use for scoring. This relation can be a view, table, or a 
    customized relation (if an alias is used at the end of the relation). 
    For example: (SELECT ... FROM ...) x
pos_label: int/float/str, optional
	To compute the CSI, one of the response column classes must be the 
	positive one. The parameter 'pos_label' represents this class.

Returns
-------
float
	score
	"""
    tn, fn, fp, tp = compute_tn_fn_fp_tp(y_true, y_score, input_relation, pos_label)
    csi = tp / (tp + fn + fp) if (tp + fn + fp != 0) else 0
    return csi


@save_verticapy_logs
def f1_score(
    y_true: str,
    y_score: str,
    input_relation: Union[str, vDataFrame],
    pos_label: Union[int, float, str] = 1,
):
    """
Computes the F1 Score.

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction.
input_relation: str/vDataFrame
	Relation to use for scoring. This relation can be a view, table, or a 
    customized relation (if an alias is used at the end of the relation). 
    For example: (SELECT ... FROM ...) x
pos_label: int/float/str, optional
	To compute the F1 Score, one of the response column classes must be the 
	positive one. The parameter 'pos_label' represents this class.

Returns
-------
float
	score
	"""
    tn, fn, fp, tp = compute_tn_fn_fp_tp(y_true, y_score, input_relation, pos_label)
    recall = tp / (tp + fn) if (tp + fn != 0) else 0
    precision = tp / (tp + fp) if (tp + fp != 0) else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall != 0)
        else 0
    )
    return f1


@save_verticapy_logs
def informedness(
    y_true: str,
    y_score: str,
    input_relation: Union[str, vDataFrame],
    pos_label: Union[int, float, str] = 1,
):
    """
Computes the Informedness.

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction.
input_relation: str/vDataFrame
	Relation to use for scoring. This relation can be a view, table, or a 
    customized relation (if an alias is used at the end of the relation). 
    For example: (SELECT ... FROM ...) x
pos_label: int/float/str, optional
	To compute the informedness, one of the response column classes must be the 
	positive one. The parameter 'pos_label' represents this class.

Returns
-------
float
	score
	"""
    tn, fn, fp, tp = compute_tn_fn_fp_tp(y_true, y_score, input_relation, pos_label)
    tpr = tp / (tp + fn) if (tp + fn != 0) else 0
    tnr = tn / (tn + fp) if (tn + fp != 0) else 0
    return tpr + tnr - 1


@save_verticapy_logs
def log_loss(
    y_true: str,
    y_score: str,
    input_relation: Union[str, vDataFrame],
    pos_label: Union[int, float, str] = 1,
):
    """
Computes the Log Loss.

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction Probability.
input_relation: str/vDataFrame
	Relation to use for scoring. This relation can be a view, table, or a 
    customized relation (if an alias is used at the end of the relation). 
    For example: (SELECT ... FROM ...) x
pos_label: int/float/str, optional
	To compute the log loss, one of the response column classes must be the 
	positive one. The parameter 'pos_label' represents this class.

Returns
-------
float
	score
	"""
    metric = f"""
        AVG(CASE 
                WHEN {{0}} = '{pos_label}' 
                    THEN - LOG({{1}}::float + 1e-90) 
                ELSE - LOG(1 - {{1}}::float + 1e-90) 
            END)"""
    return compute_metric_query(
        metric, y_true, y_score, input_relation, "Computing the Log Loss."
    )


@save_verticapy_logs
def markedness(
    y_true: str,
    y_score: str,
    input_relation: Union[str, vDataFrame],
    pos_label: Union[int, float, str] = 1,
):
    """
Computes the Markedness.

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction.
input_relation: str/vDataFrame
	Relation to use for scoring. This relation can be a view, table, or a 
    customized relation (if an alias is used at the end of the relation). 
    For example: (SELECT ... FROM ...) x
pos_label: int/float/str, optional
	To compute the markedness, one of the response column classes must be the 
	positive one. The parameter 'pos_label' represents this class.

Returns
-------
float
	score
	"""
    tn, fn, fp, tp = compute_tn_fn_fp_tp(y_true, y_score, input_relation, pos_label)
    ppv = tp / (tp + fp) if (tp + fp != 0) else 0
    npv = tn / (tn + fn) if (tn + fn != 0) else 0
    return ppv + npv - 1


@save_verticapy_logs
def matthews_corrcoef(
    y_true: str,
    y_score: str,
    input_relation: Union[str, vDataFrame],
    pos_label: Union[int, float, str] = 1,
):
    """
Computes the Matthews Correlation Coefficient.

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction.
input_relation: str/vDataFrame
	Relation to use for scoring. This relation can be a view, table, or a 
    customized relation (if an alias is used at the end of the relation). 
    For example: (SELECT ... FROM ...) x
pos_label: int/float/str, optional
	To compute the Matthews Correlation Coefficient, one of the response column 
	class must be the positive one. The parameter 'pos_label' represents this 
	class.

Returns
-------
float
	score
	"""
    tn, fn, fp, tp = compute_tn_fn_fp_tp(y_true, y_score, input_relation, pos_label)
    mcc = (
        (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if (tp + fp != 0) and (tp + fn != 0) and (tn + fp != 0) and (tn + fn != 0)
        else 0
    )
    return mcc


@check_minimum_version
@save_verticapy_logs
def multilabel_confusion_matrix(
    y_true: str, y_score: str, input_relation: Union[str, vDataFrame], labels: list,
):
    """
Computes the Multi Label Confusion Matrix.

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
labels: list
	List of the response column categories.

Returns
-------
tablesample
 	An object containing the result. For more information, see
 	utilities.tablesample.
	"""
    if isinstance(input_relation, str):
        relation = input_relation
    else:
        relation = input_relation.__genSQL__()
    num_classes = str(len(labels))
    query = f"""
        SELECT 
          CONFUSION_MATRIX(obs, response 
          USING PARAMETERS num_classes = {num_classes}) OVER() 
       FROM (SELECT DECODE({y_true}"""
    for idx, l in enumerate(labels):
        query += f", '{l}', {idx}"
    query += f") AS obs, DECODE({y_score}"
    for idx, l in enumerate(labels):
        query += f", '{l}', {idx}"
    query += f") AS response FROM {relation}) VERTICAPY_SUBTABLE;"
    result = to_tablesample(query=query, title="Computing Confusion Matrix.")
    del result.values["comment"]
    result = result.transpose()
    result.values["actual_class"] = labels
    result = result.transpose()
    matrix = {"index": labels}
    for elem in result.values:
        if elem != "actual_class":
            matrix[elem] = result.values[elem]
    result.values = matrix
    return result


@save_verticapy_logs
def negative_predictive_score(
    y_true: str,
    y_score: str,
    input_relation: Union[str, vDataFrame],
    pos_label: Union[int, float, str] = 1,
):
    """
Computes the Negative Predictive Score.

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction.
input_relation: str/vDataFrame
	Relation to use for scoring. This relation can be a view, table, or a 
    customized relation (if an alias is used at the end of the relation). 
    For example: (SELECT ... FROM ...) x
pos_label: int/float/str, optional
	To compute the Negative Predictive Score, one of the response column class 
	must be the positive one. The parameter 'pos_label' represents this class.

Returns
-------
float
	score
	"""
    tn, fn, fp, tp = compute_tn_fn_fp_tp(y_true, y_score, input_relation, pos_label)
    npv = tn / (tn + fn) if (tn + fn != 0) else 0
    return npv


@save_verticapy_logs
def prc_auc(
    y_true: str,
    y_score: str,
    input_relation: Union[str, vDataFrame],
    pos_label: Union[int, float, str] = 1,
    nbins: int = 10000,
):
    """
Computes the area under the curve (AUC) of a Precision-Recall (PRC) curve.

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction probability.
input_relation: str/vDataFrame
	Relation to use for scoring. This relation can be a view, table, or a 
    customized relation (if an alias is used at the end of the relation). 
    For example: (SELECT ... FROM ...) x
pos_label: int/float/str, optional
	To compute the PRC AUC, one of the response column classes must be the 
	positive one. The parameter 'pos_label' represents this class.
nbins: int, optional
    An integer value that determines the number of decision boundaries. Decision 
    boundaries are set at equally-spaced intervals between 0 and 1, inclusive.
    The greater number of decision boundaries, the greater precision, but 
    the greater decrease in performance. Maximum value: 999,999. If negative, the
    maximum value is used.

Returns
-------
float
	score
	"""
    return prc_curve(
        y_true, y_score, input_relation, pos_label, nbins=nbins, auc_prc=True
    )


@save_verticapy_logs
def precision_score(
    y_true: str,
    y_score: str,
    input_relation: Union[str, vDataFrame],
    pos_label: Union[int, float, str] = 1,
):
    """
Computes the Precision Score.

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction.
input_relation: str/vDataFrame
	Relation to use for scoring. This relation can be a view, table, or a 
    customized relation (if an alias is used at the end of the relation). 
    For example: (SELECT ... FROM ...) x
pos_label: int/float/str, optional
	To compute the Precision Score, one of the response column classes must be 
	the positive one. The parameter 'pos_label' represents this class.

Returns
-------
float
	score
	"""
    tn, fn, fp, tp = compute_tn_fn_fp_tp(y_true, y_score, input_relation, pos_label)
    precision = tp / (tp + fp) if (tp + fp != 0) else 0
    return precision


@save_verticapy_logs
def recall_score(
    y_true: str,
    y_score: str,
    input_relation: Union[str, vDataFrame],
    pos_label: Union[int, float, str] = 1,
):
    """
Computes the Recall Score.

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction.
input_relation: str/vDataFrame
	Relation to use for scoring. This relation can be a view, table, or a 
    customized relation (if an alias is used at the end of the relation). 
    For example: (SELECT ... FROM ...) x
pos_label: int/float/str, optional
	To compute the Recall Score, one of the response column classes must be 
	the positive one. The parameter 'pos_label' represents this class.

Returns
-------
float
	score
	"""
    tn, fn, fp, tp = compute_tn_fn_fp_tp(y_true, y_score, input_relation, pos_label)
    recall = tp / (tp + fn) if (tp + fn != 0) else 0
    return recall


@save_verticapy_logs
def specificity_score(
    y_true: str,
    y_score: str,
    input_relation: Union[str, vDataFrame],
    pos_label: Union[int, float, str] = 1,
):
    """
Computes the Specificity Score.

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction.
input_relation: str/vDataFrame
	Relation to use for scoring. This relation can be a view, table, or a 
    customized relation (if an alias is used at the end of the relation). 
    For example: (SELECT ... FROM ...) x
pos_label: int/float/str, optional
	To compute the Specificity Score, one of the response column classes must 
	be the positive one. The parameter 'pos_label' represents this class.

Returns
-------
float
	score
	"""
    tn, fn, fp, tp = compute_tn_fn_fp_tp(y_true, y_score, input_relation, pos_label)
    tnr = tn / (tn + fp) if (tn + fp != 0) else 0
    return tnr


#
# TOOLS
#


def aic_score(
    y_true: str, y_score: str, input_relation: Union[str, vDataFrame], k: int = 1,
):
    return aic_bic(y_true=y_true, y_score=y_score, input_relation=input_relation, k=k)[
        0
    ]


def bic_score(
    y_true: str, y_score: str, input_relation: Union[str, vDataFrame], k: int = 1,
):
    return aic_bic(y_true=y_true, y_score=y_score, input_relation=input_relation, k=k)[
        1
    ]


FUNCTIONS_CLASSIFICATION_DICTIONNARY = {
    "aic": aic_score,
    "bic": bic_score,
    "accuracy": accuracy_score,
    "acc": accuracy_score,
    "auc": auc,
    "prc_auc": prc_auc,
    "best_cutoff": roc_curve,
    "best_threshold": roc_curve,
    "recall": recall_score,
    "tpr": recall_score,
    "precision": precision_score,
    "ppv": precision_score,
    "specificity": specificity_score,
    "tnr": specificity_score,
    "negative_predictive_value": negative_predictive_score,
    "npv": negative_predictive_score,
    "log_loss": log_loss,
    "logloss": log_loss,
    "f1": f1_score,
    "mcc": matthews_corrcoef,
    "bm": informedness,
    "informedness": informedness,
    "mk": markedness,
    "markedness": markedness,
    "csi": critical_success_index,
    "critical_success_index": critical_success_index,
    "roc_curve": roc_curve,
    "roc": roc_curve,
    "prc_curve": prc_curve,
    "prc": prc_curve,
    "lift_chart": lift_chart,
    "lift": lift_chart,
}

FUNCTIONS_REGRESSION_DICTIONNARY = {
    "aic": aic_score,
    "bic": bic_score,
    "r2": r2_score,
    "rsquared": r2_score,
    "mae": mean_absolute_error,
    "mean_absolute_error": mean_absolute_error,
    "mse": mean_squared_error,
    "mean_squared_error": mean_squared_error,
    "msle": mean_squared_log_error,
    "mean_squared_log_error": mean_squared_log_error,
    "max": max_error,
    "max_error": max_error,
    "median": median_absolute_error,
    "median_absolute_error": median_absolute_error,
    "var": explained_variance,
    "explained_variance": explained_variance,
}

FUNCTIONS_DICTIONNARY = {
    **FUNCTIONS_CLASSIFICATION_DICTIONNARY,
    **FUNCTIONS_REGRESSION_DICTIONNARY,
}
