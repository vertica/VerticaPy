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
from collections.abc import Iterable
from typing import Union
import numpy as np

from verticapy._typing import PythonNumber, PythonScalar, SQLRelation
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._sys import _executeSQL
from verticapy._utils._sql._vertica_version import check_minimum_version

from verticapy.core.tablesample.base import TableSample

from verticapy.machine_learning.metrics.regression import _compute_metric_query

"""
General Metrics.
"""


@save_verticapy_logs
def accuracy_score(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    pos_label: PythonScalar = None,
) -> float:
    """
    Computes the Accuracy Score.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction.
    input_relation: SQLRelation
        Relation  to  use  for  scoring.  This  relation 
        can  be a view, table, or a customized  relation 
        (if an alias is used at the end of the relation). 
        For example: (SELECT ... FROM ...) x
    pos_label: PythonScalar, optional
        Label to use to identify the positive class. If 
        pos_label is NULL then the global accuracy will 
        be computed.

    Returns
    -------
    float
        score.
    """
    if pos_label != None:
        tn, fn, fp, tp = _compute_tn_fn_fp_tp(
            y_true, y_score, input_relation, pos_label
        )
        acc = (tp + tn) / (tp + tn + fn + fp)
        return acc
    else:
        try:
            return _compute_metric_query(
                "AVG(CASE WHEN {0} = {1} THEN 1 ELSE 0 END)",
                y_true,
                y_score,
                input_relation,
                "Computing the Accuracy Score.",
            )
        except:
            return _compute_metric_query(
                "AVG(CASE WHEN {0}::varchar = {1}::varchar THEN 1 ELSE 0 END)",
                y_true,
                y_score,
                input_relation,
                "Computing the Accuracy Score.",
            )


@save_verticapy_logs
def log_loss(
    y_true: str, y_score: str, input_relation: SQLRelation, pos_label: PythonScalar = 1,
) -> float:
    """
    Computes the Log Loss.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction Probability.
    input_relation: SQLRelation
        Relation to use for scoring. This relation can be a 
        view, table, or a customized relation (if an  alias 
        is used at the end of the relation). 
        For example: (SELECT ... FROM ...) x
    pos_label: PythonScalar, optional
        To compute the log loss,  one of the response column 
        classes  must  be  the  positive one.  The parameter 
        'pos_label' represents this class.

    Returns
    -------
    float
        score.
    """
    metric = f"""
        AVG(CASE 
                WHEN {{0}} = '{pos_label}' 
                    THEN - LOG({{1}}::float + 1e-90) 
                ELSE - LOG(1 - {{1}}::float + 1e-90) 
            END)"""
    return _compute_metric_query(
        metric, y_true, y_score, input_relation, "Computing the Log Loss."
    )


"""
Confusion Matrix Functions.
"""


def _compute_tn_fn_fp_tp(
    y_true: str, y_score: str, input_relation: SQLRelation, pos_label: PythonScalar = 1,
) -> tuple:
    """
    A helper function that  computes the confusion matrix 
    for  the specified 'pos_label' class and returns  its 
    values as a tuple of the following: 
    true negatives, false negatives, false positives, and 
    true positives.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction.
    input_relation: SQLRelation
        Relation  to use for scoring. This  relation can be a 
        view,  table, or a  customized relation (if an  alias 
        is used at the end of the relation). 
        For example: (SELECT ... FROM ...) x
    pos_label: PythonScalar, optional
        To  compute the Confusion Matrix, one of the  response 
        column classes must be the positive one. The parameter 
        'pos_label' represents this class.

    Returns
    -------
    tuple
        tn, fn, fp, tp
    """
    res = confusion_matrix(y_true, y_score, input_relation, pos_label)
    return res[0][0], res[1][0], res[0][1], res[1][1]


@check_minimum_version
@save_verticapy_logs
def confusion_matrix(
    y_true: str, y_score: str, input_relation: SQLRelation, pos_label: PythonScalar = 1,
) -> TableSample:
    """
    Computes the Confusion Matrix.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction.
    input_relation: SQLRelation
        Relation to use for scoring. This relation can 
        be a view, table, or a customized relation (if 
        an alias is used at the end of the relation). 
        For example: (SELECT ... FROM ...) x
    pos_label: str / PythonNumber, optional
        To compute the one dimension Confusion Matrix, 
        one  of the response column class must  be the 
        positive   one.  The   parameter   'pos_label' 
        represents this class.

    Returns
    -------
    TableSample
        confusion matrix.
    """
    if isinstance(input_relation, str):
        relation = input_relation
    else:
        relation = input_relation._genSQL()
    res = _executeSQL(
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
        method="fetchall",
    )
    return np.array([x[1:-1] for x in res])


@check_minimum_version
@save_verticapy_logs
def multilabel_confusion_matrix(
    y_true: str, y_score: str, input_relation: SQLRelation, labels: list,
) -> TableSample:
    """
    Computes the Multi Label Confusion Matrix.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction.
    input_relation: SQLRelation
        Relation to use for scoring. This relation can 
        be a view, table, or a customized relation (if 
        an alias is used at the end of the relation). 
        For example: (SELECT ... FROM ...) x
    labels: list
        List of the response column categories.

    Returns
    -------
    TableSample
        confusion matrix.
    """
    if isinstance(input_relation, str):
        relation = input_relation
    else:
        relation = input_relation._genSQL()
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
    res = _executeSQL(
        query=query, title="Computing Confusion Matrix.", method="fetchall",
    )
    return np.array([x[1:-1] for x in res])


"""
Confusion Matrix Metrics.
"""


@save_verticapy_logs
def critical_success_index(
    y_true: str, y_score: str, input_relation: SQLRelation, pos_label: PythonScalar = 1,
) -> float:
    """
    Computes the Critical Success Index.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction.
    input_relation: SQLRelation
        Relation to use for scoring. This relation can 
        be a view, table, or a customized relation (if 
        an alias is used at the end of the relation). 
        For example: (SELECT ... FROM ...) x
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response 
        column  classes must be the positive one.  The 
        parameter 'pos_label' represents this class.

    Returns
    -------
    float
        score.
    """
    tn, fn, fp, tp = _compute_tn_fn_fp_tp(y_true, y_score, input_relation, pos_label)
    csi = tp / (tp + fn + fp) if (tp + fn + fp != 0) else 0
    return csi


@save_verticapy_logs
def f1_score(
    y_true: str, y_score: str, input_relation: SQLRelation, pos_label: PythonScalar = 1,
) -> float:
    """
    Computes the F1 Score.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction.
    input_relation: SQLRelation
        Relation to use for scoring. This relation can 
        be a view, table, or a customized relation (if 
        an alias is used at the end of the relation). 
        For example: (SELECT ... FROM ...) x
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response 
        column  classes must be the positive one.  The 
        parameter 'pos_label' represents this class.

    Returns
    -------
    float
        score.
    """
    tn, fn, fp, tp = _compute_tn_fn_fp_tp(y_true, y_score, input_relation, pos_label)
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
    y_true: str, y_score: str, input_relation: SQLRelation, pos_label: PythonScalar = 1,
) -> float:
    """
    Computes the Informedness.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction.
    input_relation: SQLRelation
        Relation to use for scoring. This relation can 
        be a view, table, or a customized relation (if 
        an alias is used at the end of the relation). 
        For example: (SELECT ... FROM ...) x
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response 
        column  classes must be the positive one.  The 
        parameter 'pos_label' represents this class.

    Returns
    -------
    float
        score.
    """
    tn, fn, fp, tp = _compute_tn_fn_fp_tp(y_true, y_score, input_relation, pos_label)
    tpr = tp / (tp + fn) if (tp + fn != 0) else 0
    tnr = tn / (tn + fp) if (tn + fp != 0) else 0
    return tpr + tnr - 1


@save_verticapy_logs
def markedness(
    y_true: str, y_score: str, input_relation: SQLRelation, pos_label: PythonScalar = 1,
):
    """
    Computes the Markedness.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction.
    input_relation: SQLRelation
        Relation to use for scoring. This relation can 
        be a view, table, or a customized relation (if 
        an alias is used at the end of the relation). 
        For example: (SELECT ... FROM ...) x
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response 
        column  classes must be the positive one.  The 
        parameter 'pos_label' represents this class.

    Returns
    -------
    float
        score.
    """
    tn, fn, fp, tp = _compute_tn_fn_fp_tp(y_true, y_score, input_relation, pos_label)
    ppv = tp / (tp + fp) if (tp + fp != 0) else 0
    npv = tn / (tn + fn) if (tn + fn != 0) else 0
    return ppv + npv - 1


@save_verticapy_logs
def matthews_corrcoef(
    y_true: str, y_score: str, input_relation: SQLRelation, pos_label: PythonScalar = 1,
) -> float:
    """
    Computes the Matthews Correlation Coefficient.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction.
    input_relation: SQLRelation
        Relation to use for scoring. This relation can 
        be a view, table, or a customized relation (if 
        an alias is used at the end of the relation). 
        For example: (SELECT ... FROM ...) x
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response 
        column  classes must be the positive one.  The 
        parameter 'pos_label' represents this class.

    Returns
    -------
    float
        score.
    """
    tn, fn, fp, tp = _compute_tn_fn_fp_tp(y_true, y_score, input_relation, pos_label)
    mcc = (
        (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if (tp + fp != 0) and (tp + fn != 0) and (tn + fp != 0) and (tn + fn != 0)
        else 0
    )
    return mcc


@save_verticapy_logs
def negative_predictive_score(
    y_true: str, y_score: str, input_relation: SQLRelation, pos_label: PythonScalar = 1,
) -> float:
    """
    Computes the Negative Predictive Score.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction.
    input_relation: SQLRelation
        Relation to use for scoring. This relation can 
        be a view, table, or a customized relation (if 
        an alias is used at the end of the relation). 
        For example: (SELECT ... FROM ...) x
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response 
        column  classes must be the positive one.  The 
        parameter 'pos_label' represents this class.

    Returns
    -------
    float
        score.
    """
    tn, fn, fp, tp = _compute_tn_fn_fp_tp(y_true, y_score, input_relation, pos_label)
    npv = tn / (tn + fn) if (tn + fn != 0) else 0
    return npv


@save_verticapy_logs
def precision_score(
    y_true: str, y_score: str, input_relation: SQLRelation, pos_label: PythonScalar = 1,
) -> float:
    """
    Computes the Precision Score.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction.
    input_relation: SQLRelation
        Relation to use for scoring. This relation can 
        be a view, table, or a customized relation (if 
        an alias is used at the end of the relation). 
        For example: (SELECT ... FROM ...) x
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response 
        column  classes must be the positive one.  The 
        parameter 'pos_label' represents this class.

    Returns
    -------
    float
        score.
    """
    tn, fn, fp, tp = _compute_tn_fn_fp_tp(y_true, y_score, input_relation, pos_label)
    precision = tp / (tp + fp) if (tp + fp != 0) else 0
    return precision


@save_verticapy_logs
def recall_score(
    y_true: str, y_score: str, input_relation: SQLRelation, pos_label: PythonScalar = 1,
) -> float:
    """
    Computes the Recall Score.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction.
    input_relation: SQLRelation
        Relation to use for scoring. This relation can 
        be a view, table, or a customized relation (if 
        an alias is used at the end of the relation). 
        For example: (SELECT ... FROM ...) x
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response 
        column  classes must be the positive one.  The 
        parameter 'pos_label' represents this class.

    Returns
    -------
    float
        score.
    """
    tn, fn, fp, tp = _compute_tn_fn_fp_tp(y_true, y_score, input_relation, pos_label)
    recall = tp / (tp + fn) if (tp + fn != 0) else 0
    return recall


@save_verticapy_logs
def specificity_score(
    y_true: str, y_score: str, input_relation: SQLRelation, pos_label: PythonScalar = 1,
) -> float:
    """
    Computes the Specificity Score.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction.
    input_relation: SQLRelation
        Relation to use for scoring. This relation can 
        be a view, table, or a customized relation (if 
        an alias is used at the end of the relation). 
        For example: (SELECT ... FROM ...) x
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response 
        column  classes must be the positive one.  The 
        parameter 'pos_label' represents this class.

    Returns
    -------
    float
        score.
    """
    tn, fn, fp, tp = _compute_tn_fn_fp_tp(y_true, y_score, input_relation, pos_label)
    tnr = tn / (tn + fp) if (tn + fp != 0) else 0
    return tnr


"""
AUC / Lift Metrics.
"""

# Special AUC / Lift Methods.


def _compute_area(X: list, Y: list) -> float:
    """
    Computes the area under the curve.
    """
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


def _compute_function_metrics(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    pos_label: PythonScalar = 1,
    nbins: int = 30,
    fun_sql_name: str = "",
) -> list:
    """
    Returns the function metrics.
    """
    if fun_sql_name == "lift_table":
        label = "lift_curve"
    else:
        label = f"{fun_sql_name}_curve"
    if nbins < 0:
        nbins = 999999
    if isinstance(input_relation, str):
        table = input_relation
    else:
        table = input_relation._genSQL()
    if fun_sql_name == "roc":
        X = ["decision_boundary", "false_positive_rate", "true_positive_rate"]
    elif fun_sql_name == "prc":
        X = ["decision_boundary", "recall", "precision"]
    else:
        X = ["*"]
    query_result = _executeSQL(
        query=f"""
        SELECT
            {', '.join(X)}
        FROM
            (SELECT
                /*+LABEL('learn.model_selection.{label}')*/ 
                {fun_sql_name.upper()}(
                        obs, prob 
                        USING PARAMETERS 
                        num_bins = {nbins}) OVER() 
             FROM 
                (SELECT 
                    (CASE 
                        WHEN {y_true} = '{pos_label}' 
                        THEN 1 ELSE 0 END) AS obs, 
                    {y_score}::float AS prob 
                 FROM {table}) AS prediction_output) x""",
        title=f"Computing the {label.upper()}.",
        method="fetchall",
    )
    result = [
        [item[0] for item in query_result],
        [item[1] for item in query_result],
        [item[2] for item in query_result],
    ]
    if fun_sql_name == "prc":
        result[0] = [0] + result[0] + [1]
        result[1] = [1] + result[1] + [0]
        result[2] = [0] + result[2] + [1]
    return result


# Main AUC / Lift Methods.


@save_verticapy_logs
def best_cutoff(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    pos_label: PythonScalar = 1,
    nbins: int = 10000,
) -> float:
    """
    Computes the ROC AUC (Area Under Curve).

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction.
    input_relation: SQLRelation
        Relation to use for scoring. This relation can 
        be a view, table, or a customized relation (if 
        an alias is used at the end of the relation). 
        For example: (SELECT ... FROM ...) x
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response 
        column  classes must be the positive one.  The 
        parameter 'pos_label' represents this class.
    nbins: int, optional
        An integer value that determines the number of 
        decision boundaries. 
        Decision boundaries  are set at equally spaced 
        intervals between 0 and 1, inclusive. 
        Greater  values  for nbins give  more  precise 
        estimations  of the AUC,  but can  potentially 
        decrease  performance.  The  maximum value  is 
        999,999.  If negative,  the  maximum value  is 
        used.

    Returns
    -------
    float
        score.
    """
    threshold, false_positive, true_positive = _compute_function_metrics(
        y_true=y_true,
        y_score=y_score,
        input_relation=input_relation,
        pos_label=pos_label,
        nbins=nbins,
        fun_sql_name="roc",
    )
    l = [abs(y - x) for x, y in zip(false_positive, true_positive)]
    best_threshold_arg = max(zip(l, range(len(l))))[1]
    best = max(threshold[best_threshold_arg], 0.001)
    best = min(best, 0.999)
    return best


@save_verticapy_logs
def roc_auc(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    pos_label: PythonScalar = 1,
    nbins: int = 10000,
) -> float:
    """
    Computes the ROC AUC (Area Under Curve).

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction.
    input_relation: SQLRelation
        Relation to use for scoring. This relation can 
        be a view, table, or a customized relation (if 
        an alias is used at the end of the relation). 
        For example: (SELECT ... FROM ...) x
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response 
        column  classes must be the positive one.  The 
        parameter 'pos_label' represents this class.
    nbins: int, optional
        An integer value that determines the number of 
        decision boundaries. 
        Decision boundaries  are set at equally spaced 
        intervals between 0 and 1, inclusive. 
        Greater  values  for nbins give  more  precise 
        estimations  of the AUC,  but can  potentially 
        decrease  performance.  The  maximum value  is 
        999,999.  If negative,  the  maximum value  is 
        used.

    Returns
    -------
    float
        score.
	"""
    threshold, false_positive, true_positive = _compute_function_metrics(
        y_true=y_true,
        y_score=y_score,
        input_relation=input_relation,
        pos_label=pos_label,
        nbins=nbins,
        fun_sql_name="roc",
    )
    return _compute_area(true_positive, false_positive)


@save_verticapy_logs
def prc_auc(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    pos_label: PythonScalar = 1,
    nbins: int = 10000,
) -> float:
    """
    Computes the area under the curve (AUC) of a 
    Precision-Recall (PRC) curve.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction.
    input_relation: SQLRelation
        Relation to use for scoring. This relation can 
        be a view, table, or a customized relation (if 
        an alias is used at the end of the relation). 
        For example: (SELECT ... FROM ...) x
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response 
        column  classes must be the positive one.  The 
        parameter 'pos_label' represents this class.
    nbins: int, optional
        An integer value that determines the number of 
        decision boundaries. 
        Decision boundaries  are set at equally spaced 
        intervals between 0 and 1, inclusive. 
        Greater  values  for nbins give  more  precise 
        estimations  of the AUC,  but can  potentially 
        decrease  performance.  The  maximum value  is 
        999,999.  If negative,  the  maximum value  is 
        used.

    Returns
    -------
    float
        score.
    """
    threshold, recall, precision = _compute_function_metrics(
        y_true=y_true,
        y_score=y_score,
        input_relation=input_relation,
        pos_label=pos_label,
        nbins=nbins,
        fun_sql_name="prc",
    )
    return _compute_area(precision, recall)


"""
Reports.
"""


@save_verticapy_logs
def classification_report(
    y_true: str = "",
    y_score: list = [],
    input_relation: SQLRelation = "",
    labels: list = [],
    cutoff: Union[PythonNumber, list] = [],
    estimator=None,
    nbins: int = 10000,
):
    """
    Computes  a classification  report using  multiple 
    metrics  (AUC, accuracy, PRC AUC, F1...).  It will 
    consider  each category as positive and switch  to 
    the next one during the computation.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction.
    input_relation: SQLRelation
        Relation to use for scoring. This relation can 
        be a view, table, or a customized relation (if 
        an alias is used at the end of the relation). 
        For example: (SELECT ... FROM ...) x
    labels: list, optional
    	List of the response column categories to use.
    cutoff: PythonNumber / list, optional
    	Cutoff  for which the tested category will  be 
        accepted as prediction. 
    	For  multiclass classification, the list  will 
        represent the the classes  threshold. If it is 
        empty, the best cutoff will be used.
    estimator: object, optional
    	Estimator to use to compute the classification 
        report.
    nbins: int, optional
        [Used to compute ROC AUC, PRC AUC and the best 
        cutoff]
        An integer value that determines the number of 
        decision boundaries. 
        Decision boundaries  are set at equally spaced 
        intervals between 0 and 1, inclusive. 
        Greater  values  for nbins give  more  precise 
        estimations  of the AUC,  but can  potentially 
        decrease  performance.  The  maximum value  is 
        999,999.  If negative,  the  maximum value  is 
        used.

    Returns
    -------
    TableSample
     	report.
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
    for idx, pos_label in enumerate(labels):
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
            y_s = y_score[0].format(pos_label)
            y_p = y_score[0].format(pos_label)
            y_t = f"DECODE({y_true}, '{pos_label}', 1, 0)"
            matrix = confusion_matrix(y_true, y_p, input_relation, pos_label)
        tn = matrix[0][0]
        fn = matrix[1][0]
        fp = matrix[0][1]
        tp = matrix[1][1]
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
            auc_score = roc_auc(y_t, y_s, input_relation, 1)
            prc_auc_score = prc_auc(y_t, y_s, input_relation, 1)
            logloss = log_loss(y_t, y_s, input_relation, 1)
            if not (cutoff):
                current_cutoff = best_cutoff(y_t, y_s, input_relation, nbins=nbins,)
            elif isinstance(cutoff, Iterable):
                if len(cutoff) == 1:
                    current_cutoff = cutoff[0]
                else:
                    current_cutoff = cutoff[idx]
            else:
                current_cutoff = cutoff
        if len(labels) == 1:
            pos_label = "value"
        values[pos_label] = [
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
    return TableSample(values)
