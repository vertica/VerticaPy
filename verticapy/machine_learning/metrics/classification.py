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

from verticapy._typing import PythonScalar, SQLRelation
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._vertica_version import check_minimum_version

from verticapy.core.tablesample.base import TableSample
from verticapy.core.vdataframe.base import vDataFrame

from verticapy.machine_learning._utils import (
    _compute_metric_query,
    _compute_tn_fn_fp_tp,
)


@save_verticapy_logs
def accuracy_score(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
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
input_relation: SQLRelation
	Relation to use for scoring. This relation can be a view, table, or a 
    customized relation (if an alias is used at the end of the relation). 
    For example: (SELECT ... FROM ...) x
pos_label: PythonScalar, optional
	Label to use to identify the positive class. If pos_label is NULL then the
	global accuracy will be computed.

Returns
-------
float
	score
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
def auc(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    pos_label: PythonScalar = 1,
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
pos_label: PythonScalar, optional
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
    from verticapy.machine_learning.model_selection.model_validation import roc_curve

    return roc_curve(
        y_true, y_score, input_relation, pos_label, nbins=nbins, auc_roc=True
    )


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
Computes a classification report using multiple metrics (AUC, accuracy, PRC 
AUC, F1...). It will consider each category as positive and switch to the 
next one during the computation.

Parameters
----------
y_true: str, optional
	Response column.
y_score: list, optional
	List containing the probability and the prediction.
input_relation: SQLRelation, optional
	Relation to use for scoring. This relation can be a view, table, or a 
    customized relation (if an alias is used at the end of the relation). 
    For example: (SELECT ... FROM ...) x
labels: list, optional
	List of the response column categories to use.
cutoff: PythonNumber / list, optional
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
TableSample
 	An object containing the result. For more information, see
 	utilities.TableSample.
	"""
    from verticapy.machine_learning.model_selection.model_validation import roc_curve

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
    return TableSample(values)


@check_minimum_version
@save_verticapy_logs
def confusion_matrix(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
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
input_relation: SQLRelation
	Relation to use for scoring. This relation can be a view, table, or a 
    customized relation (if an alias is used at the end of the relation). 
    For example: (SELECT ... FROM ...) x
pos_label: str / PythonNumber, optional
	To compute the one dimension Confusion Matrix, one of the response column 
	class must be the positive one. The parameter 'pos_label' represents 
	this class.

Returns
-------
TableSample
 	An object containing the result. For more information, see
 	utilities.TableSample.
	"""
    if isinstance(input_relation, str):
        relation = input_relation
    else:
        relation = input_relation._genSQL()
    result = TableSample.read_sql(
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
    y_true: str, y_score: str, input_relation: SQLRelation, pos_label: PythonScalar = 1,
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
pos_label: PythonScalar, optional
	To compute the CSI, one of the response column classes must be the 
	positive one. The parameter 'pos_label' represents this class.

Returns
-------
float
	score
	"""
    tn, fn, fp, tp = _compute_tn_fn_fp_tp(y_true, y_score, input_relation, pos_label)
    csi = tp / (tp + fn + fp) if (tp + fn + fp != 0) else 0
    return csi


@save_verticapy_logs
def f1_score(
    y_true: str, y_score: str, input_relation: SQLRelation, pos_label: PythonScalar = 1,
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
pos_label: PythonScalar, optional
	To compute the F1 Score, one of the response column classes must be the 
	positive one. The parameter 'pos_label' represents this class.

Returns
-------
float
	score
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
pos_label: PythonScalar, optional
	To compute the informedness, one of the response column classes must be the 
	positive one. The parameter 'pos_label' represents this class.

Returns
-------
float
	score
	"""
    tn, fn, fp, tp = _compute_tn_fn_fp_tp(y_true, y_score, input_relation, pos_label)
    tpr = tp / (tp + fn) if (tp + fn != 0) else 0
    tnr = tn / (tn + fp) if (tn + fp != 0) else 0
    return tpr + tnr - 1


@save_verticapy_logs
def log_loss(
    y_true: str, y_score: str, input_relation: SQLRelation, pos_label: PythonScalar = 1,
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
pos_label: PythonScalar, optional
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
    return _compute_metric_query(
        metric, y_true, y_score, input_relation, "Computing the Log Loss."
    )


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
input_relation: str/vDataFrame
	Relation to use for scoring. This relation can be a view, table, or a 
    customized relation (if an alias is used at the end of the relation). 
    For example: (SELECT ... FROM ...) x
pos_label: PythonScalar, optional
	To compute the markedness, one of the response column classes must be the 
	positive one. The parameter 'pos_label' represents this class.

Returns
-------
float
	score
	"""
    tn, fn, fp, tp = _compute_tn_fn_fp_tp(y_true, y_score, input_relation, pos_label)
    ppv = tp / (tp + fp) if (tp + fp != 0) else 0
    npv = tn / (tn + fn) if (tn + fn != 0) else 0
    return ppv + npv - 1


@save_verticapy_logs
def matthews_corrcoef(
    y_true: str, y_score: str, input_relation: SQLRelation, pos_label: PythonScalar = 1,
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
pos_label: PythonScalar, optional
	To compute the Matthews Correlation Coefficient, one of the response column 
	class must be the positive one. The parameter 'pos_label' represents this 
	class.

Returns
-------
float
	score
	"""
    tn, fn, fp, tp = _compute_tn_fn_fp_tp(y_true, y_score, input_relation, pos_label)
    mcc = (
        (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if (tp + fp != 0) and (tp + fn != 0) and (tn + fp != 0) and (tn + fn != 0)
        else 0
    )
    return mcc


@check_minimum_version
@save_verticapy_logs
def multilabel_confusion_matrix(
    y_true: str, y_score: str, input_relation: SQLRelation, labels: list,
):
    """
Computes the Multi Label Confusion Matrix.

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction.
input_relation: SQLRelation
	Relation to use for scoring. This relation can be a view, table, or a 
    customized relation (if an alias is used at the end of the relation). 
    For example: (SELECT ... FROM ...) x
labels: list
	List of the response column categories.

Returns
-------
TableSample
 	An object containing the result. For more information, see
 	utilities.TableSample.
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
    result = TableSample.read_sql(query=query, title="Computing Confusion Matrix.")
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
    y_true: str, y_score: str, input_relation: SQLRelation, pos_label: PythonScalar = 1,
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
pos_label: PythonScalar, optional
	To compute the Negative Predictive Score, one of the response column class 
	must be the positive one. The parameter 'pos_label' represents this class.

Returns
-------
float
	score
	"""
    tn, fn, fp, tp = _compute_tn_fn_fp_tp(y_true, y_score, input_relation, pos_label)
    npv = tn / (tn + fn) if (tn + fn != 0) else 0
    return npv


@save_verticapy_logs
def prc_auc(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    pos_label: PythonScalar = 1,
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
pos_label: PythonScalar, optional
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
    from verticapy.machine_learning.model_selection.model_validation import prc_curve

    return prc_curve(
        y_true, y_score, input_relation, pos_label, nbins=nbins, auc_prc=True
    )


@save_verticapy_logs
def precision_score(
    y_true: str, y_score: str, input_relation: SQLRelation, pos_label: PythonScalar = 1,
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
pos_label: PythonScalar, optional
	To compute the Precision Score, one of the response column classes must be 
	the positive one. The parameter 'pos_label' represents this class.

Returns
-------
float
	score
	"""
    tn, fn, fp, tp = _compute_tn_fn_fp_tp(y_true, y_score, input_relation, pos_label)
    precision = tp / (tp + fp) if (tp + fp != 0) else 0
    return precision


@save_verticapy_logs
def recall_score(
    y_true: str, y_score: str, input_relation: SQLRelation, pos_label: PythonScalar = 1,
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
pos_label: PythonScalar, optional
	To compute the Recall Score, one of the response column classes must be 
	the positive one. The parameter 'pos_label' represents this class.

Returns
-------
float
	score
	"""
    tn, fn, fp, tp = _compute_tn_fn_fp_tp(y_true, y_score, input_relation, pos_label)
    recall = tp / (tp + fn) if (tp + fn != 0) else 0
    return recall


@save_verticapy_logs
def specificity_score(
    y_true: str, y_score: str, input_relation: SQLRelation, pos_label: PythonScalar = 1,
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
pos_label: PythonScalar, optional
	To compute the Specificity Score, one of the response column classes must 
	be the positive one. The parameter 'pos_label' represents this class.

Returns
-------
float
	score
	"""
    tn, fn, fp, tp = _compute_tn_fn_fp_tp(y_true, y_score, input_relation, pos_label)
    tnr = tn / (tn + fp) if (tn + fp != 0) else 0
    return tnr
