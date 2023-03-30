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
from typing import Callable, Literal, Optional, Union, TYPE_CHECKING
import numpy as np

from verticapy._typing import ArrayLike, PythonNumber, PythonScalar, SQLRelation
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._sys import _executeSQL
from verticapy._utils._sql._vertica_version import check_minimum_version

from verticapy.core.tablesample.base import TableSample

if TYPE_CHECKING:
    from verticapy.machine_learning.vertica.base import VerticaModel

"""
Confusion Matrix Functions.
"""


def _compute_tn_fn_fp_tp_from_cm(cm: ArrayLike) -> tuple:
    """
    helper function to compute the final score.
    """
    return cm[0][0], cm[1][0], cm[0][1], cm[1][1]


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
    cm = confusion_matrix(y_true, y_score, input_relation, pos_label)
    return _compute_tn_fn_fp_tp_from_cm(cm)


def _compute_classes_tn_fn_fp_tp_from_cm(cm: ArrayLike) -> list[tuple]:
    """
    helper function to compute the final score.
    """
    n, m = cm.shape
    res = []
    for i in range(m):
        tp = cm[i][i]
        tn = np.diagonal(cm).sum() - cm[i][i]
        fn = cm[:, i].sum() - cm[i][i]
        fp = cm.sum() - tp - tn - fn
        res += [(tn, fn, fp, tp)]
    return res


def _compute_classes_tn_fn_fp_tp(
    y_true: str, y_score: str, input_relation: SQLRelation, labels: list,
) -> list[tuple]:
    """
    A helper function that  computes the confusion matrix 
    and returns  its values  as a tuple of the following: 
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
    labels: list
        List of the response column categories.

    Returns
    -------
    list of tuple
        tn, fn, fp, tp for each class
    """
    cm = multilabel_confusion_matrix(y_true, y_score, input_relation, labels)
    return _compute_classes_tn_fn_fp_tp_from_cm(cm)


def _compute_final_score_from_cm(
    metric: Callable,
    cm: ArrayLike,
    average: Literal["micro", "macro", "weighted", "scores"] = "weighted",
    multi: bool = False,
) -> Union[float, list[float]]:
    """
    Computes the final score by using the different results
    of the multi-confusion matrix.
    """
    if multi:
        confusion_list = _compute_classes_tn_fn_fp_tp_from_cm(cm)
        if average == "weighted":
            score = sum(
                [(args[1] + args[3]) * metric(*args) for args in confusion_list]
            )
            total = sum([(args[1] + args[3]) for args in confusion_list])
            return score / total
        elif average == "macro":
            return np.mean([metric(*args) for args in confusion_list])
        elif average == "micro":
            args = [sum([args[i] for args in confusion_list]) for i in range(4)]
            return metric(*args)
        elif average == "scores":
            return [metric(*args) for args in confusion_list]
        else:
            raise ValueError(
                f"Wrong value for parameter 'average'. Expecting: micro|macro|weighted|scores. Got {average}."
            )
    else:
        return metric(*_compute_tn_fn_fp_tp_from_cm(cm))


def _compute_final_score(
    metric: Callable,
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    average: Literal["micro", "macro", "weighted", "scores"] = "weighted",
    labels: Optional[ArrayLike] = None,
    pos_label: Optional[PythonScalar] = None,
) -> Union[float, list[float]]:
    """
    Computes the final score by using the different results
    of the multi-confusion matrix.
    """
    if pos_label == None and isinstance(labels, type(None)):
        pos_label = 1
    if pos_label == None:
        cm = multilabel_confusion_matrix(y_true, y_score, input_relation, labels)
        return _compute_final_score_from_cm(metric, cm, average=average, multi=True)
    else:
        cm = confusion_matrix(y_true, y_score, input_relation, pos_label=pos_label)
        return _compute_final_score_from_cm(metric, cm, average=average, multi=False)


@check_minimum_version
@save_verticapy_logs
def confusion_matrix(
    y_true: str, y_score: str, input_relation: SQLRelation, pos_label: PythonScalar = 1,
) -> np.ndarray:
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
    Array
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
    y_true: str, y_score: str, input_relation: SQLRelation, labels: ArrayLike,
) -> np.ndarray:
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
    labels: ArrayLike
        List   of   the  response  column  categories.

    Returns
    -------
    Array
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


def _accuracy_score(tn: int, fn: int, fp: int, tp: int) -> float:
    return (tp + tn) / (tp + tn + fn + fp)


@save_verticapy_logs
def accuracy_score(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    average: Literal["micro", "macro", "weighted", "scores"] = "weighted",
    labels: Optional[ArrayLike] = None,
    pos_label: Optional[PythonScalar] = None,
) -> Union[float, list[float]]:
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
    average: str, optional
        The method used to  compute the final score for
        multiclass-classification.
            micro    : positive  and   negative  values 
                       globally.
            macro    : average  of  the  score of  each 
                       class.
            weighted : weighted average of the score of 
                       each class.
            scores   : scores  for   all  the  classes.
    labels: ArrayLike, optional
        List   of   the  response  column   categories.
    pos_label: PythonScalar, optional
        Label to use to identify the positive class. If 
        pos_label is NULL then the global accuracy will 
        be computed.

    Returns
    -------
    float
        score.
    """
    return _compute_final_score(
        _accuracy_score,
        y_true,
        y_score,
        input_relation,
        average=average,
        labels=labels,
        pos_label=pos_label,
    )


def _critical_success_index(tn: int, fn: int, fp: int, tp: int) -> float:
    return tp / (tp + fn + fp) if (tp + fn + fp != 0) else 0


@save_verticapy_logs
def critical_success_index(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    average: Literal["micro", "macro", "weighted", "scores"] = "weighted",
    labels: Optional[ArrayLike] = None,
    pos_label: Optional[PythonScalar] = None,
) -> Union[float, list[float]]:
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
    average: str, optional
        The method used to  compute the final score for
        multiclass-classification.
            micro    : positive  and   negative  values 
                       globally.
            macro    : average  of  the  score of  each 
                       class.
            weighted : weighted average of the score of 
                       each class.
            scores   : scores  for   all  the  classes.
    labels: ArrayLike, optional
        List   of   the  response  column   categories.
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response 
        column  classes must be the positive one.  The 
        parameter 'pos_label' represents this class.

    Returns
    -------
    float
        score.
    """
    return _compute_final_score(
        _critical_success_index,
        y_true,
        y_score,
        input_relation,
        average=average,
        labels=labels,
        pos_label=pos_label,
    )


def _f1_score(tn: int, fn: int, fp: int, tp: int) -> float:
    recall = tp / (tp + fn) if (tp + fn != 0) else 0
    precision = tp / (tp + fp) if (tp + fp != 0) else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall != 0)
        else 0
    )
    return f1


@save_verticapy_logs
def f1_score(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    average: Literal["micro", "macro", "weighted", "scores"] = "weighted",
    labels: Optional[ArrayLike] = None,
    pos_label: Optional[PythonScalar] = None,
) -> Union[float, list[float]]:
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
    average: str, optional
        The method used to  compute the final score for
        multiclass-classification.
            micro    : positive  and   negative  values 
                       globally.
            macro    : average  of  the  score of  each 
                       class.
            weighted : weighted average of the score of 
                       each class.
            scores   : scores  for   all  the  classes.
    labels: ArrayLike, optional
        List   of   the  response  column   categories.
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response 
        column  classes must be the positive one.  The 
        parameter 'pos_label' represents this class.

    Returns
    -------
    float
        score.
    """
    return _compute_final_score(
        _f1_score,
        y_true,
        y_score,
        input_relation,
        average=average,
        labels=labels,
        pos_label=pos_label,
    )


def _informedness(tn: int, fn: int, fp: int, tp: int) -> float:
    tpr = tp / (tp + fn) if (tp + fn != 0) else 0
    tnr = tn / (tn + fp) if (tn + fp != 0) else 0
    return tpr + tnr - 1


@save_verticapy_logs
def informedness(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    average: Literal["micro", "macro", "weighted", "scores"] = "weighted",
    labels: Optional[ArrayLike] = None,
    pos_label: Optional[PythonScalar] = None,
) -> Union[float, list[float]]:
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
    average: str, optional
        The method used to  compute the final score for
        multiclass-classification.
            micro    : positive  and   negative  values 
                       globally.
            macro    : average  of  the  score of  each 
                       class.
            weighted : weighted average of the score of 
                       each class.
            scores   : scores  for   all  the  classes.
    labels: ArrayLike, optional
        List   of   the  response  column   categories.
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response 
        column  classes must be the positive one.  The 
        parameter 'pos_label' represents this class.

    Returns
    -------
    float
        score.
    """
    return _compute_final_score(
        _informedness,
        y_true,
        y_score,
        input_relation,
        average=average,
        labels=labels,
        pos_label=pos_label,
    )


def _markedness(tn: int, fn: int, fp: int, tp: int) -> float:
    ppv = tp / (tp + fp) if (tp + fp != 0) else 0
    npv = tn / (tn + fn) if (tn + fn != 0) else 0
    return ppv + npv - 1


@save_verticapy_logs
def markedness(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    average: Literal["micro", "macro", "weighted", "scores"] = "weighted",
    labels: Optional[ArrayLike] = None,
    pos_label: Optional[PythonScalar] = None,
) -> Union[float, list[float]]:
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
    average: str, optional
        The method used to  compute the final score for
        multiclass-classification.
            micro    : positive  and   negative  values 
                       globally.
            macro    : average  of  the  score of  each 
                       class.
            weighted : weighted average of the score of 
                       each class.
            scores   : scores  for   all  the  classes.
    labels: ArrayLike, optional
        List   of   the  response  column   categories.
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response 
        column  classes must be the positive one.  The 
        parameter 'pos_label' represents this class.

    Returns
    -------
    float
        score.
    """
    return _compute_final_score(
        _markedness,
        y_true,
        y_score,
        input_relation,
        average=average,
        labels=labels,
        pos_label=pos_label,
    )


def _matthews_corrcoef(tn: int, fn: int, fp: int, tp: int) -> float:
    return (
        (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if (tp + fp != 0) and (tp + fn != 0) and (tn + fp != 0) and (tn + fn != 0)
        else 0
    )


@save_verticapy_logs
def matthews_corrcoef(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    average: Literal["micro", "macro", "weighted", "scores"] = "weighted",
    labels: Optional[ArrayLike] = None,
    pos_label: Optional[PythonScalar] = None,
) -> Union[float, list[float]]:
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
    average: str, optional
        The method used to  compute the final score for
        multiclass-classification.
            micro    : positive  and   negative  values 
                       globally.
            macro    : average  of  the  score of  each 
                       class.
            weighted : weighted average of the score of 
                       each class.
            scores   : scores  for   all  the  classes.
    labels: ArrayLike, optional
        List   of   the  response  column   categories.
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response 
        column  classes must be the positive one.  The 
        parameter 'pos_label' represents this class.

    Returns
    -------
    float
        score.
    """
    return _compute_final_score(
        _matthews_corrcoef,
        y_true,
        y_score,
        input_relation,
        average=average,
        labels=labels,
        pos_label=pos_label,
    )


def _negative_predictive_score(tn: int, fn: int, fp: int, tp: int) -> float:
    return tn / (tn + fn) if (tn + fn != 0) else 0


@save_verticapy_logs
def negative_predictive_score(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    average: Literal["micro", "macro", "weighted", "scores"] = "weighted",
    labels: Optional[ArrayLike] = None,
    pos_label: Optional[PythonScalar] = None,
) -> Union[float, list[float]]:
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
    average: str, optional
        The method used to  compute the final score for
        multiclass-classification.
            micro    : positive  and   negative  values 
                       globally.
            macro    : average  of  the  score of  each 
                       class.
            weighted : weighted average of the score of 
                       each class.
            scores   : scores  for   all  the  classes.
    labels: ArrayLike, optional
        List   of   the  response  column   categories.
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response 
        column  classes must be the positive one.  The 
        parameter 'pos_label' represents this class.

    Returns
    -------
    float
        score.
    """
    return _compute_final_score(
        _negative_predictive_score,
        y_true,
        y_score,
        input_relation,
        average=average,
        labels=labels,
        pos_label=pos_label,
    )


def _precision_score(tn: int, fn: int, fp: int, tp: int) -> float:
    return tp / (tp + fp) if (tp + fp != 0) else 0


@save_verticapy_logs
def precision_score(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    average: Literal["micro", "macro", "weighted", "scores"] = "weighted",
    labels: Optional[ArrayLike] = None,
    pos_label: Optional[PythonScalar] = None,
) -> Union[float, list[float]]:
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
    average: str, optional
        The method used to  compute the final score for
        multiclass-classification.
            micro    : positive  and   negative  values 
                       globally.
            macro    : average  of  the  score of  each 
                       class.
            weighted : weighted average of the score of 
                       each class.
            scores   : scores  for   all  the  classes.
    labels: ArrayLike, optional
        List   of   the  response  column   categories.
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response 
        column  classes must be the positive one.  The 
        parameter 'pos_label' represents this class.

    Returns
    -------
    float
        score.
    """
    return _compute_final_score(
        _precision_score,
        y_true,
        y_score,
        input_relation,
        average=average,
        labels=labels,
        pos_label=pos_label,
    )


def _recall_score(tn: int, fn: int, fp: int, tp: int) -> float:
    return tp / (tp + fn) if (tp + fn != 0) else 0


@save_verticapy_logs
def recall_score(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    average: Literal["micro", "macro", "weighted", "scores"] = "weighted",
    labels: Optional[ArrayLike] = None,
    pos_label: Optional[PythonScalar] = None,
) -> Union[float, list[float]]:
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
    average: str, optional
        The method used to  compute the final score for
        multiclass-classification.
            micro    : positive  and   negative  values 
                       globally.
            macro    : average  of  the  score of  each 
                       class.
            weighted : weighted average of the score of 
                       each class.
            scores   : scores  for   all  the  classes.
    labels: ArrayLike, optional
        List   of   the  response  column   categories.
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response 
        column  classes must be the positive one.  The 
        parameter 'pos_label' represents this class.

    Returns
    -------
    float
        score.
    """
    return _compute_final_score(
        _recall_score,
        y_true,
        y_score,
        input_relation,
        average=average,
        labels=labels,
        pos_label=pos_label,
    )


def _specificity_score(tn: int, fn: int, fp: int, tp: int) -> float:
    return tn / (tn + fp) if (tn + fp != 0) else 0


@save_verticapy_logs
def specificity_score(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    average: Literal["micro", "macro", "weighted", "scores"] = "weighted",
    labels: Optional[ArrayLike] = None,
    pos_label: Optional[PythonScalar] = None,
) -> Union[float, list[float]]:
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
    average: str, optional
        The method used to  compute the final score for
        multiclass-classification.
            micro    : positive  and   negative  values 
                       globally.
            macro    : average  of  the  score of  each 
                       class.
            weighted : weighted average of the score of 
                       each class.
            scores   : scores  for   all  the  classes.
    labels: ArrayLike, optional
        List   of   the  response  column   categories.
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response 
        column  classes must be the positive one.  The 
        parameter 'pos_label' represents this class.

    Returns
    -------
    float
        score.
    """
    return _compute_final_score(
        _specificity_score,
        y_true,
        y_score,
        input_relation,
        average=average,
        labels=labels,
        pos_label=pos_label,
    )


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
) -> list[list]:
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
def _compute_multiclass_metric(
    metric: Callable,
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    average: Literal["micro", "macro", "weighted", "scores"] = "weighted",
    labels: Optional[ArrayLike] = None,
    nbins: int = 10000,
) -> Union[float, list[float]]:
    """
    Computes the Multiclass metric.
    """
    if average == "weighted":
        confusion_list = _compute_classes_tn_fn_fp_tp(
            y_true, y_score[1], input_relation, labels
        )
        weights = [args[1] + args[3] for args in confusion_list]
    else:
        # micro is not feasible using AUC.
        weights = [1.0 for args in labels]
    nbins_kw = {"nbins": nbins} if nbins != None else {}
    scores = [
        weights[i]
        * metric(
            y_true,
            y_score[0].format(labels[i]),
            input_relation,
            pos_label=labels[i],
            **nbins_kw,
        )
        for i in range(len(labels))
    ]
    return sum(scores) / sum(weights)


@save_verticapy_logs
def best_cutoff(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    average: Literal["micro", "macro", "weighted", "scores"] = "weighted",
    labels: Optional[ArrayLike] = None,
    pos_label: Optional[PythonScalar] = None,
    nbins: int = 10000,
) -> Union[float, list[float]]:
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
    average: str, optional
        The method used to  compute the final score for
        multiclass-classification.
            micro    : positive  and   negative  values 
                       globally.
            macro    : average  of  the  score of  each 
                       class.
            weighted : weighted average of the score of 
                       each class.
            scores   : scores  for   all  the  classes.
    labels: ArrayLike, optional
        List   of   the  response  column   categories.
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
    if pos_label != None or isinstance(labels, type(None)):
        y_s = y_score if isinstance(y_score, str) else y_score[0].format(pos_label)
        threshold, false_positive, true_positive = _compute_function_metrics(
            y_true=y_true,
            y_score=y_s,
            input_relation=input_relation,
            pos_label=pos_label,
            nbins=nbins,
            fun_sql_name="roc",
        )
        l = [abs(y - x) for x, y in zip(false_positive, true_positive)]
        best_threshold_arg = max(zip(l, range(len(l))))[1]
        best = max(threshold[best_threshold_arg], 0.001)
        return min(best, 0.999)
    else:
        return _compute_multiclass_metric(
            metric=best_cutoff,
            y_true=y_true,
            y_score=y_score,
            input_relation=input_relation,
            average=average,
            labels=labels,
            nbins=nbins,
        )


@save_verticapy_logs
def roc_auc(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    average: Literal["micro", "macro", "weighted", "scores"] = "weighted",
    labels: Optional[ArrayLike] = None,
    pos_label: Optional[PythonScalar] = None,
    nbins: int = 10000,
) -> Union[float, list[float]]:
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
    average: str, optional
        The method used to  compute the final score for
        multiclass-classification.
            micro    : positive  and   negative  values 
                       globally.
            macro    : average  of  the  score of  each 
                       class.
            weighted : weighted average of the score of 
                       each class.
            scores   : scores  for   all  the  classes.
    labels: ArrayLike, optional
        List   of   the  response  column   categories.
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
    if pos_label != None or isinstance(labels, type(None)):
        y_s = y_score if isinstance(y_score, str) else y_score[0].format(pos_label)
        threshold, false_positive, true_positive = _compute_function_metrics(
            y_true=y_true,
            y_score=y_s,
            input_relation=input_relation,
            pos_label=pos_label,
            nbins=nbins,
            fun_sql_name="roc",
        )
        return _compute_area(true_positive, false_positive)
    else:
        return _compute_multiclass_metric(
            metric=roc_auc,
            y_true=y_true,
            y_score=y_score,
            input_relation=input_relation,
            average=average,
            labels=labels,
            nbins=nbins,
        )


@save_verticapy_logs
def prc_auc(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    average: Literal["micro", "macro", "weighted", "scores"] = "weighted",
    labels: Optional[ArrayLike] = None,
    pos_label: Optional[PythonScalar] = None,
    nbins: int = 10000,
) -> Union[float, list[float]]:
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
    average: str, optional
        The method used to  compute the final score for
        multiclass-classification.
            micro    : positive  and   negative  values 
                       globally.
            macro    : average  of  the  score of  each 
                       class.
            weighted : weighted average of the score of 
                       each class.
            scores   : scores  for   all  the  classes.
    labels: ArrayLike, optional
        List   of   the  response  column   categories.
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
    if pos_label != None or isinstance(labels, type(None)):
        y_s = y_score if isinstance(y_score, str) else y_score[0].format(pos_label)
        threshold, recall, precision = _compute_function_metrics(
            y_true=y_true,
            y_score=y_s,
            input_relation=input_relation,
            pos_label=pos_label,
            nbins=nbins,
            fun_sql_name="prc",
        )
        return _compute_area(precision, recall)
    else:
        return _compute_multiclass_metric(
            metric=prc_auc,
            y_true=y_true,
            y_score=y_score,
            input_relation=input_relation,
            average=average,
            labels=labels,
            nbins=nbins,
        )


# Logloss Metric.


@save_verticapy_logs
def log_loss(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    average: Literal["micro", "macro", "weighted", "scores"] = "weighted",
    labels: Optional[ArrayLike] = None,
    pos_label: PythonScalar = 1,
) -> Union[float, list[float]]:
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
    average: str, optional
        The  method  used  to  compute  the final score for
        multiclass-classification.
            micro    : positive    and    negative   values 
                       globally.
            macro    : average   of   the   score  of  each 
                       class.
            weighted : weighted  average  of  the score  of 
                       each class.
    labels: ArrayLike, optional
        List   of    the    response   column    categories.
    pos_label: PythonScalar, optional
        To compute the log loss,  one of the response column 
        classes  must  be  the  positive one.  The parameter 
        'pos_label' represents this class.

    Returns
    -------
    float
        score.
    """
    if pos_label != None or isinstance(labels, type(None)):
        y_s = y_score if isinstance(y_score, str) else y_score[0].format(pos_label)
        return _executeSQL(
            query=f"""
                SELECT 
                    /*+LABEL('learn.metrics.logloss')*/ 
                    AVG(CASE 
                            WHEN {y_true} = '{pos_label}' 
                            THEN - LOG({y_s}::float + 1e-90) 
                            ELSE - LOG(1 - {y_s}::float + 1e-90) 
                        END) 
                FROM {input_relation} 
                WHERE {y_true} IS NOT NULL 
                  AND {y_s} IS NOT NULL;""",
            title="Computing log loss.",
            method="fetchfirstelem",
        )
    else:
        return _compute_multiclass_metric(
            metric=log_loss,
            y_true=y_true,
            y_score=y_score,
            input_relation=input_relation,
            average=average,
            labels=labels,
            nbins=None,
        )


"""
Reports.
"""
FUNCTIONS_CONFUSION_DICTIONNARY = {
    "accuracy": _accuracy_score,
    "acc": _accuracy_score,
    "recall": _recall_score,
    "tpr": _recall_score,
    "precision": _precision_score,
    "ppv": _precision_score,
    "specificity": _specificity_score,
    "tnr": _specificity_score,
    "negative_predictive_value": _negative_predictive_score,
    "npv": _negative_predictive_score,
    "f1": _f1_score,
    "f1_score": _f1_score,
    "mcc": _matthews_corrcoef,
    "bm": _informedness,
    "informedness": _informedness,
    "mk": _markedness,
    "markedness": _markedness,
    "csi": _critical_success_index,
    "critical_success_index": _critical_success_index,
}

FUNCTIONS_OTHER_METRICS_DICTIONNARY = {
    "auc": roc_auc,
    "prc_auc": prc_auc,
    "best_cutoff": best_cutoff,
    "best_threshold": best_cutoff,
    "log_loss": log_loss,
    "logloss": log_loss,
}


@save_verticapy_logs
def classification_report(
    y_true: Optional[str] = None,
    y_score: Optional[list] = None,
    input_relation: Optional[SQLRelation] = None,
    metrics: Union[None, str, list[str]] = None,
    labels: Optional[ArrayLike] = None,
    cutoff: Optional[PythonNumber] = None,
    nbins: int = 10000,
    estimator: Optional["VerticaModel"] = None,
) -> Union[float, TableSample]:
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
    metrics: list, optional
        List of the metrics to use to compute the final 
        report.
            accuracy    : Accuracy
            aic         : Akaike’s  Information  Criterion
            auc         : Area Under the Curve (ROC)
            best_cutoff : Cutoff  which optimised the  ROC 
                          Curve prediction.
            bic         : Bayesian  Information  Criterion
            bm          : Informedness = tpr + tnr - 1
            csi         : Critical Success Index 
                          = tp / (tp + fn + fp)
            f1          : F1 Score 
            logloss     : Log Loss
            mcc         : Matthews Correlation Coefficient 
            mk          : Markedness = ppv + npv - 1
            npv         : Negative Predictive Value 
                          = tn / (tn + fn)
            prc_auc     : Area Under the Curve (PRC)
            precision   : Precision = tp / (tp + fp)
            recall      : Recall = tp / (tp + fn)
            specificity : Specificity = tn / (tn + fp)
    labels: ArrayLike, optional
    	List of the response column categories to use.
    cutoff: PythonNumber, optional
    	Cutoff  for which the tested category will  be 
        accepted as prediction.
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
    estimator: object, optional
        Estimator to use to compute the classification 
        report.

    Returns
    -------
    TableSample
     	report.
	"""
    return_scalar = False
    if isinstance(metrics, str):
        metrics = [metrics]
        return_scalar = True
    if estimator:
        num_classes = len(estimator.classes_)
        labels = labels if (num_classes != 2) else [estimator.classes_[1]]
    else:
        labels = [1] if isinstance(labels, type(None)) else labels
        num_classes = len(labels) + 1
    if len(labels) == 1:
        key = "value"
    else:
        key = pos_label
    if isinstance(metrics, type(None)):
        metrics = [
            "auc",
            "prc_auc",
            "accuracy",
            "log_loss",
            "precision",
            "recall",
            "f1",
            "mcc",
            "informedness",
            "markedness",
            "csi",
        ]
    values = {"index": metrics}
    if (cutoff == None) and num_classes > 2:
        if estimator:
            cm = estimator.confusion_matrix()
        else:
            cm = multilabel_confusion_matrix(
                y_true, y_score, input_relation, labels=labels
            )
        all_cm_metrics = _compute_classes_tn_fn_fp_tp_from_cm(cm)
        is_multi = True
    else:
        all_cm_metrics = []
        is_multi = False
    for idx, pos_label in enumerate(labels):
        if is_multi:
            tn, fn, fp, tp = all_cm_metrics[idx]
        else:
            if estimator:
                cm = estimator.confusion_matrix(pos_label=pos_label, cutoff=cutoff)
            else:
                y_s = y_score[0].format(pos_label)
                y_p = y_score[1]
                y_t = f"DECODE({y_true}, '{pos_label}', 1, 0)"
                cm = confusion_matrix(y_true, y_p, input_relation, pos_label=pos_label)
            tn, tp = cm[0][0], cm[1][1]
            fn, fp = cm[1][0], cm[0][1]
        values[key] = []
        for m in metrics:
            if m in FUNCTIONS_CONFUSION_DICTIONNARY:
                fun = FUNCTIONS_CONFUSION_DICTIONNARY[m]
                values[key] += [fun(tn, fn, fp, tp)]
            elif m in FUNCTIONS_OTHER_METRICS_DICTIONNARY:
                if estimator:
                    values[key] += [
                        estimator.score(pos_label=pos_label, metric=m, nbins=nbins)
                    ]
                else:
                    fun = FUNCTIONS_OTHER_METRICS_DICTIONNARY[m]
                    values[key] += [fun(y_t, y_s, input_relation, pos_label=1)]
            else:
                possible_metrics = list(FUNCTIONS_CONFUSION_DICTIONNARY) + list(
                    FUNCTIONS_OTHER_METRICS_DICTIONNARY
                )
                possible_metrics = "|".join(possible_metrics)
                raise ValueError(
                    f"Undefined Metric '{m}'. Must be in {possible_metrics}."
                )
        if not (is_multi):
            all_cm_metrics += [(tn, fn, fp, tp)]
    res = TableSample(values)
    if num_classes > 2:
        return_scalar = False
        res_array = res.to_numpy()
        n, m = res_array.shape
        avg_macro, avg_micro, avg_weighted = [], [], []
        for i in range(n):
            avg_macro += [np.mean(res_array[i])]
            weights = np.array([args[3] + args[1] for args in all_cm_metrics])
            avg_weighted += [(res_array[i] * weights).sum() / weights.sum()]
        res.values["avg_macro"] = avg_macro
        res.values["avg_weighted"] = avg_weighted
        args = [sum([args[i] for args in all_cm_metrics]) for i in range(4)]
        avg_micro = []
        for m in metrics:
            if m in FUNCTIONS_CONFUSION_DICTIONNARY:
                fun = FUNCTIONS_CONFUSION_DICTIONNARY[m]
                avg_micro += [fun(*args)]
            else:
                avg_micro += [None]
        res.values["avg_micro"] = avg_micro
    if return_scalar:
        res_array = res.to_numpy()
        n, m = res_array.shape
        if n == 1 and m == 1:
            return float(res_array[0][0])
    return res
