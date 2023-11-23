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
from typing import Callable, Literal, Optional, Union, TYPE_CHECKING
import numpy as np
from verticapy._typing import (
    ArrayLike,
    NoneType,
    PythonNumber,
    PythonScalar,
    SQLRelation,
)
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._sys import _executeSQL
from verticapy._utils._sql._vertica_version import check_minimum_version
from verticapy._utils._map import param_docstring

from verticapy.core.tablesample.base import TableSample

if TYPE_CHECKING:
    from verticapy.machine_learning.vertica.base import VerticaModel

"""
Confusion Matrix Functions.
"""


PARAMETER_DESCRIPTIONS = {
    "y_true": """    y_true: str
        Response column.""",
    "y_score": """    y_score: str
        Prediction.""",
    "input_relation": """    input_relation: SQLRelation
        Relation used for scoring. This relation can 
        be a view, table, or a customized relation (if 
        an alias is used at the end of the relation). 
        For example: (SELECT ... FROM ...) x""",
    "average": """    average: str, optional
        The method used to  compute the final score for
        multiclass-classification.
            micro    : positive  and   negative  values 
                    globally.
            macro    : average  of  the  score of  each 
                    class.
            weighted : weighted average of the score of 
                    each class.
            scores   : scores  for   all  the  classes.""",
    "labels": """    labels: ArrayLike
        List   of   the  response  column   categories.""",
    "cm_pos_label": """    pos_label: str / PythonNumber, optional
        To compute the one dimensional confusion 
        matrix, one  of the response column classes must
        be the positive class. The parameter 'pos_label' 
        represents this class.""",
}


def _compute_tn_fn_fp_tp_from_cm(cm: ArrayLike) -> tuple:
    """
    helper function to compute the final score.
    """
    return round(cm[0][0]), round(cm[1][0]), round(cm[0][1]), round(cm[1][1])


def _compute_tn_fn_fp_tp(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    pos_label: PythonScalar = 1,
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
        To  compute the confusion matrix, one of the  response
        column classes must be the positive class. The
        parameter 'pos_label' represents this class.

    Returns
    -------
    tuple
        tn, fn, fp, tp
    """
    cm = confusion_matrix(y_true, y_score, input_relation, pos_label=pos_label)
    return _compute_tn_fn_fp_tp_from_cm(cm)


def _compute_classes_tn_fn_fp_tp_from_cm(cm: ArrayLike) -> list[tuple]:
    """
    helper function to compute the final score.
    """
    m = cm.shape[1]
    res = []
    for i in range(m):
        tp = cm[i][i]
        fp = cm[:, i].sum() - cm[i][i]
        fn = cm[i, :].sum() - cm[i][i]
        tn = cm.sum() - fp - fn - tp
        res += [(round(tn), round(fn), round(fp), round(tp))]
    return res


def _compute_classes_tn_fn_fp_tp(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    labels: ArrayLike,
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
    labels: ArrayLike
        List of the response column categories.

    Returns
    -------
    list of tuple
        tn, fn, fp, tp for each class
    """
    cm = confusion_matrix(y_true, y_score, input_relation, labels=labels)
    return _compute_classes_tn_fn_fp_tp_from_cm(cm)


def _compute_final_score_from_cm(
    metric: Callable,
    cm: ArrayLike,
    average: Literal[None, "binary", "micro", "macro", "scores", "weighted"] = None,
    multi: bool = False,
) -> Union[float, list[float]]:
    """
    Computes the final score by using the different results
    of the multi-confusion matrix.
    """
    if metric == _accuracy_score and isinstance(average, NoneType):
        return np.trace(cm) / np.sum(cm)
    elif metric == _balanced_accuracy_score and isinstance(average, NoneType):
        return _compute_final_score_from_cm(
            metric=_recall_score, cm=cm, average="macro", multi=multi
        )
    elif multi:
        confusion_list = _compute_classes_tn_fn_fp_tp_from_cm(cm)
        if average == "binary":
            if len(confusion_list) > 1:
                raise IndexError(
                    "Too many values in 'confusion_list' for parameter average='binary'."
                )
            args = confusion_list[0]
            return metric(*args)
        elif average == "weighted":
            score = sum((args[1] + args[3]) * metric(*args) for args in confusion_list)
            total = sum((args[1] + args[3]) for args in confusion_list)
            return score / total
        elif average == "macro":
            return np.mean([metric(*args) for args in confusion_list])
        elif average == "micro":
            args = [sum(args[i] for args in confusion_list) for i in range(4)]
            return metric(*args)
        elif isinstance(average, NoneType) or average == "scores":
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
    average: Literal[None, "binary", "micro", "macro", "scores", "weighted"] = None,
    labels: Optional[ArrayLike] = None,
    pos_label: Optional[PythonScalar] = None,
) -> Union[float, list[float]]:
    """
    Computes the final score by using the different results
    of the multi-confusion matrix.
    """
    if (
        not (isinstance(pos_label, NoneType))
        and not (isinstance(average, NoneType))
        and average != "binary"
    ):
        raise ValueError(
            "The 'pos_label' parameter can only be used when the 'average' "
            "parameter is set to 'binary' or left undefined. This error can "
            "also occur when you are using a binary classifier; in that case, "
            "the 'average' parameter can only be set to 'binary' or left undefined."
        )
    if not (isinstance(pos_label, NoneType)) and not (isinstance(labels, NoneType)):
        labels = None
    if (
        isinstance(pos_label, NoneType)
        and isinstance(labels, NoneType)
        and average == "binary"
    ):
        pos_label = 1
    elif isinstance(pos_label, NoneType) and isinstance(labels, NoneType):
        labels = _executeSQL(
            query=f"""SELECT DISTINCT({y_true}) FROM {input_relation} WHERE {y_true} IS NOT NULL""",
            title="Computing 'y_true' distinct categories.",
            method="fetchall",
        )
        labels = np.array(labels)[:, 0]
    if isinstance(pos_label, NoneType):
        kwargs, multi = {"labels": labels}, True
    else:
        kwargs, multi = {"pos_label": pos_label}, False
    cm = confusion_matrix(y_true, y_score, input_relation, **kwargs)
    return _compute_final_score_from_cm(metric, cm, average=average, multi=multi)


@check_minimum_version
@save_verticapy_logs
def confusion_matrix(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    labels: Optional[ArrayLike] = None,
    pos_label: Optional[PythonScalar] = None,
) -> np.ndarray:
    """
    Computes the confusion matrix.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction.
    input_relation: SQLRelation
        Relation  used for  scoring.  This  relation
        can  be a view, table, or a customized  relation
        (if an alias is used at the end of the relation).
        For example: (SELECT ... FROM ...) x
    average: str, optional
        The method used to  compute the final score for
        multiclass-classification.

        - binary:
            considers one of the classes  as
            positive  and  use  the   binary
            confusion  matrix to compute the
            score.

        - micro:
            positive  and   negative  values globally.

        - macro:
            average  of  the  score of  each class.

        - score:
            scores  for   all  the  classes.

        - weighted :
            weighted average of the score of
            each class.

        - None:
            accuracy.

    labels: ArrayLike, optional
        List   of   the  response  column   categories.
    pos_label: PythonScalar, optional
        Label used to identify the positive class. If
        pos_label is NULL then the global accuracy is
        be computed.

    Returns
    -------
    Array
        confusion matrix.

    Examples
    ---------

    We should first import verticapy.

    .. ipython:: python

        import verticapy as vp

    Binary Classification
    ^^^^^^^^^^^^^^^^^^^^^^

    Let's create a small dataset that has:

    - true value
    - predicted value

    .. ipython:: python

        data = vp.vDataFrame(
            {
                "y_true": [1, 1, 0, 0, 1],
                "y_pred": [1, 1, 1, 0, 1],
            },
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import confusion_matrix

    Now we can conveniently calculate the score:

    .. ipython:: python

        confusion_matrix(
            y_true  = "y_true",
            y_score = "y_pred",
            input_relation = data,
        )

    It is also possible to directly compute the score
    from the vDataFrame:

    .. ipython:: python

        data.score(
            y_true  = "y_true",
            y_score = "y_pred",
            metric  = "confusion_matrix",
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    Multi-class Classification
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Let's create a small dataset that has:

    - true value with more than two classes
    - predicted value

    .. ipython:: python

        data = vp.vDataFrame(
            {
                "y_true": [1, 2, 0, 0, 1],
                "y_pred": [1, 2, 0, 1, 1],
            },
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import confusion_matrix

    Now we can conveniently calculate the score:

    .. ipython:: python

        confusion_matrix(
            y_true  = "y_true",
            y_score = "y_pred",
            labels = [0,1,2],
            input_relation = data,
        )

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """
    if isinstance(pos_label, NoneType) and isinstance(labels, NoneType):
        pos_label = 1
    if not (isinstance(pos_label, NoneType)):
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
                 FROM {input_relation}) VERTICAPY_SUBTABLE;""",
            title="Computing Confusion matrix.",
            method="fetchall",
        )
        return np.round(np.array([x[1:-1] for x in res])).astype(int)
    elif not (isinstance(labels, NoneType)) and len(labels) > 0:
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
        query += f") AS response FROM {input_relation}) VERTICAPY_SUBTABLE;"
        res = _executeSQL(
            query=query,
            title="Computing Confusion Matrix.",
            method="fetchall",
        )
        return np.round(np.array([x[1:-1] for x in res])).astype(int)
    else:
        raise ValueError("Parameters 'labels' and 'pos_label' can not be both empty.")


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
    average: Literal[None, "binary", "micro", "macro", "scores", "weighted"] = None,
    labels: Optional[ArrayLike] = None,
    pos_label: Optional[PythonScalar] = None,
) -> float:
    """
    Computes the Accuracy score.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction.
    input_relation: SQLRelation
        Relation  used for  scoring.  This  relation
        can  be a view, table, or a customized  relation
        (if an alias is used at the end of the relation).
        For example: (SELECT ... FROM ...) x
    average: str, optional
        The method used to  compute the final score for
        multiclass-classification.

        - binary:
            considers one of the classes  as
            positive  and  use  the   binary
            confusion  matrix to compute the
            score.

        - micro:
            positive  and   negative  values globally.

        - macro:
            average  of  the  score of  each class.

        - score:
            scores  for   all  the  classes.

        - weighted :
            weighted average of the score of
            each class.

        - None:
            accuracy.

    labels: ArrayLike, optional
        List   of   the  response  column   categories.
    pos_label: PythonScalar, optional
        Label used to identify the positive class. If
        pos_label is NULL then the global accuracy is
        be computed.

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
                "y_true": [1, 1, 0, 0, 1],
                "y_pred": [1, 1, 1, 0, 1],
            },
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import accuracy_score

    Now we can conveniently calculate the score:

    .. ipython:: python

        accuracy_score(
            y_true  = "y_true",
            y_score = "y_pred",
            input_relation = data,
        )

    .. note::

        For multi-class classification, we can select
        the ``average`` method for averaging from the
        following options:
        - binary
        - micro
        - macro
        - scores
        - weighted

    It is also possible to directly compute the score
    from the vDataFrame:

    .. ipython:: python

        data.score(
            y_true  = "y_true",
            y_score = "y_pred",
            metric  = "accuracy",
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """
    return _compute_final_score(
        _accuracy_score,
        **locals(),
    )


def _balanced_accuracy_score(tn: int, fn: int, fp: int, tp: int) -> float:
    return (_recall_score(**locals()) + _specificity_score(**locals())) / 2


@save_verticapy_logs
def balanced_accuracy_score(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    average: Literal[None, "binary", "micro", "macro", "scores", "weighted"] = None,
    labels: Optional[ArrayLike] = None,
    pos_label: Optional[PythonScalar] = None,
) -> float:
    """
    Computes the Balanced Accuracy.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction.
    input_relation: SQLRelation
        Relation used for scoring. This relation can
        be a view, table, or a customized relation (if
        an alias is used at the end of the relation).
        For example: (SELECT ... FROM ...) x
    average: str, optional
        The method used to  compute the final score for
        multiclass-classification.

        - binary:
            considers one of the classes  as
            positive  and  use  the   binary
            confusion  matrix to compute the
            score.

        - micro:
            positive  and   negative  values globally.

        - macro:
            average  of  the  score of  each class.

        - score:
            scores  for   all  the  classes.

        - weighted :
            weighted average of the score of
            each class.

        - None:
            accuracy.

    labels: ArrayLike, optional
        List   of   the  response  column   categories.
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response
        column classes must be the positive class. The
        parameter 'pos_label' represents this class.

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
                "y_true": [1, 1, 0, 0, 1],
                "y_pred": [1, 1, 1, 0, 1],
            },
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import balanced_accuracy_score

    Now we can conveniently calculate the score:

    .. ipython:: python

        balanced_accuracy_score(
            y_true  = "y_true",
            y_score = "y_pred",
            input_relation = data,
        )

    .. note::

        For multi-class classification, we can select
        the ``average`` method for averaging from the
        following options:
        - binary
        - micro
        - macro
        - scores
        - weighted

    It is also possible to directly compute the score
    from the vDataFrame:

    .. ipython:: python

        data.score(
            y_true  = "y_true",
            y_score = "y_pred",
            metric  = "balanced_accuracy",
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """
    return _compute_final_score(
        _balanced_accuracy_score,
        **locals(),
    )


def _critical_success_index(tn: int, fn: int, fp: int, tp: int) -> float:
    return tp / (tp + fn + fp) if (tp + fn + fp != 0) else 0.0


@save_verticapy_logs
def critical_success_index(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    average: Literal[None, "binary", "micro", "macro", "scores", "weighted"] = None,
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
        Relation used for scoring. This relation can
        be a view, table, or a customized relation (if
        an alias is used at the end of the relation).
        For example: (SELECT ... FROM ...) x
    average: str, optional
        The method used to  compute the final score for
        multiclass-classification.

        - binary:
            considers one of the classes  as
            positive  and  use  the   binary
            confusion  matrix to compute the
            score.

        - micro:
            positive  and   negative  values globally.

        - macro:
            average  of  the  score of  each class.

        - score:
            scores  for   all  the  classes.

        - weighted :
            weighted average of the score of
            each class.

        - None:
            accuracy.

        If  empty,  the  behaviour  is  similar to  the
        'scores' option.
    labels: ArrayLike, optional
        List   of   the  response  column   categories.
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response
        column classes must be the positive class. The
        parameter 'pos_label' represents this class.

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
                "y_true": [1, 1, 0, 0, 1],
                "y_pred": [1, 1, 1, 0, 1],
            },
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import critical_success_index

    Now we can conveniently calculate the score:

    .. ipython:: python

        critical_success_index(
            y_true  = "y_true",
            y_score = "y_pred",
            input_relation = data,
        )

    .. note::

        For multi-class classification, we can select
        the ``average`` method for averaging from the
        following options:
        - binary
        - micro
        - macro
        - scores
        - weighted

    It is also possible to directly compute the score
    from the vDataFrame:

    .. ipython:: python

        data.score(
            y_true  = "y_true",
            y_score = "y_pred",
            metric  = "critical_success_index",
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """
    return _compute_final_score(
        _critical_success_index,
        **locals(),
    )


def _diagnostic_odds_ratio(tn: int, fn: int, fp: int, tp: int) -> float:
    lrp, lrn = (
        _positive_likelihood_ratio(**locals()),
        _negative_likelihood_ratio(**locals()),
    )
    return lrp / lrn if lrn != 0.0 else np.inf


@save_verticapy_logs
def diagnostic_odds_ratio(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    average: Literal[None, "binary", "micro", "macro", "scores", "weighted"] = None,
    labels: Optional[ArrayLike] = None,
    pos_label: Optional[PythonScalar] = None,
) -> Union[float, list[float]]:
    """
    Computes the Diagnostic odds ratio.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction.
    input_relation: SQLRelation
        Relation used for scoring. This relation can
        be a view, table, or a customized relation (if
        an alias is used at the end of the relation).
        For example: (SELECT ... FROM ...) x
    average: str, optional
        The method used to  compute the final score for
        multiclass-classification.

        - binary:
            considers one of the classes  as
            positive  and  use  the   binary
            confusion  matrix to compute the
            score.

        - micro:
            positive  and   negative  values globally.

        - macro:
            average  of  the  score of  each class.

        - score:
            scores  for   all  the  classes.

        - weighted :
            weighted average of the score of
            each class.

        - None:
            accuracy.
        If  empty,  the  behaviour  is  similar to  the
        'scores' option.
    labels: ArrayLike, optional
        List   of   the  response  column   categories.
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response
        column classes must be the positive class. The
        parameter 'pos_label' represents this class.

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
                "y_true": [1, 1, 0, 0, 1],
                "y_pred": [1, 1, 1, 0, 1],
            },
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import diagnostic_odds_ratio

    Now we can conveniently calculate the score:

    .. ipython:: python

        diagnostic_odds_ratio(
            y_true  = "y_true",
            y_score = "y_pred",
            input_relation = data,
        )

    .. note::

        For multi-class classification, we can select
        the ``average`` method for averaging from the
        following options:
        - binary
        - micro
        - macro
        - scores
        - weighted

    It is also possible to directly compute the score
    from the vDataFrame:

    .. ipython:: python

        data.score(
            y_true  = "y_true",
            y_score = "y_pred",
            metric  = "diagnostic_odds_ratio",
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """
    return _compute_final_score(
        _diagnostic_odds_ratio,
        **locals(),
    )


def _f1_score(tn: int, fn: int, fp: int, tp: int) -> float:
    p, r = _precision_score(**locals()), _recall_score(**locals())
    return 2 * (p * r) / (p + r) if (p + r != 0) else 0.0


@save_verticapy_logs
def f1_score(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    average: Literal[None, "binary", "micro", "macro", "scores", "weighted"] = None,
    labels: Optional[ArrayLike] = None,
    pos_label: Optional[PythonScalar] = None,
) -> Union[float, list[float]]:
    """
    Computes the F1 score.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction.
    input_relation: SQLRelation
        Relation used for scoring. This relation can
        be a view, table, or a customized relation (if
        an alias is used at the end of the relation).
        For example: (SELECT ... FROM ...) x
    average: str, optional
        The method used to  compute the final score for
        multiclass-classification.

        - binary:
            considers one of the classes  as
            positive  and  use  the   binary
            confusion  matrix to compute the
            score.

        - micro:
            positive  and   negative  values globally.

        - macro:
            average  of  the  score of  each class.

        - score:
            scores  for   all  the  classes.

        - weighted :
            weighted average of the score of
            each class.

        - None:
            accuracy.
        If  empty,  the  behaviour  is  similar to  the
        'scores' option.
    labels: ArrayLike, optional
        List   of   the  response  column   categories.
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response
        column classes must be the positive class. The
        parameter 'pos_label' represents this class.

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
                "y_true": [1, 1, 0, 0, 1],
                "y_pred": [1, 1, 1, 0, 1],
            },
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import f1_score

    Now we can conveniently calculate the score:

    .. ipython:: python

        f1_score(
            y_true  = "y_true",
            y_score = "y_pred",
            input_relation = data,
        )

    .. note::

        For multi-class classification, we can select
        the ``average`` method for averaging from the
        following options:
        - binary
        - micro
        - macro
        - scores
        - weighted

    It is also possible to directly compute the score
    from the vDataFrame:

    .. ipython:: python

        data.score(
            y_true  = "y_true",
            y_score = "y_pred",
            metric  = "f1_score",
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """
    return _compute_final_score(
        _f1_score,
        **locals(),
    )


def _false_negative_rate(tn: int, fn: int, fp: int, tp: int) -> float:
    return 1 - _recall_score(**locals())


@save_verticapy_logs
def false_negative_rate(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    average: Literal[None, "binary", "micro", "macro", "scores", "weighted"] = None,
    labels: Optional[ArrayLike] = None,
    pos_label: Optional[PythonScalar] = None,
) -> Union[float, list[float]]:
    """
    Computes the False Negative Rate.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction.
    input_relation: SQLRelation
        Relation used for scoring. This relation can
        be a view, table, or a customized relation (if
        an alias is used at the end of the relation).
        For example: (SELECT ... FROM ...) x
    average: str, optional
        The method used to  compute the final score for
        multiclass-classification.

        - binary:
            considers one of the classes  as
            positive  and  use  the   binary
            confusion  matrix to compute the
            score.

        - micro:
            positive  and   negative  values globally.

        - macro:
            average  of  the  score of  each class.

        - score:
            scores  for   all  the  classes.

        - weighted :
            weighted average of the score of
            each class.

        - None:
            accuracy.
        If  empty,  the  behaviour  is  similar to  the
        'scores' option.
    labels: ArrayLike, optional
        List   of   the  response  column   categories.
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response
        column classes must be the positive class. The
        parameter 'pos_label' represents this class.

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
                "y_true": [1, 1, 0, 0, 1],
                "y_pred": [1, 1, 1, 0, 1],
            },
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import false_negative_rate

    Now we can conveniently calculate the score:

    .. ipython:: python

        false_negative_rate(
            y_true  = "y_true",
            y_score = "y_pred",
            input_relation = data,
        )

    .. note::

        For multi-class classification, we can select
        the ``average`` method for averaging from the
        following options:
        - binary
        - micro
        - macro
        - scores
        - weighted

    It is also possible to directly compute the score
    from the vDataFrame:

    .. ipython:: python

        data.score(
            y_true  = "y_true",
            y_score = "y_pred",
            metric  = "false_negative_rate",
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """
    return _compute_final_score(_false_negative_rate, **locals())


def _false_positive_rate(tn: int, fn: int, fp: int, tp: int) -> float:
    return 1 - _specificity_score(**locals())


@save_verticapy_logs
def false_positive_rate(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    average: Literal[None, "binary", "micro", "macro", "scores", "weighted"] = None,
    labels: Optional[ArrayLike] = None,
    pos_label: Optional[PythonScalar] = None,
) -> Union[float, list[float]]:
    """
    Computes the False Positive Rate.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction.
    input_relation: SQLRelation
        Relation used for scoring. This relation can
        be a view, table, or a customized relation (if
        an alias is used at the end of the relation).
        For example: (SELECT ... FROM ...) x
    average: str, optional
        The method used to  compute the final score for
        multiclass-classification.

        - binary:
            considers one of the classes  as
            positive  and  use  the   binary
            confusion  matrix to compute the
            score.

        - micro:
            positive  and   negative  values globally.

        - macro:
            average  of  the  score of  each class.

        - score:
            scores  for   all  the  classes.

        - weighted :
            weighted average of the score of
            each class.

        - None:
            accuracy.
        If  empty,  the  behaviour  is  similar to  the
        'scores' option.
    labels: ArrayLike, optional
        List   of   the  response  column   categories.
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response
        column classes must be the positive class. The
        parameter 'pos_label' represents this class.

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
                "y_true": [1, 1, 0, 0, 1],
                "y_pred": [1, 1, 1, 0, 1],
            },
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import false_positive_rate

    Now we can conveniently calculate the score:

    .. ipython:: python

        false_positive_rate(
            y_true  = "y_true",
            y_score = "y_pred",
            input_relation = data,
        )

    .. note::

        For multi-class classification, we can select
        the ``average`` method for averaging from the
        following options:
        - binary
        - micro
        - macro
        - scores
        - weighted

    It is also possible to directly compute the score
    from the vDataFrame:

    .. ipython:: python

        data.score(
            y_true  = "y_true",
            y_score = "y_pred",
            metric  = "false_positive_rate",
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """
    return _compute_final_score(
        _false_positive_rate,
        **locals(),
    )


def _false_discovery_rate(tn: int, fn: int, fp: int, tp: int) -> float:
    return 1 - _precision_score(**locals())


@save_verticapy_logs
def false_discovery_rate(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    average: Literal[None, "binary", "micro", "macro", "scores", "weighted"] = None,
    labels: Optional[ArrayLike] = None,
    pos_label: Optional[PythonScalar] = None,
) -> Union[float, list[float]]:
    """
    Computes the False Discovery Rate.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction.
    input_relation: SQLRelation
        Relation used for scoring. This relation can
        be a view, table, or a customized relation (if
        an alias is used at the end of the relation).
        For example: (SELECT ... FROM ...) x
    average: str, optional
        The method used to  compute the final score for
        multiclass-classification.

        - binary:
            considers one of the classes  as
            positive  and  use  the   binary
            confusion  matrix to compute the
            score.

        - micro:
            positive  and   negative  values globally.

        - macro:
            average  of  the  score of  each class.

        - score:
            scores  for   all  the  classes.

        - weighted :
            weighted average of the score of
            each class.

        - None:
            accuracy.
        If  empty,  the  behaviour  is  similar to  the
        'scores' option.
    labels: ArrayLike, optional
        List   of   the  response  column   categories.
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response
        column classes must be the positive class. The
        parameter 'pos_label' represents this class.

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
                "y_true": [1, 1, 0, 0, 1],
                "y_pred": [1, 1, 1, 0, 1],
            },
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import false_discovery_rate

    Now we can conveniently calculate the score:

    .. ipython:: python

        false_discovery_rate(
            y_true  = "y_true",
            y_score = "y_pred",
            input_relation = data,
        )

    .. note::

        For multi-class classification, we can select
        the ``average`` method for averaging from the
        following options:
        - binary
        - micro
        - macro
        - scores
        - weighted

    It is also possible to directly compute the score
    from the vDataFrame:

    .. ipython:: python

        data.score(
            y_true  = "y_true",
            y_score = "y_pred",
            metric  = "false_discovery_rate",
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """
    return _compute_final_score(
        _false_discovery_rate,
        **locals(),
    )


def _false_omission_rate(tn: int, fn: int, fp: int, tp: int) -> float:
    return 1 - _negative_predictive_score(**locals())


@save_verticapy_logs
def false_omission_rate(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    average: Literal[None, "binary", "micro", "macro", "scores", "weighted"] = None,
    labels: Optional[ArrayLike] = None,
    pos_label: Optional[PythonScalar] = None,
) -> Union[float, list[float]]:
    """
    Computes the False Omission Rate.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction.
    input_relation: SQLRelation
        Relation used for scoring. This relation can
        be a view, table, or a customized relation (if
        an alias is used at the end of the relation).
        For example: (SELECT ... FROM ...) x
    average: str, optional
        The method used to  compute the final score for
        multiclass-classification.

        - binary:
            considers one of the classes  as
            positive  and  use  the   binary
            confusion  matrix to compute the
            score.

        - micro:
            positive  and   negative  values globally.

        - macro:
            average  of  the  score of  each class.

        - score:
            scores  for   all  the  classes.

        - weighted :
            weighted average of the score of
            each class.

        - None:
            accuracy.
        If  empty,  the  behaviour  is  similar to  the
        'scores' option.
    labels: ArrayLike, optional
        List   of   the  response  column   categories.
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response
        column classes must be the positive class. The
        parameter 'pos_label' represents this class.

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
                "y_true": [1, 1, 0, 0, 1],
                "y_pred": [1, 1, 1, 0, 1],
            },
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import false_omission_rate

    Now we can conveniently calculate the score:

    .. ipython:: python

        false_omission_rate(
            y_true  = "y_true",
            y_score = "y_pred",
            input_relation = data,
        )

    .. note::

        For multi-class classification, we can select
        the ``average`` method for averaging from the
        following options:
        - binary
        - micro
        - macro
        - scores
        - weighted

    It is also possible to directly compute the score
    from the vDataFrame:

    .. ipython:: python

        data.score(
            y_true  = "y_true",
            y_score = "y_pred",
            metric  = "false_omission_rate",
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """
    return _compute_final_score(
        _false_omission_rate,
        **locals(),
    )


def _fowlkes_mallows_index(tn: int, fn: int, fp: int, tp: int) -> float:
    return np.sqrt(_precision_score(**locals()) * _recall_score(**locals()))


@save_verticapy_logs
def fowlkes_mallows_index(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    average: Literal[None, "binary", "micro", "macro", "scores", "weighted"] = None,
    labels: Optional[ArrayLike] = None,
    pos_label: Optional[PythonScalar] = None,
) -> Union[float, list[float]]:
    """
    Computes the Fowlkes–Mallows index.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction.
    input_relation: SQLRelation
        Relation used for scoring. This relation can
        be a view, table, or a customized relation (if
        an alias is used at the end of the relation).
        For example: (SELECT ... FROM ...) x
    average: str, optional
        The method used to  compute the final score for
        multiclass-classification.

        - binary:
            considers one of the classes  as
            positive  and  use  the   binary
            confusion  matrix to compute the
            score.

        - micro:
            positive  and   negative  values globally.

        - macro:
            average  of  the  score of  each class.

        - score:
            scores  for   all  the  classes.

        - weighted :
            weighted average of the score of
            each class.

        - None:
            accuracy.
        If  empty,  the  behaviour  is  similar to  the
        'scores' option.
    labels: ArrayLike, optional
        List   of   the  response  column   categories.
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response
        column classes must be the positive class. The
        parameter 'pos_label' represents this class.

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
                "y_true": [1, 1, 0, 0, 1],
                "y_pred": [1, 1, 1, 0, 1],
            },
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import fowlkes_mallows_index

    Now we can conveniently calculate the score:

    .. ipython:: python

        fowlkes_mallows_index(
            y_true  = "y_true",
            y_score = "y_pred",
            input_relation = data,
        )

    .. note::

        For multi-class classification, we can select
        the ``average`` method for averaging from the
        following options:
        - binary
        - micro
        - macro
        - scores
        - weighted

    It is also possible to directly compute the score
    from the vDataFrame:

    .. ipython:: python

        data.score(
            y_true  = "y_true",
            y_score = "y_pred",
            metric  = "fowlkes_mallows_index",
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """
    return _compute_final_score(
        _fowlkes_mallows_index,
        **locals(),
    )


def _informedness(tn: int, fn: int, fp: int, tp: int) -> float:
    return _recall_score(**locals()) + _specificity_score(**locals()) - 1


@save_verticapy_logs
def informedness(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    average: Literal[None, "binary", "micro", "macro", "scores", "weighted"] = None,
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
        Relation used for scoring. This relation can
        be a view, table, or a customized relation (if
        an alias is used at the end of the relation).
        For example: (SELECT ... FROM ...) x
    average: str, optional
        The method used to  compute the final score for
        multiclass-classification.

        - binary:
            considers one of the classes  as
            positive  and  use  the   binary
            confusion  matrix to compute the
            score.

        - micro:
            positive  and   negative  values globally.

        - macro:
            average  of  the  score of  each class.

        - score:
            scores  for   all  the  classes.

        - weighted :
            weighted average of the score of
            each class.

        - None:
            accuracy.
        If  empty,  the  behaviour  is  similar to  the
        'scores' option.
    labels: ArrayLike, optional
        List   of   the  response  column   categories.
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response
        column classes must be the positive class. The
        parameter 'pos_label' represents this class.

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
                "y_true": [1, 1, 0, 0, 1],
                "y_pred": [1, 1, 1, 0, 1],
            },
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import informedness

    Now we can conveniently calculate the score:

    .. ipython:: python

        informedness(
            y_true  = "y_true",
            y_score = "y_pred",
            input_relation = data,
        )

    .. note::

        For multi-class classification, we can select
        the ``average`` method for averaging from the
        following options:
        - binary
        - micro
        - macro
        - scores
        - weighted

    It is also possible to directly compute the score
    from the vDataFrame:

    .. ipython:: python

        data.score(
            y_true  = "y_true",
            y_score = "y_pred",
            metric  = "informedness",
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """
    return _compute_final_score(
        _informedness,
        **locals(),
    )


def _markedness(tn: int, fn: int, fp: int, tp: int) -> float:
    ppv = tp / (tp + fp) if (tp + fp != 0) else 0.0
    npv = tn / (tn + fn) if (tn + fn != 0) else 0.0
    return ppv + npv - 1


@save_verticapy_logs
def markedness(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    average: Literal[None, "binary", "micro", "macro", "scores", "weighted"] = None,
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

        - binary:
            considers one of the classes  as
            positive  and  use  the   binary
            confusion  matrix to compute the
            score.

        - micro:
            positive  and   negative  values globally.

        - macro:
            average  of  the  score of  each class.

        - score:
            scores  for   all  the  classes.

        - weighted :
            weighted average of the score of
            each class.

        - None:
            accuracy.
        If  empty,  the  behaviour  is  similar to  the
        'scores' option.
    labels: ArrayLike, optional
        List   of   the  response  column   categories.
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response
        column classes must be the positive class. The
        parameter 'pos_label' represents this class.

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
                "y_true": [1, 1, 0, 0, 1],
                "y_pred": [1, 1, 1, 0, 1],
            },
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import markedness

    Now we can conveniently calculate the score:

    .. ipython:: python

        markedness(
            y_true  = "y_true",
            y_score = "y_pred",
            input_relation = data,
        )

    .. note::

        For multi-class classification, we can select
        the ``average`` method for averaging from the
        following options:
        - binary
        - micro
        - macro
        - scores
        - weighted

    It is also possible to directly compute the score
    from the vDataFrame:

    .. ipython:: python

        data.score(
            y_true  = "y_true",
            y_score = "y_pred",
            metric  = "markedness",
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """
    return _compute_final_score(
        _markedness,
        **locals(),
    )


def _matthews_corrcoef(tn: int, fn: int, fp: int, tp: int) -> float:
    return (
        (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if (tp + fp != 0) and (tp + fn != 0) and (tn + fp != 0) and (tn + fn != 0)
        else 0.0
    )


@save_verticapy_logs
def matthews_corrcoef(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    average: Literal[None, "binary", "micro", "macro", "scores", "weighted"] = None,
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

        - binary:
            considers one of the classes  as
            positive  and  use  the   binary
            confusion  matrix to compute the
            score.

        - micro:
            positive  and   negative  values globally.

        - macro:
            average  of  the  score of  each class.

        - score:
            scores  for   all  the  classes.

        - weighted :
            weighted average of the score of
            each class.

        - None:
            accuracy.
        If  empty,  the  behaviour  is  similar to  the
        'scores' option.
    labels: ArrayLike, optional
        List   of   the  response  column   categories.
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response
        column classes must be the positive class. The
        parameter 'pos_label' represents this class.

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
                "y_true": [1, 1, 0, 0, 1],
                "y_pred": [1, 1, 1, 0, 1],
            },
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import matthews_corrcoef

    Now we can conveniently calculate the score:

    .. ipython:: python

        matthews_corrcoef(
            y_true  = "y_true",
            y_score = "y_pred",
            input_relation = data,
        )

    .. note::

        For multi-class classification, we can select
        the ``average`` method for averaging from the
        following options:
        - binary
        - micro
        - macro
        - scores
        - weighted

    It is also possible to directly compute the score
    from the vDataFrame:

    .. ipython:: python

        data.score(
            y_true  = "y_true",
            y_score = "y_pred",
            metric  = "mcc",
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """
    return _compute_final_score(
        _matthews_corrcoef,
        **locals(),
    )


def _negative_predictive_score(tn: int, fn: int, fp: int, tp: int) -> float:
    return tn / (tn + fn) if (tn + fn != 0) else 0.0


@save_verticapy_logs
def negative_predictive_score(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    average: Literal[None, "binary", "micro", "macro", "scores", "weighted"] = None,
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

        - binary:
            considers one of the classes  as
            positive  and  use  the   binary
            confusion  matrix to compute the
            score.

        - micro:
            positive  and   negative  values globally.

        - macro:
            average  of  the  score of  each class.

        - score:
            scores  for   all  the  classes.

        - weighted :
            weighted average of the score of
            each class.

        - None:
            accuracy.
        If  empty,  the  behaviour  is  similar to  the
        'scores' option.
    labels: ArrayLike, optional
        List   of   the  response  column   categories.
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response
        column classes must be the positive class. The
        parameter 'pos_label' represents this class.

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
                "y_true": [1, 1, 0, 0, 1],
                "y_pred": [1, 1, 1, 0, 1],
            },
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import negative_predictive_score

    Now we can conveniently calculate the score:

    .. ipython:: python

        negative_predictive_score(
            y_true  = "y_true",
            y_score = "y_pred",
            input_relation = data,
        )

    .. note::

        For multi-class classification, we can select
        the ``average`` method for averaging from the
        following options:
        - binary
        - micro
        - macro
        - scores
        - weighted

    It is also possible to directly compute the score
    from the vDataFrame:

    .. ipython:: python

        data.score(
            y_true  = "y_true",
            y_score = "y_pred",
            metric  = "negative_predictive_value",
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """
    return _compute_final_score(
        _negative_predictive_score,
        **locals(),
    )


def _negative_likelihood_ratio(tn: int, fn: int, fp: int, tp: int) -> float:
    return _false_negative_rate(**locals()) / _specificity_score(**locals())


@save_verticapy_logs
def negative_likelihood_ratio(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    average: Literal[None, "binary", "micro", "macro", "scores", "weighted"] = None,
    labels: Optional[ArrayLike] = None,
    pos_label: Optional[PythonScalar] = None,
) -> Union[float, list[float]]:
    """
    Computes the Positive Likelihood ratio.

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

        - binary:
            considers one of the classes  as
            positive  and  use  the   binary
            confusion  matrix to compute the
            score.

        - micro:
            positive  and   negative  values globally.

        - macro:
            average  of  the  score of  each class.

        - score:
            scores  for   all  the  classes.

        - weighted :
            weighted average of the score of
            each class.

        - None:
            accuracy.
        If  empty,  the  behaviour  is  similar to  the
        'scores' option.
    labels: ArrayLike, optional
        List   of   the  response  column   categories.
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response
        column classes must be the positive class. The
        parameter 'pos_label' represents this class.

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
                "y_true": [1, 1, 0, 0, 1],
                "y_pred": [1, 1, 1, 0, 1],
            },
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import negative_likelihood_ratio

    Now we can conveniently calculate the score:

    .. ipython:: python

        negative_likelihood_ratio(
            y_true  = "y_true",
            y_score = "y_pred",
            input_relation = data,
        )

    .. note::

        For multi-class classification, we can select
        the ``average`` method for averaging from the
        following options:
        - binary
        - micro
        - macro
        - scores
        - weighted

    It is also possible to directly compute the score
    from the vDataFrame:

    .. ipython:: python

        data.score(
            y_true  = "y_true",
            y_score = "y_pred",
            metric  = "negative_likelihood_ratio",
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """
    return _compute_final_score(
        _negative_likelihood_ratio,
        **locals(),
    )


def _positive_likelihood_ratio(tn: int, fn: int, fp: int, tp: int) -> float:
    tpr, fpr = _recall_score(**locals()), _false_positive_rate(**locals())
    return tpr / fpr if fpr != 0 else 0.0


@save_verticapy_logs
def positive_likelihood_ratio(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    average: Literal[None, "binary", "micro", "macro", "scores", "weighted"] = None,
    labels: Optional[ArrayLike] = None,
    pos_label: Optional[PythonScalar] = None,
) -> Union[float, list[float]]:
    """
    Computes the Positive Likelihood ratio.

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

        - binary:
            considers one of the classes  as
            positive  and  use  the   binary
            confusion  matrix to compute the
            score.

        - micro:
            positive  and   negative  values globally.

        - macro:
            average  of  the  score of  each class.

        - score:
            scores  for   all  the  classes.

        - weighted :
            weighted average of the score of
            each class.

        - None:
            accuracy.
        If  empty,  the  behaviour  is  similar to  the
        'scores' option.
    labels: ArrayLike, optional
        List   of   the  response  column   categories.
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response
        column classes must be the positive class. The
        parameter 'pos_label' represents this class.

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
                "y_true": [1, 1, 0, 0, 1],
                "y_pred": [1, 1, 1, 0, 1],
            },
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import positive_likelihood_ratio

    Now we can conveniently calculate the score:

    .. ipython:: python

        positive_likelihood_ratio(
            y_true  = "y_true",
            y_score = "y_pred",
            input_relation = data,
        )

    .. note::

        For multi-class classification, we can select
        the ``average`` method for averaging from the
        following options:
        - binary
        - micro
        - macro
        - scores
        - weighted

    It is also possible to directly compute the score
    from the vDataFrame:

    .. ipython:: python

        data.score(
            y_true  = "y_true",
            y_score = "y_pred",
            metric  = "positive_likelihood_ratio",
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """
    return _compute_final_score(
        _positive_likelihood_ratio,
        **locals(),
    )


def _precision_score(tn: int, fn: int, fp: int, tp: int) -> float:
    return tp / (tp + fp) if (tp + fp != 0) else 0.0


@save_verticapy_logs
def precision_score(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    average: Literal[None, "binary", "micro", "macro", "scores", "weighted"] = None,
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

        - binary:
            considers one of the classes  as
            positive  and  use  the   binary
            confusion  matrix to compute the
            score.

        - micro:
            positive  and   negative  values globally.

        - macro:
            average  of  the  score of  each class.

        - score:
            scores  for   all  the  classes.

        - weighted :
            weighted average of the score of
            each class.

        - None:
            accuracy.
        If  empty,  the  behaviour  is  similar to  the
        'scores' option.
    labels: ArrayLike, optional
        List   of   the  response  column   categories.
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response
        column classes must be the positive class. The
        parameter 'pos_label' represents this class.

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
                "y_true": [1, 1, 0, 0, 1],
                "y_pred": [1, 1, 1, 0, 1],
            },
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import precision_score

    Now we can conveniently calculate the score:

    .. ipython:: python

        precision_score(
            y_true  = "y_true",
            y_score = "y_pred",
            input_relation = data,
        )

    .. note::

        For multi-class classification, we can select
        the ``average`` method for averaging from the
        following options:
        - binary
        - micro
        - macro
        - scores
        - weighted

    It is also possible to directly compute the score
    from the vDataFrame:

    .. ipython:: python

        data.score(
            y_true  = "y_true",
            y_score = "y_pred",
            metric  = "precision",
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """
    return _compute_final_score(
        _precision_score,
        **locals(),
    )


def _prevalence_threshold(tn: int, fn: int, fp: int, tp: int) -> float:
    fpr, tpr = _false_positive_rate(**locals()), _recall_score(**locals())
    return np.sqrt(fpr) / (np.sqrt(tpr) + np.sqrt(fpr)) if ((tpr + fpr) != 0) else 0.0


@save_verticapy_logs
def prevalence_threshold(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    average: Literal[None, "binary", "micro", "macro", "scores", "weighted"] = None,
    labels: Optional[ArrayLike] = None,
    pos_label: Optional[PythonScalar] = None,
) -> Union[float, list[float]]:
    """
    Computes the Prevalence Threshold.

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

        - binary:
            considers one of the classes  as
            positive  and  use  the   binary
            confusion  matrix to compute the
            score.

        - micro:
            positive  and   negative  values globally.

        - macro:
            average  of  the  score of  each class.

        - score:
            scores  for   all  the  classes.

        - weighted :
            weighted average of the score of
            each class.

        - None:
            accuracy.
        If  empty,  the  behaviour  is  similar to  the
        'scores' option.
    labels: ArrayLike, optional
        List   of   the  response  column   categories.
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response
        column classes must be the positive class. The
        parameter 'pos_label' represents this class.

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
                "y_true": [1, 1, 0, 0, 1],
                "y_pred": [1, 1, 1, 0, 1],
            },
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import prevalence_threshold

    Now we can conveniently calculate the score:

    .. ipython:: python

        prevalence_threshold(
            y_true  = "y_true",
            y_score = "y_pred",
            input_relation = data,
        )

    .. note::

        For multi-class classification, we can select
        the ``average`` method for averaging from the
        following options:
        - binary
        - micro
        - macro
        - scores
        - weighted

    It is also possible to directly compute the score
    from the vDataFrame:

    .. ipython:: python

        data.score(
            y_true  = "y_true",
            y_score = "y_pred",
            metric  = "prevalence_threshold",
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """
    return _compute_final_score(
        _prevalence_threshold,
        **locals(),
    )


def _recall_score(tn: int, fn: int, fp: int, tp: int) -> float:
    return tp / (tp + fn) if (tp + fn != 0) else 0.0


@save_verticapy_logs
def recall_score(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    average: Literal[None, "binary", "micro", "macro", "scores", "weighted"] = None,
    labels: Optional[ArrayLike] = None,
    pos_label: Optional[PythonScalar] = None,
) -> Union[float, list[float]]:
    """
    Computes the Recall score.

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

        - binary:
            considers one of the classes  as
            positive  and  use  the   binary
            confusion  matrix to compute the
            score.

        - micro:
            positive  and   negative  values globally.

        - macro:
            average  of  the  score of  each class.

        - score:
            scores  for   all  the  classes.

        - weighted :
            weighted average of the score of
            each class.

        - None:
            accuracy.
        If  empty,  the  behaviour  is  similar to  the
        'scores' option.
    labels: ArrayLike, optional
        List   of   the  response  column   categories.
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response
        column classes must be the positive class. The
        parameter 'pos_label' represents this class.

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
                "y_true": [1, 1, 0, 0, 1],
                "y_pred": [1, 1, 1, 0, 1],
            },
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import recall_score

    Now we can conveniently calculate the score:

    .. ipython:: python

        recall_score(
            y_true  = "y_true",
            y_score = "y_pred",
            input_relation = data,
        )

    .. note::

        For multi-class classification, we can select
        the ``average`` method for averaging from the
        following options:
        - binary
        - micro
        - macro
        - scores
        - weighted

    It is also possible to directly compute the score
    from the vDataFrame:

    .. ipython:: python

        data.score(
            y_true  = "y_true",
            y_score = "y_pred",
            metric  = "recall",
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """
    return _compute_final_score(
        _recall_score,
        **locals(),
    )


def _specificity_score(tn: int, fn: int, fp: int, tp: int) -> float:
    return tn / (tn + fp) if (tn + fp != 0) else 0.0


@save_verticapy_logs
def specificity_score(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    average: Literal[None, "binary", "micro", "macro", "scores", "weighted"] = None,
    labels: Optional[ArrayLike] = None,
    pos_label: Optional[PythonScalar] = None,
) -> Union[float, list[float]]:
    """
    Computes the Specificity score.

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

        - binary:
            considers one of the classes  as
            positive  and  use  the   binary
            confusion  matrix to compute the
            score.

        - micro:
            positive  and   negative  values globally.

        - macro:
            average  of  the  score of  each class.

        - score:
            scores  for   all  the  classes.

        - weighted :
            weighted average of the score of
            each class.

        - None:
            accuracy.
        If  empty,  the  behaviour  is  similar to  the
        'scores' option.
    labels: ArrayLike, optional
        List   of   the  response  column   categories.
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response
        column classes must be the positive class. The
        parameter 'pos_label' represents this class.

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
                "y_true": [1, 1, 0, 0, 1],
                "y_pred": [1, 1, 1, 0, 1],
            },
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import specificity_score

    Now we can conveniently calculate the score:

    .. ipython:: python

        specificity_score(
            y_true  = "y_true",
            y_score = "y_pred",
            input_relation = data,
        )

    .. note::

        For multi-class classification, we can select
        the ``average`` method for averaging from the
        following options:
        - binary
        - micro
        - macro
        - scores
        - weighted

    It is also possible to directly compute the score
    from the vDataFrame:

    .. ipython:: python

        data.score(
            y_true  = "y_true",
            y_score = "y_pred",
            metric  = "specificity",
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """
    return _compute_final_score(
        _specificity_score,
        **locals(),
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
    pos_label: Optional[PythonScalar] = None,
    nbins: int = 30,
    fun_sql_name: Optional[str] = None,
) -> list[list[float]]:
    """
    Returns the function metrics.
    """
    if isinstance(pos_label, NoneType):
        pos_label = 1
    if fun_sql_name == "lift_table":
        label = "lift_curve"
    else:
        label = f"{fun_sql_name}_curve"
    if nbins < 0:
        nbins = 999999
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
                 FROM {input_relation}) AS prediction_output) x""",
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

# Functions to simplify the code.


def _check_labels(y_true, labels, input_relation):
    distinct_values = _executeSQL(
        query=f"SELECT DISTINCT {y_true} AS value FROM {input_relation}",
        method="fetchall",
    )
    if set(item[0] for item in distinct_values) != set(labels):
        raise ValueError("Mismatch between distinct values and provided labels")


@save_verticapy_logs
def _compute_multiclass_metric(
    metric: Callable,
    y_true: str,
    y_score: Union[str, ArrayLike],
    input_relation: SQLRelation,
    average: Literal[None, "binary", "micro", "macro", "scores", "weighted"] = None,
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
        weights = [1.0 for args in labels]
    nbins_kw = {"nbins": nbins} if not isinstance(nbins, NoneType) else {}
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


def _compute_micro_multiclass_metric(
    y_true: str,
    y_score: ArrayLike,
    input_relation: SQLRelation,
    labels: ArrayLike,
    nbins: int,
    fun_sql_name: Optional[str],
):
    if fun_sql_name == "roc":
        X = ["decision_boundary", "false_positive_rate", "true_positive_rate"]
    elif fun_sql_name == "prc":
        X = ["decision_boundary", "recall", "precision"]
    labels_query = ""
    for i in range(len(labels)):
        labels_query += f"""
                    WHEN {labels[i]} THEN {y_score[i]}"""
    decode_query = f"""
                    SELECT
                    CASE WHEN {y_true} = distinct_values.value THEN 1 ELSE 0 END AS decoded_value,
                    CASE distinct_values.value
                    {labels_query}
                    ELSE NULL
                    END AS prob_value
                    FROM
                    {input_relation},
                    (SELECT DISTINCT {y_true} AS value FROM {input_relation}) AS distinct_values"""

    query = f"""
            SELECT
                {', '.join(X)}
            FROM
                (SELECT
                    {fun_sql_name.upper()}(
                            obs, prob 
                            USING PARAMETERS 
                            num_bins = {nbins}) OVER() 
                FROM (
                SELECT
                t.decoded_value as obs,
                t.prob_value::float as prob
                FROM (
                {decode_query}) AS t) AS subquery
            ) x
            """
    # The following query does exactly what is required except that I have to tell it that the distinct values are 0,1,2
    query_result = _executeSQL(
        query=query,
        title=f"Computing the",
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


def _get_yscore(
    y_score: Union[str, ArrayLike],
    labels: Optional[ArrayLike] = None,
    pos_label: Optional[PythonScalar] = None,
) -> str:
    """
    Returns the 'y_score' to use to compute the final metric.
    """
    if isinstance(y_score, str):
        return y_score
    elif (len(y_score) == 2) and ("{}" in y_score[0]):
        return y_score[0].format(pos_label)
    elif not isinstance(labels, NoneType) and pos_label in labels:
        idx = list(labels).index(pos_label)
        return y_score[idx]
    elif len(y_score) == 2:
        return y_score[1]
    else:
        raise ValueError("Wrong parameter 'y_score'.")


@save_verticapy_logs
def _compute_final_auc_score(
    y_true: str,
    y_score: Union[str, ArrayLike],
    input_relation: SQLRelation,
    average: Literal[None, "binary", "micro", "macro", "scores", "weighted"] = None,
    labels: Optional[ArrayLike] = None,
    pos_label: Optional[PythonScalar] = None,
    nbins: int = 10000,
    fun_sql_name: Literal["roc", "prc", None] = None,
) -> Union[float, list[float]]:
    """
    Method used to simplify the code and compute
    the AUC.
    """
    if (
        not isinstance(pos_label, NoneType)
        or isinstance(labels, NoneType)
        and isinstance(y_score, str)
    ):
        false_positive, true_positive = _compute_function_metrics(
            y_true=y_true,
            y_score=_get_yscore(y_score, labels, pos_label),
            input_relation=input_relation,
            pos_label=pos_label,
            nbins=nbins,
            fun_sql_name=fun_sql_name,
        )[1:]
        return _compute_area(true_positive, false_positive)
    elif isinstance(y_score, str):
        raise ValueError(
            "Type Error: 'y_score' must be an ArrayLike, not a string. "
            "It is expected to be a list of strings, with each string "
            "representing the probabilities of the classes."
        )
    elif average == "micro":
        _check_labels(y_true=y_true, labels=labels, input_relation=input_relation)
        _, false_positive, true_positive = _compute_micro_multiclass_metric(
            y_true, y_score, input_relation, labels, nbins, fun_sql_name=fun_sql_name
        )
        return _compute_area(true_positive, false_positive)
    elif average == "macro":
        _check_labels(y_true=y_true, labels=labels, input_relation=input_relation)
        fpr = {}
        tpr = {}
        for i in range(len(y_score)):
            fpr[i], tpr[i] = _compute_function_metrics(
                y_true=y_true,
                y_score=y_score[i],
                input_relation=input_relation,
                pos_label=labels[i],
                nbins=nbins,
                fun_sql_name=fun_sql_name,
            )[1:]

        fpr_grid = np.linspace(0.0, 1.0, nbins)
        mean_tpr = np.zeros_like(fpr_grid)

        for i in range(len(y_score)):
            sorted_pairs = sorted(zip(fpr[i], tpr[i]))
            fpr_sorted, tpr_sorted = zip(*sorted_pairs)
            mean_tpr += np.interp(
                fpr_grid, fpr_sorted, tpr_sorted
            )  # linear interpolation

        # Average it and compute AUC
        mean_tpr /= len(y_score)
        roc_auc = _compute_area(mean_tpr.tolist()[::-1], fpr_grid.tolist()[::-1])
        return roc_auc
    elif average in (
        "weighted",
        "scores",
        None,
    ):
        all_scores = []
        for i in range(len(y_score)):
            recall, precision = _compute_function_metrics(
                y_true=y_true,
                y_score=y_score[i],
                input_relation=input_relation,
                pos_label=labels[i],
                nbins=nbins,
                fun_sql_name=fun_sql_name,
            )[1:]
            all_scores += [_compute_area(precision, recall)]
        if average in (
            "scores",
            None,
        ):
            return all_scores
        else:
            cardinality = _executeSQL(
                query=f"""
                    SELECT
                        {y_true},
                        COUNT(*)
                    FROM {input_relation}
                    GROUP BY 1""",
                title=f"Computing the cardinality of all categories.",
                method="fetchall",
            )
            score = 0.0
            weights = 0.0
            for i, l in enumerate(labels):
                for c in cardinality:
                    if str(c[0]) == str(l):
                        score += c[1] * all_scores[i]
                        weights += c[1]
                        break
            return score / weights
    else:
        correct_values = [
            "None",
            "'binary'",
            "'micro'",
            "'macro'",
            "'scores'",
            "'weighted'",
        ]
        raise ValueError(
            f"Incorrect Parameter 'average': Found '{average}'; "
            f"Expected: {', '.join(correct_values)}"
        )


# Main functions.


@save_verticapy_logs
def best_cutoff(
    y_true: str,
    y_score: Union[str, ArrayLike],
    input_relation: SQLRelation,
    average: Literal[None, "binary", "micro", "macro", "scores", "weighted"] = None,
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
    y_score: str | ArrayLike
        Prediction.
    input_relation: SQLRelation
        Relation to use for scoring. This relation can
        be a view, table, or a customized relation (if
        an alias is used at the end of the relation).
        For example: (SELECT ... FROM ...) x
    average: str, optional
        The method used to  compute the final score for
        multiclass-classification.

        - binary:
            considers one of the classes  as
            positive  and  use  the   binary
            confusion  matrix to compute the
            score.

        - micro:
            positive  and   negative  values globally.

        - macro:
            average  of  the  score of  each class.

        - score:
            scores  for   all  the  classes.

        - weighted :
            weighted average of the score of
            each class.

        - None:
            accuracy.
        If  empty,  the  behaviour  is  similar to  the
        'scores' option.
    labels: ArrayLike, optional
        List   of   the  response  column   categories.
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response
        column classes must be the positive class. The
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
                "y_true": [1, 1, 0, 0, 1],
                "y_pred": [1, 1, 1, 0, 1],
            },
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import best_cutoff

    Now we can conveniently calculate the score:

    .. ipython:: python

        best_cutoff(
            y_true  = "y_true",
            y_score = "y_pred",
            input_relation = data,
        )

    .. note::

        For multi-class classification, we can select
        the ``average`` method for averaging from the
        following options:
        - binary
        - micro
        - macro
        - scores
        - weighted

    It is also possible to directly compute the score
    from the vDataFrame:

    .. ipython:: python

        data.score(
            y_true  = "y_true",
            y_score = "y_pred",
            metric  = "best_cutoff",
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """
    if not isinstance(pos_label, NoneType) or isinstance(labels, NoneType):
        threshold, false_positive, true_positive = _compute_function_metrics(
            y_true=y_true,
            y_score=_get_yscore(y_score, labels, pos_label),
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
def average_precision_score(
    y_true: str,
    y_score: Union[str, ArrayLike],
    input_relation: SQLRelation,
    average: Literal[None, "micro", "macro", "scores", "weighted"] = None,
    labels: Optional[ArrayLike] = None,
    pos_label: Optional[PythonScalar] = None,
    nbins: int = 10000,
) -> float:
    """
    Computes the Average precision Score.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str |  ArrayLike
        Prediction.
    input_relation: SQLRelation
        Relation to use for scoring. This relation can
        be a view, table, or a customized relation (if
        an alias is used at the end of the relation).
        For example: (SELECT ... FROM ...) x
    average: str, optional
        The method used to  compute the final score for
        multiclass-classification.

        - micro:
            positive  and   negative  values globally.

        - macro:
            average  of  the  score of  each class.

        - scores:
            scores  for   all  the  classes.

        - weighted:
            weighted average of the score of each class.

        If  empty,  the  behaviour  is  similar to  the
        'scores' option.
    labels: ArrayLike, optional
        List   of   the  response  column   categories.
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response
        column classes must be the positive class. The
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
                "y_true": [1, 1, 0, 0, 1],
                "y_pred": [1, 1, 1, 0, 1],
            },
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import average_precision_score

    Now we can conveniently calculate the score:

    .. ipython:: python

        average_precision_score(
            y_true  = "y_true",
            y_score = "y_pred",
            input_relation = data,
        )

    .. note::

        For multi-class classification, we can select
        the ``average`` method for averaging from the
        following options:
        - binary
        - micro
        - macro
        - scores
        - weighted

    It is also possible to directly compute the score
    from the vDataFrame:

    .. ipython:: python

        data.score(
            y_true  = "y_true",
            y_score = "y_pred",
            metric  = "average_precision_score",
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """
    if not isinstance(pos_label, NoneType) or isinstance(labels, NoneType):
        recall, precision = _compute_function_metrics(
            y_true=y_true,
            y_score=_get_yscore(y_score, labels, pos_label),
            input_relation=input_relation,
            pos_label=pos_label,
            nbins=nbins,
            fun_sql_name="prc",
        )[1:]
        return -np.sum(np.diff(recall) * np.array(precision)[:-1])
    else:
        _check_labels(y_true=y_true, labels=labels, input_relation=input_relation)
    if average == "micro":
        _, recall, precision = _compute_micro_multiclass_metric(
            y_true, y_score, input_relation, labels, nbins, fun_sql_name="prc"
        )
        return -np.sum(np.diff(recall) * np.array(precision)[:-1])
    elif average == "macro":
        aps = 0
        for i, label in enumerate(labels):
            aps += average_precision_score(
                y_true=y_true,
                y_score=y_score[i],
                input_relation=input_relation,
                average=None,
                labels=None,
                pos_label=label,
                nbins=nbins,
            )
        return aps / len(labels)
    elif average == "weighted":
        aps = 0
        all_weights = 0
        for i, label in enumerate(labels):
            weight = len(input_relation[input_relation[y_true] == label])
            aps += weight * average_precision_score(
                y_true=y_true,
                y_score=y_score[i],
                input_relation=input_relation,
                average=None,
                labels=None,
                pos_label=label,
                nbins=nbins,
            )
            all_weights += weight
        return aps / all_weights
    else:
        aps = []
        for i, label in enumerate(labels):
            aps.append(
                average_precision_score(
                    y_true=y_true,
                    y_score=y_score[i],
                    input_relation=input_relation,
                    average=None,
                    labels=None,
                    pos_label=label,
                    nbins=nbins,
                )
            )
        return aps

    return None


@save_verticapy_logs
def roc_auc_score(
    y_true: str,
    y_score: Union[str, ArrayLike],
    input_relation: SQLRelation,
    average: Literal[None, "binary", "micro", "macro", "scores", "weighted"] = None,
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
    y_score: str |  ArrayLike
        When 'pos_label' and 'labels' are not defined, it
        should be a list of  probabilities represented by
        SQL  code for  the different classes in  the same
        order   as  the  labels.  Otherwise,   'pos_label'
        represents   the   main   class,   and   'y_score'
        represents its probability.
    input_relation: SQLRelation
        Relation to use for scoring. This relation can
        be a view, table, or a customized relation (if
        an alias is used at the end of the relation).
        For example: (SELECT ... FROM ...) x
    average: str, optional
        The method used to  compute the final score for
        multiclass-classification.

        - binary:
            considers one of the classes  as
            positive  and  use  the   binary
            confusion  matrix to compute the
            score.

        - micro:
            positive  and   negative  values globally.

        - macro:
            average  of  the  score of  each class.

        - score:
            scores  for   all  the  classes.

        - weighted :
            weighted average of the score of
            each class.

        - None:
            accuracy.

        If  empty,  the  behaviour  is  similar to  the
        'scores' option.
    labels: ArrayLike, optional
        List   of   the  response  column   categories.
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response
        column classes must be the positive class. The
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

    Examples
    ---------

    We should first import verticapy.

    .. ipython:: python

        import verticapy as vp

    Let's create a small dataset that has:

    - true value
    - probability of the true value

    .. important::

        This classification metric does not use the
        predicted value. Instead, it measures the performance
        of a classification model by evaluating the likelihood
        of the true labels given the predicted probabilities.

    .. ipython:: python

        data = vp.vDataFrame(
            {
                "y_true": [1, 1, 0, 0, 1],
                "y_prob": [0.5, 0.9, 0.2, 0.5, 0.6],
            },
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import roc_auc_score

    Now we can conveniently calculate the score:

    .. ipython:: python

        roc_auc_score(
            y_true  = "y_true",
            y_score = "y_prob",
            input_relation = data,
        )

    .. note::

        For multi-class classification, we can select
        the ``average`` method for averaging from the
        following options:
        - binary
        - micro
        - macro
        - scores
        - weighted

    It is also possible to directly compute the score
    from the vDataFrame:

    .. ipython:: python

        data.score(
            y_true  = "y_true",
            y_score = "y_prob",
            metric  = "roc_auc",
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """
    return _compute_final_auc_score(
        y_true=y_true,
        y_score=y_score,
        input_relation=input_relation,
        average=average,
        labels=labels,
        pos_label=pos_label,
        nbins=nbins,
        fun_sql_name="roc",
    )


@save_verticapy_logs
def prc_auc_score(
    y_true: str,
    y_score: Union[str, ArrayLike],
    input_relation: SQLRelation,
    average: Literal[None, "binary", "micro", "macro", "scores", "weighted"] = None,
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
    y_score: str | ArrayLike
        When 'pos_label' and 'labels' are not defined, it
        should be a list of  probabilities represented by
        SQL  code for  the different classes in  the same
        order   as  the  labels.  Otherwise,   'pos_label'
        represents   the   main   class,   and   'y_score'
        represents its probability.
    input_relation: SQLRelation
        Relation to use for scoring. This relation can
        be a view, table, or a customized relation (if
        an alias is used at the end of the relation).
        For example: (SELECT ... FROM ...) x
    average: str, optional
        The method used to  compute the final score for
        multiclass-classification.

        - binary:
            considers one of the classes  as
            positive  and  use  the   binary
            confusion  matrix to compute the
            score.

        - micro:
            positive  and   negative  values globally.

        - macro:
            average  of  the  score of  each class.

        - score:
            scores  for   all  the  classes.

        - weighted :
            weighted average of the score of
            each class.

        If  empty,  the  behaviour  is  similar to  the
        'scores' option.
    labels: ArrayLike, optional
        List   of   the  response  column   categories.
    pos_label: PythonScalar, optional
        To  compute  the metric, one of  the  response
        column classes must be the positive class. The
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

    Examples
    ---------

    We should first import verticapy.

    .. ipython:: python

        import verticapy as vp

    Let's create a small dataset that has:

    - true value
    - probability of the true value

    .. important::

        This classification metric does not use the
        predicted value. Instead, it measures the performance
        of a classification model by evaluating the likelihood
        of the true labels given the predicted probabilities.

    .. ipython:: python

        data = vp.vDataFrame(
            {
                "y_true": [1, 1, 0, 0, 1],
                "y_pred": [0.5, 0.9, 0.2, 0.5, 0.6],
            },
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import prc_auc_score

    Now we can conveniently calculate the score:

    .. ipython:: python

        prc_auc_score(
            y_true  = "y_true",
            y_score = "y_pred",
            input_relation = data,
        )

    .. note::

        For multi-class classification, we can select
        the ``average`` method for averaging from the
        following options:
        - binary
        - micro
        - macro
        - scores
        - weighted

    It is also possible to directly compute the score
    from the vDataFrame:

    .. ipython:: python

        data.score(
            y_true  = "y_true",
            y_score = "y_pred",
            metric  = "prc_auc",
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """
    return _compute_final_auc_score(
        y_true=y_true,
        y_score=y_score,
        input_relation=input_relation,
        average=average,
        labels=labels,
        pos_label=pos_label,
        nbins=nbins,
        fun_sql_name="prc",
    )


# Logloss Metric.


@save_verticapy_logs
def log_loss(
    y_true: str,
    y_score: Union[str, ArrayLike],
    input_relation: SQLRelation,
    average: Literal[None, "binary", "micro", "macro", "scores", "weighted"] = None,
    labels: Optional[ArrayLike] = None,
    pos_label: PythonScalar = 1,
) -> Union[float, list[float]]:
    """
    Computes the Log Loss.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str | ArrayLike
        Prediction Probability.
    input_relation: SQLRelation
        Relation to use for scoring. This relation can be a
        view, table, or a customized relation (if an  alias
        is used at the end of the relation).
        For example: (SELECT ... FROM ...) x
    average: str, optional
        The  method  used  to  compute  the final score for
        multiclass-classification.

        - binary:
            considers one of the classes  as
            positive  and  use  the   binary
            confusion  matrix to compute the
            score.

        - micro:
            positive  and   negative  values globally.

        - macro:
            average  of  the  score of  each class.

        - score:
            scores  for   all  the  classes.

        - weighted :
            weighted average of the score of
            each class.

        - None:
            accuracy.

        If  empty,  the  behaviour  is  similar to  the
        'scores' option.
    labels: ArrayLike, optional
        List   of    the    response   column    categories.
    pos_label: PythonScalar, optional
        To compute the log loss,  one of the response column
        classes must  be  the  positive class. The parameter
        'pos_label' represents this class.

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
    - probability of the true value

    .. important::

        This classification metric does not use the
        predicted value. Instead, it measures the performance
        of a classification model by evaluating the likelihood
        of the true labels given the predicted probabilities.

    .. ipython:: python

        data = vp.vDataFrame(
            {
                "y_true": [1, 1, 0, 0, 1],
                "y_prob": [0.5, 0.9, 0.2, 0.5, 0.6],
            },
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import log_loss

    Now we can conveniently calculate the score:

    .. ipython:: python

        log_loss(
            y_true  = "y_true",
            y_score = "y_prob",
            input_relation = data,
        )

    .. note::

        For multi-class classification, we can select
        the ``average`` method for averaging from the
        following options:
        - binary
        - micro
        - macro
        - scores
        - weighted

    It is also possible to directly compute the score
    from the vDataFrame:

    .. ipython:: python

        data.score(
            y_true  = "y_true",
            y_score = "y_prob",
            metric  = "log_loss",
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """
    if not isinstance(pos_label, NoneType) or isinstance(labels, NoneType):
        y_s = _get_yscore(y_score, labels, pos_label)
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
    "balanced_accuracy_score": _balanced_accuracy_score,
    "ba": _balanced_accuracy_score,
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
    "false_negative_rate": _false_negative_rate,
    "fnr": _false_negative_rate,
    "false_positive_rate": _false_positive_rate,
    "fpr": _false_positive_rate,
    "false_discovery_rate": _false_discovery_rate,
    "fdr": _false_discovery_rate,
    "false_omission_rate": _false_omission_rate,
    "for": _false_omission_rate,
    "positive_likelihood_ratio": _positive_likelihood_ratio,
    "lr+": _positive_likelihood_ratio,
    "negative_likelihood_ratio": _negative_likelihood_ratio,
    "lr-": _negative_likelihood_ratio,
    "diagnostic_odds_ratio": _diagnostic_odds_ratio,
    "dor": _diagnostic_odds_ratio,
    "mcc": _matthews_corrcoef,
    "bm": _informedness,
    "informedness": _informedness,
    "mk": _markedness,
    "markedness": _markedness,
    "ts": _critical_success_index,
    "csi": _critical_success_index,
    "critical_success_index": _critical_success_index,
    "fowlkes_mallows_index": _fowlkes_mallows_index,
    "fm": _fowlkes_mallows_index,
    "prevalence_threshold": _prevalence_threshold,
    "pt": _prevalence_threshold,
}

FUNCTIONS_OTHER_METRICS_DICTIONNARY = {
    "auc": roc_auc_score,
    "roc_auc": roc_auc_score,
    "prc_auc": prc_auc_score,
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
    metrics (AUC, accuracy, PRC AUC, F1...). In the case
    of multiclass classification, it  considers each
    category as positive and switches to the next one
    during the computation.

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
        List of the metrics used to compute the final
        report.

        - accuracy:
            Accuracy

        - aic:
            Akaike's  Information  Criterion

        - auc:
            Area Under the Curve (ROC)

        - ba:
            Balanced Accuracy

            .. math::

                ba = (tpr + tnr) / 2

        - best_cutoff:
            Cutoff  which optimised the  ROC
            Curve prediction.

        - bic:
            Bayesian  Information  Criterion

        - bm:
            Informedness

            .. math::

                tpr + tnr - 1

        - csi:
            Critical Success Index

            .. math::

                csi = tp / (tp + fn + fp)

        - f1:
            F1 Score

        - fdr:
            False Discovery Rate

            .. math::

                1 - ppv

        - fm:
            Fowlkes-Mallows index

            .. math::

                fm = sqrt(ppv * tpr)

        - fnr:
            False Negative Rate

            .. math::

                fnr = fn / (fn + tp)

        - for:
            False Omission Rate = 1 - npv

        - fpr:
            .. math::

                False Positive Rate = fp / (fp + tn)

        - logloss:
            Log Loss

        - lr+:
            Positive Likelihood Ratio

            .. math::

                lr+ = tpr / fpr

        - lr-:
            Negative Likelihood Ratio

            .. math::

                lr- = fnr / tnr

        - dor:
            Diagnostic Odds Ratio

        - mc:
            Matthews Correlation Coefficient

        - mk:
            Markedness

            .. math::

                mk = ppv + npv - 1

        - npv:
            Negative Predictive Value

            .. math::

                npv = tn / (tn + fn)

        - prc_auc:
            Area Under the Curve (PRC)

        - precision:
            .. math::

                Precision = tp / (tp + fp)

        - pt:
            Prevalence Threshold

            .. math::

                pt = sqrt(fpr) / (sqrt(tpr) + sqrt(fpr))

        - recall:
            .. math::

                Recall = tp / (tp + fn)

        - specificity:
            Specificity = tn / (tn + fp)

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
        Estimator used to compute the classification
        report.

    Returns
    -------
    TableSample
        report.

    Examples
    ---------

    We should first import verticapy.

    .. ipython:: python

        import verticapy as vp

    Binary Classification
    ^^^^^^^^^^^^^^^^^^^^^^^

    Let's create a small dataset that has:

    - true value
    - probability of the true value
    - predicted value

    .. ipython:: python

        data = vp.vDataFrame(
            {
                "y_true": [1, 1, 0, 0, 1],
                "y_prob": [0.8, 0.2, 0.1, 0.6, 0.8],
                "y_pred": [1, 0, 0, 1, 1]
            },
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import classification_report

    Now we can conveniently calculate the score:

    .. ipython:: python

        classification_report(
            y_true  = "y_true",
            y_score = ["y_prob", "y_pred"],
            input_relation = data,
        )

    .. important::

        In binary classification, ``y_score`` should
        be a list of two column names:
        - Probability of true value
        - Prediction value

        In the case of multi-class, ``y_score``,
        is the list of two elements:
        - list of column names for class probabilities
          for each class
        - Prediction value

    .. note::

        For multi-class classification, we can select
        the ``average`` method for averaging from the
        following options:
        - binary
        - micro
        - macro
        - scores
        - weighted

    It is also possible to directly compute the score
    from the vDataFrame:

    .. ipython:: python

        data.score(
            y_true  = "y_true",
            y_score = ["y_prob", "y_pred"],
            metric  = "classification_report",
        )

    .. note::

        VerticaPy uses simple SQL queries to compute various metrics.
        You can use the :py:meth:`verticapy.set_option` function with
        the ``sql_on`` parameter to enable SQL generation and examine
        the generated queries.

    Multi-class Classification
    ^^^^^^^^^^^^^^^^^^^^^^^^^^

    Let's create a small dataset that has:

    - true value with more than two classes
    - probability of each class
    - predicted value

    .. ipython:: python

        data = vp.vDataFrame(
            {
                "y_true": [1, 2, 0, 0, 1],
                "y_prob_0": [0.1, 0.1, 0.1, 0.1, 0.1],
                "y_prob_1": [0.8, 0.6, 0.4, 0.6, 0.2],
                "y_prob_2": [0.1, 0.3, 0.5, 0.3, 0.7],
                "y_pred": [1, 2, 0, 1, 1],
            },
        )

    Next, we import the metric:

    .. ipython:: python

        from verticapy.machine_learning.metrics import classification_report

    Now we can conveniently calculate the score:

    .. ipython:: python

        classification_report(
            y_true  = "y_true",
            y_score =[["y_prob_0","y_prob_1","y_prob_1"], "y_pred"],
            labels = [0,1,2],
            input_relation = data,
        )

    .. seealso::

        | :py:meth:`verticapy.vDataFrame.score` : Computes the input ML metric.
    """

    # Initialization

    # Case when a list of probabilities is used
    prob_list = False
    if (
        isinstance(y_score, list)
        and (len(y_score) > 0)
        and isinstance(y_score[0], list)
    ):
        new_score = "CASE"
        n = len(y_score[0])
        if isinstance(labels, NoneType):
            labels = [i for i in range(n)]
        elif len(labels) < n:
            raise ValueError(
                f"Incorrect Parameter Label. The size should be: {n}; Found: {len(labels)}"
            )
        for i, yi in enumerate(y_score[0]):
            new_score += f" WHEN '{{0}}' = '{labels[i]}' THEN {yi}"
        new_score += " END"
        y_score[0] = new_score
        prob_list = True

    # Other parameters
    return_scalar = False
    if isinstance(metrics, str):
        metrics = [metrics]
        return_scalar = True
    if estimator:
        num_classes = len(estimator.classes_)
        labels = labels if (num_classes != 2) else [estimator.classes_[1]]
    else:
        labels = [1] if isinstance(labels, NoneType) else labels
        num_classes = len(labels) + 1
    if isinstance(metrics, NoneType):
        metrics = [
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
        ]
    values = {"index": metrics}

    # Computation
    if isinstance(cutoff, NoneType) and num_classes > 2:
        if estimator:
            cm = estimator.confusion_matrix()
        elif isinstance(y_score, (NoneType, str)):
            cm = confusion_matrix(y_true, y_score, input_relation, labels=labels)
        else:
            cm = confusion_matrix(y_true, y_score[1], input_relation, labels=labels)
        all_cm_metrics = _compute_classes_tn_fn_fp_tp_from_cm(cm)
        is_multi = True
    else:
        all_cm_metrics = []
        is_multi = False
    for idx, pos_label in enumerate(labels):
        y_s = "undefined"
        y_t = "undefined"
        if is_multi:
            tn, fn, fp, tp = all_cm_metrics[idx]
            if prob_list:
                y_s = y_score[0].format(pos_label)
                y_t = f"DECODE({y_true}, '{pos_label}', 1, 0)"
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
        if len(labels) == 1:
            key = "value"
        else:
            key = pos_label
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
        if not is_multi:
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
        args = [sum(args[i] for args in all_cm_metrics) for i in range(4)]
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
