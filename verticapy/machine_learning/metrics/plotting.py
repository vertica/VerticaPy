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
from typing import Optional
import numpy as np

from matplotlib.axes import Axes

from verticapy._typing import PythonScalar, SQLRelation
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._vertica_version import check_minimum_version

from verticapy.core.tablesample.base import TableSample

from verticapy.machine_learning.metrics.classification import (
    _compute_area,
    _compute_function_metrics,
)

from verticapy.plotting._utils import PlottingUtils


@check_minimum_version
@save_verticapy_logs
def lift_chart(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    pos_label: PythonScalar = 1,
    nbins: int = 30,
    ax: Optional[Axes] = None,
    **style_kwargs,
) -> TableSample:
    """
    Draws the Lift Chart.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction Probability.
    input_relation: SQLRelation
        Relation to use for scoring. This relation can 
        be a view, table, or a customized relation (if 
        an alias  is used at the end of the relation). 
        For example: (SELECT ... FROM ...) x
    pos_label: PythonScalar, optional
        To compute the Lift Chart, one of the response 
        column  classes must be the positive one.  The 
        parameter  'pos_label' represents this  class.
    nbins: int, optional
        An integer value that determines the number of 
        decision  boundaries.  Decision boundaries are 
        set at equally-spaced intervals  between 0 and 
        1, inclusive.
    ax: Axes, optional
        [Only for MATPLOTLIB]
        The axes to plot on.
    **style_kwargs
        Any   optional  parameter  to  pass  to   the 
        Matplotlib functions.

    Returns
    -------
    TableSample
        decision_boundary, positive_prediction_ratio, lift
    """
    decision_boundary, positive_prediction_ratio, lift = _compute_function_metrics(
        y_true=y_true,
        y_score=y_score,
        input_relation=input_relation,
        pos_label=pos_label,
        nbins=nbins,
        fun_sql_name="lift_table",
    )
    lift = np.nan_to_num(lift, nan=np.nanmax(lift))
    decision_boundary.reverse()
    vpy_plt, kwargs = PlottingUtils()._get_plotting_lib(
        class_name="LiftChart",
        matplotlib_kwargs={"ax": ax,},
        style_kwargs=style_kwargs,
    )
    data = {
        "x": np.array(decision_boundary),
        "y": np.array(positive_prediction_ratio),
        "z": np.array(lift),
    }
    layout = {
        "title": "Lift Table",
        "x_label": "Cumulative Data Fraction",
        "y_label": "Cumulative Capture Rate",
        "z_label": "Cumulative Lift",
    }
    vpy_plt.LiftChart(data=data, layout=layout).draw(**kwargs)
    return TableSample(
        values={
            "decision_boundary": decision_boundary,
            "positive_prediction_ratio": positive_prediction_ratio,
            "lift": list(lift),
        }
    )


@check_minimum_version
@save_verticapy_logs
def prc_curve(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    pos_label: PythonScalar = 1,
    nbins: int = 30,
    ax: Optional[Axes] = None,
    **style_kwargs,
) -> TableSample:
    """
    Draws the PRC Curve.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction Probability.
    input_relation: SQLRelation
        Relation to use for scoring. This relation can 
        be a view, table, or a customized relation (if 
        an alias  is used at the end of the relation). 
        For example: (SELECT ... FROM ...) x
    pos_label: PythonScalar, optional
        To compute the PRC Curve,  one of the response 
        column  classes must be the positive one.  The 
        parameter  'pos_label' represents this  class.
    nbins: int, optional
        An integer value that determines the number of 
        decision  boundaries.  Decision boundaries are 
        set at equally-spaced intervals  between 0 and 
        1, inclusive.
    ax: Axes, optional
        [Only for MATPLOTLIB]
        The axes to plot on.
    **style_kwargs
        Any   optional  parameter  to  pass  to   the 
        Matplotlib functions.

    Returns
    -------
    TableSample
        threshold, recall, precision
    """
    threshold, recall, precision = _compute_function_metrics(
        y_true=y_true,
        y_score=y_score,
        input_relation=input_relation,
        pos_label=pos_label,
        nbins=nbins,
        fun_sql_name="prc",
    )
    auc = _compute_area(precision, recall)
    vpy_plt, kwargs = PlottingUtils()._get_plotting_lib(
        class_name="PRCCurve", matplotlib_kwargs={"ax": ax,}, style_kwargs=style_kwargs,
    )
    data = {"x": np.array(recall), "y": np.array(precision), "auc": auc}
    layout = {
        "title": "PRC Curve",
        "x_label": "Recall",
        "y_label": "Precision",
    }
    vpy_plt.PRCCurve(data=data, layout=layout).draw(**kwargs)
    return TableSample(
        values={"threshold": threshold, "recall": recall, "precision": precision,}
    )


@check_minimum_version
@save_verticapy_logs
def roc_curve(
    y_true: str,
    y_score: str,
    input_relation: SQLRelation,
    pos_label: PythonScalar = 1,
    nbins: int = 30,
    cutoff_curve: bool = False,
    ax: Optional[Axes] = None,
    **style_kwargs,
) -> TableSample:
    """
    Draws the ROC Curve.

    Parameters
    ----------
    y_true: str
        Response column.
    y_score: str
        Prediction Probability.
    input_relation: SQLRelation
        Relation to use for scoring. This relation can 
        be a view, table, or a customized relation (if 
        an alias  is used at the end of the relation). 
        For example: (SELECT ... FROM ...) x
    pos_label: PythonScalar, optional
        To compute the ROC Curve,  one of the response 
        column  classes must be the positive one.  The 
        parameter  'pos_label' represents this  class.
    nbins: int, optional
        An integer value that determines the number of 
        decision  boundaries.  Decision boundaries are 
        set at equally-spaced intervals  between 0 and 
        1, inclusive.
    ax: Axes, optional
        [Only for MATPLOTLIB]
        The axes to plot on.
    **style_kwargs
        Any   optional  parameter  to  pass  to   the 
        Matplotlib functions.

    Returns
    -------
    TableSample
        threshold, false_positive, true_positive
    """
    threshold, false_positive, true_positive = _compute_function_metrics(
        y_true=y_true,
        y_score=y_score,
        input_relation=input_relation,
        pos_label=pos_label,
        nbins=nbins,
        fun_sql_name="roc",
    )
    auc = _compute_area(true_positive, false_positive)
    if cutoff_curve:
        vpy_plt, kwargs = PlottingUtils()._get_plotting_lib(
            class_name="CutoffCurve",
            matplotlib_kwargs={"ax": ax,},
            style_kwargs=style_kwargs,
        )
        data = {
            "x": np.array(threshold),
            "y": 1 - np.array(false_positive),
            "z": np.array(true_positive),
            "auc": auc,
        }
        layout = {
            "title": "Cutoff Curve",
            "x_label": "Decision Boundary",
            "y_label": "Specificity",
            "z_label": "Sensitivity",
        }
        vpy_plt.CutoffCurve(data=data, layout=layout).draw(**kwargs)
    else:
        vpy_plt, kwargs = PlottingUtils()._get_plotting_lib(
            class_name="ROCCurve",
            matplotlib_kwargs={"ax": ax,},
            style_kwargs=style_kwargs,
        )
        data = {"x": np.array(false_positive), "y": np.array(true_positive), "auc": auc}
        layout = {
            "title": "ROC Curve",
            "x_label": "False Positive Rate (1-Specificity)",
            "y_label": "True Positive Rate (Sensitivity)",
        }
        vpy_plt.ROCCurve(data=data, layout=layout).draw(**kwargs)
    return TableSample(
        values={
            "threshold": threshold,
            "false_positive": false_positive,
            "true_positive": true_positive,
        }
    )
