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

from matplotlib.axes import Axes
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from verticapy._config.colors import get_colors
import verticapy._config.config as conf
from verticapy._typing import PythonScalar, SQLRelation
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._vertica_version import check_minimum_version

from verticapy.core.tablesample.base import TableSample

from verticapy.machine_learning.metrics.classification import (
    _compute_area,
    _compute_function_metrics,
)

from verticapy.plotting.base import PlottingBase


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
    decision_boundary.reverse()
    if not (ax):
        fig, ax = plt.subplots()
        if conf._get_import_success("jupyter"):
            fig.set_size_inches(8, 6)
    ax.set_xlabel("Cumulative Data Fraction")
    max_value = max([0 if elem != elem else elem for elem in lift])
    lift = [max_value if elem != elem else elem for elem in lift]
    param1 = {"color": get_colors()[0]}
    ax.plot(
        decision_boundary, lift, **PlottingBase._update_dict(param1, style_kwargs, 0)
    )
    param2 = {"color": get_colors()[1]}
    ax.plot(
        decision_boundary,
        positive_prediction_ratio,
        **PlottingBase._update_dict(param2, style_kwargs, 1),
    )
    color1, color2 = get_colors(style_kwargs, 0), get_colors(style_kwargs, 1)
    if color1 == color2:
        color2 = get_colors()[1]
    ax.fill_between(
        decision_boundary, positive_prediction_ratio, lift, facecolor=color1, alpha=0.2,
    )
    ax.fill_between(
        decision_boundary,
        [0 for elem in decision_boundary],
        positive_prediction_ratio,
        facecolor=color2,
        alpha=0.2,
    )
    ax.set_title("Lift Table")
    ax.set_axisbelow(True)
    ax.grid()
    color1 = mpatches.Patch(color=color1, label="Cumulative Lift")
    color2 = mpatches.Patch(color=color2, label="Cumulative Capture Rate")
    ax.legend(handles=[color1, color2], loc="center left", bbox_to_anchor=[1, 0.5])
    ax.set_xlim(0, 1)
    ax.set_ylim(0)
    return TableSample(
        values={
            "decision_boundary": decision_boundary,
            "positive_prediction_ratio": positive_prediction_ratio,
            "lift": lift,
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
    if not (ax):
        fig, ax = plt.subplots()
        if conf._get_import_success("jupyter"):
            fig.set_size_inches(8, 6)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    param = {"color": get_colors(style_kwargs, 0)}
    ax.plot(recall, precision, **PlottingBase._update_dict(param, style_kwargs))
    ax.fill_between(
        recall,
        [0 for item in recall],
        precision,
        facecolor=get_colors(style_kwargs, 0),
        alpha=0.1,
    )
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_title("PRC Curve")
    ax.text(
        0.995,
        0,
        f"AUC = {round(auc, 4) * 100}%",
        verticalalignment="bottom",
        horizontalalignment="right",
        fontsize=11.5,
    )
    ax.set_axisbelow(True)
    ax.grid()
    return TableSample(
        values={"threshold": threshold, "recall": recall, "precision": precision}
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
    if not (ax):
        fig, ax = plt.subplots()
        if conf._get_import_success("jupyter"):
            fig.set_size_inches(8, 6)
    color1, color2 = get_colors(style_kwargs, 0), get_colors(style_kwargs, 1)
    if color1 == color2:
        color2 = get_colors()[1]
    if cutoff_curve:
        ax.plot(
            threshold,
            [1 - item for item in false_positive],
            label="Specificity",
            **PlottingBase._update_dict({"color": get_colors()[0]}, style_kwargs),
        )
        ax.plot(
            threshold,
            true_positive,
            label="Sensitivity",
            **PlottingBase._update_dict({"color": get_colors()[1]}, style_kwargs),
        )
        ax.fill_between(
            threshold,
            [1 - item for item in false_positive],
            true_positive,
            facecolor="black",
            alpha=0.02,
        )
        ax.set_xlabel("Decision Boundary")
        ax.set_title("Cutoff Curve")
        ax.legend(loc="center left", bbox_to_anchor=[1, 0.5])
    else:
        ax.set_xlabel("False Positive Rate (1-Specificity)")
        ax.set_ylabel("True Positive Rate (Sensitivity)")
        ax.plot(
            false_positive,
            true_positive,
            **PlottingBase._update_dict({"color": get_colors()[0]}, style_kwargs),
        )
        ax.fill_between(
            false_positive, false_positive, true_positive, facecolor=color1, alpha=0.1,
        )
        ax.fill_between([0, 1], [0, 0], [0, 1], facecolor=color2, alpha=0.1)
        ax.plot([0, 1], [0, 1], color=color2)
        ax.set_title("ROC Curve")
        ax.text(
            0.995,
            0,
            f"AUC = {round(auc, 4) * 100}%",
            verticalalignment="bottom",
            horizontalalignment="right",
            fontsize=11.5,
        )
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_axisbelow(True)
    ax.grid()
    return TableSample(
        values={
            "threshold": threshold,
            "false_positive": false_positive,
            "true_positive": true_positive,
        }
    )
