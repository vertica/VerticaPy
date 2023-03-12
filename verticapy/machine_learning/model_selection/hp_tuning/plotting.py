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
from typing import Optional, Union
from collections.abc import Iterable
import numpy as np

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from verticapy._config.colors import get_colors
from verticapy._typing import PythonScalar, SQLColumns, SQLRelation
from verticapy._utils._sql._collect import save_verticapy_logs

from verticapy.core.tablesample.base import TableSample
from verticapy.core.vdataframe.base import vDataFrame

from verticapy.plotting.base import PlottingBase
import verticapy.plotting._matplotlib as vpy_matplotlib_plt

from verticapy.machine_learning.model_selection.hp_tuning.cv import grid_search_cv

from verticapy.machine_learning.vertica.base import VerticaModel

"""
Tracking Over-fitting.
"""


@save_verticapy_logs
def validation_curve(
    estimator: VerticaModel,
    param_name: str,
    param_range: list,
    input_relation: SQLRelation,
    X: SQLColumns,
    y: str,
    metric: str = "auto",
    cv: int = 3,
    pos_label: PythonScalar = None,
    cutoff: float = -1,
    std_coeff: float = 1,
    ax: Optional[Axes] = None,
    **style_kwargs,
) -> TableSample:
    """
    Draws the validation curve.

    Parameters
    ----------
    estimator: VerticaModel
        Vertica estimator with a fit method.
    param_name: str
        Parameter name.
    param_range: list
        Parameter Range.
    input_relation: SQLRelation
        Relation to use to train the model.
    X: SQLColumns
        List of the predictor columns.
    y: str
        Response Column.
    metric: str, optional
        Metric used to do the model evaluation.
            auto: logloss for classification & rmse for 
                  regression.
        For Classification:
            accuracy    : Accuracy
            auc         : Area Under the Curve (ROC)
            bm          : Informedness 
                          = tpr + tnr - 1
            csi         : Critical Success Index 
                          = tp / (tp + fn + fp)
            f1          : F1 Score 
            logloss     : Log Loss
            mcc         : Matthews Correlation Coefficient 
            mk          : Markedness 
                          = ppv + npv - 1
            npv         : Negative Predictive Value 
                          = tn / (tn + fn)
            prc_auc     : Area Under the Curve (PRC)
            precision   : Precision 
                          = tp / (tp + fp)
            recall      : Recall 
                          = tp / (tp + fn)
            specificity : Specificity 
                          = tn / (tn + fp)
        For Regression:
            max    : Max error
            mae    : Mean absolute error
            median : Median absolute error
            mse    : Mean squared error
            msle   : Mean squared log error
            r2     : R-squared coefficient
            r2a    : R2 adjusted
            rmse   : Root-mean-squared error
            var    : Explained variance
    cv: int, optional
        Number of folds.
    pos_label: PythonScalar, optional
        The main class to be considered as positive 
        (classification only).
    cutoff: float, optional
        The  model  cutoff  (classification  only).
    std_coeff: float, optional
        Value of the standard deviation coefficient 
        used  to compute the area plot around  each 
        score.
    ax: Axes, optional
        The axes to plot on.
    **style_kwargs
        Any  optional  parameter  to  pass  to  the 
        Matplotlib functions.

    Returns
    -------
    TableSample
        training_score_lower, training_score, 
        training_score_upper, test_score_lower,
        test_score, test_score_upper
    """
    if isinstance(X, str):
        X = [X]
    if not (isinstance(param_range, Iterable)) or isinstance(param_range, str):
        param_range = [param_range]
    gs_result = grid_search_cv(
        estimator,
        {param_name: param_range},
        input_relation,
        X,
        y,
        metric,
        cv,
        pos_label,
        cutoff,
        True,
        False,
        False,
    )
    gs_result_final = [
        (
            gs_result["parameters"][i][param_name],
            gs_result["avg_score"][i],
            gs_result["avg_train_score"][i],
            gs_result["score_std"][i],
            gs_result["score_train_std"][i],
        )
        for i in range(len(param_range))
    ]
    gs_result_final.sort(key=lambda tup: tup[0])
    x = np.array([s[0] for s in gs_result_final])
    Y = np.column_stack(
        (
            [s[2] - std_coeff * s[4] for s in gs_result_final],
            [s[2] for s in gs_result_final],
            [s[2] + std_coeff * s[4] for s in gs_result_final],
            [s[1] - std_coeff * s[3] for s in gs_result_final],
            [s[1] for s in gs_result_final],
            [s[1] + std_coeff * s[3] for s in gs_result_final],
        )
    )
    result = TableSample(
        {
            param_name: x,
            "training_score_lower": Y[:, 0],
            "training_score": Y[:, 1],
            "training_score_upper": Y[:, 2],
            "test_score_lower": Y[:, 3],
            "test_score": Y[:, 4],
            "test_score_upper": Y[:, 5],
        }
    )
    data = {"x": x, "Y": Y}
    layout = {"columns": ["train", "test"], "order_by": param_name}
    vpy_matplotlib_plt.RangeCurve(data=data, layout=layout).draw(
        ax=ax, y_label=metric, **style_kwargs
    )
    return result


"""
TSA - Finding ARIMA parameters.
"""


@save_verticapy_logs
def plot_acf_pacf(
    vdf: vDataFrame,
    column: str,
    ts: str,
    by: SQLColumns = [],
    p: Union[int, list] = 15,
    **style_kwargs,
) -> TableSample:
    """
    Draws the ACF and PACF Charts.

    Parameters
    ----------
    vdf: vDataFrame
        Input vDataFrame.
    column: str
        Response column.
    ts: str
        vDataColumn  used as timeline. It will be to use 
        to order the data. It can be a numerical or type 
        date   like   (date,   datetime,   timestamp...) 
        vDataColumn.
    by: list, optional
        vDataColumns used in the partition.
    p: int | list, optional
        Int  equals  to  the  maximum  number  of lag to 
        consider during the computation or  List of  the 
        different lags to include during the computation.
        p must be positive or a list of positive integers.
    **style_kwargs
        Any optional  parameter to pass to the Matplotlib 
        functions.

    Returns
    -------
    TableSample
        acf, pacf, confidence
    """
    if isinstance(by, str):
        by = [by]
    tmp_style = {}
    for elem in style_kwargs:
        if elem not in ("color", "colors"):
            tmp_style[elem] = style_kwargs[elem]
    if "color" in style_kwargs:
        color = style_kwargs["color"]
    else:
        color = get_colors()[0]
    by, column, ts = vdf._format_colnames(by, column, ts)
    acf = vdf.acf(ts=ts, column=column, by=by, p=p, show=False)
    pacf = vdf.pacf(ts=ts, column=column, by=by, p=p, show=False)
    result = TableSample(
        {
            "index": [i for i in range(0, len(acf.values["value"]))],
            "acf": acf.values["value"],
            "pacf": pacf.values["value"],
            "confidence": pacf.values["confidence"],
        }
    )
    fig = plt.figure(figsize=(10, 6))
    plt.rcParams["axes.facecolor"] = "#FCFCFC"
    ax1 = fig.add_subplot(211)
    x, y, confidence = (
        result.values["index"],
        result.values["acf"],
        result.values["confidence"],
    )
    plt.xlim(-1, x[-1] + 1)
    ax1.bar(x, y, width=0.007 * len(x), color="#444444", zorder=1, linewidth=0)
    param = {
        "s": 90,
        "marker": "o",
        "facecolors": color,
        "edgecolors": "black",
        "zorder": 2,
    }
    ax1.scatter(x, y, **PlottingBase._update_dict(param, tmp_style))
    ax1.plot(
        [-1] + x + [x[-1] + 1],
        [0 for elem in range(len(x) + 2)],
        color=color,
        zorder=0,
    )
    ax1.fill_between(x, confidence, color="#FE5016", alpha=0.1)
    ax1.fill_between(x, [-elem for elem in confidence], color="#FE5016", alpha=0.1)
    ax1.set_title("Autocorrelation")
    y = result.values["pacf"]
    ax2 = fig.add_subplot(212)
    ax2.bar(x, y, width=0.007 * len(x), color="#444444", zorder=1, linewidth=0)
    ax2.scatter(x, y, **PlottingBase._update_dict(param, tmp_style))
    ax2.plot(
        [-1] + x + [x[-1] + 1],
        [0 for elem in range(len(x) + 2)],
        color=color,
        zorder=0,
    )
    ax2.fill_between(x, confidence, color="#FE5016", alpha=0.1)
    ax2.fill_between(x, [-elem for elem in confidence], color="#FE5016", alpha=0.1)
    ax2.set_title("Partial Autocorrelation")
    plt.show()
    return result
