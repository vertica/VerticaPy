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
from typing import Literal, Optional, Union
from collections.abc import Iterable

import numpy as np

from verticapy._typing import PlottingObject, PythonScalar, SQLColumns, SQLRelation
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import format_type

from verticapy.core.tablesample.base import TableSample
from verticapy.core.vdataframe.base import vDataFrame

from verticapy.machine_learning.model_selection.hp_tuning.cv import grid_search_cv
from verticapy.machine_learning.vertica.base import VerticaModel

from verticapy.plotting._utils import PlottingUtils

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
    average: Literal["binary", "micro", "macro", "weighted"] = "weighted",
    pos_label: Optional[PythonScalar] = None,
    cutoff: float = -1,
    std_coeff: float = 1,
    chart: Optional[PlottingObject] = None,
    show: Optional[bool] = False,
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
        Relation used to train the model.
    X: SQLColumns
        List of the predictor columns.
    y: str
        Response Column.
    metric: str, optional
        Metric used to for model evaluation.
            auto: logloss for classification & rmse for
                  regression.
        For Classification:
            accuracy    : Accuracy
            auc         : Area Under the Curve (ROC)
            ba          : Balanced Accuracy
                          = (tpr + tnr) / 2
            bm          : Informedness
                          = tpr + tnr - 1
            csi         : Critical Success Index
                          = tp / (tp + fn + fp)
            f1          : F1 Score
            fdr         : False Discovery Rate = 1 - ppv
            fm          : Fowlkes–Mallows index
                          = sqrt(ppv * tpr)
            fnr         : False Negative Rate
                          = fn / (fn + tp)
            for         : False Omission Rate = 1 - npv
            fpr         : False Positive Rate
                          = fp / (fp + tn)
            logloss     : Log Loss
            lr+         : Positive Likelihood Ratio
                          = tpr / fpr
            lr-         : Negative Likelihood Ratio
                          = fnr / tnr
            dor         : Diagnostic Odds Ratio
            mcc         : Matthews Correlation Coefficient
            mk          : Markedness
                          = ppv + npv - 1
            npv         : Negative Predictive Value
                          = tn / (tn + fn)
            prc_auc     : Area Under the Curve (PRC)
            precision   : Precision
                          = tp / (tp + fp)
            pt          : Prevalence Threshold
                          = sqrt(fpr) / (sqrt(tpr) + sqrt(fpr))
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
    average: str, optional
        The method used to  compute the final score for
        multiclass-classification.
            binary   : considers one of the classes  as
                       positive  and  use  the   binary
                       confusion  matrix to compute the
                       score.
            micro    : positive  and   negative  values
                       globally.
            macro    : average  of  the  score of  each
                       class.
            weighted : weighted average of the score of
                       each class.
    pos_label: PythonScalar, optional
        The main class to be considered as positive
        (classification only).
    cutoff: float, optional
        The  model  cutoff  (classification  only).
    std_coeff: float, optional
        Value of the standard deviation coefficient
        used  to compute the area plot around  each
        score.
    chart: PlottingObject, optional
        The chart object to plot on.
    show: bool, optional
        Select whether you want to get the chart as the output only.
    **style_kwargs
        Any  optional  parameter  to  pass  to  the
        Plotting functions.

    Returns
    -------
    TableSample
        training_score_lower, training_score,
        training_score_upper, test_score_lower,
        test_score, test_score_upper
    """
    X = format_type(X, dtype=list)
    if not isinstance(param_range, Iterable) or isinstance(param_range, str):
        param_range = [param_range]
    gs_result = grid_search_cv(
        estimator,
        {param_name: param_range},
        input_relation,
        X,
        y,
        metric=metric,
        cv=cv,
        average=average,
        pos_label=pos_label,
        cutoff=cutoff,
        training_score=True,
        skip_error=False,
        print_info=False,
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
    vpy_plt, kwargs = PlottingUtils().get_plotting_lib(
        class_name="RangeCurve",
        chart=chart,
        style_kwargs=style_kwargs,
    )
    data = {"x": x, "Y": Y}
    layout = {"columns": ["train", "test"], "order_by": param_name, "y_label": metric}
    vpy_plt.RangeCurve(data=data, layout=layout).draw(**kwargs)
    if show:
        return vpy_plt.RangeCurve(data=data, layout=layout).draw(**kwargs)
    return result


"""
TSA - Finding ARIMA parameters.
"""


@save_verticapy_logs
def plot_acf_pacf(
    vdf: vDataFrame,
    column: str,
    ts: str,
    by: Optional[SQLColumns] = None,
    p: Union[int, list] = 15,
    show: bool = True,
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
        vDataColumn used as timeline to order the data.
        It can be a numerical or date-like type (date,
        datetime,   timestamp...) vDataColumn.
    by: list, optional
        vDataColumns used in the partition.
    p: int | list, optional
        Integer equal to the maximum  number  of lags to
        consider during the computation or a list of the
        different lags to include during the computation.
        p must be positive or a list of positive integers.
    show: bool, optional
        If  set to  True,  the  Plotting  object is
        returned.
    **style_kwargs
        Any optional  parameter to pass to the Plotting
        functions.

    Returns
    -------
    TableSample
        acf, pacf, confidence
    """
    by = format_type(by, dtype=list)
    by, column, ts = vdf.format_colnames(by, column, ts)
    acf = vdf.acf(ts=ts, column=column, by=by, p=p, show=False)
    pacf = vdf.pacf(ts=ts, column=column, by=by, p=p, show=False)
    index = [i for i in range(0, len(acf.values["value"]))]
    if show:
        vpy_plt, kwargs = PlottingUtils().get_plotting_lib(
            class_name="ACFPACFPlot",
            style_kwargs=style_kwargs,
        )
        data = {
            "x": np.array(index),
            "y0": np.array(acf.values["value"]),
            "y1": np.array(pacf.values["value"]),
            "z": np.array(pacf.values["confidence"]),
        }
        layout = {
            "y0_label": "Autocorrelation",
            "y1_label": "Partial Autocorrelation",
        }
        return vpy_plt.ACFPACFPlot(data=data, layout=layout).draw(**kwargs)
    return TableSample(
        {
            "index": index,
            "acf": acf.values["value"],
            "pacf": pacf.values["value"],
            "confidence": pacf.values["confidence"],
        }
    )
