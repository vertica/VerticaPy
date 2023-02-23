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
import random, statistics, time
from collections.abc import Iterable
from typing import Literal, Union
from tqdm.auto import tqdm

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from verticapy._config.colors import get_colors
from verticapy._config.config import ISNOTEBOOK, _options
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._sys import _executeSQL
from verticapy._utils._sql._vertica_version import check_minimum_version
from verticapy.errors import ParameterError

from verticapy.core.tablesample.base import TableSample
from verticapy.core.vdataframe.base import vDataFrame

from verticapy.machine_learning._utils import compute_area

from verticapy.plotting._matplotlib.base import updated_dict
from verticapy.plotting._matplotlib.timeseries import range_curve


def _compute_function_metrics(
    y_true: str,
    y_score: str,
    input_relation: Union[str, vDataFrame],
    pos_label: Union[int, float, str] = 1,
    nbins: int = 30,
    fun_sql_name: str = "",
):
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


@save_verticapy_logs
def cross_validate(
    estimator,
    input_relation: Union[str, vDataFrame],
    X: Union[str, list],
    y: str,
    metric: Union[str, list] = "all",
    cv: int = 3,
    pos_label: Union[str, int, float] = None,
    cutoff: Union[int, float] = -1,
    show_time: bool = True,
    training_score: bool = False,
    **kwargs,
):
    """
Computes the K-Fold cross validation of an estimator.

Parameters
----------
estimator: object
	Vertica estimator with a fit method.
input_relation: str / vDataFrame
	Relation to use to train the model.
X: str / list
	List of the predictor columns.
y: str
	Response Column.
metric: str/list, optional
    Metric used to do the model evaluation. It can also be a list of metrics.
        all: The model will compute all the possible metrics.
    For Classification:
        accuracy    : Accuracy
        auc         : Area Under the Curve (ROC)
        best_cutoff : Cutoff which optimised the ROC Curve prediction.
        bm          : Informedness = tpr + tnr - 1
        csi         : Critical Success Index = tp / (tp + fn + fp)
        f1          : F1 Score 
        logloss     : Log Loss
        mcc         : Matthews Correlation Coefficient 
        mk          : Markedness = ppv + npv - 1
        npv         : Negative Predictive Value = tn / (tn + fn)
        prc_auc     : Area Under the Curve (PRC)
        precision   : Precision = tp / (tp + fp)
        recall      : Recall = tp / (tp + fn)
        specificity : Specificity = tn / (tn + fp)
    For Regression:
        aic    : Akaike’s information criterion
        bic    : Bayesian information criterion
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
pos_label: int/float/str, optional
	The main class to be considered as positive (classification only).
cutoff: int / float, optional
	The model cutoff (classification only).
show_time: bool, optional
    If set to True, the time and the average time will be added to the report.
training_score: bool, optional
    If set to True, the training score will be computed with the validation score.

Returns
-------
TableSample
 	An object containing the result. For more information, see
 	utilities.TableSample.
	"""
    if isinstance(X, str):
        X = [X]
    if isinstance(input_relation, str):
        input_relation = vDataFrame(input_relation)
    if cv < 2:
        raise ParameterError("Cross Validation is only possible with at least 2 folds")
    if estimator._model_subcategory == "REGRESSOR":
        all_metrics = [
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
    elif estimator._model_subcategory == "CLASSIFIER":
        all_metrics = [
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
    else:
        raise Exception(
            "Cross Validation is only possible for Regressors and Classifiers"
        )
    if metric == "all":
        final_metrics = all_metrics
    elif isinstance(metric, str):
        final_metrics = [metric]
    else:
        final_metrics = metric
    result = {"index": final_metrics}
    if training_score:
        result_train = {"index": final_metrics}
    total_time = []
    if _options["tqdm"] and (
        "tqdm" not in kwargs or ("tqdm" in kwargs and kwargs["tqdm"])
    ):
        loop = tqdm(range(cv))
    else:
        loop = range(cv)
    for i in loop:
        try:
            estimator.drop()
        except:
            pass
        random_state = _options["random_state"]
        random_state = (
            random.randint(-10e6, 10e6) if not (random_state) else random_state + i
        )
        train, test = input_relation.train_test_split(
            test_size=float(1 / cv), order_by=[X[0]], random_state=random_state
        )
        start_time = time.time()
        estimator.fit(
            train, X, y, test,
        )
        total_time += [time.time() - start_time]
        if estimator._model_subcategory == "REGRESSOR":
            if metric == "all":
                result[f"{i + 1}-fold"] = estimator.regression_report().values["value"]
                if training_score:
                    estimator.test_relation = estimator.input_relation
                    result_train[
                        f"{i + 1}-fold"
                    ] = estimator.regression_report().values["value"]
            elif isinstance(metric, str):
                result[f"{i + 1}-fold"] = [estimator.score(metric)]
                if training_score:
                    estimator.test_relation = estimator.input_relation
                    result_train[f"{i + 1}-fold"] = [estimator.score(metric)]
            else:
                result[f"{i + 1}-fold"] = [estimator.score(m) for m in metric]
                if training_score:
                    estimator.test_relation = estimator.input_relation
                    result_train[f"{i + 1}-fold"] = [estimator.score(m) for m in metric]
        else:
            if (len(estimator.classes_) > 2) and (pos_label not in estimator.classes_):
                raise ParameterError(
                    "'pos_label' must be in the estimator classes, "
                    "it must be the main class to study for the Cross Validation"
                )
            elif (len(estimator.classes_) == 2) and (
                pos_label not in estimator.classes_
            ):
                pos_label = estimator.classes_[1]
            try:
                if metric == "all":
                    result[f"{i + 1}-fold"] = estimator.classification_report(
                        labels=[pos_label], cutoff=cutoff
                    ).values["value"][0:-1]
                    if training_score:
                        estimator.test_relation = estimator.input_relation
                        result_train[f"{i + 1}-fold"] = estimator.classification_report(
                            labels=[pos_label], cutoff=cutoff
                        ).values["value"][0:-1]

                elif isinstance(metric, str):
                    result[f"{i + 1}-fold"] = [
                        estimator.score(metric, pos_label=pos_label, cutoff=cutoff)
                    ]
                    if training_score:
                        estimator.test_relation = estimator.input_relation
                        result_train[f"{i + 1}-fold"] = [
                            estimator.score(metric, pos_label=pos_label, cutoff=cutoff)
                        ]
                else:
                    result[f"{i + 1}-fold"] = [
                        estimator.score(m, pos_label=pos_label, cutoff=cutoff)
                        for m in metric
                    ]
                    if training_score:
                        estimator.test_relation = estimator.input_relation
                        result_train[f"{i + 1}-fold"] = [
                            estimator.score(m, pos_label=pos_label, cutoff=cutoff)
                            for m in metric
                        ]
            except:
                if metric == "all":
                    result[f"{i + 1}-fold"] = estimator.classification_report(
                        cutoff=cutoff
                    ).values["value"][0:-1]
                    if training_score:
                        estimator.test_relation = estimator.input_relation
                        result_train[f"{i + 1}-fold"] = estimator.classification_report(
                            cutoff=cutoff
                        ).values["value"][0:-1]
                elif isinstance(metric, str):
                    result[f"{i + 1}-fold"] = [estimator.score(metric, cutoff=cutoff)]
                    if training_score:
                        estimator.test_relation = estimator.input_relation
                        result_train[f"{i + 1}-fold"] = [
                            estimator.score(metric, cutoff=cutoff)
                        ]
                else:
                    result[f"{i + 1}-fold"] = [
                        estimator.score(m, cutoff=cutoff) for m in metric
                    ]
                    if training_score:
                        estimator.test_relation = estimator.input_relation
                        result_train[f"{i + 1}-fold"] = [
                            estimator.score(m, cutoff=cutoff) for m in metric
                        ]
        try:
            estimator.drop()
        except:
            pass
    n = len(final_metrics)
    total = [[] for item in range(n)]
    for i in range(cv):
        for k in range(n):
            total[k] += [result[f"{i + 1}-fold"][k]]
    if training_score:
        total_train = [[] for item in range(n)]
        for i in range(cv):
            for k in range(n):
                total_train[k] += [result_train[f"{i + 1}-fold"][k]]
    result["avg"], result["std"] = [], []
    if training_score:
        result_train["avg"], result_train["std"] = [], []
    for item in total:
        result["avg"] += [statistics.mean([float(elem) for elem in item])]
        result["std"] += [statistics.stdev([float(elem) for elem in item])]
    if training_score:
        for item in total_train:
            result_train["avg"] += [statistics.mean([float(elem) for elem in item])]
            result_train["std"] += [statistics.stdev([float(elem) for elem in item])]
    total_time += [
        statistics.mean([float(elem) for elem in total_time]),
        statistics.stdev([float(elem) for elem in total_time]),
    ]
    result = TableSample(values=result).transpose()
    if show_time:
        result.values["time"] = total_time
    if training_score:
        result_train = TableSample(values=result_train).transpose()
        if show_time:
            result_train.values["time"] = total_time
    if training_score:
        return result, result_train
    else:
        return result


@save_verticapy_logs
def learning_curve(
    estimator,
    input_relation: Union[str, vDataFrame],
    X: Union[str, list],
    y: str,
    sizes: list = [0.1, 0.33, 0.55, 0.78, 1.0],
    method: Literal["efficiency", "performance", "scalability"] = "efficiency",
    metric: str = "auto",
    cv: int = 3,
    pos_label: Union[int, float, str] = None,
    cutoff: Union[int, float] = -1,
    std_coeff: Union[int, float] = 1,
    ax=None,
    **style_kwds,
):
    """
Draws the learning curve.

Parameters
----------
estimator: object
    Vertica estimator with a fit method.
input_relation: str/vDataFrame
    Relation to use to train the model.
X: str / list
    List of the predictor columns.
y: str
    Response Column.
sizes: list, optional
    Different sizes of the dataset used to train the model. Multiple models
    will be trained using the different sizes.
method: str, optional
    Method used to plot the curve.
        efficiency  : draws train/test score vs sample size.
        performance : draws score vs time.
        scalability : draws time vs sample size.
metric: str, optional
    Metric used to do the model evaluation.
        auto: logloss for classification & rmse for regression.
    For Classification:
        accuracy    : Accuracy
        auc         : Area Under the Curve (ROC)
        bm          : Informedness = tpr + tnr - 1
        csi         : Critical Success Index = tp / (tp + fn + fp)
        f1          : F1 Score 
        logloss     : Log Loss
        mcc         : Matthews Correlation Coefficient 
        mk          : Markedness = ppv + npv - 1
        npv         : Negative Predictive Value = tn / (tn + fn)
        prc_auc     : Area Under the Curve (PRC)
        precision   : Precision = tp / (tp + fp)
        recall      : Recall = tp / (tp + fn)
        specificity : Specificity = tn / (tn + fp)
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
pos_label: int/float/str, optional
    The main class to be considered as positive (classification only).
cutoff: int / float, optional
    The model cutoff (classification only).
std_coeff: int / float, optional
    Value of the standard deviation coefficient used to compute the area plot 
    around each score.
ax: Matplotlib axes object, optional
    The axes to plot on.
**style_kwds
    Any optional parameter to pass to the Matplotlib functions.

Returns
-------
TableSample
    An object containing the result. For more information, see
    utilities.TableSample.
    """
    for s in sizes:
        assert 0 < s <= 1, ParameterError("Each size must be in ]0,1].")
    if estimator._model_subcategory == "REGRESSOR" and metric == "auto":
        metric = "rmse"
    elif metric == "auto":
        metric = "logloss"
    if isinstance(input_relation, str):
        input_relation = vDataFrame(input_relation)
    lc_result_final = []
    sizes = sorted(set(sizes))
    if _options["tqdm"]:
        loop = tqdm(sizes)
    else:
        loop = sizes
    for s in loop:
        relation = input_relation.sample(x=s)
        lc_result = cross_validate(
            estimator,
            relation,
            X,
            y,
            metric,
            cv,
            pos_label,
            cutoff,
            True,
            True,
            tqdm=False,
        )
        lc_result_final += [
            (
                relation.shape()[0],
                lc_result[0][metric][cv],
                lc_result[0][metric][cv + 1],
                lc_result[1][metric][cv],
                lc_result[1][metric][cv + 1],
                lc_result[0]["time"][cv],
                lc_result[0]["time"][cv + 1],
            )
        ]
    if method in ("efficiency", "scalability"):
        lc_result_final.sort(key=lambda tup: tup[0])
    else:
        lc_result_final.sort(key=lambda tup: tup[5])
    result = TableSample(
        {
            "n": [elem[0] for elem in lc_result_final],
            metric: [elem[1] for elem in lc_result_final],
            metric + "_std": [elem[2] for elem in lc_result_final],
            metric + "_train": [elem[3] for elem in lc_result_final],
            metric + "_train_std": [elem[4] for elem in lc_result_final],
            "time": [elem[5] for elem in lc_result_final],
            "time_std": [elem[6] for elem in lc_result_final],
        }
    )
    if method == "efficiency":
        X = result["n"]
        Y = [
            [
                [
                    result[metric][i] - std_coeff * result[metric + "_std"][i]
                    for i in range(len(sizes))
                ],
                result[metric],
                [
                    result[metric][i] + std_coeff * result[metric + "_std"][i]
                    for i in range(len(sizes))
                ],
            ],
            [
                [
                    result[metric + "_train"][i]
                    - std_coeff * result[metric + "_train_std"][i]
                    for i in range(len(sizes))
                ],
                result[metric + "_train"],
                [
                    result[metric + "_train"][i]
                    + std_coeff * result[metric + "_train_std"][i]
                    for i in range(len(sizes))
                ],
            ],
        ]
        x_label = "n"
        y_label = metric
        labels = [
            "test",
            "train",
        ]
    elif method == "performance":
        X = result["time"]
        Y = [
            [
                [
                    result[metric][i] - std_coeff * result[metric + "_std"][i]
                    for i in range(len(sizes))
                ],
                result[metric],
                [
                    result[metric][i] + std_coeff * result[metric + "_std"][i]
                    for i in range(len(sizes))
                ],
            ],
        ]
        x_label = "time"
        y_label = metric
        labels = []
    else:
        X = result["n"]
        Y = [
            [
                [
                    result["time"][i] - std_coeff * result["time_std"][i]
                    for i in range(len(sizes))
                ],
                result["time"],
                [
                    result["time"][i] + std_coeff * result["time_std"][i]
                    for i in range(len(sizes))
                ],
            ],
        ]
        x_label = "n"
        y_label = "time"
        labels = []
    range_curve(X, Y, x_label, y_label, ax, labels, **style_kwds)
    return result


@check_minimum_version
@save_verticapy_logs
def lift_chart(
    y_true: str,
    y_score: str,
    input_relation: Union[str, vDataFrame],
    pos_label: Union[str, int, float] = 1,
    nbins: int = 30,
    ax=None,
    **style_kwds,
):
    """
Draws the Lift Chart.

Parameters
----------
y_true: str
    Response column.
y_score: str
    Prediction Probability.
input_relation: str / vDataFrame
    Relation to use for scoring. This relation can be a view, table, or a 
    customized relation (if an alias is used at the end of the relation). 
    For example: (SELECT ... FROM ...) x
pos_label: int / float / str, optional
    To compute the Lift Chart, one of the response column classes must be the
    positive one. The parameter 'pos_label' represents this class.
nbins: int, optional
    An integer value that determines the number of decision boundaries. Decision 
    boundaries are set at equally-spaced intervals between 0 and 1, inclusive.
ax: Matplotlib axes object, optional
    The axes to plot on.
**style_kwds
    Any optional parameter to pass to the Matplotlib functions.

Returns
-------
TableSample
    An object containing the result. For more information, see
    utilities.TableSample.
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
        if ISNOTEBOOK:
            fig.set_size_inches(8, 6)
    ax.set_xlabel("Cumulative Data Fraction")
    max_value = max([0 if elem != elem else elem for elem in lift])
    lift = [max_value if elem != elem else elem for elem in lift]
    param1 = {"color": get_colors()[0]}
    ax.plot(decision_boundary, lift, **updated_dict(param1, style_kwds, 0))
    param2 = {"color": get_colors()[1]}
    ax.plot(
        decision_boundary,
        positive_prediction_ratio,
        **updated_dict(param2, style_kwds, 1),
    )
    color1, color2 = get_colors(style_kwds, 0), get_colors(style_kwds, 1)
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
    input_relation: Union[str, vDataFrame],
    pos_label: Union[str, int, float] = 1,
    nbins: int = 30,
    auc_prc: bool = False,
    ax=None,
    **style_kwds,
):
    """
Draws the PRC Curve.

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
    To compute the PRC Curve, one of the response column classes must be the
    positive one. The parameter 'pos_label' represents this class.
nbins: int, optional
    An integer value that determines the number of decision boundaries. Decision 
    boundaries are set at equally-spaced intervals between 0 and 1, inclusive.
auc_prc: bool, optional
    If set to True, the function will return the PRC AUC without drawing the 
    curve.
ax: Matplotlib axes object, optional
    The axes to plot on.
**style_kwds
    Any optional parameter to pass to the Matplotlib functions.

Returns
-------
TableSample
    An object containing the result. For more information, see
    utilities.TableSample.
    """
    threshold, recall, precision = _compute_function_metrics(
        y_true=y_true,
        y_score=y_score,
        input_relation=input_relation,
        pos_label=pos_label,
        nbins=nbins,
        fun_sql_name="prc",
    )
    auc = compute_area(precision, recall)
    if auc_prc:
        return auc
    if not (ax):
        fig, ax = plt.subplots()
        if ISNOTEBOOK:
            fig.set_size_inches(8, 6)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    param = {"color": get_colors(style_kwds, 0)}
    ax.plot(recall, precision, **updated_dict(param, style_kwds))
    ax.fill_between(
        recall,
        [0 for item in recall],
        precision,
        facecolor=get_colors(style_kwds, 0),
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
    input_relation: Union[str, vDataFrame],
    pos_label: Union[int, float, str] = 1,
    nbins: int = 30,
    auc_roc: bool = False,
    best_threshold: bool = False,
    cutoff_curve: bool = False,
    ax=None,
    **style_kwds,
):
    """
Draws the ROC Curve.

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
    To compute the PRC Curve, one of the response column classes must be the
    positive one. The parameter 'pos_label' represents this class.
nbins: int, optional
    An integer value that determines the number of decision boundaries. Decision 
    boundaries are set at equally-spaced intervals between 0 and 1, inclusive.
auc_roc: bool, optional
    If set to true, the function will return the ROC AUC without drawing the 
    curve.
best_threshold: bool, optional
    If set to True, the function will return the best threshold without drawing 
    the curve. The best threshold is the threshold of the point which is the 
    farest from the random line.
cutoff_curve: bool, optional
    If set to True, the Cutoff curve will be drawn.
ax: Matplotlib axes object, optional
    The axes to plot on.
**style_kwds
    Any optional parameter to pass to the Matplotlib functions.

Returns
-------
TableSample
    An object containing the result. For more information, see
    utilities.TableSample.
    """
    threshold, false_positive, true_positive = _compute_function_metrics(
        y_true=y_true,
        y_score=y_score,
        input_relation=input_relation,
        pos_label=pos_label,
        nbins=nbins,
        fun_sql_name="roc",
    )
    auc = compute_area(true_positive, false_positive)
    if auc_roc:
        return auc
    if best_threshold:
        l = [abs(y - x) for x, y in zip(false_positive, true_positive)]
        best_threshold_arg = max(zip(l, range(len(l))))[1]
        best = max(threshold[best_threshold_arg], 0.001)
        best = min(best, 0.999)
        return best
    if not (ax):
        fig, ax = plt.subplots()
        if ISNOTEBOOK:
            fig.set_size_inches(8, 6)
    color1, color2 = get_colors(style_kwds, 0), get_colors(style_kwds, 1)
    if color1 == color2:
        color2 = get_colors()[1]
    if cutoff_curve:
        ax.plot(
            threshold,
            [1 - item for item in false_positive],
            label="Specificity",
            **updated_dict({"color": get_colors()[0]}, style_kwds),
        )
        ax.plot(
            threshold,
            true_positive,
            label="Sensitivity",
            **updated_dict({"color": get_colors()[1]}, style_kwds),
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
            **updated_dict({"color": get_colors()[0]}, style_kwds),
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
