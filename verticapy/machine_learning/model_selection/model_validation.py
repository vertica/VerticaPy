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
import copy
import random
import time
from typing import Literal, Optional, Union

import numpy as np

from tqdm.auto import tqdm

import verticapy._config.config as conf
from verticapy._typing import (
    NoneType,
    PlottingObject,
    PythonNumber,
    PythonScalar,
    SQLColumns,
    SQLRelation,
)
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import format_type


from verticapy.core.tablesample.base import TableSample
from verticapy.core.vdataframe.base import vDataFrame

from verticapy.machine_learning.vertica.base import VerticaModel

from verticapy.plotting._utils import PlottingUtils


@save_verticapy_logs
def cross_validate(
    estimator: VerticaModel,
    input_relation: SQLRelation,
    X: SQLColumns,
    y: str,
    metrics: Union[None, str, list[str]] = None,
    cv: int = 3,
    average: Literal["binary", "micro", "macro", "weighted"] = "weighted",
    pos_label: Optional[PythonScalar] = None,
    cutoff: PythonNumber = -1,
    show_time: bool = True,
    training_score: bool = False,
    **kwargs,
) -> TableSample:
    """
    Computes the  K-Fold cross validation  of an estimator.

    Parameters
    ----------
    estimator: object
        Vertica estimator with a fit method.
    input_relation: SQLRelation
        Relation  used to train the model.
    X: SQLColumns
        List   of  the  predictor   columns.
    y: str
        Response Column.
    metrics: str / list, optional
        Metrics used to do the model evaluation. It can also
        be a list of metrics. If empty, most of the estimator
        metrics are computed.
        For Classification:
            accuracy    : Accuracy
            auc         : Area Under the Curve (ROC)
            ba          : Balanced Accuracy
                          = (tpr + tnr) / 2
            best_cutoff : Cutoff which optimised the ROC
                          Curve prediction.
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
            aic    : Akaike’s Information Criterion
            bic    : Bayesian Information Criterion
            max    : Max Error
            mae    : Mean Absolute Error
            median : Median Absolute Error
            mse    : Mean Squared Error
            msle   : Mean Squared Log Error
            qe     : quantile  error,  the quantile must be
                     included in the name. Example:
                     qe50.1% will return the quantile error
                     using q=0.501.
            r2     : R squared coefficient
            r2a    : R2 adjusted
            rmse   : Root Mean Squared Error
            var    : Explained Variance

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
    cutoff: PythonNumber, optional
        The model cutoff (classification only).
    show_time: bool, optional
        If set to True,  the  time  and the average
        time   are    added  to   the   report.
    training_score: bool, optional
        If set to True,  the training score is
        computed   with   the   validation   score.

    Returns
    -------
    TableSample
        result of the cross validation.
    """
    X = format_type(X, dtype=list)
    if isinstance(input_relation, str):
        input_relation = vDataFrame(input_relation)
    if cv < 2:
        raise ValueError("Cross Validation is only possible with at least 2 folds")
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
    if isinstance(metrics, NoneType):
        final_metrics = all_metrics
    elif isinstance(metrics, str):
        final_metrics = [metrics]
    else:
        final_metrics = copy.deepcopy(metrics)
    result = {"index": final_metrics}
    if training_score:
        result_train = {"index": final_metrics}
    total_time = []
    if conf.get_option("tqdm") and (
        "tqdm" not in kwargs or ("tqdm" in kwargs and kwargs["tqdm"])
    ):
        loop = tqdm(range(cv))
    else:
        loop = range(cv)
    for i in loop:
        estimator.drop()
        random_state = conf.get_option("random_state")
        random_state = (
            random.randint(int(-10e6), int(10e6))
            if not random_state
            else random_state + i
        )
        train, test = input_relation.train_test_split(
            test_size=float(1 / cv), order_by=[X[0]], random_state=random_state
        )
        start_time = time.time()
        estimator.fit(
            train,
            X,
            y,
            test,
            return_report=True,
        )
        total_time += [time.time() - start_time]
        fun = estimator.report
        kwargs = {"metrics": final_metrics}
        key = "value"
        if estimator._model_subcategory == "CLASSIFIER" and not (
            estimator._is_binary_classifier()
        ):
            key = f"avg_{average}"
        result[f"{i + 1}-fold"] = fun(**kwargs)[key]
        if training_score:
            estimator.test_relation = estimator.input_relation
            result_train[f"{i + 1}-fold"] = fun(**kwargs)[key]
        estimator.drop()
    n = len(final_metrics)
    total = [[] for item in range(n)]
    total_time = np.array(total_time).astype(float)
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
        result["avg"] += [np.nanmean(np.array(item).astype(float))]
        result["std"] += [np.nanstd(np.array(item).astype(float))]
    if training_score:
        for item in total_train:
            result_train["avg"] += [np.nanmean(np.array(item).astype(float))]
            result_train["std"] += [np.nanstd(np.array(item).astype(float))]

    total_time = list(total_time) + [np.nanmean(total_time), np.nanstd(total_time)]
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
    estimator: VerticaModel,
    input_relation: SQLRelation,
    X: SQLColumns,
    y: str,
    sizes: Optional[list] = None,
    method: Literal["efficiency", "performance", "scalability"] = "efficiency",
    metric: str = "auto",
    cv: int = 3,
    average: Literal["binary", "micro", "macro", "weighted"] = "weighted",
    pos_label: Optional[PythonScalar] = None,
    cutoff: PythonNumber = -1,
    std_coeff: PythonNumber = 1,
    chart: Optional[PlottingObject] = None,
    return_chart: Optional[bool] = False,
    **style_kwargs,
) -> TableSample:
    """
    Draws the learning curve.

    Parameters
    ----------
    estimator: object
        Vertica estimator with a fit method.
    input_relation: SQLRelation
        Relation  used  to train the model.
    X: SQLColumns
        List   of  the  predictor   columns.
    y: str
        Response Column.
    sizes: list, optional
        Different sizes of the dataset used
        to train the model. Multiple models
        are trained using the different
        sizes.
    method: str, optional
        Method used to plot the curve.
            efficiency  : draws train/test score
                          vs sample size.
            performance : draws score  vs  time.
            scalability : draws time  vs  sample
                          size.
    metric: str, optional
        Metric used to do the model evaluation.
            auto: logloss for classification & RMSE
                  for regression.
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
            scores   : scores  for   all  the  classes.
            weighted : weighted average of the score of
                       each class.
    pos_label: PythonScalar, optional
        The main class to be considered as positive
        (classification only).
    cutoff: PythonNumber, optional
        The  model  cutoff  (classification  only).
    std_coeff: PythonNumber, optional
        Value of the standard deviation coefficient
        used to compute the area plot around each
        score.
    chart: PlottingObject, optional
        The chart object to plot on.
    return_chart: bool, optional
        Select whether you want to get the chart as the output only.
    **style_kwargs
        Any  optional  parameter  to  pass  to  the
        Plotting functions.

    Returns
    -------
    TableSample
        result of the learning curve.
    """
    sizes = format_type(sizes, dtype=list, na_out=[0.1, 0.33, 0.55, 0.78, 1.0])
    for s in sizes:
        assert 0 < s <= 1, ValueError("Each size must be in ]0,1].")
    if estimator._model_subcategory == "REGRESSOR" and metric == "auto":
        metric = "rmse"
    elif metric == "auto":
        metric = "logloss"
    if isinstance(input_relation, str):
        input_relation = vDataFrame(input_relation)
    lc_result_final = []
    sizes = sorted(set(sizes))
    if conf.get_option("tqdm"):
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
            metrics=metric,
            cv=cv,
            average=average,
            pos_label=pos_label,
            cutoff=cutoff,
            show_time=True,
            training_score=True,
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
        x = np.array(result["n"])
        Y = np.column_stack(
            (
                [
                    result[metric][i] - std_coeff * result[metric + "_std"][i]
                    for i in range(len(sizes))
                ],
                result[metric],
                [
                    result[metric][i] + std_coeff * result[metric + "_std"][i]
                    for i in range(len(sizes))
                ],
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
            ),
        )
        order_by = "n"
        y_label = metric
        columns = [
            "test",
            "train",
        ]
    elif method == "performance":
        x = np.array(result["time"])
        Y = np.column_stack(
            (
                [
                    result[metric][i] - std_coeff * result[metric + "_std"][i]
                    for i in range(len(sizes))
                ],
                result[metric],
                [
                    result[metric][i] + std_coeff * result[metric + "_std"][i]
                    for i in range(len(sizes))
                ],
            )
        )
        order_by = "time"
        y_label = None
        columns = [metric]
    else:
        x = np.array(result["n"])
        Y = np.column_stack(
            (
                [
                    result["time"][i] - std_coeff * result["time_std"][i]
                    for i in range(len(sizes))
                ],
                result["time"],
                [
                    result["time"][i] + std_coeff * result["time_std"][i]
                    for i in range(len(sizes))
                ],
            )
        )
        order_by = "n"
        y_label = None
        columns = ["time"]
    vpy_plt, kwargs = PlottingUtils().get_plotting_lib(
        class_name="RangeCurve",
        chart=chart,
        style_kwargs=style_kwargs,
    )
    data = {"x": x, "Y": Y}
    layout = {"columns": columns, "order_by": order_by, "y_label": y_label}
    vpy_plt.RangeCurve(data=data, layout=layout).draw(**kwargs)
    if return_chart:
        return vpy_plt.RangeCurve(data=data, layout=layout).draw(**kwargs)
    return result
