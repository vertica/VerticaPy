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
from typing import Literal, Optional, Union
from tqdm.auto import tqdm
import numpy as np

from matplotlib.axes import Axes

import verticapy._config.config as conf
from verticapy._typing import PythonNumber, PythonScalar, SQLColumns, SQLRelation
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._sys import _executeSQL
from verticapy.errors import ParameterError

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
    metric: Union[str, list[str]] = "all",
    cv: int = 3,
    pos_label: PythonScalar = None,
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
    	Relation  to use to train the model.
    X: SQLColumns
    	List   of  the  predictor   columns.
    y: str
    	Response Column.
    metric: str / list, optional
        Metric used to do the model evaluation. It can also 
        be a list of metrics.
            all: The  model will  compute all the  possible 
                 metrics.
        For Classification:
            accuracy    : Accuracy
            auc         : Area Under the Curve (ROC)
            best_cutoff : Cutoff which optimised the ROC 
                          Curve prediction.
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
    pos_label: PythonScalar, optional
    	The main class to be considered as positive 
        (classification only).
    cutoff: PythonNumber, optional
    	The model cutoff (classification only).
    show_time: bool, optional
        If set to True,  the  time  and the average 
        time   will   be  added  to   the   report.
    training_score: bool, optional
        If set to True,  the training score will be 
        computed   with   the   validation   score.

    Returns
    -------
    TableSample
     	result of the cross validation.
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
    if conf.get_option("tqdm") and (
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
        random_state = conf.get_option("random_state")
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
    estimator: VerticaModel,
    input_relation: SQLRelation,
    X: SQLColumns,
    y: str,
    sizes: list = [0.1, 0.33, 0.55, 0.78, 1.0],
    method: Literal["efficiency", "performance", "scalability"] = "efficiency",
    metric: str = "auto",
    cv: int = 3,
    pos_label: PythonScalar = None,
    cutoff: PythonNumber = -1,
    std_coeff: PythonNumber = 1,
    ax: Optional[Axes] = None,
    **style_kwargs,
) -> TableSample:
    """
    Draws the learning curve.

    Parameters
    ----------
    estimator: object
        Vertica estimator with a fit method.
    input_relation: SQLRelation
        Relation  to use to train the model.
    X: SQLColumns
        List   of  the  predictor   columns.
    y: str
        Response Column.
    sizes: list, optional
        Different sizes of the dataset used 
        to train the model. Multiple models
        will be trained using the different 
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
            auto: logloss for classification & rmse 
                  for regression.
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
    cutoff: PythonNumber, optional
        The  model  cutoff  (classification  only).
    std_coeff: PythonNumber, optional
        Value of the standard deviation coefficient 
        used to compute the area plot 
        around each score.
    ax: Axes, optional
        [Only for MATPLOTLIB]
        The axes to plot on.
    **style_kwargs
        Any  optional  parameter  to  pass  to  the 
        Matplotlib functions.

    Returns
    -------
    TableSample
        result of the learning curve.
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
    vpy_plt, kwargs = PlottingUtils()._get_plotting_lib(
        class_name="RangeCurve",
        matplotlib_kwargs={"ax": ax,},
        style_kwargs=style_kwargs,
    )
    data = {"x": x, "Y": Y}
    layout = {"columns": columns, "order_by": order_by, "y_label": y_label}
    vpy_plt.RangeCurve(data=data, layout=layout).draw(**kwargs)
    return result
