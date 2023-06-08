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
import copy
import itertools
import random
from typing import Literal, Optional

import numpy as np

from tqdm.auto import tqdm

import verticapy._config.config as conf
from verticapy._typing import (
    PlottingObject,
    PythonNumber,
    PythonScalar,
    SQLColumns,
    SQLRelation,
)
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import format_type
from verticapy._utils._sql._sys import _executeSQL


from verticapy.core.tablesample.base import TableSample
from verticapy.core.vdataframe.base import vDataFrame

from verticapy.plotting._utils import PlottingUtils

from verticapy.machine_learning.metrics import aic_score, bic_score
from verticapy.machine_learning.model_selection.model_validation import cross_validate

from verticapy.machine_learning.vertica.base import VerticaModel


@save_verticapy_logs
def randomized_features_search_cv(
    estimator: VerticaModel,
    input_relation: SQLRelation,
    X: SQLColumns,
    y: str,
    metric: str = "auto",
    cv: int = 3,
    average: Literal["binary", "micro", "macro", "weighted"] = "weighted",
    pos_label: Optional[PythonScalar] = None,
    cutoff: PythonNumber = -1,
    training_score: bool = True,
    comb_limit: int = 100,
    skip_error: bool = True,
    print_info: bool = True,
    **kwargs,
) -> TableSample:
    """
    Computes  the k-fold grid search of an estimator  using
    different feature combinations. It can be used to find
    the set of variables that will optimize the model.

    Parameters
    ----------
    estimator: VerticaModel
        Vertica estimator with a fit method.
    input_relation: SQLRelation
        Relation used to train the model.
    X: SQLColumns
        List of the predictor columns.
    y: str
        Response Column.
    metric: str, optional
        Metric used for the model evaluation.
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
        The main class to be  considered as positive
        (classification only).
    cutoff: float, optional
        The  model   cutoff  (classification  only).
    training_score: bool, optional
        If set to True,  the training score  is
        computed   with    the   validation   score.
    comb_limit: int, optional
        Maximum number of features combinations used
        to train the model.
    skip_error: bool, optional
        If set to True and an error occurs, the error
        is displayed but not raised.
    print_info: bool, optional
        If set to True, prints the model information
        at each step.

    Returns
    -------
    TableSample
        result of the randomized features search.
    """
    X = format_type(X, dtype=list)
    if estimator._model_subcategory == "REGRESSOR" and metric == "auto":
        metric = "rmse"
    elif metric == "auto":
        metric = "logloss"
    if len(X) < 20:
        all_configuration = []
        for r in range(len(X) + 1):
            combinations_object = itertools.combinations(X, r)
            combinations_list = list(combinations_object)
            if combinations_list[0]:
                all_configuration += combinations_list
        if len(all_configuration) > comb_limit and comb_limit > 0:
            all_configuration = random.sample(all_configuration, comb_limit)
    else:
        all_configuration = []
        for k in range(max(comb_limit, 1)):
            config = sorted(random.sample(X, random.randint(1, len(X))))
            if config not in all_configuration:
                all_configuration += [config]
    if (
        conf.get_option("tqdm")
        and ("tqdm" not in kwargs or ("tqdm" in kwargs and kwargs["tqdm"]))
        and print_info
    ):
        loop = tqdm(all_configuration)
    else:
        loop = all_configuration
    data = []
    for config in loop:
        if config:
            config = list(config)
            try:
                current_cv = cross_validate(
                    estimator,
                    input_relation,
                    config,
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
                if training_score:
                    keys = list(current_cv[0].values)
                    data += [
                        (
                            config,
                            current_cv[0][keys[1]][cv],
                            current_cv[1][keys[1]][cv],
                            current_cv[0][keys[2]][cv],
                            current_cv[0][keys[1]][cv + 1],
                            current_cv[1][keys[1]][cv + 1],
                        )
                    ]
                    if print_info:
                        print(
                            f"Model: {str(estimator.__class__).split('.')[-1][:-2]}; "
                            f"Features: {config}; \033[91mTest_score: "
                            f"{current_cv[0][keys[1]][cv]}\033[0m; \033[92mTrain_score:"
                            f" {current_cv[1][keys[1]][cv]}\033[0m; \033[94mTime:"
                            f" {current_cv[0][keys[2]][cv]}\033[0m;"
                        )
                else:
                    keys = [v for v in current_cv.values]
                    data += [
                        (
                            config,
                            current_cv[keys[1]][cv],
                            current_cv[keys[2]][cv],
                            current_cv[keys[1]][cv + 1],
                        )
                    ]
                    if print_info:
                        print(
                            f"Model: {str(estimator.__class__).split('.')[-1][:-2]};"
                            f" Features: {config}; \033[91mTest_score: "
                            f"{current_cv[keys[1]][cv]}\033[0m; \033[94mTime:"
                            f"{current_cv[keys[2]][cv]}\033[0m;"
                        )
            except Exception as e:
                if skip_error and skip_error != "no_print":
                    print(e)
                elif not skip_error:
                    raise e
    if not data:
        if training_score:
            return TableSample(
                {
                    "parameters": [],
                    "avg_score": [],
                    "avg_train_score": [],
                    "avg_time": [],
                    "score_std": [],
                    "score_train_std": [],
                }
            )
        else:
            return TableSample(
                {
                    "parameters": [],
                    "avg_score": [],
                    "avg_time": [],
                    "score_std": [],
                }
            )
    reverse = True
    if metric in [
        "logloss",
        "max",
        "mae",
        "median",
        "mse",
        "msle",
        "rmse",
        "aic",
        "bic",
        "auto",
    ]:
        reverse = False
    data.sort(key=lambda tup: tup[1], reverse=reverse)
    if training_score:
        res = TableSample(
            {
                "features": [d[0] for d in data],
                "avg_score": [d[1] for d in data],
                "avg_train_score": [d[2] for d in data],
                "avg_time": [d[3] for d in data],
                "score_std": [d[4] for d in data],
                "score_train_std": [d[5] for d in data],
            }
        )
        if print_info and (
            "final_print" not in kwargs or kwargs["final_print"] != "no_print"
        ):
            print("\033[1mRandomized Features Search Selected Model\033[0m")
            print(
                f"{str(estimator.__class__).split('.')[-1][:-2]}; Features:"
                f" {res['features'][0]}; \033[91mTest_score: "
                f"{res['avg_score'][0]}\033[0m; \033[92mTrain_score: "
                f"{res['avg_train_score'][0]}\033[0m; \033[94mTime: "
                f"{res['avg_time'][0]}\033[0m;"
            )
    else:
        res = TableSample(
            {
                "features": [d[0] for d in data],
                "avg_score": [d[1] for d in data],
                "avg_time": [d[2] for d in data],
                "score_std": [d[3] for d in data],
            }
        )
        if print_info and (
            "final_print" not in kwargs or kwargs["final_print"] != "no_print"
        ):
            print("\033[1mRandomized Features Search Selected Model\033[0m")
            print(
                f"{str(estimator.__class__).split('.')[-1][:-2]}; Features:"
                f" {res['features'][0]}; \033[91mTest_score: "
                f"{res['avg_score'][0]}\033[0m; \033[94mTime: "
                f"{res['avg_time'][0]}\033[0m;"
            )
    return res


@save_verticapy_logs
def stepwise(
    estimator: VerticaModel,
    input_relation: SQLRelation,
    X: SQLColumns,
    y: str,
    criterion: Literal["aic", "bic"] = "bic",
    direction: Literal["forward", "backward"] = "backward",
    max_steps: int = 100,
    criterion_threshold: int = 3,
    drop_final_estimator: bool = True,
    x_order: Literal["pearson", "spearman", "random", "none", None] = "pearson",
    print_info: bool = True,
    show: bool = True,
    chart: Optional[PlottingObject] = None,
    **style_kwargs,
) -> TableSample:
    """
    Uses the Stepwise algorithm to find the most suitable
    number of features when fitting the estimator.

    Parameters
    ----------
    estimator: VerticaModel
        Vertica estimator with a fit method. It must be a
        Binary Classifier or a Regressor.
    input_relation: SQLRelation
        Relation used to train the model.
    X: SQLColumns
        List of the predictor columns.
    y: str
        Response Column.
    criterion: str, optional
        Criterion used to evaluate the model.
            aic : Akaike’s Information Criterion
            bic : Bayesian Information Criterion
    direction: str, optional
        Method for starting the stepwise search, either
        'backward' or 'forward'.
    max_steps: int, optional
        The maximum number of steps to be considered.
    criterion_threshold: int, optional
        Threshold used when comparing the models criterions.
        If the difference is less than the threshold, then
        the current 'best' model is changed.
    drop_final_estimator: bool, optional
        If set to True, the final estimator is dropped.
    x_order: str, optional
        Method for preprocessing X before using the stepwise
        algorithm.
            pearson  : X  is ordered based on the  Pearson's
                       correlation coefficient.
            spearman : X is ordered  based on the Spearman's
                       correlation coefficient.
            random   : Shuffles the vector X before applying
                       the stepwise algorithm.
            none     : Does  not  change  the  order  of  X.
    print_info: bool, optional
        If set to True, prints the model information at each
        step.
    show: bool, optional
        If  set to True, the stepwise graphic is drawn.
    chart: PlottingObject, optional
        The chart object to plot on.
    **style_kwargs
        Any  optional  parameter  to pass to the  Plotting
        functions.

    Returns
    -------
    TableSample
        result of the stepwise.
    """
    X = format_type(X, dtype=list)
    assert len(X) >= 1, ValueError("Vector X must have at least one element.")
    if not conf.get_option("overwrite_model"):
        estimator._is_already_stored(raise_error=True)
    avg = _executeSQL(
        f"SELECT /*+LABEL('stepwise')*/ AVG({y}) FROM {input_relation}",
        method="fetchfirstelem",
        print_time_sql=False,
    )
    fun = aic_score if criterion == "aic" else bic_score
    if x_order == "random":
        random.shuffle(X)
    elif x_order in ("spearman", "pearson"):
        if isinstance(input_relation, str):
            vdf = vDataFrame(input_relation)
        else:
            vdf = input_relation
        X = [
            elem
            for elem in vdf.corr(method=x_order, focus=y, columns=X, show=False)[
                "index"
            ]
        ]
        if direction == "backward":
            X.reverse()
    if print_info:
        print("\033[1m\033[4mStarting Stepwise\033[0m\033[0m")
    if conf.get_option("tqdm") and print_info:
        loop = tqdm(range(len(X)))
    else:
        loop = range(len(X))
    model_id, res, current_step = 0, [], 0
    if direction == "backward":
        estimator.drop()
        estimator.fit(input_relation, X, y)
        current_score = estimator.score(metric=criterion)
        res += [(copy.deepcopy(X), current_score, None, None, 0, None)]
        X_current = copy.deepcopy(X)
        for idx in loop:
            if print_info and idx == 0:
                print(
                    f"\033[1m[Model 0]\033[0m \033[92m{criterion}: "
                    f"{current_score}\033[0m; Variables: {X_current}"
                )
            if current_step >= max_steps:
                break
            X_test = copy.deepcopy(X_current)
            X_test.remove(X[idx])
            if len(X_test) != 0:
                estimator.drop()
                estimator.fit(input_relation, X_test, y)
                test_score = estimator.score(metric=criterion)
            else:
                test_score = fun(y, str(avg), input_relation, 0)
            score_diff = test_score - current_score
            if test_score - current_score < criterion_threshold:
                sign = "-"
                model_id += 1
                current_score = test_score
                X_current.remove(X[idx])
                if print_info:
                    print(
                        f"\033[1m[Model {model_id}]\033[0m \033[92m{criterion}: "
                        f"{test_score}\033[0m; \033[91m(-) Variable: {X[idx]}\033[0m"
                    )
            else:
                sign = "+"
            res += [(X_test, test_score, sign, X[idx], idx + 1, score_diff)]
            current_step += 1
    else:
        current_score = fun(y, str(avg), input_relation, 0)
        res += [([], current_score, None, None, 0, None)]
        X_current = []
        for idx in loop:
            if print_info and idx == 0:
                print(
                    f"\033[1m[Model 0]\033[0m \033[92m{criterion}: "
                    f"{current_score}\033[0m; Variables: {X_current}"
                )
            if current_step >= max_steps:
                break
            X_test = copy.deepcopy(X_current) + [X[idx]]
            estimator.drop()
            estimator.fit(input_relation, X_test, y)
            test_score = estimator.score(metric=criterion)
            score_diff = current_score - test_score
            if current_score - test_score > criterion_threshold:
                sign = "+"
                model_id += 1
                current_score = test_score
                X_current += [X[idx]]
                if print_info:
                    print(
                        f"\033[1m[Model {model_id}]\033[0m \033[92m{criterion}:"
                        f" {test_score}\033[0m; \033[91m(+) Variable: {X[idx]}\033[0m"
                    )
            else:
                sign = "-"
            res += [(X_test, test_score, sign, X[idx], idx + 1, score_diff)]
            current_step += 1
    if print_info:
        print(f"\033[1m\033[4mSelected Model\033[0m\033[0m\n")
        print(
            f"\033[1m[Model {model_id}]\033[0m \033[92m{criterion}:"
            f" {current_score}\033[0m; Variables: {X_current}"
        )
    features = [x[0] for x in res]
    for idx, x in enumerate(features):
        features[idx] = [item.replace('"', "") for item in x]
    importance = [x[5] if (x[5]) and x[5] > 0 else 0 for x in res]
    sum_importance = sum(importance)
    if sum_importance == 0:
        importance = [100.0 / len(importance) for x in importance]
    else:
        importance = [100 * x / sum(importance) for x in importance]
    res = TableSample(
        {
            "index": [x[4] for x in res],
            "features": features,
            criterion: [x[1] for x in res],
            "change": [x[2] for x in res],
            "variable": [x[3] for x in res],
            "importance": importance,
        }
    )
    if drop_final_estimator:
        estimator.drop()
    res.best_list_ = X_current
    if show:
        vpy_plt, kwargs = PlottingUtils().get_plotting_lib(
            class_name="StepwisePlot",
            chart=chart,
            style_kwargs=style_kwargs,
        )
        data = {
            "x": np.array([len(x) for x in res["features"]]).astype(int),
            "y": np.array(res[criterion]).astype(float),
            "s": np.array(importance),
            "c": np.array(res["variable"]),
            "sign": np.array(res["change"]),
        }
        layout = {
            "in_variables": res["features"][0],
            "out_variables": X_current,
            "x_label": "n_features",
            "y_label": criterion,
            "z_label": "importance",
            "direction": direction,
        }
        res.step_wise_ = vpy_plt.StepwisePlot(data=data, layout=layout).draw(**kwargs)
        data = {
            "importance": np.array(importance),
        }
        layout = {"columns": copy.deepcopy(res["variable"])}
        res.importance_ = vpy_plt.ImportanceBarChart(data=data, layout=layout).draw(
            **kwargs
        )
    return res
