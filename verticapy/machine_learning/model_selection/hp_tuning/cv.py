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

from tqdm.auto import tqdm

import verticapy._config.config as conf
from verticapy._typing import PythonNumber, PythonScalar, SQLColumns, SQLRelation
from verticapy._utils._gen import gen_tmp_name
from verticapy._utils._sql._format import format_type
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._sys import _executeSQL


from verticapy.core.tablesample.base import TableSample
from verticapy.core.vdataframe.base import vDataFrame

from verticapy.datasets.generators import gen_meshgrid, gen_dataset

from verticapy.machine_learning.model_selection.hp_tuning.param_gen import (
    gen_params_grid,
    parameter_grid,
)
from verticapy.machine_learning.model_selection.model_validation import cross_validate
import verticapy.machine_learning.vertica as vml
from verticapy.machine_learning.vertica.base import VerticaModel

from verticapy.sql.drop import drop

"""
RANDOM
"""


@save_verticapy_logs
def randomized_search_cv(
    estimator: VerticaModel,
    input_relation: SQLRelation,
    X: SQLColumns,
    y: str,
    metric: str = "auto",
    cv: int = 3,
    average: Literal["binary", "micro", "macro", "weighted"] = "weighted",
    pos_label: Optional[PythonScalar] = None,
    cutoff: float = -1,
    nbins: int = 1000,
    lmax: int = 4,
    optimized_grid: int = 1,
    print_info: bool = True,
) -> TableSample:
    """
    Computes  the   K-Fold  randomized  search  of  an
    estimator.

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
        The main class to be considered as positive
        (classification only).
    cutoff: float, optional
        The model cutoff (classification only).
    nbins: int, optional
        Number of bins used to compute the different
        parameters categories.
    lmax: int, optional
        Maximum length of each parameter list.
    optimized_grid: int, optional
        If set to 0, the randomness is based on the
        input parameters.
        If set to 1,  the randomness is limited  to
        some  parameters  while others  are  picked
        based on a default grid.
        If set  to 2, there is no  randomness and a
        default grid is returned.
    print_info: bool, optional
        If set to True, prints the model information
        at each step.

    Returns
    -------
    TableSample
        result of the randomized search.
    """
    X = format_type(X, dtype=list)
    param_grid = gen_params_grid(estimator, nbins, len(X), lmax, optimized_grid)
    return grid_search_cv(
        estimator,
        param_grid,
        input_relation,
        X,
        y,
        metric=metric,
        cv=cv,
        average=average,
        pos_label=pos_label,
        cutoff=cutoff,
        training_score=True,
        skip_error="no_print",
        print_info=print_info,
    )


"""
GRID SEARCH
"""


@save_verticapy_logs
def grid_search_cv(
    estimator: VerticaModel,
    param_grid: Union[dict, list],
    input_relation: SQLRelation,
    X: SQLColumns,
    y: str,
    metric: str = "auto",
    cv: int = 3,
    average: Literal["binary", "micro", "macro", "weighted"] = "weighted",
    pos_label: Optional[PythonScalar] = None,
    cutoff: PythonNumber = -1,
    training_score: bool = True,
    skip_error: Union[bool, Literal["no_print"]] = True,
    print_info: bool = True,
    **kwargs,
) -> TableSample:
    """
    Computes the k-fold grid search of an estimator.

    Parameters
    ----------
    estimator: VerticaModel
        Vertica estimator with a fit method.
    param_grid: dict/list
        Dictionary of the parameters to test. It can
        also be a list of the different combinations.
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
        The main class to  be considered as positive
        (classification only).
    cutoff: float, optional
        The  model   cutoff  (classification  only).
    training_score: bool, optional
        If set to True,  the  training score is
        computed with the validation score.
    skip_error: bool, optional
        If set to True and an error occurs, the error
        is displayed but not raised.
    print_info: bool, optional
        If set to True, prints the model information
        at each step.

    Returns
    -------
    TableSample
        Result of the the grid search.
    """
    X = format_type(X, dtype=list)
    if estimator._model_subcategory == "REGRESSOR" and metric == "auto":
        metric = "rmse"
    elif metric == "auto":
        metric = "logloss"
    if isinstance(param_grid, dict):
        for param in param_grid:
            assert isinstance(param_grid[param], Iterable) and not (
                isinstance(param_grid[param], str)
            ), ValueError(
                "When of type dictionary, the parameter 'param_grid'"
                " must be a dictionary where each value is a list of "
                f"parameters, found {type(param_grid[param])} for "
                f"parameter '{param}'."
            )
        all_configuration = parameter_grid(param_grid)
    else:
        for idx, param in enumerate(param_grid):
            assert isinstance(param, dict), ValueError(
                "When of type List, the parameter 'param_grid' must "
                f"be a list of dictionaries, found {type(param)} for elem '{idx}'."
            )
        all_configuration = param_grid
    # testing all the config
    for config in all_configuration:
        estimator.set_params(config)
    # applying all the config
    data = []
    if all_configuration == []:
        all_configuration = [{}]
    if (
        conf.get_option("tqdm")
        and ("tqdm" not in kwargs or ("tqdm" in kwargs and kwargs["tqdm"]))
        and print_info
    ):
        loop = tqdm(all_configuration)
    else:
        loop = all_configuration
    for config in loop:
        try:
            estimator.set_params(config)
            current_cv = cross_validate(
                estimator,
                input_relation,
                X,
                y,
                metrics=metric,
                cv=cv,
                average=average,
                pos_label=pos_label,
                cutoff=cutoff,
                show_time=True,
                training_score=training_score,
                tqdm=False,
            )
            if training_score:
                keys = list(current_cv[0].values)
                data += [
                    (
                        estimator.get_params(),
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
                        f"Parameters: {config}; \033[91mTest_score: "
                        f"{current_cv[0][keys[1]][cv]}\033[0m; \033[92mTrain_score:"
                        f" {current_cv[1][keys[1]][cv]}\033[0m; \033[94mTime:"
                        f" {current_cv[0][keys[2]][cv]}\033[0m;"
                    )
            else:
                keys = list(current_cv.values)
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
                        f"Model: {str(estimator.__class__).split('.')[-1][:-2]}; "
                        f"Parameters: {config}; \033[91mTest_score: "
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
        result = TableSample(
            {
                "parameters": [d[0] for d in data],
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
            print("\033[1mGrid Search Selected Model\033[0m")
            print(
                f"{str(estimator.__class__).split('.')[-1][:-2]}; "
                f"Parameters: {result['parameters'][0]}; \033"
                f"[91mTest_score: {result['avg_score'][0]}\033[0m;"
                f" \033[92mTrain_score: {result['avg_train_score'][0]}"
                f"\033[0m; \033[94mTime: {result['avg_time'][0]}\033[0m;"
            )
    else:
        result = TableSample(
            {
                "parameters": [d[0] for d in data],
                "avg_score": [d[1] for d in data],
                "avg_time": [d[2] for d in data],
                "score_std": [d[3] for d in data],
            }
        )
        if print_info and (
            "final_print" not in kwargs or kwargs["final_print"] != "no_print"
        ):
            print("\033[1mGrid Search Selected Model\033[0m")
            print(
                f"{str(estimator.__class__).split('.')[-1][:-2]}; "
                f"Parameters: {result['parameters'][0]}; \033[91mTest_score:"
                f" {result['avg_score'][0]}\033[0m; \033[94mTime:"
                f" {result['avg_time'][0]}\033[0m;"
            )
    return result


"""
BAYESIAN SEARCH
"""


@save_verticapy_logs
def bayesian_search_cv(
    estimator: VerticaModel,
    input_relation: SQLRelation,
    X: SQLColumns,
    y: str,
    metric: str = "auto",
    cv: int = 3,
    pos_label: Optional[PythonScalar] = None,
    cutoff: float = -1,
    param_grid: Union[None, dict, list] = None,
    random_nbins: int = 16,
    bayesian_nbins: Optional[int] = None,
    random_grid: bool = False,
    lmax: int = 15,
    nrows: int = 100000,
    k_tops: int = 10,
    RFmodel_params: Optional[dict] = None,
    print_info: bool = True,
    **kwargs,
) -> TableSample:
    """
    Computes the k-fold bayesian search of an estimator
    using a random forest model to estimate a probably
    optimal set of parameters.

    Parameters
    ----------
    estimator: object
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
    pos_label: PythonScalar, optional
        The main class to be considered as positive
        (classification only).
    cutoff: float, optional
        The model cutoff (classification only).
    param_grid: dict/list, optional
        Dictionary of the parameters to test. It can
        also be a list of the different combinations.
        If empty, a parameter grid is generated.
    random_nbins: int, optional
        Number of bins used to compute the different
        parameters    categories   in   the   random
        parameters generation.
    bayesian_nbins: int, optional
        Number of bins used to compute the different
        parameters  categories in the bayesian table
        generation.
    random_grid: bool, optional
        If  True,  the rows used to find the optimal
        function  are used randomnly.  Otherwise,
        they are regularly spaced.
    lmax: int, optional
        Maximum length of each parameter list.
    nrows: int, optional
        Number  of  rows to use when performing  the
        bayesian search.
    k_tops: int, optional
        When  performing  the bayesian  search,  the
        final  stage  is to  retrain  the top
        possible  combinations. 'k_tops'  represents
        the number of  models to train at this stage
        in order to find the most efficient model.
    RFmodel_params: dict, optional
        Dictionary   of   the  random  forest  model
        parameters  used  to   estimate  a  probably
        optimal set of parameters.
    print_info: bool, optional
        If True, prints the model information at each
        step.

    Returns
    -------
    TableSample
        result of the bayesian search.
    """
    RFmodel_params, param_grid = format_type(RFmodel_params, param_grid, dtype=dict)
    X = format_type(X, dtype=list)
    if print_info:
        print(f"\033[1m\033[4mStarting Bayesian Search\033[0m\033[0m\n")
        print(
            f"\033[1m\033[4mStep 1 - Computing Random Models"
            " using Grid Search\033[0m\033[0m\n"
        )
    if not param_grid:
        param_grid = gen_params_grid(estimator, random_nbins, len(X), lmax, 0)
    param_gs = grid_search_cv(
        estimator,
        param_grid,
        input_relation,
        X,
        y,
        metric=metric,
        cv=cv,
        pos_label=pos_label,
        cutoff=cutoff,
        training_score=True,
        skip_error="no_print",
        print_info=print_info,
        final_print="no_print",
    )
    if "enet" not in kwargs:
        params = []
        for param_grid in param_gs["parameters"]:
            params += list(param_grid)
        all_params = list(dict.fromkeys(params))
    else:
        all_params = ["C", "l1_ratio"]
    if not bayesian_nbins:
        bayesian_nbins = max(int(np.exp(np.log(nrows) / len(all_params))), 1)
    result = {}
    for elem in all_params:
        result[elem] = []
    for param_grid in param_gs["parameters"]:
        for item in all_params:
            if item in param_grid:
                result[item] += [param_grid[item]]
            else:
                result[item] += [None]
    result["score"] = param_gs["avg_score"]
    if "max_features" in result:
        for idx, elem in enumerate(result["max_features"]):
            if elem == "auto":
                result["max_features"][idx] = int(np.floor(np.sqrt(len(X))) + 1)
            elif elem == "max":
                result["max_features"][idx] = int(len(X))
    result = TableSample(result).to_sql()
    schema = conf.get_option("temp_schema")
    relation = gen_tmp_name(schema=schema, name="bayesian")
    drop(relation, method="table")
    _executeSQL(f"CREATE TABLE {relation} AS {result}", print_time_sql=False)
    if print_info:
        print(
            f"\033[1m\033[4mStep 2 - Fitting the RF model with "
            "the hyperparameters data\033[0m\033[0m\n"
        )
    if conf.get_option("tqdm") and print_info:
        loop = tqdm(range(1))
    else:
        loop = range(1)
    for j in loop:
        if "enet" not in kwargs:
            model_grid = gen_params_grid(
                estimator,
                nbins=bayesian_nbins,
                max_nfeatures=len(all_params),
                optimized_grid=-666,
            )
        else:
            model_grid = {
                "C": {"type": float, "range": [0.0, 10], "nbins": bayesian_nbins},
                "l1_ratio": {
                    "type": float,
                    "range": [0.0, 1.0],
                    "nbins": bayesian_nbins,
                },
            }
        all_params = list(dict.fromkeys(model_grid))
        hyper_param_estimator = vml.RandomForestRegressor(
            name=estimator.model_name, **RFmodel_params
        )
        hyper_param_estimator.fit(
            relation,
            all_params,
            "score",
            return_report=True,
        )
        if random_grid:
            vdf = gen_dataset(model_grid, nrows=nrows)
        else:
            vdf = gen_meshgrid(model_grid)
        drop(relation, method="table")
        vdf.to_db(relation, relation_type="table", inplace=True)
        vdf = hyper_param_estimator.predict(vdf, name="score")
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
        vdf.sort({"score": "desc" if reverse else "asc"})
        result = vdf.head(limit=k_tops)
        new_param_grid = []
        for i in range(k_tops):
            param_tmp_grid = {}
            for elem in result.values:
                if elem != "score":
                    param_tmp_grid[elem] = result[elem][i]
            new_param_grid += [param_tmp_grid]
    if print_info:
        print(
            f"\033[1m\033[4mStep 3 - Computing Most Probable Good "
            "Models using Grid Search\033[0m\033[0m\n"
        )
    result = grid_search_cv(
        estimator,
        new_param_grid,
        input_relation,
        X,
        y,
        metric=metric,
        cv=cv,
        pos_label=pos_label,
        cutoff=cutoff,
        training_score=True,
        skip_error="no_print",
        print_info=print_info,
        final_print="no_print",
    )
    for elem in result.values:
        result.values[elem] += param_gs[elem]
    data = []
    keys = list(result.values)
    for i in range(len(result[keys[0]])):
        data += [tuple([result[elem][i] for elem in result.values])]
    data.sort(key=lambda tup: tup[1], reverse=reverse)
    for idx, elem in enumerate(result.values):
        result.values[elem] = [item[idx] for item in data]
    hyper_param_estimator.drop()
    if print_info:
        print("\033[1mBayesian Search Selected Model\033[0m")
        print(
            f"Parameters: {result['parameters'][0]}; \033[91mTest_score:"
            f" {result['avg_score'][0]}\033[0m; \033[92mTrain_score: "
            f"{result['avg_train_score'][0]}\033[0m; \033[94mTime: "
            f"{result['avg_time'][0]}\033[0m;"
        )
    drop(relation, method="table")
    return result


"""
ENET AUTO GRID SEARCH
"""


@save_verticapy_logs
def enet_search_cv(
    input_relation: SQLRelation,
    X: SQLColumns,
    y: str,
    metric: str = "auto",
    cv: int = 3,
    estimator_type: Literal["logit", "enet", "auto"] = "auto",
    cutoff: float = -1.0,
    print_info: bool = True,
    **kwargs,
) -> TableSample:
    """
    Computes the  k-fold grid search using multiple ENet
    models.

    Parameters
    ----------
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
    estimator_type: str, optional
        Estimator Type.
            auto : detects if it is a Logit Model
                   or ENet.
            logit: Logistic Regression
            enet : ElasticNet
    cutoff: float, optional
        The model cutoff (logit only).
    print_info: bool, optional
        If   set   to  True,   prints  the  model
        information at each step.

    Returns
    -------
    TableSample
        result of the ENET search.
    """
    X = format_type(X, dtype=list)
    param_grid = parameter_grid(
        {
            "solver": ["cgd"],
            "penalty": ["enet"],
            "C": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 5.0, 10.0, 50.0, 100.0]
            if "small" not in kwargs
            else [1e-1, 1.0, 10.0],
            "l1_ratio": [0.1 * i for i in range(0, 10)]
            if "small" not in kwargs
            else [0.1, 0.5, 0.9],
        }
    )
    if estimator_type == "auto":
        if not isinstance(input_relation, vDataFrame):
            vdf = vDataFrame(input_relation)
        else:
            vdf = input_relation
        if sorted(vdf[y].distinct()) == [0, 1]:
            estimator_type = "logit"
        else:
            estimator_type = "enet"
    if estimator_type == "logit":
        estimator = vml.LogisticRegression()
    else:
        estimator = vml.ElasticNet()
    result = bayesian_search_cv(
        estimator,
        input_relation,
        X,
        y,
        metric=metric,
        cv=cv,
        pos_label=None,
        cutoff=cutoff,
        param_grid=param_grid,
        random_grid=False,
        bayesian_nbins=1000,
        print_info=print_info,
        enet=True,
    )
    return result
