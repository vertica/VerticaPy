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
import itertools, math, random
from typing import Literal, Union
from collections.abc import Iterable
from tqdm.auto import tqdm
import numpy as np

import matplotlib.pyplot as plt

from verticapy._config.colors import get_colors
import verticapy._config.config as conf
from verticapy._typing import PythonNumber, PythonScalar, SQLColumns, SQLRelation
from verticapy._utils._gen import gen_tmp_name
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._sys import _executeSQL
from verticapy.errors import ParameterError

from verticapy.core.tablesample.base import TableSample
from verticapy.core.vdataframe.base import vDataFrame

from verticapy.datasets.generators import gen_meshgrid, gen_dataset

from verticapy.plotting._matplotlib.base import updated_dict
from verticapy.plotting._matplotlib.timeseries import range_curve

from verticapy.machine_learning._utils import reverse_score
from verticapy.machine_learning.model_selection.model_validation import cross_validate

from verticapy.sql.drop import drop


@save_verticapy_logs
def bayesian_search_cv(
    estimator,
    input_relation: SQLRelation,
    X: list,
    y: str,
    metric: str = "auto",
    cv: int = 3,
    pos_label: PythonScalar = None,
    cutoff: float = -1,
    param_grid: Union[dict, list] = {},
    random_nbins: int = 16,
    bayesian_nbins: int = None,
    random_grid: bool = False,
    lmax: int = 15,
    nrows: int = 100000,
    k_tops: int = 10,
    RFmodel_params: dict = {},
    print_info: bool = True,
    **kwargs,
):
    """
Computes the k-fold bayesian search of an estimator using a random
forest model to estimate a probable optimal set of parameters.

Parameters
----------
estimator: object
    Vertica estimator with a fit method.
input_relation: SQLRelation
    Relation to use to train the model.
X: list
    List of the predictor columns.
y: str
    Response Column.
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
pos_label: PythonScalar, optional
    The main class to be considered as positive (classification only).
cutoff: float, optional
    The model cutoff (classification only).
param_grid: dict/list, optional
    Dictionary of the parameters to test. It can also be a list of the
    different combinations. If empty, a parameter grid will be generated.
random_nbins: int, optional
    Number of bins used to compute the different parameters categories
    in the random parameters generation.
bayesian_nbins: int, optional
    Number of bins used to compute the different parameters categories
    in the bayesian table generation.
random_grid: bool, optional
    If True, the rows used to find the optimal function will be
    used randomnly. Otherwise, they will be regularly spaced. 
lmax: int, optional
    Maximum length of each parameter list.
nrows: int, optional
    Number of rows to use when performing the bayesian search.
k_tops: int, optional
    When performing the bayesian search, the final stage will be to retrain the top
    possible combinations. 'k_tops' represents the number of models to train at
    this stage to find the most efficient model.
RFmodel_params: dict, optional
    Dictionary of the random forest model parameters used to estimate a probable 
    optimal set of parameters.
print_info: bool, optional
    If True, prints the model information at each step.

Returns
-------
TableSample
    An object containing the result. For more information, see
    utilities.TableSample.
    """
    if print_info:
        print(f"\033[1m\033[4mStarting Bayesian Search\033[0m\033[0m\n")
        print(
            f"\033[1m\033[4mStep 1 - Computing Random Models"
            " using Grid Search\033[0m\033[0m\n"
        )
    if not (param_grid):
        param_grid = gen_params_grid(estimator, random_nbins, len(X), lmax, 0)
    param_gs = grid_search_cv(
        estimator,
        param_grid,
        input_relation,
        X,
        y,
        metric,
        cv,
        pos_label,
        cutoff,
        True,
        "no_print",
        print_info,
        final_print="no_print",
    )
    if "enet" not in kwargs:
        params = []
        for param_grid in param_gs["parameters"]:
            params += [elem for elem in param_grid]
        all_params = list(dict.fromkeys(params))
    else:
        all_params = ["C", "l1_ratio"]
    if not (bayesian_nbins):
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
    model_name = gen_tmp_name(schema=schema, name="rf")
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
        from verticapy.machine_learning.vertica.ensemble import RandomForestRegressor

        hyper_param_estimator = RandomForestRegressor(
            name=estimator.model_name, **RFmodel_params
        )
        hyper_param_estimator.fit(relation, all_params, "score")
        if random_grid:
            vdf = gen_dataset(model_grid, nrows=nrows)
        else:
            vdf = gen_meshgrid(model_grid)
        drop(relation, method="table")
        vdf.to_db(relation, relation_type="table", inplace=True)
        vdf = hyper_param_estimator.predict(vdf, name="score")
        reverse = reverse_score(metric)
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
        metric,
        cv,
        pos_label,
        cutoff,
        True,
        "no_print",
        print_info,
        final_print="no_print",
    )
    for elem in result.values:
        result.values[elem] += param_gs[elem]
    data = []
    keys = [elem for elem in result.values]
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


@save_verticapy_logs
def enet_search_cv(
    input_relation: SQLRelation,
    X: list,
    y: str,
    metric: str = "auto",
    cv: int = 3,
    estimator_type: Literal["logit", "enet", "auto"] = "auto",
    cutoff: float = -1,
    print_info: bool = True,
    **kwargs,
):
    """
Computes the k-fold grid search using multiple ENet models.

Parameters
----------
input_relation: SQLRelation
    Relation to use to train the model.
X: list
    List of the predictor columns.
y: str
    Response Column.
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
estimator_type: str, optional
    Estimator Type.
        auto : detects if it is a Logit Model or ENet.
        logit: Logistic Regression
        enet : ElasticNet
cutoff: float, optional
    The model cutoff (logit only).
print_info: bool, optional
    If set to True, prints the model information at each step.

Returns
-------
TableSample
    An object containing the result. For more information, see
    utilities.TableSample.
    """
    import verticapy.machine_learning.vertica as vml

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
        if not (isinstance(input_relation, vDataFrame)):
            vdf = vDataFrame(input_relation)
        else:
            vdf = input_relation
        if sorted(vdf[y].distinct()) == [0, 1]:
            estimator_type = "logit"
        else:
            estimator_type = "enet"
    if estimator_type == "logit":
        estimator = vml.LogisticRegression(
            gen_tmp_name(schema=conf.get_option("temp_schema"), name="logit")
        )
    else:
        estimator = vml.ElasticNet(
            gen_tmp_name(schema=conf.get_option("temp_schema"), name="enet")
        )
    result = bayesian_search_cv(
        estimator,
        input_relation,
        X,
        y,
        metric,
        cv,
        None,
        cutoff,
        param_grid,
        random_grid=False,
        bayesian_nbins=1000,
        print_info=print_info,
        enet=True,
    )
    return result


@save_verticapy_logs
def gen_params_grid(
    estimator,
    nbins: int = 10,
    max_nfeatures: int = 3,
    lmax: int = -1,
    optimized_grid: int = 0,
):
    """
Generates the estimator grid.

Parameters
----------
estimator: object
    Vertica estimator with a fit method.
nbins: int, optional
    Number of bins used to discretize numberical features.
max_nfeatures: int, optional
    Maximum number of features used to compute Random Forest, PCA...
lmax: int, optional
    Maximum length of the parameter grid.
optimized_grid: int, optional
    If set to 0, the randomness is based on the input parameters.
    If set to 1, the randomness is limited to some parameters while others
    are picked based on a default grid.
    If set to 2, there is no randomness and a default grid is returned.
    
Returns
-------
TableSample
    An object containing the result. For more information, see
    utilities.TableSample.
    """
    import verticapy.machine_learning.vertica as vml

    params_grid = {}
    if isinstance(
        estimator, (vml.DummyTreeRegressor, vml.DummyTreeClassifier, vml.OneHotEncoder)
    ):
        return params_grid
    elif isinstance(
        estimator,
        (
            vml.RandomForestRegressor,
            vml.RandomForestClassifier,
            vml.DecisionTreeRegressor,
            vml.DecisionTreeClassifier,
        ),
    ):
        if optimized_grid == 0:
            params_grid = {
                "max_features": ["auto", "max"]
                + list(range(1, max_nfeatures, math.ceil(max_nfeatures / nbins))),
                "max_leaf_nodes": list(range(1, int(1e9), math.ceil(int(1e9) / nbins))),
                "max_depth": list(range(1, 100, math.ceil(100 / nbins))),
                "min_samples_leaf": list(
                    range(1, int(1e6), math.ceil(int(1e6) / nbins))
                ),
                "min_info_gain": [
                    elem / 1000 for elem in range(1, 1000, math.ceil(1000 / nbins))
                ],
                "nbins": list(range(2, 100, math.ceil(100 / nbins))),
            }
            if isinstance(
                estimator, (vml.RandomForestRegressor, vml.RandomForestClassifier)
            ):
                params_grid["sample"] = [
                    elem / 1000 for elem in range(1, 1000, math.ceil(1000 / nbins))
                ]
                params_grid["n_estimators"] = list(
                    range(1, 100, math.ceil(100 / nbins))
                )
        elif optimized_grid == 1:
            params_grid = {
                "max_features": ["auto", "max"],
                "max_leaf_nodes": [32, 64, 128, 1000, 1e4, 1e6, 1e9],
                "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50],
                "min_samples_leaf": [1, 2, 3, 4, 5],
                "min_info_gain": [0.0, 0.1, 0.2],
                "nbins": [10, 15, 20, 25, 30, 35, 40],
            }
            if isinstance(
                estimator, (vml.RandomForestRegressor, vml.RandomForestClassifier)
            ):
                params_grid["sample"] = [
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    1.0,
                ]
                params_grid["n_estimators"] = [1, 5, 10, 15, 20, 30, 40, 50, 100]
        elif optimized_grid == 2:
            params_grid = {
                "max_features": ["auto", "max"],
                "max_leaf_nodes": [32, 64, 128, 1000],
                "max_depth": [4, 5, 6],
                "min_samples_leaf": [1, 2],
                "min_info_gain": [0.0],
                "nbins": [32],
            }
            if isinstance(
                estimator, (vml.RandomForestRegressor, vml.RandomForestClassifier)
            ):
                params_grid["sample"] = [0.7]
                params_grid["n_estimators"] = [20]
        elif optimized_grid == -666:
            result = {
                "max_features": {
                    "type": int,
                    "range": [1, max_nfeatures],
                    "nbins": nbins,
                },
                "max_leaf_nodes": {"type": int, "range": [32, 1e9], "nbins": nbins},
                "max_depth": {"type": int, "range": [2, 30], "nbins": nbins},
                "min_samples_leaf": {"type": int, "range": [1, 15], "nbins": nbins},
                "min_info_gain": {"type": float, "range": [0.0, 0.1], "nbins": nbins,},
                "nbins": {"type": int, "range": [10, 1000], "nbins": nbins},
            }
            if isinstance(
                estimator, (vml.RandomForestRegressor, vml.RandomForestClassifier)
            ):
                result["sample"] = {
                    "type": float,
                    "range": [0.1, 1.0],
                    "nbins": nbins,
                }
                result["n_estimators"] = {
                    "type": int,
                    "range": [1, 100],
                    "nbins": nbins,
                }
            return result
    elif isinstance(estimator, (vml.LinearSVC, vml.LinearSVR)):
        if optimized_grid == 0:
            params_grid = {
                "tol": [1e-4, 1e-6, 1e-8],
                "C": [elem / 1000 for elem in range(1, 5000, math.ceil(5000 / nbins))],
                "fit_intercept": [False, True],
                "intercept_mode": ["regularized", "unregularized"],
                "max_iter": [100, 500, 1000],
            }
        elif optimized_grid == 1:
            params_grid = {
                "tol": [1e-6],
                "C": [1e-1, 0.0, 1.0, 10.0],
                "fit_intercept": [True],
                "intercept_mode": ["regularized", "unregularized"],
                "max_iter": [100],
            }
        elif optimized_grid == 2:
            params_grid = {
                "tol": [1e-6],
                "C": [0.0, 1.0],
                "fit_intercept": [True],
                "intercept_mode": ["regularized", "unregularized"],
                "max_iter": [100],
            }
        elif optimized_grid == -666:
            return {
                "tol": {"type": float, "range": [1e-8, 1e-2], "nbins": nbins},
                "C": {"type": float, "range": [0.0, 1000.0], "nbins": nbins},
                "fit_intercept": {"type": bool},
                "intercept_mode": {
                    "type": str,
                    "values": ["regularized", "unregularized"],
                },
                "max_iter": {"type": int, "range": [10, 1000], "nbins": nbins},
            }
    elif isinstance(estimator, (vml.XGBClassifier, vml.XGBRegressor)):
        if optimized_grid == 0:
            params_grid = {
                "nbins": list(range(2, 100, math.ceil(100 / nbins))),
                "max_depth": list(range(1, 20, math.ceil(100 / nbins))),
                "weight_reg": [
                    elem / 1000 for elem in range(1, 1000, math.ceil(1000 / nbins))
                ],
                "min_split_loss": [
                    elem / 1000 for elem in range(1, 1000, math.ceil(1000 / nbins))
                ],
                "learning_rate": [
                    elem / 1000 for elem in range(1, 1000, math.ceil(1000 / nbins))
                ],
                # "sample": [elem / 1000 for elem in range(1, 1000, math.ceil(1000 / nbins))],
                "tol": [1e-4, 1e-6, 1e-8],
                "max_ntree": list(range(1, 100, math.ceil(100 / nbins))),
            }
        elif optimized_grid == 1:
            params_grid = {
                "nbins": [10, 15, 20, 25, 30, 35, 40],
                "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
                "weight_reg": [0.0, 0.5, 1.0, 2.0],
                "min_split_loss": [0.0, 0.1, 0.25],
                "learning_rate": [0.01, 0.05, 0.1, 1.0],
                # "sample": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "tol": [1e-8],
                "max_ntree": [1, 10, 20, 30, 40, 50, 100],
            }
        elif optimized_grid == 2:
            params_grid = {
                "nbins": [32],
                "max_depth": [3, 4, 5],
                "weight_reg": [0.0, 0.25],
                "min_split_loss": [0.0],
                "learning_rate": [0.05, 0.1, 1.0],
                # "sample": [0.5, 0.6, 0.7],
                "tol": [1e-8],
                "max_ntree": [20],
            }
        elif optimized_grid == -666:
            return {
                "nbins": {"type": int, "range": [2, 100], "nbins": nbins},
                "max_depth": {"type": int, "range": [1, 20], "nbins": nbins},
                "weight_reg": {"type": float, "range": [0.0, 1.0], "nbins": nbins},
                "min_split_loss": {
                    "type": float,
                    "values": [0.0, 0.25],
                    "nbins": nbins,
                },
                "learning_rate": {"type": float, "range": [0.0, 1.0], "nbins": nbins,},
                "sample": {"type": float, "range": [0.0, 1.0], "nbins": nbins},
                "tol": {"type": float, "range": [1e-8, 1e-2], "nbins": nbins},
                "max_ntree": {"type": int, "range": [1, 20], "nbins": nbins},
            }
    elif isinstance(estimator, vml.NaiveBayes):
        if optimized_grid == 0:
            params_grid = {
                "alpha": [
                    elem / 1000 for elem in range(1, 1000, math.ceil(1000 / nbins))
                ]
            }
        elif optimized_grid == 1:
            params_grid = {"alpha": [0.01, 0.1, 1.0, 5.0, 10.0]}
        elif optimized_grid == 2:
            params_grid = {"alpha": [0.01, 1.0, 10.0]}
        elif optimized_grid == -666:
            return {
                "alpha": {"type": float, "range": [0.00001, 1000.0], "nbins": nbins}
            }
    elif isinstance(estimator, (vml.PCA, vml.SVD)):
        if optimized_grid == 0:
            params_grid = {
                "max_features": list(
                    range(1, max_nfeatures, math.ceil(max_nfeatures / nbins))
                )
            }
        if isinstance(estimator, vml.PCA):
            params_grid["scale"] = [False, True]
        if optimized_grid == -666:
            return {
                "scale": {"type": bool},
                "max_features": {
                    "type": int,
                    "range": [1, max_nfeatures],
                    "nbins": nbins,
                },
            }
    elif isinstance(estimator, vml.Scaler):
        params_grid = {"method": ["minmax", "robust_zscore", "zscore"]}
        if optimized_grid == -666:
            return {
                "method": {
                    "type": str,
                    "values": ["minmax", "robust_zscore", "zscore"],
                }
            }
    elif isinstance(
        estimator,
        (
            vml.KNeighborsRegressor,
            vml.KNeighborsClassifier,
            vml.LocalOutlierFactor,
            vml.NearestCentroid,
        ),
    ):
        if optimized_grid == 0:
            params_grid = {
                "p": [1, 2] + list(range(3, 100, math.ceil(100 / (nbins - 2))))
            }
            if isinstance(
                estimator,
                (
                    vml.KNeighborsRegressor,
                    vml.KNeighborsClassifier,
                    vml.LocalOutlierFactor,
                ),
            ):
                params_grid["n_neighbors"] = list(
                    range(1, 100, math.ceil(100 / (nbins)))
                )
        elif optimized_grid == 1:
            params_grid = {"p": [1, 2, 3, 4]}
            if isinstance(
                estimator,
                (
                    vml.KNeighborsRegressor,
                    vml.KNeighborsClassifier,
                    vml.LocalOutlierFactor,
                ),
            ):
                params_grid["n_neighbors"] = [1, 2, 3, 4, 5, 10, 20, 100]
        elif optimized_grid == 2:
            params_grid = {"p": [1, 2]}
            if isinstance(
                estimator,
                (
                    vml.KNeighborsRegressor,
                    vml.KNeighborsClassifier,
                    vml.LocalOutlierFactor,
                ),
            ):
                params_grid["n_neighbors"] = [5, 10]
        elif optimized_grid == -666:
            return {
                "p": {"type": int, "range": [1, 10], "nbins": nbins},
                "n_neighbors": {"type": int, "range": [1, 100], "nbins": nbins},
            }
    elif isinstance(estimator, vml.DBSCAN):
        if optimized_grid == 0:
            params_grid = {
                "p": [1, 2] + list(range(3, 100, math.ceil(100 / (nbins - 2)))),
                "eps": [
                    elem / 1000 for elem in range(1, 1000, math.ceil(1000 / nbins))
                ],
                "min_samples": list(range(1, 1000, math.ceil(1000 / nbins))),
            }
        elif optimized_grid == 1:
            params_grid = {
                "p": [1, 2, 3, 4],
                "min_samples": [1, 2, 3, 4, 5, 10, 100],
            }
        elif optimized_grid == 2:
            params_grid = {"p": [1, 2], "min_samples": [5, 10]}
        elif optimized_grid == -666:
            return {
                "p": {"type": int, "range": [1, 10], "nbins": nbins},
                "min_samples": {"type": int, "range": [1, 100], "nbins": nbins},
            }
    elif isinstance(
        estimator,
        (
            vml.LogisticRegression,
            vml.LinearRegression,
            vml.ElasticNet,
            vml.Lasso,
            vml.Ridge,
        ),
    ):
        if optimized_grid == 0:
            params_grid = {"tol": [1e-4, 1e-6, 1e-8], "max_iter": [100, 500, 1000]}
            if isinstance(estimator, vml.LogisticRegression):
                params_grid["penalty"] = ["none", "l1", "l2", "enet"]
            if isinstance(estimator, vml.LinearRegression):
                params_grid["solver"] = ["newton", "bfgs"]
            elif isinstance(
                estimator, (vml.Lasso, vml.LogisticRegression, vml.ElasticNet)
            ):
                params_grid["solver"] = ["newton", "bfgs", "cgd"]
            if isinstance(
                estimator,
                (vml.Lasso, vml.Ridge, vml.ElasticNet, vml.LogisticRegression),
            ):
                params_grid["C"] = [
                    elem / 1000 for elem in range(1, 5000, math.ceil(5000 / nbins))
                ]
            if isinstance(estimator, (vml.LogisticRegression, vml.ElasticNet)):
                params_grid["l1_ratio"] = [
                    elem / 1000 for elem in range(1, 1000, math.ceil(1000 / nbins))
                ]
        elif optimized_grid == 1:
            params_grid = {"tol": [1e-6], "max_iter": [100]}
            if isinstance(estimator, vml.LogisticRegression):
                params_grid["penalty"] = ["none", "l1", "l2", "enet"]
            if isinstance(estimator, vml.LinearRegression):
                params_grid["solver"] = ["newton", "bfgs"]
            elif isinstance(
                estimator, (vml.Lasso, vml.LogisticRegression, vml.ElasticNet)
            ):
                params_grid["solver"] = ["newton", "bfgs", "cgd"]
            if isinstance(
                estimator,
                (vml.Lasso, vml.Ridge, vml.ElasticNet, vml.LogisticRegression),
            ):
                params_grid["C"] = [1e-1, 0.0, 1.0, 10.0]
            if isinstance(estimator, vml.LogisticRegression):
                params_grid["penalty"] = ["none", "l1", "l2", "enet"]
            if isinstance(estimator, (vml.LogisticRegression, vml.ElasticNet)):
                params_grid["l1_ratio"] = [
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                ]
        elif optimized_grid == 2:
            params_grid = {"tol": [1e-6], "max_iter": [100]}
            if isinstance(estimator, vml.LogisticRegression):
                params_grid["penalty"] = ["none", "l1", "l2", "enet"]
            if isinstance(estimator, vml.LinearRegression):
                params_grid["solver"] = ["newton", "bfgs"]
            elif isinstance(
                estimator, (vml.Lasso, vml.LogisticRegression, vml.ElasticNet)
            ):
                params_grid["solver"] = ["bfgs", "cgd"]
            if isinstance(
                estimator,
                (vml.Lasso, vml.Ridge, vml.ElasticNet, vml.LogisticRegression),
            ):
                params_grid["C"] = [1.0]
            if isinstance(estimator, vml.LogisticRegression):
                params_grid["penalty"] = ["none", "l1", "l2", "enet"]
            if isinstance(estimator, (vml.LogisticRegression, vml.ElasticNet)):
                params_grid["l1_ratio"] = [0.5]
        elif optimized_grid == -666:
            result = {
                "tol": {"type": float, "range": [1e-8, 1e-2], "nbins": nbins},
                "max_iter": {"type": int, "range": [1, 1000], "nbins": nbins},
            }
            if isinstance(estimator, vml.LogisticRegression):
                result["penalty"] = {
                    "type": str,
                    "values": ["none", "l1", "l2", "enet"],
                }
            if isinstance(estimator, vml.LinearRegression):
                result["solver"] = {"type": str, "values": ["newton", "bfgs"]}
            elif isinstance(
                estimator, (vml.Lasso, vml.LogisticRegression, vml.ElasticNet)
            ):
                result["solver"] = {"type": str, "values": ["bfgs", "cgd"]}
            if isinstance(
                estimator,
                (vml.Lasso, vml.Ridge, vml.ElasticNet, vml.LogisticRegression),
            ):
                result["C"] = {
                    "type": float,
                    "range": [0.0, 1000.0],
                    "nbins": nbins,
                }
            if isinstance(estimator, vml.LogisticRegression):
                result["penalty"] = {
                    "type": str,
                    "values": ["none", "l1", "l2", "enet"],
                }
            if isinstance(estimator, (vml.LogisticRegression, vml.ElasticNet)):
                result["l1_ratio"] = {
                    "type": float,
                    "range": [0.0, 1.0],
                    "nbins": nbins,
                }
            return result
    elif isinstance(estimator, vml.KMeans):
        if optimized_grid == 0:
            params_grid = {
                "n_cluster": list(range(2, 100, math.ceil(100 / nbins))),
                "init": ["kmeanspp", "random"],
                "max_iter": [100, 500, 1000],
                "tol": [1e-4, 1e-6, 1e-8],
            }
        elif optimized_grid == 1:
            params_grid = {
                "n_cluster": [
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    15,
                    20,
                    50,
                    100,
                    200,
                    300,
                    1000,
                ],
                "init": ["kmeanspp", "random"],
                "max_iter": [1000],
                "tol": [1e-8],
            }
        elif optimized_grid == 2:
            params_grid = {
                "n_cluster": [2, 3, 4, 5, 10, 20, 100],
                "init": ["kmeanspp"],
                "max_iter": [1000],
                "tol": [1e-8],
            }
        elif optimized_grid == -666:
            return {
                "tol": {"type": float, "range": [1e-2, 1e-8], "nbins": nbins},
                "max_iter": {"type": int, "range": [1, 1000], "nbins": nbins},
                "n_cluster": {"type": int, "range": [1, 10000], "nbins": nbins},
                "init": {"type": str, "values": ["kmeanspp", "random"]},
            }
    elif isinstance(estimator, vml.KPrototypes):
        if optimized_grid == 0:
            params_grid = {
                "n_cluster": list(range(2, 100, math.ceil(100 / nbins))),
                "init": ["random"],
                "max_iter": [100, 500, 1000],
                "tol": [1e-4, 1e-6, 1e-8],
                "gamma": [0.1, 1.0, 10.0],
            }
        elif optimized_grid == 1:
            params_grid = {
                "n_cluster": [
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    15,
                    20,
                    50,
                    100,
                    200,
                    300,
                    1000,
                ],
                "init": ["random"],
                "max_iter": [1000],
                "tol": [1e-8],
                "gamma": [1.0],
            }
        elif optimized_grid == 2:
            params_grid = {
                "n_cluster": [2, 3, 4, 5, 10, 20, 100],
                "init": ["random"],
                "max_iter": [1000],
                "tol": [1e-8],
                "gamma": [1.0],
            }
        elif optimized_grid == -666:
            return {
                "tol": {"type": float, "range": [1e-2, 1e-8], "nbins": nbins},
                "max_iter": {"type": int, "range": [1, 1000], "nbins": nbins},
                "n_cluster": {"type": int, "range": [1, 10000], "nbins": nbins},
                "gamma": {"type": float, "range": [1e-2, 100], "nbins": nbins},
                "init": {"type": str, "values": ["random"]},
            }
    elif isinstance(estimator, vml.BisectingKMeans):
        if optimized_grid == 0:
            params_grid = {
                "n_cluster": list(range(2, 100, math.ceil(100 / nbins))),
                "bisection_iterations": list(range(10, 1000, math.ceil(1000 / nbins))),
                "split_method": ["size", "sum_squares"],
                "min_divisible_cluster_size": list(
                    range(2, 100, math.ceil(100 / nbins))
                ),
                "init": ["kmeanspp", "pseudo"],
                "max_iter": [100, 500, 1000],
                "tol": [1e-4, 1e-6, 1e-8],
            }
        elif optimized_grid == 1:
            params_grid = {
                "n_cluster": [
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    15,
                    20,
                    50,
                    100,
                    200,
                    300,
                    1000,
                ],
                "bisection_iterations": list(range(10, 1000, math.ceil(1000 / nbins))),
                "split_method": ["size", "sum_squares"],
                "min_divisible_cluster_size": list(
                    range(2, 100, math.ceil(100 / nbins))
                ),
                "init": ["kmeanspp", "pseudo"],
                "max_iter": [1000],
                "tol": [1e-8],
            }
        elif optimized_grid == 2:
            params_grid = {
                "n_cluster": [2, 3, 4, 5, 10, 20, 100],
                "bisection_iterations": [1, 2, 3],
                "split_method": ["sum_squares"],
                "min_divisible_cluster_size": [2, 3, 4],
                "init": ["kmeanspp"],
                "max_iter": [1000],
                "tol": [1e-8],
            }
        elif optimized_grid == -666:
            return {
                "tol": {"type": float, "range": [1e-8, 1e-2], "nbins": nbins},
                "max_iter": {"type": int, "range": [1, 1000], "nbins": nbins},
                "bisection_iterations": {
                    "type": int,
                    "range": [1, 1000],
                    "nbins": nbins,
                },
                "split_method": {"type": str, "values": ["sum_squares"]},
                "n_cluster": {"type": int, "range": [1, 10000], "nbins": nbins},
                "init": {"type": str, "values": ["kmeanspp", "pseudo"]},
            }
    params_grid = parameter_grid(params_grid)
    final_param_grid = []
    for param in params_grid:
        if "C" in param and param["C"] == 0:
            del param["C"]
            if "l1_ratio" in param:
                del param["l1_ratio"]
            if "penalty" in param:
                param["penalty"] = "none"
        if "penalty" in param:
            if (
                param["penalty"] in ("none", "l2")
                and "solver" in param
                and param["solver"] == "cgd"
            ):
                param["solver"] = "bfgs"
            if param["penalty"] in ("none", "l1", "l2") and "l1_ratio" in param:
                del param["l1_ratio"]
            if param["penalty"] == "none" and "C" in param:
                del param["C"]
            if param["penalty"] in ("l1", "enet") and "solver" in param:
                param["solver"] = "cgd"
        if param not in final_param_grid:
            final_param_grid += [param]
    if len(final_param_grid) > lmax and lmax > 0:
        final_param_grid = random.sample(final_param_grid, lmax)
    return final_param_grid


@save_verticapy_logs
def grid_search_cv(
    estimator,
    param_grid: Union[dict, list],
    input_relation: SQLRelation,
    X: SQLColumns,
    y: str,
    metric: str = "auto",
    cv: int = 3,
    pos_label: PythonScalar = None,
    cutoff: PythonNumber = -1,
    training_score: bool = True,
    skip_error: bool = True,
    print_info: bool = True,
    **kwargs,
):
    """
Computes the k-fold grid search of an estimator.

Parameters
----------
estimator: object
    Vertica estimator with a fit method.
param_grid: dict/list
    Dictionary of the parameters to test. It can also be a list of the
    different combinations.
input_relation: SQLRelation
    Relation to use to train the model.
X: list
    List of the predictor columns.
y: str
    Response Column.
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
pos_label: PythonScalar, optional
    The main class to be considered as positive (classification only).
cutoff: float, optional
    The model cutoff (classification only).
training_score: bool, optional
    If set to True, the training score will be computed with the validation score.
skip_error: bool, optional
    If set to True and an error occurs, it will be displayed and not raised.
print_info: bool, optional
    If set to True, prints the model information at each step.

Returns
-------
TableSample
    An object containing the result. For more information, see
    utilities.TableSample.
    """
    if isinstance(X, str):
        X = [X]
    if estimator._model_subcategory == "REGRESSOR" and metric == "auto":
        metric = "rmse"
    elif metric == "auto":
        metric = "logloss"
    if isinstance(param_grid, dict):
        for param in param_grid:
            assert isinstance(param_grid[param], Iterable) and not (
                isinstance(param_grid[param], str)
            ), ParameterError(
                "When of type dictionary, the parameter 'param_grid'"
                " must be a dictionary where each value is a list of "
                f"parameters, found {type(param_grid[param])} for "
                f"parameter '{param}'."
            )
        all_configuration = parameter_grid(param_grid)
    else:
        for idx, param in enumerate(param_grid):
            assert isinstance(param, dict), ParameterError(
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
                metric,
                cv,
                pos_label,
                cutoff,
                True,
                training_score,
                tqdm=False,
            )
            if training_score:
                keys = [elem for elem in current_cv[0].values]
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
                keys = [elem for elem in current_cv.values]
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
            elif not (skip_error):
                raise (e)
    if not (data):
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
                {"parameters": [], "avg_score": [], "avg_time": [], "score_std": [],}
            )
    reverse = reverse_score(metric)
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


@save_verticapy_logs
def parameter_grid(param_grid: dict):
    """
Generates the list of the different combinations of input parameters.

Parameters
----------
param_grid: dict
    Dictionary of parameters.

Returns
-------
list of dict
    List of the different combinations.
    """
    return [
        dict(zip(param_grid.keys(), values))
        for values in itertools.product(*param_grid.values())
    ]


@save_verticapy_logs
def plot_acf_pacf(
    vdf: vDataFrame,
    column: str,
    ts: str,
    by: SQLColumns = [],
    p: Union[int, list] = 15,
    **style_kwds,
):
    """
Draws the ACF and PACF Charts.

Parameters
----------
vdf: vDataFrame
    Input vDataFrame.
column: str
    Response column.
ts: str
    vcolumn used as timeline. It will be to use to order the data. 
    It can be a numerical or type date like (date, datetime, timestamp...) 
    vcolumn.
by: list, optional
    vcolumns used in the partition.
p: int/list, optional
    Int equals to the maximum number of lag to consider during the computation
    or List of the different lags to include during the computation.
    p must be positive or a list of positive integers.
**style_kwds
    Any optional parameter to pass to the Matplotlib functions.

Returns
-------
TableSample
    An object containing the result. For more information, see
    utilities.TableSample.
    """
    if isinstance(by, str):
        by = [by]
    tmp_style = {}
    for elem in style_kwds:
        if elem not in ("color", "colors"):
            tmp_style[elem] = style_kwds[elem]
    if "color" in style_kwds:
        color = style_kwds["color"]
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
    ax1.scatter(x, y, **updated_dict(param, tmp_style))
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
    ax2.scatter(x, y, **updated_dict(param, tmp_style))
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


@save_verticapy_logs
def randomized_search_cv(
    estimator,
    input_relation: SQLRelation,
    X: list,
    y: str,
    metric: str = "auto",
    cv: int = 3,
    pos_label: PythonScalar = None,
    cutoff: float = -1,
    nbins: int = 1000,
    lmax: int = 4,
    optimized_grid: int = 1,
    print_info: bool = True,
):
    """
Computes the K-Fold randomized search of an estimator.

Parameters
----------
estimator: object
    Vertica estimator with a fit method.
input_relation: SQLRelation
    Relation to use to train the model.
X: list
    List of the predictor columns.
y: str
    Response Column.
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
pos_label: PythonScalar, optional
    The main class to be considered as positive (classification only).
cutoff: float, optional
    The model cutoff (classification only).
nbins: int, optional
    Number of bins used to compute the different parameters categories.
lmax: int, optional
    Maximum length of each parameter list.
optimized_grid: int, optional
    If set to 0, the randomness is based on the input parameters.
    If set to 1, the randomness is limited to some parameters while others
    are picked based on a default grid.
    If set to 2, there is no randomness and a default grid is returned.
print_info: bool, optional
    If set to True, prints the model information at each step.

Returns
-------
TableSample
    An object containing the result. For more information, see
    utilities.TableSample.
    """
    param_grid = gen_params_grid(estimator, nbins, len(X), lmax, optimized_grid)
    return grid_search_cv(
        estimator,
        param_grid,
        input_relation,
        X,
        y,
        metric,
        cv,
        pos_label,
        cutoff,
        True,
        "no_print",
        print_info,
    )


@save_verticapy_logs
def validation_curve(
    estimator,
    param_name: str,
    param_range: list,
    input_relation: SQLRelation,
    X: list,
    y: str,
    metric: str = "auto",
    cv: int = 3,
    pos_label: PythonScalar = None,
    cutoff: float = -1,
    std_coeff: float = 1,
    ax=None,
    **style_kwds,
):
    """
Draws the validation curve.

Parameters
----------
estimator: object
    Vertica estimator with a fit method.
param_name: str
    Parameter name.
param_range: list
    Parameter Range.
input_relation: SQLRelation
    Relation to use to train the model.
X: list
    List of the predictor columns.
y: str
    Response Column.
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
pos_label: PythonScalar, optional
    The main class to be considered as positive (classification only).
cutoff: float, optional
    The model cutoff (classification only).
std_coeff: float, optional
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
    X = [elem[0] for elem in gs_result_final]
    Y = [
        [
            [elem[2] - std_coeff * elem[4] for elem in gs_result_final],
            [elem[2] for elem in gs_result_final],
            [elem[2] + std_coeff * elem[4] for elem in gs_result_final],
        ],
        [
            [elem[1] - std_coeff * elem[3] for elem in gs_result_final],
            [elem[1] for elem in gs_result_final],
            [elem[1] + std_coeff * elem[3] for elem in gs_result_final],
        ],
    ]
    result = TableSample(
        {
            param_name: X,
            "training_score_lower": Y[0][0],
            "training_score": Y[0][1],
            "training_score_upper": Y[0][2],
            "test_score_lower": Y[1][0],
            "test_score": Y[1][1],
            "test_score_upper": Y[1][2],
        }
    )
    range_curve(X, Y, param_name, metric, ax, ["train", "test"], **style_kwds)
    return result
