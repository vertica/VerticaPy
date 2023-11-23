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
import itertools
import math
import random
from typing import Any

from verticapy._utils._sql._collect import save_verticapy_logs

import verticapy.machine_learning.vertica as vml
from verticapy.machine_learning.vertica.base import VerticaModel


@save_verticapy_logs
def parameter_grid(param_grid: dict) -> list[dict]:
    """
    Generates a list of the different combinations
    of input parameters.

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
def gen_params_grid(
    estimator: VerticaModel,
    nbins: int = 10,
    max_nfeatures: int = 3,
    lmax: int = -1,
    optimized_grid: int = 0,
) -> dict[str, Any]:
    """
    Generates the estimator grid.

    Parameters
    ----------
    estimator: object
        Vertica   estimator  with  a  fit   method.
    nbins: int, optional
        Number of bins used to discretize numerical
        features.
    max_nfeatures: int, optional
        Maximum number of  features used to compute
        Random Forest, PCA...
    lmax: int, optional
        Maximum length of the parameter grid.
    optimized_grid: int, optional
        If set to 0, the randomness is based on the
        input parameters.
        If set to 1,  the randomness is limited  to
        some  parameters  while others  are  picked
        based on a default grid.
        If set  to 2, there is no  randomness and a
        default grid is returned.

    Returns
    -------
    dict
        Dictionary of parameters.
    """
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
                "min_info_gain": {
                    "type": float,
                    "range": [0.0, 0.1],
                    "nbins": nbins,
                },
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
                "learning_rate": {
                    "type": float,
                    "range": [0.0, 1.0],
                    "nbins": nbins,
                },
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
