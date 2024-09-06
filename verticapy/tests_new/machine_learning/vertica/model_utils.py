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
from itertools import chain
import sys

import numpy as np
import pytest
import sklearn.cluster as skl_cluster
import sklearn.ensemble as skl_ensemble
import sklearn.linear_model as skl_linear_model
import sklearn.tree as skl_tree
import sklearn.dummy as skl_dummy
from sklearn.preprocessing import LabelEncoder
import sklearn.svm as skl_svm
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb

import verticapy as vp
from verticapy.connection import current_cursor
import verticapy.machine_learning.vertica as vpy_linear_model
import verticapy.machine_learning.vertica.cluster as vpy_cluster
import verticapy.machine_learning.vertica.ensemble as vpy_ensemble
import verticapy.machine_learning.vertica.svm as vpy_svm
import verticapy.machine_learning.vertica.tree as vpy_tree
import verticapy.machine_learning.vertica.tsa as vpy_tsa
from verticapy.tests_new.machine_learning.vertica import (
    REGRESSION_MODELS,
    CLASSIFICATION_MODELS,
    TIMESERIES_MODELS,
    CLUSTER_MODELS,
)

le = LabelEncoder()

if sys.version_info < (3, 12):
    import tensorflow as tf


@pytest.fixture(autouse=True)
def set_plotting_lib():
    vp.set_option("plotting_lib", "matplotlib")
    yield
    vp.set_option("plotting_lib", "plotly")


def get_model_attributes(model_class):
    """
    getter function to get vertica model attributes
    """
    attributes_map = {
        **dict.fromkeys(
            ["RandomForestRegressor", "DecisionTreeRegressor", "DummyTreeRegressor"],
            [
                "n_estimators_",
                "trees_",
                "features_importance_",
                "features_importance_trees_",
            ],
        ),
        **dict.fromkeys(
            ["XGBRegressor"],
            [
                "n_estimators_",
                "eta_",
                "mean_",
                "trees_",
                "features_importance_",
                "features_importance_trees_",
            ],
        ),
        **dict.fromkeys(
            ["RandomForestClassifier", "DecisionTreeClassifier", "DummyTreeClassifier"],
            [
                "n_estimators_",
                "classes_",
                "trees_",
                "features_importance_",
                "features_importance_trees_",
            ],
        ),
        **dict.fromkeys(
            ["XGBClassifier"],
            [
                "n_estimators_",
                "classes_",
                "eta_",
                "logodds_",
                "trees_",
                "features_importance_",
                "features_importance_trees_",
            ],
        ),
        **dict.fromkeys(
            ["AR"], ["phi_", "intercept_", "features_importance_", "mse_", "n_"]
        ),
        **dict.fromkeys(["MA"], ["theta_", "mu_", "mean_", "mse_", "n_"]),
        **dict.fromkeys(
            ["ARMA"], ["phi_", "theta_", "mean_", "features_importance_", "mse_", "n_"]
        ),
        **dict.fromkeys(
            ["ARIMA"], ["phi_", "theta_", "mean_", "features_importance_", "mse_", "n_"]
        ),
        **dict.fromkeys(
            ["KMeans"],
            [
                "clusters_",
                "p_",
                "between_cluster_ss_",
                "total_ss_",
                "total_within_cluster_ss_",
                "elbow_score_",
                "converged_",
            ],
        ),
    }

    return attributes_map.get(
        model_class, ["coef_", "intercept_", "features_importance_"]
    )


def get_train_function(model_class):
    """
    getter function to get vertica train function name
    """
    train_func_map = {
        **dict.fromkeys(
            ["RandomForestRegressor", "DecisionTreeRegressor", "DummyTreeRegressor"],
            "RF_REGRESSOR",
        ),
        **dict.fromkeys(
            ["RandomForestClassifier", "DecisionTreeClassifier", "DummyTreeClassifier"],
            "RF_CLASSIFIER",
        ),
        **dict.fromkeys(["XGBRegressor"], "XGB_REGRESSOR"),
        **dict.fromkeys(["XGBClassifier"], "XGB_CLASSIFIER"),
        **dict.fromkeys(["LinearSVR"], "SVM_REGRESSOR"),
        **dict.fromkeys(["LinearSVR"], "SVM_CLASSIFIER"),
        **dict.fromkeys(["PoissonRegressor"], "POISSON_REGRESSION"),
        **dict.fromkeys(["AR"], "AUTOREGRESSOR"),
        **dict.fromkeys(["MA"], "MOVING_AVERAGE"),
        **dict.fromkeys(["ARMA", "ARIMA"], "ARIMA"),
        **dict.fromkeys(
            ["Ridge", "Lasso", "ElasticNet", "LinearRegression"],
            ["LINEAR_REG", "LINEAR_REGRESSION"],
        ),
        **dict.fromkeys(["KMeans"], "KMEANS"),
    }

    return train_func_map.get(model_class, None)


def get_predict_function(model_class):
    """
    getter function to get vertica predict function name
    """
    pred_func_map = {
        **dict.fromkeys(
            ["RandomForestRegressor", "DecisionTreeRegressor", "DummyTreeRegressor"],
            "PREDICT_RF_REGRESSOR",
        ),
        **dict.fromkeys(
            ["RandomForestClassifier", "DecisionTreeClassifier", "DummyTreeClassifier"],
            "PREDICT_RF_CLASSIFIER",
        ),
        **dict.fromkeys(["XGBRegressor"], "PREDICT_XGB_REGRESSOR"),
        **dict.fromkeys(["XGBClassifier"], "PREDICT_XGB_CLASSIFIER"),
        **dict.fromkeys(["LinearSVR"], "PREDICT_SVM_REGRESSOR"),
        **dict.fromkeys(["PoissonRegressor"], "PREDICT_POISSON_REG"),
        **dict.fromkeys(["AR"], "PREDICT_AUTOREGRESSOR"),
        **dict.fromkeys(["MA"], "PREDICT_MOVING_AVERAGE"),
        **dict.fromkeys(["ARMA", "ARIMA"], "PREDICT_ARIMA"),
        **dict.fromkeys(
            ["Ridge", "Lasso", "ElasticNet", "LinearRegression"],
            "PREDICT_LINEAR_REG",
        ),
        **dict.fromkeys(["KMeans"], "APPLY_KMEANS"),
    }

    return pred_func_map.get(model_class, "PREDICT_LINEAR_REG")


def get_function_name(model_class):
    """
    getter function - get function name
    """
    function_name_map = {
        **dict.fromkeys(
            [
                "LinearRegression",
                "Ridge",
                "Lasso",
                "ElasticNet",
                "LinearSVR",
                "PoissonRegressor",
            ],
            {
                "vpy": ["vpy_tree_classifier", "vpy_linear", "vpy_linear"],
                "py": ["py_regressor", "py_linear", "py_regressor"],
            },
        ),
        **dict.fromkeys(
            [
                "RandomForestRegressor",
                "DecisionTreeRegressor",
                "DummyTreeRegressor",
                "XGBRegressor",
            ],
            {
                "vpy": [
                    "vpy_tree_regressor",
                    "vpy_tree_regressor",
                    "vpy_tree_regressor",
                ],
                "py": ["py_regressor", "py_linear", "py_regressor"],
            },
        ),
        **dict.fromkeys(
            CLASSIFICATION_MODELS,
            {
                "vpy": [
                    "vpy_tree_classifier",
                    "vpy_tree_classifier",
                    "vpy_tree_classifier",
                ],
                "py": ["py_classifier", "py_linear", "py_classifier"],
            },
        ),
        **dict.fromkeys(
            TIMESERIES_MODELS,
            {
                "vpy": ["vpy_tree_regressor", "vpy_timeseries", "vpy_timeseries"],
                "py": ["py_timeseries", "py_timeseries", "py_timeseries"],
            },
        ),
        **dict.fromkeys(
            CLUSTER_MODELS,
            {
                "vpy": ["vpy_tree_classifier", "vpy_cluster", "vpy_cluster"],
                "py": ["py_cluster", "py_cluster", "py_cluster"],
            },
        ),
        **dict.fromkeys(
            ["TENSORFLOW", "TF"],
            {"vpy": [None, None, None], "py": ["py_tf", "py_tf", "py_tf"]},
        ),
    }
    return function_name_map.get(model_class, None)


def get_model_class(model_class):
    """
    getter function - get model class
    """
    model_class_map = {
        **dict.fromkeys(["LinearRegression"], LinearRegressionInitializer),
        **dict.fromkeys(["LinearSVR"], LinearSVRInitializer),
        **dict.fromkeys(["Ridge"], RidgeInitializer),
        **dict.fromkeys(["Lasso"], LassoInitializer),
        **dict.fromkeys(["ElasticNet"], ElasticNetInitializer),
        **dict.fromkeys(["PoissonRegressor"], PoissonRegressorInitializer),
        **dict.fromkeys(["XGBRegressor", "XGBClassifier"], XGBInitializer),
        **dict.fromkeys(
            ["RandomForestRegressor", "RandomForestClassifier"], RandomForestInitializer
        ),
        **dict.fromkeys(
            ["DecisionTreeRegressor", "DecisionTreeClassifier"], DecisionTreeInitializer
        ),
        **dict.fromkeys(
            ["DummyTreeRegressor", "DummyTreeClassifier"], DummyTreeInitializer
        ),
        **dict.fromkeys(["AR"], ARInitializer),
        **dict.fromkeys(["MA"], MAInitializer),
        **dict.fromkeys(["ARMA"], ARMAInitializer),
        **dict.fromkeys(["ARIMA"], ARIMAInitializer),
        **dict.fromkeys(["KMeans"], KMeansInitializer),
        **dict.fromkeys(["TENSORFLOW", "TF"], TFInitializer),
    }
    return model_class_map[model_class]


def get_xy(model_class):
    """
    getter function - get X and y for a given model class
    """
    xy_map = {
        **dict.fromkeys(
            REGRESSION_MODELS,
            {
                "X": ["citric_acid", "residual_sugar", "alcohol"],
                "y": ["quality"],
                "dataset_name": "winequality",
            },
        ),
        **dict.fromkeys(
            CLASSIFICATION_MODELS,
            {"X": ["age", "fare", "sex"], "y": ["survived"], "dataset_name": "titanic"},
        ),
        **dict.fromkeys(
            TIMESERIES_MODELS,
            {"X": ["date"], "y": ["passengers"], "dataset_name": "airline"},
        ),
        **dict.fromkeys(
            CLUSTER_MODELS,
            {
                "X": ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
                "y": [None],
                "dataset_name": "iris",
            },
        ),
        **dict.fromkeys(
            ["TENSORFLOW", "TF"], {"X": ["NA"], "y": ["NA"], "dataset_name": "NA"}
        ),
    }
    return xy_map.get(model_class, None)


class LinearRegressionInitializer:
    """
    Model initializer class for Linear Regression
    """

    def __init__(self, datasetup_instance, **kwargs):
        self.overwrite_model = kwargs.get("overwrite_model", False)
        self.tol = kwargs.get("tol", 1e-6)
        self.max_iter = kwargs.get("max_iter", 100)
        self.solver = kwargs.get("solver", "newton")
        self.fit_intercept = (
            kwargs.get("fit_intercept") if kwargs.get("fit_intercept") else True
        )
        self.datasetup_instance = datasetup_instance
        self.model_name = f"vpy_model_{self.datasetup_instance.model_class}"

    def vpy(self):
        """
        Model initializer function for vertica Linear Regression model
        """
        model = getattr(vpy_linear_model, self.datasetup_instance.model_class)(
            name=f"{self.datasetup_instance.schema_name}.{self.model_name}",
            overwrite_model=self.overwrite_model,
            tol=self.tol,
            max_iter=self.max_iter,
            solver=self.solver,
            fit_intercept=self.fit_intercept,
        )
        print(f"VerticaPy Training Parameters: {model.get_params()}")
        # drop if model exists with same name
        model.drop()
        return model

    def py(self):
        """
        Model initializer function for python Linear Regression model
        """
        model = getattr(skl_linear_model, self.datasetup_instance.model_class)(
            fit_intercept=self.fit_intercept
        )
        print(f"Python Training Parameters: {model.get_params()}")
        return model


class RidgeInitializer:
    """
    Model initializer class for Ridge Regression model
    """

    def __init__(self, datasetup_instance, **kwargs):
        self.overwrite_model = kwargs.get("overwrite_model", False)
        self.tol = kwargs.get("tol", 1e-6)
        self.C = kwargs.get("c", 1.0)
        self.max_iter = kwargs.get("max_iter", 100)
        self.solver = kwargs.get("solver", "newton")
        self.fit_intercept = (
            kwargs.get("fit_intercept") if kwargs.get("fit_intercept") else True
        )
        self.datasetup_instance = datasetup_instance
        self.model_name = f"vpy_model_{self.datasetup_instance.model_class}"

    def vpy(self):
        """
        Model initializer function for vertica Ridge model
        """
        model = getattr(vpy_linear_model, self.datasetup_instance.model_class)(
            name=f"{self.datasetup_instance.schema_name}.{self.model_name}",
            overwrite_model=self.overwrite_model,
            tol=self.tol,
            C=self.C,
            max_iter=self.max_iter,
            solver=self.solver,
            fit_intercept=self.fit_intercept,
        )
        print(f"VerticaPy Training Parameters: {model.get_params()}")

        # drop if model exists with same name
        model.drop()
        return model

    def py(self):
        """
        Model initializer function for python Ridge model
        """
        model = getattr(skl_linear_model, self.datasetup_instance.model_class)(
            fit_intercept=self.fit_intercept
        )
        print(f"Python Training Parameters: {model.get_params()}")
        return model


class LassoInitializer:
    """
    Model initializer class for Lasso model
    """

    def __init__(self, datasetup_instance, **kwargs):
        self.overwrite_model = kwargs.get("overwrite_model", False)
        self.tol = kwargs.get("tol", 1e-6)
        self.C = kwargs.get("c", 1.0)
        self.max_iter = kwargs.get("max_iter", 100)
        self.solver = kwargs.get("solver", "cgd")
        self.fit_intercept = (
            kwargs.get("fit_intercept") if kwargs.get("fit_intercept") else True
        )
        self.datasetup_instance = datasetup_instance
        self.model_name = f"vpy_model_{self.datasetup_instance.model_class}"

    def vpy(self):
        """
        Model initializer function for vertica Lasso model
        """
        model = getattr(vpy_linear_model, self.datasetup_instance.model_class)(
            name=f"{self.datasetup_instance.schema_name}.{self.model_name}",
            overwrite_model=self.overwrite_model,
            tol=self.tol,
            C=self.C,
            max_iter=self.max_iter,
            solver=self.solver,
            fit_intercept=self.fit_intercept,
        )
        print(f"VerticaPy Training Parameters: {model.get_params()}")

        # drop if model exists with same name
        model.drop()
        return model

    def py(self):
        """
        Model initializer function for python Lasso model
        """
        model = getattr(skl_linear_model, self.datasetup_instance.model_class)(
            fit_intercept=self.fit_intercept
        )
        print(f"Python Training Parameters: {model.get_params()}")
        return model


class ElasticNetInitializer:
    """
    Model initializer class for ElasticNet model
    """

    def __init__(self, datasetup_instance, **kwargs):
        self.overwrite_model = kwargs.get("overwrite_model", False)
        self.tol = kwargs.get("tol", 1e-6)
        self.C = kwargs.get("c", 1.0)
        self.max_iter = kwargs.get("max_iter", 100)
        self.solver = kwargs.get("solver", "cgd")
        self.l1_ratio = kwargs.get("l1_ratio", 0.5)
        self.fit_intercept = (
            kwargs.get("fit_intercept") if kwargs.get("fit_intercept") else True
        )
        self.datasetup_instance = datasetup_instance
        self.model_name = f"vpy_model_{self.datasetup_instance.model_class}"

    def vpy(self):
        """
        Model initializer function for vertica ElasticNet model
        """
        model = getattr(vpy_linear_model, self.datasetup_instance.model_class)(
            name=f"{self.datasetup_instance.schema_name}.{self.model_name}",
            overwrite_model=self.overwrite_model,
            tol=self.tol,
            C=self.C,
            max_iter=self.max_iter,
            solver=self.solver,
            l1_ratio=self.l1_ratio,
            fit_intercept=self.fit_intercept,
        )
        print(f"VerticaPy Training Parameters: {model.get_params()}")

        # drop if model exists with same name
        model.drop()
        return model

    def py(self):
        """
        Model initializer function for python ElasticNet model
        """
        model = getattr(skl_linear_model, self.datasetup_instance.model_class)(
            fit_intercept=self.fit_intercept
        )
        print(f"Python Training Parameters: {model.get_params()}")
        return model


class LinearSVRInitializer:
    """
    Model initializer class for LinearSVR model
    """

    def __init__(self, datasetup_instance, **kwargs):
        self.overwrite_model = kwargs.get("overwrite_model", False)
        self.tol = kwargs.get("tol", 1e-4)
        self.C = kwargs.get("c", 1.0)
        self.intercept_scaling = kwargs.get("intercept_scaling", 1.0)
        self.intercept_mode = kwargs.get("intercept_mode", "regularized")
        self.acceptable_error_margin = kwargs.get("acceptable_error_margin", 0.1)
        self.max_iter = kwargs.get("max_iter", 100)
        self.py_fit_intercept = kwargs.get("py_fit_intercept", True)
        self.datasetup_instance = datasetup_instance
        self.model_name = f"vpy_model_{self.datasetup_instance.model_class}"

    def vpy(self):
        """
        Model initializer function for vertica LinearSVR model
        """
        model = getattr(vpy_svm, self.datasetup_instance.model_class)(
            name=f"{self.datasetup_instance.schema_name}.{self.model_name}",
            overwrite_model=self.overwrite_model,
            tol=self.tol,
            C=self.C,
            intercept_scaling=self.intercept_scaling,
            intercept_mode=self.intercept_mode,
            acceptable_error_margin=self.acceptable_error_margin,
            max_iter=self.max_iter,
        )
        print(f"VerticaPy Training Parameters: {model.get_params()}")

        # drop if model exists with same name
        model.drop()
        return model

    def py(self):
        """
        Model initializer function for python LinearSVR model
        """
        model = getattr(skl_svm, self.datasetup_instance.model_class)(
            fit_intercept=self.py_fit_intercept
        )
        print(f"Python Training Parameters: {model.get_params()}")
        return model


class PoissonRegressorInitializer:
    """
    Model initializer class for PoissonRegressor model
    """

    def __init__(self, datasetup_instance, **kwargs):
        self.overwrite_model = kwargs.get("overwrite_model", False)
        self.penalty = kwargs.get("penalty", "l2")
        self.tol = kwargs.get("tol", 1e-6)
        self.C = kwargs.get("c", 1.0)
        self.max_iter = kwargs.get("max_iter", 100)
        self.solver = kwargs.get("solver", "newton")
        self.fit_intercept = (
            kwargs.get("fit_intercept") if kwargs.get("fit_intercept") else True
        )
        self.datasetup_instance = datasetup_instance
        self.model_name = f"vpy_model_{self.datasetup_instance.model_class}"

    def vpy(self):
        """
        Model initializer function for vertica PoissonRegressor model
        """
        model = getattr(vpy_linear_model, self.datasetup_instance.model_class)(
            name=f"{self.datasetup_instance.schema_name}.{self.model_name}",
            overwrite_model=self.overwrite_model,
            penalty=self.penalty,
            tol=self.tol,
            C=self.C,
            max_iter=self.max_iter,
            solver=self.solver,
            fit_intercept=self.fit_intercept,
        )
        print(f"VerticaPy Training Parameters: {model.get_params()}")

        # drop if model exists with same name
        model.drop()
        return model

    def py(self):
        """
        Model initializer function for python PoissonRegressor model
        """
        model = getattr(skl_linear_model, self.datasetup_instance.model_class)(
            alpha=0.00005, fit_intercept=self.fit_intercept, tol=self.tol
        )
        print(f"Python Training Parameters: {model.get_params()}")
        return model


class RandomForestInitializer:
    """
    Model initializer class for RandomForest model
    """

    def __init__(self, datasetup_instance, **kwargs):
        self.overwrite_model = kwargs.get("overwrite_model", False)
        self.ntree = kwargs.get("n_estimators", 10)
        self.mtry = kwargs.get("max_features", 2)
        self.max_breadth = kwargs.get("max_leaf_nodes", 10)
        self.sampling_size = kwargs.get("sample", 0.632)
        self.max_depth = kwargs.get("max_depth", 10)
        self.min_leaf_size = kwargs.get("min_samples_leaf", 1)
        self.min_info_gain = (
            kwargs.get("min_info_gain") if kwargs.get("min_info_gain") else 0.0
        )
        self.nbins = kwargs.get("nbins") if kwargs.get("nbins") else 32
        self.datasetup_instance = datasetup_instance
        self.model_name = f"vpy_model_{self.datasetup_instance.model_class}"

    def vpy(self):
        """
        Model initializer function for vertica RandomForest model
        """
        model = getattr(vpy_tree, self.datasetup_instance.model_class)(
            name=f"{self.datasetup_instance.schema_name}.{self.model_name}",
            overwrite_model=self.overwrite_model,
            n_estimators=self.ntree,
            max_features=self.mtry,
            max_leaf_nodes=self.max_breadth,
            sample=self.sampling_size,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_leaf_size,
            min_info_gain=self.min_info_gain,
            nbins=self.nbins,
        )
        print(f"VerticaPy Training Parameters: {model.get_params()}")

        # drop if model exists with same name
        model.drop()

        return model

    def py(self):
        """
        Model initializer function for python RandomForest model
        """
        model = getattr(skl_ensemble, self.datasetup_instance.model_class)(
            n_estimators=self.ntree,
            max_features=self.mtry,
            max_leaf_nodes=self.max_breadth,
            max_samples=self.sampling_size,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_leaf_size,
            random_state=1,
        )
        print(f"Python Training Parameters: {model.get_params()}")
        return model


class DecisionTreeInitializer:
    """
    Model initializer class for DecisionTree model
    """

    def __init__(self, datasetup_instance, **kwargs):
        self.overwrite_model = kwargs.get("overwrite_model", False)
        self.ntree = 1
        self.mtry = kwargs.get("max_features", 2)
        self.max_breadth = kwargs.get("max_leaf_nodes", 10)
        self.max_depth = kwargs.get("max_depth", 10)
        self.min_leaf_size = self.sampling_size = kwargs.get("min_samples_leaf", 1)
        self.min_info_gain = (
            kwargs.get("min_info_gain") if kwargs.get("min_info_gain") else 0.0
        )
        self.nbins = kwargs.get("nbins") if kwargs.get("nbins") else 32
        self.datasetup_instance = datasetup_instance
        self.model_name = f"vpy_model_{self.datasetup_instance.model_class}"

    def vpy(self):
        """
        Model initializer function for vertica DecisionTree model
        """
        model = getattr(vpy_tree, self.datasetup_instance.model_class)(
            name=f"{self.datasetup_instance.schema_name}.{self.model_name}",
            overwrite_model=self.overwrite_model,
            max_features=self.mtry,
            max_leaf_nodes=self.max_breadth,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_leaf_size,
            min_info_gain=self.min_info_gain,
            nbins=self.nbins,
        )
        print(f"VerticaPy Training Parameters: {model.get_params()}")

        # drop if model exists with same name
        model.drop()

        return model

    def py(self):
        """
        Model initializer function for python DecisionTree model
        """
        model = getattr(skl_tree, self.datasetup_instance.model_class)(
            max_features=self.mtry,
            max_leaf_nodes=self.max_breadth,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_leaf_size,
            random_state=1,
        )
        print(f"Python Training Parameters: {model.get_params()}")
        return model


class DummyTreeInitializer:
    """
    Model initializer class for DummyTree model
    """

    def __init__(self, datasetup_instance, **kwargs):
        self.overwrite_model = kwargs.get("overwrite_model", False)
        self.ntree = 1
        self.mtry = kwargs.get("max_features", 2)
        self.max_breadth = kwargs.get("max_leaf_nodes", 1000000000)
        self.max_depth = kwargs.get("max_depth", 100)
        self.min_leaf_size = self.sampling_size = kwargs.get("min_samples_leaf", 1)
        self.nbins = kwargs.get("nbins") if kwargs.get("nbins") else 1000
        self.datasetup_instance = datasetup_instance
        self.model_name = f"vpy_model_{self.datasetup_instance.model_class}"

    def vpy(self):
        """
        Model initializer function for vertica DummyTree model
        """
        model = getattr(vpy_tree, self.datasetup_instance.model_class)(
            name=f"{self.datasetup_instance.schema_name}.{self.model_name}",
            overwrite_model=self.overwrite_model,
        )
        print(f"VerticaPy Training Parameters: {model.get_params()}")

        # drop if model exists with same name
        model.drop()
        return model

    def py(self):
        """
        Model initializer function for python DummyTree model
        """
        model = getattr(
            skl_dummy,
            "DummyRegressor"
            if self.datasetup_instance.model_class == "DummyTreeRegressor"
            else "DummyClassifier",
        )()
        print(f"Python Training Parameters: {model.get_params()}")
        return model


class XGBInitializer:
    """
    Model initializer class for XGBoost model
    """

    def __init__(self, datasetup_instance, **kwargs):
        self.overwrite_model = kwargs.get("overwrite_model", False)
        self.max_ntree = kwargs.get("max_ntree", 10)
        self.max_depth = kwargs.get("max_depth", 10)
        self.nbins = kwargs.get("nbins", 150)
        self.split_proposal_method = kwargs.get("split_proposal_method", "'global'")
        self.tol = kwargs.get("tol", 0.001)
        self.learning_rate = kwargs.get("learning_rate", 0.1)
        self.min_split_loss = kwargs.get("min_split_loss", 0.0)
        self.weight_reg = kwargs.get("weight_reg", 0.0)
        self.sample = kwargs.get("sample", 1.0)
        self.col_sample_by_tree = kwargs.get("col_sample_by_tree", 1.0)
        self.col_sample_by_node = kwargs.get("col_sample_by_node", 1.0)
        self.datasetup_instance = datasetup_instance
        self.model_name = f"vpy_model_{self.datasetup_instance.model_class}"

    def vpy(self):
        """
        Model initializer function for vertica XGBoost model
        """
        model = getattr(vpy_ensemble, self.datasetup_instance.model_class)(
            name=f"{self.datasetup_instance.schema_name}.{self.model_name}",
            overwrite_model=self.overwrite_model,
            max_ntree=self.max_ntree,
            max_depth=self.max_depth,
            nbins=self.nbins,
            split_proposal_method=self.split_proposal_method,
            tol=self.tol,
            learning_rate=self.learning_rate,
            min_split_loss=self.min_split_loss,
            weight_reg=self.weight_reg,
            sample=self.sample,
            col_sample_by_tree=self.col_sample_by_tree,
            col_sample_by_node=self.col_sample_by_node,
        )
        print(f"VerticaPy Training Parameters: {model.get_params()}")
        # drop if model exists with same name
        model.drop()

        return model

    def py(self):
        """
        Model initializer function for python XGBoost model
        """
        model = getattr(xgb, self.datasetup_instance.model_class)(
            n_estimators=self.max_ntree,
            max_depth=self.max_depth,
            max_bin=self.nbins,
            random_state=1,
            tree_method="exact",
        )
        print(f"Python Training Parameters: {model.get_params()}")
        return model


class ARInitializer:
    """
    Model initializer class for Autoregressor model
    """

    def __init__(self, datasetup_instance, **kwargs):
        self.overwrite_model = kwargs.get("overwrite_model", False)
        self.p = kwargs.get("p", 3)
        self.method = kwargs.get("method", "ols")
        self.penalty = kwargs.get("", "none")
        self.C = kwargs.get("C", 1.0)
        self.missing = kwargs.get("missing", "linear_interpolation")
        # self.compute_mse = kwargs.get("compute_mse", True)
        self.npredictions = kwargs.get("npredictions", None)
        self.datasetup_instance = datasetup_instance
        self.model_name = f"vpy_model_{self.datasetup_instance.model_class}"

    def vpy(self):
        """
        Model initializer function for vertica Autoregressor model
        """
        model = getattr(vpy_tsa, self.datasetup_instance.model_class)(
            name=f"{self.datasetup_instance.schema_name}.{self.model_name}",
            overwrite_model=self.overwrite_model,
            p=self.p,
            method=self.method,
            penalty=self.penalty,
            C=self.C,
            missing=self.missing,
            # compute_mse=self.compute_mse,
        )
        print(f"VerticaPy Training Parameters: {model.get_params()}")

        # drop if model exists with same name
        model.drop()

        return model

    def py(self):
        """
        Model initializer function for python Autoregressor model
        """
        model = ARIMA(self.datasetup_instance.py_dataset, order=(self.p, 0, 0))

        return model


class MAInitializer:
    """
    Model initializer class for MovingAverage model
    """

    def __init__(self, datasetup_instance, **kwargs):
        self.overwrite_model = kwargs.get("overwrite_model", False)
        self.q = kwargs.get("q", 1)
        self.penalty = kwargs.get("penalty", "none")
        self.C = kwargs.get("C", 1.0)
        self.missing = kwargs.get("missing", "linear_interpolation")
        self.npredictions = kwargs.get("npredictions", None)
        self.datasetup_instance = datasetup_instance
        self.model_name = f"vpy_model_{self.datasetup_instance.model_class}"

    def vpy(self):
        """
        Model initializer function for vertica MovingAverage model
        """
        model = getattr(vpy_tsa, self.datasetup_instance.model_class)(
            name=f"{self.datasetup_instance.schema_name}.{self.model_name}",
            overwrite_model=self.overwrite_model,
            q=self.q,
            penalty=self.penalty,
            C=self.C,
            missing=self.missing,
        )
        print(f"VerticaPy Training Parameters: {model.get_params()}")

        # drop if model exists with same name
        model.drop()

        return model

    def py(self):
        """
        Model initializer function for python MovingAverage model
        """
        model = ARIMA(
            self.datasetup_instance.py_dataset,
            order=(0, 0, self.q),
        )

        return model


class ARMAInitializer:
    """
    Model initializer class for AutoRegressor MovingAverage model (d=0)
    """

    def __init__(self, datasetup_instance, **kwargs):
        self.overwrite_model = kwargs.get("overwrite_model", False)
        self.order = kwargs.get("order", (2, 1))
        self.tol = kwargs.get("tol", 1e-06)
        self.max_iter = kwargs.get("max_iter", 100)
        self.init = kwargs.get("init", "zero")
        self.missing = kwargs.get("missing", "linear_interpolation")
        # self.compute_mse = kwargs.get("compute_mse", True)
        self.npredictions = kwargs.get("npredictions", None)
        self.datasetup_instance = datasetup_instance
        self.model_name = f"vpy_model_{self.datasetup_instance.model_class}"

    def vpy(self):
        """
        Model initializer function for vertica AutoRegressor MovingAverage model (d=0) model
        """
        model = getattr(vpy_tsa, self.datasetup_instance.model_class)(
            name=f"{self.datasetup_instance.schema_name}.{self.model_name}",
            overwrite_model=self.overwrite_model,
            order=self.order,
            tol=self.tol,
            max_iter=self.max_iter,
            init=self.init,
            missing=self.missing,
            # compute_mse=self.compute_mse
        )
        print(f"VerticaPy Training Parameters: {model.get_params()}")

        # drop if model exists with same name
        model.drop()

        return model

    def py(self):
        """
        Model initializer function for python AutoRegressor MovingAverage model (d=0) model
        """
        order = self.order[:1] + (0,) + self.order[-1:]

        model = ARIMA(
            self.datasetup_instance.py_dataset,
            order=order,
        )

        return model


class ARIMAInitializer:
    """
    Model initializer class for AutoRegressor MovingAverage model (d>0)
    """

    def __init__(self, datasetup_instance, **kwargs):
        self.overwrite_model = kwargs.get("overwrite_model", False)
        self.order = kwargs.get("order", (2, 1, 1))
        self.tol = kwargs.get("tol", 1e-06)
        self.max_iter = kwargs.get("max_iter", 100)
        self.init = kwargs.get("init", "zero")
        self.missing = kwargs.get("missing", "linear_interpolation")
        # self.compute_mse = kwargs.get("compute_mse", True)
        self.npredictions = kwargs.get("npredictions", None)
        self.datasetup_instance = datasetup_instance
        self.model_name = f"vpy_model_{self.datasetup_instance.model_class}"

    def vpy(self):
        """
        Model initializer function for vertica AutoRegressor MovingAverage model (d>0) model
        """
        model = getattr(vpy_tsa, self.datasetup_instance.model_class)(
            name=f"{self.datasetup_instance.schema_name}.{self.model_name}",
            overwrite_model=self.overwrite_model,
            order=self.order,
            tol=self.tol,
            max_iter=self.max_iter,
            init=self.init,
            missing=self.missing,
            # compute_mse=self.compute_mse
        )
        print(f"VerticaPy Training Parameters: {model.get_params()}")

        # drop if model exists with same name
        model.drop()

        return model

    def py(self):
        """
        Model initializer function for python AutoRegressor MovingAverage model (d>0) model
        """
        model = ARIMA(
            self.datasetup_instance.py_dataset,
            order=self.order,
        )

        return model


class KMeansInitializer:
    """
    Model initializer class for Kmeans model
    """

    def __init__(self, datasetup_instance, **kwargs):
        self.overwrite_model = kwargs.get("overwrite_model", False)
        self.n_cluster = kwargs.get("n_cluster") if kwargs.get("n_cluster") else 8
        self.init = kwargs.get("init") if kwargs.get("init") else "kmeanspp"
        self.init_py = kwargs.get("init") if kwargs.get("init") else "k-means++"
        self.max_iter = kwargs.get("max_iter") if kwargs.get("max_iter") else 300
        self.tol = kwargs.get("tol") if kwargs.get("tol") else 1e-4
        self.datasetup_instance = datasetup_instance
        self.model_name = f"vpy_model_{self.datasetup_instance.model_class}"

    def vpy(self):
        """
        Model initializer function for vertica KMeans model
        """
        model = getattr(vpy_cluster, self.datasetup_instance.model_class)(
            name=f"{self.datasetup_instance.schema_name}.{self.model_name}",
            overwrite_model=self.overwrite_model,
            n_cluster=self.n_cluster,
            init=self.init,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        print(f"VerticaPy Training Parameters: {model.get_params()}")

        # drop if model exists with same name
        model.drop()
        return model

    def py(self):
        """
        Model initializer function for python KMeans model
        """
        model = getattr(skl_cluster, self.datasetup_instance.model_class)(
            n_clusters=self.n_cluster,
            init=self.init_py,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        return model


class TFInitializer:
    """
    Model initializer class for TensorFlow model
    """

    def __init__(self, datasetup_instance, **kwargs):
        self.datasetup_instance = datasetup_instance
        self.model_name = f"vpy_model_{self.datasetup_instance.model_class}"

    def py(self):
        """
        Model initializer function for python TensorFlow model
        """
        inputs = tf.keras.Input(shape=(28, 28, 1), name="image")
        x = tf.keras.layers.Conv2D(32, 5, activation="relu")(inputs)
        x = tf.keras.layers.MaxPooling2D(2)(x)
        x = tf.keras.layers.Conv2D(64, 5, activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(10, activation="softmax", name="OUTPUT")(x)
        model = tf.keras.Model(inputs, x)
        return model


class DataSetUp:
    """
    class for setting up datasets used in model training and prediction
    """

    def __init__(self, schema_name, model_name, model_class, X, y):
        self.schema_name = schema_name
        self.model_name = model_name
        self.model_class = model_class
        self.X = X if X else get_xy(self.model_class)["X"]
        self.y = y if y else get_xy(self.model_class)["y"]
        self.dataset_name = get_xy(self.model_class)["dataset_name"]
        self.py_dataset = None

    def vpy_tree_regressor(self):
        """
        Data setup function for vertica tree regression model(s)
        """
        # adding id column to winequality. id column is needed for seed parm for tree based model
        current_cursor().execute(
            f"ALTER TABLE {self.schema_name}.winequality ADD COLUMN IF NOT EXISTS id int"
        )
        seq_sql = f"CREATE SEQUENCE IF NOT EXISTS {self.schema_name}.sequence_auto_increment START 1"
        print(f"Sequence SQL: {seq_sql}")
        current_cursor().execute(seq_sql)
        current_cursor().execute(
            f"CREATE TABLE {self.schema_name}.winequality1 as select * from {self.schema_name}.winequality limit 0"
        )
        current_cursor().execute(
            f"insert into {self.schema_name}.winequality1 select fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol,quality,good,color, NEXTVAL('{self.schema_name}.sequence_auto_increment') from {self.schema_name}.winequality"
        )
        current_cursor().execute(f"DROP TABLE {self.schema_name}.winequality")
        current_cursor().execute(
            f"ALTER TABLE {self.schema_name}.winequality1 RENAME TO winequality"
        )

    def vpy_tree_classifier(self):
        """
        Data setup function for vertica tree classification model(s)
        """
        delete_sql = f"DELETE FROM {self.schema_name}.titanic WHERE AGE IS NULL OR FARE IS NULL OR SEX IS NULL OR SURVIVED IS NULL"
        print(f"Delete SQL: {delete_sql}")
        current_cursor().execute(delete_sql)

        # added to remove duplicate record with same name
        delete_name_sql = f"delete from {self.schema_name}.titanic where name in ('Kelly, Mr. James', 'Connolly, Miss. Kate')"
        print(f"Delete Name SQL: {delete_name_sql}")
        current_cursor().execute(delete_name_sql)

    def py_regressor(self):
        """
        Data setup function for python regression model(s)
        """
        winequality_pdf = vp.vDataFrame(
            f"select * from {self.schema_name}.{self.dataset_name}"
        ).to_pandas()
        winequality_pdf["citric_acid"] = winequality_pdf["citric_acid"].astype(float)
        winequality_pdf["residual_sugar"] = winequality_pdf["residual_sugar"].astype(
            float
        )

        self.py_dataset = winequality_pdf

    def py_classifier(self):
        """
        Data setup function for python classification model(s)
        """
        titanic_pdf = vp.vDataFrame(
            f"select * from {self.schema_name}.{self.dataset_name}"
        ).to_pandas()
        titanic_pdf.dropna(subset=["age", "fare", "sex", "survived"], inplace=True)
        titanic_pdf.drop(
            titanic_pdf[
                (titanic_pdf.name == "Kelly, Mr. James")
                | (titanic_pdf.name == "Connolly, Miss. Kate")
            ].index,
            inplace=True,
        )

        titanic_pdf["sex"] = le.fit_transform(titanic_pdf["sex"])
        titanic_pdf["age"] = titanic_pdf["age"].astype(float)
        titanic_pdf["fare"] = titanic_pdf["fare"].astype(float)

        self.py_dataset = titanic_pdf

    def py_timeseries(self):
        """
        Data setup function for python timeseries model(s)
        """
        airline_pdf = vp.vDataFrame(
            f"select * from {self.schema_name}.{self.dataset_name}"
        ).to_pandas()
        airline_pdf_ts = airline_pdf.set_index("date")

        self.py_dataset = airline_pdf_ts

    def py_cluster(self):
        """
        Data setup function for python clustering model(s)
        """
        iris_pdf = vp.vDataFrame(
            f"select * from {self.schema_name}.{self.dataset_name}"
        ).to_pandas()

        self.py_dataset = iris_pdf

    def py_tf(self):
        """
        Data setup function for python TensorFlow model(s)
        """
        nptype = np.float32

        (
            (train_eval_data, train_eval_labels),
            (
                test_data,
                test_labels,
            ),
        ) = tf.keras.datasets.mnist.load_data()

        train_eval_labels = np.asarray(train_eval_labels, dtype=nptype)
        train_eval_labels = tf.keras.utils.to_categorical(train_eval_labels)

        test_labels = np.asarray(test_labels, dtype=nptype)
        test_labels = tf.keras.utils.to_categorical(test_labels)

        #  Split the training data into two parts, training and evaluation
        data_split = np.split(train_eval_data, [55000])
        labels_split = np.split(train_eval_labels, [55000])

        train_data = data_split[0]
        train_labels = labels_split[0]

        eval_data = data_split[1]
        eval_labels = labels_split[1]

        print("Size of train_data: ", train_data.shape[0])
        print("Size of eval_data: ", eval_data.shape[0])
        print("Size of test_data: ", test_data.shape[0])

        train_data = train_data.reshape((55000, 28, 28, 1))
        eval_data = eval_data.reshape((5000, 28, 28, 1))
        test_data = test_data.reshape((10000, 28, 28, 1))

        self.py_dataset = [
            train_data,
            train_labels,
            eval_data,
            eval_labels,
            test_data,
            test_labels,
        ]


class TrainModel:
    """
    class for training a model
    """

    def __init__(self, model, datasetup_instance, model_instance):
        self.model = model
        self.datasetup_instance = datasetup_instance
        self.model_instance = model_instance

    def vpy_tree_regressor(self):
        """
        function to train vertica regression model(s)
        """
        _X = [f'"{i}"' for i in self.datasetup_instance.X]
        predictor_columns = ",".join(self.datasetup_instance.X)

        if self.datasetup_instance.model_class == "XGBRegressor":
            train_sql = f"SELECT xgb_regressor('{self.datasetup_instance.schema_name}.{self.datasetup_instance.model_name}', '{self.datasetup_instance.schema_name}.{self.datasetup_instance.dataset_name}', '{self.datasetup_instance.y[0]}', '{predictor_columns}' USING PARAMETERS exclude_columns='id', max_ntree={self.model_instance.max_ntree}, max_depth={self.model_instance.max_depth}, nbins={self.model_instance.nbins}, split_proposal_method={self.model_instance.split_proposal_method}, tol={self.model_instance.tol}, learning_rate={self.model_instance.learning_rate}, min_split_loss={self.model_instance.min_split_loss}, weight_reg={self.model_instance.weight_reg}, sample={self.model_instance.sample}, col_sample_by_tree={self.model_instance.col_sample_by_tree}, col_sample_by_node={self.model_instance.col_sample_by_node}, seed=1, id_column='id')"
        else:
            train_sql = f"SELECT rf_regressor('{self.datasetup_instance.schema_name}.{self.datasetup_instance.model_name}', '{self.datasetup_instance.schema_name}.{self.datasetup_instance.dataset_name}', '{self.datasetup_instance.y[0]}', '{predictor_columns}' USING PARAMETERS exclude_columns='id', ntree={self.model_instance.ntree}, mtry={self.model_instance.mtry}, max_breadth={self.model_instance.max_breadth}, sampling_size={self.model_instance.sampling_size}, max_depth={self.model_instance.max_depth}, min_leaf_size={self.model_instance.min_leaf_size}, nbins={self.model_instance.nbins}, seed=1, id_column='id')"

        print(f"Tree Regressor Train SQL: {train_sql}")
        current_cursor().execute(train_sql)

        self.model.input_relation = f"{self.datasetup_instance.schema_name}.{self.datasetup_instance.dataset_name}"
        self.model.test_relation = self.model.input_relation
        self.model.X = _X
        self.model.y = f'"{self.datasetup_instance.y[0]}"'
        self.model._compute_attributes()

        return self.model

    def vpy_tree_classifier(self):
        """
        function to train vertica tree classification model(s)
        """
        _X = [f'"{i}"' for i in self.datasetup_instance.X]
        predictor_columns = ",".join(self.datasetup_instance.X)

        if self.datasetup_instance.model_class == "XGBClassifier":
            train_sql = f"SELECT xgb_classifier('{self.datasetup_instance.schema_name}.{self.datasetup_instance.model_name}', '{self.datasetup_instance.schema_name}.{self.datasetup_instance.dataset_name}', '{self.datasetup_instance.y[0]}', '{predictor_columns}' USING PARAMETERS exclude_columns='name', max_ntree={self.model_instance.max_ntree}, max_depth={self.model_instance.max_depth}, nbins={self.model_instance.nbins}, split_proposal_method={self.model_instance.split_proposal_method}, tol={self.model_instance.tol}, learning_rate={self.model_instance.learning_rate}, min_split_loss={self.model_instance.min_split_loss}, weight_reg={self.model_instance.weight_reg}, sample={self.model_instance.sample}, col_sample_by_tree={self.model_instance.col_sample_by_tree}, col_sample_by_node={self.model_instance.col_sample_by_node}, seed=1, id_column='name')"
        else:
            train_sql = f"SELECT rf_classifier('{self.datasetup_instance.schema_name}.{self.datasetup_instance.model_name}', '{self.datasetup_instance.schema_name}.{self.datasetup_instance.dataset_name}', '{self.datasetup_instance.y[0]}', '{predictor_columns}' USING PARAMETERS exclude_columns='name', ntree={self.model_instance.ntree}, mtry={self.model_instance.mtry}, max_breadth={self.model_instance.max_breadth}, sampling_size={self.model_instance.sampling_size}, max_depth={self.model_instance.max_depth}, min_leaf_size={self.model_instance.min_leaf_size}, nbins={self.model_instance.nbins}, seed=1, id_column='name')"

        print(f"Tree Regressor Train SQL: {train_sql}")
        current_cursor().execute(train_sql)

        self.model.input_relation = f"{self.datasetup_instance.schema_name}.{self.datasetup_instance.dataset_name}"
        self.model.test_relation = self.model.input_relation
        self.model.X = _X
        self.model.y = f'"{self.datasetup_instance.y[0]}"'
        self.model._compute_attributes()

        return self.model

    def vpy_linear(self):
        """
        function to train vertica linear model(s)
        """
        self.model.fit(
            f"{self.datasetup_instance.schema_name}.{self.datasetup_instance.dataset_name}",
            self.datasetup_instance.X,
            f"{self.datasetup_instance.y[0]}",
        )
        return self.model

    def vpy_timeseries(self):
        """
        function to train vertica timeseries model(s)
        """
        self.model.fit(
            f"{self.datasetup_instance.schema_name}.{self.datasetup_instance.dataset_name}",
            self.datasetup_instance.X[0],
            f"{self.datasetup_instance.y[0]}",
        )
        return self.model

    def vpy_cluster(self):
        """
        function to train vertica clustering model(s)
        """
        self.model.fit(
            f"{self.datasetup_instance.schema_name}.{self.datasetup_instance.dataset_name}",
            self.datasetup_instance.X,
        )
        return self.model

    def py_linear(self):
        """
        function to train python linear model(s)
        """
        dataset = self.datasetup_instance.py_dataset
        self.model.fit(
            dataset[self.datasetup_instance.X], dataset[self.datasetup_instance.y[0]]
        )
        return self.model

    def py_timeseries(self):
        """
        function to train python timeseries model(s)
        """
        return self.model.fit()

    def py_cluster(self):
        """
        function to train clustering model(s)
        """
        dataset = self.datasetup_instance.py_dataset
        return self.model.fit(dataset[self.datasetup_instance.X])

    def py_tf(self):
        """
        function to train python tensorflow model
        """
        batch_size = 100
        epochs = 5

        self.model.compile(
            loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"]
        )

        self.model.fit(
            self.datasetup_instance.py_dataset[0],
            self.datasetup_instance.py_dataset[1],
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
        )
        self.model.summary()

        return self.model


class PredictModel:
    """
    class for making model predictions
    """

    def __init__(self, model, datasetup_instance, model_instance):
        self.model = model
        self.datasetup_instance = datasetup_instance
        self.model_instance = model_instance

    def get_pvalue(self):
        """
        getter function to get pvalue for timeseries model(s)
        """
        if self.datasetup_instance.model_class == "AR":
            p_val = self.model_instance.p
        elif self.datasetup_instance.model_class == "MA":
            p_val = self.model_instance.q
        elif self.datasetup_instance.model_class in ["ARMA", "ARIMA"]:
            p_val = self.model_instance.order[0]
        else:
            p_val = 3
        return p_val

    def vpy_tree_regressor(self):
        """
        function to make predictions using vertica tree regression model(s)
        """
        pred_vdf = self.model.predict(
            f"{self.datasetup_instance.schema_name}.{self.datasetup_instance.dataset_name}",
            name=f"{self.datasetup_instance.y[0]}_pred",
        )
        pred_prob_vdf = None
        current_cursor().execute(
            f"DROP SEQUENCE IF EXISTS {self.datasetup_instance.schema_name}.sequence_auto_increment"
        )
        return pred_vdf, pred_prob_vdf

    def vpy_tree_classifier(self):
        """
        function to make predictions using vertica tree classification model(s)
        """
        pred_vdf = self.model.predict(
            f"{self.datasetup_instance.schema_name}.{self.datasetup_instance.dataset_name}",
            name=f"{self.datasetup_instance.y[0]}_pred",
        )[f"{self.datasetup_instance.y[0]}_pred"].astype("int")
        pred_prob_vdf = self.model.predict_proba(
            f"{self.datasetup_instance.schema_name}.{self.datasetup_instance.dataset_name}",
            name=f"{self.datasetup_instance.y[0]}_pred",
        )

        # y_class = titanic_vd_fun[y].distinct()
        y_class = (
            current_cursor()
            .execute(
                f"select distinct {self.datasetup_instance.y[0]} from {self.datasetup_instance.schema_name}.{self.datasetup_instance.dataset_name}"
            )
            .fetchall()
        )
        y_class = list(chain(*y_class))

        # pred_prob_vdf[f"{y}_pred"].astype("int")
        for i in y_class:
            pred_prob_vdf[f"{self.datasetup_instance.y[0]}_pred_{i}"].astype("float")

        return pred_vdf, pred_prob_vdf

    def vpy_linear(self):
        """
        function to make predictions using vertica linear model(s)
        """
        pred_vdf = self.model.predict(
            f"{self.datasetup_instance.schema_name}.{self.datasetup_instance.dataset_name}",
            name=f"{self.datasetup_instance.y[0]}_pred",
        )
        pred_prob_vdf = None

        return pred_vdf, pred_prob_vdf

    def vpy_timeseries(self):
        """
        function to make predictions using vertica timeseries model(s)
        """
        row_cnt = (
            current_cursor()
            .execute(
                f"select count(*) from {self.datasetup_instance.schema_name}.{self.datasetup_instance.dataset_name}"
            )
            .fetchall()[0][0]
        )
        npredictions = self.model_instance.npredictions

        pred_vdf = self.model.predict(
            f"{self.datasetup_instance.schema_name}.{self.datasetup_instance.dataset_name}",
            self.datasetup_instance.X[0],
            self.datasetup_instance.y[0],
            start=self.get_pvalue(),
            npredictions=npredictions if npredictions else row_cnt,
            output_estimated_ts=True,
        )
        pred_prob_vdf = None

        return pred_vdf, pred_prob_vdf

    def vpy_cluster(self):
        """
        function to make predictions using vertica clustering model(s)
        """
        pred_vdf = self.model.predict(
            f"{self.datasetup_instance.schema_name}.{self.datasetup_instance.dataset_name}",
            X=self.datasetup_instance.X,
            name=f"{self.datasetup_instance.model_class}_cluster_ids",
        )
        pred_prob_vdf = None

        return pred_vdf, pred_prob_vdf

    def py_regressor(self):
        """
        function to make predictions using python regression model(s)
        """
        dataset = self.datasetup_instance.py_dataset
        # num_params = len(skl_model.coef_) + 1
        pred = self.model.predict(dataset[self.datasetup_instance.X])
        pred_prob = None

        return pred, pred_prob

    def py_classifier(self):
        """
        function to make predictions using python classification model(s)
        """
        dataset = self.datasetup_instance.py_dataset
        # num_params = len(skl_model.coef_) + 1
        pred = self.model.predict(dataset[self.datasetup_instance.X])
        pred_prob = self.model.predict_proba(dataset[self.datasetup_instance.X])

        return pred, pred_prob

    def py_timeseries(self):
        """
        function to make predictions using python timeseries model(s)
        """
        npred = (
            self.model_instance.npredictions + self.get_pvalue()
            if self.model_instance.npredictions
            else None
        )
        pred = self.model.predict(
            start=self.get_pvalue(), end=npred, dynamic=False
        ).values
        # y = y[p_val: npred + 1 if npred else npred].values
        pred_prob = None

        return pred, pred_prob

    def py_cluster(self):
        """
        function to make predictions using python clustering model(s)
        """
        dataset = self.datasetup_instance.py_dataset
        pred = self.model.predict(dataset[self.datasetup_instance.X])
        pred_prob = None

        return pred, pred_prob

    def py_tf(self):
        """
        function to make predictions using python tensorflow model(s)
        """
        loss, acc = self.model.evaluate(
            self.datasetup_instance.py_dataset[2], self.datasetup_instance.py_dataset[3]
        )
        print("Loss: ", loss, "  Accuracy: ", acc)
        pred = self.model.predict(self.datasetup_instance.py_dataset[4][:500])
        pred_prob = None

        return pred, pred_prob
