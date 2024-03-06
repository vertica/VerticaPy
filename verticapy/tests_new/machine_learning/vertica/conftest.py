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
from collections import namedtuple
import math
import sys

import numpy as np
import pytest
import sklearn.metrics as skl_metrics
from sklearn.preprocessing import LabelEncoder
from scipy.stats import f

import verticapy as vp
from verticapy.tests_new.machine_learning.metrics.test_classification_metrics import (
    python_metrics,
)
from verticapy.tests_new.machine_learning.vertica import (
    TIMESERIES_MODELS,
    CLUSTER_MODELS,
)
from verticapy.tests_new.machine_learning.vertica.model_utils import (
    get_function_name,
    get_model_class,
    get_xy,
    DataSetUp,
    TrainModel,
    PredictModel,
)

if sys.version_info < (3, 12):
    import tensorflow as tf

le = LabelEncoder()


@pytest.fixture(autouse=True)
def set_plotting_lib():
    vp.set_option("plotting_lib", "matplotlib")
    yield
    vp.set_option("plotting_lib", "plotly")


@pytest.fixture(name="get_vpy_model", scope="function")
def get_vpy_model_fixture(
    winequality_vpy_fun, titanic_vd_fun, airline_vd_fun, iris_vd_fun, schema_loader
):
    """
    getter function for vertica tree model
    """

    def _get_vpy_model(model_class, X=None, y=None, **kwargs):
        schema_name, model_name = schema_loader, f"vpy_model_{model_class}"
        X = get_xy(model_class)["X"] if X is None else X
        y = get_xy(model_class)["y"] if y is None else y

        data, train, pred = get_function_name(model_class)["vpy"]

        # Data preparation
        datasetup_instance = DataSetUp(schema_name, model_name, model_class, X, y)
        getattr(datasetup_instance, data)()

        # Model initialization
        model_instance = get_model_class(model_class)(datasetup_instance, **kwargs)
        model = model_instance.vpy()

        # Train
        train_instance = TrainModel(model, datasetup_instance, model_instance)
        model = getattr(train_instance, train)()

        # predic
        pred_instance = PredictModel(model, datasetup_instance, model_instance)
        pred_vdf, pred_prob_vdf = getattr(pred_instance, pred)()

        vpy = namedtuple(
            "vertica_models",
            ["model", "pred_vdf", "pred_prob_vdf", "schema_name", "model_name"],
        )(model, pred_vdf, pred_prob_vdf, schema_name, model_name)

        return vpy

    yield _get_vpy_model


@pytest.fixture(name="get_py_model", scope="function")
def get_py_model_fixture(
    winequality_vpy_fun, titanic_vd_fun, airline_vd_fun, iris_vd_fun, schema_loader
):
    """
    getter function for python model
    """

    def _get_py_model(model_class, **kwargs):
        data, train, pred = get_function_name(model_class)["py"]

        # Data preparation
        datasetup_instance = DataSetUp(schema_loader, None, model_class, None, None)
        getattr(datasetup_instance, data)()

        # Model initialization
        model_instance = get_model_class(model_class)(datasetup_instance, **kwargs)
        model = model_instance.py()

        # Train
        train_instance = TrainModel(model, datasetup_instance, model_instance)
        model = getattr(train_instance, train)()

        # Predict
        pred_instance = PredictModel(model, datasetup_instance, model_instance)
        pred, pred_prob = getattr(pred_instance, pred)()

        if model_class in TIMESERIES_MODELS:
            npred = (
                model_instance.npredictions + pred_instance.get_pvalue()
                if model_instance.npredictions
                else None
            )

            dataset = datasetup_instance.py_dataset.reset_index()
            X, y = dataset[[datasetup_instance.X[0]]], dataset[datasetup_instance.y[0]]
            y = y[pred_instance.get_pvalue() : npred + 1 if npred else npred].values
        elif model_class in ["TENSORFLOW", "TF"]:
            dataset = datasetup_instance.py_dataset
            X, y = dataset[4][:500], dataset[5][:500]
        else:
            dataset = datasetup_instance.py_dataset
            X, y = (
                dataset[datasetup_instance.X],
                None
                if model_class in CLUSTER_MODELS
                else dataset[datasetup_instance.y[0]],
            )

        py = namedtuple(
            "python_models", ["X", "y", "sm_model", "pred", "pred_prob", "model"]
        )(X, y, None, pred, pred_prob, model)

        return py

    return _get_py_model


@pytest.fixture(name="regression_metrics", scope="function")
def calculate_regression_metrics(get_py_model):
    """
    fixture to calculate python metrics
    """

    def _calculate_regression_metrics(model_class, model_obj=None, fit_intercept=True):
        if model_class in [
            "RandomForestRegressor",
            "RandomForestClassifier",
            "DecisionTreeRegressor",
            "DecisionTreeClassifier",
            "XGBRegressor",
            "DummyTreeRegressor",
            # "DummyTreeClassifier",
        ]:
            y, pred, model = model_obj.y, model_obj.pred, model_obj.model
            if model_class in ["RandomForestRegressor", "RandomForestClassifier"]:
                num_params = (
                    sum(tree.tree_.node_count for tree in model.estimators_) * 5
                    if model_class == model_class
                    else len(model.coef_) + 1
                )
                num_params = (
                    2  # setting it to 2 as per dev code(k+1 where, k=1), need to check
                )
            else:
                num_params = len(model.get_params()) + 1
        elif model_class in ["AR", "MA", "ARMA", "ARIMA"]:
            _, y, _, pred, _, model = get_py_model(model_class)
            num_params = len(list(model.params))
        else:
            _, y, _, pred, _, model = get_py_model(
                model_class, py_fit_intercept=fit_intercept
            )
            num_params = len(model.coef_) + 1

        regression_metrics_map = {}
        no_of_records = len(y)
        avg = sum(y) / no_of_records
        num_features = (
            3
            if model_class in ["DummyTreeRegressor"]
            else 1
            if model_class in ["AR", "MA", "ARMA", "ARIMA"]
            else (len(model.feature_names_in_))
        )
        # y_bar = y.mean()
        # ss_tot = ((y - y_bar) ** 2).sum()
        # ss_res = ((y - pred) ** 2).sum()

        regression_metrics_map["mse"] = getattr(skl_metrics, "mean_squared_error")(
            y, pred
        )
        regression_metrics_map["rmse"] = np.sqrt(regression_metrics_map["mse"])
        regression_metrics_map["ssr"] = sum(np.square(pred - avg))
        regression_metrics_map["sse"] = sum(np.square(y - pred))
        regression_metrics_map["dfr"] = num_features
        regression_metrics_map["dfe"] = no_of_records - num_features - 1
        regression_metrics_map["msr"] = (
            regression_metrics_map["ssr"] / regression_metrics_map["dfr"]
        )
        regression_metrics_map["_mse"] = (
            regression_metrics_map["sse"] / regression_metrics_map["dfe"]
        )
        regression_metrics_map["f"] = (
            regression_metrics_map["msr"] / regression_metrics_map["_mse"]
        )
        regression_metrics_map["p_value"] = f.sf(
            regression_metrics_map["f"], num_features, no_of_records
        )
        regression_metrics_map["mean_squared_log_error"] = (
            sum(
                pow(
                    (np.log10(pred + 1) - np.log10(y + 1)),
                    2,
                )
            )
            / no_of_records
        )
        regression_metrics_map["r2"] = regression_metrics_map[
            "r2_score"
        ] = skl_metrics.r2_score(y, pred)
        # regression_metrics_map["r2"] = regression_metrics_map["r2_score"] = 1 - (
        #     ss_res / ss_tot
        # )
        regression_metrics_map["rsquared_adj"] = 1 - (
            1 - regression_metrics_map["r2"]
        ) * (no_of_records - 1) / (no_of_records - num_features - 1)
        regression_metrics_map["aic"] = (
            no_of_records * math.log(regression_metrics_map["mse"]) + 2 * num_params
        )
        regression_metrics_map["bic"] = no_of_records * math.log(
            regression_metrics_map["mse"]
        ) + num_params * math.log(no_of_records)
        # regression_metrics_map["explained_variance_score"] = getattr(
        #     skl_metrics, "explained_variance_score"
        # )(y, pred)
        regression_metrics_map["explained_variance_score"] = 1 - np.var(
            (y - pred)
        ) / np.var(y)
        regression_metrics_map["max_error"] = getattr(skl_metrics, "max_error")(y, pred)
        regression_metrics_map["median_absolute_error"] = getattr(
            skl_metrics, "median_absolute_error"
        )(y, pred)
        regression_metrics_map["mean_absolute_error"] = getattr(
            skl_metrics, "mean_absolute_error"
        )(y, pred)
        regression_metrics_map["mean_squared_error"] = getattr(
            skl_metrics, "mean_squared_error"
        )(y, pred)
        regression_metrics_map[""] = ""

        return regression_metrics_map

    return _calculate_regression_metrics


@pytest.fixture(name="classification_metrics", scope="function")
def calculate_classification_metrics(get_py_model):
    """
    fixture to calculate python classification metrics
    """

    def _calculate_classification_metrics(model_class, model_obj=None):
        if model_obj:
            y = model_obj.y.ravel()
            pred = model_obj.pred.ravel()
            pred_prob = model_obj.pred_prob[:, 1].ravel()
        else:
            _model_obj = get_py_model(model_class)
            y = _model_obj.y.ravel()
            pred = _model_obj.pred.ravel()
            pred_prob = _model_obj.pred_prob[:, 1].ravel()

        precision, recall, _ = skl_metrics.precision_recall_curve(
            y, pred_prob, pos_label=1
        )

        classification_metrics_map = {}
        # no_of_records = len(y)
        # avg = sum(y) / no_of_records
        # num_features = 3 if model_class in ["DummyTreeClassifier"] else len(model.feature_names_in_)

        classification_metrics_map["auc"] = skl_metrics.auc(recall, precision)
        classification_metrics_map["prc_auc"] = skl_metrics.auc(recall, precision)
        classification_metrics_map["accuracy_score"] = classification_metrics_map[
            "accuracy"
        ] = skl_metrics.accuracy_score(y, pred)
        classification_metrics_map["log_loss"] = -(
            (y * np.log10(pred + 1e-90)) + (1 - y) * np.log10(1 - pred + 1e-90)
        ).mean()
        classification_metrics_map["precision_score"] = classification_metrics_map[
            "precision"
        ] = python_metrics(y_true=y, y_pred=pred, metric_name="precision_score")
        classification_metrics_map["recall_score"] = classification_metrics_map[
            "recall"
        ] = python_metrics(y_true=y, y_pred=pred, metric_name="recall_score")
        classification_metrics_map["f1_score"] = classification_metrics_map[
            "f1"
        ] = python_metrics(y_true=y, y_pred=pred, metric_name="f1_score")
        classification_metrics_map["matthews_corrcoef"] = classification_metrics_map[
            "mcc"
        ] = skl_metrics.matthews_corrcoef(y_true=y, y_pred=pred)
        classification_metrics_map["informedness"] = python_metrics(
            y_true=y, y_pred=pred, metric_name="informedness"
        )
        classification_metrics_map["markedness"] = python_metrics(
            y_true=y, y_pred=pred, metric_name="markedness"
        )
        classification_metrics_map["critical_success_index"] = python_metrics(
            y_true=y, y_pred=pred, metric_name="critical_success_index"
        )
        classification_metrics_map["fpr"] = python_metrics(
            y_true=y, y_pred=pred, metric_name="fpr"
        )
        classification_metrics_map["tpr"] = python_metrics(
            y_true=y, y_pred=pred, metric_name="tpr"
        )

        return classification_metrics_map

    return _calculate_classification_metrics
