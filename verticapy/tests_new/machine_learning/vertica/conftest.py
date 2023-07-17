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
# from verticapy.machine_learning.vertica.tree import DummyTreeRegressor
# DummyTreeRegressor.get_tree
from collections import namedtuple
import math
import verticapy.machine_learning.vertica as vpy_linear_model
import verticapy.machine_learning.vertica.svm as vpy_svm
import verticapy.machine_learning.vertica.tree as vpy_tree
from verticapy.connection import current_cursor
from verticapy.tests_new.machine_learning.metrics.test_classification_metrics import (
    python_metrics,
)
import numpy as np
import sklearn.metrics as skl_metrics
import sklearn.linear_model as skl_linear_model
import sklearn.svm as skl_svm
import sklearn.ensemble as skl_ensemble
import sklearn.tree as skl_tree
import sklearn.dummy as skl_dummy
from sklearn.preprocessing import LabelEncoder
import pytest
import statsmodels.api as sm
from scipy import stats
from scipy.stats import f
import matplotlib.pyplot as plt
from vertica_highcharts.highcharts.highcharts import Highchart
import plotly
from verticapy import vDataFrame

le = LabelEncoder()


# vpy_linear_model.LinearSVR
# skl_svm.LinearSVR
# vpy_tree.DummyTreeClassifier


@pytest.fixture(name="get_vpy_model", scope="function")
def get_vpy_model_fixture(winequality_vpy_fun, titanic_vd_fun, schema_loader):
    """
    getter function for vertica tree model
    """

    def _get_vpy_model(model_class, y_true=None, **kwargs):
        schema_name, model_name = schema_loader, "vpy_model"

        if kwargs.get("solver"):
            solver = kwargs.get("solver")
        else:
            if model_class in ["Lasso", "ElasticNet"]:
                solver = "cgd"
            else:
                solver = "Newton"

        if model_class in ["RandomForestRegressor", "RandomForestClassifier"]:
            model = getattr(vpy_tree, model_class)(
                f"{schema_name}.{model_name}",
                overwrite_model=kwargs.get("overwrite_model")
                if kwargs.get("overwrite_model")
                else False,
                n_estimators=kwargs.get("n_estimators")
                if kwargs.get("n_estimators")
                else 10,
                max_features=kwargs.get("max_features")
                if kwargs.get("max_features")
                else 1,
                max_leaf_nodes=kwargs.get("max_leaf_nodes")
                if kwargs.get("max_leaf_nodes")
                else 10,
                sample=kwargs.get("sample") if kwargs.get("sample") else 0.632,
                max_depth=kwargs.get("max_depth") if kwargs.get("max_depth") else 5,
                min_samples_leaf=kwargs.get("min_samples_leaf")
                if kwargs.get("min_samples_leaf")
                else 1,
                min_info_gain=kwargs.get("min_info_gain")
                if kwargs.get("min_info_gain")
                else 0.0,
                nbins=kwargs.get("nbins") if kwargs.get("nbins") else 32,
            )
        elif model_class in ["DecisionTreeRegressor", "DecisionTreeClassifier"]:
            model = getattr(vpy_tree, model_class)(
                f"{schema_name}.{model_name}",
                overwrite_model=kwargs.get("overwrite_model")
                if kwargs.get("overwrite_model")
                else False,
                max_features=kwargs.get("max_features")
                if kwargs.get("max_features")
                else 1,
                max_leaf_nodes=kwargs.get("max_leaf_nodes")
                if kwargs.get("max_leaf_nodes")
                else 10,
                max_depth=kwargs.get("max_depth") if kwargs.get("max_depth") else 5,
                min_samples_leaf=kwargs.get("min_samples_leaf")
                if kwargs.get("min_samples_leaf")
                else 1,
                min_info_gain=kwargs.get("min_info_gain")
                if kwargs.get("min_info_gain")
                else 0.0,
                nbins=kwargs.get("nbins") if kwargs.get("nbins") else 32,
            )
        elif model_class in ["DummyTreeRegressor", "DummyTreeClassifier"]:
            model = getattr(vpy_tree, model_class)(f"{schema_name}.{model_name}")
        elif model_class == "LinearSVR":
            model = getattr(vpy_svm, model_class)(
                f"{schema_name}.{model_name}",
                overwrite_model=kwargs.get("overwrite_model")
                if kwargs.get("overwrite_model")
                else False,
                tol=kwargs.get("tol") if kwargs.get("tol") else 1e-4,
                C=kwargs.get("c") if kwargs.get("c") else 1.0,
                intercept_scaling=kwargs.get("intercept_scaling")
                if kwargs.get("intercept_scaling")
                else 1.0,
                intercept_mode=kwargs.get("intercept_mode")
                if kwargs.get("intercept_mode")
                else "regularized",
                acceptable_error_margin=kwargs.get("acceptable_error_margin")
                if kwargs.get("acceptable_error_margin")
                else 0.1,
                max_iter=kwargs.get("max_iter") if kwargs.get("max_iter") else 100,
            )
        elif model_class == "LinearRegression":
            model = getattr(vpy_linear_model, model_class)(
                f"{schema_name}.{model_name}",
                overwrite_model=kwargs.get("overwrite_model")
                if kwargs.get("overwrite_model")
                else False,
                tol=kwargs.get("tol") if kwargs.get("tol") else 1e-6,
                max_iter=kwargs.get("max_iter") if kwargs.get("max_iter") else 100,
                solver=solver,
                fit_intercept=kwargs.get("fit_intercept")
                if kwargs.get("fit_intercept")
                else True,
            )
        elif model_class == "ElasticNet":
            model = getattr(vpy_linear_model, model_class)(
                f"{schema_name}.{model_name}",
                overwrite_model=kwargs.get("overwrite_model")
                if kwargs.get("overwrite_model")
                else False,
                tol=kwargs.get("tol") if kwargs.get("tol") else 1e-6,
                C=kwargs.get("c") if kwargs.get("c") else 1.0,
                max_iter=kwargs.get("max_iter") if kwargs.get("max_iter") else 100,
                solver=solver,
                l1_ratio=kwargs.get("l1_ratio") if kwargs.get("l1_ratio") else 0.5,
                fit_intercept=kwargs.get("fit_intercept")
                if kwargs.get("fit_intercept")
                else True,
            )
        else:
            model = getattr(vpy_linear_model, model_class)(
                f"{schema_name}.{model_name}",
                overwrite_model=kwargs.get("overwrite_model")
                if kwargs.get("overwrite_model")
                else False,
                tol=kwargs.get("tol") if kwargs.get("tol") else 1e-6,
                C=kwargs.get("c") if kwargs.get("c") else 1.0,
                max_iter=kwargs.get("max_iter") if kwargs.get("max_iter") else 100,
                solver=solver,
                fit_intercept=kwargs.get("fit_intercept")
                if kwargs.get("fit_intercept")
                else True,
            )

        print(f"VerticaPy Training Parameters: {model.get_params()}")
        model.drop()

        if model_class in ["RandomForestClassifier", "DecisionTreeClassifier", "DummyTreeClassifier"]:
            delete_sql = f"DELETE FROM {schema_name}.titanic WHERE AGE IS NULL OR FARE IS NULL OR SEX IS NULL OR SURVIVED IS NULL"
            print(f"Delete SQL: {delete_sql}")

            current_cursor().execute(delete_sql)

            if y_true is None:
                y_true = ["age", "fare", "sex"]
            model.fit(
                f"{schema_name}.titanic",
                y_true,
                "survived",
            )

            pred_vdf = model.predict(titanic_vd_fun, name="survived_pred")[
                "survived_pred"
            ].astype("int")

            pred_prob_vdf = model.predict_proba(titanic_vd_fun, name="survived_pred")

            pred_prob_vdf["survived_pred"].astype("int")
            pred_prob_vdf["survived_pred_0"].astype("float")
            pred_prob_vdf["survived_pred_1"].astype("float")

        else:
            if y_true is None:
                y_true = ["citric_acid", "residual_sugar", "alcohol"]
            model.fit(
                f"{schema_name}.winequality",
                y_true,
                "quality",
            )

            pred_vdf = model.predict(winequality_vpy_fun, name="quality_pred")
            pred_prob_vdf = None

        vpy = namedtuple(
            "vertica_models",
            ["model", "pred_vdf", "pred_prob_vdf", "schema_name", "model_name"],
        )(model, pred_vdf, pred_prob_vdf, schema_name, model_name)

        return vpy

    yield _get_vpy_model


@pytest.fixture(name="get_py_model", scope="function")
def get_py_model_fixture(winequality_vpy_fun, titanic_vd_fun):
    """
    getter function for python model
    """

    def _get_py_model(model_class, py_fit_intercept=None, **kwargs):
        # sklearn
        if model_class in ["RandomForestClassifier", "DecisionTreeClassifier", "DummyTreeClassifier"]:
            # titanic_pdf = impute_dataset(titanic_vd_fun)
            # print(titanic_pdf.columns)
            titanic_pdf = titanic_vd_fun.to_pandas()
            titanic_pdf.dropna(subset=["age", "fare"], inplace=True)

            titanic_pdf["sex"] = le.fit_transform(titanic_pdf["sex"])
            X = titanic_pdf[["age", "fare", "sex"]]
            y = titanic_pdf["survived"]
        else:
            winequality_pdf = winequality_vpy_fun.to_pandas()
            winequality_pdf["citric_acid"] = winequality_pdf["citric_acid"].astype(
                float
            )
            winequality_pdf["residual_sugar"] = winequality_pdf[
                "residual_sugar"
            ].astype(float)

            X = winequality_pdf[["citric_acid", "residual_sugar", "alcohol"]]
            y = winequality_pdf["quality"]

        if model_class in ["RandomForestRegressor", "RandomForestClassifier"]:
            model = getattr(skl_ensemble, model_class)(
                n_estimators=kwargs.get("n_estimators")
                if kwargs.get("n_estimators")
                else 10,
                max_features=kwargs.get("max_features")
                if kwargs.get("max_features")
                else 1,
                max_leaf_nodes=kwargs.get("max_leaf_nodes")
                if kwargs.get("max_leaf_nodes")
                else 10,
                max_samples=kwargs.get("sample") if kwargs.get("sample") else 0.632,
                max_depth=kwargs.get("max_depth") if kwargs.get("max_depth") else 5,
                min_samples_leaf=kwargs.get("min_samples_leaf")
                if kwargs.get("min_samples_leaf")
                else 1,
            )
        elif model_class in ["DecisionTreeRegressor", "DecisionTreeClassifier"]:
            model = getattr(skl_tree, model_class)(
                max_features=kwargs.get("max_features")
                if kwargs.get("max_features")
                else 1,
                max_leaf_nodes=kwargs.get("max_leaf_nodes")
                if kwargs.get("max_leaf_nodes")
                else 10,
                max_depth=kwargs.get("max_depth") if kwargs.get("max_depth") else 5,
                min_samples_leaf=kwargs.get("min_samples_leaf")
                if kwargs.get("min_samples_leaf")
                else 1,
            )
        elif model_class in ["DummyTreeRegressor"]:
            model = getattr(skl_dummy, 'DummyRegressor')()
        elif model_class in ["DummyTreeClassifier"]:
            model = getattr(skl_dummy, 'DummyClassifier')()
        elif model_class == "LinearSVR":
            model = getattr(skl_svm, model_class)(
                fit_intercept=py_fit_intercept if py_fit_intercept else True
            )
        else:
            model = getattr(skl_linear_model, model_class)(
                fit_intercept=py_fit_intercept if py_fit_intercept else True
            )

        print(f"Python Training Parameters: {model.get_params()}")
        model.fit(X, y)

        # num_params = len(skl_model.coef_) + 1
        pred = model.predict(X)

        if model_class in ["RandomForestClassifier", "DecisionTreeClassifier", "DummyTreeClassifier"]:
            pred_prob = model.predict_proba(X)
        else:
            pred_prob = None

        # statsmodels
        # add constant to predictor variables
        # X_sm = sm.add_constant(X)

        # fit linear regression model
        # sm_model = sm.OLS(y, X_sm).fit()

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
            "DummyTreeRegressor",
            "DummyTreeClassifier",
        ]:
            y, pred, model = model_obj.y, model_obj.pred, model_obj.model
            if model_class in ["RandomForestRegressor", "RandomForestClassifier"]:
                num_params = (
                    sum(tree.tree_.node_count for tree in model.estimators_) * 5
                    if model_class == model_class
                    else len(model.coef_) + 1
                )
            else:
                num_params = len(model.get_params()) + 1
        else:
            _, y, _, pred, _, model = get_py_model(
                model_class, py_fit_intercept=fit_intercept
            )
            num_params = len(model.coef_) + 1

        regression_metrics_map = {}
        no_of_records = len(y)
        avg = sum(y) / no_of_records
        num_features = 3 if model_class in ["DummyTreeRegressor"] else len(model.feature_names_in_)

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
        regression_metrics_map["rsquared_adj"] = 1 - (
                1 - regression_metrics_map["r2"]
        ) * (no_of_records - 1) / (no_of_records - num_features - 1)
        regression_metrics_map["aic"] = (
                no_of_records * math.log(regression_metrics_map["mse"]) + 2 * num_params
        )
        regression_metrics_map["bic"] = no_of_records * math.log(
            regression_metrics_map["mse"]
        ) + num_params * math.log(no_of_records)
        regression_metrics_map["explained_variance_score"] = getattr(
            skl_metrics, "explained_variance_score"
        )(y, pred)
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

    def _calculate_classification_metrics(
            model_class, model_obj=None, fit_intercept=True
    ):
        _model_obj = get_py_model(model_class)
        y, pred, pred_prob, model = (
            model_obj.y.ravel() if model_obj else _model_obj.y.ravel(),
            model_obj.pred.ravel() if model_obj else _model_obj.pred.ravel(),
            model_obj.pred_prob[:, 1].ravel()
            if model_obj
            else _model_obj.pred_prob[:, 1].ravel(),
            model_obj.model if model_obj else _model_obj.model,
        )

        precision, recall, thresholds = skl_metrics.roc_curve(y, pred_prob, pos_label=1)

        classification_metrics_map = {}
        no_of_records = len(y)
        avg = sum(y) / no_of_records
        num_features = 3 if model_class in ["DummyTreeClassifier"] else len(model.feature_names_in_)

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

