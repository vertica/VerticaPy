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
import numpy as np
from verticapy.connection import current_cursor
from verticapy.tests_new.machine_learning.vertica import (
    REL_TOLERANCE,
    ABS_TOLERANCE,
    rel_tolerance_map,
)
from vertica_highcharts.highcharts.highcharts import Highchart
import plotly
from scipy import stats
import pytest
import matplotlib.pyplot as plt

details_report_args = (
    "metric, expected, _rel_tolerance, _abs_tolerance",
    [
        ("Dep. Variable", '"quality"', "NA", "NA"),
        ("Model", "LinearRegression", "NA", "NA"),
        ("No. Observations", None, REL_TOLERANCE, ABS_TOLERANCE),
        ("No. Predictors", None, REL_TOLERANCE, ABS_TOLERANCE),
        ("R-squared", None, rel_tolerance_map, ABS_TOLERANCE),
        ("Adj. R-squared", None, rel_tolerance_map, ABS_TOLERANCE),
        ("F-statistic", None, rel_tolerance_map, 1e-9),
        ("Prob (F-statistic)", None, REL_TOLERANCE, ABS_TOLERANCE),
        ("Kurtosis", None, 1e-2, ABS_TOLERANCE),
        ("Skewness", None, 1e-3, ABS_TOLERANCE),
        ("Jarque-Bera (JB)", None, 1e-2, ABS_TOLERANCE),
    ],
)
anova_report_args = (
    "metric, metric_types, _rel_tolerance, _abs_tolerance",
    [
        ("df", ("dfr", "dfe"), rel_tolerance_map, ABS_TOLERANCE),
        ("ss", ("ssr", "sse"), rel_tolerance_map, ABS_TOLERANCE),
        ("ms", ("msr", "mse"), rel_tolerance_map, ABS_TOLERANCE),
        ("f", ("f", ""), rel_tolerance_map, ABS_TOLERANCE),
        ("p_value", ("p_value", ""), rel_tolerance_map, ABS_TOLERANCE),
    ],
)

regression_metrics_args = (
    "vpy_metric_name, py_metric_name, _rel_tolerance",
    [
        (("explained_variance", None), "explained_variance_score", rel_tolerance_map),
        (("max_error", None), "max_error", rel_tolerance_map),
        (("median_absolute_error", None), "median_absolute_error", rel_tolerance_map),
        (("mean_absolute_error", None), "mean_absolute_error", rel_tolerance_map),
        (("mean_squared_error", None), "mean_squared_error", rel_tolerance_map),
        (("mean_squared_log_error", None), "mean_squared_log_error", rel_tolerance_map),
        (
            (
                "rmse",
                "root_mean_squared_error",
            ),
            "rmse",
            rel_tolerance_map,
        ),
        (("r2", None), "r2_score", rel_tolerance_map),
        (("r2_adj", None), "rsquared_adj", rel_tolerance_map),
        (("aic", None), "aic", rel_tolerance_map),
        (("bic", None), "bic", rel_tolerance_map),
    ],
)

classification_metrics_args = (
    "vpy_metric_name, py_metric_name, _rel_tolerance",
    [
        (("auc", None), "auc", rel_tolerance_map),
        (("prc_auc", None), "prc_auc", rel_tolerance_map),
        (("accuracy", None), "accuracy_score", rel_tolerance_map),
        (
            ("log_loss", None),
            "log_loss",
            rel_tolerance_map,
        ),  # vertica uses log base 10, sklean uses natural log(e)
        (("precision", None), "precision_score", rel_tolerance_map),
        (("recall", None), "recall_score", rel_tolerance_map),
        (("f1_score", None), "f1_score", rel_tolerance_map),
        (("mcc", None), "matthews_corrcoef", rel_tolerance_map),
        # (("informedness", None), "informedness", rel_tolerance_map), # getting mismatch for xgb
        (("markedness", None), "markedness", rel_tolerance_map),
        (("csi", None), "critical_success_index", rel_tolerance_map),
    ],
)


@pytest.fixture
def model_params(model_class):
    """
    fixture - model parameters
    """
    model_params_map = {
        "RandomForestRegressor": (
            "n_estimators, max_features, max_leaf_nodes, sample, max_depth, min_samples_leaf, min_info_gain, nbins",
            [
                # (None, None, None, None, None, None, None, None),  # accuracy does not mach with default parameters
                (10, 1, 10, 0.632, 5, 1, None, None),
                # (5, 'max', 1e6, 0.42, 6, 8, 0.3, 20),
            ],
        ),
        "RandomForestClassifier": (
            "n_estimators, max_features, max_leaf_nodes, sample, max_depth, min_samples_leaf, min_info_gain, nbins",
            [
                # (None, None, None, None, None, None, None, None),  # accuracy does not mach with default parameters
                (10, 1, 10, 0.632, 5, 1, None, None),
                # (5, 'max', 1e6, 0.42, 6, 8, 0.3, 20),
            ],
        ),
        "DecisionTreeRegressor": (
            "max_features, max_leaf_nodes, max_depth, min_samples_leaf, min_info_gain, nbins",
            [
                # (None, None, None, None, None, None, None, None),  # accuracy does not mach with default parameters
                (1, 10, 5, 1, None, None),
                # (5, 'max', 1e6, 0.42, 6, 8, 0.3, 20),
            ],
        ),
        "DecisionTreeClassifier": (
            "max_features, max_leaf_nodes, max_depth, min_samples_leaf, min_info_gain, nbins",
            [
                # (None, None, None, None, None, None, None, None),  # accuracy does not mach with default parameters
                (1, 10, 5, 1, None, None),
                # (5, 'max', 1e6, 0.42, 6, 8, 0.3, 20),
            ],
        ),
        "XGBRegressor": (
            "max_ntree, max_depth, nbins, learning_rate, min_split_loss, weight_reg, sample, col_sample_by_tree, col_sample_by_node",
            [
                # (None, None, None, None, None, None, None, None),  # accuracy does not mach with default parameters
                (10, 5, 32, 0.1, 0.0, 0.0, 1.0, 1.0, 1.0),
                # (5, 'max', 1e6, 0.42, 6, 8, 0.3, 20),
            ],
        ),
        "XGBClassifier": (
            "max_ntree, max_depth, nbins, learning_rate, min_split_loss, weight_reg, sample, col_sample_by_tree, col_sample_by_node",
            [
                # (None, None, None, None, None, None, None, None),  # accuracy does not mach with default parameters
                (10, 5, 32, 0.1, 0.0, 0.0, 1.0, 1.0, 1.0),
                # (5, 'max', 1e6, 0.42, 6, 8, 0.3, 20),
            ],
        ),
        "DummyTreeRegressor": (),
        "DummyTreeClassifier": (),
        "Ridge": (
            "tol, c, max_iter, solver, fit_intercept",
            [
                (1e-6, 1, 100, "bfgs", True),
                (1e-4, 0.99, 200, "newton", False),
            ],
        ),
        "Lasso": (
            "tol, c, max_iter, solver, fit_intercept",
            [
                # (1e-6, 1, 100, 'bfgs', True), # bfgs not supported
                # (1e-4, 0.99, 200, 'newton', False), # newton not supported
                (1e-4, 1, 200, "cgd", False)
            ],
        ),
        "ElasticNet": (
            "tol, c, max_iter, solver, l1_ratio, fit_intercept",
            [
                # (1e-6, 1, 100, 'bfgs', 0.5, True), # bfgs not supported
                # (1e-4, 0.99, 200, 'newton', 0.9, False), # newton not supported
                (
                    1e-4,
                    1,
                    200,
                    "cgd",
                    0.5,
                    True,
                )  # l1_ratio 0.4 not matches with python
            ],
        ),
        "LinearRegression": (
            "tol, max_iter, solver, fit_intercept",
            [
                (1e-6, 100, "bfgs", True),
                (1e-4, 200, "newton", False),
                (1e-4, 200, "cgd", False),
            ],
        ),
        "LinearSVR": (
            "tol, c, intercept_scaling, intercept_mode, acceptable_error_margin, max_iter",
            [
                (1e-6, 1, 1, "regularized", 0.1, 100),
                (1e-4, 1, 0.6, "unregularized", 0, 200),
                (1e-4, 1, 0, "regularized", 1, 400),
                (1e-4, 1, 3, "unregularized", 0.5, 500),
            ],
        ),
        "PoissonRegressor": (
            "tol, penalty, c, max_iter, solver, fit_intercept",
            [
                (1e-6, "l2", 1, 100, "newton", True),
            ],
        ),
        "AR": (
            "order, method, penalty, c, missing, npredictions",
            [
                ((3, 0, 0), "ols", "none", 1, "linear_interpolation", 144),
            ],
        ),
        "MA": (
            "q, penalty, c, missing, npredictions",
            [
                (1, "none", 1, "linear_interpolation", 144),
            ],
        ),
        "ARMA": (
            "order, tol, max_iter, init, missing, npredictions",
            [
                ((2, 0, 1), 1e-6, 100, "zero", "linear_interpolation", 144),
            ],
        ),
        "ARIMA": (
            "order, tol, max_iter, init, missing, npredictions",
            [
                ((2, 1, 1), 1e-6, 100, "zero", "linear_interpolation", 144),
            ],
        ),
    }

    return model_params_map[model_class]


def regression_report_none(
    get_vpy_model,
    get_py_model,
    model_class,
    regression_metrics,
    fun_name,
    vpy_metric_name,
    py_metric_name,
    _rel_tolerance,
    model_params,
):
    """
    test function - regression/report None
    """
    _model_class_tuple = (
        None
        if model_class in ["DummyTreeRegressor", "DummyTreeClassifier"]
        else namedtuple(model_class, model_params[0])(*model_params[1][0])
    )
    vpy_model_obj = get_vpy_model(model_class)

    if model_class in ["AR", "MA", "ARMA", "ARIMA"]:
        if model_class == "MA":
            p_val = _model_class_tuple.order[2]
        else:
            p_val = _model_class_tuple.order[0]

        reg_rep = (
            vpy_model_obj.model.report(
                start=p_val,
                npredictions=_model_class_tuple.npredictions,
            )
            if fun_name == "report"
            else vpy_model_obj.model.regression_report(
                start=p_val,
                npredictions=_model_class_tuple.npredictions,
            )
        )
    else:
        reg_rep = (
            vpy_model_obj.model.report()
            if fun_name == "report"
            else vpy_model_obj.model.regression_report()
        )
    vpy_rep_map = dict(zip(reg_rep["index"], reg_rep["value"]))

    if vpy_metric_name[0] in vpy_rep_map or vpy_metric_name[1] in vpy_rep_map:
        vpy_score = vpy_rep_map[
            vpy_metric_name[0] if vpy_metric_name[1] is None else vpy_metric_name[1]
        ]
    else:
        pytest.skip(f"{vpy_metric_name[0]} metric is not applicable for {model_class}")

    if model_class in [
        "RandomForestRegressor",
        "DecisionTreeRegressor",
        "DummyTreeRegressor",
        "XGBRegressor",
    ]:
        py_model_obj = get_py_model(model_class)
        regression_metrics_map = regression_metrics(model_class, model_obj=py_model_obj)
    else:
        # py_model_obj = get_py_model(model_class)
        regression_metrics_map = regression_metrics(model_class)

    py_score = regression_metrics_map[py_metric_name]

    print(
        f"Metric Name: {vpy_metric_name, py_metric_name}, vertica: {vpy_score}, sklearn: {py_score}"
    )

    return vpy_score, py_score


def regression_report_details(
    model_class,
    get_vpy_model,
    get_py_model,
    regression_metrics,
    fun_name,
    metric,
    expected,
    _rel_tolerance,
    _abs_tolerance,
):
    """
    test function - regression/report details
    """
    vpy_model_obj = get_vpy_model(model_class)

    reg_rep_details = (
        vpy_model_obj.model.report(metrics="details")
        if fun_name == "report"
        else vpy_model_obj.model.regression_report(metrics="details")
    )
    vpy_reg_rep_details_map = dict(
        zip(reg_rep_details["index"], reg_rep_details["value"])
    )

    # Python
    if model_class in [
        "RandomForestRegressor",
        "DecisionTreeRegressor",
        "DummyTreeRegressor",
        "XGBRegressor",
    ]:
        py_model_obj = get_py_model(model_class)
        regression_metrics_map = regression_metrics(model_class, model_obj=py_model_obj)
    else:
        py_model_obj = get_py_model(model_class)
        regression_metrics_map = regression_metrics(model_class)

    if metric == "No. Observations":
        py_res = len(py_model_obj.y)
    elif metric == "Model":
        if model_class == "RandomForestRegressor":
            py_res = "RandomForestRegressor"
        elif model_class == "DecisionTreeRegressor":
            py_res = "RandomForestRegressor"  # need to check on this
        elif model_class == "XGBRegressor":
            py_res = "XGBRegressor"  # need to check on this
        elif model_class == "DummyTreeRegressor":
            py_res = "DummyTreeRegressor"
        elif model_class == "LinearSVR":
            py_res = "LinearSVR"
        elif model_class == "PoissonRegressor":
            py_res = "PoissonRegressor"
        else:
            py_res = "LinearRegression"
    elif metric == "No. Predictors":
        py_res = len(py_model_obj.X.columns)
    elif metric == "R-squared":
        py_res = regression_metrics_map["r2_score"]
    elif metric == "Adj. R-squared":
        py_res = regression_metrics_map["rsquared_adj"]
    elif metric == "F-statistic":
        py_res = regression_metrics_map["f"]
    elif metric == "Prob (F-statistic)":
        py_res = regression_metrics_map["p_value"]
    elif metric == "Kurtosis":
        py_res = stats.kurtosis(py_model_obj.y)
    elif metric == "Skewness":
        py_res = stats.skew(py_model_obj.y)
    elif metric == "Jarque-Bera (JB)":
        py_res = stats.jarque_bera(py_model_obj.y).statistic
    else:
        py_res = expected

    return vpy_reg_rep_details_map, py_res


def regression_report_anova(
    model_class,
    get_vpy_model,
    get_py_model,
    regression_metrics,
    fun_name,
    _rel_tolerance,
    _abs_tolerance,
):
    """
    test function - regression/report anova
    """
    vpy_model_obj = get_vpy_model(model_class)

    reg_rep_anova = (
        vpy_model_obj.model.report(metrics="anova")
        if fun_name == "report"
        else vpy_model_obj.model.regression_report(metrics="anova")
    )

    # Python
    if model_class in [
        "RandomForestRegressor",
        "DecisionTreeRegressor",
        "DummyTreeRegressor",
        "XGBRegressor",
    ]:
        py_model_obj = get_py_model(model_class)
        regression_metrics_map = regression_metrics(model_class, model_obj=py_model_obj)
    else:
        regression_metrics_map = regression_metrics(model_class)

    return reg_rep_anova, regression_metrics_map


def model_score(
    model_class,
    get_vpy_model,
    get_py_model,
    _metrics,
    vpy_metric_name,
    py_metric_name,
    _rel_tolerance,
    model_params,
):
    """
    test function - score
    """
    _model_class_tuple = (
        None
        if model_class in ["DummyTreeRegressor", "DummyTreeClassifier"]
        else namedtuple(model_class, model_params[0])(*model_params[1][0])
    )

    # # skipping a test
    # if (model_class in ["Ridge", "LinearRegression"] and _model_class.solver == "cgd") or (
    #         model_class in ["Lasso", "ElasticNet"]
    #         and _model_class.solver
    #         in [
    #             "bfgs",
    #             "newton",
    #         ]
    # ):
    #     pytest.skip(
    #         f"optimizer [{_model_class.solver}] is not supported for [{model_class}] model"
    #     )

    if model_class == "RandomForestRegressor":
        vpy_model_obj = get_vpy_model(
            model_class,
            n_estimators=_model_class_tuple.n_estimators,
            max_features=_model_class_tuple.max_features,
            max_leaf_nodes=_model_class_tuple.max_leaf_nodes,
            sample=_model_class_tuple.sample,
            max_depth=_model_class_tuple.max_depth,
            min_samples_leaf=_model_class_tuple.min_samples_leaf,
            min_info_gain=_model_class_tuple.min_info_gain,
            nbins=_model_class_tuple.nbins,
        )
        vpy_score = vpy_model_obj.model.score(metric=vpy_metric_name[0])
    elif model_class in ["RandomForestClassifier"]:
        vpy_model_obj = get_vpy_model(
            model_class,
            n_estimators=_model_class_tuple.n_estimators,
            max_features=_model_class_tuple.max_features,
            max_leaf_nodes=_model_class_tuple.max_leaf_nodes,
            sample=_model_class_tuple.sample,
            max_depth=_model_class_tuple.max_depth,
            min_samples_leaf=_model_class_tuple.min_samples_leaf,
            min_info_gain=_model_class_tuple.min_info_gain,
            nbins=_model_class_tuple.nbins,
        )
        vpy_score = vpy_model_obj.model.score(
            metric=vpy_metric_name[0], average="binary", pos_label=1
        )
    elif model_class == "DecisionTreeRegressor":
        vpy_model_obj = get_vpy_model(
            model_class,
            max_features=_model_class_tuple.max_features,
            max_leaf_nodes=_model_class_tuple.max_leaf_nodes,
            max_depth=_model_class_tuple.max_depth,
            min_samples_leaf=_model_class_tuple.min_samples_leaf,
            min_info_gain=_model_class_tuple.min_info_gain,
            nbins=_model_class_tuple.nbins,
        )
        vpy_score = vpy_model_obj.model.score(metric=vpy_metric_name[0])
    elif model_class == "DecisionTreeClassifier":
        vpy_model_obj = get_vpy_model(
            model_class,
            max_features=_model_class_tuple.max_features,
            max_leaf_nodes=_model_class_tuple.max_leaf_nodes,
            max_depth=_model_class_tuple.max_depth,
            min_samples_leaf=_model_class_tuple.min_samples_leaf,
            min_info_gain=_model_class_tuple.min_info_gain,
            nbins=_model_class_tuple.nbins,
        )
        vpy_score = vpy_model_obj.model.score(
            metric=vpy_metric_name[0], average="binary", pos_label=1
        )
    elif model_class == "XGBRegressor":
        vpy_model_obj = get_vpy_model(
            model_class,
            max_ntree=_model_class_tuple.max_ntree,
            max_depth=_model_class_tuple.max_depth,
            nbins=_model_class_tuple.nbins,
            learning_rate=_model_class_tuple.learning_rate,
            min_split_loss=_model_class_tuple.min_split_loss,
            weight_reg=_model_class_tuple.weight_reg,
            sample=_model_class_tuple.sample,
            col_sample_by_tree=_model_class_tuple.col_sample_by_tree,
            col_sample_by_node=_model_class_tuple.col_sample_by_node,
        )
        vpy_score = vpy_model_obj.model.score(metric=vpy_metric_name[0])
    elif model_class == "XGBClassifier":
        vpy_model_obj = get_vpy_model(
            model_class,
            max_ntree=_model_class_tuple.max_ntree,
            max_depth=_model_class_tuple.max_depth,
            nbins=_model_class_tuple.nbins,
            learning_rate=_model_class_tuple.learning_rate,
            min_split_loss=_model_class_tuple.min_split_loss,
            weight_reg=_model_class_tuple.weight_reg,
            sample=_model_class_tuple.sample,
            col_sample_by_tree=_model_class_tuple.col_sample_by_tree,
            col_sample_by_node=_model_class_tuple.col_sample_by_node,
        )
        vpy_score = vpy_model_obj.model.score(
            metric=vpy_metric_name[0], average="binary", pos_label=1
        )
    elif model_class in ["DummyTreeRegressor", "DummyTreeClassifier"]:
        vpy_model_obj = get_vpy_model(model_class)
        vpy_score = vpy_model_obj.model.score(metric=vpy_metric_name[0])
    elif model_class == "LinearSVR":
        vpy_model_obj = get_vpy_model(
            model_class,
            tol=_model_class_tuple.tol,
            c=_model_class_tuple.c,
            intercept_scaling=_model_class_tuple.intercept_scaling,
            intercept_mode=_model_class_tuple.intercept_mode,
            acceptable_error_margin=_model_class_tuple.acceptable_error_margin,
            max_iter=_model_class_tuple.max_iter,
        )
        vpy_score = vpy_model_obj.model.score(metric=vpy_metric_name[0])
    elif model_class == "PoissonRegressor":
        vpy_model_obj = get_vpy_model(
            model_class,
            penalty=_model_class_tuple.penalty,
            tol=_model_class_tuple.tol,
            C=_model_class_tuple.c,
            max_iter=_model_class_tuple.max_iter,
            solver=_model_class_tuple.solver,
            fit_intercept=_model_class_tuple.fit_intercept,
        )
        vpy_score = vpy_model_obj.model.score(metric=vpy_metric_name[0])
    elif model_class == "AR":
        vpy_model_obj = get_vpy_model(
            model_class,
            p=_model_class_tuple.order[0],
            method=_model_class_tuple.method,
            penalty=_model_class_tuple.penalty,
            C=_model_class_tuple.c,
            missing=_model_class_tuple.missing,
        )
        vpy_score = vpy_model_obj.model.score(
            metric=vpy_metric_name[0],
            start=_model_class_tuple.order[0],
            npredictions=_model_class_tuple.npredictions,
        )
    elif model_class == "MA":
        vpy_model_obj = get_vpy_model(
            model_class,
            q=_model_class_tuple.q,
            penalty=_model_class_tuple.penalty,
            C=_model_class_tuple.c,
            missing=_model_class_tuple.missing,
        )
        vpy_score = vpy_model_obj.model.score(
            metric=vpy_metric_name[0],
            start=_model_class_tuple.q,
            npredictions=_model_class_tuple.npredictions,
        )
    elif model_class in ["ARMA", "ARIMA"]:
        vpy_model_obj = get_vpy_model(
            model_class,
            order=_model_class_tuple.order,
            tol=_model_class_tuple.tol,
            max_iter=_model_class_tuple.max_iter,
            init=_model_class_tuple.init,
            missing=_model_class_tuple.missing,
        )
        vpy_score = vpy_model_obj.model.score(
            metric=vpy_metric_name[0],
            start=_model_class_tuple.order[0],
            npredictions=_model_class_tuple.npredictions,
        )
    elif model_class == "LinearRegression":
        vpy_model_obj = get_vpy_model(
            model_class,
            tol=_model_class_tuple.tol,
            max_iter=_model_class_tuple.max_iter,
            solver=_model_class_tuple.solver,
            fit_intercept=_model_class_tuple.fit_intercept,
        )
        vpy_score = vpy_model_obj.model.score(metric=vpy_metric_name[0])
    elif model_class == "ElasticNet":
        vpy_model_obj = get_vpy_model(
            model_class,
            tol=_model_class_tuple.tol,
            c=_model_class_tuple.c,
            max_iter=_model_class_tuple.max_iter,
            solver=_model_class_tuple.solver,
            l1_ratio=_model_class_tuple.l1_ratio,
            fit_intercept=_model_class_tuple.fit_intercept,
        )
        vpy_score = vpy_model_obj.model.score(metric=vpy_metric_name[0])
    elif model_class in ["Ridge", "Lasso"]:
        vpy_model_obj = get_vpy_model(
            model_class,
            tol=_model_class_tuple.tol,
            c=_model_class_tuple.c,
            max_iter=_model_class_tuple.max_iter,
            solver=_model_class_tuple.solver,
            fit_intercept=_model_class_tuple.fit_intercept,
        )
        vpy_score = vpy_model_obj.model.score(metric=vpy_metric_name[0])
    else:
        pytest.skip(f"Invalid parameters {_model_class_tuple} for {model_class}")

    if model_class in [
        "RandomForestClassifier",
        "DecisionTreeClassifier",
        "DummyTreeClassifier",
        "XGBClassifier",
    ]:
        vpy_model_obj.pred_vdf.drop(
            columns=["survived_pred"]
        )  # this is added if parameters runs in loop
    elif model_class in [
        "AR",
        "MA",
        "ARMA",
        "ARIMA",
    ]:
        vpy_model_obj.pred_vdf.drop(
            columns=["prediction"]
        )  # this is added if parameters runs in loop
    else:
        vpy_model_obj.pred_vdf.drop(
            columns=["quality_pred"]
        )  # this is added if parameters runs in loop

    # python
    if model_class in ["RandomForestRegressor"]:
        py_model_obj = get_py_model(
            model_class,
            n_estimators=_model_class_tuple.n_estimators,
            max_features=_model_class_tuple.max_features,
            max_leaf_nodes=_model_class_tuple.max_leaf_nodes,
            sample=_model_class_tuple.sample,
            max_depth=_model_class_tuple.max_depth,
            min_samples_leaf=_model_class_tuple.min_samples_leaf,
        )
        metrics_map = _metrics(model_class, model_obj=py_model_obj)
        py_score = metrics_map[py_metric_name]
    elif model_class in ["RandomForestClassifier"]:
        py_model_obj = get_py_model(
            model_class,
            n_estimators=_model_class_tuple.n_estimators,
            max_features=_model_class_tuple.max_features,
            max_leaf_nodes=_model_class_tuple.max_leaf_nodes,
            sample=_model_class_tuple.sample,
            max_depth=_model_class_tuple.max_depth,
            min_samples_leaf=_model_class_tuple.min_samples_leaf,
        )
        _metrics = _metrics(model_class, model_obj=py_model_obj)
        py_score = _metrics[py_metric_name]
    elif model_class in ["DecisionTreeRegressor"]:
        py_model_obj = get_py_model(
            model_class,
            max_features=_model_class_tuple.max_features,
            max_leaf_nodes=_model_class_tuple.max_leaf_nodes,
            max_depth=_model_class_tuple.max_depth,
            min_samples_leaf=_model_class_tuple.min_samples_leaf,
        )
        metrics_map = _metrics(model_class, model_obj=py_model_obj)
        py_score = metrics_map[py_metric_name]
    elif model_class in ["DecisionTreeClassifier"]:
        py_model_obj = get_py_model(
            model_class,
            max_features=_model_class_tuple.max_features,
            max_leaf_nodes=_model_class_tuple.max_leaf_nodes,
            max_depth=_model_class_tuple.max_depth,
            min_samples_leaf=_model_class_tuple.min_samples_leaf,
        )
        _metrics = _metrics(model_class, model_obj=py_model_obj)
        py_score = _metrics[py_metric_name]
    elif model_class in ["XGBRegressor"]:
        py_model_obj = get_py_model(
            model_class,
            n_estimators=_model_class_tuple.max_ntree,
            max_depth=_model_class_tuple.max_depth,
            max_bin=_model_class_tuple.nbins,
            learning_rate=_model_class_tuple.learning_rate,
            gamma=_model_class_tuple.min_split_loss,
            reg_alpha=_model_class_tuple.weight_reg,
            reg_lambda=_model_class_tuple.weight_reg,
            subsample=_model_class_tuple.sample,
            colsample_bytree=_model_class_tuple.col_sample_by_tree,
            colsample_bynode=_model_class_tuple.col_sample_by_node,
        )
        metrics_map = _metrics(model_class, model_obj=py_model_obj)
        py_score = metrics_map[py_metric_name]
    elif model_class in ["XGBClassifier"]:
        py_model_obj = get_py_model(
            model_class,
            n_estimators=_model_class_tuple.max_ntree,
            max_depth=_model_class_tuple.max_depth,
            max_bin=_model_class_tuple.nbins,
            learning_rate=_model_class_tuple.learning_rate,
            gamma=_model_class_tuple.min_split_loss,
            reg_alpha=_model_class_tuple.weight_reg,
            reg_lambda=_model_class_tuple.weight_reg,
            subsample=_model_class_tuple.sample,
            colsample_bytree=_model_class_tuple.col_sample_by_tree,
            colsample_bynode=_model_class_tuple.col_sample_by_node,
        )
        _metrics = _metrics(model_class, model_obj=py_model_obj)
        py_score = _metrics[py_metric_name]
    elif model_class in ["DummyTreeRegressor", "DummyTreeClassifier"]:
        py_model_obj = get_py_model(model_class)
        metrics_map = _metrics(model_class, model_obj=py_model_obj)
        py_score = metrics_map[py_metric_name]
    elif model_class in ["LinearSVR"]:
        metrics_map = _metrics(model_class, fit_intercept=True)
        py_score = metrics_map[py_metric_name]
    elif model_class in ["PoissonRegressor"]:
        metrics_map = _metrics(model_class, fit_intercept=True)
        py_score = metrics_map[py_metric_name]
    elif model_class in [
        "AR",
        "MA",
        "ARMA",
        "ARIMA",
    ]:
        py_model_obj = get_py_model(model_class, order=_model_class_tuple.order)
        metrics_map = _metrics(model_class, model_obj=py_model_obj)
        py_score = metrics_map[py_metric_name]
    else:
        metrics_map = _metrics(
            model_class, fit_intercept=_model_class_tuple.fit_intercept
        )
        py_score = metrics_map[py_metric_name]

    return vpy_score, py_score


@pytest.mark.parametrize(
    "model_class",
    [
        "RandomForestRegressor",
        "RandomForestClassifier",
        "DecisionTreeRegressor",
        "DecisionTreeClassifier",
        # "DummyTreeRegressor",
        # "DummyTreeClassifier",
        "XGBRegressor",
        "XGBClassifier",
        "Ridge",
        "Lasso",
        "ElasticNet",
        "LinearRegression",
        # "LinearSVR",
        "PoissonRegressor",
        "AR",
        "MA",
        "ARMA",
        "ARIMA",
    ],
)
# @pytest.mark.parametrize("model_class", ["XGBClassifier"])
class TestBaseModelMethods:
    """
    test class for linear models
    """

    @pytest.fixture
    def get_models(self, model_class, get_vpy_model, get_py_model):
        """
        test function - get_models
        """
        _get_models = namedtuple("_get_models", ["vpy", "py"])(
            get_vpy_model(model_class), get_py_model(model_class)
        )

        yield _get_models

    def test_predict(self, get_models, model_class):
        """
        test function - predict
        """

        if model_class in [
            "RandomForestClassifier",
            "DecisionTreeClassifier",
            "DummyTreeClassifier",
            "XGBClassifier",
        ]:
            vpy_res = (
                get_models.vpy.pred_vdf[["survived_pred"]].to_numpy().ravel().sum()
            )
            py_res = get_models.py.pred.sum()

            assert vpy_res == pytest.approx(py_res, rel=rel_tolerance_map[model_class])

        elif model_class in [
            "AR",
            "MA",
            "ARMA",
            "ARIMA",
        ]:
            assert get_models.vpy.pred_vdf[
                ["prediction"]
            ].to_numpy().mean() == pytest.approx(
                get_models.py.pred.mean(), rel=rel_tolerance_map[model_class]
            )
        else:
            assert get_models.vpy.pred_vdf[
                ["quality_pred"]
            ].to_numpy().mean() == pytest.approx(
                get_models.py.pred.mean(), rel=rel_tolerance_map[model_class]
            )

    def test_contour(self, model_class, get_vpy_model):
        """
        test function - contour
        """
        if model_class in [
            "RandomForestClassifier",
            "DecisionTreeClassifier",
            "DummyTreeClassifier",
            "XGBClassifier",
        ]:
            vpy_res = get_vpy_model(model_class, X=["age", "fare"]).model.contour()
        elif model_class in [
            "AR",
            "MA",
            "ARMA",
            "ARIMA",
        ]:
            pytest.skip(f"contour function is not available for {model_class} model")
        else:
            vpy_res = get_vpy_model(
                model_class, X=["residual_sugar", "alcohol"]
            ).model.contour()

        assert isinstance(vpy_res, (plt.Axes, plotly.graph_objs.Figure, Highchart))

    def test_deploysql(self, get_models, model_class):
        """
        test function - deploySQL
        """
        if model_class in [
            "RandomForestRegressor",
            "DecisionTreeRegressor",
            "DummyTreeRegressor",
        ]:
            pred_fun_name = "PREDICT_RF_REGRESSOR"
        elif model_class in [
            "RandomForestClassifier",
            "DecisionTreeClassifier",
            "DummyTreeClassifier",
        ]:
            pred_fun_name = "PREDICT_RF_CLASSIFIER"
        elif model_class == "XGBRegressor":
            pred_fun_name = "PREDICT_XGB_REGRESSOR"
        elif model_class == "XGBClassifier":
            pred_fun_name = "PREDICT_XGB_CLASSIFIER"
        elif model_class == "LinearSVR":
            pred_fun_name = "PREDICT_SVM_REGRESSOR"
        elif model_class == "PoissonRegressor":
            pred_fun_name = "PREDICT_POISSON_REG"
        elif model_class == "AR":
            pred_fun_name = "PREDICT_AUTOREGRESSOR"
        elif model_class == "MA":
            pred_fun_name = "PREDICT_MOVING_AVERAGE"
        elif model_class in ["ARMA", "ARIMA"]:
            pred_fun_name = "PREDICT_ARIMA"
        else:
            pred_fun_name = "PREDICT_LINEAR_REG"

        vpy_pred_sql = get_models.vpy.model.deploySQL()
        if model_class in ["AR", "MA", "ARMA", "ARIMA"]:
            pred_sql = f"""{pred_fun_name}( USING PARAMETERS model_name = '{get_models.vpy.schema_name}.{get_models.vpy.model_name}', add_mean = True, npredictions = 10 ) OVER ()"""
        else:
            pred_sql = f"""{pred_fun_name}("{get_models.py.X.columns[0]}", "{get_models.py.X.columns[1]}", "{get_models.py.X.columns[2]}" USING PARAMETERS model_name = '{get_models.vpy.schema_name}.{get_models.vpy.model_name}', match_by_pos = 'true')"""

        assert vpy_pred_sql == pred_sql

    def test_drop(self, get_models):
        """
        test function - drop
        """
        model_sql = f"SELECT model_name FROM models WHERE schema_name='{get_models.vpy.schema_name}' and model_name = '{get_models.vpy.model_name}'"

        current_cursor().execute(model_sql)
        assert current_cursor().fetchone()[0] == get_models.vpy.model_name

        get_models.vpy.model.drop()
        current_cursor().execute(model_sql)
        assert current_cursor().fetchone() is None

    def test_get_attributes(self, get_models, model_class):
        """
        test function - get_attributes
        """
        if model_class in [
            "RandomForestRegressor",
            "DecisionTreeRegressor",
            "DummyTreeRegressor",
        ]:
            vpy_model_attributes = [
                "n_estimators_",
                "trees_",
                "features_importance_",
                "features_importance_trees_",
            ]
        elif model_class == "XGBRegressor":
            vpy_model_attributes = [
                "n_estimators_",
                "eta_",
                "mean_",
                "trees_",
                "features_importance_",
                "features_importance_trees_",
            ]
        elif model_class in [
            "RandomForestClassifier",
            "DecisionTreeClassifier",
            "DummyTreeClassifier",
        ]:
            vpy_model_attributes = [
                "n_estimators_",
                "classes_",
                "trees_",
                "features_importance_",
                "features_importance_trees_",
            ]
        elif model_class == "XGBClassifier":
            vpy_model_attributes = [
                "n_estimators_",
                "classes_",
                "eta_",
                "logodds_",
                "trees_",
                "features_importance_",
                "features_importance_trees_",
            ]
        elif model_class == "AR":
            vpy_model_attributes = [
                "phi_",
                "intercept_",
                "features_importance_",
                "mse_",
                "n_",
            ]
        elif model_class == "MA":
            vpy_model_attributes = [
                "theta_",
                "mu_",
                "mean_",
                "mse_",
                "n_",
            ]
        elif model_class == "ARMA":
            vpy_model_attributes = [
                "phi_",
                "theta_",
                "mean_",
                "features_importance_",
                "mse_",
                "n_",
            ]
        elif model_class == "ARIMA":
            vpy_model_attributes = [
                "phi_",
                "theta_",
                "mean_",
                "features_importance_",
                "mse_",
                "n_",
            ]
        else:
            vpy_model_attributes = ["coef_", "intercept_", "features_importance_"]

        assert get_models.vpy.model.get_attributes() == vpy_model_attributes

    def test_get_params(self, get_models, model_class):
        """
        test function - get_params
        """
        if model_class in ["RandomForestRegressor", "RandomForestClassifier"]:
            model_params_map = {
                "n_estimators": 10,
                "max_features": 2,
                "max_leaf_nodes": 10,
                "sample": 0.632,
                "max_depth": 10,
                "min_samples_leaf": 1,
                "min_info_gain": 0.0,
                "nbins": 32,
            }
        elif model_class in ["DecisionTreeRegressor", "DecisionTreeClassifier"]:
            model_params_map = {
                "max_features": 2,
                "max_leaf_nodes": 10,
                "max_depth": 10,
                "min_samples_leaf": 1,
                "min_info_gain": 0.0,
                "nbins": 32,
            }
        elif model_class in ["XGBRegressor", "XGBClassifier"]:
            model_params_map = {
                "max_ntree": 10,
                "max_depth": 10,
                "nbins": 150,
                "split_proposal_method": "global",
                "tol": 0.001,
                "learning_rate": 0.1,
                "min_split_loss": 0.0,
                "weight_reg": 0.0,
                "sample": 1.0,
                "col_sample_by_tree": 1.0,
                "col_sample_by_node": 1.0,
            }
        elif model_class in ["DummyTreeRegressor", "DummyTreeClassifier"]:
            model_params_map = {}
        elif model_class == "LinearSVR":
            model_params_map = {
                "tol": 1e-04,
                "C": 1.0,
                "intercept_scaling": 1.0,
                "intercept_mode": "regularized",
                "acceptable_error_margin": 0.1,
                "max_iter": 100,
            }
        elif model_class == "PoissonRegressor":
            model_params_map = {
                "penalty": "l2",
                "tol": 1e-06,
                "C": 1,
                "max_iter": 100,
                "solver": "newton",
                "fit_intercept": True,
            }
        elif model_class == "AR":
            model_params_map = {
                "p": 3,
                "method": "ols",
                "penalty": "none",
                "C": 1.0,
                "missing": "linear_interpolation",
            }
        elif model_class == "MA":
            model_params_map = {
                "q": 1,
                "penalty": "none",
                "C": 1.0,
                "missing": "linear_interpolation",
            }
        elif model_class == "ARMA":
            model_params_map = {
                "order": (2, 1),
                "tol": 1e-06,
                "max_iter": 100,
                "init": "zero",
                "missing": "linear_interpolation",
            }
        elif model_class == "ARIMA":
            model_params_map = {
                "order": (2, 1, 1),
                "tol": 1e-06,
                "max_iter": 100,
                "init": "zero",
                "missing": "linear_interpolation",
            }
        else:
            model_params_map = {
                "tol": 1e-06,
                "max_iter": 100,
                "solver": "newton",
                "fit_intercept": True,
            }

        if model_class == "Ridge":
            model_params_map["C"] = 1
        elif model_class == "Lasso":
            model_params_map["C"] = 1
            model_params_map["solver"] = "cgd"
        elif model_class == "ElasticNet":
            model_params_map["C"] = 1
            model_params_map["solver"] = "cgd"
            model_params_map["l1_ratio"] = 0.5

        assert get_models.vpy.model.get_params() == pytest.approx(
            model_params_map, rel=REL_TOLERANCE
        )

    @pytest.mark.parametrize(
        "attributes, expected",
        [
            (
                "attr_name",
                [
                    "details",
                    "regularization",
                    "iteration_count",
                    "rejected_row_count",
                    "accepted_row_count",
                    "call_string",
                ],
            ),
            (
                "attr_fields",
                [
                    "predictor, coefficient, std_err, t_value, p_value",
                    "type, lambda",
                    "iteration_count",
                    "rejected_row_count",
                    "accepted_row_count",
                    "call_string",
                ],
            ),
            ("#_of_rows", [4, 1, 1, 1, 1, 1]),
        ],
    )
    def test_get_vertica_attributes(
        self, get_models, model_class, attributes, expected
    ):
        """
        test function - get_vertica_attributes
        """
        model_attributes = get_models.vpy.model.get_vertica_attributes()

        if model_class in [
            "RandomForestRegressor",
            "RandomForestClassifier",
            "DecisionTreeRegressor",
            "DecisionTreeClassifier",
            "DummyTreeRegressor",
            "DummyTreeClassifier",
            "XGBRegressor",
            "XGBClassifier",
        ]:
            attr_map = {
                "attr_name": [
                    "tree_count",
                    "rejected_row_count",
                    "accepted_row_count",
                    "call_string",
                    "details",
                ],
                "attr_fields": [
                    "tree_count",
                    "rejected_row_count",
                    "accepted_row_count",
                    "call_string",
                    "predictor, type",
                ],
                "#_of_rows": [1, 1, 1, 1, 3],
            }
            expected = attr_map[attributes]
            if model_class == "XGBRegressor":
                attr_map["attr_name"].append("initial_prediction")
                attr_map["attr_fields"].append("initial_prediction")
                attr_map["#_of_rows"].append(1)
            elif model_class == "XGBClassifier":
                attr_map["attr_name"].append("initial_prediction")
                attr_map["attr_fields"].append("response_label, value")
                attr_map["#_of_rows"].append(2)

        elif model_class == "LinearSVR":
            attr_map = {
                "attr_name": [
                    "details",
                    "accepted_row_count",
                    "rejected_row_count",
                    "iteration_count",
                    "call_string",
                ],
                "attr_fields": [
                    "predictor, coefficient",
                    "accepted_row_count",
                    "rejected_row_count",
                    "iteration_count",
                    "call_string",
                ],
                "#_of_rows": [4, 1, 1, 1, 1],
            }
            expected = attr_map[attributes]
        elif model_class == "PoissonRegressor":
            attr_map = {
                "attr_name": [
                    "details",
                    "regularization",
                    "iteration_count",
                    "rejected_row_count",
                    "accepted_row_count",
                    "call_string",
                ],
                "attr_fields": [
                    "predictor, coefficient, std_err, z_value, p_value",
                    "type, lambda",
                    "iteration_count",
                    "rejected_row_count",
                    "accepted_row_count",
                    "call_string",
                ],
                "#_of_rows": [4, 1, 1, 1, 1, 1],
            }
            expected = attr_map[attributes]
        elif model_class == "AR":
            attr_map = {
                "attr_name": [
                    "coefficients",
                    "lag_order",
                    "lambda",
                    "mean_squared_error",
                    "rejected_row_count",
                    "accepted_row_count",
                    "timeseries_name",
                    "timestamp_name",
                    "missing_method",
                    "call_string",
                ],
                "attr_fields": [
                    "parameter, value",
                    "lag_order",
                    "lambda",
                    "mean_squared_error",
                    "rejected_row_count",
                    "accepted_row_count",
                    "timeseries_name",
                    "timestamp_name",
                    "missing_method",
                    "call_string",
                ],
                "#_of_rows": [4, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            }
            expected = attr_map[attributes]
        elif model_class == "MA":
            attr_map = {
                "attr_name": [
                    "coefficients",
                    "mean",
                    "lag_order",
                    "lambda",
                    "mean_squared_error",
                    "rejected_row_count",
                    "accepted_row_count",
                    "timeseries_name",
                    "timestamp_name",
                    "missing_method",
                    "call_string",
                ],
                "attr_fields": [
                    "parameter, value",
                    "mean",
                    "lag_order",
                    "lambda",
                    "mean_squared_error",
                    "rejected_row_count",
                    "accepted_row_count",
                    "timeseries_name",
                    "timestamp_name",
                    "missing_method",
                    "call_string",
                ],
                "#_of_rows": [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            }
            expected = attr_map[attributes]
        elif model_class in ["ARMA", "ARIMA"]:
            attr_map = {
                "attr_name": [
                    "coefficients",
                    "p",
                    "d",
                    "q",
                    "mean",
                    "regularization",
                    "lambda",
                    "mean_squared_error",
                    "rejected_row_count",
                    "accepted_row_count",
                    "timeseries_name",
                    "timestamp_name",
                    "missing_method",
                    "call_string",
                ],
                "attr_fields": [
                    "parameter, value",
                    "p",
                    "d",
                    "q",
                    "mean",
                    "regularization",
                    "lambda",
                    "mean_squared_error",
                    "rejected_row_count",
                    "accepted_row_count",
                    "timeseries_name",
                    "timestamp_name",
                    "missing_method",
                    "call_string",
                ],
                "#_of_rows": [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            }
            expected = attr_map[attributes]

        assert model_attributes[attributes] == expected

    def test_set_params(self, get_models, model_class):
        """
        test function - set_params
        """

        if model_class in ["RandomForestRegressor", "RandomForestClassifier"]:
            params = {"n_estimators": 100, "max_depth": 50, "nbins": 100}
        elif model_class in ["DecisionTreeRegressor", "DecisionTreeClassifier"]:
            params = {"max_depth": 50, "nbins": 100}
        elif model_class in ["XGBRegressor", "XGBClassifier"]:
            params = {"max_depth": 50, "nbins": 100}
        elif model_class in ["DummyTreeRegressor", "DummyTreeClassifier"]:
            params = {}
        elif model_class == "LinearSVR":
            params = {"intercept_mode": "unregularized", "max_iter": 500}
        elif model_class == "ElasticNet":
            params = {"l1_ratio": 0.01, "C": 0.12, "solver": "newton", "max_iter": 500}
        elif model_class == "AR":
            params = {"p": 10, "C": 0.12, "penalty": "l2", "missing": "drop"}
        elif model_class == "MA":
            params = {"q": 10, "C": 0.12, "penalty": "l2", "missing": "drop"}
        elif model_class == "ARMA":
            params = {"order": (2, 4, 2), "tol": 1, "init": "hr", "missing": "drop"}
        elif model_class == "ARIMA":
            params = {"order": (2, 2), "tol": 1, "init": "hr", "missing": "drop"}
        else:
            params = {"solver": "cgd", "max_iter": 500}

        get_models.vpy.model.set_params(params)

        assert {
            k: get_models.vpy.model.get_params()[k] for k in params
        } == pytest.approx(params)

    def test_summarize(self, get_models):
        """
        test function - summarize
        """
        vpy_model_summary = get_models.vpy.model.summarize()
        vt_model_summary_sql = f"SELECT GET_MODEL_SUMMARY(USING PARAMETERS MODEL_NAME='{get_models.vpy.schema_name}.{get_models.vpy.model_name}')"
        vt_model_summary = (
            current_cursor().execute(vt_model_summary_sql).fetchall()[0][0]
        )

        assert vpy_model_summary == vt_model_summary

    def test_to_python(
        self,
        get_models,
        model_class,
    ):
        """
        test function - to_python
        """
        if model_class in [
            "RandomForestClassifier",
            "DecisionTreeClassifier",
            "DummyTreeClassifier",
            "XGBClassifier",
        ]:
            py_res = get_models.vpy.model.to_python()(get_models.py.X)[10]
            vpy_res = get_models.vpy.pred_vdf[["survived_pred"]].to_numpy()[10]
        elif model_class in ["AR", "MA", "ARMA", "ARIMA"]:
            pytest.skip(f"to_python function is not available for {model_class} model")
        else:
            py_res = get_models.vpy.model.to_python()(get_models.py.X)[10]
            vpy_res = get_models.vpy.pred_vdf[["quality_pred"]].to_numpy()[10]

        assert vpy_res == pytest.approx(
            np.exp(py_res) if model_class == "PoissonRegressor" else py_res,
            rel=rel_tolerance_map[model_class],
        )

    def test_to_sql(self, get_models, model_class):
        """
        test function - to_sql
        """
        pred_fun_map = {
            **dict.fromkeys(
                [
                    "RandomForestRegressor",
                    "DecisionTreeRegressor",
                    "DummyTreeRegressor",
                ],
                "PREDICT_RF_REGRESSOR",
            ),
            **dict.fromkeys(
                [
                    "RandomForestClassifier",
                    "DecisionTreeClassifier",
                    "DummyTreeClassifier",
                ],
                "PREDICT_RF_CLASSIFIER",
            ),
            "XGBRegressor": "PREDICT_XGB_REGRESSOR",
            "XGBClassifier": "PREDICT_XGB_CLASSIFIER",
            "LinearSVR": "PREDICT_SVM_REGRESSOR",
            "PoissonRegressor": "PREDICT_POISSON_REG",
            "AR": "PREDICT_AUTOREGRESSOR",
            **dict.fromkeys(
                ["Ridge", "Lasso", "ElasticNet", "LinearRegression"],
                "PREDICT_LINEAR_REG",
            ),
        }

        if model_class in [
            "RandomForestClassifier",
            "DecisionTreeClassifier",
            "DummyTreeClassifier",
            "XGBClassifier",
        ]:
            pred_sql = f"SELECT {pred_fun_map[model_class]}(* USING PARAMETERS model_name = '{get_models.vpy.model.model_name}', match_by_pos=True)::int, {get_models.vpy.model.to_sql()}::int FROM (SELECT 30.0 AS age, 45.0 AS fare, 'male' AS sex) x"
        elif model_class in ["AR", "MA", "ARMA", "ARIMA"]:
            pytest.skip(f"to_sql function is not available for {model_class} model")
        else:
            pred_sql = f"SELECT {pred_fun_map[model_class]}(3.0, 11.0, 93.0 USING PARAMETERS model_name = '{get_models.vpy.model.model_name}', match_by_pos=True)::float, {get_models.vpy.model.to_sql([3.0, 11.0, 93.0])}::float"

        current_cursor().execute(pred_sql)
        prediction = current_cursor().fetchone()
        assert prediction[0] == pytest.approx(
            np.exp(prediction[1])
            if model_class == "PoissonRegressor"
            else prediction[1]
        )

    def test_does_model_exists(self, get_models):
        """
        test function - does_model_exists
        """
        model_name_with_schema = (
            f"{get_models.vpy.schema_name}.{get_models.vpy.model_name}"
        )
        assert (
            get_models.vpy.model.does_model_exists(name=model_name_with_schema) is True
        )

        try:
            get_models.vpy.model.does_model_exists(
                name=model_name_with_schema, raise_error=True
            )
        except NameError as error:
            assert error.args[0] == "The model 'vpy_model' already exists!"

        assert get_models.vpy.model.does_model_exists(
            name=model_name_with_schema, return_model_type=True
        )[1] in [
            "LINEAR_REGRESSION",
            "SVM_REGRESSOR",
            "RF_REGRESSOR",
            "RF_CLASSIFIER",
            "XGB_REGRESSOR",
            "XGB_CLASSIFIER",
            "POISSON_REGRESSION",
            "AUTOREGRESSOR",
            "MOVING_AVERAGE",
            "ARIMA",
        ]

        get_models.vpy.model.drop()
        assert (
            get_models.vpy.model.does_model_exists(name=model_name_with_schema) is False
        )

    @pytest.mark.parametrize(
        "match_index_attr, expected", [("valid_colum", 2), ("invalid_colum", None)]
    )
    def test_get_match_index(self, get_models, match_index_attr, expected, model_class):
        """
        test function - get_match_index
        """
        if model_class in ["AR", "MA", "ARMA", "ARIMA"]:
            x = get_models.py.X.columns[0]
            col_list = [x]
            expected = 0 if match_index_attr == "valid_colum" else expected
        else:
            x = get_models.py.X.columns[2]
            col_list = get_models.py.X.columns

        if match_index_attr == "valid_colum":
            vpy_res = get_models.vpy.model.get_match_index(x=x, col_list=col_list)
        else:
            vpy_res = get_models.vpy.model.get_match_index(
                x=match_index_attr, col_list=col_list
            )

        assert vpy_res == expected

    def test_get_plotting_lib(self, get_models):
        """
        test function - get_plotting_lib
        """
        plotting_lib = get_models.vpy.model.get_plotting_lib(
            class_name="RegressionPlot"
        )[0].RegressionPlot.__module__
        assert (
            "matplotlib" in plotting_lib
            or "plotly" in plotting_lib
            or "highcharts" in plotting_lib
        )

    @pytest.mark.parametrize(
        "key_name",
        [
            "index",
            "importance",
            "sign",
        ],
    )
    def test_features_importance(self, get_vpy_model, model_class, key_name):
        """
        test function - features_importance
        """
        features_importance_map = {"sign": [1, 1, 1]}
        _X = None

        if model_class in [
            "RandomForestClassifier",
            "DecisionTreeClassifier",
            "DummyTreeClassifier",
            "XGBClassifier",
        ]:
            features_importance_map["index"] = [
                "sex",
                "fare",
                "age",
            ]
            if model_class in ["RandomForestClassifier"]:
                # features_importance_map["index"] = [
                #     "sex",
                #     "fare",
                #     "age",
                # ]
                features_importance_map["importance"] = [74.4, 12.88, 12.72]
            elif model_class in ["DecisionTreeClassifier"]:
                # features_importance_map["index"] = [
                #     "sex",
                #     "age",
                #     "fare",
                # ]
                features_importance_map["importance"] = [76.4, 12.41, 11.19]
            elif model_class in ["XGBClassifier"]:
                # features_importance_map["index"] = [
                #     "sex",
                #     "fare",
                #     "age",
                # ]
                features_importance_map["importance"] = [97.88, 1.39, 0.73]
            elif model_class in ["DummyTreeClassifier"]:
                features_importance_map["index"] = [
                    "fare",
                    "sex",
                    "age",
                ]
                features_importance_map["importance"] = [38.9, 36.61, 24.49]
        elif model_class in [
            "RandomForestRegressor",
            "DecisionTreeRegressor",
            "XGBRegressor",
        ]:
            features_importance_map["index"] = [
                "alcohol",
                "citric_acid",
                "residual_sugar",
            ]
            if model_class in ["RandomForestRegressor"]:
                features_importance_map["importance"] = [82.67, 12.91, 4.42]
            elif model_class in ["DecisionTreeRegressor"]:
                features_importance_map["importance"] = [83.65, 11.37, 4.98]
            elif model_class in ["XGBRegressor"]:
                features_importance_map["index"] = [
                    "alcohol",
                    "residual_sugar",
                    "citric_acid",
                ]
                features_importance_map["importance"] = [64.86, 17.58, 17.57]
        elif model_class == "AR":
            features_importance_map["index"] = [
                '""passengers""[t-1]',
                '""passengers""[t-2]',
                '""passengers""[t-3]',
            ]
            features_importance_map["importance"] = [
                0.62945595267205,
                0.27631678771967194,
                0.09422725960827791,
            ]
            features_importance_map["sign"] = [1.0, -1.0, 1.0]
        elif model_class == "MA":
            pytest.skip("Features Importance can not be computed for Moving Averages")
        elif model_class in ["ARMA", "ARIMA"]:
            features_importance_map["index"] = [
                '""passengers""[t-1]',
                '""passengers""[t-2]',
            ]
            features_importance_map["importance"] = [
                0.6018412076652686,
                0.3981587923347315,
            ]
            features_importance_map["sign"] = (
                [1.0, 1.0] if model_class == "ARMA" else [1.0, -1.0]
            )
        else:
            features_importance_map["index"] = [
                "alcohol",
                "residual_sugar",
                "citric_acid",
            ]
            if model_class in ["DummyTreeRegressor"]:
                # features_importance_map["index"] = [
                #     "alcohol",
                #     "residual_sugar",
                #     "citric_acid",
                # ]
                features_importance_map["importance"] = [37.61, 36.56, 25.83]
            elif model_class == "Ridge":
                # features_importance_map["index"] = ['alcohol', 'residual_sugar', 'citric_acid']
                features_importance_map["importance"] = [52.3, 32.63, 15.07]
            elif model_class == "LinearRegression":
                # features_importance_map["index"] = ['alcohol', 'residual_sugar', 'citric_acid']
                features_importance_map["importance"] = [52.25, 32.58, 15.17]
            elif model_class == "LinearSVR":
                # features_importance_map["index"] = ['alcohol', 'residual_sugar', 'citric_acid']
                features_importance_map["importance"] = [52.68, 33.27, 14.05]
            elif model_class == "PoissonRegressor":
                # features_importance_map["index"] = ['alcohol', 'residual_sugar', 'citric_acid']
                features_importance_map["importance"] = [51.94, 32.48, 15.58]
            elif model_class in ["Lasso", "ElasticNet"]:
                _X = features_importance_map["index"] = [
                    "total_sulfur_dioxide",
                    "residual_sugar",
                    "alcohol",
                ]
                features_importance_map["importance"] = [100, 0, 0]
                features_importance_map["sign"] = [-1, 0, 0]

        f_imp = get_vpy_model(model_class, X=_X).model.features_importance(show=False)
        # print(f_imp[key_name])

        assert features_importance_map[key_name] == pytest.approx(
            f_imp[key_name], rel=1e-0
        )

    def test_plot(self, model_class, get_vpy_model):
        """
        test function - plot
        """
        try:
            if model_class in [
                "RandomForestClassifier",
                "DecisionTreeClassifier",
                "DummyTreeClassifier",
                "XGBClassifier",
            ]:
                vpy_res = get_vpy_model(model_class, X=["age", "fare"])[0].plot()
            elif model_class in [
                "AR",
                "MA",
                "ARMA",
                "ARIMA",
            ]:
                vpy_res = get_vpy_model(model_class, X="date")[0].plot(
                    start=5
                )  # need to remove start parm. once 847 issue is fixed
            else:
                vpy_res = get_vpy_model(model_class, X=["residual_sugar", "alcohol"])[
                    0
                ].plot()
        except NotImplementedError:
            pytest.skip(
                f"plot function is not implemented for {model_class} model class - NotImplementedError"
            )

        assert isinstance(vpy_res, (plt.Axes, plotly.graph_objs.Figure, Highchart))

    def test_to_memmodel(self, get_models, model_class):
        """
        test function - to_mmmodel
        """
        if model_class in [
            "AR",
            "MA",
            "ARMA",
            "ARIMA",
        ]:
            pytest.skip(
                f"to_memmodel function is not available for {model_class} model"
            )
        else:
            mmodel = get_models.vpy.model.to_memmodel()
            mm_res = mmodel.predict(get_models.py.X)
            py_res = get_models.vpy.model.to_python()(get_models.py.X)

        assert mm_res == pytest.approx(py_res, rel=REL_TOLERANCE)
