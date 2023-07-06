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
from collections import namedtuple
import math
import verticapy.machine_learning.vertica as vpy_linear_model
import verticapy.machine_learning.vertica.svm as vpy_svm
from verticapy.connection import current_cursor
import numpy as np
import sklearn.metrics as skl_metrics
import sklearn.linear_model as skl_linear_model
import sklearn.svm as skl_svm
import pytest
import statsmodels.api as sm
from scipy import stats
from scipy.stats import f
import matplotlib.pyplot as plt
from vertica_highcharts.highcharts.highcharts import Highchart
import plotly

REL_TOLERANCE = 1e-6
ABS_TOLERANCE = 1e-12

regression_metrics_args = (
    "vpy_metric_name, py_metric_name, _rel_tolerance",
    [
        (("explained_variance",), "explained_variance_score", 1e-2),
        (("max_error",), "max_error", 1e-2),
        (("median_absolute_error",), "median_absolute_error", 1e-2),
        (("mean_absolute_error",), "mean_absolute_error", 1e-2),
        (("mean_squared_error",), "mean_squared_error", 1e-2),
        (("mean_squared_log_error",), "mean_squared_log_error", 1e-2),
        (
            (
                "rmse",
                "root_mean_squared_error",
            ),
            "rmse",
            1e-2,
        ),
        (("r2",), "r2_score", 1e-2),
        (("r2_adj",), "rsquared_adj", 1e-2),
        (("aic",), "aic", 1e-2),
        (("bic",), "bic", 1e-2),
    ],
)


def calculate_regression_metrics(linear_model_name, get_py_model, fit_intercept=True):
    """
    function to calculate python metrics
    """
    X, y, _, skl_pred, skl_model = get_py_model(
        linear_model_name, py_fit_intercept=fit_intercept
    )
    regression_metrics_map = {}
    n = len(y)
    avg = sum(y) / n
    num_features = k = len(skl_model.feature_names_in_)
    num_params = len(skl_model.coef_) + 1

    regression_metrics_map["mse"] = getattr(skl_metrics, "mean_squared_error")(
        y, skl_pred
    )
    regression_metrics_map["rmse"] = np.sqrt(regression_metrics_map["mse"])
    regression_metrics_map["ssr"] = sum(np.square(skl_pred - avg))
    regression_metrics_map["sse"] = sum(np.square(y - skl_pred))
    regression_metrics_map["dfr"] = k
    regression_metrics_map["dfe"] = n - k - 1
    regression_metrics_map["msr"] = (
        regression_metrics_map["ssr"] / regression_metrics_map["dfr"]
    )
    regression_metrics_map["_mse"] = (
        regression_metrics_map["sse"] / regression_metrics_map["dfe"]
    )
    regression_metrics_map["f"] = (
        regression_metrics_map["msr"] / regression_metrics_map["_mse"]
    )
    regression_metrics_map["p_value"] = f.sf(regression_metrics_map["f"], k, n)
    regression_metrics_map["mean_squared_log_error"] = (
        sum(
            pow(
                (np.log10(skl_pred + 1) - np.log10(y + 1)),
                2,
            )
        )
        / n
    )
    regression_metrics_map["r2"] = regression_metrics_map[
        "r2_score"
    ] = skl_metrics.r2_score(y, skl_pred)
    regression_metrics_map["rsquared_adj"] = 1 - (1 - regression_metrics_map["r2"]) * (
        n - 1
    ) / (n - num_features - 1)
    regression_metrics_map["aic"] = (
        n * math.log(regression_metrics_map["mse"]) + 2 * num_params
    )
    regression_metrics_map["bic"] = n * math.log(
        regression_metrics_map["mse"]
    ) + num_params * math.log(n)
    regression_metrics_map["explained_variance_score"] = getattr(
        skl_metrics, "explained_variance_score"
    )(y, skl_pred)
    regression_metrics_map["max_error"] = getattr(skl_metrics, "max_error")(y, skl_pred)
    regression_metrics_map["median_absolute_error"] = getattr(
        skl_metrics, "median_absolute_error"
    )(y, skl_pred)
    regression_metrics_map["mean_absolute_error"] = getattr(
        skl_metrics, "mean_absolute_error"
    )(y, skl_pred)
    regression_metrics_map["mean_squared_error"] = getattr(
        skl_metrics, "mean_squared_error"
    )(y, skl_pred)
    regression_metrics_map[""] = ""

    return regression_metrics_map


@pytest.fixture(name="get_vpy_model", scope="function")
def get_vpy_model_fixture(winequality_vpy_fun, schema_loader):
    """
    getter function for vertica model
    """

    def _get_vpy_model(
        linear_model_name,
        y_true=None,
        vpy_tol=None,
        vpy_c=None,
        vpy_max_iter=None,
        vpy_solver=None,
        vpy_fit_intercept=None,
        # ElasticNet
        vpy_l1_ratio=None,
        # LinearSVR
        vpy_intercept_scaling=None,
        vpy_intercept_mode=None,
        vpy_acceptable_error_margin=None,
    ):
        schema_name, model_name = schema_loader, "vpy_lr_model"

        if vpy_solver:
            vpy_solver = vpy_solver
        else:
            if linear_model_name in ["Lasso", "ElasticNet"]:
                vpy_solver = "cgd"
            else:
                vpy_solver = "Newton"

        if linear_model_name == "LinearSVR":
            vpy_model = getattr(vpy_svm, linear_model_name)(
                f"{schema_name}.{model_name}",
                tol=vpy_tol if vpy_tol else 1e-4,
                C=vpy_c if vpy_c else 1.0,
                intercept_scaling=vpy_intercept_scaling
                if vpy_intercept_scaling
                else 1.0,
                intercept_mode=vpy_intercept_mode
                if vpy_intercept_mode
                else "regularized",
                acceptable_error_margin=vpy_acceptable_error_margin
                if vpy_acceptable_error_margin
                else 0.1,
                max_iter=vpy_max_iter if vpy_max_iter else 100,
            )
        elif linear_model_name == "LinearRegression":
            vpy_model = getattr(vpy_linear_model, linear_model_name)(
                f"{schema_name}.{model_name}",
                tol=vpy_tol if vpy_tol else 1e-6,
                max_iter=vpy_max_iter if vpy_max_iter else 100,
                solver=vpy_solver,
                fit_intercept=vpy_fit_intercept if vpy_fit_intercept else True,
            )
        elif linear_model_name == "ElasticNet":
            vpy_model = getattr(vpy_linear_model, linear_model_name)(
                f"{schema_name}.{model_name}",
                tol=vpy_tol if vpy_tol else 1e-6,
                C=vpy_c if vpy_c else 1.0,
                max_iter=vpy_max_iter if vpy_max_iter else 100,
                solver=vpy_solver,
                l1_ratio=vpy_l1_ratio if vpy_l1_ratio else 0.5,
                fit_intercept=vpy_fit_intercept if vpy_fit_intercept else True,
            )
        else:
            vpy_model = getattr(vpy_linear_model, linear_model_name)(
                f"{schema_name}.{model_name}",
                tol=vpy_tol if vpy_tol else 1e-6,
                C=vpy_c if vpy_c else 1.0,
                max_iter=vpy_max_iter if vpy_max_iter else 100,
                solver=vpy_solver,
                fit_intercept=vpy_fit_intercept if vpy_fit_intercept else True,
            )

        print(f"VerticaPy Training Parameters: {vpy_model.get_params()}")
        vpy_model.drop()

        if y_true is None:
            y_true = ["citric_acid", "residual_sugar", "alcohol"]
        vpy_model.fit(
            f"{schema_name}.winequality",
            y_true,
            "quality",
        )
        vpy_pred_vdf = vpy_model.predict(winequality_vpy_fun, name="quality_pred")

        return vpy_model, vpy_pred_vdf, schema_name, model_name

    yield _get_vpy_model


@pytest.fixture(name="get_py_model", scope="function")
def get_py_model_fixture(winequality_vpy_fun):
    """
    getter function for python model
    """

    def _get_py_model(linear_model_name, py_fit_intercept=None):
        # sklearn
        winequality_pdf = winequality_vpy_fun.to_pandas()
        winequality_pdf["citric_acid"] = winequality_pdf["citric_acid"].astype(float)
        winequality_pdf["residual_sugar"] = winequality_pdf["residual_sugar"].astype(
            float
        )

        X = winequality_pdf[["citric_acid", "residual_sugar", "alcohol"]]
        y = winequality_pdf["quality"]

        if linear_model_name == "LinearSVR":
            obj_name = skl_svm
        else:
            obj_name = skl_linear_model

        skl_model = getattr(obj_name, linear_model_name)(
            fit_intercept=py_fit_intercept if py_fit_intercept else True
        )

        print(f"Python Training Parameters: {skl_model.get_params()}")
        skl_model.fit(X, y)

        # num_params = len(skl_model.coef_) + 1
        skl_pred = skl_model.predict(X)

        # statsmodels
        # add constant to predictor variables
        X_sm = sm.add_constant(X)

        # fit linear regression model
        sm_model = sm.OLS(y, X_sm).fit()

        return X, y, sm_model, skl_pred, skl_model

    return _get_py_model


# @pytest.mark.parametrize("linear_model_name", ["LinearSVR"])
# @pytest.mark.parametrize("linear_model_name", ["Ridge", "Lasso", "ElasticNet", "LinearRegression", "LinearSVR"])
@pytest.mark.parametrize("linear_model_name", ["Ridge", "Lasso", "ElasticNet", "LinearRegression"])
class TestLinearModel:
    """
    test class for linear models
    """

    @pytest.fixture
    def get_models(self, linear_model_name, get_vpy_model, get_py_model):
        vpy_model, vpy_pred_vdf, schema_name, model_name = get_vpy_model(
            linear_model_name
        )
        X, y, sm_model, skl_pred, skl_model = get_py_model(linear_model_name)

        vpy = namedtuple(
            "vertica_models", ["vpy_model", "vpy_pred_vdf", "schema_name", "model_name"]
        )(vpy_model, vpy_pred_vdf, schema_name, model_name)
        py = namedtuple(
            "python_models", ["X", "y", "sm_model", "skl_pred", "skl_model"]
        )(X, y, sm_model, skl_pred, skl_model)

        _get_models = namedtuple("_get_models", ["vpy", "py"])(vpy, py)

        yield _get_models

    def test_predict(self, get_models):
        """
        test function - predict
        """

        assert get_models.vpy.vpy_pred_vdf[
            ["quality_pred"]
        ].to_numpy().ravel() == pytest.approx(get_models.py.skl_pred, rel=REL_TOLERANCE)

    @pytest.mark.parametrize("fun_name", ["regression", "report"])
    def test_regression_report_none(
        self, get_models, linear_model_name, get_py_model, fun_name
    ):
        """
        test function - regression/report None
        """
        reg_rep = (
            get_models.vpy.vpy_model.report()
            if fun_name == "report"
            else get_models.vpy.vpy_model.regression_report()
        )
        vpy_reg_rep_map = dict(zip(reg_rep["index"], reg_rep["value"]))

        regression_metrics_map = calculate_regression_metrics(
            linear_model_name, get_py_model
        )

        for vpy_metric in reg_rep["index"]:
            idx = [
                list(set(v) & set(reg_rep["index"]))[0]
                if set(v) & set(reg_rep["index"])
                else ""
                for v, p, _ in regression_metrics_args[1]
            ].index(vpy_metric)
            py_metric_name = regression_metrics_args[1][idx][1]
            vpy_score = vpy_reg_rep_map[vpy_metric]
            py_score = regression_metrics_map[py_metric_name]
            _rel_tolerance = regression_metrics_args[1][idx][2]

            print(
                f"Metric Name: {vpy_metric, py_metric_name}, vertica: {vpy_score}, sklearn: {py_score}"
            )
            assert vpy_score == pytest.approx(
                py_score,
                rel=_rel_tolerance,
            )

    @pytest.mark.parametrize(
        "metric, expected, _rel_tolerance, _abs_tolerance",
        [
            ("Dep. Variable", '"quality"', "NA", "NA"),
            ("Model", "LinearRegression", "NA", "NA"),
            ("No. Observations", None, REL_TOLERANCE, ABS_TOLERANCE),
            ("No. Predictors", None, REL_TOLERANCE, ABS_TOLERANCE),
            ("R-squared", None, REL_TOLERANCE, ABS_TOLERANCE),
            ("Adj. R-squared", None, REL_TOLERANCE, ABS_TOLERANCE),
            ("F-statistic", None, REL_TOLERANCE, 1e-9),
            ("Prob (F-statistic)", None, REL_TOLERANCE, ABS_TOLERANCE),
            ("Kurtosis", None, 1e-2, ABS_TOLERANCE),
            ("Skewness", None, 1e-3, ABS_TOLERANCE),
            ("Jarque-Bera (JB)", None, 1e-2, ABS_TOLERANCE),
        ],
    )
    @pytest.mark.parametrize("fun_name", ["regression", "report"])
    def test_regression_report_details(
        self,
        get_models,
        linear_model_name,
        get_py_model,
        fun_name,
        metric,
        expected,
        _rel_tolerance,
        _abs_tolerance,
    ):
        """
        test function - regression/report details
        """
        reg_rep_details = (
            get_models.vpy.vpy_model.report(metrics="details")
            if fun_name == "report"
            else get_models.vpy.vpy_model.regression_report(metrics="details")
        )
        vpy_reg_rep_details_map = dict(
            zip(reg_rep_details["index"], reg_rep_details["value"])
        )

        regression_metrics_map = calculate_regression_metrics(
            linear_model_name, get_py_model
        )

        if metric == "No. Observations":
            py_res = len(get_models.py.y)
        elif metric == "No. Predictors":
            py_res = len(get_models.py.X.columns)
        elif metric == "R-squared":
            py_res = regression_metrics_map["r2_score"]
        elif metric == "Adj. R-squared":
            py_res = regression_metrics_map["rsquared_adj"]
        elif metric == "F-statistic":
            py_res = regression_metrics_map["f"]
        elif metric == "Prob (F-statistic)":
            py_res = regression_metrics_map["p_value"]
        elif metric == "Kurtosis":
            py_res = stats.kurtosis(get_models.py.y)
        elif metric == "Skewness":
            py_res = stats.skew(get_models.py.y)
        elif metric == "Jarque-Bera (JB)":
            py_res = stats.jarque_bera(get_models.py.y).statistic
        else:
            py_res = expected

        if py_res == 0:
            assert vpy_reg_rep_details_map[metric] == pytest.approx(
                py_res, abs=_abs_tolerance
            )
        else:
            assert vpy_reg_rep_details_map[metric] == pytest.approx(
                py_res, rel=_rel_tolerance
            )

    @pytest.mark.parametrize(
        "metric, metric_types",
        [
            ("df", ("dfr", "dfe")),
            ("ss", ("ssr", "sse")),
            ("ms", ("msr", "mse")),
            ("f", ("f", "")),
            ("p_value", ("p_value", "")),
        ],
    )
    @pytest.mark.parametrize("fun_name", ["regression", "report"])
    def test_regression_report_anova(
        self,
        get_models,
        linear_model_name,
        get_py_model,
        fun_name,
        metric,
        metric_types,
    ):
        """
        test function - regression/report anova
        """
        reg_rep_anova = (
            get_models.vpy.vpy_model.report(metrics="anova")
            if fun_name == "report"
            else get_models.vpy.vpy_model.regression_report(metrics="anova")
        )

        regression_metrics_map = calculate_regression_metrics(
            linear_model_name, get_py_model
        )

        for vpy_res, metric_type in zip(reg_rep_anova[metric], metric_types):
            py_res = regression_metrics_map[metric_type]

            if py_res == 0:
                assert vpy_res == pytest.approx(py_res, abs=1e-9)
            else:
                assert vpy_res == pytest.approx(py_res, rel=1e-3)

    @pytest.mark.parametrize("fit_attr", ["coef_", "intercept_", "score"])
    def test_fit(self, get_models, fit_attr):
        """
        test function - fit
        """
        if fit_attr == "score":
            vpy_res = getattr(get_models.vpy.vpy_model, fit_attr)()
            py_res = getattr(get_models.py.skl_model, fit_attr)(
                get_models.py.X, get_models.py.y
            )
        else:
            vpy_res = getattr(get_models.vpy.vpy_model, fit_attr)
            py_res = getattr(get_models.py.skl_model, fit_attr)

        assert vpy_res == pytest.approx(py_res, rel=REL_TOLERANCE)

    def test_contour(self, linear_model_name, get_vpy_model):
        """
        test function - contour
        """
        vpy_res = get_vpy_model(
            linear_model_name, y_true=["residual_sugar", "alcohol"]
        )[0].contour()

        assert (
            isinstance(vpy_res, plt.Axes)
            or isinstance(vpy_res, plotly.graph_objs.Figure)
            or isinstance(vpy_res, Highchart)
        )

    def test_deploysql(self, get_models, linear_model_name):
        """
        test function - deploySQL
        """
        if linear_model_name == "LinearSVR":
            pred_fun_name = "PREDICT_SVM_REGRESSOR"
        else:
            pred_fun_name = "PREDICT_LINEAR_REG"

        vpy_pred_sql = get_models.vpy.vpy_model.deploySQL()
        pred_sql = f"""{pred_fun_name}("{get_models.py.X.columns[0]}", "{get_models.py.X.columns[1]}", "{get_models.py.X.columns[2]}" USING PARAMETERS model_name = '{get_models.vpy.schema_name}.{get_models.vpy.model_name}', match_by_pos = 'true')"""

        assert vpy_pred_sql == pred_sql

    def test_drop(self, get_models):
        """
        test function - drop
        """
        model_sql = f"SELECT model_name FROM models WHERE schema_name='{get_models.vpy.schema_name}' and model_name = '{get_models.vpy.model_name}'"

        current_cursor().execute(model_sql)
        assert current_cursor().fetchone()[0] == get_models.vpy.model_name

        get_models.vpy.vpy_model.drop()
        current_cursor().execute(model_sql)
        assert current_cursor().fetchone() is None

    def test_get_attributes(self, get_models):
        """
        test function - get_attributes
        """

        assert get_models.vpy.vpy_model.get_attributes() == []

    def test_get_params(self, get_models, linear_model_name):
        """
        test function - get_params
        """

        if linear_model_name == "LinearSVR":
            vpy_linear_model_params_map = {
                "tol": 1e-04,
                "C": 1.0,
                "intercept_scaling": 1.0,
                "intercept_mode": "regularized",
                "acceptable_error_margin": 0.1,
                "max_iter": 100,
            }
        else:
            vpy_linear_model_params_map = {
                "tol": 1e-06,
                "max_iter": 100,
                "solver": "newton",
                "fit_intercept": True,
            }

        if linear_model_name == "Ridge":
            vpy_linear_model_params_map["C"] = 1
        elif linear_model_name == "Lasso":
            vpy_linear_model_params_map["C"] = 1
            vpy_linear_model_params_map["solver"] = "cgd"
        elif linear_model_name == "ElasticNet":
            vpy_linear_model_params_map["C"] = 1
            vpy_linear_model_params_map["solver"] = "cgd"
            vpy_linear_model_params_map["l1_ratio"] = 0.5

        assert get_models.vpy.vpy_model.get_params() == pytest.approx(
            vpy_linear_model_params_map, rel=REL_TOLERANCE
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
        self, get_models, linear_model_name, attributes, expected
    ):
        """
        test function - get_vertica_attributes
        """
        svr_attr_map = {
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
        model_attributes = get_models.vpy.vpy_model.get_vertica_attributes()

        if linear_model_name == "LinearSVR":
            expected = svr_attr_map[attributes]

        assert model_attributes[attributes] == expected

    def test_set_params(self, get_models, linear_model_name):
        """
        test function - set_params
        """

        if linear_model_name == "LinearSVR":
            params = {"intercept_mode": "unregularized", "max_iter": 500}
        elif linear_model_name == "ElasticNet":
            params = {"l1_ratio": 0.01, "C": 0.12, "solver": "newton", "max_iter": 500}
        else:
            params = {"solver": "cgd", "max_iter": 500}

        get_models.vpy.vpy_model.set_params(params)

        assert {
            k: get_models.vpy.vpy_model.get_params()[k] for k in params
        } == pytest.approx(params)

    def test_summarize(self, get_models):
        """
        test function - summarize
        """
        vpy_model_summary = get_models.vpy.vpy_model.summarize()
        vt_model_summary_sql = f"SELECT GET_MODEL_SUMMARY(USING PARAMETERS MODEL_NAME='{get_models.vpy.schema_name}.{get_models.vpy.model_name}')"
        vt_model_summary = (
            current_cursor().execute(vt_model_summary_sql).fetchall()[0][0]
        )

        assert vpy_model_summary == vt_model_summary

    def test_to_python(self, get_models):
        """
        test function - to_python
        """
        py_res = get_models.vpy.vpy_model.to_python(return_proba=False)(get_models.py.X)
        vpy_res = get_models.vpy.vpy_pred_vdf[["quality_pred"]].to_numpy().ravel()

        assert vpy_res == pytest.approx(py_res, rel=REL_TOLERANCE)

    def test_to_sql(self, get_models, linear_model_name):
        """
        test function - to_sql
        """
        if linear_model_name == "LinearSVR":
            pred_fun_name = "PREDICT_SVM_REGRESSOR"
        else:
            pred_fun_name = "PREDICT_LINEAR_REG"

        pred_sql = f"SELECT {pred_fun_name}(3.0, 11.0, 93. USING PARAMETERS model_name = '{get_models.vpy.vpy_model.model_name}', match_by_pos=True)::float, {get_models.vpy.vpy_model.to_sql([3.0, 11.0, 93.0])}::float"

        current_cursor().execute(pred_sql)
        prediction = current_cursor().fetchone()
        assert prediction[0] == pytest.approx(prediction[1])

    def test_does_model_exists(self, get_models):
        """
        test function - does_model_exists
        """
        model_name_with_schema = (
            f"{get_models.vpy.schema_name}.{get_models.vpy.model_name}"
        )
        assert (
            get_models.vpy.vpy_model.does_model_exists(name=model_name_with_schema)
            is True
        )

        try:
            get_models.vpy.vpy_model.does_model_exists(
                name=model_name_with_schema, raise_error=True
            )
        except NameError as e:
            assert e.args[0] == "The model 'vpy_lr_model' already exists !"

        assert get_models.vpy.vpy_model.does_model_exists(
            name=model_name_with_schema, return_model_type=True
        ) in ["LINEAR_REGRESSION", "SVM_REGRESSOR"]

        get_models.vpy.vpy_model.drop()
        assert (
            get_models.vpy.vpy_model.does_model_exists(name=model_name_with_schema)
            is False
        )

    @pytest.mark.parametrize(
        "match_index_attr, expected", [("valid_colum", 2), ("invalid_colum", None)]
    )
    def test_get_match_index(self, get_models, match_index_attr, expected):
        """
        test function - get_match_index
        """
        if match_index_attr == "valid_colum":
            vpy_res = get_models.vpy.vpy_model.get_match_index(
                x=get_models.py.X.columns[2], col_list=get_models.py.X.columns
            )
        else:
            vpy_res = get_models.vpy.vpy_model.get_match_index(
                x=match_index_attr, col_list=get_models.py.X.columns
            )

        assert vpy_res == expected

    def test_get_plotting_lib(self, get_models):
        """
        test function - get_plotting_lib
        """
        plotting_lib = get_models.vpy.vpy_model.get_plotting_lib(
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
    def test_features_importance(self, get_vpy_model, linear_model_name, key_name):
        """
        test function - features_importance
        """
        features_importance_map = {
            "index": ["alcohol", "residual_sugar", "citric_acid"],
            "sign": [1, 1, 1],
        }

        if linear_model_name == "Ridge":
            features_importance_map["importance"] = [52.3, 32.63, 15.07]
        elif linear_model_name in ["Lasso", "ElasticNet"]:
            features_importance_map["index"] = [
                "total_sulfur_dioxide",
                "residual_sugar",
                "alcohol",
            ]
            features_importance_map["importance"] = [100, 0, 0]
            features_importance_map["sign"] = [-1, 0, 0]
        elif linear_model_name == "LinearRegression":
            features_importance_map["importance"] = [52.25, 32.58, 15.17]
        elif linear_model_name == "LinearSVR":
            features_importance_map["importance"] = [52.68, 33.27, 14.05]

        f_imp = get_vpy_model(
            linear_model_name, y_true=features_importance_map["index"]
        )[0].features_importance(show=False)

        assert features_importance_map[key_name] == f_imp[key_name]

    def test_plot(self, linear_model_name, get_vpy_model):
        """
        test function - plot
        """
        vpy_res = get_vpy_model(
            linear_model_name, y_true=["residual_sugar", "alcohol"]
        )[0].plot()

        assert (
            isinstance(vpy_res, plt.Axes)
            or isinstance(vpy_res, plotly.graph_objs.Figure)
            or isinstance(vpy_res, Highchart)
        )

    def test_to_memmodel(self, get_models):
        """
        test function - to_mmmodel
        """
        mmodel = get_models.vpy.vpy_model.to_memmodel()
        mm_res = mmodel.predict(get_models.py.X)
        py_res = get_models.vpy.vpy_model.to_python()(get_models.py.X)

        assert mm_res == pytest.approx(py_res, rel=REL_TOLERANCE)

    @pytest.mark.parametrize(*regression_metrics_args)
    @pytest.mark.parametrize(
        "tol, c, max_iter, solver, l1_ratio, fit_intercept, intercept_scaling, intercept_mode, acceptable_error_margin",
        [
            # Ridge/Lasso/ElasticNet/LinearRegression
            (1e-6, 1, 20, "bfgs", 1, False, 1.0, "regularized", 0.1),
            (1e-6, 0, 2000, "bfgs", 0.5, True, 2.0, "unregularized", 0.1),
            (1e-6, 5, 50, "newton", 0, True, 0.6, "regularized", 0.1),
            (1e-6, 2, 1000, "newton", 0.1, False, 0, "unregularized", 0.1),
            (1e-4, 3, 40, "bfgs", 0.9, False, 0.9, "regularized", 0.1),
            (0.99, 0.001, 20, "newton", 0.0001, True, 0.01, "regularized", 0.1),
            (1e-10, 0.99, 20, "bfgs", 0.99, True, 0.9, "regularized", 0.1),
            # Only Lasso/ElasticNet
            (1e-6, 1.0001, 20, "cgd", 0.5, True, 0.1, "regularized", 0.5),
            (1e-6, 9.1111, 20, "cgd", 1.0, False, 0.6, "unregularized", 1),
            (0.99, 0, 20, "cgd", 0, True, 0.80, "regularized", 0),
            (1e-10, 100, 20, "cgd", 0.05, False, 0.0, "unregularized", 0.25),
        ],
    )
    def test_score(
        self,
        linear_model_name,
        get_vpy_model,
        get_py_model,
        tol,
        c,
        max_iter,
        solver,
        l1_ratio,
        fit_intercept,
        intercept_scaling,
        intercept_mode,
        acceptable_error_margin,
        vpy_metric_name,
        py_metric_name,
        _rel_tolerance,
    ):
        """
        test function - score
        """
        # skipping a test
        if (linear_model_name in ["Ridge", "LinearRegression"] and solver == "cgd") or (
            linear_model_name in ["Lasso", "ElasticNet"]
            and solver
            in [
                "bfgs",
                "newton",
            ]
        ):
            pytest.skip(
                f"optimizer [{solver}] is not supported for [{linear_model_name}] model"
            )

        if linear_model_name == "LinearSVR":
            vpy_score = get_vpy_model(
                linear_model_name,
                vpy_tol=tol,
                vpy_c=c,
                vpy_intercept_scaling=intercept_scaling,
                vpy_intercept_mode=intercept_mode,
                vpy_acceptable_error_margin=acceptable_error_margin,
                vpy_max_iter=max_iter,
            )[0].score(metric=vpy_metric_name[0])
        elif linear_model_name == "LinearRegression":
            vpy_score = get_vpy_model(
                linear_model_name,
                vpy_tol=tol,
                vpy_max_iter=max_iter,
                vpy_solver=solver,
                vpy_fit_intercept=fit_intercept,
            )[0].score(metric=vpy_metric_name[0])
        elif linear_model_name == "ElasticNet":
            vpy_score = get_vpy_model(
                linear_model_name,
                vpy_tol=tol,
                vpy_c=c,
                vpy_max_iter=max_iter,
                vpy_solver=solver,
                vpy_l1_ratio=l1_ratio,
                vpy_fit_intercept=fit_intercept,
            )[0].score(metric=vpy_metric_name[0])
        else:
            vpy_score = get_vpy_model(
                linear_model_name,
                vpy_tol=tol,
                vpy_c=c,
                vpy_max_iter=max_iter,
                vpy_solver=solver,
                vpy_fit_intercept=fit_intercept,
            )[0].score(metric=vpy_metric_name[0])

        # python
        regression_metrics_map = calculate_regression_metrics(
            linear_model_name, get_py_model, fit_intercept=fit_intercept
        )
        py_score = regression_metrics_map[py_metric_name]

        print(
            f"Metric Name: {py_metric_name}, vertica: {vpy_score}, sklearn: {py_score}"
        )
        assert vpy_score == pytest.approx(py_score, rel=_rel_tolerance)
