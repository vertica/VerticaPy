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
import math
import verticapy.machine_learning.vertica as vpy_linear_model
from verticapy.connection import current_cursor
import numpy as np
import sklearn.metrics as skl_metrics
import sklearn.linear_model as skl_linear_model
import pytest
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
from vertica_highcharts.highcharts.highcharts import Highchart
import plotly

REL_TOLERANCE = 1e-5

regression_metrics_args = (
    "vpy_metric_name, py_metric_name, _rel_tolerance",
    [
        (("explained_variance",), "explained_variance_score", REL_TOLERANCE),
        (("max_error",), "max_error", REL_TOLERANCE),
        (("median_absolute_error",), "median_absolute_error", 1e-2),
        (("mean_absolute_error",), "mean_absolute_error", REL_TOLERANCE),
        (("mean_squared_error",), "mean_squared_error", REL_TOLERANCE),
        (("mean_squared_log_error",), "mean_squared_log_error", REL_TOLERANCE),
        (
            (
                "rmse",
                "root_mean_squared_error",
            ),
            "rmse",
            REL_TOLERANCE,
        ),
        (("r2",), "r2_score", REL_TOLERANCE),
        (("r2_adj",), "rsquared_adj", REL_TOLERANCE),
        (("aic",), "aic", REL_TOLERANCE),
        (("bic",), "bic", REL_TOLERANCE),
    ],
)


def calculate_python_regression_metrics(
    get_py_model, metric_name="r2_score", fit_intercept=True
):
    """
    function to calculate python metrics
    """
    _, y, sm_model, skl_pred, skl_model = get_py_model(py_fit_intercept=fit_intercept)
    mse = getattr(skl_metrics, "mean_squared_error")(y, skl_pred)
    num_params = len(skl_model.coef_) + 1
    n = len(y)

    if metric_name == "rmse":
        score = np.sqrt(mse)
    elif metric_name == "mean_squared_log_error":
        score = (
            sum(
                pow(
                    (np.log10(skl_pred + 1) - np.log10(y + 1)),
                    2,
                )
            )
            / n
        )
    elif metric_name == "rsquared_adj":
        score = sm_model.rsquared_adj
    elif metric_name == "aic":
        score = n * math.log(mse) + 2 * num_params
    elif metric_name == "bic":
        score = n * math.log(mse) + num_params * math.log(n)
    else:
        score = getattr(skl_metrics, metric_name)(y, skl_pred)

    return score


@pytest.fixture(name="get_vpy_model", scope="function")
def get_vpy_model_fixture(winequality_vpy_fun, schema_loader):
    """
    getter function for vertica model
    """

    def _get_vpy_model(
        y_true=None,
        vpy_tol=None,
        vpy_max_iter=None,
        vpy_solver=None,
        vpy_fit_intercept=None,
    ):
        schema_name, model_name = schema_loader, "vpy_lr_model"
        vpy_model = vpy_linear_model.LinearRegression(
            f"{schema_name}.{model_name}",
            tol=vpy_tol if vpy_tol else 1e-6,
            max_iter=vpy_max_iter if vpy_max_iter else 100,
            solver=vpy_solver if vpy_solver else "Newton",
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

    def _get_py_model(py_fit_intercept=None):
        # sklearn
        winequality_pdf = winequality_vpy_fun.to_pandas()
        winequality_pdf["citric_acid"] = winequality_pdf["citric_acid"].astype(float)
        winequality_pdf["residual_sugar"] = winequality_pdf["residual_sugar"].astype(
            float
        )

        X = winequality_pdf[["citric_acid", "residual_sugar", "alcohol"]]
        y = winequality_pdf["quality"]

        skl_model = skl_linear_model.LinearRegression(
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


class TestLinearRegressionMethods:
    """
    test class for linear regression
    """

    def test_predict(self, get_vpy_model, get_py_model):
        """
        test function - predict
        """
        _, vpy_pred_vdf, _, _ = get_vpy_model()
        _, _, _, skl_pred, _ = get_py_model()

        assert vpy_pred_vdf[["quality_pred"]].to_numpy().ravel() == pytest.approx(
            skl_pred, rel=REL_TOLERANCE
        )

    @pytest.mark.parametrize("report_metric", [None, "details", "anova"])
    def test_regression_report(
        self, get_vpy_model, get_py_model, report_metric, fun_name=None
    ):
        """
        test function - regression report
        """
        # regression report values are tested as part of regression metrics test -
        # /VerticaPy/verticapy/tests_new/machine_learning/metrics/test_regression_metrics.py
        # Only, extra metrics would be tested here

        vpy_model, _, _, _ = get_vpy_model()
        X, y, sm_model, _, _ = get_py_model()

        # ****************** regression report *********************************************
        if report_metric is None:
            reg_rep = (
                vpy_model.report()
                if fun_name == "report"
                else vpy_model.regression_report()
            )
            vpy_reg_rep_map = dict(zip(reg_rep["index"], reg_rep["value"]))
            for vpy_metric in reg_rep["index"]:
                idx = [
                    list(set(v) & set(reg_rep["index"]))[0]
                    if set(v) & set(reg_rep["index"])
                    else ""
                    for v, p, _ in regression_metrics_args[1]
                ].index(vpy_metric)
                py_metric_name = regression_metrics_args[1][idx][1]
                vpy_score = vpy_reg_rep_map[vpy_metric]
                py_score = calculate_python_regression_metrics(
                    get_py_model, metric_name=py_metric_name
                )
                _rel_tolerance = regression_metrics_args[1][idx][2]

                print(
                    f"Metric Name: {vpy_metric, py_metric_name}, vertica: {vpy_score}, sklearn: {py_score}"
                )
                assert vpy_score == pytest.approx(
                    py_score,
                    rel=_rel_tolerance,
                )
        # ****************** detailed report *********************************************
        elif report_metric == "details":
            reg_rep_details = (
                vpy_model.report(metrics=report_metric)
                if fun_name == "report"
                else vpy_model.regression_report(metrics=report_metric)
            )
            vpy_reg_rep_details_map = dict(
                zip(reg_rep_details["index"], reg_rep_details["value"])
            )

            py_reg_rep_details_map = {
                "Dep. Variable": '"quality"',
                "Model": "LinearRegression",
                "No. Observations": len(y),
                "No. Predictors": len(X.columns),
                "R-squared": sm_model.rsquared,
                "Adj. R-squared": sm_model.rsquared_adj,
                "F-statistic": sm_model.fvalue,
                "Prob (F-statistic)": sm_model.f_pvalue,
                "Kurtosis": stats.kurtosis(y),
                "Skewness": stats.skew(y),
                "Jarque-Bera (JB)": stats.jarque_bera(y).statistic,
            }

            assert vpy_reg_rep_details_map == pytest.approx(
                py_reg_rep_details_map, rel=1e-2
            )
            # ****************** anova report *********************************************
        else:
            vpy_reg_rep_anova_map, py_reg_rep_anova_map = {}, {}
            reg_rep_anova = (
                vpy_model.report(metrics=report_metric)
                if fun_name == "report"
                else vpy_model.regression_report(metrics=report_metric)
            )
            for metric in ["Df", "SS", "MS", "F", "p_value"]:
                (
                    vpy_reg_rep_anova_map[f"model_{metric}"],
                    vpy_reg_rep_anova_map[f"residual_{metric}"],
                    _,
                ) = reg_rep_anova[metric]

                # python stats model
                if metric == "Df":
                    (
                        py_reg_rep_anova_map[f"model_{metric}"],
                        py_reg_rep_anova_map[f"residual_{metric}"],
                    ) = (sm_model.df_model, sm_model.df_resid)
                elif metric == "SS":
                    (
                        py_reg_rep_anova_map[f"model_{metric}"],
                        py_reg_rep_anova_map[f"residual_{metric}"],
                    ) = (sm_model.ess, sm_model.ssr)
                elif metric == "MS":
                    (
                        py_reg_rep_anova_map[f"model_{metric}"],
                        py_reg_rep_anova_map[f"residual_{metric}"],
                    ) = (sm_model.mse_model, sm_model.mse_resid)
                elif metric == "F":
                    (
                        py_reg_rep_anova_map[f"model_{metric}"],
                        py_reg_rep_anova_map[f"residual_{metric}"],
                    ) = (sm_model.fvalue, "")
                elif metric == "p_value":
                    (
                        py_reg_rep_anova_map[f"model_{metric}"],
                        py_reg_rep_anova_map[f"residual_{metric}"],
                    ) = (sm_model.f_pvalue, "")

            assert vpy_reg_rep_anova_map == pytest.approx(
                py_reg_rep_anova_map, rel=REL_TOLERANCE
            )

    @pytest.mark.parametrize("report_metric", [None, "details", "anova"])
    def test_report(self, get_vpy_model, get_py_model, report_metric):
        """
        test function - report
        """
        self.test_regression_report(
            get_vpy_model, get_py_model, report_metric, fun_name="report"
        )

    @pytest.mark.parametrize("fit_attr", ["coef_", "intercept_", "score"])
    def test_fit(self, get_vpy_model, get_py_model, fit_attr):
        """
        test function - fir
        """
        vpy_model, _, _, _ = get_vpy_model()
        X, y, _, _, skl_model = get_py_model()

        if fit_attr == "score":
            vpy_res = getattr(vpy_model, fit_attr)()
            py_res = getattr(skl_model, fit_attr)(X, y)
        else:
            vpy_res = getattr(vpy_model, fit_attr)
            py_res = getattr(skl_model, fit_attr)

        assert vpy_res == pytest.approx(py_res, rel=REL_TOLERANCE)

    def test_contour(self, get_vpy_model):
        """
        test function - contour
        """
        vpy_res = get_vpy_model(y_true=["residual_sugar", "alcohol"])[0].contour()

        assert (
            isinstance(vpy_res, plt.Axes)
            or isinstance(vpy_res, plotly.graph_objs.Figure)
            or isinstance(vpy_res, Highchart)
        )

    def test_deploysql(self, get_vpy_model, get_py_model):
        """
        test function - deploySQL
        """
        vpy_model, _, schema_name, model_name = get_vpy_model()
        X, _, _, _, _ = get_py_model()

        vpy_pred_sql = vpy_model.deploySQL()
        pred_sql = f"""PREDICT_LINEAR_REG("{X.columns[0]}", "{X.columns[1]}", "{X.columns[2]}" USING PARAMETERS model_name = '{schema_name}.{model_name}', match_by_pos = 'true')"""

        assert vpy_pred_sql == pred_sql

    def test_drop(self, get_vpy_model):
        """
        test function - drop
        """
        vpy_model, _, schema_name, model_name = get_vpy_model()
        model_sql = f"SELECT model_name FROM models WHERE schema_name='{schema_name}' and model_name = '{model_name}'"

        current_cursor().execute(model_sql)
        assert current_cursor().fetchone()[0] == model_name

        vpy_model.drop()
        current_cursor().execute(model_sql)
        assert current_cursor().fetchone() is None

    def test_get_attributes(self, get_vpy_model):
        """
        test function - get_attributes
        """
        vpy_model, _, _, _ = get_vpy_model()

        assert vpy_model.get_attributes() == []

    def test_get_params(self, get_vpy_model):
        """
        test function - get_params
        """
        vpy_model, _, _, _ = get_vpy_model()

        vpy_lr_parms_map = {
            "tol": 1e-06,
            "max_iter": 100,
            "solver": "newton",
            "fit_intercept": True,
        }

        assert vpy_model.get_params() == pytest.approx(
            vpy_lr_parms_map, rel=REL_TOLERANCE
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
    def test_get_vertica_attributes(self, get_vpy_model, attributes, expected):
        """
        test function - get_vertica_attributes
        """
        vpy_model, _, _, _ = get_vpy_model()
        model_attributes = vpy_model.get_vertica_attributes()

        assert model_attributes[attributes] == expected

    def test_set_params(self, get_vpy_model):
        """
        test function - set_params
        """
        vpy_model, _, _, _ = get_vpy_model()

        params = {"solver": "cgd", "max_iter": 500}
        vpy_model.set_params(params)

        assert {k: vpy_model.get_params()[k] for k in params} == pytest.approx(params)

    def test_summarize(self, get_vpy_model):
        """
        test function - summarize
        """
        vpy_model, _, schema_name, model_name = get_vpy_model()

        vpy_model_summary = vpy_model.summarize()
        vt_model_summary_sql = f"SELECT GET_MODEL_SUMMARY(USING PARAMETERS MODEL_NAME='{schema_name}.{model_name}')"
        vt_model_summary = (
            current_cursor().execute(vt_model_summary_sql).fetchall()[0][0]
        )

        assert vpy_model_summary == vt_model_summary

    def test_to_python(self, get_vpy_model, get_py_model):
        """
        test function - to_python
        """
        vpy_model, vpy_pred_vdf, _, _ = get_vpy_model()
        X, _, _, _, _ = get_py_model()

        py_res = vpy_model.to_python(return_proba=False)(X)
        vpy_res = vpy_pred_vdf[["quality_pred"]].to_numpy().ravel()

        assert vpy_res == pytest.approx(py_res, rel=REL_TOLERANCE)

    def test_to_sql(self, get_vpy_model):
        """
        test function - to_sql
        """
        vpy_model, _, _, _ = get_vpy_model()

        current_cursor().execute(
            f"SELECT PREDICT_LINEAR_REG(3.0, 11.0, 93. USING PARAMETERS model_name = '{vpy_model.model_name}', match_by_pos=True)::float, {vpy_model.to_sql([3.0, 11.0, 93.0])}::float"
        )
        prediction = current_cursor().fetchone()
        assert prediction[0] == pytest.approx(prediction[1])

    def test_does_model_exists(self, get_vpy_model):
        """
        test function - does_model_exists
        """
        vpy_model, _, schema_name, model_name = get_vpy_model()

        model_name_with_schema = f"{schema_name}.{model_name}"
        assert vpy_model.does_model_exists(name=model_name_with_schema) is True

        try:
            vpy_model.does_model_exists(name=model_name_with_schema, raise_error=True)
        except NameError as e:
            assert e.args[0] == "The model 'vpy_lr_model' already exists!"

        assert (
            vpy_model.does_model_exists(
                name=model_name_with_schema, return_model_type=True
            )
            == "LINEAR_REGRESSION"
        )

        vpy_model.drop()
        assert vpy_model.does_model_exists(name=model_name_with_schema) is False

    @pytest.mark.parametrize(
        "match_index_attr, expected", [("valid_colum", 2), ("invalid_colum", None)]
    )
    def test_get_match_index(
        self, get_vpy_model, get_py_model, match_index_attr, expected
    ):
        """
        test function - get_match_index
        """
        vpy_model, _, _, _ = get_vpy_model()
        X, _, _, _, _ = get_py_model()

        if match_index_attr == "valid_colum":
            vpy_res = vpy_model.get_match_index(x=X.columns[2], col_list=X.columns)
        else:
            vpy_res = vpy_model.get_match_index(x=match_index_attr, col_list=X.columns)

        assert vpy_res == expected

    def test_get_plotting_lib(self, get_vpy_model):
        """
        test function - get_plotting_lib
        """
        plotting_lib = (
            get_vpy_model()[0]
            .get_plotting_lib(class_name="RegressionPlot")[0]
            .RegressionPlot.__module__
        )
        assert (
            "matplotlib" in plotting_lib
            or "plotly" in plotting_lib
            or "highcharts" in plotting_lib
        )

    @pytest.mark.parametrize(
        "key_name, expected",
        [
            ("index", ["alcohol", "residual_sugar", "citric_acid"]),
            ("importance", [52.25, 32.58, 15.17]),
            ("sign", [1, 1, 1]),
        ],
    )
    def test_features_importance(self, get_vpy_model, key_name, expected):
        """
        test function - features_importance
        """
        f_imp = get_vpy_model()[0].features_importance(show=False)

        assert f_imp[key_name] == expected

    def test_plot(self, get_vpy_model):
        """
        test function - plot
        """
        vpy_res = get_vpy_model(y_true=["residual_sugar", "alcohol"])[0].plot()

        assert (
            isinstance(vpy_res, plt.Axes)
            or isinstance(vpy_res, plotly.graph_objs.Figure)
            or isinstance(vpy_res, Highchart)
        )

    def test_to_memmodel(self, get_vpy_model, get_py_model):
        """
        test function - to_mmmodel
        """
        vpy_model, _, _, _ = get_vpy_model()
        X, _, _, _, _ = get_py_model()

        mmodel = vpy_model.to_memmodel()
        mm_res = mmodel.predict(X)
        py_res = vpy_model.to_python()(X)

        assert mm_res == pytest.approx(py_res, rel=REL_TOLERANCE)

    @pytest.mark.parametrize(*regression_metrics_args)
    @pytest.mark.parametrize(
        "tol, max_iter, solver, fit_intercept",
        [
            (1e-6, 20, "bfgs", False),
            (1e-6, 2000, "bfgs", True),
            (1e-6, 50, "newton", True),
            (1e-6, 1000, "newton", False),
            (1e-2, 40, "bfgs", False),
            (0.99, 20, "newton", True),
            (1e-10, 20, "bfgs", True),
        ],
    )
    def test_score(
        self,
        get_vpy_model,
        get_py_model,
        tol,
        max_iter,
        solver,
        fit_intercept,
        vpy_metric_name,
        py_metric_name,
        _rel_tolerance,
    ):
        """
        test function - score
        """
        vpy_score = get_vpy_model(
            vpy_tol=tol,
            vpy_max_iter=max_iter,
            vpy_solver=solver,
            vpy_fit_intercept=fit_intercept,
        )[0].score(metric=vpy_metric_name[0])

        # python
        py_score = calculate_python_regression_metrics(
            get_py_model, metric_name=py_metric_name, fit_intercept=fit_intercept
        )

        print(
            f"Metric Name: {py_metric_name}, vertica: {vpy_score}, sklearn: {py_score}"
        )
        assert vpy_score == pytest.approx(py_score, rel=_rel_tolerance)
