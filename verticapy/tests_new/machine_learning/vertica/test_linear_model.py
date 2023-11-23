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
import pytest
import sklearn.metrics as skl_metrics
from verticapy.tests_new.machine_learning.vertica.test_base_model_methods import (
    rel_tolerance_map,
    regression_metrics_args,
    model_params,
    model_score,
    anova_report_args,
    details_report_args,
    regression_report_none,
    regression_report_details,
    regression_report_anova,
)


@pytest.mark.parametrize(
    "model_class",
    [
        "Ridge",
        "Lasso",
        "ElasticNet",
        "LinearRegression",
        # "LinearSVR",
        "PoissonRegressor",
        "AR",
        # "MA",
        "ARMA",
        # "ARIMA",
    ],
)
class TestLinearModel:
    """
    test class - TestLinearModel
    """

    @pytest.mark.parametrize(
        "fit_attr, ts_fit_attr, py_ts_fit_attr",
        [
            ("coef_", "phi_", "arparams"),
            ("intercept_", "intercept_", "params"),
            ("score", "mse_", "mse"),
        ],
    )
    def test_fit(
        self,
        get_vpy_model,
        get_py_model,
        model_class,
        fit_attr,
        ts_fit_attr,
        py_ts_fit_attr,
        model_params,
    ):
        """
        test function - fit
        """
        vpy_model_obj, py_model_obj = (
            get_vpy_model(model_class),
            get_py_model(model_class),
        )

        if model_class in ["AR", "MA", "ARMA", "ARIMA"]:
            _model_class_tuple = (
                None
                if model_class in ["DummyTreeRegressor", "DummyTreeClassifier"]
                else namedtuple(model_class, model_params[0])(*model_params[1][0])
            )
            if ts_fit_attr == "phi_":
                vpy_res = getattr(vpy_model_obj.model, ts_fit_attr)
                py_res = list(getattr(py_model_obj.model, py_ts_fit_attr))
            elif ts_fit_attr == "intercept_":
                if model_class in ["AR"]:
                    pytest.skip(
                        f"statsmodels ARIMA has high intercept value then independent AutoReg{model_class} function"
                    )
                elif model_class in ["ARMA", "ARIMA"]:
                    pytest.skip(
                        f"Vertica {model_class} model does not have {ts_fit_attr} attributes"
                    )
                else:
                    vpy_res = getattr(vpy_model_obj.model, ts_fit_attr)
                    py_res = list(getattr(py_model_obj.model, py_ts_fit_attr))[0]
            else:
                # mse in summary table does not match with manual calculation for ARIMA.
                # so, using manual calculation to match with python
                vpy_model_obj = get_vpy_model(model_class)
                vpy_res = vpy_model_obj.model.score(
                    start=_model_class_tuple.order[2]
                    if model_class == "MA"
                    else _model_class_tuple.order[0],
                    npredictions=len(vpy_model_obj.pred_vdf),
                    metric="mse",
                )

                py_model_obj = get_py_model(model_class, npredictions=None)
                py_res = getattr(skl_metrics, "mean_squared_error")(
                    py_model_obj.y, py_model_obj.pred
                )
        else:
            if fit_attr == "score":
                vpy_res = getattr(vpy_model_obj.model, fit_attr)()
                py_res = getattr(py_model_obj.model, fit_attr)(
                    py_model_obj.X, py_model_obj.y
                )
            else:
                vpy_res = getattr(vpy_model_obj.model, fit_attr)
                py_res = getattr(py_model_obj.model, fit_attr)

        assert vpy_res == pytest.approx(py_res, rel=rel_tolerance_map[model_class])

    @pytest.mark.parametrize(*regression_metrics_args)
    @pytest.mark.parametrize("fun_name", ["regression", "report"])
    def test_regression_report_none(
        self,
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

        vpy_score, py_score = regression_report_none(
            get_vpy_model,
            get_py_model,
            model_class,
            regression_metrics,
            fun_name,
            vpy_metric_name,
            py_metric_name,
            _rel_tolerance,
            model_params,
        )
        assert vpy_score == pytest.approx(
            py_score,
            rel=_rel_tolerance[model_class]
            if isinstance(_rel_tolerance, dict)
            else _rel_tolerance,
        )

    @pytest.mark.parametrize(*details_report_args)
    @pytest.mark.parametrize("fun_name", ["regression", "report"])
    def test_regression_report_details(
        self,
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
        if model_class in [
            "AR",
            "MA",
            "ARMA",
            "ARIMA",
        ]:
            pytest.skip(
                f"report function with metrics details/anova is not available for {model_class} model"
            )
        else:
            vpy_reg_rep_details_map, py_res = regression_report_details(
                model_class,
                get_vpy_model,
                get_py_model,
                regression_metrics,
                fun_name,
                metric,
                expected,
                _rel_tolerance,
                _abs_tolerance,
            )

            if py_res == 0:
                assert vpy_reg_rep_details_map[metric] == pytest.approx(
                    py_res, abs=_abs_tolerance
                )
            else:
                assert vpy_reg_rep_details_map[metric] == pytest.approx(
                    py_res,
                    rel=_rel_tolerance[model_class]
                    if isinstance(_rel_tolerance, dict)
                    else _rel_tolerance,
                )

    @pytest.mark.parametrize(*anova_report_args)
    @pytest.mark.parametrize("fun_name", ["regression", "report"])
    def test_regression_report_anova(
        self,
        model_class,
        get_vpy_model,
        get_py_model,
        regression_metrics,
        fun_name,
        metric,
        metric_types,
        _rel_tolerance,
        _abs_tolerance,
    ):
        """
        test function - regression/report anova
        """
        if model_class in [
            "AR",
            "MA",
            "ARMA",
            "ARIMA",
        ]:
            pytest.skip(
                f"report function with metrics details/anova is not available for {model_class} model"
            )
        else:
            reg_rep_anova, regression_metrics_map = regression_report_anova(
                model_class,
                get_vpy_model,
                get_py_model,
                regression_metrics,
                fun_name,
                _rel_tolerance,
                _abs_tolerance,
            )

            for vpy_res, metric_type in zip(reg_rep_anova[metric], metric_types):
                py_res = regression_metrics_map[metric_type]

                if py_res == 0:
                    assert vpy_res == pytest.approx(py_res, abs=1e-9)
                else:
                    assert vpy_res == pytest.approx(
                        py_res,
                        rel=_rel_tolerance[model_class]
                        if isinstance(_rel_tolerance, dict)
                        else _rel_tolerance,
                    )

    @pytest.mark.parametrize(*regression_metrics_args)
    def test_score(
        self,
        model_class,
        get_vpy_model,
        get_py_model,
        regression_metrics,
        vpy_metric_name,
        py_metric_name,
        _rel_tolerance,
        model_params,
    ):
        """
        test function - test_score
        """
        vpy_score, py_score = model_score(
            model_class,
            get_vpy_model,
            get_py_model,
            regression_metrics,
            vpy_metric_name,
            py_metric_name,
            _rel_tolerance,
            model_params,
        )

        print(
            f"Metric Name: {py_metric_name}, vertica: {vpy_score}, sklearn: {py_score}"
        )

        assert vpy_score == pytest.approx(py_score, rel=_rel_tolerance[model_class])
