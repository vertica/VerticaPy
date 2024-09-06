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
import pytest
import sklearn.metrics as skl_metrics
import verticapy.machine_learning.metrics.regression as vpy_metrics
import numpy as np
import verticapy.learn.linear_model as vpy_linear_model
import statsmodels.api as sm
from statsmodels.formula.api import ols
from verticapy import drop
import sklearn.linear_model as skl_linear_model
from math import log


@pytest.fixture(scope="module")
def vpy_lr_model_pred(winequality_vpy, schema_loader):
    vpy_lr_model = vpy_linear_model.LinearRegression(f"{schema_loader}.vpy_lr_model")
    vpy_lr_model.drop()
    vpy_lr_model.fit(
        f"{schema_loader}.winequality",
        ["citric_acid", "residual_sugar", "alcohol"],
        "quality",
    )
    num_pred = len(vpy_lr_model.X)
    vpy_lr_pred_vdf = vpy_lr_model.predict(winequality_vpy, name="quality_pred")

    yield vpy_lr_model, vpy_lr_pred_vdf, num_pred

    vpy_lr_model.drop()
    drop(name=f"{schema_loader}.vpy_lr_pred")


@pytest.mark.parametrize(
    "sql_relation_type, expected",
    [
        ("table", ""),
        ("view", ""),
        ("temporary", ""),
        # pytest.param('invalid', '', marks=pytest.mark.xfail)
    ],
)
@pytest.mark.parametrize(
    "vpy_regression_metrics, py_regression_metrics, func_params, is_skl_metrics",
    [
        (
            "aic_score",
            "aic",
            {},
            "n",
        ),  # need to check with Badr on displaying message/in doc on using mse for aic. statsmodel uses OLS(ordnary least sqauare) as default and gives different result
        (
            "bic_score",
            "bic",
            {},
            "n",
        ),  # need to check with Badr on displaying message/in doc on using mse for bic. statsmodel uses OLS(ordnary least sqauare) as default and gives different result
        (
            "explained_variance",
            "explained_variance_score",
            {},
            "y",
        ),  # function name change in skl
        ("max_error", "max_error", {}, "y"),
        ("mean_absolute_error", "mean_absolute_error", {}, "y"),
        (
            "mean_squared_error",
            "mean_squared_error",
            {},
            "y",
        ),  # need info on root parameter
        (
            "mean_squared_log_error",
            "mean_squared_log_error",
            {},
            "n",
        ),  # need to check with Badr on displaying message on log base, vertica has default base 10, sklean uses natural log (e)
        (
            "median_absolute_error",
            "median_absolute_error",
            {},
            "y",
        ),  # fail -- decimal precision issue
        (
            "quantile_error",
            "quantile_error",
            {},
            "n",
        ),  # fail -- decimal precision issue
        ("r2_score", "r2_score", {"k": 3, "adj": True}, "y"),
        (
            "r2_score",
            "r2_score",
            {"k": 3, "adj": False},
            "y",
        ),  # fail -- decimal precision issue
        ("anova_table", "anova_table", {}, "n")
        # Need to check with Badr on adding these metrics in vpy as these are present in skl
        # Mean Poisson, Gamma, and Tweedie deviances
        # Pinball loss
        # D² score
    ],
)
class TestRegressionMetrics:
    def test_master_regression_metrics(
        self,
        sql_relation_type,
        expected,
        vpy_lr_model_pred,
        vpy_regression_metrics,
        py_regression_metrics,
        func_params,
        is_skl_metrics,
        schema_loader,
    ):
        rel_tolerance = 1e-2
        vpy_lr_model, vpy_lr_pred_vdf, num_pred = vpy_lr_model_pred

        # converts to pandas dataframe for non vertica framework
        vpy_lr_pred_pdf = vpy_lr_pred_vdf.to_pandas()
        vpy_lr_pred_pdf["citric_acid"] = vpy_lr_pred_pdf["citric_acid"].astype(float)
        vpy_lr_pred_pdf["residual_sugar"] = vpy_lr_pred_pdf["residual_sugar"].astype(
            float
        )

        X = vpy_lr_pred_pdf[["citric_acid", "residual_sugar", "alcohol"]]
        y = vpy_lr_pred_pdf["quality"]

        # ******************************** verticapy logic ***************************************
        def get_vertica_metrics():
            # verticapy dataframe to vertica db
            drop(
                name=f"{schema_loader}.vpy_lr_pred",
            )
            vpy_lr_pred_vdf.to_db(
                name=f"{schema_loader}.vpy_lr_pred",
                relation_type=f"{sql_relation_type}",
            )

            if vpy_regression_metrics in ["quantile_error"]:
                _vpy_res = getattr(vpy_metrics, vpy_regression_metrics)(
                    y_true="quality",
                    y_score="quality_pred",
                    input_relation=f"{schema_loader}.vpy_lr_pred",
                    q=0.72,
                )
            elif vpy_regression_metrics in ["r2_score"]:
                if func_params["adj"]:
                    _vpy_res = getattr(vpy_metrics, vpy_regression_metrics)(
                        y_true="quality",
                        y_score="quality_pred",
                        input_relation=f"{schema_loader}.vpy_lr_pred",
                        adj=func_params["adj"],
                    )
                else:
                    _vpy_res = getattr(vpy_metrics, vpy_regression_metrics)(
                        y_true="quality",
                        y_score="quality_pred",
                        input_relation=f"{schema_loader}.vpy_lr_pred",
                        adj=func_params["adj"],
                    )
            elif vpy_regression_metrics in ["anova_table"]:
                _vpy_res = (
                    getattr(vpy_metrics, vpy_regression_metrics)(
                        y_true="quality",
                        y_score="quality_pred",
                        input_relation=f"{schema_loader}.vpy_lr_pred",
                        k=1,
                    )
                    .to_pandas()
                    .replace("", 0)
                    .values.tolist()[:2]
                )
                _vpy_res = np.array(_vpy_res)
            elif vpy_regression_metrics in ["aic_score", "bic_score"]:
                _vpy_res = getattr(vpy_metrics, vpy_regression_metrics)(
                    y_true="quality",
                    y_score="quality_pred",
                    input_relation=f"{schema_loader}.vpy_lr_pred",
                    k=num_pred,
                )
            else:
                _vpy_res = getattr(vpy_metrics, vpy_regression_metrics)(
                    y_true="quality",
                    y_score="quality_pred",
                    input_relation=f"{schema_loader}.vpy_lr_pred",
                )

            return _vpy_res

        # *************************** python based metrics logic *********************************
        def get_python_metrics():
            if is_skl_metrics == "y":
                if py_regression_metrics in ["r2_score"]:
                    r2_score = getattr(skl_metrics, py_regression_metrics)(
                        vpy_lr_pred_pdf["quality"], vpy_lr_pred_pdf["quality_pred"]
                    )
                    if func_params["adj"]:
                        _skl_res = 1 - (1 - r2_score) * (len(vpy_lr_pred_pdf) - 1) / (
                            len(vpy_lr_pred_pdf) - num_pred - 1
                        )
                    else:
                        _skl_res = r2_score
                else:
                    _skl_res = getattr(skl_metrics, py_regression_metrics)(
                        vpy_lr_pred_pdf["quality"], vpy_lr_pred_pdf["quality_pred"]
                    )
            else:
                if py_regression_metrics in ["quantile_error"]:
                    _skl_res = abs(
                        vpy_lr_pred_pdf["quality"] - vpy_lr_pred_pdf["quality_pred"]
                    ).quantile(0.72)
                elif py_regression_metrics in ["anova_table"]:
                    cw_lm = ols("quality ~ quality_pred", data=vpy_lr_pred_pdf).fit()
                    _skl_res = sm.stats.anova_lm(cw_lm, typ=1).fillna(0).values.tolist()
                    _skl_res = np.array(_skl_res)
                    # print(skl_res)
                elif py_regression_metrics in ["aic", "bic"]:
                    skl_model = skl_linear_model.LinearRegression()
                    skl_model.fit(X, y)
                    num_params = len(skl_model.coef_) + 1
                    pred = skl_model.predict(X)
                    mse = skl_metrics.mean_squared_error(y, pred)
                    n = len(y)

                    if py_regression_metrics in ["aic"]:
                        aic = n * log(mse) + 2 * num_params
                        _skl_res = aic
                    else:
                        bic = n * log(mse) + num_params * log(n)
                        _skl_res = bic
                elif py_regression_metrics in ["mean_squared_log_error"]:
                    _skl_res = sum(
                        pow(
                            (
                                np.log10(vpy_lr_pred_pdf["quality_pred"] + 1)
                                - np.log10(vpy_lr_pred_pdf["quality"] + 1)
                            ),
                            2,
                        )
                    ) / len(vpy_lr_pred_pdf)
                else:
                    assert False, "Invalid metric ....................."
            return _skl_res

        vpy_res = get_vertica_metrics()
        skl_res = get_python_metrics()

        print(f"vertica: {vpy_res}, sklearn: {skl_res}")
        assert vpy_res == pytest.approx(skl_res, rel=rel_tolerance)
