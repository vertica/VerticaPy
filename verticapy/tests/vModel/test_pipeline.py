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

# Pytest
import pytest

# VerticaPy
from verticapy import drop, set_option, TableSample
from verticapy.connection import current_cursor
from verticapy.datasets import load_winequality
from verticapy.learn.linear_model import LinearRegression, LogisticRegression
from verticapy.learn.preprocessing import Scaler, MinMaxScaler
from verticapy.learn.pipeline import Pipeline

set_option("print_info", False)


@pytest.fixture(scope="module")
def winequality_vd():
    winequality = load_winequality()
    yield winequality
    drop(
        name="public.winequality",
    )


@pytest.fixture(scope="module")
def model(winequality_vd):
    model_class = Pipeline(
        [
            (
                "ScalerWine",
                Scaler(
                    "std_model_test",
                ),
            ),
            (
                "LinearRegressionWine",
                LinearRegression(
                    "linreg_model_test",
                ),
            ),
        ]
    )
    model_class.drop()
    model_class.fit(
        "public.winequality",
        ["citric_acid", "residual_sugar", "alcohol"],
        "quality",
    )
    yield model_class
    model_class.drop()


class TestPipeline:
    def test_index(self, model):
        assert model[0]._model_type == "Scaler"
        assert model[0:][0][0] == "ScalerWine"

    def test_drop(self, winequality_vd):
        model_class = Pipeline(
            [
                (
                    "ScalerWine",
                    Scaler(
                        "std_model_test_drop",
                    ),
                ),
                (
                    "LinearRegressionWine",
                    LinearRegression(
                        "linreg_model_test_drop",
                    ),
                ),
            ]
        )
        model_class.drop()
        model_class.fit(winequality_vd, ["alcohol"], "quality")
        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name IN ('linreg_model_test_drop', 'std_model_test_drop')"
        )
        assert len(current_cursor().fetchall()) == 2
        model_class.drop()
        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name IN ('linreg_model_test_drop', 'std_model_test_drop')"
        )
        assert current_cursor().fetchone() is None

    def test_get_params(self, model):
        assert model.get_params() == {
            "LinearRegressionWine": {
                "fit_intercept": True,
                "max_iter": 100,
                "solver": "newton",
                "tol": 1e-06,
            },
            "ScalerWine": {"method": "zscore"},
        }

    def test_set_params(self, model):
        model.set_params({"ScalerWine": {"method": "robust_zscore"}})
        assert model.get_params()["ScalerWine"] == {"method": "robust_zscore"}
        model.set_params({"ScalerWine": {"method": "zscore"}})

    def test_get_predicts(self, winequality_vd, model):
        winequality_copy = winequality_vd.copy()
        winequality_copy = model.predict(
            winequality_copy,
            X=["citric_acid", "residual_sugar", "alcohol"],
            name="predicted_quality",
        )

        assert winequality_copy["predicted_quality"].mean() == pytest.approx(
            5.818378, abs=1e-6
        )

    def test_report(self, model):
        reg_rep = model.report()

        assert reg_rep["index"] == [
            "explained_variance",
            "max_error",
            "median_absolute_error",
            "mean_absolute_error",
            "mean_squared_error",
            "root_mean_squared_error",
            "r2",
            "r2_adj",
            "aic",
            "bic",
        ]
        assert reg_rep["value"][0] == pytest.approx(0.219816, abs=1e-6)
        assert reg_rep["value"][1] == pytest.approx(3.592465, abs=1e-6)
        assert reg_rep["value"][2] == pytest.approx(0.496031, abs=1e-6)
        assert reg_rep["value"][3] == pytest.approx(0.609075, abs=1e-6)
        assert reg_rep["value"][4] == pytest.approx(0.594856, abs=1e-6)
        assert reg_rep["value"][5] == pytest.approx(0.7712695123858948, abs=1e-6)
        assert reg_rep["value"][6] == pytest.approx(0.219816, abs=1e-6)
        assert reg_rep["value"][7] == pytest.approx(0.21945605202370688, abs=1e-6)
        assert reg_rep["value"][8] == pytest.approx(-3366.75686210436, abs=1e-6)
        assert reg_rep["value"][9] == pytest.approx(-3339.65156943384, abs=1e-6)

        model_class = Pipeline(
            [
                (
                    "ScalerWine",
                    Scaler("logstd_model_test"),
                ),
                (
                    "LogisticRegressionWine",
                    LogisticRegression("logreg_model_test"),
                ),
            ]
        )
        model_class.drop()
        model_class.fit("public.winequality", ["alcohol"], "good")
        cls_rep1 = model_class.report().transpose()
        assert cls_rep1["auc"][0] == pytest.approx(0.7642901826299067)
        assert cls_rep1["prc_auc"][0] == pytest.approx(0.45326090911518313)
        assert cls_rep1["accuracy"][0] == pytest.approx(0.8131445282438048)
        assert cls_rep1["log_loss"][0] == pytest.approx(0.182720882885624)
        assert cls_rep1["precision"][0] == pytest.approx(0.5595463137996219)
        assert cls_rep1["recall"][0] == pytest.approx(0.2317932654659358)
        assert cls_rep1["f1_score"][0] == pytest.approx(0.3277962347729789)
        assert cls_rep1["mcc"][0] == pytest.approx(0.2719537880298097)
        assert cls_rep1["informedness"][0] == pytest.approx(0.18715725014026519)
        assert cls_rep1["markedness"][0] == pytest.approx(0.3951696381964047)
        assert cls_rep1["csi"][0] == pytest.approx(0.19602649006622516)
        model_class.drop()

    def test_score(self, model):
        # method = "max"
        assert model.score(metric="max") == pytest.approx(3.592465, abs=1e-6)
        # method = "mae"
        assert model.score(metric="mae") == pytest.approx(0.609075, abs=1e-6)
        # method = "median"
        assert model.score(metric="median") == pytest.approx(0.496031, abs=1e-6)
        # method = "mse"
        assert model.score(metric="mse") == pytest.approx(0.594856660735976, abs=1e-6)
        # method = "rmse"
        assert model.score(metric="rmse") == pytest.approx(0.7712695123858948, abs=1e-6)
        # method = "msl"
        assert model.score(metric="msle") == pytest.approx(0.002509, abs=1e-6)
        # method = "r2"
        assert model.score() == pytest.approx(0.219816, abs=1e-6)
        # method = "r2a"
        assert model.score(metric="r2a") == pytest.approx(0.21945605202370688, abs=1e-6)
        # method = "var"
        assert model.score(metric="var") == pytest.approx(0.219816, abs=1e-6)
        # method = "aic"
        assert model.score(metric="aic") == pytest.approx(-3366.75686210436, abs=1e-6)
        # method = "bic"
        assert model.score(metric="bic") == pytest.approx(-3339.65156943384, abs=1e-6)

    def test_transform(self, winequality_vd, model):
        model_class = Pipeline(
            [
                (
                    "ScalerWine",
                    Scaler("logstd_model_test"),
                ),
                (
                    "ScalerWine",
                    MinMaxScaler("logmm_model_test"),
                ),
            ]
        )
        model_class.drop()
        model_class.fit("public.winequality", ["alcohol"])
        winequality_copy = winequality_vd.copy()
        winequality_copy = model_class.transform(winequality_copy, X=["alcohol"])
        assert winequality_copy["alcohol"].mean() == pytest.approx(
            0.361130555239542, abs=1e-6
        )

        model_class.drop()

    def test_inverse_transform(self, winequality_vd, model):
        model_class = Pipeline(
            [
                (
                    "ScalerWine",
                    Scaler("logstd_model_test"),
                ),
                (
                    "ScalerWine",
                    MinMaxScaler("logmm_model_test"),
                ),
            ]
        )
        model_class.drop()
        model_class.fit("public.winequality", ["alcohol"])
        winequality_copy = winequality_vd.copy()
        winequality_copy = model_class.inverse_transform(
            winequality_copy,
            X=["alcohol"],
        )
        assert winequality_copy["alcohol"].mean() == pytest.approx(
            80.3934257349546, abs=1e-6
        )

        model_class.drop()

    def test_model_from_vDF(self, winequality_vd):
        model_test = Pipeline(
            [
                (
                    "ScalerWine",
                    Scaler(
                        "std_model_test_vdf",
                    ),
                ),
                (
                    "LinearRegressionWine",
                    LinearRegression(
                        "linreg_model_test_vdf",
                    ),
                ),
            ]
        )
        model_test.drop()
        model_test.fit(
            winequality_vd, ["citric_acid", "residual_sugar", "alcohol"], "quality"
        )
        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name IN ('std_model_test_vdf', 'linreg_model_test_vdf')"
        )
        assert len(current_cursor().fetchall()) == 2
        model_test.drop()

    def test_overwrite_model(self, winequality_vd):
        model = Pipeline(
            [
                (
                    "ScalerWine",
                    Scaler(
                        "std_test_overwrite_model",
                    ),
                ),
                (
                    "LinearRegressionWine",
                    LinearRegression(
                        "linreg_test_overwrite_model",
                    ),
                ),
            ]
        )
        model.drop()
        model.fit(winequality_vd, ["alcohol"], "quality")

        # overwrite_model is false by default
        with pytest.raises(NameError) as exception_info:
            model.fit(winequality_vd, ["alcohol"], "quality")
        assert exception_info.match(
            "The model 'std_test_overwrite_model' already exists!"
        )

        # overwriting the model when overwrite_model is specified true
        model = Pipeline(
            [
                (
                    "ScalerWine",
                    Scaler(
                        "std_test_overwrite_model",
                    ),
                ),
                (
                    "LinearRegressionWine",
                    LinearRegression(
                        "linreg_test_overwrite_model",
                    ),
                ),
            ],
            overwrite_model=True,
        )
        model.fit(winequality_vd, ["alcohol"], "quality")

        # cleaning up
        model.drop()
