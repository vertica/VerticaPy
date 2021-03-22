# (c) Copyright [2018-2021] Micro Focus or one of its affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest, warnings, sys, os, verticapy
from verticapy.learn.linear_model import LinearRegression, LogisticRegression
from verticapy.learn.preprocessing import StandardScaler, MinMaxScaler
from verticapy.learn.pipeline import Pipeline
from verticapy.utilities import tablesample
from verticapy import drop, set_option, vertica_conn
import matplotlib.pyplot as plt

set_option("print_info", False)


@pytest.fixture(scope="module")
def winequality_vd(base):
    from verticapy.datasets import load_winequality

    winequality = load_winequality(cursor=base.cursor)
    yield winequality
    with warnings.catch_warnings(record=True) as w:
        drop(name="public.winequality", cursor=base.cursor)


@pytest.fixture(scope="module")
def model(base, winequality_vd):
    model_class = Pipeline(
        [
            ("NormalizerWine", StandardScaler("std_model_test", cursor=base.cursor)),
            (
                "LinearRegressionWine",
                LinearRegression("linreg_model_test", cursor=base.cursor),
            ),
        ]
    )
    model_class.drop()
    model_class.fit(
        "public.winequality", ["citric_acid", "residual_sugar", "alcohol"], "quality"
    )
    yield model_class
    model_class.drop()


class TestPipeline:
    def test_index(self, model):
        assert model[0].type == "Normalizer"
        assert model[0:][0][0] == "NormalizerWine"

    def test_drop(self, base, winequality_vd):
        model_class = Pipeline(
            [
                (
                    "NormalizerWine",
                    StandardScaler("std_model_test_drop", cursor=base.cursor),
                ),
                (
                    "LinearRegressionWine",
                    LinearRegression("linreg_model_test_drop", cursor=base.cursor),
                ),
            ]
        )
        model_class.drop()
        model_class.fit(winequality_vd, ["alcohol"], "quality")
        model_class.cursor.execute(
            "SELECT model_name FROM models WHERE model_name IN ('linreg_model_test_drop', 'std_model_test_drop')"
        )
        assert len(model_class.cursor.fetchall()) == 2
        model_class.drop()
        model_class.cursor.execute(
            "SELECT model_name FROM models WHERE model_name IN ('linreg_model_test_drop', 'std_model_test_drop')"
        )
        assert model_class.cursor.fetchone() is None

    def test_get_params(self, model):
        assert model.get_params() == {
            "LinearRegressionWine": {
                "max_iter": 100,
                "penalty": "none",
                "solver": "newton",
                "tol": 1e-06,
            },
            "NormalizerWine": {"method": "zscore"},
        }

    def test_set_params(self, model):
        model.set_params({"NormalizerWine": {"method": "robust_zscore"}})
        assert model.get_params()["NormalizerWine"] == {"method": "robust_zscore"}
        model.set_params({"NormalizerWine": {"method": "zscore"}})

    def test_to_sklearn(self, model):
        md = model.to_sklearn()
        test_record = tablesample(
            {"citric_acid": [3.0], "residual_sugar": [11.0], "alcohol": [93.0]}
        ).to_vdf(cursor=model.cursor)
        prediction = model.predict(
            test_record, ["citric_acid", "residual_sugar", "alcohol"]
        )[0][-1]
        assert prediction == pytest.approx(md.predict([[3.0, 11.0, 93.0]])[0][0])

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
        assert reg_rep["value"][8] == pytest.approx(-3366.7617912479104, abs=1e-6)
        assert reg_rep["value"][9] == pytest.approx(-3339.65156943384, abs=1e-6)

        model_class = Pipeline(
            [
                (
                    "NormalizerWine",
                    StandardScaler("logstd_model_test", cursor=model.cursor),
                ),
                (
                    "LogisticRegressionWine",
                    LogisticRegression("logreg_model_test", cursor=model.cursor),
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
        assert cls_rep1["f1_score"][0] == pytest.approx(0.37307094353346476)
        assert cls_rep1["mcc"][0] == pytest.approx(0.2719537880298097)
        assert cls_rep1["informedness"][0] == pytest.approx(0.18715725014026519)
        assert cls_rep1["markedness"][0] == pytest.approx(0.3951696381964047)
        assert cls_rep1["csi"][0] == pytest.approx(0.19602649006622516)
        assert cls_rep1["cutoff"][0] == pytest.approx(0.5)

        model_class.drop()

    def test_score(self, model):
        # method = "max"
        assert model.score(method="max") == pytest.approx(3.592465, abs=1e-6)
        # method = "mae"
        assert model.score(method="mae") == pytest.approx(0.609075, abs=1e-6)
        # method = "median"
        assert model.score(method="median") == pytest.approx(0.496031, abs=1e-6)
        # method = "mse"
        assert model.score(method="mse") == pytest.approx(0.594856660735976, abs=1e-6)
        # method = "rmse"
        assert model.score(method="rmse") == pytest.approx(0.7712695123858948, abs=1e-6)
        # method = "msl"
        assert model.score(method="msle") == pytest.approx(0.002509, abs=1e-6)
        # method = "r2"
        assert model.score() == pytest.approx(0.219816, abs=1e-6)
        # method = "r2a"
        assert model.score(method="r2a") == pytest.approx(0.21945605202370688, abs=1e-6)
        # method = "var"
        assert model.score(method="var") == pytest.approx(0.219816, abs=1e-6)
        # method = "aic"
        assert model.score(method="aic") == pytest.approx(-3366.7617912479104, abs=1e-6)
        # method = "bic"
        assert model.score(method="bic") == pytest.approx(-3339.65156943384, abs=1e-6)

    def test_set_cursor(self, model):
        cur = vertica_conn(
            "vp_test_config",
            os.path.dirname(verticapy.__file__) + "/tests/verticaPy_test_tmp.conf",
        ).cursor()
        model.set_cursor(cur)
        model.cursor.execute("SELECT 1;")
        result = model.cursor.fetchone()
        assert result[0] == 1

    def test_transform(self, winequality_vd, model):
        model_class = Pipeline(
            [
                (
                    "NormalizerWine",
                    StandardScaler("logstd_model_test", cursor=model.cursor),
                ),
                (
                    "NormalizerWine",
                    MinMaxScaler("logmm_model_test", cursor=model.cursor),
                ),
            ]
        )
        model_class.drop()
        model_class.fit("public.winequality", ["alcohol"])
        winequality_copy = winequality_vd.copy()
        winequality_copy = model_class.transform(winequality_copy, X=["alcohol"],)
        assert winequality_copy["alcohol"].mean() == pytest.approx(
            0.361130555239542, abs=1e-6
        )

        model_class.drop()

    def test_inverse_transform(self, winequality_vd, model):
        model_class = Pipeline(
            [
                (
                    "NormalizerWine",
                    StandardScaler("logstd_model_test", cursor=model.cursor),
                ),
                (
                    "NormalizerWine",
                    MinMaxScaler("logmm_model_test", cursor=model.cursor),
                ),
            ]
        )
        model_class.drop()
        model_class.fit("public.winequality", ["alcohol"])
        winequality_copy = winequality_vd.copy()
        winequality_copy = model_class.inverse_transform(
            winequality_copy, X=["alcohol"],
        )
        assert winequality_copy["alcohol"].mean() == pytest.approx(
            80.3934257349546, abs=1e-6
        )

        model_class.drop()

    def test_model_from_vDF(self, base, winequality_vd):
        model_test = Pipeline(
            [
                (
                    "NormalizerWine",
                    StandardScaler("std_model_test_vdf", cursor=base.cursor),
                ),
                (
                    "LinearRegressionWine",
                    LinearRegression("linreg_model_test_vdf", cursor=base.cursor),
                ),
            ]
        )
        model_test.drop()
        model_test.fit(
            winequality_vd, ["citric_acid", "residual_sugar", "alcohol"], "quality"
        )
        model_test.cursor.execute(
            "SELECT model_name FROM models WHERE model_name IN ('std_model_test_vdf', 'linreg_model_test_vdf')"
        )
        assert len(base.cursor.fetchall()) == 2
        model_test.drop()
