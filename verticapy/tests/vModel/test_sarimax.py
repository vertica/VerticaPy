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
from verticapy.learn.tsa import SARIMAX
from verticapy import drop, set_option, vertica_conn, create_verticapy_schema
import matplotlib.pyplot as plt

set_option("print_info", False)


@pytest.fixture(scope="module")
def amazon_vd(base):
    from verticapy.datasets import load_amazon

    amazon = load_amazon(cursor=base.cursor)
    yield amazon
    with warnings.catch_warnings(record=True) as w:
        drop(name="public.amazon", cursor=base.cursor)


@pytest.fixture(scope="module")
def model(base, amazon_vd):
    try:
        create_verticapy_schema(base.cursor)
    except:
        pass
    model_class = SARIMAX("sarimax_model_test", cursor=base.cursor, p=1, d=1, q=1, s=12, P=1, D=1, Q=1, max_pik=20)
    model_class.drop()
    model_class.fit("public.amazon", "number", "date",)
    yield model_class
    model_class.drop()


class TestSARIMAX:
    def test_repr(self, model):
        assert "Additional Info" in model.__repr__()
        model_repr = SARIMAX("sarimax_repr", cursor=model.cursor)
        model_repr.drop()
        assert model_repr.__repr__() == "<SARIMAX>"

    def test_deploySQL(self, model):
        assert 'VerticaPy_y_copy' in model.deploySQL()

    def test_drop(self, base):
        model_test = SARIMAX("sarimax_model_test_drop", cursor=base.cursor)
        model_test.drop()
        model_test.fit("public.amazon", "number", "date",)

        base.cursor.execute(
            "SELECT model_name FROM verticapy.models WHERE model_name IN ('sarimax_model_test_drop', '\"sarimax_model_test_drop\"')"
        )
        assert base.cursor.fetchone()[0] in ("sarimax_model_test_drop", '"sarimax_model_test_drop"')

        model_test.drop()
        base.cursor.execute(
            "SELECT model_name FROM verticapy.models WHERE model_name IN ('sarimax_model_test_drop', '\"sarimax_model_test_drop\"')"
        )
        assert base.cursor.fetchone() is None

    def test_get_attr(self, model):
        m_att = model.get_attr()

        assert m_att["attr_name"] == [
            "coef",
            "ma_avg",
            "ma_piq",
        ]

        m_att_details = model.get_attr(attr_name="coef")

        assert m_att_details["predictor"] == [
            "Intercept",
            "ar1",
            "ar12",
            "ma1",
            "ma12",
        ]
        assert m_att_details["coefficient"][0] == pytest.approx(-0.0206811318986692, abs=1e-6)
        assert m_att_details["coefficient"][1] == pytest.approx(-0.472445862105583, abs=1e-6)
        assert m_att_details["coefficient"][2] == pytest.approx(-0.283486934349855, abs=1e-6)
        assert m_att_details["coefficient"][3] == pytest.approx(-0.289912044494682, abs=1e-6)
        assert m_att_details["coefficient"][4] == pytest.approx(-0.5845016482145707, abs=1e-6)

        assert model.get_attr(attr_name="ma_avg") == pytest.approx(-0.271509332267827, abs=1e-6)
        assert model.get_attr(attr_name="ma_piq")["coefficient"][0:2] == [pytest.approx(-1, abs=1e-6), pytest.approx(0.289912044494682, abs=1e-6)]

    def test_get_params(self, model):
        assert model.get_params() == {'D': 1,
                                      'P': 1,
                                      'Q': 1,
                                      'd': 1,
                                      'max_iter': 1000,
                                      'max_pik': 20,
                                      'p': 1,
                                      'papprox_ma': 200,
                                      'q': 1,
                                      's': 12,
                                      'solver': 'Newton',
                                      'tol': 0.0001}

    def test_get_plot(self, model,):
        result = model.plot(color="r", nlead=10, nlast=10, dynamic=True,)
        assert len(result.get_default_bbox_extra_artists()) == 18
        plt.close("all")

    def test_get_predicts(self, amazon_vd, model):
        result = model.predict(
            amazon_vd,
            name="predict",
            nlead=10,
        )

        assert result["predict"].avg() == pytest.approx(
            140.036629403195, abs=1e-6
        )

    def test_regression_report(self, model):
        reg_rep = model.regression_report()

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
        assert reg_rep["value"][0] == pytest.approx(-0.772114907643978, abs=1e-6)
        assert float(reg_rep["value"][1]) == pytest.approx(28044.894016038184, abs=1e-6)
        assert reg_rep["value"][2] == pytest.approx(337.270572384534, abs=1e-6)
        assert reg_rep["value"][3] == pytest.approx(1089.45677026604, abs=1e-6)
        assert reg_rep["value"][4] == pytest.approx(4992560.88725707, abs=1e-6)
        assert reg_rep["value"][5] == pytest.approx(2234.403922136074, abs=1e-6)
        assert reg_rep["value"][6] == pytest.approx(-0.823855565723365, abs=1e-6)
        assert reg_rep["value"][7] == pytest.approx(-0.8249868143297991, abs=1e-6)
        assert reg_rep["value"][8] == pytest.approx(99553.01717600155, abs=1e-6)
        assert reg_rep["value"][9] == pytest.approx(99586.8701476537, abs=1e-6)

    def test_score(self, model):
        # method = "max"
        assert model.score(method="max") == pytest.approx(28044.894016038183, abs=1e-6)
        # method = "mae"
        assert model.score(method="mae") == pytest.approx(1089.45677026604, abs=1e-6)
        # method = "median"
        assert model.score(method="median") == pytest.approx(337.270572384534, abs=1e-6)
        # method = "mse"
        assert model.score(method="mse") == pytest.approx(4992560.88725705, abs=1e-6)
        # method = "rmse"
        assert model.score(method="rmse") == pytest.approx(2234.4039221360695, abs=1e-6)
        # method = "r2"
        assert model.score() == pytest.approx(-0.823855565723365, abs=1e-6)
        # method = "r2a"
        assert model.score(method="r2a") == pytest.approx(-0.8249868143297991, abs=1e-6)
        # method = "var"
        assert model.score(method="var") == pytest.approx(-0.772114907643978, abs=1e-6)
        # method = "aic"
        assert model.score(method="aic") == pytest.approx(99484.6564122794, abs=1e-6)
        # method = "bic"
        assert model.score(method="bic") == pytest.approx(99518.50938393154, abs=1e-6)

    def test_set_cursor(self, model):
        cur = vertica_conn(
            "vp_test_config",
            os.path.dirname(verticapy.__file__) + "/tests/verticaPy_test_tmp.conf",
        ).cursor()
        model.set_cursor(cur)
        model.cursor.execute("SELECT 1;")
        result = model.cursor.fetchone()
        assert result[0] == 1

    def test_set_params(self, model):
        model.set_params({"p": 2})

        assert model.get_params()["p"] == 2

    def test_model_from_vDF(self, base, amazon_vd):
        model_class = SARIMAX("sarimax_model_test_vdf", cursor=base.cursor, p=1, d=1, q=1, s=12, P=1, D=1, Q=1, max_pik=20)
        model_class.drop()
        model_class.fit(amazon_vd, "number", "date",)

        base.cursor.execute(
            "SELECT model_name FROM verticapy.models WHERE model_name IN ('sarimax_model_test_vdf', '\"sarimax_model_test_vdf\"')"
        )
        assert base.cursor.fetchone()[0] in ("sarimax_model_test_vdf", '"sarimax_model_test_vdf"')

        model_class.drop()
