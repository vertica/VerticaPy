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

import pytest, sys, os, verticapy
from verticapy.learn.linear_model import Ridge
from verticapy import drop, set_option, vertica_conn
from decimal import Decimal
import matplotlib.pyplot as plt

set_option("print_info", False)


@pytest.fixture(scope="module")
def winequality_vd(base):
    from verticapy.datasets import load_winequality

    winequality = load_winequality(cursor=base.cursor)
    yield winequality
    drop(name="public.winequality", cursor=base.cursor)


@pytest.fixture(scope="module")
def model(base, winequality_vd):
    base.cursor.execute("DROP MODEL IF EXISTS ridge_model_test")
    model_class = Ridge("ridge_model_test", cursor=base.cursor)
    model_class.fit(
        "public.winequality", ["citric_acid", "residual_sugar", "alcohol"], "quality"
    )
    yield model_class
    model_class.drop()


class TestRidge:
    def test_repr(self, model):
        assert "|coefficient|std_err |t_value |p_value" in model.__repr__()
        model_repr = Ridge("lin_repr")
        model_repr.drop()
        assert model_repr.__repr__() == "<LinearRegression>"

    def test_contour(self, base, winequality_vd):
        model_test = Ridge("model_contour", cursor=base.cursor)
        model_test.drop()
        model_test.fit(
            winequality_vd,
            ["citric_acid", "residual_sugar",],
            "quality",
        )
        result = model_test.contour()
        assert len(result.get_default_bbox_extra_artists()) == 30
        model_test.drop()

    def test_deploySQL(self, model):
        expected_sql = 'PREDICT_LINEAR_REG("citric_acid", "residual_sugar", "alcohol" USING PARAMETERS model_name = \'ridge_model_test\', match_by_pos = \'true\')'
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_drop(self, base):
        base.cursor.execute("DROP MODEL IF EXISTS ridge_model_test_drop")
        model_test = Ridge("ridge_model_test_drop", cursor=base.cursor)
        model_test.fit("public.winequality", ["alcohol"], "quality")

        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'ridge_model_test_drop'"
        )
        assert base.cursor.fetchone()[0] == "ridge_model_test_drop"

        model_test.drop()
        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'ridge_model_test_drop'"
        )
        assert base.cursor.fetchone() is None

    def test_features_importance(self, model):
        fim = model.features_importance()

        assert fim["index"] == ["alcohol", "residual_sugar", "citric_acid"]
        assert fim["importance"] == [52.3, 32.63, 15.07]
        assert fim["sign"] == [1, 1, 1]

    def test_get_attr(self, model):
        m_att = model.get_attr()

        assert m_att["attr_name"] == [
            "details",
            "regularization",
            "iteration_count",
            "rejected_row_count",
            "accepted_row_count",
            "call_string",
        ]
        assert m_att["attr_fields"] == [
            "predictor, coefficient, std_err, t_value, p_value",
            "type, lambda",
            "iteration_count",
            "rejected_row_count",
            "accepted_row_count",
            "call_string",
        ]
        assert m_att["#_of_rows"] == [4, 1, 1, 1, 1, 1]

        m_att_details = model.get_attr(attr_name="details")

        assert m_att_details["predictor"] == [
            "Intercept",
            "citric_acid",
            "residual_sugar",
            "alcohol",
        ]
        assert m_att_details["coefficient"][0] == pytest.approx(
            1.77574980319025, abs=1e-6
        )
        assert m_att_details["coefficient"][1] == pytest.approx(
            0.431005879933288, abs=1e-6
        )
        assert m_att_details["coefficient"][2] == pytest.approx(
            0.0237636413018576, abs=1e-6
        )
        assert m_att_details["coefficient"][3] == pytest.approx(
            0.359894749137091, abs=1e-6
        )
        assert m_att_details["std_err"][3] == pytest.approx(
            0.00860813464286587, abs=1e-6
        )
        assert m_att_details["t_value"][3] == pytest.approx(41.8086802853809, abs=1e-6)
        assert m_att_details["p_value"][1] == pytest.approx(8.96677134128099e-11)

        m_att_regularization = model.get_attr("regularization")

        assert m_att_regularization["type"][0] == "l2"
        assert m_att_regularization["lambda"][0] == 1

        assert model.get_attr("iteration_count")["iteration_count"][0] == 1
        assert model.get_attr("rejected_row_count")["rejected_row_count"][0] == 0
        assert model.get_attr("accepted_row_count")["accepted_row_count"][0] == 6497
        assert (
            model.get_attr("call_string")["call_string"][0]
            == "linear_reg('public.ridge_model_test', 'public.winequality', '\"quality\"', '\"citric_acid\", \"residual_sugar\", \"alcohol\"'\nUSING PARAMETERS optimizer='newton', epsilon=1e-06, max_iterations=100, regularization='l2', lambda=1, alpha=0.5)"
        )

    def test_get_params(self, model):
        assert model.get_params() == {
            "solver": "newton",
            "penalty": "l2",
            "max_iter": 100,
            "C": 1.0,
            "tol": 1e-06,
        }

    def test_get_plot(self, base, winequality_vd):
        base.cursor.execute("DROP MODEL IF EXISTS model_test_plot")
        model_test = Ridge("model_test_plot", cursor=base.cursor)
        model_test.fit(winequality_vd, ["alcohol"], "quality")
        result = model_test.plot()
        assert len(result.get_default_bbox_extra_artists()) == 9
        plt.close("all")
        model_test.drop()

    def test_to_sklearn(self, model):
        md = model.to_sklearn()
        model.cursor.execute(
            "SELECT PREDICT_LINEAR_REG(3.0, 11.0, 93. USING PARAMETERS model_name = '{}', match_by_pos=True)".format(
                model.name
            )
        )
        prediction = model.cursor.fetchone()[0]
        assert prediction == pytest.approx(md.predict([[3.0, 11.0, 93.0]])[0][0])

    def test_to_python(self, model):
        model.cursor.execute(
            "SELECT PREDICT_LINEAR_REG(3.0, 11.0, 93. USING PARAMETERS model_name = '{}', match_by_pos=True)".format(
                model.name
            )
        )
        prediction = model.cursor.fetchone()[0]
        assert prediction == pytest.approx(model.to_python(return_str=False)([[3.0, 11.0, 93.0]])[0])

    def test_to_sql(self, model):
        model.cursor.execute(
            "SELECT PREDICT_LINEAR_REG(3.0, 11.0, 93. USING PARAMETERS model_name = '{}', match_by_pos=True)::float, {}::float".format(
                model.name, model.to_sql([3.0, 11.0, 93.])
            )
        )
        prediction = model.cursor.fetchone()
        assert prediction[0] == pytest.approx(prediction[1])

    @pytest.mark.skip(reason="shap doesn't want to get installed.")
    def test_shapExplainer(self, model):
        explainer = model.shapExplainer()
        assert explainer.expected_value[0] == pytest.approx(5.81837771)

    def test_get_predicts(self, winequality_vd, model):
        winequality_copy = winequality_vd.copy()
        model.predict(
            winequality_copy,
            X=["citric_acid", "residual_sugar", "alcohol"],
            name="predicted_quality",
        )

        assert winequality_copy["predicted_quality"].mean() == pytest.approx(
            5.8183779, abs=1e-6
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
        assert reg_rep["value"][0] == pytest.approx(0.219816244842147, abs=1e-6)
        assert reg_rep["value"][1] == pytest.approx(3.59213874427945, abs=1e-6)
        assert reg_rep["value"][2] == pytest.approx(0.495516023908698, abs=1e-3)
        assert reg_rep["value"][3] == pytest.approx(0.60908330928705, abs=1e-6)
        assert reg_rep["value"][4] == pytest.approx(0.594856874272792, abs=1e-6)
        assert reg_rep["value"][5] == pytest.approx(0.7712696508179172, abs=1e-6)
        assert reg_rep["value"][6] == pytest.approx(0.219816244842152, abs=1e-6)
        assert reg_rep["value"][7] == pytest.approx(0.21945577183037412, abs=1e-6)
        assert reg_rep["value"][8] == pytest.approx(-3366.7594590080057, abs=1e-6)
        assert reg_rep["value"][9] == pytest.approx(-3339.6492371939366, abs=1e-6)

        reg_rep_details = model.regression_report("details")
        assert reg_rep_details["value"][2:] == [
            6497.0,
            3,
            pytest.approx(0.219816244842152),
            pytest.approx(0.21945577183037412),
            pytest.approx(609.4456850593101),
            pytest.approx(0.0),
            pytest.approx(0.232322269343305),
            pytest.approx(0.189622693372695),
            pytest.approx(53.1115447611131),
        ]

        reg_rep_anova = model.regression_report("anova")
        assert reg_rep_anova["SS"] == [
            pytest.approx(1088.2688789226),
            pytest.approx(3864.78511215033),
            pytest.approx(4953.68570109281),
        ]
        assert reg_rep_anova["MS"][:-1] == [
            pytest.approx(362.7562929742),
            pytest.approx(0.5952233346912568),
        ]

    def test_score(self, model):
        # method = "max"
        assert model.score(method="max") == pytest.approx(3.59213874427945, abs=1e-6)
        # method = "mae"
        assert model.score(method="mae") == pytest.approx(0.60908330928705, abs=1e-6)
        # method = "median"
        assert model.score(method="median") == pytest.approx(
            0.495516023908698, abs=1e-3
        )
        # method = "mse"
        assert model.score(method="mse") == pytest.approx(0.594856874272793, abs=1e-6)
        # method = "rmse"
        assert model.score(method="rmse") == pytest.approx(0.7712696508179179, abs=1e-6)
        # method = "msl"
        assert model.score(method="msle") == pytest.approx(
            0.00250970549028931, abs=1e-6
        )
        # method = "r2"
        assert model.score(method="r2") == pytest.approx(0.219816244842152, abs=1e-6)
        # method = "r2a"
        assert model.score(method="r2a") == pytest.approx(0.21945577183037412, abs=1e-6)
        # method = "var"
        assert model.score(method="var") == pytest.approx(0.219816244842147, abs=1e-6)
        # method = "aic"
        assert model.score(method="aic") == pytest.approx(-3366.7594590080057, abs=1e-6)
        # method = "bic"
        assert model.score(method="bic") == pytest.approx(-3339.6492371939366, abs=1e-6)

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
        model.set_params({"max_iter": 1000})

        assert model.get_params()["max_iter"] == 1000

    def test_model_from_vDF(self, base, winequality_vd):
        base.cursor.execute("DROP MODEL IF EXISTS ridge_from_vDF")
        model_test = Ridge("ridge_from_vDF", cursor=base.cursor)
        model_test.fit(winequality_vd, ["alcohol"], "quality")

        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'ridge_from_vDF'"
        )
        assert base.cursor.fetchone()[0] == "ridge_from_vDF"

        model_test.drop()
