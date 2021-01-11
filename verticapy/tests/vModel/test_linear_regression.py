# (c) Copyright [2018-2020] Micro Focus or one of its affiliates.
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

import pytest, warnings, sys
from verticapy.learn.linear_model import LinearRegression
from verticapy import drop_table
import matplotlib.pyplot as plt

from verticapy import set_option

set_option("print_info", False)


@pytest.fixture(scope="module")
def winequality_vd(base):
    from verticapy.learn.datasets import load_winequality

    winequality = load_winequality(cursor=base.cursor)
    yield winequality
    with warnings.catch_warnings(record=True) as w:
        drop_table(name="public.winequality", cursor=base.cursor)


@pytest.fixture(scope="module")
def model(base, winequality_vd):
    base.cursor.execute("DROP MODEL IF EXISTS linreg_model_test")
    model_class = LinearRegression("linreg_model_test", cursor=base.cursor)
    model_class.fit(
        "public.winequality", ["citric_acid", "residual_sugar", "alcohol"], "quality"
    )
    yield model_class
    model_class.drop()


class TestLinearRegression:
    def test_deploySQL(self, model):
        expected_sql = 'PREDICT_LINEAR_REG("citric_acid", "residual_sugar", "alcohol" USING PARAMETERS model_name = \'linreg_model_test\', match_by_pos = \'true\')'
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_drop(self, base):
        base.cursor.execute("DROP MODEL IF EXISTS linreg_model_test_drop")
        model_test = LinearRegression("linreg_model_test_drop", cursor=base.cursor)
        model_test.fit("public.winequality", ["alcohol"], "quality")

        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'linreg_model_test_drop'"
        )
        assert base.cursor.fetchone()[0] == "linreg_model_test_drop"

        model_test.drop()
        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'linreg_model_test_drop'"
        )
        assert base.cursor.fetchone() is None

    def test_features_importance(self, model):
        fim = model.features_importance()

        assert fim["index"] == ["alcohol", "residual_sugar", "citric_acid"]
        assert fim["importance"] == [52.25, 32.58, 15.17]
        assert fim["sign"] == [1, 1, 1]
        plt.close()

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
        assert m_att_details["coefficient"][0] == pytest.approx(1.774512, abs=1e-6)
        assert m_att_details["coefficient"][1] == pytest.approx(0.434204, abs=1e-6)
        assert m_att_details["coefficient"][2] == pytest.approx(0.023752, abs=1e-6)
        assert m_att_details["coefficient"][3] == pytest.approx(0.359921, abs=1e-6)
        assert m_att_details["std_err"][3] == pytest.approx(0.008608, abs=1e-6)
        assert m_att_details["t_value"][3] == pytest.approx(41.8089205202906, abs=1e-6)
        assert m_att_details["p_value"][3] == pytest.approx(0)

        m_att_regularization = model.get_attr("regularization")

        assert m_att_regularization["type"][0] == "none"
        assert m_att_regularization["lambda"][0] == 1

        assert model.get_attr("iteration_count")["iteration_count"][0] == 1
        assert model.get_attr("rejected_row_count")["rejected_row_count"][0] == 0
        assert model.get_attr("accepted_row_count")["accepted_row_count"][0] == 6497
        assert (
            model.get_attr("call_string")["call_string"][0]
            == "linear_reg('public.linreg_model_test', 'public.winequality', '\"quality\"', '\"citric_acid\", \"residual_sugar\", \"alcohol\"'\nUSING PARAMETERS optimizer='newton', epsilon=1e-06, max_iterations=100, regularization='none', lambda=1, alpha=0.5)"
        )

    def test_get_params(self, model):
        assert model.get_params() == {
            "solver": "newton",
            "penalty": "none",
            "max_iter": 100,
            "tol": 1e-06,
        }

    @pytest.mark.skip(reason="test not implemented")
    def test_get_plot(self):
        pass

    def test_to_sklearn(self, model):
        md = model.to_sklearn()
        model.cursor.execute(
            "SELECT PREDICT_LINEAR_REG(3.0, 11.0, 93. USING PARAMETERS model_name = '{}', match_by_pos=True)".format(
                model.name
            )
        )
        prediction = model.cursor.fetchone()[0]
        assert prediction == pytest.approx(md.predict([[3.0, 11.0, 93.0]])[0][0])

    @pytest.mark.skip(reason="shap doesn't want to work on python3.6")
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
            5.818378, abs=1e-6
        )

    def test_regression_report(self, model):
        reg_rep = model.regression_report()

        assert reg_rep["index"] == [
            "explained_variance",
            "max_error",
            "median_absolute_error",
            "mean_absolute_error",
            "mean_squared_error",
            "r2",
        ]
        assert reg_rep["value"][0] == pytest.approx(0.219816, abs=1e-6)
        assert reg_rep["value"][1] == pytest.approx(3.592465, abs=1e-6)
        assert reg_rep["value"][2] == pytest.approx(0.496031, abs=1e-6)
        assert reg_rep["value"][3] == pytest.approx(0.609075, abs=1e-6)
        assert reg_rep["value"][4] == pytest.approx(0.594856, abs=1e-6)
        assert reg_rep["value"][5] == pytest.approx(0.219816, abs=1e-6)

    def test_score(self, model):
        # method = "max"
        assert model.score(method="max") == pytest.approx(3.592465, abs=1e-6)
        # method = "mae"
        assert model.score(method="mae") == pytest.approx(0.609075, abs=1e-6)
        # method = "median"
        assert model.score(method="median") == pytest.approx(0.496031, abs=1e-6)
        # method = "mse"
        assert model.score(method="mse") == pytest.approx(0.609075, abs=1e-6)
        # method = "msl"
        assert model.score(method="msle") == pytest.approx(0.002509, abs=1e-6)
        # method = "r2"
        assert model.score() == pytest.approx(0.219816, abs=1e-6)
        # method = "var"
        assert model.score(method="var") == pytest.approx(0.219816, abs=1e-6)

    def test_set_cursor(self, base):
        model_test = LinearRegression("linear_reg_cursor_test", cursor=base.cursor)
        # TODO: creat a new cursor
        model_test.set_cursor(base.cursor)
        model_test.drop()
        model_test.fit("public.winequality", ["alcohol"], "quality")

        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'linear_reg_cursor_test'"
        )
        assert base.cursor.fetchone()[0] == "linear_reg_cursor_test"

        model_test.drop()

    def test_set_params(self, model):
        model.set_params({"max_iter": 1000})

        assert model.get_params()["max_iter"] == 1000

    def test_model_from_vDF(self, base, winequality_vd):
        base.cursor.execute("DROP MODEL IF EXISTS linreg_from_vDF")
        model_test = LinearRegression("linreg_from_vDF", cursor=base.cursor)
        model_test.fit(winequality_vd, ["alcohol"], "quality")

        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'linreg_from_vDF'"
        )
        assert base.cursor.fetchone()[0] == "linreg_from_vDF"

        model_test.drop()
