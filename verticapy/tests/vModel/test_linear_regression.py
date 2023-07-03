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

# Standard Python Modules
import pytest

# Other Modules
import matplotlib.pyplot as plt

# VerticaPy
from verticapy.tests.conftest import get_version
from verticapy import drop, set_option
from verticapy.connection import current_cursor
from verticapy.datasets import load_winequality
from verticapy.learn.linear_model import LinearRegression

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
    model_class = LinearRegression(
        "linreg_model_test",
    )
    model_class.drop()
    model_class.fit(
        "public.winequality",
        ["citric_acid", "residual_sugar", "alcohol"],
        "quality",
    )
    yield model_class
    model_class.drop()


class TestLinearRegression:
    def test_repr(self, model):
        assert model.__repr__() == "<LinearRegression>"

    def test_contour(self, winequality_vd):
        model_test = LinearRegression(
            "model_contour",
        )
        model_test.drop()
        model_test.fit(
            winequality_vd,
            ["citric_acid", "residual_sugar"],
            "quality",
        )
        result = model_test.contour()
        assert len(result.get_default_bbox_extra_artists()) == 32
        model_test.drop()

    def test_deploySQL(self, model):
        expected_sql = 'PREDICT_LINEAR_REG("citric_acid", "residual_sugar", "alcohol" USING PARAMETERS model_name = \'linreg_model_test\', match_by_pos = \'true\')'
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_drop(self):
        current_cursor().execute("DROP MODEL IF EXISTS linreg_model_test_drop")
        model_test = LinearRegression(
            "linreg_model_test_drop",
        )
        model_test.fit("public.winequality", ["alcohol"], "quality")

        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'linreg_model_test_drop'"
        )
        assert current_cursor().fetchone()[0] == "linreg_model_test_drop"

        model_test.drop()
        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'linreg_model_test_drop'"
        )
        assert current_cursor().fetchone() is None

    def test_features_importance(self, model):
        fim = model.features_importance(show=False)

        assert fim["index"] == ["alcohol", "residual_sugar", "citric_acid"]
        assert fim["importance"] == [52.25, 32.58, 15.17]
        assert fim["sign"] == [1, 1, 1]
        plt.close("all")

    def test_get_vertica_attributes(self, model):
        m_att = model.get_vertica_attributes()

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

        m_att_details = model.get_vertica_attributes(attr_name="details")

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

        m_att_regularization = model.get_vertica_attributes("regularization")

        assert m_att_regularization["type"][0] == "none"
        assert m_att_regularization["lambda"][0] == 1

        assert (
            model.get_vertica_attributes("iteration_count")["iteration_count"][0] == 1
        )
        assert (
            model.get_vertica_attributes("rejected_row_count")["rejected_row_count"][0]
            == 0
        )
        assert (
            model.get_vertica_attributes("accepted_row_count")["accepted_row_count"][0]
            == 6497
        )

        if get_version()[0] < 12:
            assert (
                model.get_vertica_attributes("call_string")["call_string"][0]
                == "linear_reg('public.linreg_model_test', 'public.winequality', '\"quality\"', '\"citric_acid\", \"residual_sugar\", \"alcohol\"'\nUSING PARAMETERS optimizer='newton', epsilon=1e-06, max_iterations=100, regularization='none', lambda=1, alpha=0.5)"
            )
        else:
            assert (
                model.get_vertica_attributes("call_string")["call_string"][0]
                == "linear_reg('public.linreg_model_test', 'public.winequality', '\"quality\"', '\"citric_acid\", \"residual_sugar\", \"alcohol\"'\nUSING PARAMETERS optimizer='newton', epsilon=1e-06, max_iterations=100, regularization='none', lambda=1, alpha=0.5, fit_intercept=true)"
            )

    def test_get_params(self, model):
        assert model.get_params() == {
            "solver": "newton",
            "max_iter": 100,
            "tol": 1e-06,
            "fit_intercept": True,
        }

    def test_get_plot(self, winequality_vd):
        current_cursor().execute("DROP MODEL IF EXISTS model_test_plot")
        model_test = LinearRegression(
            "model_test_plot",
        )
        model_test.fit(winequality_vd, ["alcohol"], "quality")
        result = model_test.plot(color="r")
        assert len(result.get_default_bbox_extra_artists()) == 9
        plt.close("all")
        model_test.drop()
        model_test.fit(winequality_vd, ["alcohol", "residual_sugar"], "quality")
        result = model_test.plot(color="r")
        assert len(result.get_default_bbox_extra_artists()) == 3
        plt.close("all")
        model_test.drop()

    def test_to_python(self, model):
        current_cursor().execute(
            "SELECT PREDICT_LINEAR_REG(3.0, 11.0, 93. USING PARAMETERS model_name = '{}', match_by_pos=True)".format(
                model.model_name
            )
        )
        prediction = current_cursor().fetchone()[0]
        assert prediction == pytest.approx(model.to_python()([[3.0, 11.0, 93.0]])[0])

    def test_to_sql(self, model):
        current_cursor().execute(
            "SELECT PREDICT_LINEAR_REG(3.0, 11.0, 93. USING PARAMETERS model_name = '{}', match_by_pos=True)::float, {}::float".format(
                model.model_name, model.to_sql([3.0, 11.0, 93.0])
            )
        )
        prediction = current_cursor().fetchone()
        assert prediction[0] == pytest.approx(prediction[1])

    def test_to_memmodel(self, model, winequality_vd):
        mmodel = model.to_memmodel()
        res = mmodel.predict([[3.0, 11.0, 93.0], [11.0, 1.0, 99.0]])
        res_py = model.to_python()([[3.0, 11.0, 93.0], [11.0, 1.0, 99.0]])
        assert res[0] == res_py[0]
        assert res[1] == res_py[1]
        vdf = winequality_vd.copy()
        vdf["prediction_sql"] = mmodel.predict_sql(
            ["citric_acid", "residual_sugar", "alcohol"]
        )
        model.predict(vdf, name="prediction_vertica_sql")
        score = vdf.score("prediction_sql", "prediction_vertica_sql", metric="r2")
        assert score == pytest.approx(1.0)

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
        assert reg_rep["value"][9] == pytest.approx(-3339.6515694338464, abs=1e-6)

        reg_rep_details = model.regression_report(metrics="details")
        assert reg_rep_details["value"][2:] == [
            6497.0,
            3,
            pytest.approx(0.219816524906085),
            pytest.approx(0.21945605202370688),
            pytest.approx(609.8004472783862),
            pytest.approx(0.0),
            pytest.approx(0.232322269343305),
            pytest.approx(0.189622693372695),
            pytest.approx(53.1115447611131),
        ]

        reg_rep_anova = model.regression_report(metrics="anova")
        assert reg_rep_anova["SS"] == [
            pytest.approx(1088.90197629059),
            pytest.approx(3864.78372480164),
            pytest.approx(4953.68570109281),
        ]
        assert reg_rep_anova["MS"][:-1] == [
            pytest.approx(362.9673254301967),
            pytest.approx(0.5952231210228923),
        ]

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
        assert model.score(metric="bic") == pytest.approx(-3339.6515694338464, abs=1e-6)

    def test_set_params(self, model):
        model.set_params({"max_iter": 1000})

        assert model.get_params()["max_iter"] == 1000

    def test_model_from_vDF(self, winequality_vd):
        current_cursor().execute("DROP MODEL IF EXISTS linreg_from_vDF")
        model_test = LinearRegression(
            "linreg_from_vDF",
        )
        model_test.fit(winequality_vd, ["alcohol"], "quality")

        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'linreg_from_vDF'"
        )
        assert current_cursor().fetchone()[0] == "linreg_from_vDF"

        model_test.drop()

    def test_overwrite_model(self, winequality_vd):
        model = LinearRegression("test_overwrite_model")
        model.drop()  # to isulate this test from any previous left over
        model.fit(winequality_vd, ["alcohol"], "quality")

        # overwrite_model is false by default
        with pytest.raises(NameError) as exception_info:
            model.fit(winequality_vd, ["alcohol"], "quality")
        assert exception_info.match("The model 'test_overwrite_model' already exists!")

        # overwriting the model when overwrite_model is specified true
        model = LinearRegression("test_overwrite_model", overwrite_model=True)
        model.fit(winequality_vd, ["alcohol"], "quality")

        # cleaning up
        model.drop()
