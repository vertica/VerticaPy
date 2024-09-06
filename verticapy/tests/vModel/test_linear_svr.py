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

# Other Modules
import matplotlib.pyplot as plt

# VerticaPy
from verticapy import drop, set_option
from verticapy.connection import current_cursor
from verticapy.datasets import load_winequality
from verticapy.learn.svm import LinearSVR

# Matplotlib skip
import matplotlib

matplotlib_version = matplotlib.__version__
skip_plt = pytest.mark.skipif(
    matplotlib_version > "3.5.2",
    reason="Test skipped on matplotlib version greater than 3.5.2",
)

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
    model_class = LinearSVR(
        "lsvr_model_test",
    )
    model_class.drop()
    model_class.fit(
        "public.winequality",
        ["citric_acid", "residual_sugar", "alcohol"],
        "quality",
    )
    yield model_class
    model_class.drop()


class TestLinearSVR:
    def test_repr(self, model):
        assert model.__repr__() == "<LinearSVR>"

    @skip_plt
    def test_contour(self, winequality_vd):
        model_test = LinearSVR(
            "model_contour",
        )
        model_test.drop()
        model_test.fit(
            winequality_vd,
            ["citric_acid", "residual_sugar"],
            "quality",
        )
        result = model_test.contour()
        assert len(result.get_default_bbox_extra_artists()) == 38
        model_test.drop()

    def test_deploySQL(self, model):
        expected_sql = 'PREDICT_SVM_REGRESSOR("citric_acid", "residual_sugar", "alcohol" USING PARAMETERS model_name = \'lsvr_model_test\', match_by_pos = \'true\')'
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_drop(self):
        current_cursor().execute("DROP MODEL IF EXISTS lsvr_model_test_drop")
        model_test = LinearSVR(
            "lsvr_model_test_drop",
        )
        model_test.fit("public.winequality", ["alcohol"], "quality")

        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'lsvr_model_test_drop'"
        )
        assert current_cursor().fetchone()[0] == "lsvr_model_test_drop"

        model_test.drop()
        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'lsvr_model_test_drop'"
        )
        assert current_cursor().fetchone() is None

    def test_features_importance(self, model):
        fim = model.features_importance(show=False)

        assert fim["index"] == ["alcohol", "residual_sugar", "citric_acid"]
        assert fim["importance"] == [52.68, 33.27, 14.05]
        assert fim["sign"] == [1, 1, 1]
        plt.close("all")

    def test_get_vertica_attributes(self, model):
        m_att = model.get_vertica_attributes()

        assert m_att["attr_name"] == [
            "details",
            "accepted_row_count",
            "rejected_row_count",
            "iteration_count",
            "call_string",
        ]
        assert m_att["attr_fields"] == [
            "predictor, coefficient",
            "accepted_row_count",
            "rejected_row_count",
            "iteration_count",
            "call_string",
        ]
        assert m_att["#_of_rows"] == [4, 1, 1, 1, 1]

        m_att_details = model.get_vertica_attributes(attr_name="details")

        assert m_att_details["predictor"] == [
            "Intercept",
            "citric_acid",
            "residual_sugar",
            "alcohol",
        ]
        assert m_att_details["coefficient"][0] == pytest.approx(
            1.67237120425236, abs=1e-6
        )
        assert m_att_details["coefficient"][1] == pytest.approx(
            0.410055106076537, abs=1e-6
        )
        assert m_att_details["coefficient"][2] == pytest.approx(
            0.0247255999942058, abs=1e-6
        )
        assert m_att_details["coefficient"][3] == pytest.approx(
            0.369955366024044, abs=1e-6
        )

        assert (
            model.get_vertica_attributes("iteration_count")["iteration_count"][0] == 5
        )
        assert (
            model.get_vertica_attributes("rejected_row_count")["rejected_row_count"][0]
            == 0
        )
        assert (
            model.get_vertica_attributes("accepted_row_count")["accepted_row_count"][0]
            == 6497
        )
        assert (
            model.get_vertica_attributes("call_string")["call_string"][0]
            == "SELECT svm_regressor('public.lsvr_model_test', 'public.winequality', '\"quality\"', '\"citric_acid\", \"residual_sugar\", \"alcohol\"'\nUSING PARAMETERS error_tolerance=0.1, C=1, max_iterations=100, intercept_mode='regularized', intercept_scaling=1, epsilon=0.0001);"
        )

    def test_get_params(self, model):
        assert model.get_params() == {
            "tol": 0.0001,
            "C": 1.0,
            "max_iter": 100,
            "intercept_scaling": 1.0,
            "intercept_mode": "regularized",
            "acceptable_error_margin": 0.1,
        }

    @skip_plt
    def test_get_plot(self, winequality_vd):
        current_cursor().execute("DROP MODEL IF EXISTS model_test_plot")
        model_test = LinearSVR(
            "model_test_plot",
        )
        model_test.fit("public.winequality", ["alcohol"], "quality")
        result = model_test.plot()
        assert len(result.get_default_bbox_extra_artists()) == 9
        plt.close("all")
        model_test.drop()

    def test_to_python(self, model):
        current_cursor().execute(
            "SELECT PREDICT_SVM_REGRESSOR(3.0, 11.0, 93. USING PARAMETERS model_name = '{}', match_by_pos=True)".format(
                model.model_name
            )
        )
        prediction = current_cursor().fetchone()[0]
        assert prediction == pytest.approx(model.to_python()([[3.0, 11.0, 93.0]])[0])

    def test_to_sql(self, model):
        current_cursor().execute(
            "SELECT PREDICT_SVM_REGRESSOR(3.0, 11.0, 93. USING PARAMETERS model_name = '{}', match_by_pos=True)::float, {}::float".format(
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
            5.8191136575806, abs=1e-6
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
        assert reg_rep["value"][0] == pytest.approx(0.219641599658795, abs=1e-6)
        assert reg_rep["value"][1] == pytest.approx(3.61156861855927, abs=1e-6)
        assert reg_rep["value"][2] == pytest.approx(0.49469564704003, abs=1e-3)
        assert reg_rep["value"][3] == pytest.approx(0.608521836351418, abs=1e-6)
        assert reg_rep["value"][4] == pytest.approx(0.594990575399229, abs=1e-6)
        assert reg_rep["value"][5] == pytest.approx(0.7713563219415707, abs=1e-6)
        assert reg_rep["value"][6] == pytest.approx(0.219640889304706, abs=1e-6)
        assert reg_rep["value"][7] == pytest.approx(0.21928033527235014, abs=1e-6)
        assert reg_rep["value"][8] == pytest.approx(-3365.29441626357, abs=1e-6)
        assert reg_rep["value"][9] == pytest.approx(-3338.189123593071, abs=1e-6)

        reg_rep_details = model.regression_report(metrics="details")
        assert reg_rep_details["value"][2:] == [
            6497.0,
            3,
            pytest.approx(0.219640889304706),
            pytest.approx(0.21928033527235014),
            pytest.approx(640.932567311251),
            pytest.approx(0.0),
            pytest.approx(0.232322269343305),
            pytest.approx(0.189622693372695),
            pytest.approx(53.1115447611131),
        ]

        reg_rep_anova = model.regression_report(metrics="anova")
        assert reg_rep_anova["SS"] == [
            pytest.approx(1144.75129867412),
            pytest.approx(3865.65376836879),
            pytest.approx(4953.68570109281),
        ]
        assert reg_rep_anova["MS"][:-1] == [
            pytest.approx(381.5837662247067),
            pytest.approx(0.595357118184012),
        ]

    def test_score(self, model):
        # method = "max"
        assert model.score(metric="max") == pytest.approx(3.61156861855927, abs=1e-6)
        # method = "mae"
        assert model.score(metric="mae") == pytest.approx(0.608521836351418, abs=1e-6)
        # method = "median"
        assert model.score(metric="median") == pytest.approx(0.49469564704003, abs=1e-3)
        # method = "mse"
        assert model.score(metric="mse") == pytest.approx(0.594990575399229, abs=1e-6)
        # method = "rmse"
        assert model.score(metric="rmse") == pytest.approx(0.7713563219415712, abs=1e-6)
        # method = "msl"
        assert model.score(metric="msle") == pytest.approx(
            0.00251024411036473, abs=1e-6
        )
        # method = "r2"
        assert model.score() == pytest.approx(0.219640889304706, abs=1e-6)
        # method = "r2a"
        assert model.score(metric="r2a") == pytest.approx(0.21928033527235014, abs=1e-6)
        # method = "var"
        assert model.score(metric="var") == pytest.approx(0.219641599658795, abs=1e-6)
        # method = "aic"
        assert model.score(metric="aic") == pytest.approx(-3365.29441626357, abs=1e-6)
        # method = "bic"
        assert model.score(metric="bic") == pytest.approx(-3338.189123593071, abs=1e-6)

    def test_set_params(self, model):
        model.set_params({"max_iter": 1000})

        assert model.get_params()["max_iter"] == 1000

    def test_model_from_vDF(self, winequality_vd):
        current_cursor().execute("DROP MODEL IF EXISTS lsvr_from_vDF")
        model_test = LinearSVR(
            "lsvr_from_vDF",
        )
        model_test.fit(winequality_vd, ["alcohol"], "quality")

        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'lsvr_from_vDF'"
        )
        assert current_cursor().fetchone()[0] == "lsvr_from_vDF"

        model_test.drop()

    def test_optional_name(self):
        model = LinearSVR()
        assert model.model_name is not None
