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
from verticapy import (
    drop,
    set_option,
)
from verticapy.connection import current_cursor
from verticapy.datasets import load_titanic
from verticapy.learn.neighbors import KNeighborsRegressor

# Matplotlib skip
import matplotlib

matplotlib_version = matplotlib.__version__
skip_plt = pytest.mark.skipif(
    matplotlib_version > "3.5.2",
    reason="Test skipped on matplotlib version greater than 3.5.2",
)

set_option("print_info", False)


@pytest.fixture(scope="module")
def titanic_vd():
    titanic = load_titanic()
    yield titanic
    drop(
        name="public.titanic",
    )


@pytest.fixture(scope="module")
def model(titanic_vd):
    model_class = KNeighborsRegressor(
        "knn_model_test",
    )
    model_class.drop()
    model_class.fit("public.titanic", ["age", "fare"], "survived")
    yield model_class
    model_class.drop()


class TestKNeighborsRegressor:
    def test_repr(self, model):
        assert model.__repr__() == "<KNeighborsRegressor>"

    def test_get_attributes(self, model):
        m_att = model.get_attributes()
        assert m_att == ["n_neighbors_", "p_"]
        m_att = model.get_attributes("n_neighbors")
        assert m_att == model.parameters["n_neighbors"]
        m_att = model.get_attributes("p")
        assert m_att == model.parameters["p"]

    @skip_plt
    def test_contour(self, titanic_vd):
        model_test = KNeighborsRegressor(
            "model_contour",
        )
        model_test.drop()
        model_test.fit(
            titanic_vd,
            ["age", "fare"],
            "survived",
        )
        result = model_test.contour()
        assert len(result.get_default_bbox_extra_artists()) == 34
        model_test.drop()

    def test_deploySQL(self, model):
        expected_sql = '(SELECT "age", "fare", "survived", AVG(predict_neighbors) AS predict_neighbors FROM ( SELECT x."age", x."fare", x."survived", ROW_NUMBER() OVER(PARTITION BY x."age", x."fare", row_id ORDER BY POWER(POWER(ABS(x."age" - y."age"), 2) + POWER(ABS(x."fare" - y."fare"), 2), 1 / 2)) AS ordered_distance, y."survived" AS predict_neighbors, row_id FROM (SELECT *, ROW_NUMBER() OVER() AS row_id FROM public.titanic WHERE "age" IS NOT NULL AND "fare" IS NOT NULL) x CROSS JOIN (SELECT * FROM public.titanic WHERE "age" IS NOT NULL AND "fare" IS NOT NULL) y) z WHERE ordered_distance <= 5 GROUP BY "age", "fare", "survived", row_id) knr_table'
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_get_params(self, model):
        assert model.get_params() == {"n_neighbors": 5, "p": 2}

    def test_get_predicts(self, titanic_vd, model):
        titanic_copy = titanic_vd.copy()
        titanic_copy = model.predict(
            titanic_copy,
            X=["age", "fare"],
            name="predicted_quality",
            inplace=False,
        )

        assert titanic_copy["predicted_quality"].mean() == pytest.approx(
            0.378313253012048, abs=1e-6
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
        assert reg_rep["value"][0] == pytest.approx(0.32196148887151, abs=1e-6)
        assert reg_rep["value"][1] == pytest.approx(1.0, abs=1e-6)
        assert reg_rep["value"][2] == pytest.approx(0.2, abs=1e-6)
        assert reg_rep["value"][3] == pytest.approx(0.319076305220884, abs=1e-6)
        assert reg_rep["value"][4] == pytest.approx(0.161887550200803, abs=1e-6)
        assert reg_rep["value"][5] == pytest.approx(0.40235251981415876, abs=1e-6)
        assert reg_rep["value"][6] == pytest.approx(0.321109086681745, abs=1e-6)
        assert reg_rep["value"][7] == pytest.approx(0.3197417333820103, abs=1e-6)
        assert reg_rep["value"][8] == pytest.approx(-1807.52756734861, abs=1e-6)
        assert reg_rep["value"][9] == pytest.approx(-1792.8586642855382, abs=1e-6)

        reg_rep_details = model.regression_report(metrics="details")
        assert reg_rep_details["value"][2:] == [
            1234.0,
            2,
            pytest.approx(0.321109086681745),
            pytest.approx(0.3200060957584009),
            pytest.approx(239.85598471783427),
            pytest.approx(9.340149253171836e-86),
            pytest.approx(-1.68576262213743),
            pytest.approx(0.56300284427369),
            pytest.approx(212.604974886025),
        ]

        reg_rep_anova = model.regression_report(metrics="anova")
        assert reg_rep_anova["SS"] == [
            pytest.approx(77.894016064257),
            pytest.approx(161.24),
            pytest.approx(237.505020080321),
        ]
        assert reg_rep_anova["MS"][:-1] == [
            pytest.approx(38.9470080321285),
            pytest.approx(0.16237663645518632),
        ]

    def test_score(self, model):
        # method = "max"
        assert model.score(metric="max") == pytest.approx(1.0, abs=1e-6)
        # method = "mae"
        assert model.score(metric="mae") == pytest.approx(0.319076305220884, abs=1e-6)
        # method = "median"
        assert model.score(metric="median") == pytest.approx(0.2, abs=1e-6)
        # method = "mse"
        assert model.score(metric="mse") == pytest.approx(0.161887550200803, abs=1e-6)
        # method = "rmse"
        assert model.score(metric="rmse") == pytest.approx(
            0.40235251981415876, abs=1e-6
        )
        # method = "msl"
        assert model.score(metric="msle") == pytest.approx(0.0148862189812457, abs=1e-6)
        # method = "r2"
        assert model.score() == pytest.approx(0.321109086681745, abs=1e-6)
        # method = "r2a"
        assert model.score(metric="r2a") == pytest.approx(0.3197417333820103, abs=1e-6)
        # method = "var"
        assert model.score(metric="var") == pytest.approx(0.32196148887151, abs=1e-6)
        # method = "aic"
        assert model.score(metric="aic") == pytest.approx(-1807.52756734861, abs=1e-6)
        # method = "bic"
        assert model.score(metric="bic") == pytest.approx(-1792.8586642855369, abs=1e-6)

    def test_set_params(self, model):
        model.set_params({"p": 1})

        assert model.get_params()["p"] == 1

    def test_model_from_vDF(self, titanic_vd):
        model_test = KNeighborsRegressor(
            "knn_from_vDF",
        )
        model_test.drop()
        model_test.fit(titanic_vd, ["age"], "survived")
        assert model_test.score() == pytest.approx(-0.122616967579114)
        model_test.drop()

    def test_optional_name(self):
        model = KNeighborsRegressor()
        assert model.model_name is not None
