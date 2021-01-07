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
from verticapy.learn.ensemble import RandomForestRegressor
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
    base.cursor.execute("DROP MODEL IF EXISTS rfr_model_test")
    base.cursor.execute("CREATE TABLE IF NOT EXISTS public.wine2 AS SELECT row_number() over() AS id, * from public.winequality order by fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol, quality, good, color")

    base.cursor.execute("SELECT rf_regressor('rfr_model_test', 'public.wine2', 'quality', 'citric_acid, residual_sugar, alcohol' USING PARAMETERS mtry=2, ntree=5, max_leaf_nodes=100, sampling_size=0.7, max_depth=4, min_leaf_size=2, min_info_gain=0.001, nbins=30, seed=1, id_column='id')")

    # I could use load_model but it is buggy
    model_class = RandomForestRegressor("rfr_model_test", cursor=base.cursor, n_estimators = 5,
                                         max_features = 2, max_leaf_nodes = 100, sample = 0.7,
                                         max_depth = 4, min_samples_leaf = 2, min_info_gain = 0.001, nbins = 30)
    model_class.input_relation = 'public.wine2'
    model_class.test_relation = model_class.input_relation
    model_class.X = ["citric_acid", "residual_sugar", "alcohol"]
    model_class.y = "quality"

    yield model_class
    model_class.drop()


class TestRFR:
    def test_deploySQL(self, model):
        expected_sql = "PREDICT_RF_REGRESSOR(citric_acid, residual_sugar, alcohol USING PARAMETERS model_name = 'rfr_model_test', match_by_pos = 'true')"
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_drop(self, base):
        base.cursor.execute("DROP MODEL IF EXISTS rfr_model_test_drop")
        model_test = RandomForestRegressor("rfr_model_test_drop", cursor=base.cursor)
        model_test.fit("public.winequality", ["alcohol"], "quality")

        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'rfr_model_test_drop'"
        )
        assert base.cursor.fetchone()[0] == "rfr_model_test_drop"

        model_test.drop()
        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'rfr_model_test_drop'"
        )
        assert base.cursor.fetchone() is None

    def test_features_importance(self, model):
        fim = model.features_importance()

        assert fim["index"] == ["alcohol", "citric_acid", "residual_sugar"]
        assert fim["importance"] == [85.89, 10.56, 3.56]
        assert fim["sign"] == [1, 1, 1]
        plt.close()

    def test_get_attr(self, model):
        m_att = model.get_attr()

        assert m_att["attr_name"] == [
            'tree_count',
            'rejected_row_count',
            'accepted_row_count',
            'call_string',
            'details'
        ]
        assert m_att["attr_fields"] == [
            'tree_count',
            'rejected_row_count',
            'accepted_row_count',
            'call_string', 'predictor, type'
        ]
        assert m_att["#_of_rows"] == [1, 1, 1, 1, 3]

        m_att_details = model.get_attr(attr_name="details")

        assert m_att_details["predictor"] == [
            "citric_acid",
            "residual_sugar",
            "alcohol",
        ]
        assert m_att_details["type"] == ['float or numeric', 'float or numeric', 'float or numeric']

        assert model.get_attr("tree_count")["tree_count"][0] == 5
        assert model.get_attr("rejected_row_count")["rejected_row_count"][0] == 0
        assert model.get_attr("accepted_row_count")["accepted_row_count"][0] == 6497
        assert (
            model.get_attr("call_string")["call_string"][0]
            == "SELECT rf_regressor('public.rfr_model_test', 'public.wine2', '\"quality\"', 'citric_acid, residual_sugar, alcohol' USING PARAMETERS exclude_columns='', ntree=5, mtry=2, sampling_size=0.7, max_depth=4, max_breadth=32, min_leaf_size=2, min_info_gain=0.001, nbins=30);"
        )

    def test_get_params(self, model):
        assert model.get_params() == {
            'n_estimators': 5,
            'max_features': 2,
            'max_leaf_nodes': 100,
            'sample': 0.7,
            'max_depth': 4,
            'min_samples_leaf': 2,
            'min_info_gain': 0.001,
            'nbins': 30
        }

    @pytest.mark.skip(reason="test not implemented")
    def test_get_plot(self):
        pass

    def test_to_sklearn(self, model):
        md = model.to_sklearn()
        model.cursor.execute(
            "SELECT PREDICT_RF_REGRESSOR(3.0, 11.0, 93.0 USING PARAMETERS model_name = '{}', match_by_pos=True)".format(
                model.name
            )
        )
        prediction = model.cursor.fetchone()[0]
        assert prediction == pytest.approx(md.predict([[3.0, 11.0, 93.0]])[0])

    @pytest.mark.skip(reason="The method to_shapExplainer is not available for model type RandomForestRegressor")
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
            5.8172285092582, abs=1e-6
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
        assert reg_rep["value"][0] == pytest.approx(0.248517919020818, abs=1e-6)
        assert reg_rep["value"][1] == pytest.approx(3.54752363446373, abs=1e-6)
        assert reg_rep["value"][2] == pytest.approx(0.494468592066868, abs=1e-6)
        assert reg_rep["value"][3] == pytest.approx(0.612363301928436, abs=1e-6)
        assert reg_rep["value"][4] == pytest.approx(0.572974391189618, abs=1e-6)
        assert reg_rep["value"][5] == pytest.approx(0.248516186899441, abs=1e-6)

    def test_score(self, model):
        # method = "max"
        assert model.score(method="max") == pytest.approx(3.54752363446373, abs=1e-6)
        # method = "mae"
        assert model.score(method="mae") == pytest.approx(0.612363301928436, abs=1e-6)
        # method = "median"
        assert model.score(method="median") == pytest.approx(0.494468592066868, abs=1e-6)
        # method = "mse"
        assert model.score(method="mse") == pytest.approx(0.612363301928436, abs=1e-6)
        # method = "msl"
        assert model.score(method="msle") == pytest.approx(0.00242033414556773, abs=1e-6)
        # method = "r2"
        assert model.score() == pytest.approx(0.248516186899441, abs=1e-6)
        # method = "var"
        assert model.score(method="var") == pytest.approx(0.248517919020818, abs=1e-6)

    def test_set_cursor(self, base):
        model_test = RandomForestRegressor("rfr_cursor_test", cursor=base.cursor)
        # TODO: creat a new cursor
        model_test.set_cursor(base.cursor)
        model_test.drop()
        model_test.fit("public.winequality", ["alcohol"], "quality")

        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'rfr_cursor_test'"
        )
        assert base.cursor.fetchone()[0] == "rfr_cursor_test"

        model_test.drop()

    def test_set_params(self, model):
        model.set_params({"nbins": 1000})

        assert model.get_params()["nbins"] == 1000

    def test_model_from_vDF(self, base, winequality_vd):
        base.cursor.execute("DROP MODEL IF EXISTS rfr_from_vDF")
        model_test = RandomForestRegressor("rfr_from_vDF", cursor=base.cursor)
        model_test.fit(winequality_vd, ["alcohol"], "quality")

        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'rfr_from_vDF'"
        )
        assert base.cursor.fetchone()[0] == "rfr_from_vDF"

        model_test.drop()
