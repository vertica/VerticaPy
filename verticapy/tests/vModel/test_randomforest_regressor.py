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
from verticapy.learn.ensemble import RandomForestRegressor
from verticapy import vDataFrame, drop, set_option, vertica_conn
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
def titanic_vd(base):
    from verticapy.datasets import load_titanic

    titanic = load_titanic(cursor=base.cursor)
    yield titanic
    with warnings.catch_warnings(record=True) as w:
        drop(name="public.titanic", cursor=base.cursor)

@pytest.fixture(scope="module")
def rfr_data_vd(base):
    base.cursor.execute("DROP TABLE IF EXISTS public.rfr_data")
    base.cursor.execute(
        'CREATE TABLE IF NOT EXISTS public.rfr_data(Id INT, transportation INT, gender VARCHAR, "owned cars" INT, cost VARCHAR, income CHAR(4))'
    )
    base.cursor.execute("INSERT INTO rfr_data VALUES (1, 0, 'Male', 0, 'Cheap', 'Low')")
    base.cursor.execute("INSERT INTO rfr_data VALUES (2, 0, 'Male', 1, 'Cheap', 'Med')")
    base.cursor.execute(
        "INSERT INTO rfr_data VALUES (3, 1, 'Female', 1, 'Cheap', 'Med')"
    )
    base.cursor.execute(
        "INSERT INTO rfr_data VALUES (4, 0, 'Female', 0, 'Cheap', 'Low')"
    )
    base.cursor.execute("INSERT INTO rfr_data VALUES (5, 0, 'Male', 1, 'Cheap', 'Med')")
    base.cursor.execute(
        "INSERT INTO rfr_data VALUES (6, 1, 'Male', 0, 'Standard', 'Med')"
    )
    base.cursor.execute(
        "INSERT INTO rfr_data VALUES (7, 1, 'Female', 1, 'Standard', 'Med')"
    )
    base.cursor.execute(
        "INSERT INTO rfr_data VALUES (8, 2, 'Female', 1, 'Expensive', 'Hig')"
    )
    base.cursor.execute(
        "INSERT INTO rfr_data VALUES (9, 2, 'Male', 2, 'Expensive', 'Med')"
    )
    base.cursor.execute(
        "INSERT INTO rfr_data VALUES (10, 2, 'Female', 2, 'Expensive', 'Hig')"
    )
    base.cursor.execute("COMMIT")

    rfr_data = vDataFrame(input_relation="public.rfr_data", cursor=base.cursor)
    yield rfr_data
    with warnings.catch_warnings(record=True) as w:
        drop(name="public.rfr_data", cursor=base.cursor)


@pytest.fixture(scope="module")
def model(base, rfr_data_vd):
    base.cursor.execute("DROP MODEL IF EXISTS rfr_model_test")

    base.cursor.execute(
        "SELECT rf_regressor('rfr_model_test', 'public.rfr_data', 'TransPortation', '*' USING PARAMETERS exclude_columns='id, transportation', mtry=4, ntree=3, max_breadth=100, sampling_size=1, max_depth=6, min_leaf_size=1, min_info_gain=0.0, nbins=40, seed=1, id_column='id')"
    )

    # I could use load_model but it is buggy
    model_class = RandomForestRegressor(
        "rfr_model_test",
        cursor=base.cursor,
        n_estimators=3,
        max_features=4,
        max_leaf_nodes=100,
        sample=1.0,
        max_depth=6,
        min_samples_leaf=1,
        min_info_gain=0.0,
        nbins=40,
    )
    model_class.input_relation = "public.rfr_data"
    model_class.test_relation = model_class.input_relation
    model_class.X = ['"Gender"', '"owned cars"', '"cost"', '"income"']
    model_class.y = '"TransPortation"'

    yield model_class
    model_class.drop()


class TestRFR:
    def test_repr(self, model):
        assert "rf_regressor" in model.__repr__()
        model_repr = RandomForestRegressor("model_repr")
        model_repr.drop()
        assert model_repr.__repr__() == "<RandomForestRegressor>"

    def test_contour(self, base, titanic_vd):
        model_test = RandomForestRegressor("model_contour", cursor=base.cursor)
        model_test.drop()
        model_test.fit(
            titanic_vd,
            ["age", "fare",],
            "survived",
        )
        result = model_test.contour()
        assert len(result.get_default_bbox_extra_artists()) == 38
        model_test.drop()

    def test_deploySQL(self, model):
        expected_sql = "PREDICT_RF_REGRESSOR(\"Gender\", \"owned cars\", \"cost\", \"income\" USING PARAMETERS model_name = 'rfr_model_test', match_by_pos = 'true')"
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_drop(self, base):
        base.cursor.execute("DROP MODEL IF EXISTS rfr_model_test_drop")
        model_test = RandomForestRegressor("rfr_model_test_drop", cursor=base.cursor)
        model_test.fit(
            "public.rfr_data",
            ["Gender", '"owned cars"', "cost", "income"],
            "TransPortation",
        )

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

        assert fim["index"] == ["cost", "owned cars", "gender", "income"]
        assert fim["importance"] == [88.41, 7.25, 4.35, 0.0]
        assert fim["sign"] == [1, 1, 1, 0]
        plt.close("all")

    def test_get_attr(self, model):
        m_att = model.get_attr()

        assert m_att["attr_name"] == [
            "tree_count",
            "rejected_row_count",
            "accepted_row_count",
            "call_string",
            "details",
        ]
        assert m_att["attr_fields"] == [
            "tree_count",
            "rejected_row_count",
            "accepted_row_count",
            "call_string",
            "predictor, type",
        ]
        assert m_att["#_of_rows"] == [1, 1, 1, 1, 4]

        m_att_details = model.get_attr(attr_name="details")

        assert m_att_details["predictor"] == ["gender", "owned cars", "cost", "income"]
        assert m_att_details["type"] == [
            "char or varchar",
            "int",
            "char or varchar",
            "char or varchar",
        ]

        assert model.get_attr("tree_count")["tree_count"][0] == 3
        assert model.get_attr("rejected_row_count")["rejected_row_count"][0] == 0
        assert model.get_attr("accepted_row_count")["accepted_row_count"][0] == 10
        assert (
            model.get_attr("call_string")["call_string"][0]
            == "SELECT rf_regressor('public.rfr_model_test', 'public.rfr_data', '\"transportation\"', '*' USING PARAMETERS exclude_columns='id, transportation', ntree=3, mtry=4, sampling_size=1, max_depth=6, max_breadth=100, min_leaf_size=1, min_info_gain=0, nbins=40);"
        )

    def test_get_params(self, model):
        assert model.get_params() == {
            "n_estimators": 3,
            "max_features": 4,
            "max_leaf_nodes": 100,
            "sample": 1,
            "max_depth": 6,
            "min_samples_leaf": 1,
            "min_info_gain": 0,
            "nbins": 40,
        }

    def test_get_plot(self, base, winequality_vd):
        base.cursor.execute("DROP MODEL IF EXISTS model_test_plot")
        model_test = RandomForestRegressor("model_test_plot", cursor=base.cursor)
        model_test.fit(winequality_vd, ["alcohol"], "quality")
        result = model_test.plot()
        assert len(result.get_default_bbox_extra_artists()) == 9
        plt.close("all")
        model_test.drop()

    @pytest.mark.skip(reason="sklearn tree only work for numerical values.")
    def test_to_sklearn(self, model):
        md = model.to_sklearn()
        model.cursor.execute(
            "SELECT PREDICT_RF_REGRESSOR('Male', 0, 'Cheap', 'Low' USING PARAMETERS model_name = '{}', match_by_pos=True)".format(
                model.name
            )
        )
        prediction = model.cursor.fetchone()[0]
        assert prediction == pytest.approx(md.predict([["Male", 0, "Cheap", "Low"]])[0])

    def test_to_python(self, model):
        model.cursor.execute(
            "SELECT PREDICT_RF_REGRESSOR('Male', 0, 'Cheap', 'Low' USING PARAMETERS model_name = '{}', match_by_pos=True)::float".format(
                model.name
            )
        )
        prediction = model.cursor.fetchone()[0]
        assert prediction == pytest.approx(model.to_python(return_str=False)([['Male', 0, 'Cheap', 'Low']])[0])

    def test_to_sql(self, model):
        model.cursor.execute(
            "SELECT PREDICT_RF_REGRESSOR(* USING PARAMETERS model_name = '{}', match_by_pos=True)::float, {}::float FROM (SELECT 'Male' AS \"Gender\", 0 AS \"owned cars\", 'Cheap' AS \"cost\", 'Low' AS \"income\") x".format(
                model.name, model.to_sql()
            )
        )
        prediction = model.cursor.fetchone()
        assert prediction[0] == pytest.approx(prediction[1])

    @pytest.mark.skip(reason="not yet available")
    def test_shapExplainer(self, model):
        explainer = model.shapExplainer()
        assert explainer.expected_value[0] == pytest.approx(5.81837771)

    def test_get_predicts(self, rfr_data_vd, model):
        rfr_data_copy = rfr_data_vd.copy()
        model.predict(
            rfr_data_copy,
            X=["Gender", '"owned cars"', "cost", "income"],
            name="predicted_quality",
        )

        assert rfr_data_copy["predicted_quality"].mean() == pytest.approx(0.9, abs=1e-6)

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
        assert reg_rep["value"][0] == pytest.approx(1.0, abs=1e-6)
        assert reg_rep["value"][1] == pytest.approx(0.0, abs=1e-6)
        assert reg_rep["value"][2] == pytest.approx(0.0, abs=1e-6)
        assert reg_rep["value"][3] == pytest.approx(0.0, abs=1e-6)
        assert reg_rep["value"][4] == pytest.approx(0.0, abs=1e-6)
        assert reg_rep["value"][5] == pytest.approx(0.0, abs=1e-6)
        assert reg_rep["value"][6] == pytest.approx(1.0, abs=1e-6)
        assert reg_rep["value"][7] == pytest.approx(1.0, abs=1e-6)
        assert reg_rep["value"][8] == pytest.approx(-float("inf"), abs=1e-6)
        assert reg_rep["value"][9] == pytest.approx(-float("inf"), abs=1e-6)

        reg_rep_details = model.regression_report("details")
        assert reg_rep_details["value"][2:] == [
            10.0,
            4,
            pytest.approx(1.0),
            pytest.approx(1.0),
            float("inf"),
            pytest.approx(0.0),
            pytest.approx(-1.73372940858763),
            pytest.approx(0.223450528977454),
            pytest.approx(3.76564442746721),
        ]

        reg_rep_anova = model.regression_report("anova")
        assert reg_rep_anova["SS"] == [
            pytest.approx(6.9),
            pytest.approx(0.0),
            pytest.approx(6.9),
        ]
        assert reg_rep_anova["MS"][:-1] == [pytest.approx(1.725), pytest.approx(0.0)]

    def test_score(self, model):
        # method = "max"
        assert model.score(method="max") == pytest.approx(0, abs=1e-6)
        # method = "mae"
        assert model.score(method="mae") == pytest.approx(0, abs=1e-6)
        # method = "median"
        assert model.score(method="median") == pytest.approx(0, abs=1e-6)
        # method = "mse"
        assert model.score(method="mse") == pytest.approx(0.0, abs=1e-6)
        # method = "rmse"
        assert model.score(method="rmse") == pytest.approx(0.0, abs=1e-6)
        # method = "msl"
        assert model.score(method="msle") == pytest.approx(0.0, abs=1e-6)
        # method = "r2"
        assert model.score() == pytest.approx(1.0, abs=1e-6)
        # method = "r2a"
        assert model.score(method="r2a") == pytest.approx(1.0, abs=1e-6)
        # method = "var"
        assert model.score(method="var") == pytest.approx(1.0, abs=1e-6)
        # method = "aic"
        assert model.score(method="aic") == pytest.approx(-float("inf"), abs=1e-6)
        # method = "bic"
        assert model.score(method="bic") == pytest.approx(-float("inf"), abs=1e-6)

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
        model.set_params({"max_features": 1000})

        assert model.get_params()["max_features"] == 1000

    def test_model_from_vDF(self, base, rfr_data_vd):
        base.cursor.execute("DROP MODEL IF EXISTS rfr_from_vDF")
        model_test = RandomForestRegressor("rfr_from_vDF", cursor=base.cursor)
        model_test.fit(rfr_data_vd, ["gender"], "transportation")

        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'rfr_from_vDF'"
        )
        assert base.cursor.fetchone()[0] == "rfr_from_vDF"

        model_test.drop()

    def test_export_graphviz(self, model):
        gvz_tree_0 = model.export_graphviz(tree_id=0)
        expected_gvz_0 = 'digraph Tree{\n1 [label = "cost == Expensive ?", color="blue"];\n1 -> 2 [label = "yes", color = "black"];\n1 -> 3 [label = "no", color = "black"];\n2 [label = "prediction: 2.000000, variance: 0", color="red"];\n3 [label = "cost == Cheap ?", color="blue"];\n3 -> 6 [label = "yes", color = "black"];\n3 -> 7 [label = "no", color = "black"];\n6 [label = "gender == Female ?", color="blue"];\n6 -> 12 [label = "yes", color = "black"];\n6 -> 13 [label = "no", color = "black"];\n12 [label = "owned cars < 0.050000 ?", color="blue"];\n12 -> 24 [label = "yes", color = "black"];\n12 -> 25 [label = "no", color = "black"];\n24 [label = "prediction: 0.000000, variance: 0", color="red"];\n25 [label = "prediction: 1.000000, variance: 0", color="red"];\n13 [label = "prediction: 0.000000, variance: 0", color="red"];\n7 [label = "prediction: 1.000000, variance: 0", color="red"];\n}'

        assert gvz_tree_0 == expected_gvz_0

    def test_get_tree(self, model):
        tree_1 = model.get_tree(tree_id=1)

        assert tree_1["prediction"] == [
            None,
            "2.000000",
            None,
            None,
            "1.000000",
            None,
            "0.000000",
            "0.000000",
            "1.000000",
        ]

    def test_plot_tree(self, model):
        result = model.plot_tree()
        assert result.by_attr()[0:3] == "[1]"
