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
from verticapy.learn.tree import DummyTreeRegressor
from verticapy import vDataFrame, drop, set_option, vertica_conn
import matplotlib.pyplot as plt

set_option("print_info", False)


@pytest.fixture(scope="module")
def tr_data_vd(base):
    base.cursor.execute("DROP TABLE IF EXISTS public.tr_data")
    base.cursor.execute(
        'CREATE TABLE IF NOT EXISTS public.tr_data(Id INT, transportation INT, gender VARCHAR, "owned cars" INT, cost VARCHAR, income CHAR(4))'
    )
    base.cursor.execute("INSERT INTO tr_data VALUES (1, 0, 'Male', 0, 'Cheap', 'Low')")
    base.cursor.execute("INSERT INTO tr_data VALUES (2, 0, 'Male', 1, 'Cheap', 'Med')")
    base.cursor.execute(
        "INSERT INTO tr_data VALUES (3, 1, 'Female', 1, 'Cheap', 'Med')"
    )
    base.cursor.execute(
        "INSERT INTO tr_data VALUES (4, 0, 'Female', 0, 'Cheap', 'Low')"
    )
    base.cursor.execute("INSERT INTO tr_data VALUES (5, 0, 'Male', 1, 'Cheap', 'Med')")
    base.cursor.execute(
        "INSERT INTO tr_data VALUES (6, 1, 'Male', 0, 'Standard', 'Med')"
    )
    base.cursor.execute(
        "INSERT INTO tr_data VALUES (7, 1, 'Female', 1, 'Standard', 'Med')"
    )
    base.cursor.execute(
        "INSERT INTO tr_data VALUES (8, 2, 'Female', 1, 'Expensive', 'Hig')"
    )
    base.cursor.execute(
        "INSERT INTO tr_data VALUES (9, 2, 'Male', 2, 'Expensive', 'Med')"
    )
    base.cursor.execute(
        "INSERT INTO tr_data VALUES (10, 2, 'Female', 2, 'Expensive', 'Hig')"
    )
    base.cursor.execute("COMMIT")

    tr_data = vDataFrame(input_relation="public.tr_data", cursor=base.cursor)
    yield tr_data
    with warnings.catch_warnings(record=True) as w:
        drop(name="public.tr_data", cursor=base.cursor)


@pytest.fixture(scope="module")
def model(base, tr_data_vd):
    base.cursor.execute("DROP MODEL IF EXISTS tr_model_test")

    base.cursor.execute(
        "SELECT rf_regressor('tr_model_test', 'public.tr_data', 'TransPortation', '*' USING PARAMETERS exclude_columns='id, transportation', mtry=4, ntree=1, max_breadth=1e9, sampling_size=1, max_depth=100, min_leaf_size=1, min_info_gain=0.0, nbins=1000, seed=1, id_column='id')"
    )

    # I could use load_model but it is buggy
    model_class = DummyTreeRegressor("tr_model_test", cursor=base.cursor)
    model_class.input_relation = "public.tr_data"
    model_class.test_relation = model_class.input_relation
    model_class.X = ['"Gender"', '"owned cars"', '"cost"', '"income"']
    model_class.y = '"TransPortation"'

    yield model_class
    model_class.drop()

@pytest.fixture(scope="module")
def titanic_vd(base):
    from verticapy.datasets import load_titanic

    titanic = load_titanic(cursor=base.cursor)
    yield titanic
    with warnings.catch_warnings(record=True) as w:
        drop(name="public.titanic", cursor=base.cursor)


class TestDummyTreeRegressor:
    def test_repr(self, model):
        assert "SELECT rf_regressor('public.tr_model_test'," in model.__repr__()
        model_repr = DummyTreeRegressor("RF_repr")
        model_repr.drop()
        assert model_repr.__repr__() == "<RandomForestRegressor>"

    def test_contour(self, base, titanic_vd):
        model_test = DummyTreeRegressor("model_contour", cursor=base.cursor)
        model_test.drop()
        model_test.fit(
            titanic_vd,
            ["age", "fare",],
            "survived",
        )
        result = model_test.contour()
        assert len(result.get_default_bbox_extra_artists()) == 34
        model_test.drop()

    def test_deploySQL(self, model):
        expected_sql = "PREDICT_RF_REGRESSOR(\"Gender\", \"owned cars\", \"cost\", \"income\" USING PARAMETERS model_name = 'tr_model_test', match_by_pos = 'true')"
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_drop(self, base):
        base.cursor.execute("DROP MODEL IF EXISTS tr_model_test_drop")
        model_test = DummyTreeRegressor("tr_model_test_drop", cursor=base.cursor)
        model_test.fit(
            "public.tr_data",
            ["Gender", '"owned cars"', "cost", "income"],
            "TransPortation",
        )

        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'tr_model_test_drop'"
        )
        assert base.cursor.fetchone()[0] == "tr_model_test_drop"

        model_test.drop()
        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'tr_model_test_drop'"
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

        assert model.get_attr("tree_count")["tree_count"][0] == 1
        assert model.get_attr("rejected_row_count")["rejected_row_count"][0] == 0
        assert model.get_attr("accepted_row_count")["accepted_row_count"][0] == 10
        assert (
            model.get_attr("call_string")["call_string"][0]
            == "SELECT rf_regressor('public.tr_model_test', 'public.tr_data', '\"transportation\"', '*' USING PARAMETERS exclude_columns='id, transportation', ntree=1, mtry=4, sampling_size=1, max_depth=100, max_breadth=1000000000, min_leaf_size=1, min_info_gain=0, nbins=1000);"
        )

    def test_get_params(self, model):
        assert model.get_params() == {
            "n_estimators": 1,
            "max_features": "max",
            "max_leaf_nodes": 1000000000,
            "sample": 1.0,
            "max_depth": 100,
            "min_samples_leaf": 1,
            "min_info_gain": 0,
            "nbins": 1000,
        }

    @pytest.mark.skip(reason="pb with sklearn trees")
    def test_to_sklearn(self, base):
        base.cursor.execute("DROP MODEL IF EXISTS tr_model_sk_test")

        base.cursor.execute(
            "SELECT rf_regressor('tr_model_sk_test', 'public.tr_data', 'TransPortation', '\"owned cars\"' USING PARAMETERS mtry=1, ntree=1, max_breadth=1e9, sampling_size=1, max_depth=100, min_leaf_size=1, min_info_gain=0.0, nbins=1000, seed=1, id_column='id')"
        )

        # I could use load_model but it is buggy
        model_sk = DummyTreeRegressor("tr_model_sk_test", cursor=base.cursor)
        model_sk.input_relation = "public.tr_data"
        model_sk.test_relation = model_sk.input_relation
        model_sk.X = ['"owned cars"']
        model_sk.y = "TransPortation"

        md = model_sk.to_sklearn()
        model_sk.cursor.execute(
            "SELECT PREDICT_RF_REGRESSOR(1 USING PARAMETERS model_name = '{}', match_by_pos=True)".format(
                model_sk.name
            )
        )
        prediction = model_sk.cursor.fetchone()[0]
        assert prediction == pytest.approx(md.predict([1])[0])

        model_sk.drop()

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

    def test_get_predicts(self, tr_data_vd, model):
        tr_data_copy = tr_data_vd.copy()
        model.predict(
            tr_data_copy,
            X=["Gender", '"owned cars"', "cost", "income"],
            name="predicted_quality",
        )

        assert tr_data_copy["predicted_quality"].mean() == pytest.approx(0.9, abs=1e-6)

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
        # Nothing will change as Dummy Trees have no parameters
        model.set_params({"max_features": 100})
        assert model.get_params()["max_features"] == "max"

    def test_model_from_vDF(self, base, tr_data_vd):
        base.cursor.execute("DROP MODEL IF EXISTS tr_from_vDF")
        model_test = DummyTreeRegressor("tr_from_vDF", cursor=base.cursor)
        model_test.fit(tr_data_vd, ["gender"], "transportation")

        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'tr_from_vDF'"
        )
        assert base.cursor.fetchone()[0] == "tr_from_vDF"

        model_test.drop()

    def test_export_graphviz(self, model):
        gvz_tree_0 = model.export_graphviz(tree_id=0)
        expected_gvz_0 = 'digraph Tree{\n1 [label = "cost == Expensive ?", color="blue"];\n1 -> 2 [label = "yes", color = "black"];\n1 -> 3 [label = "no", color = "black"];\n2 [label = "prediction: 2.000000, variance: 0", color="red"];\n3 [label = "cost == Cheap ?", color="blue"];\n3 -> 6 [label = "yes", color = "black"];\n3 -> 7 [label = "no", color = "black"];\n6 [label = "gender == Female ?", color="blue"];\n6 -> 12 [label = "yes", color = "black"];\n6 -> 13 [label = "no", color = "black"];\n12 [label = "owned cars < 0.002000 ?", color="blue"];\n12 -> 24 [label = "yes", color = "black"];\n12 -> 25 [label = "no", color = "black"];\n24 [label = "prediction: 0.000000, variance: 0", color="red"];\n25 [label = "prediction: 1.000000, variance: 0", color="red"];\n13 [label = "prediction: 0.000000, variance: 0", color="red"];\n7 [label = "prediction: 1.000000, variance: 0", color="red"];\n}'

        assert gvz_tree_0 == expected_gvz_0

    def test_get_tree(self, model):
        tree_0 = model.get_tree()

        assert tree_0["prediction"] == [
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
