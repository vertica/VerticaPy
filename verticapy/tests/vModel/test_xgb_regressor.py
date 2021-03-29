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
from verticapy.learn.ensemble import XGBoostRegressor
from verticapy.tests.conftest import get_version
from verticapy import vDataFrame, drop, version, set_option, vertica_conn
import matplotlib.pyplot as plt
import numpy as np

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
def xgbr_data_vd(base):
    base.cursor.execute("DROP TABLE IF EXISTS public.xgbr_data")
    base.cursor.execute(
        'CREATE TABLE IF NOT EXISTS public.xgbr_data(Id INT, transportation INT, gender VARCHAR, "owned cars" INT, cost VARCHAR, income CHAR(4))'
    )
    base.cursor.execute(
        "INSERT INTO xgbr_data VALUES (1, 0, 'Male', 0, 'Cheap', 'Low')"
    )
    base.cursor.execute(
        "INSERT INTO xgbr_data VALUES (2, 0, 'Male', 1, 'Cheap', 'Med')"
    )
    base.cursor.execute(
        "INSERT INTO xgbr_data VALUES (3, 1, 'Female', 1, 'Cheap', 'Med')"
    )
    base.cursor.execute(
        "INSERT INTO xgbr_data VALUES (4, 0, 'Female', 0, 'Cheap', 'Low')"
    )
    base.cursor.execute(
        "INSERT INTO xgbr_data VALUES (5, 0, 'Male', 1, 'Cheap', 'Med')"
    )
    base.cursor.execute(
        "INSERT INTO xgbr_data VALUES (6, 1, 'Male', 0, 'Standard', 'Med')"
    )
    base.cursor.execute(
        "INSERT INTO xgbr_data VALUES (7, 1, 'Female', 1, 'Standard', 'Med')"
    )
    base.cursor.execute(
        "INSERT INTO xgbr_data VALUES (8, 2, 'Female', 1, 'Expensive', 'Hig')"
    )
    base.cursor.execute(
        "INSERT INTO xgbr_data VALUES (9, 2, 'Male', 2, 'Expensive', 'Med')"
    )
    base.cursor.execute(
        "INSERT INTO xgbr_data VALUES (10, 2, 'Female', 2, 'Expensive', 'Hig')"
    )
    base.cursor.execute("COMMIT")

    xgbr_data = vDataFrame(input_relation="public.xgbr_data", cursor=base.cursor)
    yield xgbr_data
    with warnings.catch_warnings(record=True) as w:
        drop(name="public.xgbr_data", cursor=base.cursor)

@pytest.fixture(scope="module")
def model(base, xgbr_data_vd):
    base.cursor.execute("DROP MODEL IF EXISTS xgbr_model_test")

    base.cursor.execute(
        "SELECT xgb_regressor('xgbr_model_test', 'public.xgbr_data', 'TransPortation', '*' USING PARAMETERS exclude_columns='id, transportation', min_split_loss=0.1, max_ntree=3, learning_rate=0.2, sampling_size=1, max_depth=6, nbins=40, seed=1, id_column='id')"
    )

    # I could use load_model but it is buggy
    model_class = XGBoostRegressor(
        "xgbr_model_test",
        cursor=base.cursor,
        max_ntree=3,
        min_split_loss=0.1,
        learning_rate=0.2,
        sample=1.0,
        max_depth=6,
        nbins=40,
    )
    model_class.input_relation = "public.xgbr_data"
    model_class.test_relation = model_class.input_relation
    model_class.X = ['"Gender"', '"owned cars"', '"cost"', '"income"']
    model_class.y = '"TransPortation"'

    yield model_class
    model_class.drop()

@pytest.mark.skipif(get_version()[0] < 10 or (get_version()[0] == 10 and get_version()[1] == 0), reason="requires vertica 10.1 or higher")
class TestXGBR:
    def test_contour(self, base, titanic_vd):
        model_test = XGBoostRegressor("model_contour", cursor=base.cursor)
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
        expected_sql = "PREDICT_XGB_REGRESSOR(\"Gender\", \"owned cars\", \"cost\", \"income\" USING PARAMETERS model_name = 'xgbr_model_test', match_by_pos = 'true')"
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_drop(self, base):
        base.cursor.execute("DROP MODEL IF EXISTS xgbr_model_test_drop")
        model_test = XGBoostRegressor("xgbr_model_test_drop", cursor=base.cursor)
        model_test.fit(
            "public.xgbr_data",
            ['"Gender"', '"owned cars"', '"cost"', '"income"'],
            "TransPortation",
        )

        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'xgbr_model_test_drop'"
        )
        assert base.cursor.fetchone()[0] == "xgbr_model_test_drop"

        model_test.drop()
        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'xgbr_model_test_drop'"
        )
        assert base.cursor.fetchone() is None

    def test_to_sql(self, model):
        model.cursor.execute(
            "SELECT PREDICT_XGB_REGRESSOR(* USING PARAMETERS model_name = '{}', match_by_pos=True)::float, {}::float FROM (SELECT 'Male' AS \"Gender\", 0 AS \"owned cars\", 'Cheap' AS \"cost\", 'Low' AS \"income\") x".format(
                model.name, model.to_sql()
            )
        )
        prediction = model.cursor.fetchone()
        assert prediction[0] == pytest.approx(prediction[1])

    @pytest.mark.skip(reason="not yet available")
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

        assert m_att_details["predictor"] == [
            "gender",
            "owned cars",
            "cost",
            "income",
        ]
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
            == "xgb_regressor('public.xgbr_model_test', 'public.xgbr_data', '\"transportation\"', '*' USING PARAMETERS exclude_columns='id, transportation', max_ntree=3, max_depth=6, nbins=40, objective=squarederror, split_proposal_method=global, epsilon=0.001, learning_rate=0.2, min_split_loss=0.1, weight_reg=0, sampling_size=1, seed=1, id_column='id')"
        )

    def test_get_params(self, model):
        assert model.get_params() == {
            "max_ntree": 3,
            "min_split_loss": 0.1,
            "learning_rate": 0.2,
            "sample": 1.0,
            "max_depth": 6,
            "nbins": 40,
            "objective": "squarederror",
            "split_proposal_method": "global",
            "tol": 0.001,
            "weight_reg": 0.0,
        }

    @pytest.mark.skip(reason="not yet available.")
    def test_to_sklearn(self, model):
        md = model.to_sklearn()
        model.cursor.execute(
            "SELECT PREDICT_xgb_REGRESSOR('Male', 0, 'Cheap', 'Low' USING PARAMETERS model_name = '{}', match_by_pos=True)".format(
                model.name
            )
        )
        prediction = model.cursor.fetchone()[0]
        assert prediction == pytest.approx(
            md.predict([["Male", 0, "Cheap", "Low"]])[0]
        )

    @pytest.mark.skip(reason="not yet available.")
    def test_shapExplainer(self, model):
        explainer = model.shapExplainer()
        assert explainer.expected_value[0] == pytest.approx(5.81837771)

    def test_get_predicts(self, xgbr_data_vd, model):
        xgbr_data_copy = xgbr_data_vd.copy()
        model.predict(
            xgbr_data_copy,
            X=["Gender", '"owned cars"', "cost", "income"],
            name="predicted_quality",
        )

        assert xgbr_data_copy["predicted_quality"].mean() == pytest.approx(
            0.9, abs=1e-6
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
        assert reg_rep["value"][0] == pytest.approx(0.737856, abs=1e-6)
        assert reg_rep["value"][1] == pytest.approx(0.5632, abs=1e-6)
        assert reg_rep["value"][2] == pytest.approx(0.4608, abs=1e-6)
        assert reg_rep["value"][3] == pytest.approx(0.36864, abs=1e-6)
        assert reg_rep["value"][4] == pytest.approx(0.18087936, abs=1e-6)
        assert reg_rep["value"][5] == pytest.approx(0.42529914178140543, abs=1e-6)
        assert reg_rep["value"][6] == pytest.approx(0.737856, abs=1e-6)
        assert reg_rep["value"][7] == pytest.approx(0.5281407999999999, abs=1e-6)
        assert reg_rep["value"][8] == pytest.approx(7.900750107239094, abs=1e-6)
        assert reg_rep["value"][9] == pytest.approx(-5.586324427790675, abs=1e-6)

        reg_rep_details = model.regression_report("details")
        assert reg_rep_details["value"][2:] == [
            10.0,
            4,
            pytest.approx(0.737856),
            pytest.approx(0.5281407999999999),
            pytest.approx(1.13555908203125),
            pytest.approx(0.3938936106224664),
            pytest.approx(-1.73372940858763),
            pytest.approx(0.223450528977454),
            pytest.approx(3.76564442746721),
        ]

        reg_rep_anova = model.regression_report("anova")
        assert reg_rep_anova["SS"] == [
            pytest.approx(1.6431936),
            pytest.approx(1.8087936),
            pytest.approx(6.9),
        ]
        assert reg_rep_anova["MS"][:-1] == [
            pytest.approx(0.4107984),
            pytest.approx(0.36175872),
        ]

    def test_score(self, model):
        # method = "max"
        assert model.score(method="max") == pytest.approx(0.5632, abs=1e-6)
        # method = "mae"
        assert model.score(method="mae") == pytest.approx(0.36864, abs=1e-6)
        # method = "median"
        assert model.score(method="median") == pytest.approx(0.4608, abs=1e-6)
        # method = "mse"
        assert model.score(method="mse") == pytest.approx(0.18087936, abs=1e-6)
        # method = "rmse"
        assert model.score(method="rmse") == pytest.approx(
            0.42529914178140543, abs=1e-6
        )
        # method = "msl"
        assert model.score(method="msle") == pytest.approx(
            0.0133204031846029, abs=1e-6
        )
        # method = "r2"
        assert model.score() == pytest.approx(0.737856, abs=1e-6)
        # method = "r2a"
        assert model.score(method="r2a") == pytest.approx(
            0.5281407999999999, abs=1e-6
        )
        # method = "var"
        assert model.score(method="var") == pytest.approx(0.737856, abs=1e-6)
        # method = "aic"
        assert model.score(method="aic") == pytest.approx(
            7.900750107239094, abs=1e-6
        )
        # method = "bic"
        assert model.score(method="bic") == pytest.approx(
            -5.586324427790675, abs=1e-6
        )

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
        model.set_params({"max_ntree": 5})

        assert model.get_params()["max_ntree"] == 5

    def test_model_from_vDF(self, base, xgbr_data_vd):
        base.cursor.execute("DROP MODEL IF EXISTS xgbr_from_vDF")
        model_test = XGBoostRegressor("xgbr_from_vDF", cursor=base.cursor)
        model_test.fit(xgbr_data_vd, ["gender"], "transportation")

        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'xgbr_from_vDF'"
        )
        assert base.cursor.fetchone()[0] == "xgbr_from_vDF"

        model_test.drop()

    def test_export_graphviz(self, model):
        gvz_tree_0 = model.export_graphviz(tree_id=0)
        expected_gvz_0 = 'digraph Tree{\n1 [label = "cost == Expensive ?", color="blue"];\n1 -> 2 [label = "yes", color = "black"];\n1 -> 3 [label = "no", color = "black"];\n2 [label = "prediction: 1.100000", color="red"];\n3 [label = "cost == Cheap ?", color="blue"];\n3 -> 6 [label = "yes", color = "black"];\n3 -> 7 [label = "no", color = "black"];\n6 [label = "gender == Female ?", color="blue"];\n6 -> 10 [label = "yes", color = "black"];\n6 -> 11 [label = "no", color = "black"];\n10 [label = "income == Low ?", color="blue"];\n10 -> 14 [label = "yes", color = "black"];\n10 -> 15 [label = "no", color = "black"];\n14 [label = "prediction: -0.900000", color="red"];\n15 [label = "prediction: 0.100000", color="red"];\n11 [label = "prediction: -0.900000", color="red"];\n7 [label = "prediction: 0.100000", color="red"];\n}'

        assert gvz_tree_0 == expected_gvz_0

    def test_get_tree(self, model):
        tree_1 = model.get_tree(tree_id=1)

        assert tree_1["prediction"] == [
            None,
            "0.880000",
            None,
            None,
            "0.080000",
            None,
            "-0.720000",
            "-0.720000",
            "0.080000",
        ]

    def test_get_plot(self, base, winequality_vd):
        base.cursor.execute("DROP MODEL IF EXISTS model_test_plot")
        model_test = XGBoostRegressor("model_test_plot", cursor=base.cursor)
        model_test.fit(winequality_vd, ["alcohol"], "quality")
        result = model_test.plot()
        assert len(result.get_default_bbox_extra_artists()) == 9
        plt.close("all")
        model_test.drop()

    def test_plot_tree(self, model):
        result = model.plot_tree()
        assert result.by_attr()[0:3] == "[1]"
