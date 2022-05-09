# (c) Copyright [2018-2022] Micro Focus or one of its affiliates.
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

# Pytest
import pytest

# Standard Python Modules
import os

# Other Modules
import matplotlib.pyplot as plt
import xgboost as xgb

# VerticaPy
from verticapy.tests.conftest import get_version
from verticapy import (
    vDataFrame,
    drop,
    set_option,
)
from verticapy.connect import current_cursor
from verticapy.datasets import load_winequality, load_titanic, load_dataset_reg
from verticapy.learn.ensemble import XGBoostRegressor

set_option("print_info", False)


@pytest.fixture(scope="module")
def winequality_vd():
    winequality = load_winequality()
    yield winequality
    drop(name="public.winequality")


@pytest.fixture(scope="module")
def titanic_vd():
    titanic = load_titanic()
    yield titanic
    drop(name="public.titanic")


@pytest.fixture(scope="module")
def xgbr_data_vd():
    xgbr_data = load_dataset_reg(table_name="xgbr_data", schema="public")
    yield xgbr_data
    drop(name="public.xgbr_data", method="table")


@pytest.fixture(scope="module")
def model(xgbr_data_vd):
    current_cursor().execute("DROP MODEL IF EXISTS xgbr_model_test")

    current_cursor().execute(
        "SELECT xgb_regressor('xgbr_model_test', 'public.xgbr_data', 'TransPortation', '*' USING PARAMETERS exclude_columns='id, transportation', min_split_loss=0.1, max_ntree=3, learning_rate=0.2, sampling_size=1, max_depth=6, nbins=40, seed=1, id_column='id')"
    )

    # I could use load_model but it is buggy
    model_class = XGBoostRegressor(
        "xgbr_model_test",
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
    model_class.prior_ = model_class.get_prior()

    yield model_class
    model_class.drop()


@pytest.mark.skipif(
    get_version()[0] < 10 or (get_version()[0] == 10 and get_version()[1] == 0),
    reason="requires vertica 10.1 or higher",
)
class TestXGBR:
    def test_contour(self, titanic_vd):
        model_test = XGBoostRegressor("model_contour",)
        model_test.drop()
        model_test.fit(
            titanic_vd, ["age", "fare"], "survived",
        )
        result = model_test.contour()
        assert len(result.get_default_bbox_extra_artists()) in (37, 34)
        model_test.drop()

    def test_deploySQL(self, model):
        expected_sql = 'PREDICT_XGB_REGRESSOR("Gender", "owned cars", "cost", "income" USING PARAMETERS model_name = \'xgbr_model_test\', match_by_pos = \'true\')'
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_drop(self):
        current_cursor().execute("DROP MODEL IF EXISTS xgbr_model_test_drop")
        model_test = XGBoostRegressor("xgbr_model_test_drop",)
        model_test.fit(
            "public.xgbr_data",
            ['"Gender"', '"owned cars"', '"cost"', '"income"'],
            "TransPortation",
        )

        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'xgbr_model_test_drop'"
        )
        assert current_cursor().fetchone()[0] == "xgbr_model_test_drop"

        model_test.drop()
        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'xgbr_model_test_drop'"
        )
        assert current_cursor().fetchone() is None

    def test_to_python(self, model, titanic_vd):
        current_cursor().execute(
            "SELECT PREDICT_XGB_REGRESSOR('Male', 0, 'Cheap', 'Low' USING PARAMETERS model_name = '{}', match_by_pos=True)::float".format(
                model.name
            )
        )
        prediction = current_cursor().fetchone()[0]
        assert prediction == pytest.approx(
            float(model.to_python()([["Male", 0, "Cheap", "Low"]])[0])
        )

    def test_to_sql(self, model):
        current_cursor().execute(
            "SELECT PREDICT_XGB_REGRESSOR(* USING PARAMETERS model_name = '{}', match_by_pos=True)::float, {}::float FROM (SELECT 'Male' AS \"Gender\", 0 AS \"owned cars\", 'Cheap' AS \"cost\", 'Low' AS \"income\") x".format(
                model.name, model.to_sql()
            )
        )
        prediction = current_cursor().fetchone()
        assert prediction[0] == pytest.approx(prediction[1])

    def test_to_memmodel(self, model):
        mmodel = model.to_memmodel()
        res = mmodel.predict(
            [["Male", 0, "Cheap", "Low"], ["Female", 1, "Expensive", "Low"]]
        )
        res_py = model.to_python()(
            [["Male", 0, "Cheap", "Low"], ["Female", 1, "Expensive", "Low"]]
        )
        assert res[0] == res_py[0]
        assert res[1] == res_py[1]
        vdf = vDataFrame("public.xgbr_data")
        vdf["prediction_sql"] = mmodel.predict_sql(
            ['"Gender"', '"owned cars"', '"cost"', '"income"']
        )
        model.predict(vdf, name="prediction_vertica_sql")
        score = vdf.score("prediction_sql", "prediction_vertica_sql", "r2")
        assert score == pytest.approx(1.0)

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
            "initial_prediction",
        ]
        assert m_att["attr_fields"] == [
            "tree_count",
            "rejected_row_count",
            "accepted_row_count",
            "call_string",
            "predictor, type",
            "initial_prediction",
        ]
        assert m_att["#_of_rows"] == [1, 1, 1, 1, 4, 1]

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
            "xgb_regressor('public.xgbr_model_test', 'public.xgbr_data', '\"transportation\"', '*' USING PARAMETERS"
            in model.get_attr("call_string")["call_string"][0]
        )

    def test_get_params(self, model):
        assert model.get_params() == {
            "max_ntree": 3,
            "min_split_loss": 0.1,
            "learning_rate": 0.2,
            "sample": 1.0,
            "max_depth": 6,
            "nbins": 40,
            "split_proposal_method": "global",
            "tol": 0.001,
            "weight_reg": 0.0,
        } or model.get_params() == {
            "max_ntree": 3,
            "min_split_loss": 0.1,
            "learning_rate": 0.2,
            "sample": 1.0,
            "max_depth": 6,
            "nbins": 40,
            "split_proposal_method": "global",
            "tol": 0.001,
            "weight_reg": 0.0,
            "col_sample_by_tree": 1.0,
            "col_sample_by_node": 1.0,
        }

    def test_get_predicts(self, xgbr_data_vd, model):
        xgbr_data_copy = xgbr_data_vd.copy()
        model.predict(
            xgbr_data_copy,
            X=["Gender", '"owned cars"', "cost", "income"],
            name="predicted_quality",
        )

        assert xgbr_data_copy["predicted_quality"].mean() in (
            pytest.approx(0.9, abs=1e-6),
            pytest.approx(0.908453240740741, abs=1e-6),
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
        assert reg_rep["value"][0] in (
            pytest.approx(0.737856, abs=1e-6),
            pytest.approx(0.60448287427822, abs=1e-6),
        )
        assert reg_rep["value"][1] in (
            pytest.approx(0.5632, abs=1e-6),
            pytest.approx(0.6755375, abs=1e-6),
        )
        assert reg_rep["value"][2] in (
            pytest.approx(0.4608, abs=1e-6),
            pytest.approx(0.5527125, abs=1e-6),
        )
        assert reg_rep["value"][3] in (
            pytest.approx(0.36864, abs=1e-6),
            pytest.approx(0.454394259259259, abs=1e-6),
        )
        assert reg_rep["value"][4] in (
            pytest.approx(0.18087936, abs=1e-6),
            pytest.approx(0.272978274027049, abs=1e-6),
        )
        assert reg_rep["value"][5] in (
            pytest.approx(0.42529914178140543, abs=1e-6),
            pytest.approx(0.5224732280481451, abs=1e-6),
        )
        assert reg_rep["value"][6] in (
            pytest.approx(0.737856, abs=1e-6),
            pytest.approx(0.604379313004277, abs=1e-6),
        )
        assert reg_rep["value"][7] in (
            pytest.approx(0.5281407999999999, abs=1e-6),
            pytest.approx(0.2878827634076987, abs=1e-6),
        )
        assert reg_rep["value"][8] in (
            pytest.approx(7.900750107239094, abs=1e-6),
            pytest.approx(12.016369307174802, abs=1e-6),
        )
        assert reg_rep["value"][9] in (
            pytest.approx(-5.586324427790675, abs=1e-6),
            pytest.approx(-1.4707052278549675, abs=1e-6),
        )

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
        ] or reg_rep_details["value"][2:] == [
            10.0,
            4,
            pytest.approx(0.604379313004277),
            pytest.approx(0.2878827634076987),
            pytest.approx(0.4418800475932531),
            pytest.approx(0.7760066793800073),
            pytest.approx(-1.73372940858763),
            pytest.approx(0.223450528977454),
            pytest.approx(3.76564442746721),
        ]

        reg_rep_anova = model.regression_report("anova")
        assert reg_rep_anova["SS"] == [
            pytest.approx(1.6431936),
            pytest.approx(1.8087936),
            pytest.approx(6.9),
        ] or reg_rep_anova["SS"] == [
            pytest.approx(0.964989221751972),
            pytest.approx(2.72978274027049),
            pytest.approx(6.9),
        ]
        assert reg_rep_anova["MS"][:-1] == [
            pytest.approx(0.4107984),
            pytest.approx(0.36175872),
        ] or reg_rep_anova["MS"][:-1] == [
            pytest.approx(0.241247305437993),
            pytest.approx(0.545956548054098),
        ]

    def test_score(self, model):
        # method = "max"
        assert model.score(method="max") in (
            pytest.approx(0.5632, abs=1e-6),
            pytest.approx(0.6755375, abs=1e-6),
        )
        # method = "mae"
        assert model.score(method="mae") in (
            pytest.approx(0.36864, abs=1e-6),
            pytest.approx(0.454394259259259, abs=1e-6),
        )
        # method = "median"
        assert model.score(method="median") in (
            pytest.approx(0.4608, abs=1e-6),
            pytest.approx(0.5527125, abs=1e-6),
        )
        # method = "mse"
        assert model.score(method="mse") in (
            pytest.approx(0.18087936, abs=1e-6),
            pytest.approx(0.272978274027049, abs=1e-6),
        )
        # method = "rmse"
        assert model.score(method="rmse") in (
            pytest.approx(0.42529914178140543, abs=1e-6),
            pytest.approx(0.5224732280481451, abs=1e-6),
        )
        # method = "msl"
        assert model.score(method="msle") in (
            pytest.approx(0.0133204031846029, abs=1e-6),
            pytest.approx(0.0195048419826687, abs=1e-6),
        )
        # method = "r2"
        assert model.score() in (
            pytest.approx(0.737856, abs=1e-6),
            pytest.approx(0.604379313004277, abs=1e-6),
        )
        # method = "r2a"
        assert model.score(method="r2a") in (
            pytest.approx(0.5281407999999999, abs=1e-6),
            pytest.approx(0.2878827634076987, abs=1e-6),
        )
        # method = "var"
        assert model.score(method="var") in (
            pytest.approx(0.737856, abs=1e-6),
            pytest.approx(0.60448287427822, abs=1e-6),
        )
        # method = "aic"
        assert model.score(method="aic") in (
            pytest.approx(7.900750107239094, abs=1e-6),
            pytest.approx(12.016369307174802, abs=1e-6),
        )
        # method = "bic"
        assert model.score(method="bic") in (
            pytest.approx(-5.586324427790675, abs=1e-6),
            pytest.approx(-1.4707052278549675, abs=1e-6),
        )

    def test_set_params(self, model):
        model.set_params({"max_ntree": 5})

        assert model.get_params()["max_ntree"] == 5

    def test_model_from_vDF(self, xgbr_data_vd):
        current_cursor().execute("DROP MODEL IF EXISTS xgbr_from_vDF")
        model_test = XGBoostRegressor("xgbr_from_vDF",)
        model_test.fit(xgbr_data_vd, ["gender"], "transportation")

        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'xgbr_from_vDF'"
        )
        assert current_cursor().fetchone()[0] == "xgbr_from_vDF"

        model_test.drop()

    def test_to_graphviz(self, model):
        gvz_tree_1 = model.to_graphviz(
            tree_id=1,
            classes_color=["red", "blue", "green"],
            round_pred=4,
            percent=True,
            vertical=False,
            node_style={"shape": "box", "style": "filled"},
            arrow_style={"color": "blue"},
            leaf_style={"shape": "circle", "style": "filled"},
        )
        assert 'digraph Tree{\ngraph [rankdir = "LR"];\n0' in gvz_tree_1
        assert "0 -> 1" in gvz_tree_1

    def test_get_tree(self, model):
        tree_1 = model.get_tree(tree_id=1)
        assert tree_1["prediction"][0] == None
        assert tree_1["prediction"][1] in ("0.880000", "0.701250")
        assert tree_1["prediction"][2] == None
        assert tree_1["prediction"][3] == None
        assert tree_1["prediction"][4] in ("0.080000", "0.057778")
        assert tree_1["prediction"][5] == None
        assert tree_1["prediction"][6] in ("-0.720000", "-0.573750")
        assert tree_1["prediction"][7] in ("-0.720000", "-0.405000")
        assert tree_1["prediction"][8] in ("0.080000", "0.045000")

    def test_get_plot(self, winequality_vd):
        current_cursor().execute("DROP MODEL IF EXISTS model_test_plot")
        model_test = XGBoostRegressor("model_test_plot",)
        model_test.fit(winequality_vd, ["alcohol"], "quality")
        result = model_test.plot()
        assert len(result.get_default_bbox_extra_artists()) in (9, 12)
        plt.close("all")
        model_test.drop()

    def test_plot_tree(self, model):
        result = model.plot_tree()
        assert model.to_graphviz() == result.source.strip()

    def test_to_json(self, titanic_vd):
        titanic = titanic_vd.copy()
        titanic.fillna()
        path = "verticapy_test_xgbr.json"
        X = ["pclass", "age", "survived"]
        y = "fare"
        model = XGBoostRegressor(
            "verticapy_xgb_regressor_test", max_ntree=10, max_depth=5
        )
        model.drop()
        model.fit(titanic, X, y)
        X_test = titanic[X].to_numpy()
        y_test_vertica = model.to_python()(X_test)
        if os.path.exists(path):
            os.remove(path)
        model.to_json(path)
        model_python = xgb.XGBRegressor()
        model_python.load_model(path)
        y_test_python = model_python.predict(X_test)
        result = (y_test_vertica - y_test_python) ** 2
        result = result.sum() / len(result)
        assert result == pytest.approx(0.0, abs=1.0e-11)
        model.drop()
        os.remove(path)
