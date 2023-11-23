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
from verticapy import (
    vDataFrame,
    drop,
    set_option,
)
from verticapy.tests.conftest import get_version
from verticapy.connection import current_cursor
from verticapy.datasets import load_titanic, load_dataset_reg
from verticapy.learn.ensemble import IsolationForest

# Matplotlib skip
import matplotlib

matplotlib_version = matplotlib.__version__
skip_plt = pytest.mark.skipif(
    matplotlib_version > "3.5.2",
    reason="Test skipped on matplotlib version greater than 3.5.2",
)

set_option("print_info", False)
set_option("random_state", 1)


@pytest.fixture(scope="module")
def iforest_data_vd():
    iforest_data = load_dataset_reg(table_name="iforest_data", schema="public")
    yield iforest_data
    drop(name="public.iforest_data", method="table")


@pytest.fixture(scope="module")
def model(iforest_data_vd):
    model_class = IsolationForest(
        "iforest_model_test",
        n_estimators=100,
        max_depth=10,
        nbins=32,
        sample=0.632,
        col_sample_by_tree=0.8,
    )
    model_class.drop()
    X = ["Gender", "owned cars", "cost", "income", "TransPortation"]
    model_class.fit("public.iforest_data", X)

    yield model_class
    model_class.drop()


@pytest.fixture(scope="module")
def titanic_vd():
    titanic = load_titanic()
    yield titanic
    drop(
        name="public.titanic",
    )


@pytest.mark.skipif(
    get_version()[0] < 12,
    reason="requires vertica 12.0 or higher",
)
class TestIsolationForest:
    def test_repr(self, model):
        assert model.__repr__() == "<IsolationForest>"

    @skip_plt
    def test_contour(self, titanic_vd):
        model_test = IsolationForest(
            "model_contour_iF",
        )
        model_test.drop()
        model_test.fit(
            titanic_vd,
            ["age", "fare"],
        )
        result = model_test.contour()
        assert len(result.get_default_bbox_extra_artists()) > 30
        model_test.drop()

    def test_deploySQL(self, model):
        expected_sql = (
            '((APPLY_IFOREST("Gender", "owned cars", "cost", "income", "TransPortation" '
            "USING PARAMETERS model_name = 'iforest_model_test', match_by_pos = 'true', "
            "threshold = 0.7)).is_anomaly)::int"
        )
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

        expected_sql = (
            '((APPLY_IFOREST("Gender", "owned cars", "cost", "income", "TransPortation" '
            "USING PARAMETERS model_name = 'iforest_model_test', match_by_pos = 'true', "
            "contamination = 0.05)).is_anomaly)::int"
        )
        result_sql = model.deploySQL(contamination=0.05)

        assert result_sql == expected_sql

        expected_sql = (
            '(APPLY_IFOREST("Gender", "owned cars", "cost", '
            '"income", "TransPortation" USING PARAMETERS '
            "model_name = 'iforest_model_test', "
            "match_by_pos = 'true')).anomaly_score"
        )
        result_sql = model.deploySQL(return_score=True)

        assert result_sql == expected_sql

    def test_drop(self, iforest_data_vd):
        current_cursor().execute("DROP MODEL IF EXISTS iforest_model_test_drop")
        model_test = IsolationForest(
            "iforest_model_test_drop",
        )
        model_test.fit(
            iforest_data_vd,
            ["Gender", '"owned cars"', "cost", "income", "TransPortation"],
        )

        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'iforest_model_test_drop'"
        )
        assert current_cursor().fetchone()[0] == "iforest_model_test_drop"

        model_test.drop()
        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'iforest_model_test_drop'"
        )
        assert current_cursor().fetchone() is None

    def test_get_vertica_attributes(self, model):
        m_att = model.get_vertica_attributes()

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
        assert m_att["#_of_rows"] == [1, 1, 1, 1, 5]

        m_att_details = model.get_vertica_attributes(attr_name="details")

        assert m_att_details["predictor"] == [
            "gender",
            "owned cars",
            "cost",
            "income",
            "transportation",
        ]
        assert m_att_details["type"] == [
            "char or varchar",
            "int",
            "char or varchar",
            "char or varchar",
            "int",
        ]

        assert model.get_vertica_attributes("tree_count")["tree_count"][0] == 100
        assert (
            model.get_vertica_attributes("rejected_row_count")["rejected_row_count"][0]
            == 0
        )
        assert (
            model.get_vertica_attributes("accepted_row_count")["accepted_row_count"][0]
            == 10
        )
        assert (
            "SELECT iforest('public.iforest_model_test',"
            in model.get_vertica_attributes("call_string")["call_string"][0]
        )

    def test_get_params(self, model):
        assert model.get_params() == {
            "n_estimators": 100,
            "sample": 0.632,
            "max_depth": 10,
            "nbins": 32,
            "col_sample_by_tree": 0.8,
        }

    def test_to_python(self, model):
        current_cursor().execute(
            "SELECT (APPLY_IFOREST('Male', 0, 'Cheap', 'Low', 1 USING PARAMETERS model_name = '{}', match_by_pos=True)).anomaly_score::float".format(
                model.model_name
            )
        )
        prediction = current_cursor().fetchone()[0]
        assert prediction == pytest.approx(
            model.to_python()([["Male", 0, "Cheap", "Low", 1]])[0],
            10e-2,
        )

    @pytest.mark.skip(reason="This needs to be investigated")
    def test_to_sql(self, model):
        current_cursor().execute(
            "SELECT (APPLY_IFOREST(* USING PARAMETERS model_name = '{}', match_by_pos=True)).anomaly_score::float, {}::float FROM (SELECT 'Male' AS \"Gender\", 0 AS \"owned cars\", 'Cheap' AS \"cost\", 'Low' AS \"income\", 1 AS Transportation) x".format(
                model.model_name, model.to_sql()
            )
        )
        prediction = current_cursor().fetchone()
        assert prediction[0] == pytest.approx(prediction[1], 10e-2)

    def test_to_memmodel(self, model, iforest_data_vd):
        mmodel = model.to_memmodel()
        res = mmodel.predict(
            [["Male", 0, "Cheap", "Low", 1], ["Female", 1, "Expensive", "Low", 1]]
        )
        res_py = model.to_python()(
            [["Male", 0, "Cheap", "Low", 1], ["Female", 1, "Expensive", "Low", 1]]
        )
        assert res[0] == pytest.approx(res_py[0])
        assert res[1] == pytest.approx(res_py[1])
        iforest_data_vd["prediction_sql"] = mmodel.predict_sql(
            ['"Gender"', '"owned cars"', '"cost"', '"income"', '"TransPortation"']
        )
        model.predict(iforest_data_vd, name="prediction_vertica_sql")
        # score = iforest_data_vd.score("prediction_sql", "prediction_vertica_sql", metric="r2") # Numeric Overflow
        # assert score == pytest.approx(1.0, 10e-1) # The score is not perfectly matching, we have to understand why

    def test_get_predicts(self, iforest_data_vd, model):
        iforest_data_copy = iforest_data_vd.copy()
        model.predict(
            iforest_data_copy,
            X=["Gender", '"owned cars"', "cost", "income", "TransPortation"],
            name="anomaly",
        )

        assert iforest_data_copy["anomaly"].mean() == pytest.approx(0.0, abs=1e-6)

        # TODO - contamination when v12.0.1 is in place

    def test_get_decision_function(self, iforest_data_vd, model):
        iforest_data_copy = iforest_data_vd.copy()
        model.decision_function(
            iforest_data_copy,
            X=["Gender", '"owned cars"', "cost", "income", "TransPortation"],
            name="anomaly_score",
        )

        assert iforest_data_copy["anomaly_score"].mean() == pytest.approx(
            0.516816506833607, abs=1e-6
        )

    def test_set_params(self, model):
        model.set_params({"n_estimators": 666})

        assert model.get_params()["n_estimators"] == 666

    def test_model_from_vDF(self, iforest_data_vd):
        current_cursor().execute("DROP MODEL IF EXISTS iForest_from_vdf")
        model_test = IsolationForest(
            "iForest_from_vdf",
        )
        model_test.fit(iforest_data_vd, ["gender"])

        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'iForest_from_vdf'"
        )
        assert current_cursor().fetchone()[0] == "iForest_from_vdf"

        model_test.drop()

    def test_to_graphviz(self, model):
        gvz_tree_0 = model.to_graphviz(
            tree_id=0,
            classes_color=["red", "blue", "green"],
            round_pred=4,
            percent=True,
            vertical=False,
            node_style={"shape": "box", "style": "filled"},
            arrow_style={"color": "blue"},
            leaf_style={"shape": "circle", "style": "filled"},
        )
        assert 'digraph Tree{\ngraph [rankdir = "LR"];\n0' in gvz_tree_0
        assert "0 -> 1" in gvz_tree_0

    def test_get_tree(self, model):
        tree_0 = model.get_tree()

        assert tree_0["leaf_path_length"] == [
            None,
            "1.000000",
            None,
            "2.000000",
            None,
            "3.000000",
            "3.000000",
        ]

    def test_plot_tree(self, model):
        result = model.plot_tree()
        assert model.to_graphviz() == result.source.strip()

    def test_optional_name(self):
        model = IsolationForest()
        assert model.model_name is not None
