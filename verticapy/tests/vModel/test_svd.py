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
from verticapy.learn.decomposition import SVD
from verticapy import drop, set_option, vertica_conn

set_option("print_info", False)


@pytest.fixture(scope="module")
def winequality_vd(base):
    from verticapy.datasets import load_winequality

    winequality = load_winequality(cursor=base.cursor)
    yield winequality
    with warnings.catch_warnings(record=True) as w:
        drop(name="public.winequality", cursor=base.cursor)


@pytest.fixture(scope="module")
def model(base, winequality_vd):
    base.cursor.execute("DROP MODEL IF EXISTS SVD_model_test")
    model_class = SVD("SVD_model_test", cursor=base.cursor)
    model_class.fit("public.winequality", ["citric_acid", "residual_sugar", "alcohol"])
    yield model_class
    model_class.drop()


class TestSVD:
    def test_repr(self, model):
        assert "SVD" in model.__repr__()
        model_repr = SVD("model_repr")
        model_repr.drop()
        assert model_repr.__repr__() == "<SVD>"

    def test_deploySQL(self, model):
        expected_sql = 'APPLY_SVD("citric_acid", "residual_sugar", "alcohol" USING PARAMETERS model_name = \'SVD_model_test\', match_by_pos = \'true\', cutoff = 1)'
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_deployInverseSQL(self, model):
        expected_sql = 'APPLY_INVERSE_SVD("citric_acid", "residual_sugar", "alcohol" USING PARAMETERS model_name = \'SVD_model_test\', match_by_pos = \'true\')'
        result_sql = model.deployInverseSQL()

        assert result_sql == expected_sql

    def test_drop(self, base):
        base.cursor.execute("DROP MODEL IF EXISTS SVD_model_test_drop")
        model_test = SVD("SVD_model_test_drop", cursor=base.cursor)
        model_test.fit("public.winequality", ["alcohol", "quality"])

        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'SVD_model_test_drop'"
        )
        assert base.cursor.fetchone()[0] == "SVD_model_test_drop"

        model_test.drop()
        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'SVD_model_test_drop'"
        )
        assert base.cursor.fetchone() is None

    def test_get_attr(self, model):
        m_att = model.get_attr()

        assert m_att["attr_name"] == [
            "columns",
            "singular_values",
            "right_singular_vectors",
            "counters",
            "call_string",
        ]
        assert m_att["attr_fields"] == [
            "index, name",
            "index, value, explained_variance, accumulated_explained_variance",
            "index, vector1, vector2, vector3",
            "counter_name, counter_value",
            "call_string",
        ]
        assert m_att["#_of_rows"] == [3, 3, 3, 3, 1]

        m_att_details = model.get_attr(attr_name="singular_values")

        assert m_att_details["value"][0] == pytest.approx(968.964362586858, abs=1e-6)
        assert m_att_details["value"][1] == pytest.approx(354.585184720344, abs=1e-6)
        assert m_att_details["value"][2] == pytest.approx(11.7281921567471, abs=1e-6)

    def test_get_params(self, model):
        assert model.get_params() == {"method": "lapack", "n_components": 0}

    def test_to_sklearn(self, model):
        md = model.to_sklearn()
        model.cursor.execute(
            "SELECT APPLY_SVD(citric_acid, residual_sugar, alcohol USING PARAMETERS model_name = '{}', match_by_pos=True) FROM (SELECT 3.0 AS citric_acid, 11.0 AS residual_sugar, 93. AS alcohol) x".format(
                model.name
            )
        )
        prediction = model.cursor.fetchone()
        assert prediction == pytest.approx(md.transform([[3.0, 11.0, 93.0]])[0])

    def test_to_python(self, model):
        model.cursor.execute(
            "SELECT APPLY_SVD(citric_acid, residual_sugar, alcohol USING PARAMETERS model_name = '{}', match_by_pos=True) FROM (SELECT 3.0 AS citric_acid, 11.0 AS residual_sugar, 93. AS alcohol) x".format(
                model.name
            )
        )
        prediction = model.cursor.fetchone()
        assert prediction == pytest.approx(model.to_python(return_str=False)([[3.0, 11.0, 93.0]])[0])

    def test_to_sql(self, model):
        model.cursor.execute(
            "SELECT APPLY_SVD(citric_acid, residual_sugar, alcohol USING PARAMETERS model_name = '{}', match_by_pos=True) FROM (SELECT 3.0 AS citric_acid, 11.0 AS residual_sugar, 93. AS alcohol) x".format(
                model.name
            )
        )
        prediction = [float(elem) for elem in model.cursor.fetchone()]
        model.cursor.execute(
            "SELECT {} FROM (SELECT 3.0 AS citric_acid, 11.0 AS residual_sugar, 93. AS alcohol) x".format(
                model.to_sql()
            )
        )
        prediction2 = [float(elem) for elem in model.cursor.fetchone()]
        assert prediction == pytest.approx(prediction2)

    def test_get_transform(self, winequality_vd, model):
        winequality_trans = model.transform(
            winequality_vd, X=["citric_acid", "residual_sugar", "alcohol"]
        )
        assert winequality_trans["col1"].mean() == pytest.approx(
            0.0121807874344058, abs=1e-6
        )
        assert winequality_trans["col2"].mean() == pytest.approx(
            -0.00200082024084619, abs=1e-6
        )
        assert winequality_trans["col3"].mean() == pytest.approx(
            0.000194341623203586, abs=1e-6
        )

    def test_get_inverse_transform(self, winequality_vd, model):
        winequality_trans = model.transform(
            winequality_vd, X=["citric_acid", "residual_sugar", "alcohol"]
        )
        winequality_trans = model.inverse_transform(
            winequality_trans, X=["col1", "col2", "col3"]
        )
        assert winequality_trans["citric_acid"].mean() == pytest.approx(
            winequality_vd["citric_acid"].mean(), abs=1e-6
        )
        assert winequality_trans["residual_sugar"].mean() == pytest.approx(
            winequality_vd["residual_sugar"].mean(), abs=1e-6
        )
        assert winequality_trans["alcohol"].mean() == pytest.approx(
            winequality_vd["alcohol"].mean(), abs=1e-6
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

    def test_svd_score(self, model):
        result = model.score()
        assert result["Score"][0] == pytest.approx(0.0, abs=1e-6)
        assert result["Score"][1] == pytest.approx(0.0, abs=1e-6)
        assert result["Score"][2] == pytest.approx(0.0, abs=1e-6)

    def test_set_params(self, model):
        model.set_params({"n_components": 3})
        assert model.get_params()["n_components"] == 3

    def test_model_from_vDF(self, base, winequality_vd):
        base.cursor.execute("DROP MODEL IF EXISTS SVD_vDF")
        model_test = SVD("SVD_vDF", cursor=base.cursor)
        model_test.fit(winequality_vd, ["alcohol", "quality"])
        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'SVD_vDF'"
        )
        assert base.cursor.fetchone()[0] == "SVD_vDF"
        model_test.drop()
