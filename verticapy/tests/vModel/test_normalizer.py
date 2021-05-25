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
from verticapy.learn.preprocessing import Normalizer, StandardScaler, RobustScaler, MinMaxScaler
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
    base.cursor.execute("DROP MODEL IF EXISTS norm_model_test")
    model_class = Normalizer("norm_model_test", cursor=base.cursor)
    model_class.fit("public.winequality", ["citric_acid", "residual_sugar", "alcohol"])
    yield model_class
    model_class.drop()


class TestNormalizer:
    def test_repr(self, model):
        assert "column_name  |  avg   |std_dev" in model.__repr__()
        model_repr = Normalizer("model_repr")
        model_repr.drop()
        assert model_repr.__repr__() == "<Normalizer>"

    def test_Normalizer_subclasses(self):
        result = StandardScaler("model_test")
        assert result.parameters["method"] == "zscore"
        result = RobustScaler("model_test")
        assert result.parameters["method"] == "robust_zscore"
        result = MinMaxScaler("model_test")
        assert result.parameters["method"] == "minmax"

    def test_deploySQL(self, model):
        expected_sql = 'APPLY_NORMALIZE("citric_acid", "residual_sugar", "alcohol" USING PARAMETERS model_name = \'norm_model_test\', match_by_pos = \'true\')'
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_deployInverseSQL(self, model):
        expected_sql = 'REVERSE_NORMALIZE("citric_acid", "residual_sugar", "alcohol" USING PARAMETERS model_name = \'norm_model_test\', match_by_pos = \'true\')'
        result_sql = model.deployInverseSQL()

        assert result_sql == expected_sql

    def test_drop(self, base):
        base.cursor.execute("DROP MODEL IF EXISTS norm_model_test_drop")
        model_test = Normalizer("norm_model_test_drop", cursor=base.cursor)
        model_test.fit("public.winequality", ["alcohol", "quality"])

        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'norm_model_test_drop'"
        )
        assert base.cursor.fetchone()[0] == "norm_model_test_drop"

        model_test.drop()
        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'norm_model_test_drop'"
        )
        assert base.cursor.fetchone() is None

    def test_get_attr(self, model):
        m_att = model.get_attr()

        assert m_att["attr_name"] == [
            "details",
        ]
        assert m_att["attr_fields"] == [
            "column_name, avg, std_dev",
        ]
        assert m_att["#_of_rows"] == [3]

        m_att_details = model.get_attr(attr_name="details")

        assert m_att_details["column_name"] == [
            "citric_acid",
            "residual_sugar",
            "alcohol",
        ]
        assert m_att_details["avg"][0] == pytest.approx(0.318633215330152, abs=1e-6)
        assert m_att_details["avg"][1] == pytest.approx(5.44323533938741, abs=1e-6)
        assert m_att_details["avg"][2] == pytest.approx(10.4918008311528, abs=1e-6)
        assert m_att_details["std_dev"][0] == pytest.approx(0.145317864897592, abs=1e-6)
        assert m_att_details["std_dev"][1] == pytest.approx(4.75780374314742, abs=1e-6)
        assert m_att_details["std_dev"][2] == pytest.approx(1.192711748871)

    def test_get_params(self, model):
        assert model.get_params() == {"method": "zscore"}

    def test_to_sklearn(self, model):
        # Zscore
        md = model.to_sklearn()
        model.cursor.execute(
            "SELECT APPLY_NORMALIZE(citric_acid, residual_sugar, alcohol USING PARAMETERS model_name = '{}', match_by_pos=True) FROM (SELECT 3.0 AS citric_acid, 11.0 AS residual_sugar, 93. AS alcohol) x".format(
                model.name
            )
        )
        prediction = model.cursor.fetchone()[0]
        assert prediction == pytest.approx(md.transform([[3.0, 11.0, 93.0]])[0][0])
        # Minmax
        model2 = Normalizer("norm_model_test2", cursor=model.cursor, method="minmax")
        model2.drop()
        model2.fit("public.winequality", ["citric_acid", "residual_sugar", "alcohol"])
        md = model2.to_sklearn()
        model2.cursor.execute(
            "SELECT APPLY_NORMALIZE(citric_acid, residual_sugar, alcohol USING PARAMETERS model_name = '{}', match_by_pos=True) FROM (SELECT 3.0 AS citric_acid, 11.0 AS residual_sugar, 93. AS alcohol) x".format(
                model2.name
            )
        )
        prediction = model2.cursor.fetchone()[0]
        model2.drop()
        assert prediction == pytest.approx(md.transform([[3.0, 11.0, 93.0]])[0][0])
        # Robust Zscore
        model3 = Normalizer(
            "norm_model_test2", cursor=model.cursor, method="robust_zscore"
        )
        model3.drop()
        model3.fit("public.winequality", ["citric_acid", "residual_sugar", "alcohol"])
        md = model3.to_sklearn()
        model3.cursor.execute(
            "SELECT APPLY_NORMALIZE(citric_acid, residual_sugar, alcohol USING PARAMETERS model_name = '{}', match_by_pos=True) FROM (SELECT 3.0 AS citric_acid, 11.0 AS residual_sugar, 93. AS alcohol) x".format(
                model3.name
            )
        )
        prediction = model3.cursor.fetchone()[0]
        model3.drop()
        assert prediction == pytest.approx(md.transform([[3.0, 11.0, 93.0]])[0][0])

    def test_to_python(self, model):
        # Zscore
        model.cursor.execute(
            "SELECT APPLY_NORMALIZE(citric_acid, residual_sugar, alcohol USING PARAMETERS model_name = '{}', match_by_pos=True) FROM (SELECT 3.0 AS citric_acid, 11.0 AS residual_sugar, 93. AS alcohol) x".format(
                model.name
            )
        )
        prediction = model.cursor.fetchone()[0]
        assert prediction == pytest.approx(model.to_python(return_str=False)([[3.0, 11.0, 93.0]])[0][0])
        # Minmax
        model2 = Normalizer("norm_model_test2", cursor=model.cursor, method="minmax")
        model2.drop()
        model2.fit("public.winequality", ["citric_acid", "residual_sugar", "alcohol"])
        model2.cursor.execute(
            "SELECT APPLY_NORMALIZE(citric_acid, residual_sugar, alcohol USING PARAMETERS model_name = '{}', match_by_pos=True) FROM (SELECT 3.0 AS citric_acid, 11.0 AS residual_sugar, 93. AS alcohol) x".format(
                model2.name
            )
        )
        prediction = model2.cursor.fetchone()[0]
        assert prediction == pytest.approx(model2.to_python(return_str=False)([[3.0, 11.0, 93.0]])[0][0])
        model2.drop()
        # Robust Zscore
        model3 = Normalizer(
            "norm_model_test2", cursor=model.cursor, method="robust_zscore"
        )
        model3.drop()
        model3.fit("public.winequality", ["citric_acid", "residual_sugar", "alcohol"])
        model3.cursor.execute(
            "SELECT APPLY_NORMALIZE(citric_acid, residual_sugar, alcohol USING PARAMETERS model_name = '{}', match_by_pos=True) FROM (SELECT 3.0 AS citric_acid, 11.0 AS residual_sugar, 93. AS alcohol) x".format(
                model3.name
            )
        )
        prediction = model3.cursor.fetchone()[0]
        assert prediction == pytest.approx(model3.to_python(return_str=False)([[3.0, 11.0, 93.0]])[0][0])
        model3.drop()

    def test_to_sql(self, model):
        # Zscore
        model.cursor.execute(
            "SELECT APPLY_NORMALIZE(citric_acid, residual_sugar, alcohol USING PARAMETERS model_name = '{}', match_by_pos=True) FROM (SELECT 3.0 AS citric_acid, 11.0 AS residual_sugar, 93. AS alcohol) x".format(
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
        # Minmax
        model2 = Normalizer("norm_model_test2", cursor=model.cursor, method="minmax")
        model2.drop()
        model2.fit("public.winequality", ["citric_acid", "residual_sugar", "alcohol"])
        model2.cursor.execute(
            "SELECT APPLY_NORMALIZE(citric_acid, residual_sugar, alcohol USING PARAMETERS model_name = '{}', match_by_pos=True) FROM (SELECT 3.0 AS citric_acid, 11.0 AS residual_sugar, 93. AS alcohol) x".format(
                model2.name
            )
        )
        prediction = [float(elem) for elem in model2.cursor.fetchone()]
        model2.cursor.execute(
            "SELECT {} FROM (SELECT 3.0 AS citric_acid, 11.0 AS residual_sugar, 93. AS alcohol) x".format(
                model2.to_sql()
            )
        )
        prediction2 = [float(elem) for elem in model2.cursor.fetchone()]
        assert prediction == pytest.approx(prediction2)
        model2.drop()
        # Robust Zscore
        model3 = Normalizer(
            "norm_model_test2", cursor=model.cursor, method="robust_zscore"
        )
        model3.drop()
        model3.fit("public.winequality", ["citric_acid", "residual_sugar", "alcohol"])
        model3.cursor.execute(
            "SELECT APPLY_NORMALIZE(citric_acid, residual_sugar, alcohol USING PARAMETERS model_name = '{}', match_by_pos=True) FROM (SELECT 3.0 AS citric_acid, 11.0 AS residual_sugar, 93. AS alcohol) x".format(
                model3.name
            )
        )
        prediction = [float(elem) for elem in model3.cursor.fetchone()]
        model3.cursor.execute(
            "SELECT {} FROM (SELECT 3.0 AS citric_acid, 11.0 AS residual_sugar, 93. AS alcohol) x".format(
                model3.to_sql()
            )
        )
        prediction2 = [float(elem) for elem in model3.cursor.fetchone()]
        assert prediction == pytest.approx(prediction2)
        model3.drop()

    def test_get_transform(self, winequality_vd, model):
        # Zscore
        winequality_trans = model.transform(
            winequality_vd, X=["citric_acid", "residual_sugar", "alcohol"]
        )
        assert winequality_trans["citric_acid"].mean() == pytest.approx(0.0, abs=1e-6)
        assert winequality_trans["residual_sugar"].mean() == pytest.approx(
            0.0, abs=1e-6
        )
        assert winequality_trans["alcohol"].mean() == pytest.approx(0.0, abs=1e-6)
        # Minmax
        model2 = Normalizer("norm_model_test2", cursor=model.cursor, method="minmax")
        model2.drop()
        model2.fit("public.winequality", ["citric_acid", "residual_sugar", "alcohol"])
        winequality_trans = model2.transform(
            winequality_vd, X=["citric_acid", "residual_sugar", "alcohol"]
        )
        assert winequality_trans["citric_acid"].min() == pytest.approx(0.0, abs=1e-6)
        assert winequality_trans["residual_sugar"].max() == pytest.approx(1.0, abs=1e-6)
        assert winequality_trans["alcohol"].min() == pytest.approx(0.0, abs=1e-6)
        model2.drop()
        # Robust Zscore
        model3 = Normalizer(
            "norm_model_test2", cursor=model.cursor, method="robust_zscore"
        )
        model3.drop()
        model3.fit("public.winequality", ["citric_acid", "residual_sugar", "alcohol"])
        winequality_trans = model3.transform(
            winequality_vd, X=["citric_acid", "residual_sugar", "alcohol"]
        )
        assert winequality_trans["citric_acid"].median() == pytest.approx(0.0, abs=1e-6)
        assert winequality_trans["residual_sugar"].median() == pytest.approx(
            0.0, abs=1e-6
        )
        assert winequality_trans["alcohol"].median() == pytest.approx(0.0, abs=1e-6)
        model3.drop()

    def test_get_inverse_transform(self, winequality_vd, model):
        winequality_trans = model.inverse_transform(
            winequality_vd, X=["citric_acid", "residual_sugar", "alcohol"]
        )
        assert winequality_trans["citric_acid"].mean() == pytest.approx(
            0.364936313867385, abs=1e-6
        )
        assert winequality_trans["residual_sugar"].mean() == pytest.approx(
            31.3410808119571, abs=1e-6
        )
        assert winequality_trans["alcohol"].mean() == pytest.approx(
            23.0054949492833, abs=1e-6
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
        model.set_params({"method": "robust_zscore"})
        assert model.get_params()["method"] == "robust_zscore"
        model.set_params({"method": "zscore"})
        assert model.get_params()["method"] == "zscore"

    def test_model_from_vDF(self, base, winequality_vd):
        base.cursor.execute("DROP MODEL IF EXISTS norm_vDF")
        model_test = Normalizer("norm_vDF", cursor=base.cursor)
        model_test.fit(winequality_vd, ["alcohol", "quality"])
        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'norm_vDF'"
        )
        assert base.cursor.fetchone()[0] == "norm_vDF"
        model_test.drop()
