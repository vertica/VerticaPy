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
from verticapy.learn.preprocessing import OneHotEncoder
from verticapy import drop, set_option, vertica_conn

set_option("print_info", False)


@pytest.fixture(scope="module")
def titanic_vd(base):
    from verticapy.datasets import load_titanic

    titanic = load_titanic(cursor=base.cursor)
    yield titanic
    with warnings.catch_warnings(record=True) as w:
        drop(name="public.titanic", cursor=base.cursor)


@pytest.fixture(scope="module")
def model(base, titanic_vd):
    base.cursor.execute("DROP MODEL IF EXISTS ohe_model_test")
    model_class = OneHotEncoder("ohe_model_test", cursor=base.cursor, drop_first=False)
    model_class.fit("public.titanic", ["pclass", "sex", "embarked"])
    yield model_class
    model_class.drop()


class TestOneHotEncoder:
    def test_repr(self, model):
        assert "one_hot_encoder_fit" in model.__repr__()
        model_repr = OneHotEncoder("model_repr")
        model_repr.drop()
        assert model_repr.__repr__() == "<OneHotEncoder>"

    def test_deploySQL(self, model):
        expected_sql = "APPLY_ONE_HOT_ENCODER(\"pclass\", \"sex\", \"embarked\" USING PARAMETERS model_name = 'ohe_model_test', match_by_pos = 'true', drop_first = False, ignore_null = True, separator = '_', column_naming = 'indices')"
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_drop(self, base):
        base.cursor.execute("DROP MODEL IF EXISTS ohe_model_test_drop")
        model_test = OneHotEncoder("ohe_model_test_drop", cursor=base.cursor)
        model_test.fit("public.titanic", ["pclass", "embarked"])

        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'ohe_model_test_drop'"
        )
        assert base.cursor.fetchone()[0] == "ohe_model_test_drop"

        model_test.drop()
        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'ohe_model_test_drop'"
        )
        assert base.cursor.fetchone() is None

    def test_get_attr(self, model):
        m_att = model.get_attr()

        assert m_att["attr_name"] == [
            "call_string",
            "integer_categories",
            "varchar_categories",
        ]
        assert m_att["attr_fields"] == [
            "call_string",
            "category_name, category_level, category_level_index",
            "category_name, category_level, category_level_index",
        ]
        assert m_att["#_of_rows"] == [1, 3, 6]

        m_att_details = model.get_attr(attr_name="integer_categories")

        assert m_att_details["category_name"] == [
            "pclass",
            "pclass",
            "pclass",
        ]
        assert m_att_details["category_level"][0] == pytest.approx(1, abs=1e-6)
        assert m_att_details["category_level"][1] == pytest.approx(2, abs=1e-6)
        assert m_att_details["category_level"][2] == pytest.approx(3, abs=1e-6)
        assert m_att_details["category_level_index"][0] == pytest.approx(0, abs=1e-6)
        assert m_att_details["category_level_index"][1] == pytest.approx(1, abs=1e-6)
        assert m_att_details["category_level_index"][2] == pytest.approx(2)

    def test_get_params(self, model):
        assert model.get_params() == {
            "column_naming": "indices",
            "drop_first": False,
            "extra_levels": {},
            "ignore_null": True,
            "null_column_name": "null",
            "separator": "_",
        }

    def test_to_sklearn(self, model):
        md = model.to_sklearn()
        model.cursor.execute(
            "SELECT pclass_0, pclass_1, pclass_2, sex_0, sex_1, embarked_0, embarked_1, embarked_2 FROM (SELECT APPLY_ONE_HOT_ENCODER(pclass, sex, embarked USING PARAMETERS model_name = '{}', match_by_pos=True, drop_first=False) FROM (SELECT 1 AS pclass, 'female' AS sex, 'S' AS embarked) x) x".format(
                model.name
            )
        )
        prediction = model.cursor.fetchone()
        assert prediction == pytest.approx(
            md.transform([[1, "female", "S"]]).toarray()[0]
        )

    def test_to_sql(self, model):
        model.cursor.execute(
            "SELECT pclass_0, pclass_1, pclass_2, sex_0, sex_1, embarked_0, embarked_1, embarked_2 FROM (SELECT APPLY_ONE_HOT_ENCODER(pclass, sex, embarked USING PARAMETERS model_name = '{}', match_by_pos=True, drop_first=False) FROM (SELECT 1 AS pclass, 'female' AS sex, 'S' AS embarked) x) x".format(
                model.name
            )
        )
        prediction = [float(elem) for elem in model.cursor.fetchone()]
        model.cursor.execute(
            "SELECT pclass_0, pclass_1, pclass_2, sex_0, sex_1, embarked_0, embarked_1, embarked_2 FROM (SELECT {} FROM (SELECT 1 AS pclass, 'female' AS sex, 'S' AS embarked) x) x".format(
                model.to_sql()
            )
        )
        prediction2 = [float(elem) for elem in model.cursor.fetchone()]
        assert prediction == pytest.approx(prediction2)

    def test_get_transform(self, titanic_vd, model):
        titanic_trans = model.transform(titanic_vd, X=["pclass", "sex", "embarked"])
        assert titanic_trans["pclass_1"].mean() == pytest.approx(
            0.209886547811994, abs=1e-6
        )
        assert titanic_trans["pclass_2"].mean() == pytest.approx(
            0.537277147487844, abs=1e-6
        )
        assert titanic_trans["embarked_1"].mean() == pytest.approx(
            0.086038961038961, abs=1e-6
        )

    @pytest.mark.skip(reason="Vertica OHE has no inverse transform from now")
    def test_get_inverse_transform(self, titanic_vd, model):
        pass

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
        model.set_params({"ignore_null": False})
        assert model.get_params()["ignore_null"] == False

    def test_model_from_vDF(self, base, titanic_vd):
        base.cursor.execute("DROP MODEL IF EXISTS ohe_vDF")
        model_test = OneHotEncoder("ohe_vDF", cursor=base.cursor, drop_first=False)
        model_test.fit(titanic_vd, ["pclass", "embarked"])
        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'ohe_vDF'"
        )
        assert base.cursor.fetchone()[0] == "ohe_vDF"
        model_test.drop()
