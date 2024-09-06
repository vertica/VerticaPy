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
from verticapy import drop, set_option
from verticapy.connection import current_cursor
from verticapy.datasets import load_titanic
from verticapy.learn.preprocessing import OneHotEncoder

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
    model_class = OneHotEncoder("ohe_model_test", drop_first=False)
    model_class.drop()
    model_class.fit("public.titanic", ["pclass", "sex", "embarked"])
    yield model_class
    model_class.drop()


class TestOneHotEncoder:
    def test_repr(self, model):
        assert model.__repr__() == "<OneHotEncoder>"

    def test_deploySQL(self, model):
        expected_sql = "APPLY_ONE_HOT_ENCODER(\"pclass\", \"sex\", \"embarked\" USING PARAMETERS model_name = 'ohe_model_test', match_by_pos = 'true', drop_first = 'false', ignore_null = 'true', separator = '_', column_naming = 'indices')"
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_drop(self):
        current_cursor().execute("DROP MODEL IF EXISTS ohe_model_test_drop")
        model_test = OneHotEncoder(
            "ohe_model_test_drop",
        )
        model_test.fit("public.titanic", ["pclass", "embarked"])

        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'ohe_model_test_drop'"
        )
        assert current_cursor().fetchone()[0] == "ohe_model_test_drop"

        model_test.drop()
        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'ohe_model_test_drop'"
        )
        assert current_cursor().fetchone() is None

    def test_get_vertica_attributes(self, model):
        m_att = model.get_vertica_attributes()

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

        m_att_details = model.get_vertica_attributes(attr_name="integer_categories")

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

    def test_to_sql(self, model):
        current_cursor().execute(
            "SELECT pclass_1, pclass_2, sex_1, embarked_1, embarked_2 FROM (SELECT APPLY_ONE_HOT_ENCODER(pclass, sex, embarked USING PARAMETERS model_name = '{}', match_by_pos=True, drop_first=True) FROM (SELECT 1 AS pclass, 'female' AS sex, 'S' AS embarked) x) x".format(
                model.model_name
            )
        )
        prediction = [float(elem) for elem in current_cursor().fetchone()]
        current_cursor().execute(
            "SELECT pclass_1, pclass_2, sex_1, embarked_1, embarked_2 FROM (SELECT {} FROM (SELECT 1 AS pclass, 'female' AS sex, 'S' AS embarked) x) x".format(
                ", ".join([", ".join(elem) for elem in model.to_sql()])
            )
        )
        prediction2 = [float(elem) for elem in current_cursor().fetchone()]
        assert prediction == pytest.approx(prediction2)

    def test_to_memmodel(self, model):
        current_cursor().execute(
            "SELECT pclass_0, pclass_1, pclass_2, sex_0, sex_1, embarked_0, embarked_1, embarked_2 FROM (SELECT APPLY_ONE_HOT_ENCODER(pclass, sex, embarked USING PARAMETERS model_name = '{}', match_by_pos=True, drop_first=False) FROM (SELECT 1 AS pclass, 'female' AS sex, 'S' AS embarked) x) x".format(
                model.model_name
            )
        )
        prediction = [float(elem) for elem in current_cursor().fetchone()]
        current_cursor().execute(
            "SELECT pclass_0, pclass_1, pclass_2, sex_0, sex_1, embarked_0, embarked_1, embarked_2 FROM (SELECT {} FROM (SELECT 1 AS pclass, 'female' AS sex, 'S' AS embarked) x) x".format(
                ", ".join(
                    [
                        ", ".join(elem)
                        for elem in model.to_memmodel().transform_sql(
                            ["pclass", "sex", "embarked"]
                        )
                    ]
                )
            )
        )
        prediction2 = [float(elem) for elem in current_cursor().fetchone()]
        assert prediction == pytest.approx(prediction2)
        prediction3 = model.to_memmodel().transform([[1, "female", "S"]])
        assert prediction[0] == pytest.approx(prediction3[0][0])
        assert prediction[1] == pytest.approx(prediction3[0][1])
        assert prediction[2] == pytest.approx(prediction3[0][2])
        assert prediction[3] == pytest.approx(prediction3[0][3])
        assert prediction[4] == pytest.approx(prediction3[0][4])
        assert prediction[5] == pytest.approx(prediction3[0][5])
        assert prediction[6] == pytest.approx(prediction3[0][6])
        assert prediction[7] == pytest.approx(prediction3[0][7])

    def test_to_python(self, model):
        current_cursor().execute(
            "SELECT pclass_0, pclass_1, pclass_2, sex_0, sex_1, embarked_0, embarked_1, embarked_2, 0 FROM (SELECT APPLY_ONE_HOT_ENCODER(pclass, sex, embarked USING PARAMETERS model_name = '{}', match_by_pos=True, drop_first=False) FROM (SELECT 1 AS pclass, 'female' AS sex, 'S' AS embarked) x) x".format(
                model.model_name
            )
        )
        prediction = [int(elem) for elem in current_cursor().fetchone()]
        prediction2 = model.to_python()([[1, "female", "S"]])[0]
        assert len(prediction) == len(prediction2)
        assert prediction[0] == prediction2[0]
        assert prediction[1] == prediction2[1]
        assert prediction[2] == prediction2[2]
        assert prediction[3] == prediction2[3]
        assert prediction[4] == prediction2[4]
        assert prediction[5] == prediction2[5]
        assert prediction[6] == prediction2[6]
        assert prediction[7] == prediction2[7]

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

    def test_set_params(self, model):
        model.set_params({"ignore_null": False})
        assert not model.get_params()["ignore_null"]

    def test_model_from_vDF(self, titanic_vd):
        current_cursor().execute("DROP MODEL IF EXISTS ohe_vDF")
        model_test = OneHotEncoder("ohe_vDF", drop_first=False)
        model_test.fit(titanic_vd, ["pclass", "embarked"])
        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'ohe_vDF'"
        )
        assert current_cursor().fetchone()[0] == "ohe_vDF"
        model_test.drop()

    def test_optional_name(self):
        model = OneHotEncoder()
        assert model.model_name is not None
