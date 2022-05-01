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

# VerticaPy
from verticapy import (
    drop,
    set_option,
    create_verticapy_schema,
)
from verticapy.connect import current_cursor
from verticapy.datasets import load_titanic
from verticapy.learn.preprocessing import CountVectorizer

set_option("print_info", False)


@pytest.fixture(scope="module")
def titanic_vd():
    titanic = load_titanic()
    yield titanic
    drop(name="public.titanic",)


@pytest.fixture(scope="module")
def model(titanic_vd):
    create_verticapy_schema()
    model_class = CountVectorizer("model_test_countvectorizer",)
    model_class.drop()
    model_class.fit("public.titanic", ["name"])
    yield model_class
    model_class.drop()


class TestCountVectorizer:
    def test_repr(self, model):
        assert "Vocabulary" in model.__repr__()
        model_repr = CountVectorizer("model_repr")
        model_repr.drop()
        assert model_repr.__repr__() == "<CountVectorizer>"

    def test_get_attr(self, model):
        m_att = model.get_attr()
        assert m_att["attr_name"] == [
            "lowercase",
            "max_df",
            "min_df",
            "max_features",
            "ignore_special",
            "max_text_size",
            "vocabulary",
            "stop_words",
        ]
        m_att = model.get_attr("lowercase")
        assert m_att == model.parameters["lowercase"]
        m_att = model.get_attr("max_df")
        assert m_att == model.parameters["max_df"]
        m_att = model.get_attr("min_df")
        assert m_att == model.parameters["min_df"]
        m_att = model.get_attr("max_features")
        assert m_att == model.parameters["max_features"]
        m_att = model.get_attr("ignore_special")
        assert m_att == model.parameters["ignore_special"]
        m_att = model.get_attr("max_text_size")
        assert m_att == model.parameters["max_text_size"]
        m_att = model.get_attr("vocabulary")
        assert m_att == model.parameters["vocabulary"]
        m_att = model.get_attr("stop_words")
        assert m_att == model.parameters["stop_words"]

    def test_deploySQL(self, model):
        expected_sql = (
            "SELECT \n                    * \n                 FROM"
            " (SELECT \n                          token, \n        "
            "                  cnt / SUM(cnt) OVER () AS df, \n    "
            "                      cnt, \n                         "
            " rnk \n                 FROM (SELECT \n               "
            "           token, \n                          COUNT(*)"
            " AS cnt, \n                          RANK() OVER (ORDER"
            " BY COUNT(*) DESC) AS rnk \n                       FROM"
            " model_test_countvectorizer GROUP BY 1) VERTICAPY_SUBTABLE)"
            " VERTICAPY_SUBTABLE \n                       WHERE (df "
            "BETWEEN 0.0 AND 1.0)"
        )
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_drop(self, titanic_vd):
        model_test = CountVectorizer("model_test_drop")
        model_test.drop()
        model_test.fit(titanic_vd, ["name"])
        current_cursor().execute(
            "SELECT model_name FROM verticapy.models WHERE model_name IN ('model_test_drop', '\"model_test_drop\"')"
        )
        assert current_cursor().fetchone()[0] in (
            "model_test_drop",
            '"model_test_drop"',
        )
        model_test.drop()
        current_cursor().execute(
            "SELECT model_name FROM verticapy.models WHERE model_name IN ('model_test_drop', '\"model_test_drop\"')"
        )
        assert current_cursor().fetchone() is None

    def test_get_attr(self, model):
        assert sorted(model.vocabulary_)[0:3] == ["a", "aaron", "abbing"]
        assert model.stop_words_ == []

    def test_get_params(self, model):
        assert model.get_params() == {
            "ignore_special": True,
            "lowercase": True,
            "max_df": 1.0,
            "max_features": -1,
            "max_text_size": 2000,
            "min_df": 0.0,
        }

    def test_get_transform(self, model):
        result = model.transform().sort(["rnk"])
        assert result["token"][0] == "mr"
        assert result["df"][0] == pytest.approx(0.14816310052482842)
        assert result["cnt"][0] == pytest.approx(734)
        assert result["rnk"][0] == pytest.approx(1)
        assert result.shape() == (1841, 4)

    def test_set_params(self, model):
        model.set_params({"lowercase": False})
        assert model.get_params()["lowercase"] == False
        model.set_params({"lowercase": True})
        assert model.get_params()["lowercase"] == True

    def test_model_from_vDF(self, titanic_vd):
        model_class = CountVectorizer("model_test_vdf",)
        model_class.drop()
        model_class.fit(titanic_vd, ["name"])
        assert model_class.transform().shape() == (1841, 4)
        model_class.drop()
