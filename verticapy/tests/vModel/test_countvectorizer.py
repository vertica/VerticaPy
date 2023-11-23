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
from vertica_python.errors import QueryError

# VerticaPy
from verticapy import (
    drop,
    set_option,
)
from verticapy.connection import current_cursor
from verticapy.datasets import load_titanic
from verticapy.learn.preprocessing import CountVectorizer
from verticapy._utils._sql._format import clean_query

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
    model_class = CountVectorizer(
        "model_test_countvectorizer",
    )
    model_class.drop()
    model_class.fit("public.titanic", ["name"])
    yield model_class
    model_class.drop()


class TestCountVectorizer:
    def test_repr(self, model):
        assert model.__repr__() == "<CountVectorizer>"

    def test_get_vertica_attributes(self, model):
        m_att = model.get_attributes()
        assert m_att == ["stop_words_", "vocabulary_", "n_errors_"]

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

        assert result_sql == clean_query(expected_sql)

    def test_get_attributes(self, model):
        assert sorted(model.vocabulary_)[0:3] == ["a", "aaron", "abbing"]
        assert len(model.stop_words_) == 0

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
        assert not model.get_params()["lowercase"]
        model.set_params({"lowercase": True})
        assert model.get_params()["lowercase"]

    def test_model_from_vDF(self, titanic_vd):
        model_class = CountVectorizer(
            "model_test_vdf",
        )
        model_class.drop()
        model_class.fit(titanic_vd, ["name"])
        assert model_class.transform().shape() == (1841, 4)
        model_class.drop()

    def test_AutoDataPrep_overwrite_model(self, titanic_vd):
        model = CountVectorizer("test_overwrite_model")
        model.drop()
        model.fit(titanic_vd, ["name"])

        # overwrite_model is false by default
        with pytest.raises(QueryError) as exception_info:
            model.fit(titanic_vd)
        assert "throwErrorFunctionDoesNotMatchCandidates" in str(exception_info.value)

        # overwriting the model when overwrite_model is specified true
        model = CountVectorizer("test_overwrite_model", overwrite_model=True)
        model.fit(titanic_vd, ["name"])

        # cleaning up
        model.drop()

    def test_optional_name(self):
        model = CountVectorizer()
        assert model.model_name is not None
