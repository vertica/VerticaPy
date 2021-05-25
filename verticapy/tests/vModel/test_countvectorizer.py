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
from verticapy.learn.preprocessing import CountVectorizer
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
    try:
        create_verticapy_schema(base.cursor)
    except:
        pass
    model_class = CountVectorizer("model_test", cursor=base.cursor)
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

    def test_deploySQL(self, model):
        expected_sql = 'SELECT * FROM (SELECT token, cnt / SUM(cnt) OVER () AS df, cnt, rnk FROM (SELECT token, COUNT(*) AS cnt, RANK() OVER (ORDER BY COUNT(*) DESC) AS rnk FROM model_test GROUP BY 1) VERTICAPY_SUBTABLE) VERTICAPY_SUBTABLE WHERE (df BETWEEN 0.0 AND 1.0)'
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_drop(self, model, titanic_vd):
        model_test = CountVectorizer("model_test_drop", cursor=model.cursor)
        model_test.drop()
        model_test.fit(titanic_vd, ["name"])
        model_test.cursor.execute(
            "SELECT model_name FROM verticapy.models WHERE model_name IN ('model_test_drop', '\"model_test_drop\"')"
        )
        assert model_test.cursor.fetchone()[0] in ("model_test_drop", '"model_test_drop"')
        model_test.drop()
        model_test.cursor.execute(
            "SELECT model_name FROM models WHERE model_name LIKE 'model_test_drop'"
        )
        assert model_test.cursor.fetchone() is None

    def test_get_attr(self, model):
        assert sorted(model.vocabulary_)[0:3] == ['a', 'aaron', 'abbing',]
        assert model.stop_words_ == []

    def test_get_params(self, model):
        assert model.get_params() == {'ignore_special': True,
                                      'lowercase': True,
                                      'max_df': 1.0,
                                      'max_features': -1,
                                      'max_text_size': 2000,
                                      'min_df': 0.0}

    def test_get_transform(self, model):
        result = model.transform().sort(["rnk"])
        assert result["token"][0] == "mr"
        assert result["df"][0] == pytest.approx(0.14816310052482842)
        assert result["cnt"][0] == pytest.approx(734)
        assert result["rnk"][0] == pytest.approx(1)
        assert result.shape() == (1841, 4)

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
        model.set_params({"lowercase": False})
        assert model.get_params()["lowercase"] == False
        model.set_params({"lowercase": True})
        assert model.get_params()["lowercase"] == True

    def test_model_from_vDF(self, base, titanic_vd):
        model_class = CountVectorizer("model_test_vdf", cursor=base.cursor)
        model_class.drop()
        model_class.fit(titanic_vd, ["name"])
        assert model_class.transform().shape() == (1841, 4)
        model_class.drop()
