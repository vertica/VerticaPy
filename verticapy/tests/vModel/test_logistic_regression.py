# (c) Copyright [2018-2020] Micro Focus or one of its affiliates.
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

import pytest
from verticapy.learn.linear_model import LogisticRegression


@pytest.fixture(scope="module")
def model(base):
    from verticapy.learn.datasets import load_titanic
    from verticapy import drop_table

    titanic = load_titanic(cursor=base.cursor)

    base.cursor.execute("DROP MODEL IF EXISTS logreg_model_test")
    model_class = LogisticRegression("logreg_model_test", cursor=base.cursor)
    model_class.fit("public.titanic", ["age", "fare"], "survived")
    yield model_class
    model_class.drop()
    drop_table(name = "public.titanic", cursor = base.cursor)


class TestLogisticRegression:
    def test_deploySQL(self, model):
        expected_sql = "PREDICT_LOGISTIC_REG(\"age\", \"fare\" USING PARAMETERS model_name = 'logreg_model_test', type = 'probability', match_by_pos = 'true')"
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_drop(self, base):
        base.cursor.execute("DROP MODEL IF EXISTS logreg_model_test_drop")
        model_test = LogisticRegression("logreg_model_test_drop", cursor=base.cursor)
        model_test.fit("public.titanic", ["age", "fare"], "survived")

        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'logreg_model_test_drop'"
        )
        assert base.cursor.fetchone()[0] == "logreg_model_test_drop"

        model_test.drop()
        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'logreg_model_test_drop'"
        )
        assert base.cursor.fetchone() is None

    @pytest.mark.skip(reason="test not implemented")
    def test_features_importance(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_get_model_attribute(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_get_model_fun(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_get_params(self):
        pass
