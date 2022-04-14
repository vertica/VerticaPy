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

# Pytest
import pytest

# Standard Python Modules
import warnings, os

# VerticaPy
from verticapy import drop, set_option
from verticapy.datasets import load_titanic
from verticapy.sql import sql

set_option("print_info", False)


@pytest.fixture(scope="module")
def titanic_vd():
    titanic = load_titanic()
    yield titanic
    drop(name="public.titanic")


class TestSQL:
    def test_sql(self, titanic_vd):
        result = sql("", "SELECT * FROM titanic;")
        assert result.shape() == (1234, 14)
        result = sql(
            "",
            """DROP MODEL IF EXISTS model_test;
               SELECT LINEAR_REG('model_test', 'public.titanic', 'survived', 'age, fare'); 
               SELECT PREDICT_LINEAR_REG(3.0, 4.0 
                      USING PARAMETERS model_name='model_test', 
                                       match_by_pos=True) AS predict;""",
        )
        assert result["predict"][0] == pytest.approx(0.395335892040411)
        result = sql("", "DROP MODEL IF EXISTS model_test; SELECT 1 AS col;;")
        assert result["col"][0] == 1
        result = sql(
            "-i {}/tests/sql/queries.sql".format(os.path.dirname(verticapy.__file__)),
            "",
        )
        assert result["predict"][0] == pytest.approx(0.395335892040411)
        result = sql("DROP MODEL IF EXISTS model_test; SELECT 1 AS col;;", "")
        assert result["col"][0] == 1
