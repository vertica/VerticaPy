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

import pytest, warnings
from verticapy import drop, set_option
from verticapy.connect import *
from verticapy.sql import *

set_option("print_info", False)

@pytest.fixture(scope="module")
def titanic_vd(base):
    from verticapy.datasets import load_titanic

    titanic = load_titanic(cursor=base.cursor)
    yield titanic
    with warnings.catch_warnings(record=True) as w:
        drop(name="public.titanic", cursor=base.cursor)

class TestSQL:
    def test_sql(self, base, titanic_vd):
        d = read_dsn("vp_test_config", os.path.dirname(verticapy.__file__) + "/tests/verticaPy_test_tmp.conf",)
        new_auto_connection(d, "VerticaDSN_test")
        change_auto_connection("VerticaDSN_test")
        result = sql("-limit 30", "SELECT * FROM titanic;")
        assert len(result["age"]) == 30
        result = sql("-vdf True", "SELECT * FROM titanic;")
        assert result.shape() == (1234, 14)
        result = sql("-vdf True", "DROP MODEL IF EXISTS model_test; SELECT LINEAR_REG('model_test', 'public.titanic', 'survived', 'age, fare'); SELECT PREDICT_LINEAR_REG(3.0, 4.0 USING PARAMETERS model_name='model_test', match_by_pos=True) AS predict;")
        assert result["predict"][0] == pytest.approx(0.395335892040411)
        result = sql("-limit 30", "DROP MODEL IF EXISTS model_test; SELECT 1 AS col;;")
        assert result["col"][0] == 1
