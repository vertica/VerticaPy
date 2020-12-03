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

import pytest, warnings
from verticapy import drop_table, set_option, str_sql
import verticapy.stats as st

set_option("print_info", False)


@pytest.fixture(scope="module")
def titanic_vd(base):
    from verticapy.learn.datasets import load_titanic

    titanic = load_titanic(cursor=base.cursor)
    yield titanic
    with warnings.catch_warnings(record=True) as w:
        drop_table(
            name="public.titanic", cursor=base.cursor,
        )


class TestStats:
    def test_abs(self, titanic_vd):
        assert st.abs(titanic_vd["age"]) == 'ABS("age")'

    def test_acos(self, titanic_vd):
        assert st.acos(titanic_vd["age"]) == 'ACOS("age")'

    def test_asin(self, titanic_vd):
        assert st.asin(titanic_vd["age"]) == 'ASIN("age")'

    def test_atan(self, titanic_vd):
        assert st.atan(titanic_vd["age"]) == 'ATAN("age")'

    def test_cbrt(self, titanic_vd):
        assert st.cbrt(titanic_vd["age"]) == 'CBRT("age")'

    def test_ceil(self, titanic_vd):
        assert st.ceil(titanic_vd["age"]) == 'CEIL("age")'

    def test_comb(self, titanic_vd):
        assert st.comb(3, 6) == "(3)! / ((6)! * (3 - 6)!)"

    def test_cos(self, titanic_vd):
        assert st.cos(titanic_vd["age"]) == 'COS("age")'

    def test_cosh(self, titanic_vd):
        assert st.cosh(titanic_vd["age"]) == 'COSH("age")'

    def test_cot(self, titanic_vd):
        assert st.cot(titanic_vd["age"]) == 'COT("age")'

    def test_degrees(self, titanic_vd):
        assert st.degrees(titanic_vd["age"]) == 'DEGREES("age")'

    def test_distance(self, titanic_vd):
        assert (
            st.distance(
                titanic_vd["age"],
                titanic_vd["fare"],
                titanic_vd["age"],
                titanic_vd["fare"],
            )
            == 'DISTANCE("age", "fare", "age", "fare", 6371.009)'
        )

    def test_exp(self, titanic_vd):
        assert st.exp(titanic_vd["age"]) == 'EXP("age")'

    def test_factorial(self, titanic_vd):
        assert st.factorial(titanic_vd["age"]) == '("age")!'

    def test_floor(self, titanic_vd):
        assert st.floor(titanic_vd["age"]) == 'FLOOR("age")'

    def test_gamma(self, titanic_vd):
        assert st.gamma(titanic_vd["age"]) == '("age - 1")!'

    def test_hash(self, titanic_vd):
        assert st.hash(titanic_vd["age"]) == 'HASH("age")'
        assert st.hash(titanic_vd["age"], titanic_vd["fare"]) == 'HASH("age", "fare")'

    def test_isfinite(self, titanic_vd):
        assert (
            st.isfinite(titanic_vd["age"])
            == '(("age") = ("age")) AND (ABS("age") < \'inf\'::float)'
        )

    def test_isinf(self, titanic_vd):
        assert st.isinf(titanic_vd["age"]) == "ABS(\"age\") = 'inf'::float"

    def test_isnan(self, titanic_vd):
        assert st.isnan(titanic_vd["age"]) == '(("age") != ("age"))'

    def test_lgamma(self, titanic_vd):
        assert st.lgamma(titanic_vd["age"]) == 'LN(("age" - 1)!)'

    def test_ln(self, titanic_vd):
        assert st.ln(titanic_vd["age"]) == 'LN("age")'

    def test_log(self, titanic_vd):
        assert st.log(titanic_vd["age"]) == 'LOG(10, "age")'

    def test_radians(self, titanic_vd):
        assert st.radians(titanic_vd["age"]) == 'RADIANS("age")'

    def test_random(self):
        assert st.random() == "RANDOM()"

    def test_randomint(self):
        assert st.randomint(10) == "RANDOMINT(10)"

    def test_round(self, titanic_vd):
        assert st.round(titanic_vd["age"], 3) == 'ROUND("age", 3)'

    def test_sign(self, titanic_vd):
        assert st.sign(titanic_vd["age"]) == 'SIGN("age")'

    def test_sin(self, titanic_vd):
        assert st.sin(titanic_vd["age"]) == 'SIN("age")'

    def test_sinh(self, titanic_vd):
        assert st.sinh(titanic_vd["age"]) == 'SINH("age")'

    def test_sqrt(self, titanic_vd):
        assert st.sqrt(titanic_vd["age"]) == 'SQRT("age")'

    def test_tan(self, titanic_vd):
        assert st.tan(titanic_vd["age"]) == 'TAN("age")'

    def test_tanh(self, titanic_vd):
        assert st.tanh(titanic_vd["age"]) == 'TANH("age")'

    def test_trunc(self, titanic_vd):
        assert st.trunc(titanic_vd["age"], 3) == 'TRUNC("age", 3)'

    def test_constants(self):
        assert st.pi == str_sql("PI()")
        assert st.e == str_sql("EXP(1)")
        assert st.tau == str_sql("2 * PI()")
        assert st.inf == str_sql("'inf'::float")
        assert st.nan == str_sql("'nan'::float")
