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


@pytest.fixture(scope="module")
def amazon_vd(base):
    from verticapy.learn.datasets import load_amazon

    amazon = load_amazon(cursor=base.cursor)
    yield amazon
    with warnings.catch_warnings(record=True) as w:
        drop_table(
            name="public.amazon", cursor=base.cursor,
        )


class TestStats:
    def test_regexp_instr(self, titanic_vd):
        assert (
            str(st.regexp_instr(titanic_vd["name"], "([A-Za-z])+\\. "))
            == "REGEXP_INSTR(\"name\", '([A-Za-z])+\\. ', 1, 1, 0)"
        )

    def test_regexp_like(self, titanic_vd):
        assert (
            str(st.regexp_like(titanic_vd["name"], "([A-Za-z])+\\. "))
            == "REGEXP_LIKE(\"name\", '([A-Za-z])+\\. ')"
        )

    def test_regexp_replace(self, titanic_vd):
        assert (
            str(st.regexp_replace(titanic_vd["name"], "([A-Za-z])+\\. ", "\\.\\."))
            == "REGEXP_REPLACE(\"name\", '([A-Za-z])+\\. ', '\\.\\.', 1, 1)"
        )

    def test_regexp_substr(self, titanic_vd):
        assert (
            str(st.regexp_substr(titanic_vd["name"], "([A-Za-z])+\\. "))
            == "REGEXP_SUBSTR(\"name\", '([A-Za-z])+\\. ', 1, 1)"
        )

    def test_regexp_count(self, titanic_vd):
        assert (
            str(st.regexp_count(titanic_vd["name"], "([A-Za-z])+\\. "))
            == "REGEXP_COUNT(\"name\", '([A-Za-z])+\\. ', 1)"
        )

    def test_regexp_ilike(self, titanic_vd):
        assert (
            str(st.regexp_ilike(titanic_vd["name"], "([A-Za-z])+\\. "))
            == "REGEXP_ILIKE(\"name\", '([A-Za-z])+\\. ')"
        )

    def test_length(self, titanic_vd):
        assert str(st.length(titanic_vd["name"])) == 'LENGTH("name")'

    def test_lower(self, titanic_vd):
        assert str(st.lower(titanic_vd["name"])) == 'LOWER("name")'

    def test_substr(self, titanic_vd):
        assert str(st.substr(titanic_vd["name"], 1, 2)) == 'SUBSTR("name", 1, 2)'

    def test_upper(self, titanic_vd):
        assert str(st.upper(titanic_vd["name"])) == 'UPPER("name")'

    def test_apply(self, titanic_vd):
        assert str(st.apply("avg", titanic_vd["age"])) == 'AVG("age")'
        assert str(st.apply("dense_rank")) == "DENSE_RANK()"

    def test_avg(self, titanic_vd):
        assert str(st.avg(titanic_vd["age"])) == 'AVG("age")'

    def test_bool_and(self, titanic_vd):
        assert str(st.bool_and(titanic_vd["age"])) == 'BOOL_AND("age")'

    def test_bool_or(self, titanic_vd):
        assert str(st.bool_or(titanic_vd["age"])) == 'BOOL_OR("age")'

    def test_bool_xor(self, titanic_vd):
        assert str(st.bool_xor(titanic_vd["age"])) == 'BOOL_XOR("age")'

    def test_conditional_change_event(self, titanic_vd):
        assert (
            str(st.conditional_change_event(titanic_vd["age"] > 5))
            == 'CONDITIONAL_CHANGE_EVENT(("age") > (5))'
        )

    def test_conditional_true_event(self, titanic_vd):
        assert (
            str(st.conditional_true_event(titanic_vd["age"] > 5))
            == 'CONDITIONAL_TRUE_EVENT(("age") > (5))'
        )

    def test_count(self, titanic_vd):
        assert str(st.count(titanic_vd["age"])) == 'COUNT("age")'

    def test_max(self, titanic_vd):
        assert str(st.max(titanic_vd["age"])) == 'MAX("age")'

    def test_median(self, titanic_vd):
        assert str(st.median(titanic_vd["age"])) == 'APPROXIMATE_MEDIAN("age")'

    def test_min(self, titanic_vd):
        assert str(st.min(titanic_vd["age"])) == 'MIN("age")'

    def test_nth_value(self, titanic_vd):
        assert str(st.nth_value(titanic_vd["age"], 2)) == 'NTH_VALUE("age", 2)'

    def test_lag(self, titanic_vd):
        assert str(st.lag(titanic_vd["age"])) == 'LAG("age", 1)'

    def test_lead(self, titanic_vd):
        assert str(st.lead(titanic_vd["age"])) == 'LEAD("age", 1)'

    def test_quantile(self, titanic_vd):
        assert (
            str(st.quantile(titanic_vd["age"], 0.3))
            == 'APPROXIMATE_PERCENTILE("age" USING PARAMETERS percentile = 0.3)'
        )

    def test_rank(self):
        assert str(st.rank()) == "RANK()"

    def test_row_number(self):
        assert str(st.row_number()) == "ROW_NUMBER()"

    def test_std(self, titanic_vd):
        assert str(st.std(titanic_vd["age"])) == 'STDDEV("age")'

    def test_sum(self, titanic_vd):
        assert str(st.sum(titanic_vd["age"])) == 'SUM("age")'

    def test_var(self, titanic_vd):
        assert str(st.var(titanic_vd["age"])) == 'VARIANCE("age")'

    def test_abs(self, titanic_vd):
        assert str(st.abs(titanic_vd["age"])) == 'ABS("age")'

    def test_acos(self, titanic_vd):
        assert str(st.acos(titanic_vd["age"])) == 'ACOS("age")'

    def test_asin(self, titanic_vd):
        assert str(st.asin(titanic_vd["age"])) == 'ASIN("age")'

    def test_atan(self, titanic_vd):
        assert str(st.atan(titanic_vd["age"])) == 'ATAN("age")'

    def test_case_when(self, titanic_vd):
        assert (
            str(st.case_when(titanic_vd["age"] > 5, 11, 1993))
            == 'CASE WHEN ("age") > (5) THEN 11 ELSE 1993 END'
        )

    def test_cbrt(self, titanic_vd):
        assert str(st.cbrt(titanic_vd["age"])) == 'CBRT("age")'

    def test_ceil(self, titanic_vd):
        assert str(st.ceil(titanic_vd["age"])) == 'CEIL("age")'

    def test_coalesce(self, titanic_vd):
        assert str(st.coalesce(titanic_vd["age"], 30)) == 'COALESCE("age", 30)'

    def test_comb(self, titanic_vd):
        assert str(st.comb(3, 6)) == "(3)! / ((6)! * (3 - 6)!)"

    def test_cos(self, titanic_vd):
        assert str(st.cos(titanic_vd["age"])) == 'COS("age")'

    def test_cosh(self, titanic_vd):
        assert str(st.cosh(titanic_vd["age"])) == 'COSH("age")'

    def test_cot(self, titanic_vd):
        assert str(st.cot(titanic_vd["age"])) == 'COT("age")'

    def test_date(self, amazon_vd):
        assert str(st.date(amazon_vd["date"])) == 'DATE("date")'

    def test_day(self, amazon_vd):
        assert str(st.day(amazon_vd["date"])) == 'DAY("date")'

    def test_dayofweek(self, amazon_vd):
        assert str(st.dayofweek(amazon_vd["date"])) == 'DAYOFWEEK("date")'

    def test_dayofyear(self, amazon_vd):
        assert str(st.dayofyear(amazon_vd["date"])) == 'DAYOFYEAR("date")'

    def test_degrees(self, titanic_vd):
        assert str(st.degrees(titanic_vd["age"])) == 'DEGREES("age")'

    def test_decode(self, titanic_vd):
        assert (
            str(st.decode(titanic_vd["pclass"], 1, 0, 1)) == 'DECODE("pclass", 1, 0, 1)'
        )

    def test_distance(self, titanic_vd):
        assert (
            str(
                st.distance(
                    titanic_vd["age"],
                    titanic_vd["fare"],
                    titanic_vd["age"],
                    titanic_vd["fare"],
                )
            )
            == 'DISTANCE("age", "fare", "age", "fare", 6371.009)'
        )

    def test_exp(self, titanic_vd):
        assert str(st.exp(titanic_vd["age"])) == 'EXP("age")'

    def test_extract(self, amazon_vd):
        assert (
            str(st.extract(amazon_vd["date"], "MONTH"))
            == "DATE_PART('MONTH', \"date\")"
        )

    def test_factorial(self, titanic_vd):
        assert str(st.factorial(titanic_vd["age"])) == '("age")!'

    def test_floor(self, titanic_vd):
        assert str(st.floor(titanic_vd["age"])) == 'FLOOR("age")'

    def test_gamma(self, titanic_vd):
        assert str(st.gamma(titanic_vd["age"])) == '("age" - 1)!'

    def test_getdate(self):
        assert str(st.getdate()) == "GETDATE()"

    def test_getutcdate(self):
        assert str(st.getutcdate()) == "GETUTCDATE()"

    def test_hash(self, titanic_vd):
        assert str(st.hash(titanic_vd["age"])) == 'HASH("age")'
        assert (
            str(st.hash(titanic_vd["age"], titanic_vd["fare"])) == 'HASH("age", "fare")'
        )

    def test_hour(self, amazon_vd):
        assert str(st.hour(amazon_vd["date"])) == 'HOUR("date")'

    def test_interval(self, amazon_vd):
        assert str(st.interval("1 day")) == "('1 day')::interval"

    def test_isfinite(self, titanic_vd):
        assert (
            str(st.isfinite(titanic_vd["age"]))
            == '(("age") = ("age")) AND (ABS("age") < \'inf\'::float)'
        )

    def test_isinf(self, titanic_vd):
        assert str(st.isinf(titanic_vd["age"])) == "ABS(\"age\") = 'inf'::float"

    def test_isnan(self, titanic_vd):
        assert str(st.isnan(titanic_vd["age"])) == '(("age") != ("age"))'

    def test_lgamma(self, titanic_vd):
        assert str(st.lgamma(titanic_vd["age"])) == 'LN(("age" - 1)!)'

    def test_ln(self, titanic_vd):
        assert str(st.ln(titanic_vd["age"])) == 'LN("age")'

    def test_log(self, titanic_vd):
        assert str(st.log(titanic_vd["age"])) == 'LOG(10, "age")'

    def test_minute(self, amazon_vd):
        assert str(st.minute(amazon_vd["date"])) == 'MINUTE("date")'

    def test_microsecond(self, amazon_vd):
        assert str(st.microsecond(amazon_vd["date"])) == 'MICROSECOND("date")'

    def test_month(self, amazon_vd):
        assert str(st.month(amazon_vd["date"])) == 'MONTH("date")'

    def test_nullifzero(self, amazon_vd):
        assert str(st.nullifzero(amazon_vd["number"])) == 'NULLIFZERO("number")'

    def test_overlaps(self):
        assert (
            str(st.overlaps("11-03-1993", "12-03-1993", "11-30-1993", "11-30-1994"))
            == "('11-03-1993', '12-03-1993') OVERLAPS ('11-30-1993', '11-30-1994')"
        )

    def test_quarter(self, amazon_vd):
        assert str(st.quarter(amazon_vd["date"])) == 'QUARTER("date")'

    def test_radians(self, titanic_vd):
        assert str(st.radians(titanic_vd["age"])) == 'RADIANS("age")'

    def test_seeded_random(self):
        assert str(st.seeded_random(10)) == "SEEDED_RANDOM(10)"

    def test_random(self):
        assert str(st.random()) == "RANDOM()"

    def test_randomint(self):
        assert str(st.randomint(10)) == "RANDOMINT(10)"

    def test_round(self, titanic_vd):
        assert str(st.round(titanic_vd["age"], 3)) == 'ROUND("age", 3)'

    def test_round_date(self, amazon_vd):
        assert str(st.round_date(amazon_vd["date"])) == "ROUND(\"date\", 'DD')"

    def test_second(self, amazon_vd):
        assert str(st.second(amazon_vd["date"])) == 'SECOND("date")'

    def test_sign(self, titanic_vd):
        assert str(st.sign(titanic_vd["age"])) == 'SIGN("age")'

    def test_sin(self, titanic_vd):
        assert str(st.sin(titanic_vd["age"])) == 'SIN("age")'

    def test_sinh(self, titanic_vd):
        assert str(st.sinh(titanic_vd["age"])) == 'SINH("age")'

    def test_sqrt(self, titanic_vd):
        assert str(st.sqrt(titanic_vd["age"])) == 'SQRT("age")'

    def test_tan(self, titanic_vd):
        assert str(st.tan(titanic_vd["age"])) == 'TAN("age")'

    def test_tanh(self, titanic_vd):
        assert str(st.tanh(titanic_vd["age"])) == 'TANH("age")'

    def test_timestamp(self, amazon_vd):
        assert str(st.timestamp("05/09/1959")) == "('05/09/1959')::timestamp"

    def test_trunc(self, titanic_vd):
        assert str(st.trunc(titanic_vd["age"], 3)) == 'TRUNC("age", 3)'

    def test_week(self, amazon_vd):
        assert str(st.week(amazon_vd["date"])) == 'WEEK("date")'

    def test_year(self, amazon_vd):
        assert str(st.year(amazon_vd["date"])) == 'YEAR("date")'

    def test_zeroifnull(self, amazon_vd):
        assert str(st.zeroifnull(amazon_vd["date"])) == 'ZEROIFNULL("date")'

    def test_constants(self):
        assert str(st.pi) == "PI()"
        assert str(st.e) == "EXP(1)"
        assert str(st.tau) == "2 * PI()"
        assert str(st.inf) == "'inf'::float"
        assert str(st.nan) == "'nan'::float"
