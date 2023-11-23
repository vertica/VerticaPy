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
from verticapy.datasets import load_titanic, load_airline_passengers, load_amazon
import verticapy.stats as st
from verticapy.learn.linear_model import LinearRegression

set_option("print_info", False)


@pytest.fixture(scope="module")
def titanic_vd():
    titanic = load_titanic()
    yield titanic
    drop(name="public.titanic")


@pytest.fixture(scope="module")
def airline_vd():
    airline = load_airline_passengers()
    yield airline
    drop(name="public.airline_passengers")


@pytest.fixture(scope="module")
def amazon_vd():
    amazon = load_amazon()
    yield amazon
    drop(name="public.amazon")


class TestStats:
    def test_adfuller(self, amazon_vd):
        # testing without trend
        result = st.adfuller(
            amazon_vd,
            column="number",
            ts="date",
            by=["state"],
            p=40,
            with_trend=False,
        )
        assert result["value"][0] == pytest.approx(-0.4059507552046538, 1e-2)
        assert result["value"][1] == pytest.approx(0.684795156687264, 1e-2)
        assert not result["value"][-1]

        # testing with trend
        result = st.adfuller(
            amazon_vd,
            column="number",
            ts="date",
            by=["state"],
            p=40,
            with_trend=True,
        )
        assert result["value"][0] == pytest.approx(-0.4081159118011171, 1e-2)
        assert result["value"][1] == pytest.approx(0.683205052234998, 1e-2)
        assert not result["value"][-1]

    def test_cochrane_orcutt(self, airline_vd):
        airline_copy = airline_vd.copy()
        airline_copy["passengers_bias"] = (
            airline_copy["passengers"] ** 2 - 50 * st.random()
        )
        drop("lin_cochrane_orcutt_model_test", method="model")
        model = LinearRegression("lin_cochrane_orcutt_model_test")
        model.fit(airline_copy, ["passengers_bias"], "passengers")
        result = st.cochrane_orcutt(
            model,
            airline_copy,
            ts="date",
            prais_winsten=True,
        )
        assert result.intercept_ == pytest.approx(25.8582027191416, 1e-2)
        assert result.coef_[0] == pytest.approx(0.00123563974547625, 1e-2)
        model.drop()

    def test_durbin_watson(self, amazon_vd):
        result = st.durbin_watson(amazon_vd, eps="number", ts="date", by=["state"])
        assert result == pytest.approx(0.583991056156811, 1e-2)

    def test_het_arch(self, amazon_vd):
        result = st.het_arch(amazon_vd, eps="number", ts="date", by=["state"], p=2)
        assert result == (
            pytest.approx(883.1114423462593),
            pytest.approx(1.7170654186230018e-192),
            pytest.approx(511.33952787255066),
            pytest.approx(7.432857276079368e-207),
        )

    def test_het_breuschpagan(self, amazon_vd):
        result = amazon_vd.groupby(["date"], ["AVG(number) AS number"])
        result["lag_number"] = "LAG(number) OVER (ORDER BY date)"
        result = st.het_breuschpagan(result, eps="number", X=["lag_number"])
        assert result == (
            pytest.approx(68.30399583311137),
            pytest.approx(1.4013672773820152e-16),
            pytest.approx(94.83553579040095),
            pytest.approx(4.570574529673344e-19),
        )

    def test_het_goldfeldquandt(self, amazon_vd):
        vdf = amazon_vd.groupby(["date"], ["AVG(number) AS number"])
        vdf["lag_number"] = "LAG(number) OVER (ORDER BY date)"
        result = st.het_goldfeldquandt(vdf, y="number", X=["lag_number"])
        assert result == (
            pytest.approx(30.17263128858259),
            pytest.approx(1.3988910574921388e-55),
        )
        result2 = st.het_goldfeldquandt(
            vdf, y="number", X=["lag_number"], alternative="decreasing"
        )
        assert result2 == (
            pytest.approx(30.17263128858259),
            pytest.approx(0.9999999999999999),
        )
        result3 = st.het_goldfeldquandt(
            vdf, y="number", X=["lag_number"], alternative="two-sided"
        )
        assert result3 == (
            pytest.approx(30.17263128858259),
            pytest.approx(0.0),
        )

    def test_het_white(self, amazon_vd):
        result = amazon_vd.groupby(["date"], ["AVG(number) AS number"])
        result["lag_number"] = "LAG(number) OVER (ORDER BY date)"
        result = st.het_white(result, eps="number", X=["lag_number"])
        assert result == (
            pytest.approx(72.93566993238099),
            pytest.approx(1.3394533546740618e-17),
            pytest.approx(104.09070850396219),
            pytest.approx(1.6997687814870292e-20),
        )

    def test_jarque_bera(self, amazon_vd):
        result = st.jarque_bera(amazon_vd, column="number")
        assert result[0] == pytest.approx(930829.520860999, 1e-2)
        assert result[1] == pytest.approx(0.0, 1e-2)

    def test_kurtosistest(self, amazon_vd):
        result = st.kurtosistest(amazon_vd, column="number")
        assert result[0] == pytest.approx(47.31605467852915, 1e-2)
        assert result[1] == pytest.approx(0.0, 1e-2)

    def test_normaltest(self, amazon_vd):
        result = st.normaltest(amazon_vd, column="number")
        assert result[0] == pytest.approx(7645.980976250067, 1e-2)
        assert result[1] == pytest.approx(0.0, 1e-2)

    def test_ljungbox(self, amazon_vd):
        # testing Ljung–Box
        result = st.ljungbox(
            amazon_vd,
            column="number",
            ts="date",
            by=["state"],
            p=40,
            box_pierce=False,
        )
        assert result["Serial Correlation"][-1]
        assert result["p_value"][-1] == pytest.approx(0.0)
        assert result["Ljung–Box Test Statistic"][-1] == pytest.approx(
            33724.41181636157, 1e-2
        )

        # testing Box-Pierce
        result = st.ljungbox(
            amazon_vd,
            column="number",
            ts="date",
            by=["state"],
            p=40,
            box_pierce=True,
        )
        assert result["Serial Correlation"][-1]
        assert result["p_value"][-1] == pytest.approx(0.0)
        assert result["Box-Pierce Test Statistic"][-1] == pytest.approx(
            33601.96361200001, 1e-2
        )

    def test_mkt(self, amazon_vd):
        result = amazon_vd.groupby(["date"], ["AVG(number) AS number"])
        result = st.mkt(result, column="number", ts="date")
        assert result["value"][0] == pytest.approx(2.579654773618437, 1e-2)
        assert result["value"][1] == pytest.approx(3188.0, 1e-2)
        assert result["value"][2] == pytest.approx(1235.43662996799, 1e-2)
        assert result["value"][3] == pytest.approx(0.009889912917327177, 1e-2)
        assert result["value"][4]
        assert result["value"][5] == "increasing"

    def test_seasonal_decompose(self, airline_vd):
        result = st.seasonal_decompose(
            airline_vd,
            "Passengers",
            "date",
            period=12,
            mult=True,
            polynomial_order=-1,
        )
        assert result["passengers_trend"].avg() == pytest.approx(266.398518668831)
        assert result["passengers_seasonal"].avg() == pytest.approx(1.0)
        assert result["passengers_epsilon"].avg() == pytest.approx(1.05417531440333)
        result2 = st.seasonal_decompose(
            airline_vd,
            "Passengers",
            "date",
            period=12,
            mult=False,
            polynomial_order=-1,
        )
        assert result2["passengers_trend"].avg() == pytest.approx(266.398518668831)
        assert result2["passengers_seasonal"].avg() == pytest.approx(
            1.1842378929335e-15
        )
        assert result2["passengers_epsilon"].avg() == pytest.approx(13.9000924422799)

        result2 = st.seasonal_decompose(
            airline_vd,
            "Passengers",
            "date",
            mult=True,
            polynomial_order=2,
            estimate_seasonality=True,
        )
        assert result2["passengers_trend"].avg() == pytest.approx(280.298611111111)
        assert result2["passengers_seasonal"].avg() == pytest.approx(1.0)
        assert result2["passengers_epsilon"].avg() == pytest.approx(1.00044316689124)

    def test_skewtest(self, amazon_vd):
        result = st.skewtest(amazon_vd, column="number")
        assert result[0] == pytest.approx(73.53347500226347, 1e-2)
        assert result[1] == pytest.approx(0.0, 1e-2)

    def test_variance_inflation_factor(self, titanic_vd):
        result = st.variance_inflation_factor(
            titanic_vd, ["pclass", "survived", "age", "fare"]
        )
        assert result["VIF"][0] == pytest.approx(1.8761429731878563, 1e-2)
        assert result["VIF"][1] == pytest.approx(1.1859478232661875, 1e-2)
        assert result["VIF"][2] == pytest.approx(1.250542149908016, 1e-2)
        assert result["VIF"][3] == pytest.approx(1.4940557668701793, 1e-2)

    @pytest.mark.skip(reason="this test will be valid for Vertica v12.0.2")
    def test_jaro_distance(self, titanic_vd):
        assert (
            str(st.edit_distance(titanic_vd["name"], "Laurent"))
            == "JARO_DISTANCE(\"name\", 'Laurent')"
        )

    @pytest.mark.skip(reason="this test will be valid for Vertica v12.0.2")
    def test_jaro_winkler_distance(self, titanic_vd):
        assert (
            str(st.edit_distance(titanic_vd["name"], "Laurent"))
            == "JARO_WINKLER_DISTANCE(\"name\", 'Laurent')"
        )

    def test_edit_distance(self, titanic_vd):
        assert (
            str(st.edit_distance(titanic_vd["name"], "Laurent"))
            == "EDIT_DISTANCE(\"name\", 'Laurent')"
        )

    def test_soundex(self, titanic_vd):
        assert str(st.soundex(titanic_vd["name"])) == 'SOUNDEX("name")'

    def test_soundex_matches(self, titanic_vd):
        assert (
            str(st.soundex_matches(titanic_vd["name"], "Laurent"))
            == "SOUNDEX_MATCHES(\"name\", 'Laurent')"
        )

    def test_regexp_count(self, titanic_vd):
        assert (
            str(st.regexp_count(titanic_vd["name"], "([A-Za-z])+\\. "))
            == "REGEXP_COUNT(\"name\", '([A-Za-z])+\\. ', 1)"
        )

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

    def test_atan2(self, titanic_vd):
        assert (
            str(st.atan2(titanic_vd["age"], titanic_vd["fare"]))
            == 'ATAN2("age", "fare")'
        )

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
        assert str(st.PI) == "PI()"
        assert str(st.E) == "EXP(1)"
        assert str(st.TAU) == "2 * PI()"
        assert str(st.INF) == "'inf'::float"
        assert str(st.NAN) == "'nan'::float"
