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

import pytest, datetime, warnings
from verticapy import vDataFrame, drop_table, errors

from verticapy import set_option

set_option("print_info", False)


@pytest.fixture(scope="module")
def amazon_vd(base):
    from verticapy.learn.datasets import load_amazon

    amazon = load_amazon(cursor=base.cursor)
    yield amazon
    with warnings.catch_warnings(record=True) as w:
        drop_table(
            name="public.amazon", cursor=base.cursor,
        )


@pytest.fixture(scope="module")
def iris_vd(base):
    from verticapy.learn.datasets import load_iris

    iris = load_iris(cursor=base.cursor)
    yield iris
    with warnings.catch_warnings(record=True) as w:
        drop_table(name="public.iris", cursor=base.cursor)


@pytest.fixture(scope="module")
def smart_meters_vd(base):
    from verticapy.learn.datasets import load_smart_meters

    smart_meters = load_smart_meters(cursor=base.cursor)
    yield smart_meters
    with warnings.catch_warnings(record=True) as w:
        drop_table(name="public.smart_meters", cursor=base.cursor)


@pytest.fixture(scope="module")
def titanic_vd(base):
    from verticapy.learn.datasets import load_titanic

    titanic = load_titanic(cursor=base.cursor)
    yield titanic
    with warnings.catch_warnings(record=True) as w:
        drop_table(name="public.titanic", cursor=base.cursor)


class TestvDFFeatureEngineering:
    def test_vDF_cummax(self, amazon_vd):
        amazon_copy = amazon_vd.copy()
        amazon_copy.cummax(
            column="number", by=["state"], order_by=["date"], name="cummax_number"
        )

        assert amazon_copy["cummax_number"].max() == 25963.0

    def test_vDF_cummin(self, amazon_vd):
        amazon_copy = amazon_vd.copy()
        amazon_copy.cummin(
            column="number", by=["state"], order_by=["date"], name="cummin_number"
        )

        assert amazon_copy["cummin_number"].min() == 0.0

    def test_vDF_cumprod(self, iris_vd):
        iris_copy = iris_vd.copy()
        iris_copy.cumprod(
            column="PetalWidthCm",
            by=["Species"],
            order_by=["PetalLengthCm"],
            name="cumprod_number",
        )

        assert iris_copy["cumprod_number"].max() == 1347985569095150.0

    def test_vDF_cumsum(self, amazon_vd):
        amazon_copy = amazon_vd.copy()
        amazon_copy.cumsum(
            column="number", by=["state"], order_by=["date"], name="cumsum_number"
        )

        assert amazon_copy["cumsum_number"].max() == pytest.approx(723629.0)

    def test_vDF_rolling(self, titanic_vd):
        # func = "aad"
        titanic_copy = titanic_vd.copy()
        titanic_copy.rolling(
            func="aad",
            column="age",
            preceding=10,
            following=1,
            by=["pclass"],
            order_by={"name": "asc", "ticket": "desc"},
            method="rows",
            rule="past",
            name="aad",
        )
        assert titanic_copy["aad"].max() == pytest.approx(38.0)

        # func = "beta"
        titanic_copy = titanic_vd.copy()
        titanic_copy.rolling(
            func="beta",
            column="age",
            column2="fare",
            preceding=100,
            following=100,
            by=["pclass"],
            order_by={"name": "asc", "ticket": "desc"},
            name="beta",
        )
        titanic_copy["beta"].max() == pytest.approx(0.181693993120358)

        # func = "count"
        titanic_copy = titanic_vd.copy()
        titanic_copy.rolling(
            func="count",
            column="age",
            preceding=10,
            following=1,
            by=["pclass"],
            order_by={"name": "asc", "ticket": "desc"},
            name="count",
        )
        assert titanic_copy["count"].max() == 12

        # func = "corr"
        titanic_copy = titanic_vd.copy()
        titanic_copy.rolling(
            func="corr",
            column="age",
            column2="fare",
            preceding=10,
            following=1,
            name="corr",
            order_by={"name": "asc", "ticket": "desc"},
        )
        assert titanic_copy["corr"].median() == pytest.approx(0.317169567973457)

        # func = "cov"
        titanic_copy = titanic_vd.copy()
        titanic_copy.rolling(
            func="cov",
            column="age",
            column2="fare",
            preceding=10,
            following=1,
            name="cov",
            order_by={"name": "asc", "ticket": "desc"},
        )
        assert titanic_copy["cov"].median() == pytest.approx(101.433815700758)

        # func = "kurtosis"
        titanic_copy = titanic_vd.copy()
        titanic_copy.rolling(
            func="kurtosis",
            column="age",
            preceding=10,
            following=1,
            name="kurtosis",
            order_by={"name": "asc", "ticket": "desc"},
        )
        assert titanic_copy["kurtosis"].min() == pytest.approx(-9.98263002423947)

        # func = "jb"
        titanic_copy = titanic_vd.copy()
        titanic_copy.rolling(
            func="jb",
            column="age",
            preceding=10,
            following=1,
            name="jb",
            order_by={"name": "asc", "ticket": "desc"},
        )
        assert titanic_copy["jb"].min() == pytest.approx(0.000815735789122907)

        # func = "max"
        titanic_copy = titanic_vd.copy()
        titanic_copy.rolling(
            func="max",
            column="age",
            preceding=10,
            following=1,
            name="max",
            order_by={"name": "asc", "ticket": "desc"},
        )
        assert titanic_copy["max"].min() == pytest.approx(24)

        # func = "mean"
        titanic_copy = titanic_vd.copy()
        titanic_copy.rolling(
            func="mean",
            column="age",
            preceding=10,
            following=1,
            name="mean",
            order_by={"name": "asc", "ticket": "desc"},
        )
        assert titanic_copy["mean"].min() == pytest.approx(11.4545454545455)

        # func = "min"
        titanic_copy = titanic_vd.copy()
        titanic_copy.rolling(
            func="min",
            column="age",
            preceding=10,
            following=1,
            name="min",
            order_by={"name": "asc", "ticket": "desc"},
        )
        assert titanic_copy["min"].max() == pytest.approx(34.5)

        # func = "prod"
        titanic_copy = titanic_vd.copy()
        titanic_copy.rolling(
            func="prod",
            column="age",
            preceding=10,
            following=1,
            name="prod",
            order_by={"name": "asc", "ticket": "desc"},
        )
        assert titanic_copy["prod"].min() == pytest.approx(409.92)

        # func = "range"
        titanic_copy = titanic_vd.copy()
        titanic_copy.rolling(
            func="range",
            column="age",
            preceding=10,
            following=1,
            name="range",
            order_by={"name": "asc", "ticket": "desc"},
        )
        assert titanic_copy["range"].max() == pytest.approx(72)

        # func = "sem"
        titanic_copy = titanic_vd.copy()
        titanic_copy.rolling(
            func="sem",
            column="age",
            preceding=10,
            following=1,
            name="sem",
            order_by={"name": "asc", "ticket": "desc"},
        )
        assert titanic_copy["sem"].median() == pytest.approx(4.28774427352888)

        # func = "skewness"
        titanic_copy = titanic_vd.copy()
        titanic_copy.rolling(
            func="skewness",
            column="age",
            preceding=10,
            following=1,
            name="skewness",
            order_by={"name": "asc", "ticket": "desc"},
        )
        assert titanic_copy["skewness"].max() == pytest.approx(21.6801549178229)

        # func = "sum"
        titanic_copy = titanic_vd.copy()
        titanic_copy.rolling(
            func="sum",
            column="age",
            preceding=10,
            following=1,
            name="sum",
            order_by={"name": "asc", "ticket": "desc"},
        )
        assert titanic_copy["sum"].max() == pytest.approx(545)

        # func = "std"
        titanic_copy = titanic_vd.copy()
        titanic_copy.rolling(
            func="std",
            column="pclass",
            preceding=10,
            following=1,
            name="std",
            order_by={"name": "asc", "ticket": "desc"},
        )
        assert titanic_copy["std"].median() == pytest.approx(0.792961461098759)

        # func = "var"
        titanic_copy = titanic_vd.copy()
        titanic_copy.rolling(
            func="var",
            column="pclass",
            preceding=10,
            following=1,
            name="var",
            order_by={"name": "asc", "ticket": "desc"},
        )
        assert titanic_copy["var"].median() == pytest.approx(0.628787878787879)

    def test_vDF_analytic(self, titanic_vd):
        # func = "aad"
        titanic_copy = titanic_vd.copy()
        titanic_copy.analytic(func="aad", column="age", by=["pclass"], name="aad")
        assert titanic_copy["aad"].max() == pytest.approx(12.1652677287016)

        # func = "beta"
        titanic_copy = titanic_vd.copy()
        titanic_copy.analytic(
            func="beta", column="age", column2="fare", by=["pclass"], name="beta"
        )
        assert titanic_copy["beta"].min() == pytest.approx(-0.293055805788566)

        # func = "count"
        titanic_copy = titanic_vd.copy()
        titanic_copy.analytic(
            func="count", column="age", by=["pclass"], name="age_count"
        )
        assert titanic_copy["age_count"].max() == 477

        # func = "corr"
        titanic_copy = titanic_vd.copy()
        titanic_copy.analytic(func="corr", column="age", column2="fare", name="corr")
        assert titanic_copy["corr"].median() == pytest.approx(0.316603147524037)

        # func = "cov"
        titanic_copy = titanic_vd.copy()
        titanic_copy.analytic(func="cov", column="age", column2="fare", name="cov")
        assert titanic_copy["cov"].median() == pytest.approx(240.60639320099)

        # func = "ema"
        titanic_copy = titanic_vd.copy()
        titanic_copy.analytic(
            func="ema",
            column="age",
            by=["survived"],
            name="ema",
            order_by={"name": "asc", "ticket": "desc"},
        )
        assert titanic_copy["ema"].min() == pytest.approx(5.98894892036915)

        # func = "first_value"
        titanic_copy = titanic_vd.copy()
        titanic_copy.analytic(
            func="first_value",
            column="age",
            name="first_value",
            order_by={"name": "asc", "ticket": "desc"},
        )
        assert titanic_copy["first_value"].min() == pytest.approx(42)

        # func = "iqr"
        titanic_copy = titanic_vd.copy()
        titanic_copy.analytic(func="iqr", column="age", name="iqr")
        assert titanic_copy["iqr"].min() == pytest.approx(18)

        # func = "dense_rank"
        titanic_copy = titanic_vd.copy()
        titanic_copy.analytic(
            func="dense_rank",
            name="dense_rank",
            order_by={"name": "asc", "ticket": "desc"},
        )
        assert titanic_copy["dense_rank"].max() == pytest.approx(1234)

        # func = "kurtosis"
        titanic_copy = titanic_vd.copy()
        titanic_copy.analytic(func="kurtosis", column="age", name="kurtosis")
        assert titanic_copy["kurtosis"].min() == pytest.approx(0.1568969133)

        # func = "jb"
        titanic_copy = titanic_vd.copy()
        titanic_copy.analytic(func="jb", column="age", name="jb")
        assert titanic_copy["jb"].min() == pytest.approx(28.802353)

        # func = "lead"
        titanic_copy = titanic_vd.copy()
        titanic_copy.analytic(
            func="lead",
            column="age",
            name="lead",
            order_by={"name": "asc", "ticket": "desc"},
        )
        assert titanic_copy["lead"].min() == pytest.approx(0.33)

        # func = "lag"
        titanic_copy = titanic_vd.copy()
        titanic_copy.analytic(
            func="lag",
            column="age",
            name="lag",
            order_by={"name": "asc", "ticket": "desc"},
        )
        assert titanic_copy["lag"].min() == pytest.approx(0.33)

        # func = "last_value" -
        titanic_copy = titanic_vd.copy()
        titanic_copy.analytic(
            func="last_value",
            column="age",
            name="last_value",
            order_by={"name": "asc", "ticket": "desc"},
        )
        assert titanic_copy["last_value"].min() == pytest.approx(0.33)

        # func = "mad"
        titanic_copy = titanic_vd.copy()
        titanic_copy.analytic(func="mad", column="age", name="mad")
        assert titanic_copy["mad"].min() == pytest.approx(11.14643931)

        # func = "max"
        titanic_copy = titanic_vd.copy()
        titanic_copy.analytic(func="max", column="age", name="max")
        assert titanic_copy["max"].min() == pytest.approx(80)

        # func = "mean"
        titanic_copy = titanic_vd.copy()
        titanic_copy.analytic(func="mean", column="age", name="mean")
        assert titanic_copy["mean"].min() == pytest.approx(30.15245737)

        # func = "median"
        titanic_copy = titanic_vd.copy()
        titanic_copy.analytic(func="median", column="age", name="median")
        assert titanic_copy["median"].min() == pytest.approx(28)

        # func = "min"
        titanic_copy = titanic_vd.copy()
        titanic_copy.analytic(func="min", column="age", name="min")
        assert titanic_copy["min"].max() == pytest.approx(0.33)

        # func = "mode"
        titanic_copy = titanic_vd.copy()
        titanic_copy.analytic(func="mode", column="embarked", name="mode")
        assert titanic_copy["mode"].distinct() == ["S"]
        assert titanic_copy["mode_count"].distinct() == [873]

        # func = "q%"
        titanic_copy = titanic_vd.copy()
        titanic_copy.analytic(func="20%", column="age", name="q20%")
        assert titanic_copy["q20%"].max() == pytest.approx(19)

        # func = "pct_change"
        titanic_copy = titanic_vd.copy()
        titanic_copy.analytic(
            func="pct_change",
            column="age",
            name="pct_change",
            order_by={"name": "asc", "ticket": "desc"},
        )
        assert titanic_copy["pct_change"].max() == pytest.approx(103.03030303030303)

        # func = "percent_rank"
        titanic_copy = titanic_vd.copy()
        titanic_copy.analytic(
            func="percent_rank",
            name="percent_rank",
            order_by={"name": "asc", "ticket": "desc"},
        )
        assert titanic_copy["percent_rank"].max() == pytest.approx(1)

        # func = "prod"
        titanic_copy = titanic_vd.copy()
        titanic_copy.analytic(func="prod", column="fare", by=["pclass"], name="prod")
        assert titanic_copy["prod"].min() == 0

        # func = "range"
        titanic_copy = titanic_vd.copy()
        titanic_copy.analytic(func="range", column="age", name="range")
        assert titanic_copy["range"].max() == pytest.approx(79.67)

        # func = "rank"
        titanic_copy = titanic_vd.copy()
        titanic_copy.analytic(
            func="rank", name="rank", order_by={"name": "asc", "ticket": "desc"}
        )
        assert titanic_copy["rank"].max() == pytest.approx(1234)

        # func = "row_number"
        titanic_copy = titanic_vd.copy()
        titanic_copy.analytic(
            func="row_number",
            name="row_number",
            order_by={"name": "asc", "ticket": "desc"},
        )
        assert titanic_copy["row_number"].max() == pytest.approx(1234)

        # func = "sem"
        titanic_copy = titanic_vd.copy()
        titanic_copy.analytic(func="sem", column="age", name="sem")
        assert titanic_copy["sem"].median() == pytest.approx(0.457170684605937)

        # func = "skewness"
        titanic_copy = titanic_vd.copy()
        titanic_copy.analytic(func="skewness", column="age", name="skewness")
        assert titanic_copy["skewness"].max() == pytest.approx(0.4088764607)

        # func = "sum"
        titanic_copy = titanic_vd.copy()
        titanic_copy.analytic(func="sum", column="age", name="sum")
        assert titanic_copy["sum"].max() == pytest.approx(30062)

        # func = "std"
        titanic_copy = titanic_vd.copy()
        titanic_copy.analytic(func="std", column="age", name="std")
        assert titanic_copy["std"].median() == pytest.approx(14.4353046299159)

        # func = "unique"
        titanic_copy = titanic_vd.copy()
        titanic_copy.analytic(func="unique", column="pclass", name="unique")
        assert titanic_copy["unique"].max() == pytest.approx(3)

        # func = "var"
        titanic_copy = titanic_vd.copy()
        titanic_copy.analytic(func="var", column="age", name="var")
        assert titanic_copy["var"].median() == pytest.approx(208.378019758472)

    def test_vDF_asfreq(self, smart_meters_vd):
        # bfill method
        result1 = smart_meters_vd.asfreq(
            ts="time", rule="1 hour", method={"val": "bfill"}, by=["id"]
        )
        result1.sort({"id": "asc", "time": "asc"})

        assert result1.shape() == (148189, 3)
        assert result1["time"][2] == datetime.datetime(2014, 1, 1, 13, 0)
        assert result1["id"][2] == 0
        assert result1["val"][2] == pytest.approx(0.029)

        # ffill
        result2 = smart_meters_vd.asfreq(
            ts="time", rule="1 hour", method={"val": "ffill"}, by=["id"]
        )
        result2.sort({"id": "asc", "time": "asc"})

        assert result2.shape() == (148189, 3)
        assert result2["time"][2] == datetime.datetime(2014, 1, 1, 13, 0)
        assert result2["id"][2] == 0
        assert result2["val"][2] == pytest.approx(0.277)

        # linear method
        result3 = smart_meters_vd.asfreq(
            ts="time", rule="1 hour", method={"val": "linear"}, by=["id"]
        )
        result3.sort({"id": "asc", "time": "asc"})

        assert result3.shape() == (148189, 3)
        assert result3["time"][2] == datetime.datetime(2014, 1, 1, 13, 0)
        assert result3["id"][2] == 0
        assert result3["val"][2] == pytest.approx(0.209363636363636)

    def test_vDF_sessionize(self, smart_meters_vd):
        smart_meters_copy = smart_meters_vd.copy()

        # expected exception
        with pytest.raises(errors.QueryError) as exception_info:
            smart_meters_copy.sessionize(
                ts="time", by=["id"], session_threshold="1 time", name="slot"
            )
        # checking the error message
        assert exception_info.match("seems to be incorrect")

        smart_meters_copy.sessionize(
            ts="time", by=["id"], session_threshold="1 hour", name="slot"
        )
        smart_meters_copy.sort({"id": "asc", "time": "asc"})

        assert smart_meters_copy.shape() == (11844, 4)
        assert smart_meters_copy["time"][2] == datetime.datetime(2014, 1, 2, 10, 45)
        assert smart_meters_copy["val"][2] == 0.321
        assert smart_meters_copy["id"][2] == 0
        assert smart_meters_copy["slot"][2] == 2

    def test_vDF_case_when(self, titanic_vd):
        titanic_copy = titanic_vd.copy()
        titanic_copy.case_when(
            "age_category",
            titanic_copy["age"] < 12,
            "children",
            titanic_copy["age"] < 18,
            "teenagers",
            titanic_copy["age"] > 60,
            "seniors",
            titanic_copy["age"] < 25,
            "young adults",
            "adults",
        )

        assert titanic_copy["age_category"].distinct() == [
            "adults",
            "children",
            "seniors",
            "teenagers",
            "young adults",
        ]

    def test_vDF_eval(self, titanic_vd):
        # new feature creation
        titanic_copy = titanic_vd.copy()
        titanic_copy.eval(name="family_size", expr="parch + sibsp + 1")

        assert titanic_copy["family_size"].max() == 11

        # Customized SQL code evaluation
        titanic_copy = titanic_vd.copy()
        titanic_copy.eval(
            name="has_life_boat", expr="CASE WHEN boat IS NULL THEN 0 ELSE 1 END"
        )

        assert titanic_copy["boat"].count() == titanic_copy["has_life_boat"].sum()

    def test_vDF_abs(self, titanic_vd):
        # Testing vDataFrame.abs
        titanic_copy = titanic_vd.copy()

        titanic_copy.normalize(["fare", "age"])
        assert titanic_copy["fare"].min() == pytest.approx(-0.64513441)
        assert titanic_copy["age"].min() == pytest.approx(-2.0659389)

        titanic_copy.abs(["fare", "age"])
        assert titanic_copy["fare"].min() == pytest.approx(0.001082821)
        assert titanic_copy["age"].min() == pytest.approx(0.010561423)

        # Testing vDataFrame[].abs
        titanic_copy = titanic_vd.copy()
        titanic_copy["fare"].normalize()
        titanic_copy["fare"].abs()

        assert titanic_copy["fare"].min() == pytest.approx(0.001082821)

    def test_vDF_apply(self, titanic_vd):
        ### Testing vDataFrame.apply
        titanic_copy = titanic_vd.copy()

        titanic_copy.apply(
            func={
                "boat": "DECODE({}, NULL, 0, 1)",
                "age": "COALESCE(age, AVG({}) OVER (PARTITION BY pclass, sex))",
                "name": "REGEXP_SUBSTR({}, ' ([A-Za-z])+\\.')",
            }
        )

        assert titanic_copy["boat"].sum() == titanic_vd["boat"].count()
        assert titanic_copy["age"].std() == pytest.approx(13.234162542)
        assert len(titanic_copy["name"].distinct()) == 16

        ### Testing vDataFrame[].apply
        titanic_copy = titanic_vd.copy()
        titanic_copy["age"].apply(func="POWER({}, 2)")
        assert titanic_copy["age"].min() == pytest.approx(0.1089)

        titanic_copy = titanic_vd.copy()
        titanic_copy["age"].apply(func="POWER({}, 2)", copy_name="age_pow_2")
        assert titanic_copy["age_pow_2"].min() == pytest.approx(0.1089)

    def test_vDF_apply_fun(self, titanic_vd):
        # func = "abs"
        titanic_copy = titanic_vd.copy()
        titanic_copy["age"].apply_fun(func="abs")

        assert titanic_copy["age"].min() == pytest.approx(0.33)

        # func = "acos"
        titanic_copy = titanic_vd.copy()
        titanic_copy["survived"].apply_fun(func="acos")

        assert titanic_copy["survived"].max() == pytest.approx(1.57079632)

        # func = "asin"
        titanic_copy = titanic_vd.copy()
        titanic_copy["survived"].apply_fun(func="asin")

        assert titanic_copy["survived"].max() == pytest.approx(1.57079632)

        # func = "atan"
        titanic_copy = titanic_vd.copy()
        titanic_copy["survived"].apply_fun(func="atan")

        assert titanic_copy["survived"].max() == pytest.approx(0.7853981633)

        # func = "cbrt"
        titanic_copy = titanic_vd.copy()
        titanic_copy["age"].apply_fun(func="cbrt")

        assert titanic_copy["age"].min() == pytest.approx(0.691042323)

        # func = "ceil"
        titanic_copy = titanic_vd.copy()
        titanic_copy["age"].apply_fun(func="ceil")

        assert titanic_copy["age"].min() == pytest.approx(1)

        # func = "cos"
        titanic_copy = titanic_vd.copy()
        titanic_copy["age"].apply_fun(func="cos")

        assert titanic_copy["age"].min() == pytest.approx(-0.9999608263)

        # func = "cosh"
        titanic_copy = titanic_vd.copy()
        titanic_copy["age"].apply_fun(func="cosh")

        assert titanic_copy["age"].min() == pytest.approx(1.05494593)

        # func = "cot"
        titanic_copy = titanic_vd.copy()
        titanic_copy["age"].apply_fun(func="exp")

        assert titanic_copy["age"].min() == pytest.approx(1.390968128)

        # func = "floor"
        titanic_copy = titanic_vd.copy()
        titanic_copy["age"].apply_fun(func="floor")

        assert titanic_copy["age"].min() == pytest.approx(0)

        # func = "ln"
        titanic_copy = titanic_vd.copy()
        titanic_copy["age"].apply_fun(func="ln")

        assert titanic_copy["age"].max() == pytest.approx(4.382026634)

        # func = "log"
        titanic_copy = titanic_vd.copy()
        titanic_copy["age"].apply_fun(func="log")

        assert titanic_copy["age"].min() == pytest.approx(-8.312950414)

        # func = "log10"
        titanic_copy = titanic_vd.copy()
        titanic_copy["age"].apply_fun(func="log10")

        assert titanic_copy["age"].max() == pytest.approx(1.903089986)

        # func = "mod"
        titanic_copy = titanic_vd.copy()
        titanic_copy["age"].apply_fun(func="mod")

        assert titanic_copy["age"].max() == pytest.approx(1.5)

        # func = "pow"
        titanic_copy = titanic_vd.copy()
        titanic_copy["age"].apply_fun(func="pow", x=3)

        assert titanic_copy["age"].min() == pytest.approx(0.035937)

        # func = "round"
        titanic_copy = titanic_vd.copy()
        titanic_copy["fare"].apply_fun(func="round")

        assert titanic_copy["fare"].max() == pytest.approx(512.33)

        # func = "sign"
        titanic_copy = titanic_vd.copy()
        titanic_copy["age"].apply_fun(func="sign")

        assert titanic_copy["age"].min() == pytest.approx(1)

        # func = "sin"
        titanic_copy = titanic_vd.copy()
        titanic_copy["age"].apply_fun(func="sin")

        assert titanic_copy["age"].min() == pytest.approx(-0.9999902065)

        # func = "sinh"
        titanic_copy = titanic_vd.copy()
        titanic_copy["age"].apply_fun(func="sinh")

        assert titanic_copy["age"].min() == pytest.approx(0.3360221975)

        # func = "sqrt"
        titanic_copy = titanic_vd.copy()
        titanic_copy["age"].apply_fun(func="sqrt")

        assert titanic_copy["age"].min() == pytest.approx(0.5744562646)

        # func = "tan"
        titanic_copy = titanic_vd.copy()
        titanic_copy["age"].apply_fun(func="tan")

        assert titanic_copy["age"].min() == pytest.approx(-225.9508464)

        # func = "tanh"
        titanic_copy = titanic_vd.copy()
        titanic_copy["age"].apply_fun(func="tanh")

        assert titanic_copy["age"].min() == pytest.approx(0.3185207769)

    def test_vDF_applymap(self, titanic_vd):
        titanic_copy = titanic_vd.copy()
        titanic_copy.applymap(func="COALESCE({}, 0)", numeric_only=True)

        assert titanic_copy["age"].count() == 1234

    def test_vDF_data_part(self, smart_meters_vd):
        smart_meters_copy = smart_meters_vd.copy()
        smart_meters_copy["time"].date_part("hour")

        assert len(smart_meters_copy["time"].distinct()) == 24

    def test_vDF_round(self, smart_meters_vd):
        smart_meters_copy = smart_meters_vd.copy()
        smart_meters_copy["val"].round(n=1)

        assert smart_meters_copy["val"].mode() == 0.1000000

    def test_vDF_slice(self, smart_meters_vd):
        # start = True
        smart_meters_copy = smart_meters_vd.copy()
        smart_meters_copy["time"].slice(length=1, unit="hour")

        assert smart_meters_copy["time"].min() == datetime.datetime(2014, 1, 1, 1, 0)

        # start = False
        smart_meters_copy = smart_meters_vd.copy()
        smart_meters_copy["time"].slice(length=1, unit="hour", start=False)

        assert smart_meters_copy["time"].min() == datetime.datetime(2014, 1, 1, 2, 0)

    def test_vDF_regexp(self, titanic_vd):
        # method = "count"
        titanic_copy = titanic_vd.copy()
        titanic_copy.regexp(column="name", pattern="son", method="count", name="name2")

        assert titanic_copy["name2"].max() == 2

        # method = "ilike"
        titanic_copy = titanic_vd.copy()
        titanic_copy.regexp(
            column="name", pattern="mrs.", method="ilike", occurrence=1, name="name2"
        )

        assert titanic_copy["name2"].sum() == 185

        # method = "instr"
        titanic_copy = titanic_vd.copy()
        titanic_copy.regexp(
            column="name", pattern="Mrs.", method="instr", position=2, name="name2"
        )

        assert titanic_copy["name2"].max() == 23

        # method = "like"
        titanic_copy = titanic_vd.copy()
        titanic_copy.regexp(
            column="name", pattern="Mrs.", method="like", position=2, name="name2"
        )

        assert titanic_copy["name2"].sum() == 185

        # method = "not_ilike"
        titanic_copy = titanic_vd.copy()
        titanic_copy.regexp(
            column="name", pattern="mrs.", method="not_ilike", position=2, name="name2"
        )

        assert titanic_copy["name2"].sum() == 1049

        # method = "not_like"
        titanic_copy = titanic_vd.copy()
        titanic_copy.regexp(
            column="name", pattern="Mrs.", method="not_like", position=2, name="name2"
        )

        assert titanic_copy["name2"].sum() == 1049

        # method = "replace"
        titanic_copy = titanic_vd.copy()
        titanic_copy.regexp(
            column="name",
            pattern="Mrs.",
            method="replace",
            return_position=5,
            replacement="Mr.",
            name="name2",
        )
        titanic_copy.sort(["name2"])

        assert titanic_copy["name2"][3] == "Abbott, Mr. Stanton (Rosa Hunt)"

        # method = "substr"
        titanic_copy = titanic_vd.copy()
        titanic_copy.regexp(
            column="name", pattern="[^,]+", method="substr", occurrence=2, name="name2"
        )
        titanic_copy.sort(["name2"])

        assert titanic_copy["name2"][3] == " Col. John Jacob"

    def test_vDF_str_contains(self, titanic_vd):
        titanic_copy = titanic_vd.copy()
        titanic_copy["name"].str_contains(pat=" ([A-Za-z])+\\.")

        assert titanic_copy["name"].dtype() == "boolean"

    def test_vDF_count(self, titanic_vd):
        titanic_copy = titanic_vd.copy()
        titanic_copy["name"].str_count(pat=" ([A-Za-z])+\\.")

        assert titanic_copy["name"].distinct() == [1, 2]

    def test_vDF_str_extract(self, titanic_vd):
        titanic_copy = titanic_vd.copy()
        titanic_copy["name"].str_extract(pat=" ([A-Za-z])+\\.")

        assert len(titanic_copy["name"].distinct()) == 16

    def test_vDF_str_replace(self, titanic_vd):
        titanic_copy = titanic_vd.copy()
        titanic_copy["name"].str_replace(
            to_replace=" ([A-Za-z])+\\.", value="VERTICAPY"
        )

        assert "VERTICAPY" in titanic_copy["name"][0]

    def test_vDF_str_slice(self, titanic_vd):
        titanic_copy = titanic_vd.copy()
        titanic_copy["name"].str_slice(start=0, step=3)

        assert len(titanic_copy["name"].distinct()) == 165

    def test_vDF_add(self, titanic_vd):
        titanic_copy = titanic_vd.copy()
        titanic_copy["age"].add(2)

        assert titanic_copy["age"].mean() == pytest.approx(titanic_vd["age"].mean() + 2)

    def test_vDF_div(self, titanic_vd):
        titanic_copy = titanic_vd.copy()
        titanic_copy["age"].div(2)

        assert titanic_copy["age"].mean() == pytest.approx(titanic_vd["age"].mean() / 2)

    def test_vDF_mul(self, titanic_vd):
        titanic_copy = titanic_vd.copy()
        titanic_copy["age"].mul(2)

        assert titanic_copy["age"].mean() == pytest.approx(titanic_vd["age"].mean() * 2)

    def test_vDF_sub(self, titanic_vd):
        titanic_copy = titanic_vd.copy()
        titanic_copy["age"].sub(2)

        assert titanic_copy["age"].mean() == pytest.approx(titanic_vd["age"].mean() - 2)

    def test_vDF_add_copy(self, titanic_vd):
        titanic_copy = titanic_vd.copy()
        titanic_copy["age"].add_copy(name="copy_age")

        assert titanic_copy["copy_age"].mean() == titanic_copy["age"].mean()

    def test_vDF_copy(self, titanic_vd):
        titanic_copy = titanic_vd.copy()

        assert titanic_copy.get_columns() == titanic_vd.get_columns()
