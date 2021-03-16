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
from verticapy import vDataFrame, drop

from verticapy import set_option

set_option("print_info", False)


@pytest.fixture(scope="module")
def titanic_vd(base):
    from verticapy.datasets import load_titanic

    titanic = load_titanic(cursor=base.cursor)
    yield titanic
    with warnings.catch_warnings(record=True) as w:
        drop(
            name="public.titanic", cursor=base.cursor,
        )


@pytest.fixture(scope="module")
def market_vd(base):
    from verticapy.datasets import load_market

    market = load_market(cursor=base.cursor)
    yield market
    with warnings.catch_warnings(record=True) as w:
        drop(
            name="public.market", cursor=base.cursor,
        )


@pytest.fixture(scope="module")
def amazon_vd(base):
    from verticapy.datasets import load_amazon

    amazon = load_amazon(cursor=base.cursor)
    yield amazon
    with warnings.catch_warnings(record=True) as w:
        drop(
            name="public.amazon", cursor=base.cursor,
        )


class TestvDFDescriptiveStat:
    def test_vDF_aad(self, titanic_vd):
        # testing vDataFrame.aad
        result = titanic_vd.aad(columns=["age", "fare", "parch"])
        assert result["aad"][0] == pytest.approx(11.2547854194)
        assert result["aad"][1] == pytest.approx(30.6258659424)
        assert result["aad"][2] == pytest.approx(0.58208012314)

        # testing vDataFrame[].aad
        assert titanic_vd["age"].aad() == result["aad"][0]
        assert titanic_vd["fare"].aad() == result["aad"][1]
        assert titanic_vd["parch"].aad() == result["aad"][2]

    def test_vDF_agg(self, titanic_vd):
        # testing vDataFrame.agg
        result1 = titanic_vd.agg(
            func=["unique", "top", "min", "10%", "50%", "90%", "max"],
            columns=["age", "fare", "pclass", "survived"],
        )
        assert result1["unique"][0] == 96
        assert result1["unique"][1] == 277
        assert result1["unique"][2] == 3
        assert result1["unique"][3] == 2
        assert result1["top"][0] is None
        assert result1["top"][1] == pytest.approx(8.05)
        assert result1["top"][2] == 3
        assert result1["top"][3] == 0
        assert result1["min"][0] == pytest.approx(0.330)
        assert result1["min"][1] == 0
        assert result1["min"][2] == 1
        assert result1["min"][3] == 0
        assert result1["10%"][0] == pytest.approx(14.5)
        assert result1["10%"][1] == pytest.approx(7.5892)
        assert result1["10%"][2] == 1
        assert result1["10%"][3] == 0
        assert result1["50%"][0] == 28
        assert result1["50%"][1] == pytest.approx(14.4542)
        assert result1["50%"][2] == 3
        assert result1["50%"][3] == 0
        assert result1["90%"][0] == 50
        assert result1["90%"][1] == pytest.approx(79.13)
        assert result1["90%"][2] == 3
        assert result1["90%"][3] == 1
        assert result1["max"][0] == 80
        assert result1["max"][1] == pytest.approx(512.3292)
        assert result1["max"][2] == 3
        assert result1["max"][3] == 1

        result1_1 = titanic_vd.agg(
            func=["unique", "top", "min", "10%", "50%", "90%", "max"],
            columns=["age", "fare", "pclass", "survived"],
            ncols_block=2,
        )
        assert result1_1["unique"][0] == 96
        assert result1_1["unique"][1] == 277
        assert result1_1["unique"][2] == 3
        assert result1_1["unique"][3] == 2
        assert result1_1["top"][0] is None
        assert result1_1["top"][1] == pytest.approx(8.05)
        assert result1_1["top"][2] == 3
        assert result1_1["top"][3] == 0
        assert result1_1["min"][0] == pytest.approx(0.330)
        assert result1_1["min"][1] == 0
        assert result1_1["min"][2] == 1
        assert result1_1["min"][3] == 0
        assert result1_1["10%"][0] == pytest.approx(14.5)
        assert result1_1["10%"][1] == pytest.approx(7.5892)
        assert result1_1["10%"][2] == 1
        assert result1_1["10%"][3] == 0
        assert result1_1["50%"][0] == 28
        assert result1_1["50%"][1] == pytest.approx(14.4542)
        assert result1_1["50%"][2] == 3
        assert result1_1["50%"][3] == 0
        assert result1_1["90%"][0] == 50
        assert result1_1["90%"][1] == pytest.approx(79.13)
        assert result1_1["90%"][2] == 3
        assert result1_1["90%"][3] == 1
        assert result1_1["max"][0] == 80
        assert result1_1["max"][1] == pytest.approx(512.3292)
        assert result1_1["max"][2] == 3
        assert result1_1["max"][3] == 1

        result2 = titanic_vd.agg(
            func=[
                "aad",
                "approx_unique",
                "count",
                "cvar",
                "dtype",
                "iqr",
                "kurtosis",
                "jb",
                "mad",
                "mean",
                "median",
                "mode",
                "percent",
                "prod",
                "range",
                "sem",
                "skewness",
                "sum",
                "std",
                "top2",
                "top2_percent",
                "var",
            ],
            columns=["age", "pclass"],
        )
        assert result2["aad"][0] == pytest.approx(11.254785419447906)
        assert result2["aad"][1] == pytest.approx(0.768907165691)
        assert result2["approx_unique"][0] == 96
        assert result2["approx_unique"][1] == 3
        assert result2["count"][0] == 997
        assert result2["count"][1] == 1234
        assert result2["cvar"][0] == pytest.approx(63.32653061)
        assert result2["cvar"][1] is None
        assert result2["dtype"][0] == "numeric(6,3)"
        assert result2["dtype"][1] == "int"
        assert result2["iqr"][0] == pytest.approx(18)
        assert result2["iqr"][1] == pytest.approx(2)
        assert result2["kurtosis"][0] == pytest.approx(0.1568969133)
        assert result2["kurtosis"][1] == pytest.approx(-1.34962169)
        assert result2["jb"][0] == pytest.approx(28.533863175)
        assert result2["jb"][1] == pytest.approx(163.25695108)
        assert result2["mad"][0] == pytest.approx(8)
        assert result2["mad"][1] == pytest.approx(0)
        assert result2["mean"][0] == pytest.approx(30.1524573721163)
        assert result2["mean"][1] == pytest.approx(2.28444084278768)
        assert result2["median"][0] == 28
        assert result2["median"][1] == 3
        assert result2["mode"][0] is None
        assert result2["mode"][1] == 3
        assert result2["percent"][0] == pytest.approx(80.794)
        assert result2["percent"][1] == pytest.approx(100)
        assert result2["prod"][0] == float("inf")
        assert result2["prod"][1] == float("inf")
        assert result2["range"][0] == pytest.approx(79.670)
        assert result2["range"][1] == 2
        assert result2["sem"][0] == pytest.approx(0.457170684)
        assert result2["sem"][1] == pytest.approx(0.023983078)
        assert result2["skewness"][0] == pytest.approx(0.408876460)
        assert result2["skewness"][1] == pytest.approx(-0.57625856)
        assert result2["sum"][0] == pytest.approx(30062.000)
        assert result2["sum"][1] == 2819
        assert result2["std"][0] == pytest.approx(14.4353046299159)
        assert result2["std"][1] == pytest.approx(0.842485636190292)
        assert result2["top2"][0] == pytest.approx(24.000)
        assert result2["top2"][1] == 1
        assert result2["top2_percent"][0] == pytest.approx(3.566)
        assert result2["top2_percent"][1] == pytest.approx(25.284)
        assert result2["var"][0] == pytest.approx(208.3780197)
        assert result2["var"][1] == pytest.approx(0.709782047)

        # making sure that vDataFrame.aggregate is the same
        result1_1 = titanic_vd.aggregate(
            func=["unique", "top", "min", "10%", "max"], columns=["age"]
        )
        assert result1_1["unique"][0] == result1["unique"][0]
        assert result1_1["top"][0] == result1["top"][0]
        assert result1_1["min"][0] == result1["min"][0]
        assert result1_1["10%"][0] == result1["10%"][0]
        assert result1_1["max"][0] == result1["max"][0]

        result2_2 = titanic_vd.aggregate(
            func=[
                "aad",
                "approx_unique",
                "count",
                "cvar",
                "dtype",
                "iqr",
                "kurtosis",
                "jb",
                "mad",
                "mean",
                "median",
                "mode",
                "percent",
                "prod",
                "range",
                "sem",
                "skewness",
                "sum",
                "std",
                "top2",
                "top2_percent",
                "var",
            ],
            columns=["age"],
        )
        assert result2_2["aad"][0] == result2["aad"][0]
        assert result2_2["approx_unique"][0] == result2["approx_unique"][0]
        assert result2_2["count"][0] == result2["count"][0]
        assert result2_2["cvar"][0] == result2["cvar"][0]
        assert result2_2["dtype"][0] == result2["dtype"][0]
        assert result2_2["iqr"][0] == result2["iqr"][0]
        assert result2_2["kurtosis"][0] == result2["kurtosis"][0]
        assert result2_2["jb"][0] == result2["jb"][0]
        assert result2_2["mad"][0] == result2["mad"][0]
        assert result2_2["mean"][0] == result2["mean"][0]
        assert result2_2["median"][0] == result2["median"][0]
        assert result2_2["mode"][0] == result2["mode"][0]
        assert result2_2["percent"][0] == result2["percent"][0]
        assert result2_2["prod"][0] == result2["prod"][0]
        assert result2_2["range"][0] == result2["range"][0]
        assert result2_2["sem"][0] == result2["sem"][0]
        assert result2_2["skewness"][0] == result2["skewness"][0]
        assert result2_2["sum"][0] == result2["sum"][0]
        assert result2_2["std"][0] == result2["std"][0]
        assert result2_2["top2"][0] == result2["top2"][0]
        assert result2_2["top2_percent"][0] == result2["top2_percent"][0]
        assert result2_2["var"][0] == result2["var"][0]

        # testing vDataFrame[].agg
        result3 = titanic_vd["age"].agg(
            func=[
                "unique",
                "top",
                "min",
                "10%",
                "max",
                "aad",
                "approx_unique",
                "count",
                "cvar",
                "dtype",
                "iqr",
                "kurtosis",
                "jb",
                "mad",
                "mean",
                "median",
                "mode",
                "percent",
                "prod",
                "range",
                "sem",
                "skewness",
                "sum",
                "std",
                "top2",
                "top2_percent",
                "var",
            ]
        )
        assert result3["age"][0] == 96
        assert result3["age"][1] is None
        assert result3["age"][2] == pytest.approx(0.33)
        assert result3["age"][3] == pytest.approx(14.5)
        assert result3["age"][4] == 80
        assert result3["age"][5] == pytest.approx(11.254785419447906)
        assert result3["age"][6] == 96
        assert result3["age"][7] == 997
        assert result3["age"][8] == pytest.approx(63.3265306122449)
        assert result3["age"][9] == "numeric(6,3)"
        assert result3["age"][10] == 18
        assert result3["age"][11] == pytest.approx(0.15689691331997)
        assert result3["age"][12] == pytest.approx(28.5338631758186)
        assert result3["age"][13] == 8
        assert result3["age"][14] == pytest.approx(30.1524573721163)
        assert result3["age"][15] == 28
        assert result3["age"][16] is None
        assert result3["age"][17] == 80.794
        assert result3["age"][18] == float("inf")
        assert result3["age"][19] == pytest.approx(79.67)
        assert result3["age"][20] == pytest.approx(0.457170684605937)
        assert result3["age"][21] == pytest.approx(0.408876460779437)
        assert result3["age"][22] == 30062
        assert result3["age"][23] == pytest.approx(14.4353046299159)
        assert result3["age"][24] == 24
        assert result3["age"][25] == pytest.approx(3.566)
        assert result3["age"][26] == pytest.approx(208.378019758472)

        # testing vDataFrame[].aggregate
        result3_3 = titanic_vd["age"].aggregate(
            func=[
                "unique",
                "top",
                "min",
                "10%",
                "max",
                "aad",
                "approx_unique",
                "count",
                "cvar",
                "dtype",
                "iqr",
                "kurtosis",
                "jb",
                "mad",
                "mean",
                "median",
                "mode",
                "percent",
                "prod",
                "range",
                "sem",
                "skewness",
                "sum",
                "std",
                "top2",
                "top2_percent",
                "var",
            ]
        )
        assert result3_3["age"][0] == result3["age"][0]
        assert result3_3["age"][1] == result3["age"][1]
        assert result3_3["age"][2] == result3["age"][2]
        assert result3_3["age"][3] == result3["age"][3]
        assert result3_3["age"][4] == result3["age"][4]
        assert result3_3["age"][5] == result3["age"][5]
        assert result3_3["age"][6] == result3["age"][6]
        assert result3_3["age"][7] == result3["age"][7]
        assert result3_3["age"][8] == result3["age"][8]
        assert result3_3["age"][9] == result3["age"][9]
        assert result3_3["age"][10] == result3["age"][10]
        assert result3_3["age"][11] == result3["age"][11]
        assert result3_3["age"][12] == result3["age"][12]
        assert result3_3["age"][13] == result3["age"][13]
        assert result3_3["age"][14] == result3["age"][14]
        assert result3_3["age"][15] == result3["age"][15]
        assert result3_3["age"][16] == result3["age"][16]
        assert result3_3["age"][17] == result3["age"][17]
        assert result3_3["age"][18] == result3["age"][18]
        assert result3_3["age"][19] == result3["age"][19]
        assert result3_3["age"][20] == result3["age"][20]
        assert result3_3["age"][21] == result3["age"][21]
        assert result3_3["age"][22] == result3["age"][22]
        assert result3_3["age"][23] == result3["age"][23]
        assert result3_3["age"][24] == result3["age"][24]
        assert result3_3["age"][25] == result3["age"][25]
        assert result3_3["age"][26] == result3["age"][26]

    def test_vDF_all(self, titanic_vd):
        result = titanic_vd.all(columns=["survived"])
        assert result["bool_and"][0] == 0.0

    def test_vDF_any(self, titanic_vd):
        result = titanic_vd.any(columns=["survived"])
        assert result["bool_or"][0] == 1.0

    def test_vDF_avg(self, titanic_vd):
        # tests for vDataFrame.avg()
        result = titanic_vd.avg(columns=["age", "fare", "parch"])
        assert result["avg"][0] == pytest.approx(30.15245737)
        assert result["avg"][1] == pytest.approx(33.96379367)
        assert result["avg"][2] == pytest.approx(0.378444084)

        # there is an expected exception for categorical columns
        from vertica_python.errors import QueryError

        with pytest.raises(QueryError) as exception_info:
            titanic_vd.avg(columns=["embarked"])
        # checking the error message
        assert exception_info.match("Could not convert")

        # tests for vDataFrame.mean()
        result2 = titanic_vd.mean(columns=["age"])
        assert result2["avg"][0] == result["avg"][0]

        # tests for vDataFrame[].avg()
        assert titanic_vd["age"].avg() == result["avg"][0]

        # tests for vDataFrame[].mean()
        assert titanic_vd["age"].mean() == result["avg"][0]

    def test_vDF_count(self, titanic_vd):
        # tests for vDataFrame.count()
        result = titanic_vd.count(desc=False)

        assert result["count"][0] == 118
        assert result["count"][1] == 286
        assert result["count"][2] == 439
        assert result["percent"][0] == pytest.approx(9.562)
        assert result["percent"][1] == pytest.approx(23.177)
        assert result["percent"][2] == pytest.approx(35.575)

        # tests for vDataFrame[].count()
        assert titanic_vd["age"].count() == 997

        # there is an expected exception for non-existant columns
        with pytest.raises(AttributeError) as exception_info:
            titanic_vd["haha"].count()
        # checking the error message
        assert exception_info.match("'vDataFrame' object has no attribute 'haha'")

    def test_vDF_describe(self, titanic_vd):
        # testing vDataFrame.describe()
        result1 = titanic_vd.describe(method="all").transpose()

        assert result1["count"][0] == 1234
        assert result1["unique"][0] == 3
        assert result1["top"][0] == 3
        assert result1["top_percent"][0] == pytest.approx(53.728)
        assert result1["avg"][0] == pytest.approx(2.284440842)
        assert result1["stddev"][0] == pytest.approx(0.842485636)
        assert result1["min"][0] == 1
        assert result1["25%"][0] == pytest.approx(1.0)
        assert result1["50%"][0] == pytest.approx(3.0)
        assert result1["75%"][0] == pytest.approx(3.0)
        assert result1["max"][0] == pytest.approx(3)
        assert result1["range"][0] == 2
        assert result1["empty"][0] is None

        assert result1["count"][5] == 1233
        assert result1["unique"][5] == 277
        assert result1["top"][5] == 8.05
        assert result1["top_percent"][5] == pytest.approx(4.7)
        assert result1["avg"][5] == pytest.approx(33.9637936)
        assert result1["stddev"][5] == pytest.approx(52.646072)
        assert result1["min"][5] == 0
        assert result1["25%"][5] == pytest.approx(7.8958)
        assert result1["50%"][5] == pytest.approx(14.4542)
        assert result1["75%"][5] == pytest.approx(31.3875)
        assert result1["max"][5] == pytest.approx(512.32920)
        assert result1["range"][5] == pytest.approx(512.32920)
        assert result1["empty"][5] is None

        result2 = titanic_vd.describe(method="categorical")

        assert result2["dtype"][7] == "varchar(36)"
        assert result2["unique"][7] == 887
        assert result2["count"][7] == 1234
        assert result2["top"][7] == "CA. 2343"
        assert result2["top_percent"][7] == pytest.approx(0.81)

        result3 = titanic_vd.describe(method="length")

        assert result3["dtype"][9] == "varchar(30)"
        assert result3["percent"][9] == pytest.approx(23.177)
        assert result3["count"][9] == 286
        assert result3["unique"][9] == 182
        assert result3["empty"][9] == 0
        assert result3["avg_length"][9] == pytest.approx(3.72027972)
        assert result3["stddev_length"][9] == pytest.approx(2.28313602)
        assert result3["min_length"][9] == 1
        assert result3["25%_length"][9] == 3
        assert result3["50%_length"][9] == 3
        assert result3["75%_length"][9] == 3
        assert result3["max_length"][9] == 15

        result4 = titanic_vd.describe(method="numerical")

        assert result4["count"][1] == 1234
        assert result4["mean"][1] == pytest.approx(0.36466774)
        assert result4["std"][1] == pytest.approx(0.48153201)
        assert result4["min"][1] == 0
        assert result4["25%"][1] == 0
        assert result4["50%"][1] == 0
        assert result4["75%"][1] == 1
        assert result4["max"][1] == 1
        assert result4["unique"][1] == 2.0

        result4_1 = titanic_vd.describe(method="numerical", ncols_block=2)

        assert result4_1["count"][1] == 1234
        assert result4_1["mean"][1] == pytest.approx(0.36466774)
        assert result4_1["std"][1] == pytest.approx(0.48153201)
        assert result4_1["min"][1] == 0
        assert result4_1["25%"][1] == 0
        assert result4_1["50%"][1] == 0
        assert result4_1["75%"][1] == 1
        assert result4_1["max"][1] == 1
        assert result4_1["unique"][1] == 2.0

        result5 = titanic_vd.describe(method="range")

        assert result5["dtype"][2] == "numeric(6,3)"
        assert result5["percent"][2] == pytest.approx(80.794)
        assert result5["count"][2] == 997
        assert result5["unique"][2] == 96
        assert result5["min"][2] == pytest.approx(0.33)
        assert result5["max"][2] == 80
        assert result5["range"][2] == pytest.approx(79.67)

        result6 = titanic_vd.describe(method="statistics")

        assert result6["dtype"][3] == "int"
        assert result6["percent"][3] == 100
        assert result6["count"][3] == 1234
        assert result6["unique"][3] == 7
        assert result6["avg"][3] == pytest.approx(0.504051863857374)
        assert result6["stddev"][3] == pytest.approx(1.04111727241629)
        assert result6["min"][3] == 0
        assert result6["1%"][3] == pytest.approx(0.0)
        assert result6["10%"][3] == pytest.approx(0.0)
        assert result6["25%"][3] == 0
        assert result6["median"][3] == 0
        assert result6["75%"][3] == 1
        assert result6["90%"][3] == 1.0
        assert result6["99%"][3] == pytest.approx(5.0)
        assert result6["max"][3] == 8
        assert result6["skewness"][3] == pytest.approx(3.7597831)
        assert result6["kurtosis"][3] == pytest.approx(19.21388533)

    def test_vDF_describe_index(self, market_vd):
        # testing vDataFrame[].describe
        result1 = market_vd["Form"].describe(method="categorical", max_cardinality=3)

        assert result1["value"][0] == '"Form"'
        assert result1["value"][1] == "varchar(32)"
        assert result1["value"][2] == 37.0
        assert result1["value"][3] == 314.0
        assert result1["value"][4] == 90
        assert result1["value"][5] == 90
        assert result1["value"][6] == 57
        assert result1["value"][7] == 47

        result2 = market_vd["Price"].describe(method="numerical")

        assert result2["value"][0] == '"Price"'
        assert result2["value"][1] == "float"
        assert result2["value"][2] == 308.0
        assert result2["value"][3] == 314
        assert result2["value"][4] == pytest.approx(2.07751098)
        assert result2["value"][5] == pytest.approx(1.51037749)
        assert result2["value"][6] == pytest.approx(0.31663877)
        assert result2["value"][7] == pytest.approx(1.07276187)
        assert result2["value"][8] == pytest.approx(1.56689808)
        assert result2["value"][9] == pytest.approx(2.60376599)
        assert result2["value"][10] == pytest.approx(10.163712)

        result3 = market_vd["Form"].describe(method="cat_stats", numcol="Price")

        assert result3["count"][3] == 2
        assert result3["percent"][3] == pytest.approx(0.63694267515)
        assert result3["mean"][3] == pytest.approx(4.6364768)
        assert result3["std"][3] == pytest.approx(0.6358942)
        assert result3["min"][3] == pytest.approx(4.1868317)
        assert result3["10%"][3] == pytest.approx(4.2767607)
        assert result3["25%"][3] == pytest.approx(4.4116542)
        assert result3["50%"][3] == pytest.approx(4.6364768)
        assert result3["75%"][3] == pytest.approx(4.8612994)
        assert result3["90%"][3] == pytest.approx(4.9961929)
        assert result3["max"][3] == pytest.approx(5.0861220)

    def test_vDF_distinct(self, amazon_vd):
        result = amazon_vd["state"].distinct()
        result.sort()
        assert result == [
            "ACRE",
            "ALAGOAS",
            "AMAPÁ",
            "AMAZONAS",
            "BAHIA",
            "CEARÁ",
            "DISTRITO FEDERAL",
            "ESPÍRITO SANTO",
            "GOIÁS",
            "MARANHÃO",
            "MATO GROSSO",
            "MATO GROSSO DO SUL",
            "MINAS GERAIS",
            "PARANÁ",
            "PARAÍBA",
            "PARÁ",
            "PERNAMBUCO",
            "PIAUÍ",
            "RIO DE JANEIRO",
            "RIO GRANDE DO NORTE",
            "RIO GRANDE DO SUL",
            "RONDÔNIA",
            "RORAIMA",
            "SANTA CATARINA",
            "SERGIPE",
            "SÃO PAULO",
            "TOCANTINS",
        ]

    def test_vDF_duplicated(self, market_vd):
        result = market_vd.duplicated(columns=["Form", "Name"])

        assert result.count == 151
        assert len(result.values) == 3

    def test_vDF_kurt(self, titanic_vd):
        # testing vDataFrame.kurt
        result1 = titanic_vd.kurt(columns=["age", "fare", "parch"])

        assert result1["kurtosis"][0] == pytest.approx(0.15689691)
        assert result1["kurtosis"][1] == pytest.approx(26.2543152)
        assert result1["kurtosis"][2] == pytest.approx(22.6438022)

        # testing vDataFrame.kurtosis
        result2 = titanic_vd.kurtosis(columns=["age", "fare", "parch"])

        assert result2["kurtosis"][0] == result1["kurtosis"][0]
        assert result2["kurtosis"][1] == result1["kurtosis"][1]
        assert result2["kurtosis"][2] == result1["kurtosis"][2]

        # testing vDataFrame[].kurt
        assert titanic_vd["age"].kurt() == result1["kurtosis"][0]
        assert titanic_vd["fare"].kurt() == result1["kurtosis"][1]
        assert titanic_vd["parch"].kurt() == result1["kurtosis"][2]

        # testing vDataFrame[].kurtosis
        assert titanic_vd["age"].kurtosis() == result1["kurtosis"][0]
        assert titanic_vd["fare"].kurtosis() == result1["kurtosis"][1]
        assert titanic_vd["parch"].kurtosis() == result1["kurtosis"][2]

    def test_vDF_mad(self, titanic_vd):
        # testing vDataFrame.mad
        result1 = titanic_vd.mad(columns=["age", "fare", "parch"])

        assert result1["mad"][0] == pytest.approx(8.0)
        assert result1["mad"][1] == pytest.approx(6.9042)
        assert result1["mad"][2] == pytest.approx(0.0)

        # testing vDataFrame[].mad
        assert titanic_vd["age"].mad() == result1["mad"][0]
        assert titanic_vd["fare"].mad() == result1["mad"][1]
        assert titanic_vd["parch"].mad() == result1["mad"][2]

    def test_vDF_max(self, titanic_vd):
        # testing vDataFrame.max
        result1 = titanic_vd.max(columns=["age", "fare", "parch"])

        assert result1["max"][0] == pytest.approx(80.0)
        assert result1["max"][1] == pytest.approx(512.3292)
        assert result1["max"][2] == pytest.approx(9.0)

        # testing vDataFrame[].max
        assert titanic_vd["age"].max() == result1["max"][0]
        assert titanic_vd["fare"].max() == result1["max"][1]
        assert titanic_vd["parch"].max() == result1["max"][2]

    def test_vDF_median(self, titanic_vd):
        # testing vDataFrame.median
        result = titanic_vd.median(columns=["age", "fare", "parch"])

        assert result["median"][0] == pytest.approx(28.0)
        assert result["median"][1] == pytest.approx(14.4542)
        assert result["median"][2] == pytest.approx(0.0)

        # testing vDataFrame[].median
        assert titanic_vd["age"].median() == result["median"][0]
        assert titanic_vd["fare"].median() == result["median"][1]
        assert titanic_vd["parch"].median() == result["median"][2]

    def test_vDF_min(self, titanic_vd):
        # testing vDataFrame.min
        result = titanic_vd.min(columns=["age", "fare", "parch"])

        assert result["min"][0] == pytest.approx(0.33)
        assert result["min"][1] == pytest.approx(0.0)
        assert result["min"][2] == pytest.approx(0.0)

        # testing vDataFrame[].median
        assert titanic_vd["age"].min() == result["min"][0]
        assert titanic_vd["fare"].min() == result["min"][1]
        assert titanic_vd["parch"].min() == result["min"][2]

    def test_vDF_mode(self, market_vd):
        # testing vDataFrame[].mod
        assert market_vd["Name"].mode() == "Pineapple"
        assert market_vd["Name"].mode(n=2) == "Carrots"

    def test_vDF_nlargest(self, market_vd):
        result = market_vd["Price"].nlargest(n=2)

        assert result["Name"][0] == "Mangoes"
        assert result["Form"][0] == "Dried"
        assert result["Price"][0] == pytest.approx(10.1637125)
        assert result["Name"][1] == "Mangoes"
        assert result["Form"][1] == "Dried"
        assert result["Price"][1] == pytest.approx(8.50464930)

    def test_vDF_nsmallest(self, market_vd):
        result = market_vd["Price"].nsmallest(n=2)

        assert result["Name"][0] == "Watermelon"
        assert result["Form"][0] == "Fresh"
        assert result["Price"][0] == pytest.approx(0.31663877)
        assert result["Name"][1] == "Watermelon"
        assert result["Form"][1] == "Fresh"
        assert result["Price"][1] == pytest.approx(0.33341203)

    def test_vDF_nunique(self, titanic_vd):
        result = titanic_vd.nunique(columns=["pclass", "embarked", "survived", "cabin"])

        assert result["unique"][0] == 3.0
        assert result["unique"][1] == 3.0
        assert result["unique"][2] == 2.0
        assert result["unique"][3] == 182.0

    def test_vDF_numh(self, market_vd, amazon_vd):
        assert market_vd["Price"].numh(method="auto") == pytest.approx(0.984707376)
        assert market_vd["Price"].numh(method="freedman_diaconis") == pytest.approx(
            0.450501738
        )
        assert market_vd["Price"].numh(method="sturges") == pytest.approx(0.984707376)
        assert amazon_vd["date"].numh(method="auto") == pytest.approx(44705828.571428575)
        assert amazon_vd["date"].numh(method="freedman_diaconis") == pytest.approx(
            33903959.714834176
        )
        assert amazon_vd["date"].numh(method="sturges") == pytest.approx(44705828.571428575)


    def test_vDF_prod(self, market_vd):
        # testing vDataFrame.prod
        result1 = market_vd.prod(columns=["Price"])

        assert result1["prod"][0] == pytest.approx(1.9205016913e71)

        # testing vDataFrame.product
        result2 = market_vd.product(columns=["Price"])

        assert result2["prod"][0] == result1["prod"][0]

        # testing vDataFrame[].prod
        assert market_vd["price"].prod() == result1["prod"][0]

        # testing vDataFrame[].product
        assert market_vd["price"].product() == result1["prod"][0]

    def test_vDF_quantile(self, titanic_vd):
        # testing vDataFrame.quantile
        result = titanic_vd.quantile(q=[0.22, 0.9], columns=["age", "fare"])

        assert result["22.0%"][0] == pytest.approx(20.0)
        assert result["90.0%"][0] == pytest.approx(50.0)
        assert result["22.0%"][1] == pytest.approx(7.8958)
        assert result["90.0%"][1] == pytest.approx(79.13)

        # testing vDataFrame[].quantile
        assert titanic_vd["age"].quantile(x=0.5) == pytest.approx(28.0)
        assert titanic_vd["fare"].quantile(x=0.1) == pytest.approx(7.5892)

    def test_vDF_score(self, base, titanic_vd):
        from verticapy.learn.linear_model import LogisticRegression

        model = LogisticRegression(
            name="public.LR_titanic",
            cursor=base.cursor,
            tol=1e-4,
            C=1.0,
            max_iter=100,
            solver="CGD",
            penalty="ENet",
            l1_ratio=0.5,
        )

        model.drop()  # dropping the model in case of its existance
        model.fit("public.titanic", ["fare", "age"], "survived")
        model.predict(titanic_vd, name="survived_pred")

        # Computing AUC
        auc = titanic_vd.score(y_true="survived", y_score="survived_pred", method="auc")
        assert auc == pytest.approx(0.7051784997146537)

        # Computing MSE
        mse = titanic_vd.score(y_true="survived", y_score="survived_pred", method="mse")
        assert mse == pytest.approx(0.228082579110535)

        # Drawing ROC Curve
        roc_res = titanic_vd.score(
            y_true="survived", y_score="survived_pred", method="roc", nbins=1000,
        )
        assert roc_res["threshold"][3] == 0.003
        assert roc_res["false_positive"][3] == 1.0
        assert roc_res["true_positive"][3] == 1.0
        assert roc_res["threshold"][300] == 0.3
        assert roc_res["false_positive"][300] == pytest.approx(1.0)
        assert roc_res["true_positive"][300] == pytest.approx(1.0)
        assert roc_res["threshold"][900] == 0.9
        assert roc_res["false_positive"][900] == pytest.approx(0.0148760330578512)
        assert roc_res["true_positive"][900] == pytest.approx(0.061381074168798)

        # Drawing PRC Curve
        prc_res = titanic_vd.score(
            y_true="survived", y_score="survived_pred", method="prc", nbins=1000,
        )
        assert prc_res["threshold"][3] == 0.002
        assert prc_res["recall"][3] == 1.0
        assert prc_res["precision"][3] == pytest.approx(0.3925702811)
        assert prc_res["threshold"][300] == 0.299
        assert prc_res["recall"][300] == pytest.approx(1.0)
        assert prc_res["precision"][300] == pytest.approx(0.392570281124498)
        assert prc_res["threshold"][900] == 0.899
        assert prc_res["recall"][900] == pytest.approx(0.061381074168798)
        assert prc_res["precision"][900] == pytest.approx(0.727272727272727)

        # dropping the created model
        model.drop()

    def test_vDF_sem(self, titanic_vd):
        # testing vDataFrame.sem
        result = titanic_vd.sem(columns=["age", "fare"])

        assert result["sem"][0] == pytest.approx(0.457170684)
        assert result["sem"][1] == pytest.approx(1.499285853)

        # testing vDataFrame[].sem
        assert titanic_vd["parch"].sem() == pytest.approx(0.024726611)

    def test_vDF_shape(self, market_vd):
        assert market_vd.shape() == (314, 3)

    def test_vDF_skew(self, titanic_vd):
        # testing vDataFrame.skew
        result1 = titanic_vd.skew(columns=["age", "fare"])

        assert result1["skewness"][0] == pytest.approx(0.408876460)
        assert result1["skewness"][1] == pytest.approx(4.300699188)

        # testing vDataFrame.skewness
        result2 = titanic_vd.skewness(columns=["age", "fare"])

        assert result2["skewness"][0] == result1["skewness"][0]
        assert result2["skewness"][1] == result1["skewness"][1]

        # testing vDataFrame[].skew
        assert titanic_vd["parch"].skew() == pytest.approx(3.798019282)
        # testing vDataFrame[].skewness
        assert titanic_vd["parch"].skewness() == pytest.approx(3.798019282)

    def test_vDF_std(self, titanic_vd):
        # testing vDataFrame.std
        result = titanic_vd.std(columns=["fare"])

        assert result["stddev"][0] == pytest.approx(52.64607298)

        # testing vDataFrame[].std
        assert titanic_vd["parch"].std() == pytest.approx(0.868604707)

    def test_vDF_sum(self, titanic_vd):
        # testing vDataFrame.sum
        result = titanic_vd.sum(columns=["fare", "parch"])

        assert result["sum"][0] == pytest.approx(41877.3576)
        assert result["sum"][1] == pytest.approx(467.0)

        # testing vDataFrame[].sum
        assert titanic_vd["age"].sum() == pytest.approx(30062.0)

    def test_vDF_topk(self, market_vd):
        result = market_vd["Name"].topk(k=3)

        assert result["count"][0] == 12
        assert result["percent"][0] == pytest.approx(3.822)
        assert result["count"][1] == 10
        assert result["percent"][1] == pytest.approx(3.185)
        assert result["count"][2] == 8
        assert result["percent"][2] == pytest.approx(2.548)

    def test_vDF_value_counts(self, market_vd):
        result = market_vd["Name"].value_counts(k=2)

        assert result["value"][0] == '"Name"'
        assert result["value"][1] == "varchar(32)"
        assert result["value"][2] == 73.0
        assert result["value"][3] == 314.0
        assert result["value"][4] == 284
        assert result["value"][5] == 12
        assert result["value"][6] == 10
        assert result["index"][6] == "Carrots"

    def test_vDF_var(self, titanic_vd):
        # testing vDataFrame.var
        result = titanic_vd.var(columns=["age", "parch"])

        assert result["variance"][0] == pytest.approx(208.3780197)
        assert result["variance"][1] == pytest.approx(0.754474138)

        # testing vDataFrame[].var
        assert titanic_vd["fare"].var() == pytest.approx(2771.6090005)
