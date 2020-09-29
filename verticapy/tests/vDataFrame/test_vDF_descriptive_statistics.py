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
from verticapy import vDataFrame
from verticapy import drop_table
from decimal import Decimal 


@pytest.fixture(scope="module")
def titanic_vd(base):
    from verticapy.learn.datasets import load_titanic

    titanic = load_titanic(cursor=base.cursor)
    yield titanic
    drop_table(name = "public.titanic", cursor = base.cursor)

@pytest.fixture(scope="module")
def market_vd(base):
    from verticapy.learn.datasets import load_market

    market = load_market(cursor=base.cursor)
    yield market
    drop_table(name = "public.market", cursor = base.cursor)

@pytest.fixture(scope="module")
def amazon_vd(base):
    from verticapy.learn.datasets import load_amazon

    amazon = load_amazon(cursor=base.cursor)
    yield amazon
    drop_table(name = "public.amazon", cursor = base.cursor)


class TestvDFDescriptiveStat:
    def test_vDF_aad(self, titanic_vd):
        # testing vDataFrame.aad
        result = titanic_vd.aad(columns=["age", "fare", "parch"])
        assert result.values["aad"][0] == pytest.approx(11.2547854194)
        assert result.values["aad"][1] == pytest.approx(30.6258659424)
        assert result.values["aad"][2] == pytest.approx(0.58208012314)

        # testing vDataFrame[].aad
        assert titanic_vd["age"].aad() == result.values["aad"][0]
        assert titanic_vd["fare"].aad() == result.values["aad"][1]
        assert titanic_vd["parch"].aad() == result.values["aad"][2]

    def test_vDF_agg(self, titanic_vd):
        # testing vDataFrame.agg
        result1 = titanic_vd.agg(func = ["unique", "top", "min", "10%", "50%", "90%", "max"],
                                 columns = ["age", "fare", "pclass", "survived"])
        assert result1.values["unique"][0] == 96
        assert result1.values["unique"][1] == 277
        assert result1.values["unique"][2] == 3
        assert result1.values["unique"][3] == 2
        assert result1.values["top"][0] is None
        assert result1.values["top"][1] == pytest.approx(8.05)
        assert result1.values["top"][2] == 3
        assert result1.values["top"][3] == 0
        assert result1.values["min"][0] == Decimal('0.330') # Why does it need to have Decimal?
        assert result1.values["min"][1] == 0
        assert result1.values["min"][2] == 1
        assert result1.values["min"][3] == 0
        assert result1.values["10%"][0] == pytest.approx(14.5)
        assert result1.values["10%"][1] == pytest.approx(7.5892)
        assert result1.values["10%"][2] == 1
        assert result1.values["10%"][3] == 0
        assert result1.values["50%"][0] == 28
        assert result1.values["50%"][1] == pytest.approx(14.4542)
        assert result1.values["50%"][2] == 3
        assert result1.values["50%"][3] == 0
        assert result1.values["90%"][0] == 50
        assert result1.values["90%"][1] == pytest.approx(79.13)
        assert result1.values["90%"][2] == 3
        assert result1.values["90%"][3] == 1
        assert result1.values["max"][0] == 80
        assert result1.values["max"][1] == pytest.approx(Decimal(512.3292)) # Why Decimal?
        assert result1.values["max"][2] == 3
        assert result1.values["max"][3] == 1

        result2 = titanic_vd.agg(func = ["aad", "approx_unique", "count", "cvar", "dtype", "iqr",
                                         "kurtosis", "jb", "mad", "mean", "median", "mode",
                                         "percent", "prod", "range", "sem", "skewness", "sum",
                                         "std", "top2", "top2_percent", "var"],
                                 columns = ["age", "pclass"])
        assert result2.values["aad"][0] == '11.254785419447906' # Why string?
        assert result2.values["aad"][1] == pytest.approx(Decimal('0.768907165691')) # Why string-Decimal?
        assert result2.values["approx_unique"][0] == '96' # Why string?
        assert result2.values["approx_unique"][1] == '3' # Why string?
        assert result2.values["count"][0] == 997
        assert result2.values["count"][1] == '1234' # Why string?
        assert result2.values["cvar"][0] == pytest.approx(63.32653061)
        assert result2.values["cvar"][1] is None
        assert result2.values["dtype"][0] == 'numeric(6,3)'
        assert result2.values["dtype"][1] == 'int'
        assert result2.values["iqr"][0] == pytest.approx(18)
        assert result2.values["iqr"][1] == pytest.approx(2)
        assert result2.values["kurtosis"][0] == pytest.approx(0.1568969133)
        assert result2.values["kurtosis"][1] == pytest.approx(-1.34962169)
        assert result2.values["jb"][0] == pytest.approx(28.533863175)
        assert result2.values["jb"][1] == pytest.approx(163.25695108)
        assert result2.values["mad"][0] == pytest.approx(8)
        assert result2.values["mad"][1] == pytest.approx(0)
        assert result2.values["mean"][0] == '30.1524573721163' # Why string?
        assert result2.values["mean"][1] == '2.28444084278768' # Why string?
        assert result2.values["median"][0] == '28' # Why string?
        assert result2.values["median"][1] == '3' # Why string?
        assert result2.values["mode"][0] is None
        assert result2.values["mode"][1] == '3'
        assert result2.values["percent"][0] == pytest.approx(80.794)
        assert result2.values["percent"][1] == pytest.approx(100)
        assert result2.values["prod"][0] == float('inf')
        assert result2.values["prod"][1] == float('inf')
        assert result2.values["range"][0] == Decimal('79.670') # Why Decimal?
        assert result2.values["range"][1] == 2
        assert result2.values["sem"][0] == pytest.approx(0.457170684)
        assert result2.values["sem"][1] == pytest.approx(0.023983078)
        assert result2.values["skewness"][0] == pytest.approx(0.408876460)
        assert result2.values["skewness"][1] == pytest.approx(-0.57625856)
        assert result2.values["sum"][0] == Decimal('30062.000') # Why Decimal?
        assert result2.values["sum"][1] == 2819
        assert result2.values["std"][0] == '14.4353046299159' # Why string?
        assert result2.values["std"][1] == '0.842485636190292' # Why string?
        assert result2.values["top2"][0] == '24.000' # Why string?
        assert result2.values["top2"][1] == '1' # Why string?
        assert result2.values["top2_percent"][0] == Decimal('3.566') # Why Decimal?
        assert result2.values["top2_percent"][1] == Decimal('25.284') # Why Decimal?
        assert result2.values["var"][0] == pytest.approx(208.3780197)
        assert result2.values["var"][1] == pytest.approx(0.709782047)

        # making sure that vDataFrame.aggregate is the same
        result1_1 = titanic_vd.aggregate(func = ["unique", "top", "min", "10%", "max"],
                                         columns = ["age"])
        #assert result1_1.values["unique"][0] == result1.values["unique"][0] # NOT THE SAME!!!
        assert result1_1.values["top"][0] == result1.values["top"][0]
        #assert result1_1.values["min"][0] == result1.values["min"][0] # NOT THE SAME!!!
        #assert result1_1.values["10%"][0] == result1.values["10%"][0] # NOT THE SAME!!!
        #assert result1_1.values["max"][0] == result1.values["max"][0] # NOT THE SAME!!!

        result2_2 = titanic_vd.aggregate(func = ["aad", "approx_unique", "count", "cvar", "dtype", "iqr",
                                                 "kurtosis", "jb", "mad", "mean", "median", "mode",
                                                 "percent", "prod", "range", "sem", "skewness", "sum",
                                                 "std", "top2", "top2_percent", "var"],
                                         columns = ["age"])
        assert result2_2.values["aad"][0] == result2.values["aad"][0]
        assert result2_2.values["approx_unique"][0] == result2.values["approx_unique"][0]
        #assert result2_2.values["count"][0] == result2.values["count"][0] # NOT THE SAME!!!
        #assert result2_2.values["cvar"][0] == result2.values["cvar"][0] # NOT THE SAME!!!
        assert result2_2.values["dtype"][0] == result2.values["dtype"][0]
        #assert result2_2.values["iqr"][0] == result2.values["iqr"][0] # NOT THE SAME!!!
        #assert result2_2.values["kurtosis"][0] == result2.values["kurtosis"][0] # NOT THE SAME!!!
        #assert result2_2.values["jb"][0] == result2.values["jb"][0]
        #assert result2_2.values["mad"][0] == result2.values["mad"][0]
        assert result2_2.values["mean"][0] == result2.values["mean"][0]
        assert result2_2.values["median"][0] == result2.values["median"][0]
        assert result2_2.values["mode"][0] == result2.values["mode"][0]
        #assert result2_2.values["percent"][0] == result2.values["percent"][0]
        #assert result2_2.values["prod"][0] == result2.values["prod"][0]
        #assert result2_2.values["range"][0] == result2.values["range"][0]
        #assert result2_2.values["sem"][0] == result2.values["sem"][0]
        #assert result2_2.values["skewness"][0] == result2.values["skewness"][0]
        #assert result2_2.values["sum"][0] == result2.values["sum"][0]
        #assert result2_2.values["std"][0] == result2.values["std"][0]
        #assert result2_2.values["top2"][0] == result2.values["top2"][0]
        #assert result2_2.values["top2_percent"][0] == result2.values["top2_percent"][0]
        #assert result2_2.values["var"][0] == result2.values["var"][0]

        # testing vDataFrame[].agg
        result3 = titanic_vd["age"].agg(func = ["unique", "top", "min", "10%", "max",
                                                "aad", "approx_unique", "count", "cvar", "dtype", "iqr",
                                                "kurtosis", "jb", "mad", "mean", "median", "mode",
                                                "percent", "prod", "range", "sem", "skewness", "sum",
                                                "std", "top2", "top2_percent", "var"])
        # It is NOT nice that it requires '"age"' as keyword instead of "age"
        assert result3.values['"age"'][0] == '96'
        assert result3.values['"age"'][1] is None
        assert result3.values['"age"'][2] == '0.33'
        assert result3.values['"age"'][3] == '14.5'
        assert result3.values['"age"'][4] == '80'
        assert result3.values['"age"'][5] == '11.254785419447906' # Why string?
        assert result3.values['"age"'][6] == '96'
        assert result3.values['"age"'][7] == '997'
        assert result3.values['"age"'][8] == '63.3265306122449'
        assert result3.values['"age"'][9] == 'numeric(6,3)'
        assert result3.values['"age"'][10] == '18'
        assert result3.values['"age"'][11] == '0.15689691331997'
        assert result3.values['"age"'][12] == '28.5338631758186'
        assert result3.values['"age"'][13] == '8'
        assert result3.values['"age"'][14] == '30.1524573721163' # Why string?
        assert result3.values['"age"'][15] == '28' 
        assert result3.values['"age"'][16] is None
        assert result3.values['"age"'][17] == '80.794'
        assert result3.values['"age"'][18] == 'inf'
        assert result3.values['"age"'][19] == '79.67'
        assert result3.values['"age"'][20] == '0.457170684605937'
        assert result3.values['"age"'][21] == '0.408876460779437'
        assert result3.values['"age"'][22] == '30062'
        assert result3.values['"age"'][23] == '14.4353046299159'
        assert result3.values['"age"'][24] == '24' # Why string?
        assert result3.values['"age"'][25] == '3.566'
        assert result3.values['"age"'][26] == '208.378019758472'

        # testing vDataFrame[].aggregate
        result3_3 = titanic_vd["age"].aggregate(func = ["unique", "top", "min", "10%", "max",
                                                        "aad", "approx_unique", "count", "cvar", "dtype", "iqr",
                                                        "kurtosis", "jb", "mad", "mean", "median", "mode",
                                                        "percent", "prod", "range", "sem", "skewness", "sum",
                                                        "std", "top2", "top2_percent", "var"])
        assert result3_3.values['"age"'][0] == result3.values['"age"'][0]
        assert result3_3.values['"age"'][1] == result3.values['"age"'][1]
        assert result3_3.values['"age"'][2] == result3.values['"age"'][2]
        assert result3_3.values['"age"'][3] == result3.values['"age"'][3]
        assert result3_3.values['"age"'][4] == result3.values['"age"'][4]
        assert result3_3.values['"age"'][5] == result3.values['"age"'][5]
        assert result3_3.values['"age"'][6] == result3.values['"age"'][6]
        assert result3_3.values['"age"'][7] == result3.values['"age"'][7]
        assert result3_3.values['"age"'][8] == result3.values['"age"'][8]
        assert result3_3.values['"age"'][9] == result3.values['"age"'][9]
        assert result3_3.values['"age"'][10] == result3.values['"age"'][10]
        assert result3_3.values['"age"'][11] == result3.values['"age"'][11]
        assert result3_3.values['"age"'][12] == result3.values['"age"'][12]
        assert result3_3.values['"age"'][13] == result3.values['"age"'][13]
        assert result3_3.values['"age"'][14] == result3.values['"age"'][14]
        assert result3_3.values['"age"'][15] == result3.values['"age"'][15]
        assert result3_3.values['"age"'][16] == result3.values['"age"'][16]
        assert result3_3.values['"age"'][17] == result3.values['"age"'][17]
        assert result3_3.values['"age"'][18] == result3.values['"age"'][18]
        assert result3_3.values['"age"'][19] == result3.values['"age"'][19]
        assert result3_3.values['"age"'][20] == result3.values['"age"'][20]
        assert result3_3.values['"age"'][21] == result3.values['"age"'][21]
        assert result3_3.values['"age"'][22] == result3.values['"age"'][22]
        assert result3_3.values['"age"'][23] == result3.values['"age"'][23]
        assert result3_3.values['"age"'][24] == result3.values['"age"'][24]
        assert result3_3.values['"age"'][25] == result3.values['"age"'][25]
        assert result3_3.values['"age"'][26] == result3.values['"age"'][26]

    def test_vDF_all(self, titanic_vd):
        result = titanic_vd.all(columns = ["survived"])
        assert result.values["bool_and"][0] == 0.0

    def test_vDF_any(self, titanic_vd):
        result = titanic_vd.any(columns = ["survived"])
        assert result.values["bool_or"][0] == 1.0

    def test_vDF_avg(self, titanic_vd):
        # tests for vDataFrame.avg()
        result = titanic_vd.avg(columns = ["age", "fare", "parch"])
        assert result.values["avg"][0] == pytest.approx(30.15245737)
        assert result.values["avg"][1] == pytest.approx(33.96379367)
        assert result.values["avg"][2] == pytest.approx(0.378444084)

        # there is an expected exception for categorical columns
        from vertica_python.errors import QueryError
        with pytest.raises(QueryError) as exception_info:
            titanic_vd.avg(columns = ["embarked"])
        # checking the error message
        assert exception_info.match("Could not convert \"C\" from column titanic.embarked to a float8")

        # tests for vDataFrame.mean()
        result2 = titanic_vd.mean(columns = ["age"])
        assert result2.values['avg'][0] == result.values["avg"][0]

        # tests for vDataFrame[].avg()
        assert titanic_vd["age"].avg() == result.values["avg"][0]

        # tests for vDataFrame[].mean()
        assert titanic_vd["age"].mean() == result.values["avg"][0]

    def test_vDF_count(self, titanic_vd):
        # tests for vDataFrame.count()
        result = titanic_vd.count(desc = False)

        assert result.values["count"][0] == 118
        assert result.values["count"][1] == 286
        assert result.values["count"][2] == 439
        assert result.values["percent"][0] == pytest.approx(9.562)
        assert result.values["percent"][1] == pytest.approx(23.177)
        assert result.values["percent"][2] == pytest.approx(35.575)

        # tests for vDataFrame[].count()
        assert titanic_vd["age"].count() == 997

        # there is an expected exception for non-existant columns
        with pytest.raises(AttributeError) as exception_info:
            titanic_vd["haha"].count()
        # checking the error message
        assert exception_info.match("'vDataFrame' object has no attribute 'haha'")

    def test_vDF_describe(self, titanic_vd):
        # testing vDataFrame.describe()
        result1 = titanic_vd.describe(method = "all")

        assert result1.values["count"][0] == 1234
        assert result1.values["unique"][0] == 3
        assert result1.values["top"][0] == 3
        assert result1.values["top_percent"][0] == Decimal('53.728') # Why this format?
        assert result1.values["avg"][0] == pytest.approx(2.284440842)
        assert result1.values["stddev"][0] == pytest.approx(0.842485636)
        assert result1.values["min"][0] == 1
        assert result1.values["25%"][0] == pytest.approx(1.0)
        assert result1.values["50%"][0] == pytest.approx(3.0)
        assert result1.values["75%"][0] == pytest.approx(3.0)
        assert result1.values["max"][0] == pytest.approx(3)
        assert result1.values["range"][0] == 2
        assert result1.values["empty"][0] is None

        assert result1.values["count"][5] == 1233
        assert result1.values["unique"][5] == 277
        assert result1.values["top"][5] == 8.05
        assert result1.values["top_percent"][5] == Decimal('4.7') # Why this format?
        assert result1.values["avg"][5] == pytest.approx(33.9637936)
        assert result1.values["stddev"][5] == pytest.approx(52.646072)
        assert result1.values["min"][5] == 0
        assert result1.values["25%"][5] == pytest.approx(7.8958)
        assert result1.values["50%"][5] == pytest.approx(14.4542)
        assert result1.values["75%"][5] == pytest.approx(31.3875)
        assert result1.values["max"][5] == pytest.approx(512.32920)
        assert result1.values["range"][5] == Decimal('512.32920') # why this format
        assert result1.values["empty"][5] is None

        result2 = titanic_vd.describe(method = "categorical")

        assert result2.values["dtype"][7] == 'varchar(36)'
        assert result2.values["unique"][7] == '887' # Why string?
        assert result2.values["count"][7] == '1234' # Why string?
        assert result2.values["top"][7] == 'CA. 2343'
        assert result2.values["top_percent"][7] == '0.81' # Why this format?

        result3 = titanic_vd.describe(method = "length")

        assert result3.values["dtype"][9] == 'varchar(30)'
        assert result3.values["percent"][9] == '23.177' # Why string?
        assert result3.values["count"][9] == '286' # Why string?
        assert result3.values["unique"][9] == '182' # Why string?
        assert result3.values["empty"][9] == 0
        assert result3.values["avg_length"][9] == pytest.approx(3.72027972)
        assert result3.values["stddev_length"][9] == pytest.approx(2.28313602)
        assert result3.values["min_length"][9] == 1
        assert result3.values["25%_length"][9] == 3
        assert result3.values["50%_length"][9] == 3
        assert result3.values["75%_length"][9] == 3
        assert result3.values["max_length"][9] == 15

        result4 = titanic_vd.describe(method = "numerical")

        assert result4.values["count"][1] == 1234
        assert result4.values["mean"][1] == pytest.approx(0.36466774)
        assert result4.values["std"][1] == pytest.approx(0.48153201)
        assert result4.values["min"][1] == 0
        assert result4.values["25%"][1] == 0
        assert result4.values["50%"][1] == 0
        assert result4.values["75%"][1] == 1
        assert result4.values["max"][1] == 1
        assert result4.values["unique"][1] == 2.0

        result5 = titanic_vd.describe(method = "range")

        assert result5.values["dtype"][2] == 'numeric(6,3)'
        assert result5.values["percent"][2] == '80.794' # Why string?
        assert result5.values["count"][2] == '997' # Why string?
        assert result5.values["unique"][2] == '96' # Why string?
        assert result5.values["min"][2] == '0.33' # Why string?
        assert result5.values["max"][2] == '80' # Why string?
        assert result5.values["range"][2] == '79.67' # Why string?

        result6 = titanic_vd.describe(method = "statistics")

        assert result6.values["dtype"][3] == 'int'
        assert result6.values["percent"][3] == '100' # Why string?
        assert result6.values["count"][3] == '1234' # Why string?
        assert result6.values["unique"][3] == '7' # Why string?
        assert result6.values["avg"][3] == '0.504051863857374' # Why string?
        assert result6.values["stddev"][3] == '1.04111727241629' # Why string?
        assert result6.values["min"][3] == '0' # Why string?
        assert result6.values["1%"][3] == pytest.approx(0.0)
        assert result6.values["10%"][3] == pytest.approx(0.0)
        assert result6.values["25%"][3] == '0' # Why string?
        assert result6.values["median"][3] == '0' # Why string?
        assert result6.values["75%"][3] == '1' # Why string?
        assert result6.values["90%"][3] == 1.0
        assert result6.values["99%"][3] == pytest.approx(5.0)
        assert result6.values["max"][3] == '8' # Why string?
        assert result6.values["skewness"][3] == pytest.approx(3.7597831)
        assert result6.values["kurtosis"][3] == pytest.approx(19.21388533)

    def test_vDF_describe_index(self, market_vd):
        # testing vDataFrame[].describe
        result1 = market_vd["Form"].describe(method = "categorical", max_cardinality = 3)

        assert result1.values["value"][0] == '"Form"'
        assert result1.values["value"][1] == 'varchar(32)'
        assert result1.values["value"][2] == 37.0
        assert result1.values["value"][3] == 314.0
        assert result1.values["value"][4] == 90
        assert result1.values["value"][5] == 90
        assert result1.values["value"][6] == 57
        assert result1.values["value"][7] == 47

        result2 = market_vd["Price"].describe(method = "numerical")

        assert result2.values["value"][0] == '"Price"'
        assert result2.values["value"][1] == 'float'
        assert result2.values["value"][2] == 308.0
        assert result2.values["value"][3] == 314
        assert result2.values["value"][4] == pytest.approx(2.07751098)
        assert result2.values["value"][5] == pytest.approx(1.51037749)
        assert result2.values["value"][6] == pytest.approx(0.31663877)
        assert result2.values["value"][7] == pytest.approx(1.07276187)
        assert result2.values["value"][8] == pytest.approx(1.56689808)
        assert result2.values["value"][9] == pytest.approx(2.60376599)
        assert result2.values["value"][10] == pytest.approx(10.163712)

        result3 = market_vd["Form"].describe(method = "cat_stats", numcol = "Price")

        assert result3.values["count"][3] == 2
        assert result3.values["percent"][3] == pytest.approx(Decimal('0.63694267515')) # Why this format?
        assert result3.values["mean"][3] == pytest.approx(4.6364768)
        assert result3.values["std"][3] == pytest.approx(0.6358942)
        assert result3.values["min"][3] == pytest.approx(4.1868317)
        assert result3.values["10%"][3] == pytest.approx(4.2767607)
        assert result3.values["25%"][3] == pytest.approx(4.4116542)
        assert result3.values["50%"][3] == pytest.approx(4.6364768)
        assert result3.values["75%"][3] == pytest.approx(4.8612994)
        assert result3.values["90%"][3] == pytest.approx(4.9961929)
        assert result3.values["max"][3] == pytest.approx(5.0861220)

    def test_vDF_distinct(self, amazon_vd):
        result = amazon_vd["state"].distinct()
        assert result == ['Acre', 'Alagoas', 'Amapa', 'Amazonas', 'Bahia', 'Ceara', 'Distrito Federal',
                          'Espirito Santo', 'Goias', 'Maranhao', 'Mato Grosso', 'Minas Gerais',
                          'Para', 'Paraiba', 'Pernambuco', 'Piau', 'Rio', 'Rondonia', 'Roraima',
                          'Santa Catarina', 'Sao Paulo', 'Sergipe', 'Tocantins']

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_duplicated(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_isin(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_kurt(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_mad(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_max(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_median(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_min(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_mode(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_nlargest(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_nsmallest(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_nunique(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_numh(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_prod(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_quantile(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_score(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_sem(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_shape(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_skew(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_statistics(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_std(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_sum(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_topk(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_value_counts(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_var(self):
        pass
