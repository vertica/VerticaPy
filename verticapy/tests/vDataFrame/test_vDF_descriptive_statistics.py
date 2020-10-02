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
        assert exception_info.match("Could not convert")

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

    def test_vDF_duplicated(self, market_vd):
        result = market_vd.duplicated(columns = ["Form", "Name"])

        assert result.count == 151
        assert len(result.values) == 3

    def test_vDF_isin(self, market_vd):
        result = market_vd.groupby(columns = ["Form", "Name"],
                                expr = ["AVG(Price) AS avg_price",
                                        "STDDEV(Price) AS std"])

        assert result.shape() == (159, 4)

    def test_vDF_isin(self, amazon_vd):
        # testing vDataFrame.isin
        assert amazon_vd.isin({"state": ["Rio", "Acre"], "number": [0, 0]}) == [True, True]

        # testing vDataFrame[].isin
        assert amazon_vd["state"].isin(val = ["Rio", "Acre", "Paris"]) == [True, True, False]

    def test_vDF_kurt(self, titanic_vd):
        # testing vDataFrame.kurt
        result1 = titanic_vd.kurt(columns = ["age", "fare", "parch"])

        assert result1.values['kurtosis'][0] == pytest.approx(0.15689691)
        assert result1.values['kurtosis'][1] == pytest.approx(26.2543152)
        assert result1.values['kurtosis'][2] == pytest.approx(22.6438022)

        # testing vDataFrame.kurtosis
        result2 = titanic_vd.kurtosis(columns = ["age", "fare", "parch"])

        assert result2.values['kurtosis'][0] == result1.values['kurtosis'][0]
        assert result2.values['kurtosis'][1] == result1.values['kurtosis'][1]
        assert result2.values['kurtosis'][2] == result1.values['kurtosis'][2]

        # testing vDataFrame[].kurt
        assert titanic_vd['age'].kurt() == result1.values['kurtosis'][0]
        assert titanic_vd['fare'].kurt() == result1.values['kurtosis'][1]
        assert titanic_vd['parch'].kurt() == result1.values['kurtosis'][2]

        # testing vDataFrame[].kurtosis
        assert titanic_vd['age'].kurtosis() == result1.values['kurtosis'][0]
        assert titanic_vd['fare'].kurtosis() == result1.values['kurtosis'][1]
        assert titanic_vd['parch'].kurtosis() == result1.values['kurtosis'][2]

    def test_vDF_mad(self, titanic_vd):
        # testing vDataFrame.mad
        result1 = titanic_vd.mad(columns = ["age", "fare", "parch"])

        assert result1.values["mad"][0] == pytest.approx(8.0)
        assert result1.values["mad"][1] == pytest.approx(6.9042)
        assert result1.values["mad"][2] == pytest.approx(0.0)

        # testing vDataFrame[].mad
        assert titanic_vd["age"].mad() == result1.values["mad"][0]
        assert titanic_vd["fare"].mad() == result1.values["mad"][1]
        assert titanic_vd["parch"].mad() == result1.values["mad"][2]

    def test_vDF_max(self, titanic_vd):
        # testing vDataFrame.max
        result1 = titanic_vd.max(columns = ["age", "fare", "parch"])

        assert result1.values["max"][0] == pytest.approx(80.0)
        assert result1.values["max"][1] == pytest.approx(512.3292)
        assert result1.values["max"][2] == pytest.approx(9.0)

        # testing vDataFrame[].max
        assert titanic_vd["age"].max() == result1.values["max"][0]
        assert titanic_vd["fare"].max() == result1.values["max"][1]
        assert titanic_vd["parch"].max() == result1.values["max"][2]

    def test_vDF_median(self, titanic_vd):
        # testing vDataFrame.median
        result = titanic_vd.median(columns = ["age", "fare", "parch"])

        assert result.values["median"][0] == pytest.approx(28.0)
        assert result.values["median"][1] == pytest.approx(14.4542)
        assert result.values["median"][2] == pytest.approx(0.0)

        # testing vDataFrame[].median
        assert titanic_vd["age"].median() == result.values["median"][0]
        assert titanic_vd["fare"].median() == result.values["median"][1]
        assert titanic_vd["parch"].median() == result.values["median"][2]

    def test_vDF_min(self, titanic_vd):
        # testing vDataFrame.min
        result = titanic_vd.min(columns = ["age", "fare", "parch"])

        assert result.values["min"][0] == pytest.approx(0.33)
        assert result.values["min"][1] == pytest.approx(0.0)
        assert result.values["min"][2] == pytest.approx(0.0)

        # testing vDataFrame[].median
        assert titanic_vd["age"].min() == result.values["min"][0]
        assert titanic_vd["fare"].min() == result.values["min"][1]
        assert titanic_vd["parch"].min() == result.values["min"][2]

    def test_vDF_mode(self, market_vd):
        # testing vDataFrame[].mod
        assert market_vd["Name"].mode() == 'Pineapple'
        assert market_vd["Name"].mode(n = 2) == 'Carrots'

    def test_vDF_nlargest(self, market_vd):
        result = market_vd["Price"].nlargest(n = 2)

        assert result.values["Name"][0] == 'Mangoes'
        assert result.values["Form"][0] == 'Dried'
        assert result.values["Price"][0] == pytest.approx(10.1637125)
        assert result.values["Name"][1] == 'Mangoes'
        assert result.values["Form"][1] == 'Dried'
        assert result.values["Price"][1] == pytest.approx(8.50464930)

    def test_vDF_nsmallest(self, market_vd):
        result = market_vd["Price"].nsmallest(n = 2)

        assert result.values["Name"][0] == 'Watermelon'
        assert result.values["Form"][0] == 'Fresh'
        assert result.values["Price"][0] == pytest.approx(0.31663877)
        assert result.values["Name"][1] == 'Watermelon'
        assert result.values["Form"][1] == 'Fresh'
        assert result.values["Price"][1] == pytest.approx(0.33341203)

    def test_vDF_nunique(self, titanic_vd):
        result = titanic_vd.nunique(columns = ["pclass", "embarked", "survived", "cabin"])

        assert result.values["unique"][0] == 3.0
        assert result.values["unique"][1] == 3.0
        assert result.values["unique"][2] == 2.0
        assert result.values["unique"][3] == 182.0

    def test_vDF_numh(self, market_vd):
        assert market_vd["Price"].numh(method = "auto") == pytest.approx(0.984707376)
        assert market_vd["Price"].numh(method = "freedman_diaconis") == pytest.approx(0.450501738)
        assert market_vd["Price"].numh(method = "sturges") == pytest.approx(0.984707376)

    def test_vDF_prod(self, market_vd):
        # testing vDataFrame.prod
        result1 = market_vd.prod(columns = ["Price"])

        assert result1.values["prod"][0] == pytest.approx(1.9205016913e+71)

        # testing vDataFrame.product
        result2 = market_vd.product(columns = ["Price"])

        assert result2.values["prod"][0] == result1.values["prod"][0]

        # testing vDataFrame[].prod
        assert market_vd["price"].prod() == result1.values["prod"][0]

        # testing vDataFrame[].product
        assert market_vd["price"].product() == result1.values["prod"][0]

    def test_vDF_quantile(self, titanic_vd):
        # testing vDataFrame.quantile
        result = titanic_vd.quantile(q= [0.22, 0.9], columns = ["age", "fare"])

        assert result.values["22.0%"][0] == pytest.approx(20.0)
        assert result.values["90.0%"][0] == pytest.approx(50.0)
        assert result.values["22.0%"][1] == pytest.approx(7.8958)
        assert result.values["90.0%"][1] == pytest.approx(79.13)

        # testing vDataFrame[].quantile
        assert titanic_vd["age"].quantile(x = 0.5) == pytest.approx(28.0)
        assert titanic_vd["fare"].quantile(x = 0.1) == pytest.approx(7.5892)

    def test_vDF_score(self, base, titanic_vd):
        from verticapy.learn.linear_model import LogisticRegression
        model = LogisticRegression(name = "public.LR_titanic", cursor=base.cursor,
                                   tol = 1e-4, C = 1.0, max_iter = 100, 
                                   solver = 'CGD', l1_ratio = 0.5)

        model.drop() # dropping the model in case of its existance
        model.fit("public.titanic", ["fare", "age"], "survived")
        model.predict(titanic_vd, name = "survived_pred")

        # Computing AUC
        auc = titanic_vd.score(y_true  = "survived", y_score = "survived_pred", method  = "auc")
        assert auc == pytest.approx(0.697476274)

        # Computing MSE
        mse = titanic_vd.score(y_true  = "survived", y_score = "survived_pred", method  = "mse")
        assert mse == pytest.approx(0.224993557)

        # Drawing ROC Curve
        roc_res = titanic_vd.score(y_true  = "survived", y_score = "survived_pred", method  = "roc")
        assert roc_res.values["threshold"][3] == 0.003
        assert roc_res.values["false_positive"][3] == 1.0
        assert roc_res.values["true_positive"][3] == 1.0
        assert roc_res.values["threshold"][300] == 0.3
        assert roc_res.values["false_positive"][300] == pytest.approx(0.9900826446)
        assert roc_res.values["true_positive"][300] == pytest.approx(0.9974424552)
        assert roc_res.values["threshold"][900] == 0.9
        assert roc_res.values["false_positive"][900] == pytest.approx(0.01818181818)
        assert roc_res.values["true_positive"][900] == pytest.approx(0.06649616368)

        # Drawing PRC Curve
        prc_res = titanic_vd.score(y_true  = "survived", y_score = "survived_pred", method  = "prc")
        assert prc_res.values["threshold"][3] == 0.002
        assert prc_res.values["recall"][3] == 1.0
        assert prc_res.values["precision"][3] == pytest.approx(0.3925702811)
        assert prc_res.values["threshold"][300] == 0.299
        assert prc_res.values["recall"][300] == pytest.approx(1.0)
        assert prc_res.values["precision"][300] == pytest.approx(0.3949494949)
        assert prc_res.values["threshold"][900] == 0.899
        assert prc_res.values["recall"][900] == pytest.approx(0.06649616368)
        assert prc_res.values["precision"][900] == pytest.approx(0.7027027027)

        # dropping the created model
        model.drop()

    def test_vDF_sem(self, titanic_vd):
        # testing vDataFrame.sem
        result = titanic_vd.sem(columns = ["age", "fare"])

        assert result.values["sem"][0] == pytest.approx(0.457170684)
        assert result.values["sem"][1] == pytest.approx(1.499285853)

        # testing vDataFrame[].sem
        assert titanic_vd["parch"].sem() == pytest.approx(0.024726611)

    def test_vDF_shape(self, market_vd):
        assert market_vd.shape() == (314, 3)

    def test_vDF_skew(self, titanic_vd):
        # testing vDataFrame.skew
        result1 = titanic_vd.skew(columns = ["age", "fare"])

        assert result1.values["skewness"][0] == pytest.approx(0.408876460)
        assert result1.values["skewness"][1] == pytest.approx(4.300699188)

        # testing vDataFrame.skewness
        result2 = titanic_vd.skewness(columns = ["age", "fare"])

        assert result2.values["skewness"][0] == result1.values["skewness"][0]
        assert result2.values["skewness"][1] == result1.values["skewness"][1]

        # testing vDataFrame[].skew
        assert titanic_vd["parch"].skew() == pytest.approx(3.798019282)
        # testing vDataFrame[].skewness
        assert titanic_vd["parch"].skewness() == pytest.approx(3.798019282)

    def test_vDF_std(self, titanic_vd):
        # testing vDataFrame.std
        result = titanic_vd.std(columns = ["fare"])

        assert result.values["stddev"][0] == pytest.approx(52.64607298)

        # testing vDataFrame[].std
        assert titanic_vd["parch"].std() == pytest.approx(0.868604707)

    def test_vDF_sum(self, titanic_vd):
        # testing vDataFrame.sum
        result = titanic_vd.sum(columns = ["fare", "parch"])

        assert result.values["sum"][0] == pytest.approx(41877.3576)
        assert result.values["sum"][1] == pytest.approx(467.0)

        # testing vDataFrame[].sum
        assert titanic_vd["age"].sum() == pytest.approx(30062.0)

    def test_vDF_topk(self, market_vd):
        result = market_vd["Name"].topk(k = 3)

        assert result.values["count"][0] == 12
        assert result.values["percent"][0] == pytest.approx(3.822)
        assert result.values["count"][1] == 10
        assert result.values["percent"][1] == pytest.approx(3.185)
        assert result.values["count"][2] == 8
        assert result.values["percent"][2] == pytest.approx(2.548)

    def test_vDF_value_counts(self, market_vd):
        result = market_vd["Name"].value_counts(k = 2)

        assert result.values["value"][0] == '"Name"'
        assert result.values["value"][1] == "varchar(32)"
        assert result.values["value"][2] == 73.0
        assert result.values["value"][3] == 314.0
        assert result.values["value"][4] == 284
        assert result.values["value"][5] == 12
        assert result.values["value"][6] == 10
        assert result.values["index"][6] == 'Carrots'

    def test_vDF_var(self, titanic_vd):
        # testing vDataFrame.var
        result = titanic_vd.var(columns = ["age", "parch"])

        assert result.values["variance"][0] == pytest.approx(208.3780197)
        assert result.values["variance"][1] == pytest.approx(0.754474138)

        # testing vDataFrame[].var
        assert titanic_vd["fare"].var() == pytest.approx(2771.6090005)
