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


@pytest.fixture(scope="module")
def titanic_vd(base):
    from verticapy.learn.datasets import load_titanic
    from verticapy import drop_table

    titanic = load_titanic(cursor=base.cursor)
    yield titanic
    drop_table(name = "public.titanic", cursor = base.cursor)
    


class TestvDFDescriptiveStat:
    def test_vDF_aad(self, titanic_vd):
        # testing vDataFrame.aad
        result = titanic_vd.aad(columns=["age", "fare", "parch"])
        assert result.values["aad"][0] == pytest.approx(11.2547854194)
        assert result.values["aad"][1] == pytest.approx(30.6258659424)
        assert result.values["aad"][2] == pytest.approx(0.58208012314)

        # testing vDataFrame[].aad
        assert titanic_vd["age"].aad() == pytest.approx(11.254785419447906)
        assert titanic_vd["fare"].aad() == pytest.approx(30.625865942462237)
        assert titanic_vd["parch"].aad() == pytest.approx(0.5820801231451393)

    def test_vDF_agg(self, titanic_vd):
        from decimal import Decimal

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

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_all(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_any(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_avg(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_count(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_describe(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_distinct(self):
        pass

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
