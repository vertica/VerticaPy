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


@pytest.fixture(scope="module")
def titanic_vd(base):
    from verticapy.learn.datasets import load_titanic

    titanic = load_titanic(cursor=base.cursor)
    yield titanic
    drop_table(name = "public.titanic", cursor = base.cursor)

@pytest.fixture(scope="module")
def amazon_vd(base):
    from verticapy.learn.datasets import load_amazon

    amazon = load_amazon(cursor=base.cursor)
    yield amazon
    drop_table(name = "public.amazon", cursor = base.cursor)


@pytest.fixture(scope="module")
def titanic_vd(base):
    from verticapy.learn.datasets import load_titanic

    titanic = load_titanic(cursor=base.cursor)
    yield titanic
    drop_table(name="public.titanic", cursor=base.cursor)


@pytest.fixture(scope="module")
def amazon_vd(base):
    from verticapy.learn.datasets import load_amazon

    amazon = load_amazon(cursor=base.cursor)
    yield amazon
    drop_table(name="public.amazon", cursor=base.cursor)


class TestvDFCorrelation:
    def test_vDF_acf(self, amazon_vd):
        # The results of acf are not exactly the same on different runs
        # spearmann method
        result1 = amazon_vd.acf(ts = "date", 
                                column = "number", 
                                p = 20,
                                by = ["state"],
                                unit = "month",
                                method = "spearman",
                                show = False)

        assert result1.values["value"][0] == pytest.approx(1)
        assert result1.values["confidence"][0] == pytest.approx(0.0243968418)
        # the values are not very stable
        assert result1.values["value"][10] == pytest.approx(0.48, abs = 2e-3)
        assert result1.values["confidence"][10] == pytest.approx(0.068, abs = 5e-4)

        # pearson method
        result2 = amazon_vd.acf(ts = "date", 
                                column = "number", 
                                by = ["state"],
                                p = [1, 3, 6, 7],
                                unit = "year",
                                method = "pearson",
                                show = False)

        assert result2.values["value"][0] == pytest.approx(1.0)
        assert result2.values["confidence"][0] == pytest.approx(0.0243968418)
        assert result2.values["value"][4] == pytest.approx(0.366, abs = 5e-3)
        assert result2.values["confidence"][4] == pytest.approx(0.039, abs = 1e-3)

        # Autocorrelation Heatmap for each 'month' lag
        result3 = amazon_vd.acf(ts = "date", 
                                column = "number", 
                                by = ["state"],
                                p = 12,
                                unit = "month",
                                method = "pearson", 
                                round_nb = 3, 
                                acf_type = "heatmap",
                                show = False)

        assert result3.values["index"][1] == '"lag_12_number"'
        assert result3.values['"number"'][1] == pytest.approx(0.708, abs = 0.05)
        assert result3.values["index"][5] == '"lag_10_number"'
        assert result3.values['"number"'][5] == pytest.approx(0.299, abs = 0.01)

        # Autocorrelation Line for each 'month' lag
        result4 = amazon_vd.acf(ts = "date", 
                                column = "number", 
                                by = ["state"],
                                p = 12,
                                unit = "month",
                                method = "pearson",
                                acf_type = "line",
                                show = False)

        assert result4.values["value"][1] == pytest.approx(0.66, abs = 5e-2)
        assert result4.values["confidence"][1] == pytest.approx(0.034, abs = 1e-2)
        assert result4.values["value"][6] == pytest.approx(-0.097, abs = 1e-3)
        assert result4.values["confidence"][6] == pytest.approx(0.049, abs = 1e-2)

    def test_vDF_corr(self, titanic_vd):
        #
        # PEARSON
        #
        # testing vDataFrame.corr (method = 'pearson')
        result1 = titanic_vd.corr(
            columns=["survived", "age", "fare"], show=False, method="pearson"
        )
        assert result1["survived"][0] == 1.0
        assert result1["survived"][1] == pytest.approx(-0.0422446185581737)
        assert result1["survived"][2] == pytest.approx(0.264150360783869)
        assert result1["age"][0] == pytest.approx(-0.0422446185581737)
        assert result1["age"][1] == 1.0
        assert result1["age"][2] == pytest.approx(0.178575164117464)
        assert result1["fare"][0] == pytest.approx(0.264150360783869)
        assert result1["fare"][1] == pytest.approx(0.178575164117464)
        assert result1["fare"][2] == 1.0

        # Checking focus
        result2 = titanic_vd.corr(method = "pearson", focus = "survived", show = False)
        assert result2.values['"survived"'][1] == pytest.approx(-0.336, abs = 1e-3)
        assert result2.values['"survived"'][2] == pytest.approx(0.264, abs = 1e-3)

        #
        # SPEARMAN
        #
        # testing vDataFrame.corr (method = 'spearman')
        result3 = titanic_vd.corr(
            columns=["survived", "age", "fare"],method = "spearman", show = False)
        assert result3["survived"][0] == 1.0
        assert result3["survived"][1] == pytest.approx(-0.0909763969172584)
        assert result3["survived"][2] == pytest.approx(0.322409613111481)
        assert result3["age"][0] == pytest.approx(-0.0909763969172584)
        assert result3["age"][1] == 1.0
        assert result3["age"][2] == pytest.approx(0.00451935857538285)
        assert result3["fare"][0] == pytest.approx(0.322409613111481)
        assert result3["fare"][1] == pytest.approx(0.00451935857538285)
        assert result3["fare"][2] == 1.0

        #
        # CRAMER'S V
        #
        # testing vDataFrame.corr (method = 'cramer')
        result4 = titanic_vd.corr(
            columns=["survived", "pclass", "embarked"], show=False, method="cramer"
        )
        assert result4["survived"][0] == 1.0
        assert result4["survived"][1] == pytest.approx(0.235827362803827)
        assert result4["survived"][2] == pytest.approx(0.128304316649469)
        assert result4["pclass"][0] == pytest.approx(0.235827362803827)
        assert result4["pclass"][1] == 1.0
        assert result4["pclass"][2] == pytest.approx(0.2214160584227)
        assert result4["embarked"][0] == pytest.approx(0.128304316649469)
        assert result4["embarked"][1] == pytest.approx(0.2214160584227)
        assert result4["embarked"][2] == 1.0

        #
        # KENDALL
        #
        # testing vDataFrame.corr (method = 'kendall')
        result5 = titanic_vd.corr(
            columns=["survived", "age", "fare"], show=False, method="kendall"
        )
        assert result5["survived"][0] == 1.0
        assert result5["survived"][1] == pytest.approx(-0.033166080419339265)
        assert result5["survived"][2] == pytest.approx(0.3894712709557243)
        assert result5["age"][0] == pytest.approx(-0.033166080419339265)
        assert result5["age"][1] == 1.0
        assert result5["age"][2] == pytest.approx(0.1324522539797924)
        assert result5["fare"][0] == pytest.approx(0.3894712709557243)
        assert result5["fare"][1] == pytest.approx(0.1324522539797924)
        assert result5["fare"][2] == 1.0

        #
        # BISERIAL POINT
        #
        # testing vDataFrame.corr (method = 'biserial')
        result6 = titanic_vd.corr(
            columns=["survived", "age", "fare"], show=False, method="biserial"
        )
        assert result6["survived"][0] == 1.0
        assert result6["survived"][1] == pytest.approx(-0.0422234273762242)
        assert result6["survived"][2] == pytest.approx(0.264043222121672)
        assert result6["age"][0] == pytest.approx(-0.0422234273762242)
        assert result6["age"][1] == 1.0
        assert result6["fare"][0] == pytest.approx(0.264043222121672)
        assert result6["fare"][2] == 1.0

    def test_vDF_cov(self, titanic_vd):
        # testing vDataFrame.cov
        result1 = titanic_vd.cov(columns=["survived", "age", "fare"], show=False)
        assert result1["survived"][0] == pytest.approx(0.231685181342251)
        assert result1["survived"][1] == pytest.approx(-0.297583583247234)
        assert result1["survived"][2] == pytest.approx(6.69214075159394)
        assert result1["age"][0] == pytest.approx(-0.297583583247234)
        assert result1["age"][1] == pytest.approx(208.169014723609)
        assert result1["age"][2] == pytest.approx(145.057125218791)
        assert result1["fare"][0] == pytest.approx(6.69214075159394)
        assert result1["fare"][1] == pytest.approx(145.057125218791)
        assert result1["fare"][2] == pytest.approx(2769.36114247479)

        # testing vDataFrame.cov with focus
        result1_f = titanic_vd.cov(
            columns=["survived", "age", "fare"], focus="survived", show=False
        )
        assert result1_f["survived"][0] == pytest.approx(6.69214075159394)
        assert result1_f["survived"][1] == pytest.approx(-0.297583583247234)
        assert result1_f["survived"][2] == pytest.approx(0.231685181342251)

    def test_vDF_pacf(self, amazon_vd):
        # testing vDataFrame.pacf
        result = amazon_vd.pacf(
            column="number", ts="date", by=["state"], p=5, show=False
        )
        assert result["value"][0] == 1.0
        assert result["value"][1] == pytest.approx(0.514, abs = 0.01)
        assert result["value"][2] == pytest.approx(0.133, abs = 0.01)
        #assert result["value"][3] == pytest.approx(-0.029, abs = 0.01)
        #assert result["value"][4] == pytest.approx(-0.046, abs = 0.01)
        assert result["value"][5] == pytest.approx(-0.053, abs = 0.01)

    def test_vDF_regr(self, titanic_vd):
        #
        # ALPHA
        #
        # testing vDataFrame.regr (method = 'alpha')
        result1 = titanic_vd.regr(
            columns=["survived", "age", "fare"], show=False, method="alpha"
        )
        assert result1["survived"][0] == 0.0
        assert result1["survived"][1] == pytest.approx(0.435280333103508)
        assert result1["survived"][2] == pytest.approx(0.282890247028015)
        assert result1["age"][0] == pytest.approx(30.6420462046205)
        assert result1["age"][1] == 0.0
        assert result1["age"][2] == pytest.approx(28.4268042866199)
        assert result1["fare"][0] == pytest.approx(23.425595019157)
        assert result1["fare"][1] == pytest.approx(16.1080039795446)
        assert result1["fare"][2] == 0.0

        # making sure that vDataFrame.regr (method = 'alpha') is the same
        result1_1 = titanic_vd.regr(
            columns=["survived", "age", "fare"], show=False, method="alpha"
        )
        assert result1_1["survived"][0] == result1["survived"][0]
        assert result1_1["survived"][1] == result1["survived"][1]
        assert result1_1["survived"][2] == result1["survived"][2]
        assert result1_1["age"][0] == result1["age"][0]
        assert result1_1["age"][1] == result1["age"][1]
        assert result1_1["age"][2] == result1["age"][2]
        assert result1_1["fare"][0] == result1["fare"][0]
        assert result1_1["fare"][1] == result1["fare"][1]
        assert result1_1["fare"][2] == result1["fare"][2]

        #
        # BETA
        #
        # testing vDataFrame.regr (method = 'beta')
        result2 = titanic_vd.regr(
            columns=["survived", "age", "fare"], show=False, method="beta"
        )
        assert result2["survived"][0] == 1.0
        assert result2["survived"][1] == pytest.approx(-0.00142952871080426)
        assert result2["survived"][2] == pytest.approx(0.00241649261591561)
        assert result2["age"][0] == pytest.approx(-1.2483889156179)
        assert result2["age"][1] == 1.0
        assert result2["age"][2] == pytest.approx(0.0456059549185254)
        assert result2["fare"][0] == pytest.approx(28.8746643141762)
        assert result2["fare"][1] == pytest.approx(0.69923081967147)
        assert result2["fare"][2] == 1.0

        # making sure that vDataFrame.regr (method = 'beta') is the same
        result2_1 = titanic_vd.regr(
            columns=["survived", "age", "fare"], show=False, method="beta"
        )
        assert result2_1["survived"][0] == result2["survived"][0]
        assert result2_1["survived"][1] == result2["survived"][1]
        assert result2_1["survived"][2] == result2["survived"][2]
        assert result2_1["age"][0] == result2["age"][0]
        assert result2_1["age"][1] == result2["age"][1]
        assert result2_1["age"][2] == result2["age"][2]
        assert result2_1["fare"][0] == result2["fare"][0]
        assert result2_1["fare"][1] == result2["fare"][1]
        assert result2_1["fare"][2] == result2["fare"][2]

        #
        # R2
        #
        # testing vDataFrame.regr (method = 'r2')
        result3 = titanic_vd.regr(
            columns=["survived", "age", "fare"], show=False, method="r2"
        )
        assert result3["survived"][0] == 1.0
        assert result3["survived"][1] == pytest.approx(0.00178460779712559)
        assert result3["survived"][2] == pytest.approx(0.0697754131022489)
        assert result3["age"][0] == pytest.approx(0.00178460779712559)
        assert result3["age"][1] == 1.0
        assert result3["age"][2] == pytest.approx(0.0318890892395806)
        assert result3["fare"][0] == pytest.approx(0.0697754131022489)
        assert result3["fare"][1] == pytest.approx(0.0318890892395806)
        assert result3["fare"][2] == 1.0

        # making sure that vDataFrame.regr (method = 'r2') is the same
        result3_1 = titanic_vd.regr(
            columns=["survived", "age", "fare"], show=False, method="r2"
        )
        assert result3_1["survived"][0] == result3["survived"][0]
        assert result3_1["survived"][1] == result3["survived"][1]
        assert result3_1["survived"][2] == result3["survived"][2]
        assert result3_1["age"][0] == result3["age"][0]
        assert result3_1["age"][1] == result3["age"][1]
        assert result3_1["age"][2] == result3["age"][2]
        assert result3_1["fare"][0] == result3["fare"][0]
        assert result3_1["fare"][1] == result3["fare"][1]
        assert result3_1["fare"][2] == result3["fare"][2]
