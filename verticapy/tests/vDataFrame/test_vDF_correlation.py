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

    titanic = load_titanic(cursor=base.cursor)
    yield titanic
    drop_table(name="public.titanic", cursor=base.cursor)


@pytest.fixture(scope="module")
def market_vd(base):
    from verticapy.learn.datasets import load_market

    market = load_market(cursor=base.cursor)
    yield market
    drop_table(name="public.market", cursor=base.cursor)


@pytest.fixture(scope="module")
def amazon_vd(base):
    from verticapy.learn.datasets import load_amazon

    amazon = load_amazon(cursor=base.cursor)
    yield amazon
    drop_table(name="public.amazon", cursor=base.cursor)


class TestvDFCorrelation:
    def test_vDF_acf(self):
        # testing vDataFrame.acf
        result1 = amazon_vd.acf(
            column="number", ts="date", by=["state"], p=5, show=False
        )
        assert result1["value"][0] == 1.0
        assert result1["value"][1] == pytest.approx(0.515)
        assert result1["value"][2] == pytest.approx(0.362)
        assert result1["value"][3] == pytest.approx(0.208)
        assert result1["value"][4] == pytest.approx(0.095)
        assert result1["value"][5] == pytest.approx(0.006)

        # making sure that vDataFrame.acf is the same
        result1_1 = amazon_vd.acf(
            column="number", ts="date", by=["state"], p=5, show=False
        )
        assert result1_1["value"][0] == result1["value"][0]
        assert result1_1["value"][1] == result1["value"][1]
        assert result1_1["value"][2] == result1["value"][2]
        assert result1_1["value"][3] == result1["value"][3]
        assert result1_1["value"][4] == result1["value"][4]
        assert result1_1["value"][5] == result1["value"][5]

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_corr(self):
        pass

    def test_vDF_cov(self):
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

        # making sure that vDataFrame.cov is the same
        result1_1 = titanic_vd.cov(columns=["survived", "age", "fare"], show=False)
        assert result1_1["survived"][0] == result1["survived"][0]
        assert result1_1["survived"][1] == result1["survived"][1]
        assert result1_1["survived"][2] == result1["survived"][2]
        assert result1_1["age"][0] == result1["age"][0]
        assert result1_1["age"][1] == result1["age"][1]
        assert result1_1["age"][2] == result1["age"][2]
        assert result1_1["fare"][0] == result1["fare"][0]
        assert result1_1["fare"][1] == result1["fare"][1]
        assert result1_1["fare"][2] == result1["fare"][2]

        # making sure that vDataFrame.cov with focus is the same
        result1_f_1 = titanic_vd.cov(
            columns=["survived", "age", "fare"], focus="survived", show=False
        )
        assert result1_f["survived"][0] == result1_f["survived"][0]
        assert result1_f["survived"][1] == result1_f["survived"][1]
        assert result1_f["survived"][2] == result1_f["survived"][2]

    def test_vDF_pacf(self):
        # testing vDataFrame.pacf
        result1 = amazon_vd.pacf(
            column="number", ts="date", by=["state"], p=5, show=False
        )
        assert result1["value"][0] == 1.0
        assert result1["value"][1] == pytest.approx(0.514716791247187)
        assert result1["value"][2] == pytest.approx(0.133201986167273)
        assert result1["value"][3] == pytest.approx(-0.0293272001119337)
        assert result1["value"][4] == pytest.approx(-0.0468372999807555)
        assert result1["value"][5] == pytest.approx(-0.053730457039713)

        # making sure that vDataFrame.pacf is the same
        result1_1 = amazon_vd.pacf(
            column="number", ts="date", by=["state"], p=5, show=False
        )
        assert result1_1["value"][0] == result1["value"][0]
        assert result1_1["value"][1] == result1["value"][1]
        assert result1_1["value"][2] == result1["value"][2]
        assert result1_1["value"][3] == result1["value"][3]
        assert result1_1["value"][4] == result1["value"][4]
        assert result1_1["value"][5] == result1["value"][5]

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_regr(self):
        pass
