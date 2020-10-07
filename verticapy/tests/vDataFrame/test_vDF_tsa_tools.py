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
from verticapy.learn.tsa.tools import *


@pytest.fixture(scope="module")
def amazon_vd(base):
    from verticapy.learn.datasets import load_amazon

    amazon = load_amazon(cursor=base.cursor)
    yield amazon
    drop_table(name="public.amazon", cursor=base.cursor)


class TestvDFStatsTools:
    @pytest.mark.xfail(reason="The results are not correct")
    def test_adfuller(self, amazon_vd):
        # testing without trend
        result = adfuller(
            amazon_vd, column="number", ts="date", by=["state"], p=40, with_trend=False
        )
        assert result["value"][0] == pytest.approx(-1.9157890754403832, 0.01)
        assert result["value"][1] == pytest.approx(0.0554440321694081, 0.1)
        assert result["value"][-1] == False

        # testing with trend
        result = adfuller(
            amazon_vd, column="number", ts="date", by=["state"], p=40, with_trend=True
        )
        assert result["value"][0] == pytest.approx(-1.9156093125782623, 0.01)
        assert result["value"][1] == pytest.approx(0.0554669415324133, 0.1)
        assert result["value"][-1] == False

    def test_durbin_watson(self, amazon_vd):
        result = amazon_vd.copy()
        result["number_lag"] = "LAG(number) OVER (PARTITION BY state ORDER BY date)"
        result = durbin_watson(
            result, column="number", ts="date", by=["state"], X=["number_lag"]
        )
        assert result["value"][0] == pytest.approx(2.13353126698345, 0.01)
        assert result["value"][1] == True

    def test_jarque_bera(self, amazon_vd):
        result = jarque_bera(amazon_vd, column="number")
        assert result["value"][0] == pytest.approx(1031620.28905652, 0.1)
        assert result["value"][-1] == False

    def test_ljungbox(self, amazon_vd):
        # testing Ljung–Box
        result = ljungbox(
            amazon_vd, column="number", ts="date", by=["state"], p=40, box_pierce=False
        )
        assert result["Serial Correlation"][-1] == True
        assert result["p_value"][-1] == pytest.approx(0.0)
        assert result["Ljung–Box Test Statistic"][-1] == pytest.approx(
            23190.670172549573, 0.1
        )

        # testing Box-Pierce
        result = ljungbox(
            amazon_vd, column="number", ts="date", by=["state"], p=40, box_pierce=True
        )
        assert result["Serial Correlation"][-1] == True
        assert result["p_value"][-1] == pytest.approx(0.0)
        assert result["Box-Pierce Test Statistic"][-1] == pytest.approx(
            23128.102620000005, 0.1
        )

    @pytest.mark.xfail(reason="It fails")
    def test_mkt(self, amazon_vd):
        result = amazon_vd.groupby(["date"], ["AVG(number) AS number"])
        result = mkt(result, column="number", ts="date")
        assert result["value"][0] == pytest.approx(2.669258629634529, 0.1)
        assert result["value"][1] == pytest.approx(3196.0, 0.1)
        assert result["value"][2] == pytest.approx(1196.96156997625, 0.1)
        assert result["value"][3] == pytest.approx(0.003800944473728418, 0.1)
        assert result["value"][4] == True
