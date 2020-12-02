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
from verticapy import vDataFrame
from verticapy.learn.tsa.tools import *

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


class TestvDFStatsTools:
    def test_adfuller(self, amazon_vd):
        # testing without trend
        result = adfuller(
            amazon_vd, column="number", ts="date", by=["state"], p=40, with_trend=False
        )
        assert result["value"][0] == pytest.approx(-0.4059507552046538, 1e-2)
        assert result["value"][1] == pytest.approx(0.684795156687264, 1e-2)
        assert result["value"][-1] == False

        # testing with trend
        result = adfuller(
            amazon_vd, column="number", ts="date", by=["state"], p=40, with_trend=True
        )
        assert result["value"][0] == pytest.approx(-0.4081159118011171, 1e-2)
        assert result["value"][1] == pytest.approx(0.683205052234998, 1e-2)
        assert result["value"][-1] == False

    def test_durbin_watson(self, amazon_vd):
        result = amazon_vd.copy()
        result["number_lag"] = "LAG(number) OVER (PARTITION BY state ORDER BY date)"
        result = durbin_watson(
            result, column="number", ts="date", by=["state"], X=["number_lag"]
        )
        assert result["value"][0] == pytest.approx(1.74223305056269, 1e-2)
        assert result["value"][1] == True

    def test_jarque_bera(self, amazon_vd):
        result = jarque_bera(amazon_vd, column="number")
        assert result["value"][0] == pytest.approx(930829.520860999, 1e-2)
        assert result["value"][-1] == False

    def test_ljungbox(self, amazon_vd):
        # testing Ljung–Box
        result = ljungbox(
            amazon_vd, column="number", ts="date", by=["state"], p=40, box_pierce=False
        )
        assert result["Serial Correlation"][-1] == True
        assert result["p_value"][-1] == pytest.approx(0.0)
        assert result["Ljung–Box Test Statistic"][-1] == pytest.approx(
            40184.55076431489, 1e-2
        )

        # testing Box-Pierce
        result = ljungbox(
            amazon_vd, column="number", ts="date", by=["state"], p=40, box_pierce=True
        )
        assert result["Serial Correlation"][-1] == True
        assert result["p_value"][-1] == pytest.approx(0.0)
        assert result["Box-Pierce Test Statistic"][-1] == pytest.approx(
            40053.87251600001, 1e-2
        )

    def test_mkt(self, amazon_vd):
        result = amazon_vd.groupby(["date"], ["AVG(number) AS number"])
        result = mkt(result, column="number", ts="date")
        assert result["value"][0] == pytest.approx(2.579654773618437, 1e-2)
        assert result["value"][1] == pytest.approx(3188.0, 1e-2)
        assert result["value"][2] == pytest.approx(1235.43662996799, 1e-2)
        assert result["value"][3] == pytest.approx(0.014317855149496753, 1e-2)
        assert result["value"][4] == True
        assert result["value"][5] == "increasing"
