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
from vertica_python.errors import QueryError
from verticapy import drop, set_option
from verticapy.datasets import load_titanic, load_market, load_amazon
from verticapy.learn.linear_model import LogisticRegression

set_option("print_info", False)


@pytest.fixture(scope="module")
def titanic_vd():
    titanic = load_titanic()
    yield titanic
    drop(name="public.titanic")


@pytest.fixture(scope="module")
def market_vd():
    market = load_market()
    yield market
    drop(name="public.market")


@pytest.fixture(scope="module")
def amazon_vd():
    amazon = load_amazon()
    yield amazon
    drop(name="public.amazon")


class TestvDFDescriptiveStat:
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

    def test_vDF_numh(self, market_vd, amazon_vd):
        assert market_vd["Price"].numh(method="auto") == pytest.approx(0.984707376)
        assert market_vd["Price"].numh(method="freedman_diaconis") == pytest.approx(
            0.450501738
        )
        assert market_vd["Price"].numh(method="sturges") == pytest.approx(0.984707376)
        assert amazon_vd["date"].numh(method="auto") == pytest.approx(
            44705828.571428575
        )
        assert amazon_vd["date"].numh(method="freedman_diaconis") == pytest.approx(
            33903959.714834176
        )
        assert amazon_vd["date"].numh(method="sturges") == pytest.approx(
            44705828.571428575
        )
