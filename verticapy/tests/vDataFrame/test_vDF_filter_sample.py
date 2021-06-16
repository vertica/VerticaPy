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
from verticapy import vDataFrame
from verticapy import drop

from verticapy import set_option

set_option("print_info", False)
set_option("random_state", 0)


@pytest.fixture(scope="module")
def smart_meters_vd(base):
    from verticapy.datasets import load_smart_meters

    smart_meters = load_smart_meters(cursor=base.cursor)
    yield smart_meters
    with warnings.catch_warnings(record=True) as w:
        drop(
            name="public.smart_meters", cursor=base.cursor,
        )


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
def amazon_vd(base):
    from verticapy.datasets import load_amazon

    amazon = load_amazon(cursor=base.cursor)
    yield amazon
    with warnings.catch_warnings(record=True) as w:
        drop(
            name="public.amazon", cursor=base.cursor,
        )


class TestvDFFilterSample:
    def test_vDF_search(self, titanic_vd):
        # testing with one condition
        result1 = titanic_vd.search(
            conditions="age BETWEEN 30 AND 70",
            usecols=["pclass", "boat", "embarked", "age", "family_size"],
            expr=["sibsp + parch + 1 AS family_size"],
            order_by={"age": "desc", "family_size": "asc"},
        )
        assert result1.shape() == (456, 5)
        assert result1["age"][0] == 70.0
        assert result1["age"][1] == 70.0
        assert result1["family_size"][0] == 1
        assert result1["family_size"][1] == 3
        assert result1["pclass"][0] == 2
        assert result1["pclass"][1] == 1

        # testing with multiple conditions
        result2 = titanic_vd.search(
            conditions=["age BETWEEN 30 AND 64", "pclass = 1"],
            usecols=["pclass", "boat", "embarked", "age", "family_size"],
            expr=["sibsp + parch + 1 AS family_size"],
            order_by={"age": "desc", "family_size": "asc"},
        )
        assert result2.shape() == (191, 5)
        assert result2["age"][0] == 64.0
        assert result2["age"][1] == 64.0
        assert result2["age"][2] == 64.0
        assert result2["family_size"][0] == 1
        assert result2["family_size"][1] == 2
        assert result2["family_size"][2] == 3
        assert result2["pclass"][0] == 1
        assert result2["pclass"][1] == 1

    def test_vDF_at_time(self, smart_meters_vd):
        result = smart_meters_vd.copy().at_time(ts="time", time="12:00",)
        assert result.shape() == (140, 3)

    def test_vDF_balance(self, titanic_vd):
        # hybrid
        result = titanic_vd.balance(
            "embarked", method="hybrid"
        )["embarked"].topk()
        assert 30 < result["percent"][0] < 40
        assert 30 < result["percent"][1] < 40
        assert 30 < result["percent"][2] < 40
        # under
        result = titanic_vd.balance(
            "embarked", method="under", x = 0.5
        )["embarked"].topk()
        assert 35 < result["percent"][0] < 55
        assert 35 < result["percent"][1] < 55
        assert 15 < result["percent"][2] < 30
        # over
        result = titanic_vd.balance(
            "embarked", method="over", x = 0.5
        )["embarked"].topk()
        assert 40 < result["percent"][0] < 55
        assert 15 < result["percent"][1] < 35
        assert 15 < result["percent"][2] < 35

    def test_vDF_between_time(self, smart_meters_vd):
        result = smart_meters_vd.copy().between_time(
            ts="time", start_time="12:00", end_time="14:00",
        )
        assert result.shape() == (1151, 3)

    def test_vDF_filter(self, titanic_vd):
        result = titanic_vd.copy().filter(
            ["pclass = 1 OR age > 50", "embarked = 'S'", "boat IS NOT NULL"],
        )
        assert result.shape() == (99, 14)

    def test_vDF_first(self, smart_meters_vd):
        result = smart_meters_vd.copy().first(ts="time", offset="6 months",)
        assert result.shape() == (3427, 3)

    def test_vDF_isin(self, amazon_vd):
        # testing vDataFrame.isin
        assert amazon_vd.isin(
            {"state": ["SERGIPE", "TOCANTINS"], "number": [0, 0]}
        ).shape() == (90, 3)

        # testing vDataFrame[].isin
        assert amazon_vd["state"].isin(
            val=["SERGIPE", "TOCANTINS", "PARIS"]
        ).shape() == (478, 3)

    def test_vDF_last(self, smart_meters_vd):
        result = smart_meters_vd.copy().last(ts="time", offset="1 year",)
        assert result.shape() == (7018, 3)

    def test_vDF_drop(self, titanic_vd):
        # testing vDataFrame.drop
        result = titanic_vd.copy().drop(columns=["pclass", "boat", "embarked"])
        assert result.shape() == (1234, 11)

        # testing vDataFrame[].drop
        result = titanic_vd.copy()["survived"].drop()
        assert result.shape() == (1234, 13)

    def test_vDF_drop_duplicates(self, titanic_vd):
        result = titanic_vd.copy().drop_duplicates(columns=["age", "fare", "pclass"],)
        assert result.shape() == (942, 14)

    def test_vDF_drop_outliers(self, titanic_vd):
        # testing with threshold
        result1 = titanic_vd.copy()["age"].drop_outliers(threshold=3.0,)
        assert result1.shape() == (994, 14)

        # testing without threshold
        result2 = titanic_vd.copy()["age"].drop_outliers(
            use_threshold=False, alpha=0.05,
        )
        assert result2.shape() == (900, 14)

    def test_vDF_select(self, titanic_vd):
        result = titanic_vd.select(columns=["age", "fare", "pclass"])
        assert result.shape() == (1234, 3)
        result = titanic_vd.select(
            columns=["age", "fare", "pclass", "parch + sibsp + 1 AS family_size"],
        )
        assert result.shape() == (1234, 4)

    def test_vDF_sample(self, titanic_vd):
        # testing with x
        result = titanic_vd.copy().sample(x=0.33, method="random")
        assert result.shape()[0] == pytest.approx(1234 * 0.33, 0.12)
        result2 = titanic_vd.copy().sample(
            x=0.33, method="stratified", by=["age", "pclass",]
        )
        assert result2.shape()[0] == pytest.approx(1234 * 0.33, 0.12)
        result3 = titanic_vd.copy().sample(x=0.33, method="systematic")
        assert result3.shape()[0] == pytest.approx(1234 * 0.33, 0.12)

        # testing with n
        result = titanic_vd.copy().sample(n=200, method="random")
        assert result.shape()[0] == pytest.approx(200, 0.12)
        result2 = titanic_vd.copy().sample(
            n=200, method="stratified", by=["age", "pclass",]
        )
        assert result2.shape()[0] == pytest.approx(200, 0.12)
        result3 = titanic_vd.copy().sample(n=200, method="systematic")
        assert result3.shape()[0] == pytest.approx(200, 0.12)
