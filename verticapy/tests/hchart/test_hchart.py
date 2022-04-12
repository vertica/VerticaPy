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
from verticapy import drop, set_option
from verticapy.connect import *
from verticapy.hchart import *

set_option("print_info", False)


@pytest.fixture(scope="module")
def titanic_vd():
    from verticapy.datasets import load_titanic

    titanic = load_titanic()
    yield titanic
    with warnings.catch_warnings(record=True) as w:
        drop(name="public.titanic")


@pytest.fixture(scope="module")
def amazon_vd():
    from verticapy.datasets import load_amazon

    amazon = load_amazon()
    yield amazon
    with warnings.catch_warnings(record=True) as w:
        drop(name="public.titanic")


class Test_hchart:
    def test_hchart(self, titanic_vd, amazon_vd):
        from highcharts.highcharts.highcharts import Highchart
        from highcharts.highstock.highstock import Highstock

        result = hchart("-type pearson", "SELECT * FROM titanic;")
        assert isinstance(result, Highchart)
        result = hchart("-type scatter", "SELECT age, fare FROM titanic;")
        assert isinstance(result, Highchart)
        result = hchart("-type scatter", "SELECT age, fare, pclass FROM titanic;")
        assert isinstance(result, Highchart)
        result = hchart(
            "-type scatter", "SELECT age, fare, parch, pclass FROM titanic;"
        )
        assert isinstance(result, Highchart)
        result = hchart("-type bubble", "SELECT age, fare, pclass FROM titanic;")
        assert isinstance(result, Highchart)
        result = hchart("-type bubble", "SELECT age, fare, parch, pclass FROM titanic;")
        assert isinstance(result, Highchart)
        result = hchart("-type auto", "SELECT age, fare, pclass FROM titanic;")
        assert isinstance(result, Highchart)
        result = hchart("-type auto", "SELECT age, fare, parch, pclass FROM titanic;")
        assert isinstance(result, Highchart)
        result = hchart(
            "-type auto", "SELECT pclass, COUNT(*) FROM titanic GROUP BY 1;"
        )
        assert isinstance(result, Highchart)
        result = hchart(
            "-type auto",
            "SELECT pclass, survived, COUNT(*) FROM titanic GROUP BY 1, 2;",
        )
        assert isinstance(result, Highchart)
        result = hchart("-type auto", "SELECT date, number FROM amazon;")
        assert isinstance(result, Highstock)
        result = hchart("-type auto", "SELECT date, number, state FROM amazon;")
        assert isinstance(result, Highstock)
        result = hchart("-type line", "SELECT date, number, state FROM amazon;")
        assert isinstance(result, Highstock)
