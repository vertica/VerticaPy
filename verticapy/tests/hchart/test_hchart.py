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

# Standard Python Modules
import os

# Other Modules
from vertica_highcharts.highcharts.highcharts import Highchart
from vertica_highcharts.highstock.highstock import Highstock

# VerticaPy
import verticapy
from verticapy import drop, set_option
from verticapy.datasets import load_titanic, load_amazon
from verticapy.jupyter.extensions.chart_magic import chart_magic as hchart

set_option("print_info", False)


@pytest.fixture(scope="module")
def titanic_vd():
    titanic = load_titanic()
    yield titanic
    drop(name="public.titanic")


@pytest.fixture(scope="module")
def amazon_vd():
    amazon = load_amazon()
    yield amazon
    drop(name="public.titanic")


class Test_hchart:
    @pytest.mark.skip(
        reason="Deprecated, we need to implement the functions for each graphic"
    )
    def test_hchart(self, titanic_vd, amazon_vd):
        # Test -k
        result = hchart("-k pearson", "SELECT * FROM titanic;")
        assert isinstance(result, Highchart)
        result = hchart("--kind scatter", "SELECT age, fare FROM titanic;")
        assert isinstance(result, Highchart)
        result = hchart("    -k     scatter", "SELECT age, fare, pclass FROM titanic;")
        assert isinstance(result, Highchart)
        result = hchart("-k scatter", "SELECT age, fare, parch, pclass FROM titanic;")
        assert isinstance(result, Highchart)
        result = hchart("   --kind bubble", "SELECT age, fare, pclass FROM titanic;")
        assert isinstance(result, Highchart)
        result = hchart("-k bubble", "SELECT age, fare, parch, pclass FROM titanic;")
        assert isinstance(result, Highchart)
        result = hchart("-k auto", "SELECT age, fare, pclass FROM titanic;")
        assert isinstance(result, Highchart)
        result = hchart("-k auto", "SELECT age, fare, parch, pclass FROM titanic;")
        assert isinstance(result, Highchart)
        result = hchart("-k auto", "SELECT pclass, COUNT(*) FROM titanic GROUP BY 1;")
        assert isinstance(result, Highchart)
        result = hchart(
            "-k auto",
            "SELECT pclass, survived, COUNT(*) FROM titanic GROUP BY 1, 2;",
        )
        assert isinstance(result, Highchart)
        result = hchart("-k auto", "SELECT date, number FROM amazon;")
        assert isinstance(result, Highstock)
        result = hchart("    --kind auto", "SELECT date, number, state FROM amazon;")
        assert isinstance(result, Highstock)
        result = hchart("-k line", "SELECT date, number, state FROM amazon;")
        assert isinstance(result, Highstock)

        # Test -f
        result = hchart(
            "   -k line  -f   {}/tests/hchart/queries.sql".format(
                os.path.dirname(verticapy.__file__)
            ),
            "",
        )
        assert isinstance(result, Highstock)
        result = hchart(
            "   -k line  --file     {}/tests/hchart/queries.sql  ".format(
                os.path.dirname(verticapy.__file__)
            ),
            "",
        )
        assert isinstance(result, Highstock)

        # Test -c
        result = hchart(
            "   -k line  -c   'SELECT date, number, state FROM amazon;'",
            "",
        )
        assert isinstance(result, Highstock)
        result = hchart(
            '   -k line  --command   "SELECT date, number, state FROM amazon;"',
            "",
        )
        assert isinstance(result, Highstock)

        # Export to HTML --output
        result = hchart(
            "  --output   verticapy_test_hchart",
            "SELECT date, number, state FROM amazon;",
        )
        try:
            file = open("verticapy_test_hchart.html", "r", encoding="utf-8")
            result = file.read()
            assert "<!DOCTYPE html>" in result
        except:
            os.remove("verticapy_test_hchart.html")
            file.close()
            raise
        os.remove("verticapy_test_hchart.html")
        file.close()

        # Export to HTML -o
        result = hchart(
            "  -o   verticapy_test_hchart",
            "SELECT date, number, state FROM amazon;",
        )
        try:
            file = open("verticapy_test_hchart.html", "r", encoding="utf-8")
            result = file.read()
            assert "<!DOCTYPE html>" in result
        except:
            os.remove("verticapy_test_hchart.html")
            file.close()
            raise
        os.remove("verticapy_test_hchart.html")
        file.close()
