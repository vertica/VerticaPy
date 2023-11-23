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
import datetime, os, sys

# Other Modules
import matplotlib.pyplot as plt
from vertica_highcharts.highcharts.highcharts import Highchart
from vertica_highcharts.highstock.highstock import Highstock
from IPython.display import HTML

# VerticaPy
import verticapy
from verticapy import drop, set_option
from verticapy.datasets import (
    load_titanic,
    load_amazon,
    load_commodities,
    load_iris,
    load_world,
    load_pop_growth,
    load_gapminder,
)

# Matplotlib skip
import matplotlib

matplotlib_version = matplotlib.__version__
skip_plt = pytest.mark.skipif(
    matplotlib_version > "3.5.2",
    reason="Test skipped on matplotlib version greater than 3.5.2",
)

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
    drop(name="public.amazon")


@pytest.fixture(scope="module")
def commodities_vd():
    commodities = load_commodities()
    yield commodities
    drop(name="public.commodities")


@pytest.fixture(scope="module")
def iris_vd():
    iris = load_iris()
    yield iris
    drop(name="public.iris")


@pytest.fixture(scope="module")
def world_vd():
    cities = load_world()
    yield cities
    drop(name="public.world")


@pytest.fixture(scope="module")
def pop_growth_vd():
    pop_growth = load_pop_growth()
    yield pop_growth
    drop(name="public.pop_growth")


@pytest.fixture(scope="module")
def gapminder_vd():
    gapminder = load_gapminder()
    yield gapminder
    drop(name="public.gapminder")


class TestvDFPlot:
    @skip_plt
    def test_vDF_animated(self, pop_growth_vd, amazon_vd, commodities_vd, gapminder_vd):
        result = pop_growth_vd.animated_bar(
            "year",
            ["city", "population"],
            "continent",
            1970,
            1980,
        )
        assert isinstance(result, HTML)
        plt.close("all")
        result = pop_growth_vd.animated_pie(
            "year",
            ["city", "population"],
            "continent",
            1970,
            1980,
        )
        assert isinstance(result, HTML)
        plt.close("all")
        result = pop_growth_vd.animated_bar(
            "year",
            ["city", "population"],
            "",
            1970,
            1980,
        )
        assert isinstance(result, HTML)
        plt.close("all")
        result = pop_growth_vd.animated_pie(
            "year",
            ["city", "population"],
            "",
            1970,
            1980,
        )
        assert isinstance(result, HTML)
        plt.close("all")
        result = amazon_vd.animated_plot(
            "date",
            "number",
            by="state",
        )
        assert isinstance(result, HTML)
        plt.close("all")
        result = commodities_vd.animated_plot("date", color=["r", "g", "b"])
        assert isinstance(result, HTML)
        plt.close("all")
        result = gapminder_vd.animated_scatter(
            "year",
            ["lifeExp", "gdpPercap", "country", "pop"],
            "continent",
            limit_labels=10,
            limit_over=100,
        )
        assert isinstance(result, HTML)
        plt.close("all")
        result = gapminder_vd.animated_scatter(
            "year",
            ["lifeExp", "gdpPercap", "country"],
            "continent",
            limit_labels=10,
            limit_over=100,
        )
        assert isinstance(result, HTML)
        plt.close("all")
        result = gapminder_vd.animated_scatter(
            "year",
            ["lifeExp", "gdpPercap", "pop"],
            "continent",
            limit_labels=10,
            limit_over=100,
        )
        assert isinstance(result, HTML)
        plt.close("all")
        result = gapminder_vd.animated_scatter(
            "year",
            ["lifeExp", "gdpPercap"],
            "continent",
            limit_labels=10,
            limit_over=100,
        )
        assert isinstance(result, HTML)
        plt.close("all")

    @skip_plt
    def test_vDF_geo_plot(self, world_vd):
        assert (
            len(
                world_vd["geometry"]
                .geo_plot(column="pop_est", cmap="Reds")
                .get_default_bbox_extra_artists()
            )
            == 8
        )
        plt.close("all")

    @skip_plt
    def test_vDF_scatter_matrix(self, iris_vd):
        result = iris_vd.scatter_matrix(color="b")
        assert len(result) == 4
        plt.close("all")
