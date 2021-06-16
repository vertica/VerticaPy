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

import pytest, datetime, warnings, os, verticapy
from verticapy import vDataFrame, drop, create_verticapy_schema
import matplotlib.pyplot as plt

from verticapy import set_option

set_option("print_info", False)


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
    drop(
        name="public.amazon", cursor=base.cursor,
    )

@pytest.fixture(scope="module")
def commodities_vd(base):
    from verticapy.datasets import load_commodities

    commodities = load_commodities(cursor=base.cursor)
    yield commodities
    drop(
        name="public.commodities", cursor=base.cursor,
    )


@pytest.fixture(scope="module")
def iris_vd(base):
    from verticapy.datasets import load_iris

    iris = load_iris(cursor=base.cursor)
    yield iris
    drop(
        name="public.iris", cursor=base.cursor,
    )


@pytest.fixture(scope="module")
def world_vd(base):
    from verticapy.datasets import load_world

    cities = load_world(cursor=base.cursor)
    yield cities
    with warnings.catch_warnings(record=True) as w:
        drop(
            name="public.world", cursor=base.cursor,
        )

@pytest.fixture(scope="module")
def pop_growth_vd(base):
    from verticapy.datasets import load_pop_growth

    pop_growth = load_pop_growth(cursor=base.cursor)
    yield pop_growth
    with warnings.catch_warnings(record=True) as w:
        drop(
            name="public.pop_growth", cursor=base.cursor,
        )

@pytest.fixture(scope="module")
def gapminder_vd(base):
    from verticapy.datasets import load_gapminder

    gapminder = load_gapminder(cursor=base.cursor)
    yield gapminder
    with warnings.catch_warnings(record=True) as w:
        drop(
            name="public.gapminder", cursor=base.cursor,
        )


class TestvDFPlot:

    def test_vDF_animated(self, pop_growth_vd, amazon_vd, commodities_vd, gapminder_vd):
        import matplotlib.animation as animation

        result = pop_growth_vd.animated("year", ["city", "population"], "continent", 1970, 1980, "bar", return_html=False)
        assert isinstance(result, animation.FuncAnimation)
        plt.close("all")
        result = pop_growth_vd.animated("year", ["city", "population"], "continent", 1970, 1980, "pie", return_html=False)
        assert isinstance(result, animation.FuncAnimation)
        plt.close("all")
        result = pop_growth_vd.animated("year", ["city", "population"], "", 1970, 1980, "bar", return_html=False)
        assert isinstance(result, animation.FuncAnimation)
        plt.close("all")
        result = pop_growth_vd.animated("year", ["city", "population"], "", 1970, 1980, "pie", return_html=False)
        assert isinstance(result, animation.FuncAnimation)
        plt.close("all")
        result = amazon_vd.animated("date", "number", kind="ts", by="state", return_html=False)
        assert isinstance(result, animation.FuncAnimation)
        plt.close("all")
        result = commodities_vd.animated("date", kind="ts", color = ["r", "g", "b"])
        assert isinstance(result, animation.FuncAnimation)
        plt.close("all")
        result = gapminder_vd.animated("year", ["lifeExp", "gdpPercap", "country", "pop",], "continent", kind="bubble", limit_labels=10, limit_over=100, return_html=False)
        assert isinstance(result, animation.FuncAnimation)
        plt.close("all")
        result = gapminder_vd.animated("year", ["lifeExp", "gdpPercap", "country",], "continent", kind="bubble", limit_labels=10, limit_over=100, return_html=False)
        assert isinstance(result, animation.FuncAnimation)
        plt.close("all")
        result = gapminder_vd.animated("year", ["lifeExp", "gdpPercap", "pop",], "continent", kind="bubble", limit_labels=10, limit_over=100, return_html=False)
        assert isinstance(result, animation.FuncAnimation)
        plt.close("all")
        result = gapminder_vd.animated("year", ["lifeExp", "gdpPercap",], "continent", kind="bubble", limit_labels=10, limit_over=100, return_html=False)
        assert isinstance(result, animation.FuncAnimation)
        plt.close("all")


    def test_vDF_stacked_area(self, amazon_vd):
        assert (
            len(
                amazon_vd.pivot("date", "state", "number")
                .stacked_area("date", ["ACRE", "BAHIA"], color="b",)
                .get_default_bbox_extra_artists()
            )
            == 12
        )
        plt.close("all")
        assert (
            len(
                amazon_vd.pivot("date", "state", "number")
                .stacked_area("date", ["ACRE", "BAHIA"], fully=True, color="b",)
                .get_default_bbox_extra_artists()
            )
            == 12
        )
        plt.close("all")

    def test_vDF_bar(self, titanic_vd, amazon_vd):
        # testing vDataFrame[].bar
        # auto
        result = titanic_vd["fare"].bar(color="b",)
        assert result.get_default_bbox_extra_artists()[0].get_width() == pytest.approx(
            0.7965964343598055
        )
        assert result.get_default_bbox_extra_artists()[1].get_width() == pytest.approx(
            0.12236628849270664
        )
        assert result.get_yticks()[1] == pytest.approx(42.694100000000006)
        plt.close("all")

        # auto + date
        result = amazon_vd["date"].bar(color="b",)
        assert result.get_default_bbox_extra_artists()[0].get_width() == pytest.approx(
            0.07530213820886272
        )
        assert result.get_default_bbox_extra_artists()[1].get_width() == pytest.approx(
            0.06693523396343352
        )
        assert result.get_yticks()[1] == pytest.approx(44705828.571428575)
        plt.close("all")

        # method=sum of=survived and bins=5
        result2 = titanic_vd["fare"].bar(method="sum", of="survived", bins=5, color="b",)
        assert result2.get_default_bbox_extra_artists()[0].get_width() == pytest.approx(
            391
        )
        assert result2.get_default_bbox_extra_artists()[1].get_width() == pytest.approx(
            34
        )
        assert result2.get_yticks()[1] == pytest.approx(102.46583999999999)
        plt.close("all")

        # testing vDataFrame.bar
        # auto & stacked
        for hist_type in ["auto", "stacked"]:
            result3 = titanic_vd.bar(
                columns=["pclass", "survived"],
                method="50%",
                of="fare",
                hist_type=hist_type,
                color="b",
            )
            assert result3.get_default_bbox_extra_artists()[
                0
            ].get_width() == pytest.approx(50.0)
            assert result3.get_default_bbox_extra_artists()[
                3
            ].get_width() == pytest.approx(77.9583)
            plt.close("all")
        # fully_stacked
        result4 = titanic_vd.bar(
            columns=["pclass", "survived"], hist_type="fully_stacked", color="b",
        )
        assert result4.get_default_bbox_extra_artists()[0].get_width() == pytest.approx(
            0.38782051282051283
        )
        assert result4.get_default_bbox_extra_artists()[3].get_width() == pytest.approx(
            0.6121794871794872
        )
        plt.close("all")
        # pyramid
        result5 = titanic_vd.bar(columns=["pclass", "survived"], hist_type="pyramid", color="b",)
        assert result5.get_default_bbox_extra_artists()[0].get_width() == pytest.approx(
            0.09805510534846029
        )
        assert result5.get_default_bbox_extra_artists()[3].get_width() == pytest.approx(
            -0.1547811993517018
        )
        plt.close("all")

    def test_vDF_boxplot(self, titanic_vd):
        # testing vDataFrame[].boxplot
        result = titanic_vd["age"].boxplot(color="b",)
        assert result.get_default_bbox_extra_artists()[0].get_data()[0][
            0
        ] == pytest.approx(16.07647847)
        assert result.get_default_bbox_extra_artists()[1].get_data()[0][
            0
        ] == pytest.approx(36.25)
        plt.close("all")
        result = titanic_vd["age"].boxplot(colors=["b", "r", "g",], by="pclass",)
        assert result.get_default_bbox_extra_artists()[0].get_data()[0][
            0
        ] == pytest.approx(1.0)
        assert result.get_default_bbox_extra_artists()[1].get_data()[0][
            0
        ] == pytest.approx(1.0)
        plt.close("all")

        # testing vDataFrame.boxplot
        result = titanic_vd.boxplot(columns=["age", "fare"], color="b",)
        assert result.get_default_bbox_extra_artists()[6].get_data()[1][
            0
        ] == pytest.approx(31.3875)
        assert result.get_default_bbox_extra_artists()[6].get_data()[1][
            1
        ] == pytest.approx(512.3292)
        plt.close("all")

    def test_vDF_bubble(self, iris_vd, titanic_vd,):
        # testing vDataFrame.bubble - img
        result = titanic_vd.bubble(columns=["fare", "age"], size_bubble_col="pclass", color="b", img=os.path.dirname(verticapy.__file__) + "/tests/vDataFrame/img_test.png", bbox=[0, 10, 0, 10],)
        result = result.get_default_bbox_extra_artists()[0]
        assert max([elem[0] for elem in result.get_offsets().data]) == 512.3292
        plt.close("all")
        # testing vDataFrame.bubble
        result = iris_vd.bubble(
            columns=["PetalLengthCm", "SepalLengthCm"], size_bubble_col="PetalWidthCm", color="b",
        )
        result = result.get_default_bbox_extra_artists()[0]
        assert max([elem[0] for elem in result.get_offsets().data]) == 6.9
        assert max([elem[1] for elem in result.get_offsets().data]) == 7.9
        plt.close("all")
        # testing vDataFrame.scatter using parameter catcol
        result2 = iris_vd.bubble(
            columns=["PetalLengthCm", "SepalLengthCm"],
            size_bubble_col="PetalWidthCm",
            catcol="Species",
            color="b",
        )
        result2 = result2.get_default_bbox_extra_artists()[0]
        assert max([elem[0] for elem in result2.get_offsets().data]) <= 6.9
        assert max([elem[1] for elem in result2.get_offsets().data]) <= 7.9
        plt.close("all")
        # testing vDataFrame.scatter using parameter cmap_col
        result3 = iris_vd.bubble(
            columns=["PetalLengthCm", "SepalLengthCm"],
            size_bubble_col="PetalWidthCm",
            cmap_col="SepalWidthCm",
        )
        result3 = result3.get_default_bbox_extra_artists()[0]
        assert max([elem[0] for elem in result3.get_offsets().data]) <= 6.9
        assert max([elem[1] for elem in result3.get_offsets().data]) <= 7.9
        plt.close("all")

    def test_vDF_density(self, iris_vd):
        # testing vDataFrame[].density
        try:
            create_verticapy_schema(iris_vd._VERTICAPY_VARIABLES_["cursor"])
        except:
            pass
        for kernel in ["gaussian", "logistic", "sigmoid", "silverman"]:
            result = iris_vd["PetalLengthCm"].density(kernel=kernel, nbins=20, color="b",)
            assert max(result.get_default_bbox_extra_artists()[1].get_data()[1]) < 0.25
            plt.close("all")
        for kernel in ["gaussian", "logistic", "sigmoid", "silverman"]:
            result = iris_vd["PetalLengthCm"].density(kernel=kernel, nbins=20, by="Species", color="b",)
            assert len(result.get_default_bbox_extra_artists()) < 20
            plt.close("all")
        # testing vDataFrame.density
        for kernel in ["gaussian", "logistic", "sigmoid", "silverman"]:
            result = iris_vd.density(kernel=kernel, nbins=20, color="b",)
            assert max(result.get_default_bbox_extra_artists()[5].get_data()[1]) < 0.37
            plt.close("all")

    def test_vDF_contour(self, titanic_vd):
        def func(a, b):
            return a + b + 1
        result = titanic_vd.contour(["parch", "sibsp"], func,)
        assert len(result.get_default_bbox_extra_artists()) == 32
        plt.close("all")
        result = titanic_vd.contour(["parch", "sibsp"], "parch + sibsp + 1",)
        assert len(result.get_default_bbox_extra_artists()) == 32
        plt.close("all")

    def test_vDF_geo_plot(self, world_vd):
        assert (
            len(
                world_vd["geometry"]
                .geo_plot(column="pop_est", cmap="Reds",)
                .get_default_bbox_extra_artists()
            )
            == 8
        )
        plt.close("all")

    def test_vDF_heatmap(self, iris_vd):
        result = iris_vd.heatmap(
            ["PetalLengthCm", "SepalLengthCm"],
            method="avg",
            of="SepalWidthCm",
            h=(1, 1),
        )
        assert result.get_default_bbox_extra_artists()[-2].get_size() == (5, 4)
        plt.close("all")

    def test_vDF_hexbin(self, titanic_vd):
        result = titanic_vd.hexbin(columns=["fare", "age"], img=os.path.dirname(verticapy.__file__) + "/tests/vDataFrame/img_test.png", bbox=[0, 10, 0, 10],)
        result = result.get_default_bbox_extra_artists()[0]
        assert max([elem[0] for elem in result.get_offsets()]) == pytest.approx(512.3292)
        plt.close("all")
        result = titanic_vd.hexbin(columns=["age", "fare"], method="avg", of="survived")
        result = result.get_default_bbox_extra_artists()[0]
        assert max([elem[0] for elem in result.get_offsets()]) == pytest.approx(
            80.00000007967, 1e-2
        )
        assert max([elem[1] for elem in result.get_offsets()]) == pytest.approx(
            512.3292, 1e-2
        )
        plt.close("all")

    def test_vDF_hchart(self, titanic_vd, amazon_vd):
        from highcharts.highcharts.highcharts import Highchart
        from highcharts.highstock.highstock import Highstock

        # boxplot
        result = titanic_vd.hchart(kind = "boxplot")
        assert isinstance(result, Highchart)
        # kendall
        result = titanic_vd.hchart(kind = "kendall")
        assert isinstance(result, Highchart)
        # cramer
        result = titanic_vd.hchart(kind = "cramer")
        assert isinstance(result, Highchart)
        # pearson
        result = titanic_vd.hchart(kind = "pearson")
        assert isinstance(result, Highchart)
        # spearman
        result = titanic_vd.hchart(kind = "spearman")
        assert isinstance(result, Highchart)
        # biserial
        result = titanic_vd.hchart(kind = "biserial")
        assert isinstance(result, Highchart)
        # area
        result = amazon_vd.hchart(x="date", y="number", kind = "area")
        assert isinstance(result, Highchart)
        result = amazon_vd.hchart(x="date", y="number", z="state", kind = "area")
        assert isinstance(result, Highchart)
        # line
        result = amazon_vd.hchart(x="date", y="number", kind = "line")
        assert isinstance(result, Highchart)
        result = amazon_vd.hchart(x="date", y="number", z="state", kind = "line")
        assert isinstance(result, Highchart)
        # spline
        result = amazon_vd.hchart(x="date", y="number", kind = "spline")
        assert isinstance(result, Highchart)
        result = amazon_vd.hchart(x="date", y="number", z="state", kind = "spline")
        assert isinstance(result, Highchart)
        # area_range
        result = amazon_vd.hchart(x="date", 
                                  y=["MIN(number)", "AVG(number)", "MAX(number)"], 
                                  kind = "area_range")
        assert isinstance(result, Highchart)
        # area_ts
        result = amazon_vd.hchart(x="date", y="number", kind = "area_ts")
        assert isinstance(result, Highchart)
        result = amazon_vd.hchart(x="date", y="number", z="state", kind = "area_ts")
        assert isinstance(result, Highchart)
        # bar1D
        result = titanic_vd.hchart(x="pclass", y="COUNT(*) AS cnt", kind = "bar")
        assert isinstance(result, Highchart)
        # hist1D
        result = titanic_vd.hchart(x="pclass", y="COUNT(*) AS cnt", kind = "hist")
        assert isinstance(result, Highchart)
        # donut
        result = titanic_vd.hchart(x="pclass", y="COUNT(*) AS cnt", kind = "donut")
        assert isinstance(result, Highchart)
        # donut3d
        result = titanic_vd.hchart(x="pclass", y="COUNT(*) AS cnt", kind = "donut3d")
        assert isinstance(result, Highchart)
        # pie
        result = titanic_vd.hchart(x="pclass", y="COUNT(*) AS cnt", kind = "pie")
        assert isinstance(result, Highchart)
        # pie_half
        result = titanic_vd.hchart(x="pclass", y="COUNT(*) AS cnt", kind = "pie_half")
        assert isinstance(result, Highchart)
        # pie3d
        result = titanic_vd.hchart(x="pclass", y="COUNT(*) AS cnt", kind = "pie3d")
        assert isinstance(result, Highchart)
        # bar2D / hist2D or drilldown
        result = titanic_vd.hchart(x="pclass", y="survived", z="COUNT(*) AS cnt", kind = "bar")
        assert isinstance(result, Highchart)
        result = titanic_vd.hchart(x="pclass", y="survived", z="COUNT(*) AS cnt", kind = "hist")
        assert isinstance(result, Highchart)
        result = titanic_vd.hchart(x="pclass", y="survived", z="COUNT(*) AS cnt", kind = "stacked_hist")
        assert isinstance(result, Highchart)
        result = titanic_vd.hchart(x="pclass", y="survived", z="COUNT(*) AS cnt", kind = "stacked_bar")
        assert isinstance(result, Highchart)
        result = titanic_vd.hchart(x="pclass", y="survived", z="COUNT(*) AS cnt", kind = "bar", drilldown=True)
        assert isinstance(result, Highchart)
        result = titanic_vd.hchart(x="pclass", y="survived", z="COUNT(*) AS cnt", kind = "hist", drilldown=True)
        assert isinstance(result, Highchart)
        result = titanic_vd.hchart(x="pclass", y="survived", z="COUNT(*) AS cnt", kind = "pie", drilldown=True)
        assert isinstance(result, Highchart)
        # bubble or scatter
        result = titanic_vd.hchart(x="age", y="fare", kind = "scatter",)
        assert isinstance(result, Highchart)
        result = titanic_vd.hchart(x="age", y="fare", c="survived", kind = "scatter",)
        assert isinstance(result, Highchart)
        result = titanic_vd.hchart(x="age", y="fare", z="parch", c="survived", kind = "scatter",)
        assert isinstance(result, Highchart)
        result = titanic_vd.hchart(x="age", y="fare", c="survived", kind = "bubble",)
        assert isinstance(result, Highchart)
        result = titanic_vd.hchart(x="age", y="fare", z="parch", c="survived", kind = "bubble",)
        assert isinstance(result, Highchart)
        # negative_bar
        result = titanic_vd.hchart(x="survived", y="age", z="COUNT(*) AS cnt", kind = "donut3d",)
        assert isinstance(result, Highchart)
        # spider
        result = titanic_vd.hchart(x="pclass",  kind = "spider",)
        assert isinstance(result, Highchart)
        # candlestick
        result = amazon_vd.hchart(x="date", y="number", kind = "candlestick")
        assert isinstance(result, Highstock)

    def test_vDF_hist(self, titanic_vd):
        # testing vDataFrame[].hist
        # auto
        result = titanic_vd["age"].hist(color="b")
        assert result.get_default_bbox_extra_artists()[0].get_height() == pytest.approx(
            0.050243111831442464
        )
        assert result.get_default_bbox_extra_artists()[1].get_height() == pytest.approx(
            0.029983792544570502
        )
        assert result.get_xticks()[1] == pytest.approx(7.24272727)
        plt.close("all")

        # method=avg of=survived and h=15
        result2 = titanic_vd["age"].hist(method="avg", of="survived", h=15, color="b")
        assert result2.get_default_bbox_extra_artists()[
            0
        ].get_height() == pytest.approx(0.534653465346535)
        assert result2.get_default_bbox_extra_artists()[
            1
        ].get_height() == pytest.approx(0.354838709677419)
        assert result2.get_xticks()[1] == pytest.approx(15)
        plt.close("all")

        # testing vDataFrame.hist
        # auto & stacked
        for hist_type in ["auto", "stacked"]:
            result3 = titanic_vd.hist(
                columns=["pclass", "sex"],
                method="avg",
                of="survived",
                hist_type=hist_type,
                color="b",
            )
            assert result3.get_default_bbox_extra_artists()[
                0
            ].get_height() == pytest.approx(0.964285714285714)
            assert result3.get_default_bbox_extra_artists()[
                3
            ].get_height() == pytest.approx(0.325581395348837)
            plt.close("all")
        # multi
        result4 = titanic_vd.hist(columns=["fare", "age"], hist_type="multi")
        assert result4.get_default_bbox_extra_artists()[
            0
        ].get_height() == pytest.approx(0.07374392220421394)
        assert result4.get_default_bbox_extra_artists()[
            1
        ].get_height() == pytest.approx(0.4327390599675851)
        plt.close("all")

    def test_vDF_pie(self, titanic_vd):
        # testing vDataFrame[].pie
        result = titanic_vd["pclass"].pie(method="avg", of="survived", colors=["b", "r",],)
        assert int(result.get_default_bbox_extra_artists()[6].get_text()) == 3
        assert float(
            result.get_default_bbox_extra_artists()[7].get_text()
        ) == pytest.approx(0.227753)
        plt.close("all")
        # testing vDataFrame.pie
        result = titanic_vd.pie(["sex", "pclass"], color="b",)
        assert result.get_default_bbox_extra_artists()[9].get_text() == "11.3%"
        plt.close("all")
        # testing vDataFrame[].pie - donut
        result = titanic_vd["sex"].pie(method="sum", of="survived", pie_type="donut", colors=["b", "r",],)
        assert result.get_default_bbox_extra_artists()[6].get_text() == "female"
        assert int(
            result.get_default_bbox_extra_artists()[7].get_text()
        ) == pytest.approx(302)
        plt.close("all")
        # testing vDataFrame[].pie - rose
        result = titanic_vd["sex"].pie(method="sum", of="survived", pie_type="rose",)
        assert len(result.get_default_bbox_extra_artists()) == 8
        plt.close("all")

    def test_vDF_pivot_table(self, titanic_vd):
        result = titanic_vd.pivot_table(
            columns=["age", "pclass"], method="avg", of="survived", cmap="Reds",
        )
        assert result[1][0] == pytest.approx(0.75)
        assert result[1][1] == pytest.approx(1.0)
        assert result[1][2] == pytest.approx(0.782608695652174)
        assert result[2][0] == pytest.approx(1.0)
        assert result[2][1] == pytest.approx(0.875)
        assert result[2][2] == pytest.approx(0.375)
        assert len(result[1]) == 12
        plt.close("all")

    def test_vDF_outliers_plot(self, titanic_vd):
        assert (
            len(titanic_vd.outliers_plot(["fare"]).get_default_bbox_extra_artists())
            == 24
        )
        plt.close("all")
        assert (
            len(
                titanic_vd.outliers_plot(
                    ["fare", "age"]
                ).get_default_bbox_extra_artists()
            )
            == 25
        )
        plt.close("all")

    def test_vDF_plot(self, amazon_vd):
        # testing vDataFrame[].plot
        result = amazon_vd["number"].plot(ts="date", by="state", color="b",)
        result = result.get_default_bbox_extra_artists()[0].get_data()
        assert len(result[0]) == len(result[1]) == pytest.approx(239, 1e-2)
        plt.close("all")

        # testing vDataFrame.plot
        result = amazon_vd.groupby(["date"], ["AVG(number) AS number"])
        result = result.plot(ts="date", columns=["number"], color="b",)
        result = result.get_default_bbox_extra_artists()[0].get_data()
        assert result[0][0] == datetime.date(1998, 1, 1)
        assert result[0][-1] == datetime.date(2017, 11, 1)
        assert result[1][0] == pytest.approx(0.0)
        assert result[1][-1] == pytest.approx(651.2962963)
        plt.close("all")

    def test_vDF_range_plot(self, amazon_vd):
        assert (
            len(
                amazon_vd["number"]
                .range_plot(ts="date", color="b",)
                .get_default_bbox_extra_artists()
            )
            == 10
        )
        plt.close("all")
        assert (
            len(
                amazon_vd["number"]
                .range_plot(ts="date", color="b",)
                .get_default_bbox_extra_artists()
            )
            == 10
        )
        plt.close("all")

    def test_vDF_scatter(self, iris_vd, titanic_vd,):
        # testing vDataFrame.scatter
        result = titanic_vd.scatter(columns=["fare", "age"], color="b", img=os.path.dirname(verticapy.__file__) + "/tests/vDataFrame/img_test.png", bbox=[0, 10, 0, 10],)
        result = result.get_default_bbox_extra_artists()[0]
        assert max([elem[0] for elem in result.get_offsets().data]) == 512.3292
        plt.close("all")
        result = iris_vd.scatter(columns=["PetalLengthCm", "SepalLengthCm"], color="b",)
        result = result.get_default_bbox_extra_artists()[0]
        assert max([elem[0] for elem in result.get_offsets().data]) == 6.9
        assert max([elem[1] for elem in result.get_offsets().data]) == 7.9
        plt.close("all")
        result2 = iris_vd.scatter(
            columns=["PetalLengthCm", "SepalLengthCm", "SepalWidthCm"], color="b",
        )
        result2 = result2.get_default_bbox_extra_artists()[0]
        assert max([elem[0] for elem in result2.get_offsets().data]) == 6.9
        assert max([elem[1] for elem in result2.get_offsets().data]) == 7.9
        plt.close("all")

        # testing vDataFrame.scatter using parameter catcol
        result3 = iris_vd.scatter(
            columns=["PetalLengthCm", "SepalLengthCm"], catcol="Species", color="b",
        )
        result3 = result3.get_default_bbox_extra_artists()[0]
        assert max([elem[0] for elem in result3.get_offsets().data]) <= 6.9
        assert max([elem[1] for elem in result3.get_offsets().data]) <= 7.9
        plt.close("all")
        result4 = iris_vd.scatter(
            columns=["PetalLengthCm", "SepalLengthCm", "SepalWidthCm"], catcol="Species", color="b",
        )
        result4 = result4.get_default_bbox_extra_artists()[0]
        assert max([elem[0] for elem in result3.get_offsets().data]) <= 6.9
        assert max([elem[1] for elem in result3.get_offsets().data]) <= 7.9
        plt.close("all")

    def test_vDF_scatter_matrix(self, iris_vd):
        result = iris_vd.scatter_matrix(color="b",)
        assert len(result) == 4
        plt.close("all")

    def test_vDF_spider(self, titanic_vd):
        result = titanic_vd["pclass"].spider("survived", color="b",)
        assert len(result.get_default_bbox_extra_artists()) == 9
        plt.close("all")
