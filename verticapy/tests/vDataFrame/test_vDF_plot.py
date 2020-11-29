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

import pytest, datetime, warnings
from verticapy import vDataFrame, drop_table, create_verticapy_schema
import matplotlib.pyplot as plt

from verticapy import set_option

set_option("print_info", False)


@pytest.fixture(scope="module")
def titanic_vd(base):
    from verticapy.learn.datasets import load_titanic

    titanic = load_titanic(cursor=base.cursor)
    yield titanic
    with warnings.catch_warnings(record=True) as w:
        drop_table(
            name="public.titanic", cursor=base.cursor,
        )


@pytest.fixture(scope="module")
def amazon_vd(base):
    from verticapy.learn.datasets import load_amazon

    amazon = load_amazon(cursor=base.cursor)
    yield amazon
    drop_table(
        name="public.amazon", cursor=base.cursor,
    )


@pytest.fixture(scope="module")
def iris_vd(base):
    from verticapy.learn.datasets import load_iris

    iris = load_iris(cursor=base.cursor)
    yield iris
    drop_table(
        name="public.iris", cursor=base.cursor,
    )


class TestvDFPlot:
    def test_vDF_bar(self, titanic_vd):
        # testing vDataFrame[].bar
        # auto
        result = titanic_vd["fare"].bar()
        assert result.get_default_bbox_extra_artists()[0].get_width() == pytest.approx(
            0.7965964343598055
        )
        assert result.get_default_bbox_extra_artists()[1].get_width() == pytest.approx(
            0.12236628849270664
        )
        assert result.get_yticks()[1] == pytest.approx(42.694100000000006)

        # method=sum of=survived and bins=5
        result2 = titanic_vd["fare"].bar(method="sum", of="survived", bins=5)
        assert result2.get_default_bbox_extra_artists()[0].get_width() == pytest.approx(
            391
        )
        assert result2.get_default_bbox_extra_artists()[1].get_width() == pytest.approx(
            34
        )
        assert result2.get_yticks()[1] == pytest.approx(102.46583999999999)

        # testing vDataFrame.bar
        # auto & stacked
        for hist_type in ["auto", "stacked"]:
            result3 = titanic_vd.bar(
                columns=["pclass", "survived"],
                method="50%",
                of="fare",
                hist_type=hist_type,
            )
            assert result3.get_default_bbox_extra_artists()[
                0
            ].get_width() == pytest.approx(50.0)
            assert result3.get_default_bbox_extra_artists()[
                3
            ].get_width() == pytest.approx(77.9583)
        # fully_stacked
        result4 = titanic_vd.bar(
            columns=["pclass", "survived"], hist_type="fully_stacked"
        )
        assert result4.get_default_bbox_extra_artists()[0].get_width() == pytest.approx(
            0.38782051282051283
        )
        assert result4.get_default_bbox_extra_artists()[3].get_width() == pytest.approx(
            0.6121794871794872
        )
        plt.close()

    def test_vDF_boxplot(self, titanic_vd):
        # testing vDataFrame[].boxplot
        result = titanic_vd["age"].boxplot()
        assert result.get_default_bbox_extra_artists()[0].get_data()[0][
            0
        ] == pytest.approx(16.07647847)
        assert result.get_default_bbox_extra_artists()[1].get_data()[0][
            0
        ] == pytest.approx(36.25)

        # testing vDataFrame.boxplot
        result = titanic_vd.boxplot(columns=["age", "fare"])
        assert result.get_default_bbox_extra_artists()[6].get_data()[1][
            0
        ] == pytest.approx(31.3875)
        assert result.get_default_bbox_extra_artists()[6].get_data()[1][
            1
        ] == pytest.approx(512.3292)
        plt.close()

    def test_vDF_bubble(self, iris_vd):
        # testing vDataFrame.bubble
        result = iris_vd.bubble(
            columns=["PetalLengthCm", "SepalLengthCm"], size_bubble_col="PetalWidthCm"
        )
        result = result.get_default_bbox_extra_artists()[0]
        assert max([elem[0] for elem in result.get_offsets().data]) == 6.9
        assert max([elem[1] for elem in result.get_offsets().data]) == 7.9
        # testing vDataFrame.scatter using parameter catcol
        result2 = iris_vd.bubble(
            columns=["PetalLengthCm", "SepalLengthCm"],
            size_bubble_col="PetalWidthCm",
            catcol="Species",
        )
        result2 = result2.get_default_bbox_extra_artists()[0]
        assert max([elem[0] for elem in result2.get_offsets().data]) <= 6.9
        assert max([elem[1] for elem in result2.get_offsets().data]) <= 7.9
        plt.close()

    def test_vDF_density(self, iris_vd):
        # testing vDataFrame[].density
        try:
            create_verticapy_schema(iris_vd._VERTICAPY_VARIABLES_["cursor"])
        except:
            pass
        for kernel in ["gaussian", "logistic", "sigmoid", "silverman"]:
            result = iris_vd["PetalLengthCm"].density(kernel=kernel, nbins=20)
            assert max(result.get_default_bbox_extra_artists()[1].get_data()[1]) < 0.25
            plt.close()
        # testing vDataFrame.density
        for kernel in ["gaussian", "logistic", "sigmoid", "silverman"]:
            result = iris_vd.density(kernel=kernel, nbins=20)
            assert max(result.get_default_bbox_extra_artists()[5].get_data()[1]) < 0.37
            plt.close()

    def test_vDF_donut(self, titanic_vd):
        result = titanic_vd["sex"].donut(method="sum", of="survived")
        assert result.get_default_bbox_extra_artists()[6].get_text() == "female"
        assert int(
            result.get_default_bbox_extra_artists()[7].get_text()
        ) == pytest.approx(302)
        plt.close()

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_hchart(self):
        pass

    def test_vDF_heatmap(self, iris_vd):
        result = iris_vd.heatmap(
            ["PetalLengthCm", "SepalLengthCm"],
            method="avg",
            of="SepalWidthCm",
            h=(1, 1),
        )
        assert result.get_default_bbox_extra_artists()[-2].get_size() == (5, 4)
        plt.close()

    def test_vDF_hexbin(self, titanic_vd):
        result = titanic_vd.hexbin(columns=["age", "fare"], method="avg", of="survived")
        result = result.get_default_bbox_extra_artists()[0]
        assert max([elem[0] for elem in result.get_offsets()]) == pytest.approx(
            78.0082500756865, 1e-2
        )
        assert max([elem[1] for elem in result.get_offsets()]) == pytest.approx(
            512.3292, 1e-2
        )
        plt.close()

    def test_vDF_hist(self, titanic_vd):
        # testing vDataFrame[].hist
        # auto
        result = titanic_vd["age"].hist()
        assert result.get_default_bbox_extra_artists()[0].get_height() == pytest.approx(
            0.050243111831442464
        )
        assert result.get_default_bbox_extra_artists()[1].get_height() == pytest.approx(
            0.029983792544570502
        )
        assert result.get_xticks()[1] == pytest.approx(7.24272727)

        # method=avg of=survived and h=15
        result2 = titanic_vd["age"].hist(method="avg", of="survived", h=15)
        assert result2.get_default_bbox_extra_artists()[
            0
        ].get_height() == pytest.approx(0.534653465346535)
        assert result2.get_default_bbox_extra_artists()[
            1
        ].get_height() == pytest.approx(0.354838709677419)
        assert result2.get_xticks()[1] == pytest.approx(15)

        # testing vDataFrame.hist
        # auto & stacked
        for hist_type in ["auto", "stacked"]:
            result3 = titanic_vd.hist(
                columns=["pclass", "sex"],
                method="avg",
                of="survived",
                hist_type=hist_type,
            )
            assert result3.get_default_bbox_extra_artists()[
                0
            ].get_height() == pytest.approx(0.964285714285714)
            assert result3.get_default_bbox_extra_artists()[
                3
            ].get_height() == pytest.approx(0.325581395348837)
        # multi
        result4 = titanic_vd.hist(columns=["fare", "age"], hist_type="multi")
        assert result4.get_default_bbox_extra_artists()[
            0
        ].get_height() == pytest.approx(0.07374392220421394)
        assert result4.get_default_bbox_extra_artists()[
            1
        ].get_height() == pytest.approx(0.4327390599675851)
        plt.close()

    def test_vDF_pie(self, titanic_vd):
        result = titanic_vd["pclass"].pie(method="avg", of="survived")
        assert int(result.get_default_bbox_extra_artists()[6].get_text()) == 3
        assert float(
            result.get_default_bbox_extra_artists()[7].get_text()
        ) == pytest.approx(0.227753)
        plt.close()

    def test_vDF_pivot_table(self, titanic_vd):
        result = titanic_vd.pivot_table(
            columns=["age", "pclass"], method="avg", of="survived"
        )
        assert result[1][0] == pytest.approx(0.75)
        assert result[1][1] == pytest.approx(1.0)
        assert result[1][2] == pytest.approx(0.782608695652174)
        assert result[2][0] == pytest.approx(1.0)
        assert result[2][1] == pytest.approx(0.875)
        assert result[2][2] == pytest.approx(0.375)
        assert len(result[1]) == 12
        plt.close()

    def test_vDF_plot(self, amazon_vd):
        # testing vDataFrame[].plot
        result = amazon_vd["number"].plot(ts="date", by="state")
        result = result.get_default_bbox_extra_artists()[0].get_data()
        assert len(result[0]) == len(result[1]) == pytest.approx(239, 1e-2)

        # testing vDataFrame.plot
        result = amazon_vd.groupby(["date"], ["AVG(number) AS number"])
        result = result.plot(ts="date", columns=["number"])
        result = result.get_default_bbox_extra_artists()[0].get_data()
        assert result[0][0] == datetime.date(1998, 1, 1)
        assert result[0][-1] == datetime.date(2017, 11, 1)
        assert result[1][0] == pytest.approx(0.0)
        assert result[1][-1] == pytest.approx(651.2962963)
        plt.close()

    def test_vDF_scatter(self, iris_vd):
        # testing vDataFrame.scatter
        result = iris_vd.scatter(columns=["PetalLengthCm", "SepalLengthCm"])
        result = result.get_default_bbox_extra_artists()[0]
        assert max([elem[0] for elem in result.get_offsets().data]) == 6.9
        assert max([elem[1] for elem in result.get_offsets().data]) == 7.9
        result2 = iris_vd.scatter(
            columns=["PetalLengthCm", "SepalLengthCm", "SepalWidthCm"]
        )
        result2 = result2.get_default_bbox_extra_artists()[0]
        assert max([elem[0] for elem in result2.get_offsets().data]) == 6.9
        assert max([elem[1] for elem in result2.get_offsets().data]) == 7.9

        # testing vDataFrame.scatter using parameter catcol
        result3 = iris_vd.scatter(
            columns=["PetalLengthCm", "SepalLengthCm"], catcol="Species"
        )
        result3 = result3.get_default_bbox_extra_artists()[0]
        assert max([elem[0] for elem in result3.get_offsets().data]) <= 6.9
        assert max([elem[1] for elem in result3.get_offsets().data]) <= 7.9
        result3 = iris_vd.scatter(
            columns=["PetalLengthCm", "SepalLengthCm", "SepalWidthCm"], catcol="Species"
        )
        result3 = result3.get_default_bbox_extra_artists()[0]
        assert max([elem[0] for elem in result3.get_offsets().data]) <= 6.9
        assert max([elem[1] for elem in result3.get_offsets().data]) <= 7.9
        plt.close()

    def test_vDF_scatter_matrix(self, iris_vd):
        result = iris_vd.scatter_matrix()
        assert len(result) == 4
        plt.close()
