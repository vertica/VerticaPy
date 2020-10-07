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
    drop_table(name="public.titanic", cursor=base.cursor, print_info=False)


class TestvDFPlot:
    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_bar(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_boxplot(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_bubble(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_density(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_donut(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_hchart(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_hexbin(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_hist(self, titanic_vd):
        # testing vDataFrame[].hist
        result = titanic_vd["age"].hist()
        assert result.get_default_bbox_extra_artists()[0].get_height() == pytest.approx(
            0.050243111831442464
        )
        assert result.get_default_bbox_extra_artists()[1].get_height() == pytest.approx(
            0.029983792544570502
        )
        assert result.get_default_bbox_extra_artists()[2].get_height() == pytest.approx(
            0.13452188006482982
        )
        assert result.get_default_bbox_extra_artists()[3].get_height() == pytest.approx(
            0.1952998379254457
        )
        assert result.get_default_bbox_extra_artists()[4].get_height() == pytest.approx(
            0.17017828200972449
        )
        assert result.get_default_bbox_extra_artists()[5].get_height() == pytest.approx(
            0.08184764991896272
        )
        assert result.get_default_bbox_extra_artists()[6].get_height() == pytest.approx(
            0.06888168557536467
        )
        assert result.get_default_bbox_extra_artists()[7].get_height() == pytest.approx(
            0.03727714748784441
        )
        assert result.get_default_bbox_extra_artists()[8].get_height() == pytest.approx(
            0.031604538087520256
        )
        assert result.get_default_bbox_extra_artists()[9].get_height() == pytest.approx(
            0.005672609400324149
        )
        assert result.get_default_bbox_extra_artists()[
            10
        ].get_height() == pytest.approx(0.0016207455429497568)
        assert result.get_default_bbox_extra_artists()[
            11
        ].get_height() == pytest.approx(0.0008103727714748784)
        assert result.get_xticks()[0] == pytest.approx(0.0)
        assert result.get_xticks()[1] == pytest.approx(7.24272727)
        assert result.get_xticks()[2] == pytest.approx(7.24272727 * 2)
        assert result.get_xticks()[3] == pytest.approx(7.24272727 * 3)
        assert result.get_xticks()[4] == pytest.approx(7.24272727 * 4)
        assert result.get_xticks()[5] == pytest.approx(7.24272727 * 5)
        assert result.get_xticks()[6] == pytest.approx(7.24272727 * 6)
        assert result.get_xticks()[7] == pytest.approx(7.24272727 * 7)
        assert result.get_xticks()[8] == pytest.approx(7.24272727 * 8)
        assert result.get_xticks()[9] == pytest.approx(7.24272727 * 9)
        assert result.get_xticks()[10] == pytest.approx(7.24272727 * 10)
        assert result.get_xticks()[11] == pytest.approx(7.24272727 * 11)

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_pie(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_pivot_table(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_plot(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_scatter(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_scatter_matrix(self):
        pass
