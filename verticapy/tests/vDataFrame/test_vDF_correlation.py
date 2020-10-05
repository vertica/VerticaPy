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
from verticapy import drop_table


@pytest.fixture(scope="module")
def titanic_vd(base):
    from verticapy.learn.datasets import load_titanic

    titanic = load_titanic(cursor=base.cursor)
    yield titanic
    drop_table(name = "public.titanic", cursor = base.cursor)

@pytest.fixture(scope="module")
def amazon_vd(base):
    from verticapy.learn.datasets import load_amazon

    amazon = load_amazon(cursor=base.cursor)
    yield amazon
    drop_table(name = "public.amazon", cursor = base.cursor)


class TestvDFCorrelation:
    def test_vDF_acf(self, amazon_vd):
        # spearmann method
        result1 = amazon_vd.acf(ts = "date", 
                                column = "number", 
                                p = 20,
                                by = ["state"],
                                unit = "month",
                                method = "spearman")

        assert result1.values["value"][0] == pytest.approx(1)
        assert result1.values["confidence"][0] == pytest.approx(0.0243968418)
        # the values are not very stable
        assert result1.values["value"][10] == pytest.approx(0.48, abs = 2e-3)
        assert result1.values["confidence"][10] == pytest.approx(0.068, abs = 5e-4)

        # pearson method
        result2 = amazon_vd.acf(ts = "date", 
                                column = "number", 
                                by = ["state"],
                                p = [1, 3, 6, 7],
                                unit = "year",
                                method = "pearson")

        assert result2.values["value"][0] == pytest.approx(1.0)
        assert result2.values["confidence"][0] == pytest.approx(0.0243968418)
        assert result2.values["value"][4] == pytest.approx(0.366, abs = 5e-3)
        assert result2.values["confidence"][4] == pytest.approx(0.0398389344)

        # Autocorrelation Heatmap for each 'month' lag
        result3 = amazon_vd.acf(ts = "date", 
                                column = "number", 
                                by = ["state"],
                                p = 12,
                                unit = "month",
                                method = "pearson", 
                                round_nb = 3, 
                                acf_type = "heatmap")

        assert result3.values["index"][1] == '"lag_12_number"'
        assert result3.values['"number"'][1] == pytest.approx(0.708)
        assert result3.values["index"][5] == '"lag_10_number"'
        assert result3.values['"number"'][5] == pytest.approx(0.299)

        # Autocorrelation Line for each 'month' lag
        result4 = amazon_vd.acf(ts = "date", 
                                column = "number", 
                                by = ["state"],
                                p = 12,
                                unit = "month",
                                method = "pearson",
                                acf_type = "line")

        assert result4.values["value"][1] == pytest.approx(0.66, abs = 5e-2)
        assert result4.values["confidence"][1] == pytest.approx(0.034, abs = 1e-2)
        assert result4.values["value"][6] == pytest.approx(-0.097)
        assert result4.values["confidence"][6] == pytest.approx(0.049, abs = 1e-2)

    def test_vDF_corr(self, titanic_vd):
        # Monotonic Correlation
        result1 = titanic_vd.corr(method = "spearman")
        assert result1.values['"age"'][0] == pytest.approx(-0.1670068536)
        assert result1.values['"age"'][1] == pytest.approx(-0.0909763969)

        # Categorical correlation
        result2 = titanic_vd.corr(method = "cramer")
        assert result2.values['"boat"'][0] == pytest.approx(0.4888480714)
        assert result2.values['"boat"'][1] == pytest.approx(0.4442208300)

        # Linear Correlation using only the response
        result3 = titanic_vd.corr(method = "pearson", focus = "survived")
        assert result3.values['"survived"'][1] == pytest.approx(-0.336, abs = 1e-3)
        assert result3.values['"survived"'][2] == pytest.approx(0.264, abs = 1e-3)

    def test_vDF_cov(self, titanic_vd):
        result1 = titanic_vd.cov()
        assert result1.values['"age"'][0] == pytest.approx(-4.879176375)
        assert result1.values['"age"'][1] == pytest.approx(-0.297583583)

        result2 = titanic_vd.cov(focus = "survived")
        assert result2.values['"survived"'][0] == pytest.approx(6.692, abs = 1e-3)
        assert result2.values['"survived"'][2] == pytest.approx(-0.298, abs = 1e-3)

    @pytest.mark.xfail(reason="Two issue: (1) internal need to a new cursor. (2) doesn't clean up a temp view named VERTICAPY_TEMP_MODEL_LINEAR_REGRESSION_VIEW_")
    def test_vDF_pacf(self, amazon_vd):
        # Partial Autocorrelation for each 'month' lag
        # p = 48: it will compute 48 'months' lags
        result1 = amazon_vd.pacf(ts = "date", 
                                 column = "number", 
                                 p = 48,
                                 by = ["state"],
                                 unit = "month")
        assert result1.values["value"][1] == pytest.approx(0.668, abs = 1e-3)
        assert result1.values["confidence"][1] == pytest.approx(0.033, abs = 1e-3)
        assert result1.values["value"][1] == pytest.approx(0.668, abs = 1e-3)
        assert result1.values["value"][24] == pytest.approx(0.213, abs = 1e-3)
        assert result1.values["confidence"][24] == pytest.approx(0.041, abs = 1e-3)


    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_regr(self):
        pass
