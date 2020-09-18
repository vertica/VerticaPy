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

class TestvDFDescriptiveStat():

    def test_vDF_aad(self, base):
        from verticapy.learn.datasets import load_titanic
        titanic = load_titanic(cursor = base.cursor)

        result = titanic.aad(columns = ["age", "fare", "parch"])
        assert result.values["aad"][0] == pytest.approx(11.254785419447906)
        assert result.values["aad"][1] == pytest.approx(30.625865942462237)
        assert result.values["aad"][2] == pytest.approx(0.5820801231451393)

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_agg(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_all(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_any(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_avg(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_count(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_describe(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_distinct(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_duplicated(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_isin(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_kurt(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_mad(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_max(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_median(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_min(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_mode(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_nlargest(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_nsmallest(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_nunique(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_numh(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_prod(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_quantile(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_score(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_sem(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_shape(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_skew(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_statistics(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_std(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_sum(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_topk(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_value_counts(self):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_var(self):
        pass
