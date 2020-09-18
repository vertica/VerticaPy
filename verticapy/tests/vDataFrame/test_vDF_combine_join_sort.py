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

class TestvDFCombineJoinSort():

    def test_vDF_append(self, base):
        from verticapy.learn.datasets import load_iris
        iris = load_iris(cursor = base.cursor)

        result_vDF = iris.append(iris)
        assert result_vDF.shape() == (300, 5), "testing vDataFrame.append(vDataFrame) failed"

        result_vDF = iris.append("public.iris")
        assert result_vDF.shape() == (300, 5), "testing vDataFrame.append(str) failed"

        result_vDF = iris.append(iris,
                                 expr1 = ["SepalLengthCm AS sl", "PetalLengthCm AS pl"],
                                 expr2 = ["SepalLengthCm AS sl", "PetalLengthCm AS pl"])
        assert result_vDF.shape() == (300, 2), "testing vDataFrame.append(vDataFrame, expr1, expr2) failed"

        result_vDF = iris.append(iris, union_all=False)
        assert result_vDF.shape() == (300, 5), "testing vDataFrame.append(vDataFrame, union_all) failed"

        # TODO: add at least one test where the union_all parameter makes a difference

    def testvDF_groupby(self, base):
        from verticapy.learn.datasets import load_market
        market = load_market(cursor = base.cursor)

        result_vDF = market.groupby(columns = ["Form", "Name"], expr = ["AVG(Price) AS avg_price",
                                                                        "STDDEV(Price) AS std"])
        assert result_vDF.shape() == (159, 4), "testing vDataFrame.groupby(columns, expr) failed"

        # TODO: add tests for other combination of parameters

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_join(self, base):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_narrow(self, base):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_pivot(self, base):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_sort(self, base):
        pass
