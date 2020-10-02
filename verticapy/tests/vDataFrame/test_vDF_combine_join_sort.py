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
def iris_vd(base):
    from verticapy.learn.datasets import load_iris

    iris = load_iris(cursor=base.cursor)
    yield iris
    drop_table(name = "public.iris", cursor = base.cursor)

@pytest.fixture(scope="module")
def market_vd(base):
    from verticapy.learn.datasets import load_market

    market = load_market(cursor=base.cursor)
    yield market
    drop_table(name = "public.market", cursor = base.cursor)

class TestvDFCombineJoinSort:
    @pytest.mark.xfail(reason="UNION_ALL cannot be turned off")
    def test_vDF_append(self, iris_vd):
        assert iris_vd.shape() == (150, 5)

        result_vDF = iris_vd.append(iris_vd)
        assert result_vDF.shape() == (300, 5), "testing vDataFrame.append(vDataFrame) failed"

        result_vDF = iris_vd.append("public.iris")
        assert result_vDF.shape() == (300, 5), "testing vDataFrame.append(str) failed"

        result_vDF = iris_vd.append(
            iris_vd,
            expr1=["SepalLengthCm AS sl", "PetalLengthCm AS pl"],
            expr2=["SepalLengthCm AS sl", "PetalLengthCm AS pl"],
        )
        assert result_vDF.shape() == (300, 2), "testing vDataFrame.append(vDataFrame, expr1, expr2) failed"

        # the duplicate rows 
        result_vDF = iris_vd.append(iris_vd, union_all=False)
        assert result_vDF.shape() == (150, 5), "testing vDataFrame.append(vDataFrame, union_all) failed"

    def testvDF_groupby(self, market_vd):
        result1 = market_vd.groupby(
            columns=["Form", "Name"],
            expr=["AVG(Price) AS avg_price", "STDDEV(Price) AS std"]
        )
        assert result1.shape() == (159, 4), "testing vDataFrame.groupby(columns, expr) failed"

        # check parameter
        from verticapy.errors import MissingColumn
        with pytest.raises(MissingColumn) as exception_info:
            result2 = market_vd.groupby(
                columns=["For", "Name"],
                expr=["AVG(Price) AS avg_price", "STDDEV(Price) AS std"],
                check = True
            )
        assert exception_info.match("The Virtual Column 'for' doesn't exist")

        from vertica_python.errors import VerticaSyntaxError
        with pytest.raises(VerticaSyntaxError) as exception_info:
            result2 = market_vd.groupby(
                columns=["For", "Name"],
                expr=["AVG(Price) AS avg_price", "STDDEV(Price) AS std"],
                check = False
            )
        assert exception_info.match("Syntax error at or near \"For\"")

    @pytest.mark.xfail(reason="Always returns the result of inner join")
    def test_vDF_join(self, market_vd):
        # CREATE TABLE not_fresh AS SELECT * FROM market WHERE Form != 'Fresh';
        not_fresh = market_vd.filter(expr = "Form != 'Fresh'")
        # CREATE TABLE not_dried AS SELECT * FROM market WHERE Form != 'Dried';
        not_dried = market_vd.filter(expr = "Form != 'Dried'")

        # CREATE TABLE left_join AS
        #        SELECT a.Name as Name1, b.Name as Name2
        #        FROM not_fresh AS a LEFT JOIN not_dried AS b ON a.Form = b.Form;
        left_join = not_fresh.join(not_dried, how = "left", on = {"Form": "Form"},
                                   expr1 = ["Name AS Name1"],
                                   expr2 = ["Name AS Name2"])
        assert left_join.shape() == (5886, 2)
        # SELECT COUNT(*) FROM left_join WHERE Name1 IS NULL;
        assert left_join["Name1"].count() == 0
        # SELECT COUNT(*) FROM left_join WHERE Name2 IS NULL;
        assert left_join["Name2"].count() == 30

        # CREATE TABLE right_join AS
        #        SELECT a.Name as Name1, b.Name as Name2
        #        FROM not_fresh AS a RIGHT JOIN not_dried AS b ON a.Form = b.Form;
        right_join = not_fresh.join(not_dried, how = "right", on = {"Form": "Form"},
                                    expr1 = ["Name AS Name1"],
                                    expr2 = ["Name AS Name2"])
        assert right_join.shape() == (5946, 2)
        # SELECT COUNT(*) FROM right_join WHERE Name1 IS NULL;
        assert right_join["Name1"].count() == 90
        # SELECT COUNT(*) FROM right_join WHERE Name2 IS NULL;
        assert right_join["Name2"].count() == 0

        # CREATE TABLE full_join AS
        #        SELECT a.Name as Name1, b.Name as Name2
        #        FROM not_fresh AS a FULL OUTER JOIN not_dried AS b ON a.Form = b.Form;
        full_join = not_fresh.join(not_dried, how = "full", on = {"Form": "Form"},
                                   expr1 = ["Name AS Name1"],
                                   expr2 = ["Name AS Name2"])
        assert full_join.shape() == (5976, 2)
        # SELECT COUNT(*) FROM full_join WHERE Name1 IS NULL;
        assert full_join["Name1"].count() == 90
        # SELECT COUNT(*) FROM full_join WHERE Name2 IS NULL;
        assert full_join["Name2"].count() == 30

        # CREATE TABLE inner_join AS
        #        SELECT a.Name as Name1, b.Name as Name2
        #        FROM not_fresh AS a INNER JOIN not_dried AS b ON a.Form = b.Form;
        inner_join = not_fresh.join(not_dried, how = "inner", on = {"Form": "Form"},
                                    expr1 = ["Name AS Name1"],
                                    expr2 = ["Name AS Name2"])
        assert inner_join.shape() == (5856, 2)
        # SELECT COUNT(*) FROM inner_join WHERE Name1 IS NULL;
        assert inner_join["Name1"].count() == 0
        # SELECT COUNT(*) FROM inner_join WHERE Name2 IS NULL;
        assert inner_join["Name2"].count() == 0

        # CREATE TABLE natural_join AS
        #        SELECT a.Name as Name1, b.Name as Name2
        #        FROM not_fresh AS a NATURAL JOIN not_dried AS b;
        natural_join = not_fresh.join(not_dried, how = "natural",
                                      expr1 = ["Name AS Name1"],
                                      expr2 = ["Name AS Name2"])
        assert natural_join.shape() == (194, 2)
        # SELECT COUNT(*) FROM natural_join WHERE Name1 IS NULL;
        assert natural_join["Name1"].count() == 0
        # SELECT COUNT(*) FROM natural_join WHERE Name2 IS NULL;
        assert natural_join["Name2"].count() == 0

        # CREATE TABLE cross_join AS
        #        SELECT a.Name as Name1, b.Name as Name2
        #        FROM not_fresh AS a CROSS JOIN not_dried AS b;
        corss_join = not_fresh.join(not_dried, how = "cross",
                                    expr1 = ["Name AS Name1"],
                                    expr2 = ["Name AS Name2"])
        assert cross_join.shape() == (63616, 2)
        # SELECT COUNT(*) FROM cross_join WHERE Name1 IS NULL;
        assert cross_join["Name1"].count() == 0
        # SELECT COUNT(*) FROM cross_join WHERE Name2 IS NULL;
        assert cross_join["Name2"].count() == 0

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_narrow(self, base):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_pivot(self, base):
        pass

    @pytest.mark.skip(reason="test not implemented")
    def test_vDF_sort(self, base):
        pass
