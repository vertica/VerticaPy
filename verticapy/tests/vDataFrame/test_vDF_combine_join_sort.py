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

# VerticaPy
from verticapy.errors import MissingColumn
from verticapy import drop, set_option, TableSample
from verticapy.datasets import load_iris, load_market, load_amazon, load_titanic

set_option("print_info", False)


@pytest.fixture(scope="module")
def iris_vd():
    iris = load_iris()
    yield iris
    drop(name="public.iris")


@pytest.fixture(scope="module")
def market_vd():
    market = load_market()
    yield market
    drop(name="public.market")


@pytest.fixture(scope="module")
def amazon_vd():
    amazon = load_amazon()
    yield amazon
    drop(name="public.amazon")


@pytest.fixture(scope="module")
def titanic_vd():
    titanic = load_titanic()
    yield titanic
    drop(name="public.titanic")


class TestvDFCombineJoinSort:
    def test_vDF_append(self, iris_vd):
        assert iris_vd.shape() == (150, 5)

        result_vDF = iris_vd.append(iris_vd)
        assert result_vDF.shape() == (
            300,
            5,
        ), "testing vDataFrame.append(vDataFrame) failed"

        result_vDF = iris_vd.append("public.iris")
        assert result_vDF.shape() == (
            300,
            5,
        ), "testing vDataFrame.append(str) failed"

        result_vDF = iris_vd.append(
            iris_vd,
            expr1=["SepalLengthCm AS sl", "PetalLengthCm AS pl"],
            expr2=["SepalLengthCm AS sl", "PetalLengthCm AS pl"],
        )
        assert result_vDF.shape() == (
            300,
            2,
        ), "testing vDataFrame.append(vDataFrame, expr1, expr2) failed"

        # the duplicate rows
        result_vDF = iris_vd.append(iris_vd, union_all=False)
        assert result_vDF.shape() == (
            147,
            5,
        ), "testing vDataFrame.append(vDataFrame, union_all) failed"

    def testvDF_groupby(self, market_vd, titanic_vd):
        result1 = market_vd.groupby(
            columns=["Form", "Name"],
            expr=["AVG(Price) AS avg_price", "STDDEV(Price) AS std"],
        )
        assert result1.shape() == (
            159,
            4,
        ), "testing vDataFrame.groupby(columns, expr) failed"

        with pytest.raises(MissingColumn) as exception_info:
            result2 = market_vd.groupby(
                columns=["For", "Name"],
                expr=["AVG(Price) AS avg_price", "STDDEV(Price) AS std"],
            )
        assert exception_info.match(
            "The Virtual Column 'For' doesn't exist.\nDid you mean '\"Form\"' ?"
        )
        # with rollup
        result3 = market_vd.groupby(
            columns=["Form", "Name"],
            expr=["AVG(Price) AS avg_price", "STDDEV(Price) AS std"],
            rollup=True,
        )
        assert result3.shape() == (
            197,
            4,
        ), "testing vDataFrame.groupby(columns, expr) with rollup failed"
        # using the having parameter
        result4 = market_vd.groupby(
            columns=["Form", "Name"],
            expr=["AVG(Price) AS avg_price", "STDDEV(Price) AS std"],
            rollup=True,
            having="AVG(Price) > 2",
        )
        assert result4.shape() == (
            73,
            4,
        ), "testing vDataFrame.groupby(columns, expr) with rollup and having failed"
        # with specific rollups
        result5 = market_vd.groupby(
            columns=["Form", "Name"],
            expr=["AVG(Price) AS avg_price", "STDDEV(Price) AS std"],
            rollup=[False, True],
        )
        assert result5.shape() == (
            196,
            4,
        ), "testing vDataFrame.groupby(columns, expr) with rollup failed"
        # with specific rollups + tuples
        result6 = titanic_vd.groupby(
            columns=[("pclass", "sex"), "embarked"],
            expr=["AVG(survived) AS avg_survived"],
            rollup=[True, False],
        )
        assert result6.shape() == (
            33,
            4,
        ), "testing vDataFrame.groupby(columns, expr) with rollup failed"

    def test_vDF_join(self, market_vd):
        # CREATE TABLE not_fresh AS SELECT * FROM market WHERE Form != 'Fresh';
        not_fresh = market_vd.search("Form != 'Fresh'")
        # CREATE TABLE not_dried AS SELECT * FROM market WHERE Form != 'Dried';
        not_dried = market_vd.search("Form != 'Dried'")

        # CREATE TABLE left_join AS
        #        SELECT a.Name as Name1, b.Name as Name2
        #        FROM not_fresh AS a LEFT JOIN not_dried AS b ON a.Form = b.Form;
        left_join = not_fresh.join(
            not_dried,
            how="left",
            on={"Form": "Form"},
            expr1=["Name AS Name1"],
            expr2=["Name AS Name2"],
        )
        assert left_join.shape() == (5886, 2)
        # SELECT COUNT(*) FROM left_join WHERE Name1 IS NULL;
        assert left_join["Name1"].count() == 5886
        # SELECT COUNT(*) FROM left_join WHERE Name2 IS NULL;
        assert left_join["Name2"].count() == 5886 - 30

        # CREATE TABLE right_join AS
        #        SELECT a.Name as Name1, b.Name as Name2
        #        FROM not_fresh AS a RIGHT JOIN not_dried AS b ON a.Form = b.Form;
        right_join = not_fresh.join(
            not_dried,
            how="right",
            on={"Form": "Form"},
            expr1=["Name AS Name1"],
            expr2=["Name AS Name2"],
        )
        assert right_join.shape() == (5946, 2)
        # SELECT COUNT(*) FROM right_join WHERE Name1 IS NULL;
        assert right_join["Name1"].count() == 5946 - 90
        # SELECT COUNT(*) FROM right_join WHERE Name2 IS NULL;
        assert right_join["Name2"].count() == 5946

        # CREATE TABLE full_join AS
        #        SELECT a.Name as Name1, b.Name as Name2
        #        FROM not_fresh AS a FULL OUTER JOIN not_dried AS b ON a.Form = b.Form;
        full_join = not_fresh.join(
            not_dried,
            how="full",
            on={"Form": "Form"},
            expr1=["Name AS Name1"],
            expr2=["Name AS Name2"],
        )
        assert full_join.shape() == (5976, 2)
        # SELECT COUNT(*) FROM full_join WHERE Name1 IS NULL;
        assert full_join["Name1"].count() == 5976 - 90
        # SELECT COUNT(*) FROM full_join WHERE Name2 IS NULL;
        assert full_join["Name2"].count() == 5976 - 30

        # CREATE TABLE inner_join AS
        #        SELECT a.Name as Name1, b.Name as Name2
        #        FROM not_fresh AS a INNER JOIN not_dried AS b ON a.Form = b.Form;
        inner_join = not_fresh.join(
            not_dried,
            how="inner",
            on={"Form": "Form"},
            expr1=["Name AS Name1"],
            expr2=["Name AS Name2"],
        )
        assert inner_join.shape() == (5856, 2)
        # SELECT COUNT(*) FROM inner_join WHERE Name1 IS NULL;
        assert inner_join["Name1"].count() == 5856
        # SELECT COUNT(*) FROM inner_join WHERE Name2 IS NULL;
        assert inner_join["Name2"].count() == 5856

        # CREATE TABLE natural_join AS
        #        SELECT a.Name as Name1, b.Name as Name2
        #        FROM not_fresh AS a NATURAL JOIN not_dried AS b;
        natural_join = not_fresh.join(
            not_dried,
            how="natural",
            expr1=["Name AS Name1"],
            expr2=["Name AS Name2"],
        )
        assert natural_join.shape() == (194, 2)
        # SELECT COUNT(*) FROM natural_join WHERE Name1 IS NULL;
        assert natural_join["Name1"].count() == 194
        # SELECT COUNT(*) FROM natural_join WHERE Name2 IS NULL;
        assert natural_join["Name2"].count() == 194

        # CREATE TABLE cross_join AS
        #        SELECT a.Name as Name1, b.Name as Name2
        #        FROM not_fresh AS a CROSS JOIN not_dried AS b;
        cross_join = not_fresh.join(
            not_dried, how="cross", expr1=["Name AS Name1"], expr2=["Name AS Name2"]
        )
        assert cross_join.shape() == (63616, 2)
        # SELECT COUNT(*) FROM cross_join WHERE Name1 IS NULL;
        assert cross_join["Name1"].count() == 63616
        # SELECT COUNT(*) FROM cross_join WHERE Name2 IS NULL;
        assert cross_join["Name2"].count() == 63616

        # join directly with a Vertica table
        not_dried.to_db("not_dried", relation_type="local")
        table_join = not_fresh.join(
            "v_temp_schema.not_dried",
            how="natural",
            expr1=["Name AS Name1"],
            expr2=["Name AS Name2"],
        )
        assert table_join.shape() == (194, 2)
        drop("v_temp_schema.not_dried")
        # testing different operators
        d1 = TableSample(
            {
                "name1": ["Badr", "Umar", "Arash"],
                "email1": ["b.ouali@me.fr", "umar@me.com", "a.farad@me.com"],
                "age1": [30, 28, 40],
                "fav_car1": ["BM", "Audi", "Merc"],
            }
        ).to_vdf()
        d2 = TableSample(
            {
                "name2": ["Badr", "Umar", "Arash"],
                "email2": [
                    "badr.ouali@me.fr",
                    "umar.f@me.com",
                    "arash.farad@me.com",
                ],
                "age2": [29, 26, 35],
                "fav_car2": ["BMW", "Audi", "Mercedes"],
            }
        ).to_vdf()
        vdf = d1.join(d2, on=[("name1", "name2", "=")], how="left")
        assert vdf.shape() == (3, 8)
        assert vdf["age1"].sum() == 98.0
        vdf = d1.join(d2, on=[("age1", "age2", ">")], how="left")
        assert vdf.shape() == (6, 8)
        assert vdf["age1"].sum() == 208.0
        vdf = d1.join(d2, on=[("age1", "age2", ">=")], how="left")
        assert vdf.shape() == (6, 8)
        assert vdf["age1"].sum() == 208.0
        vdf = d1.join(d2, on=[("age1", "age2", "<")], how="left")
        assert vdf.shape() == (4, 8)
        assert vdf["age1"].sum() == 126.0
        vdf = d1.join(d2, on=[("age1", "age2", "<=")], how="left")
        assert vdf.shape() == (4, 8)
        assert vdf["age1"].sum() == 126.0
        vdf = d1.join(d2, on=[("fav_car1", "fav_car2", "llike")], how="inner")
        assert vdf.shape() == (1, 8)
        assert vdf["age1"].sum() == 28.0
        vdf = d1.join(d2, on=[("fav_car1", "fav_car2", "rlike")], how="inner")
        assert vdf.shape() == (3, 8)
        assert vdf["age2"].sum() == 90.0
        # available for vertica v12.0.2
        # vdf = d1.join(d2, on=[("email1", "email2", "jaro", ">", 0.7)], how="inner")
        # assert vdf.shape() == (4, 8)
        # assert vdf["age2"].sum() == 116.0
        # vdf = d1.join(d2, on=[("email1", "email2", "jarow", ">", 0.8)], how="inner")
        # assert vdf.shape() == (4, 8)
        # assert vdf["age2"].sum() == 116.0
        vdf = d1.join(d2, on=[("email1", "email2", "lev", "<", 3)], how="inner")
        assert vdf.shape() == (1, 8)
        assert vdf["age2"].sum() == 26.0
        vdf = d1.join(d2, on=[("email1", "email2", "lev", "<", 5)], how="inner")
        assert vdf.shape() == (3, 8)
        assert vdf["age2"].sum() == 90.0
        vdf = d1.join(d2, on=[("email1", "email2", "linterpolate")], how="left")
        assert vdf.shape() == (3, 8)
        assert vdf["age2"].sum() == 61.0
        vdf = d1.join(d2, on=[("email1", "email2", "rinterpolate")], how="left")
        assert vdf.shape() == (3, 8)
        assert vdf["age2"].sum() == 61.0
        # available for vertica v12.0.2
        # vdf = d1.join(
        #    d2,
        #    on=[("email1", "email2", "jaro", ">", 0.88), ("name1", "name2", "=")],
        #    how="inner",
        # )
        # assert vdf.shape() == (2, 8)
        # assert vdf["age2"].sum() == 55.0

    def test_vDF_narrow(self, amazon_vd):
        amazon_pivot = amazon_vd.pivot(
            index="date", columns="state", values="number", aggr="sum"
        )
        amazon_narrow = amazon_pivot.narrow("date", col_name="state", val_name="number")

        assert amazon_narrow.shape() == (6453, 3)

    def test_vDF_pivot(self, amazon_vd):
        amazon_pivot = amazon_vd.pivot(
            index="date", columns="state", values="number", aggr="sum", prefix="pv_"
        )

        assert amazon_pivot.shape() == (239, 28)
        assert amazon_pivot["pv_Acre"].count() == 239

    def testvDF_polynomial_comb(self, iris_vd):
        assert iris_vd.polynomial_comb(r=3).shape() == (150, 25)

    def testvDF_recommend(self, market_vd):
        assert market_vd.recommend("Name", "Form").shape() == (126, 4)

    def test_vDF_sort(self, iris_vd):
        result1 = iris_vd.copy().sort(columns={"PetalLengthCm": "asc"})
        assert result1["PetalLengthCm"][0] == 1.0

        result2 = iris_vd.copy().sort(columns=["PetalLengthCm", "SepalWidthCm"])
        assert result2["PetalLengthCm"][0] == 1.0

        result3 = iris_vd.copy().sort(columns={"PetalLengthCm": "desc"})
        assert result3["PetalLengthCm"][0] == 6.9

        result4 = iris_vd.copy().sort(
            columns={"PetalLengthCm": "desc", "SepalWidthCm": "asc"}
        )
        assert result4["PetalLengthCm"][0] == 6.9
        assert result4["SepalWidthCm"][0] == 2.6

    @pytest.mark.skip(
        reason="this test will be implemented later. It should test a combination of operation like joins, imputing missing values..."
    )
    def test_multiple_operation(self):
        pass
