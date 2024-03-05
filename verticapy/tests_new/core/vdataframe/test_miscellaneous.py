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
from math import ceil, floor

import numpy as np
import pandas as pd

import verticapy as vp
import verticapy.sql.functions as vpf


class TestMiscellaneousVDF:
    """
    test class to test Miscellaneous functions for vDataframe
    """

    def test_repr(self, titanic_vd_fun):
        """
        test function - repr
        """
        repr_vdf = titanic_vd_fun.__repr__()
        assert "pclass" in repr_vdf
        assert "survived" in repr_vdf
        assert 10000 < len(repr_vdf) < 1000000
        repr_html_vdf = titanic_vd_fun._repr_html_()
        assert 10000 < len(repr_html_vdf) < 10000000
        assert "<table>" in repr_html_vdf
        assert "data:image/png;base64," in repr_html_vdf

        # vdc
        repr_vdc = titanic_vd_fun["age"].__repr__()
        assert "age" in repr_vdc
        assert "60" in repr_vdc
        assert 500 < len(repr_vdc) < 5000
        repr_html_vdc = titanic_vd_fun["age"]._repr_html_()
        assert 10000 < len(repr_html_vdc) < 10000000
        assert "<table>" in repr_html_vdc
        assert "data:image/png;base64," in repr_html_vdc

    def test_magic(self, titanic_vd):
        """
        test function - magic
        """
        assert (
            str(titanic_vd["name"]._in(["Madison", "Ashley", None]))
            == "(\"name\") IN ('Madison', 'Ashley', NULL)"
        )
        assert str(titanic_vd["age"]._between(1, 4)) == '("age") BETWEEN (1) AND (4)'
        assert str(titanic_vd["age"]._as("age2")) == '("age") AS age2'
        assert str(titanic_vd["age"]._distinct()) == 'DISTINCT ("age")'
        assert (
            str(
                vpf.sum(titanic_vd["age"])._over(
                    by=[titanic_vd["pclass"], titanic_vd["sex"]],
                    order_by=[titanic_vd["fare"]],
                )
            )
            == 'SUM("age") OVER (PARTITION BY "pclass", "sex" ORDER BY "fare")'
        )
        assert str(abs(titanic_vd["age"])) == 'ABS("age")'
        assert str(ceil(titanic_vd["age"])) == 'CEIL("age")'
        assert str(floor(titanic_vd["age"])) == 'FLOOR("age")'
        assert str(round(titanic_vd["age"], 2)) == 'ROUND("age", 2)'
        assert str(-titanic_vd["age"]) == '-("age")'
        assert str(+titanic_vd["age"]) == '+("age")'
        assert str(titanic_vd["age"] % 2) == 'MOD("age", 2)'
        assert str(2 % titanic_vd["age"]) == 'MOD(2, "age")'
        assert str(titanic_vd["age"] ** 2) == 'POWER("age", 2)'
        assert str(2 ** titanic_vd["age"]) == 'POWER(2, "age")'
        assert str(titanic_vd["age"] + 3) == '("age") + (3)'
        assert str(3 + titanic_vd["age"]) == '(3) + ("age")'
        assert str(titanic_vd["age"] - 3) == '("age") - (3)'
        assert str(3 - titanic_vd["age"]) == '(3) - ("age")'
        assert str(titanic_vd["age"] * 3) == '("age") * (3)'
        assert str(3 * titanic_vd["age"]) == '(3) * ("age")'
        assert str(titanic_vd["age"] // 3) == '("age") // (3)'
        assert str(3 // titanic_vd["age"]) == '(3) // ("age")'
        assert str(titanic_vd["age"] > 3) == '("age") > (3)'
        assert str(3 > titanic_vd["age"]) == '("age") < (3)'
        assert str(titanic_vd["age"] >= 3) == '("age") >= (3)'
        assert str(3 >= titanic_vd["age"]) == '("age") <= (3)'
        assert str(titanic_vd["age"] < 3) == '("age") < (3)'
        assert str(3 < titanic_vd["age"]) == '("age") > (3)'
        assert str(titanic_vd["age"] <= 3) == '("age") <= (3)'
        assert str(3 <= titanic_vd["age"]) == '("age") >= (3)'
        assert (
            str((3 >= titanic_vd["age"]) & (titanic_vd["age"] <= 50))
            == '(("age") <= (3)) AND (("age") <= (50))'
        )
        assert (
            str((3 >= titanic_vd["age"]) | (titanic_vd["age"] <= 50))
            == '(("age") <= (3)) OR (("age") <= (50))'
        )
        assert str("Mr " + titanic_vd["name"]) == "('Mr ') || (\"name\")"
        assert str(titanic_vd["name"] + " .") == "(\"name\") || (' .')"
        assert str(3 * titanic_vd["name"]) == 'REPEAT("name", 3)'
        assert str(titanic_vd["name"] * 3) == 'REPEAT("name", 3)'
        assert str(titanic_vd["age"] == 3) == '("age") = (3)'
        assert str(3 == titanic_vd["age"]) == '("age") = (3)'
        assert str(titanic_vd["age"] != 3) == '("age") != (3)'
        assert str(None != titanic_vd["age"]) == '("age") IS NOT NULL'
        assert titanic_vd["fare"][0] >= 0
        assert titanic_vd[["fare"]][0][0] >= 0
        assert titanic_vd[titanic_vd["fare"] > 500].shape()[0] == 4
        assert titanic_vd[titanic_vd["fare"] < 500].shape()[0] == 1229
        assert titanic_vd[titanic_vd["fare"] * 4 + 2 < 500].shape()[0] == 1167
        assert titanic_vd[titanic_vd["fare"] / 4 - 2 < 500].shape()[0] == 1233

    def test_sql(self, titanic_vd_fun, schema_loader):
        """
        test function - sql
        """
        sql = f"""-- Selecting some columns \n
                 SELECT 
                    age, 
                    fare 
                 FROM {schema_loader}.titanic 
                 WHERE age IS NOT NULL;"""
        vdf = vp.vDataFrame(sql)
        assert vdf.shape() == (997, 2)
        vdf = vp.vDataFrame(sql, usecols=["age"])
        assert vdf.shape() == (997, 1)


class TestVDFCreate:
    """
    test class to test vDataframe create options
    """

    def test_using_input_relation(self, titanic_vd_fun, schema_loader):
        """
        test create vDataFrame using input relation
        """
        vdf = vp.vDataFrame(input_relation=f"{schema_loader}.titanic")

        assert vdf["pclass"].count() == 1234

    def test_using_input_relation_schema(self, titanic_vd_fun, schema_loader):
        """
        test create vDataFrame using input relation and schema
        """
        vdf = vp.vDataFrame(input_relation="titanic", schema=schema_loader)

        assert vdf["pclass"].count() == 1234

    def test_using_input_relation_vdatacolumns(self, titanic_vd_fun, schema_loader):
        """
        test create vDataFrame using relation `vDataColumns`
        """
        vdf = vp.vDataFrame(
            input_relation=f"{schema_loader}.titanic",
            usecols=["age", "survived"],
        )

        assert vdf["survived"].count() == 1234

    def test_using_pandas_dataframe(self, titanic_vd_fun, schema_loader):
        """
        test create vDataFrame using pandas dataframe
        """
        pdf = pd.DataFrame(
            [[1, "first1", "last1"], [2, "first2", "last2"]],
            columns=["id", "fname", "lname"],
        )
        vdf = vp.vDataFrame(pdf)

        assert vdf["id"].count() == 2

    def test_using_list(self):
        """
        test create vDataFrame using list
        """
        vdf = vp.vDataFrame(
            input_relation=[[1, "first1", "last1"], [2, "first2", "last2"]],
            usecols=["id", "fname", "lname"],
        )

        assert vdf.shape() == (2, 3)
        assert vdf["id"].avg() == 1.5

    def test_using_np_array(self):
        """
        test create vDataFrame using numpy array
        """
        vdf = vp.vDataFrame(
            input_relation=np.array([[1, "first1", "last1"], [2, "first2", "last2"]]),
        )

        assert vdf.shape() == (2, 3)
        assert vdf["col0"].avg() == 1.5

    def test_using_tablesample(self):
        """
        test create vDataFrame using `TableSample`
        """
        tb = vp.TableSample(
            {"id": [1, 2], "fname": ["first1", "first2"], "lname": ["last1", "last2"]}
        )
        vdf = vp.vDataFrame(
            input_relation=tb,
        )

        assert vdf.shape() == (2, 3)
        assert vdf["id"].avg() == 1.5

        vdf = vp.vDataFrame(input_relation=tb, usecols=["id", "lname"])

        assert vdf.shape() == (2, 2)
        assert vdf.get_columns() == ['"id"', '"lname"']

    def test_using_dict(self):
        """
        test create vDataFrame using dictionary
        """
        tb = {"id": [1, 2], "fname": ["first1", "first2"], "lname": ["last1", "last2"]}
        vdf = vp.vDataFrame(
            input_relation=tb,
        )

        assert vdf.shape() == (2, 3)
        assert vdf["id"].avg() == 1.5

        vdf = vp.vDataFrame(input_relation=tb, usecols=["id", "lname"])

        assert vdf.shape() == (2, 2)
        assert vdf.get_columns() == ['"id"', '"lname"']

    def test_from_sql(self, titanic_vd_fun, schema_loader):
        """
        test create vDataFrame using sql
        """
        vdf = vp.vDataFrame(f"SELECT * FROM {schema_loader}.titanic")

        assert vdf["survived"].count() == 1234

        vdf = vp.vDataFrame(
            f"SELECT * FROM {schema_loader}.titanic", usecols=["survived"]
        )

        assert vdf["survived"].count() == 1234
        assert vdf.get_columns() == ['"survived"']
