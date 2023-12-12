"""
Copyright  (c)  2018-2023 Open Text  or  one  of its
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
import random
from itertools import chain
import pytest
import pandas as pd
from verticapy.utilities import pandas_to_vertica, drop


class TestJoinUnionSort:
    """
    test class for join_union_sort functions test
    """

    @pytest.mark.parametrize(
        "input_type,",
        ["vDataFrame", "relation", "expr", "union_all"],
    )
    def test_append(self, iris_vd_fun, input_type, schema_loader):
        """
        test function - append
        """
        iris_pdf = iris_vd_fun.to_pandas()

        if input_type == "relation":
            vpy_res = iris_vd_fun.append(f"{schema_loader}.iris", union_all=True)
        elif input_type == "expr":
            vpy_res = iris_vd_fun.append(
                iris_vd_fun,
                expr1=["SepalLengthCm AS sl", "PetalLengthCm AS pl"],
                expr2=["SepalLengthCm AS sl", "PetalLengthCm AS pl"],
                union_all=True,
            )
        elif input_type == "union_all":
            vpy_res = iris_vd_fun.append(f"{schema_loader}.iris", union_all=False)
        else:
            vpy_res = iris_vd_fun.append(iris_vd_fun)

        py_res = pd.concat([iris_pdf, iris_pdf], ignore_index=True)

        assert len(vpy_res) == len(
            vpy_res.drop_duplicates() if input_type == "union_all" else py_res
        )

    @pytest.mark.parametrize(
        "on, on_interpolate, how, exp1, exp2, expected",
        [
            ({"Form": "Form"}, {}, "left", ["Name AS Name1"], ["Name AS Name2"], None),
            ({"Form": "Form"}, {}, "right", ["Name AS Name1"], ["Name AS Name2"], None),
            ({"Form": "Form"}, {}, "full", ["Name AS Name1"], ["Name AS Name2"], None),
            ({"Form": "Form"}, {}, "inner", ["Name AS Name1"], ["Name AS Name2"], None),
            (
                {"Form": "Form", "Name": "Name", "Price": "Price"},
                {},
                "natural",
                ["Name AS Name1"],
                ["Name AS Name2"],
                None,
            ),
            ({}, {}, "cross", ["Name AS Name1"], ["Name AS Name2"], None),
            (
                [("Form", "Form", "=")],
                {},
                "left",
                ["Name AS Name1"],
                ["Name AS Name2"],
                None,
            ),
            (
                [("Price", "Price", ">")],
                {},
                "left",
                ["Name AS Name1"],
                ["Name AS Name2"],
                32207,
            ),
            (
                [("Price", "Price", ">=")],
                {},
                "left",
                ["Name AS Name1"],
                ["Name AS Name2"],
                32405,
            ),
            (
                [("Price", "Price", "<")],
                {},
                "left",
                ["Name AS Name1"],
                ["Name AS Name2"],
                31213,
            ),
            (
                [("Price", "Price", "<=")],
                {},
                "left",
                ["Name AS Name1"],
                ["Name AS Name2"],
                31411,
            ),
            (
                [("Name", "Name", "llike")],
                {},
                "inner",
                ["Name AS Name1"],
                ["Name AS Name2"],
                None,
            ),
            (
                [("Name", "Name", "rlike")],
                {},
                "inner",
                ["Name AS Name1"],
                ["Name AS Name2"],
                None,
            ),
            (
                [("Name", "Name", "linterpolate")],
                {},
                "inner",
                ["Name AS Name1"],
                ["Name AS Name2"],
                None,
            ),
            (
                [("Name", "Name", "rinterpolate")],
                {},
                "inner",
                ["Name AS Name1"],
                ["Name AS Name2"],
                None,
            ),
            (
                [("Form", "Form", "jaro", ">", 0.7)],
                {},
                "inner",
                ["Name AS Name1"],
                ["Name AS Name2"],
                12924,
            ),
            (
                [("Form", "Form", "jarow", ">", 0.7)],
                {},
                "inner",
                ["Name AS Name1"],
                ["Name AS Name2"],
                13742,
            ),
            (
                [("Form", "Form", "lev", ">", 3)],
                {},
                "inner",
                ["Name AS Name1"],
                ["Name AS Name2"],
                57408,
            ),
        ],
    )
    @pytest.mark.parametrize("relation_type", ["DataFrame", "table"])
    def test_join(
        self,
        market_vd,
        schema_loader,
        relation_type,
        on,
        on_interpolate,
        how,
        exp1,
        exp2,
        expected,
    ):
        """
        test function - sort
        """
        # randomly changing few records
        market_pdf = market_vd.to_pandas()
        change_pdf = market_pdf.sample(50, random_state=100).index
        market_pdf.loc[change_pdf, "Price"] += 1
        market_vd_copy = pandas_to_vertica(
            market_pdf, name=f"market_pandas_{random.random()}", schema=schema_loader
        )

        # CREATE TABLE not_fresh AS SELECT * FROM market WHERE Form != 'Fresh';
        not_fresh = market_vd_copy.search("Form != 'Fresh'")
        not_fresh_pdf = not_fresh.to_pandas()

        # CREATE TABLE not_dried AS SELECT * FROM market WHERE Form != 'Dried';
        not_dried = market_vd_copy.search("Form != 'Dried'")
        not_dried_pdf = not_dried.to_pandas()

        drop(f"{schema_loader}.not_dried")
        not_dried.to_db(f"{schema_loader}.not_dried", relation_type="Table")

        vpy_res = not_fresh.join(
            f"{schema_loader}.not_dried" if relation_type == "table" else not_dried,
            on={} if how == "natural" else on,
            on_interpolate=on_interpolate,
            how=how,
            expr1=exp1,
            expr2=exp2,
        )
        # python
        if isinstance(on, list):
            on = on[0][:-1]
            print(on[:-1])
        else:
            if how == "cross":
                on = None
            elif how == "natural":
                on = list(on.keys())
            else:
                on = [list(on.keys())[0], list(on.values())[0]]

        if not expected:
            py_res = not_fresh_pdf.merge(
                not_dried_pdf,
                on=on,
                how="outer"
                if how == "full"
                else ("inner" if how == "natural" else how),
            )
        else:
            py_res = []

        print(f"Join type: {how}, Vertica: {len(vpy_res)}, Python: {len(py_res)}")

        assert len(vpy_res) == expected if expected else len(vpy_res) == len(py_res)

        drop(f"{schema_loader}.not_dried")

    @pytest.mark.parametrize(
        "order_by,",
        [
            {"PetalLengthCm": "asc"},
            ["PetalLengthCm", "SepalWidthCm"],
            {"Species": "desc"},
            {"PetalLengthCm": "desc", "SepalWidthCm": "asc"},
        ],
    )
    def test_sort(self, iris_vd_fun, order_by):
        """
        test function - sort
        """
        if isinstance(order_by, list):
            order_by = {i: "asc" for i in order_by}

        colm_name = list(order_by.keys())[0]
        iris_pdf = iris_vd_fun.to_pandas()
        iris_pdf[colm_name] = iris_pdf[colm_name].astype(
            "str" if colm_name == "Species" else "float"
        )

        vpy_res = iris_vd_fun.sort(columns=order_by)[[colm_name]].to_list()

        py_res = iris_pdf.sort_values(
            by=list(order_by.keys()),
            ascending=[i == "asc" for i in list(order_by.values())],
        )[colm_name].tolist()

        assert list(chain(*vpy_res)) == py_res
