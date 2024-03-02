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
import pytest


class TestVDFSys:
    """
    test class for sys functions test for vDataframe class
    """

    def test_current_relation(self, titanic_vd_fun):
        """
        test function - current_relation
        """
        res = titanic_vd_fun.current_relation().split(".")[1].replace('"', "")
        assert res == "titanic"

    def test_del_catalog(self, titanic_vd_fun):
        """
        test function - del_catalog
        """
        titanic_vd_fun.describe(method="numerical")
        catalog_val = titanic_vd_fun["age"]._catalog
        assert {"max", "avg"}.issubset(catalog_val.keys())

        titanic_vd_fun.del_catalog()
        catalog_val = titanic_vd_fun["age"]._catalog
        assert not {"max", "avg"}.issubset(catalog_val.keys())

    @pytest.mark.parametrize("test_type", ["non_empty", "empty"])
    def test_empty(self, amazon_vd, test_type):
        """
        test function - empty dataframe
        """
        if test_type == "non_empty":
            assert not amazon_vd.empty()
        else:
            assert amazon_vd.drop(["number", "date", "state"]).empty()

    @pytest.mark.parametrize(
        "col, expected",
        [
            ("expected_size", 85947.0),
            ("max_size", 504492.0),
        ],
    )
    @pytest.mark.parametrize("unit", ["b", "kb", "mb", "gb", "tb"])
    def test_expected_store_usage(self, titanic_vd_fun, col, unit, expected):
        """
        test function - expected_store_usage
        """
        res = titanic_vd_fun.expected_store_usage(unit=unit)[f"{col} ({unit})"][-1]

        if unit == "b":
            assert res == expected
        elif unit == "kb":
            assert res == expected / 1024
        elif unit == "mb":
            assert res == expected / (1024 * 1024)
        elif unit == "gb":
            assert res == expected / (1024 * 1024 * 1024)
        elif unit == "tb":
            assert res == expected / (1024 * 1024 * 1024 * 1024)

    @pytest.mark.parametrize(
        "digraph, expected",
        [
            (True, "digraph G {"),
            (
                False,
                "------------------------------ \nQUERY PLAN DESCRIPTION: \n\nEXPLAIN SELECT",
            ),
        ],
    )
    def test_explain(self, titanic_vd_fun, digraph, expected):
        """
        test function - explain
        """
        res = titanic_vd_fun.explain(digraph=digraph)
        assert res.startswith(expected)

    @pytest.mark.parametrize("actions", [0, 1, 2])
    def test_info(self, titanic_vd_fun, actions):
        """
        test function - info
        """
        if actions == 0:
            assert titanic_vd_fun.info() == "The vDataFrame was never modified."
        elif actions == 1:
            res = titanic_vd_fun.filter("age > 0")
            assert res.info().startswith(
                "The vDataFrame was modified with only one action"
            )
        else:
            res = titanic_vd_fun.filter("age > 0")
            res["fare"].drop_outliers()
            assert res.info().startswith("The vDataFrame was modified many times")

    @pytest.mark.skip("Test is not stable")
    @pytest.mark.parametrize(
        "column, expected",
        [(None, 1039)],
    )
    def test_memory_usage(self, amazon_vd, column, expected):
        """
        test function - memory_usage
        """
        # values are not stable
        assert amazon_vd.memory_usage()["value"][0] == pytest.approx(expected, 1e-01)

    @pytest.mark.parametrize(
        "col1, col2, expected",
        [
            (
                "pop",
                0,
                [
                    '"pop"',
                    '"year"',
                    '"country"',
                    '"continent"',
                    '"lifeExp"',
                    '"gdpPercap"',
                ],
            ),
            (
                "year",
                "lifeExp",
                [
                    '"country"',
                    '"lifeExp"',
                    '"pop"',
                    '"continent"',
                    '"year"',
                    '"gdpPercap"',
                ],
            ),
        ],
    )
    def test_swap(self, gapminder_vd_fun, col1, col2, expected):
        """
        test function - swap two columns
        """
        gapminder_vd_fun.swap(col1, col2)
        swap_columns = gapminder_vd_fun.get_columns()

        assert expected == swap_columns


class TestVDCSys:
    """
    test class for sys functions test for vColumn class
    """

    def test_add_copy(self, titanic_vd_fun):
        """
        test function - add_copy
        """
        titanic_vd_fun["age"].add_copy(name="copy_age")

        assert titanic_vd_fun["copy_age"].mean() == titanic_vd_fun["age"].mean()

    @pytest.mark.skip("Test is not stable")
    @pytest.mark.parametrize(
        "column, expected",
        [("number", 1724)],
    )
    def test_memory_usage(self, amazon_vd, column, expected):
        """
        test function - memory_usage
        """
        # values are not stable
        assert amazon_vd[column].memory_usage() == pytest.approx(expected, 1e-01)

    def test_store_usage(self, titanic_vd):
        """
        test function - store_usage
        """
        res = titanic_vd["age"].store_usage()
        assert res == pytest.approx(5908, 1e-2)

    def test_rename(self, titanic_vd_fun):
        """
        test function - rename
        """
        titanic_vd_fun["sex"].rename("gender")
        columns = titanic_vd_fun.get_columns()
        assert '"gender"' in columns and '"sex"' not in columns
