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


class TestUtils:
    """
    test class for Utils functions test
    """

    @pytest.mark.parametrize(
        "args, columns, expected_nb_of_cols, raise_error, expected",
        [
            (None, ["home.dest", "age"], None, True, ['"home.dest"', '"age"']),
            (None, ["home.dest", "age"], 1, True, None),
            (None, ["home.dest", "age"], [0, 1], True, None),
            # (None, ["home.dest", "age"], [1, 1], False, None),
        ],
    )
    def test_format_colnames(
        self, titanic_vd_fun, args, columns, expected_nb_of_cols, raise_error, expected
    ):
        """
        test function - format_colnames
        """
        if (
            raise_error and len(columns) > max(expected_nb_of_cols)
            if isinstance(expected_nb_of_cols, list)
            else expected_nb_of_cols
        ):
            with pytest.raises(ValueError) as exception_info:
                titanic_vd_fun.format_colnames(
                    columns=columns,
                    expected_nb_of_cols=expected_nb_of_cols,
                    raise_error=raise_error,
                )
            assert exception_info.match(
                rf"The number of Virtual Columns expected is \[0|1\], found {len(columns)}."
            )
        else:
            res = titanic_vd_fun.format_colnames(
                columns=columns,
                expected_nb_of_cols=expected_nb_of_cols,
                raise_error=raise_error,
            )
            assert expected == res

    def test_get_columns(self, amazon_vd):
        """
        test function - get_columns
        """
        assert amazon_vd.get_columns() == ['"date"', '"state"', '"number"']

    @pytest.mark.parametrize(
        "x, str_check, expected",
        [("pclass", True, 0), ("pclass", None, 0), ("class", False, None)],
    )
    def test_get_match_index(self, titanic_vd_fun, x, str_check, expected):
        """
        test function - get_match_index
        """
        if str_check:
            assert (
                titanic_vd_fun.get_match_index(
                    x=x, col_list=titanic_vd_fun.get_columns(), str_check=str_check
                )
                == expected
            )
        else:
            assert (
                titanic_vd_fun.get_match_index(
                    x=x, col_list=titanic_vd_fun.get_columns()
                )
                == expected
            )

    @pytest.mark.parametrize("column, expected", [("pclass", True), ("class", False)])
    def test_is_colname_in(self, titanic_vd_fun, column, expected):
        """
        test function - is_colname_in
        """
        assert titanic_vd_fun.is_colname_in(column=column) is expected
