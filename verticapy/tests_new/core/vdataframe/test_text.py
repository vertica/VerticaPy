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
from itertools import chain
import pytest


class TestText:
    """
    test class for Text functions test
    """

    @pytest.mark.parametrize(
        "column, pattern, method, position, occurrence, replacement, return_position, name",
        [
            ("name", "son", "count", 1, None, None, None, "name_regex"),
            ("name", "Mrs", "ilike", None, 1, None, None, "name_ilike"),
            ("name", "Mrs.", "instr", 1, 1, None, 0, "name_instr"),
            ("name", "mrs", "like", None, 1, None, None, "name_like"),
            ("name", "Mrs", "not_ilike", None, 1, None, None, "name_not_ilike"),
            ("name", "Mrs.", "not_like", 1, 1, None, 0, "name_not_like"),
            ("name", "Mrs.", "replace", 1, 1, "Mr.", 0, "name_replace"),
            ("name", "([^,]+)", "substr", 1, 1, None, 0, "name_substr"),
        ],
    )
    def test_regexp(
        self,
        titanic_vd_fun,
        column,
        pattern,
        method,
        position,
        occurrence,
        replacement,
        return_position,
        name,
    ):
        """
        test function - regexp for vDataframe
        """
        titanic_pdf = titanic_vd_fun.to_pandas()

        _vpy_res = titanic_vd_fun.regexp(
            column=column,
            pattern=pattern,
            method=method,
            position=position,
            occurrence=occurrence,
            replacement=replacement,
            return_position=return_position,
            name=name,
        )[name][:5].to_list()
        vpy_res = list(chain(*_vpy_res))

        if method == "count":
            py_res = titanic_pdf[column].str.count(pat=pattern).head(5).to_list()
        elif method == "ilike":
            py_res = (
                titanic_pdf[column]
                .str.contains(pat=pattern, case=False)
                .head(5)
                .to_list()
            )
        elif method == "instr":
            py_res = titanic_pdf[column].str.find(pattern).add(1).head(5).to_list()
        elif method == "like":
            py_res = (
                titanic_pdf[column]
                .str.contains(pat=pattern, case=True)
                .head(5)
                .to_list()
            )
        elif method == "not_ilike":
            titanic_pdf[name] = titanic_pdf[column].apply(lambda x: pattern not in x)
            py_res = titanic_pdf[name].head(5).to_list()
        elif method == "not_like":
            titanic_pdf[name] = titanic_pdf[column].apply(lambda x: pattern not in x)
            py_res = titanic_pdf[name].head(5).to_list()
        elif method == "replace":
            py_res = (
                titanic_pdf[column]
                .replace(to_replace=pattern, value=replacement, regex=True)
                .head(5)
                .to_list()
            )
        elif method == "substr":
            py_res = (
                titanic_pdf[column]
                .str.extract(pat=pattern, expand=False)
                .head(5)
                .to_list()
            )

        print(
            f"method name: str_slice \nVerticaPy Result: {vpy_res} \nPython Result :{py_res}\n"
        )

        assert vpy_res == py_res

    @pytest.mark.parametrize("column, pat", [("name", r"([A-Za-z]+\.)")])
    def test_str_contains(self, titanic_vd_fun, column, pat):
        """
        test function - str_contains for vColumns
        """
        titanic_pdf = titanic_vd_fun.to_pandas()

        _vpy_res = titanic_vd_fun[column].str_contains(pat=pat)[column][:5].to_list()
        vpy_res = list(chain(*_vpy_res))

        py_res = titanic_pdf[column].str.contains(pat=pat).head(5).to_list()

        print(
            f"method name: str_slice \nVerticaPy Result: {vpy_res} \nPython Result :{py_res}\n"
        )

        assert vpy_res == py_res

    @pytest.mark.parametrize("column, pat", [("name", r"([A-Za-z]+\.)")])
    def test_str_count(self, titanic_vd_fun, column, pat):
        """
        test function - str_count for vColumns
        """
        titanic_pdf = titanic_vd_fun.to_pandas()

        _vpy_res = titanic_vd_fun[column].str_count(pat=pat)[column][:5].to_list()
        vpy_res = list(chain(*_vpy_res))

        py_res = titanic_pdf[column].str.count(pat=pat).head(5).to_list()

        print(
            f"method name: str_slice \nVerticaPy Result: {vpy_res} \nPython Result :{py_res}\n"
        )

        assert vpy_res == py_res

    @pytest.mark.parametrize("column, pat", [("name", r"([A-Za-z]+\.)")])
    def test_str_extract(self, titanic_vd_fun, column, pat):
        """
        test function - str_extract for vColumns
        """
        titanic_pdf = titanic_vd_fun.to_pandas()

        _vpy_res = titanic_vd_fun[column].str_extract(pat=pat)[column][:5].to_list()
        vpy_res = list(chain(*_vpy_res))

        py_res = (
            titanic_pdf[column].str.extract(pat=pat, expand=False).head(5).to_list()
        )

        print(
            f"method name: str_slice \nVerticaPy Result: {vpy_res} \nPython Result :{py_res}\n"
        )

        assert vpy_res == py_res

    @pytest.mark.parametrize(
        "column, to_replace, value", [("name", r" ([A-Za-z])+\.", "VERTICAPY")]
    )
    def test_str_replace(self, titanic_vd_fun, column, to_replace, value):
        """
        test function - str_replace for vColumns
        """
        titanic_pdf = titanic_vd_fun.to_pandas()

        _vpy_res = (
            titanic_vd_fun[column]
            .str_replace(to_replace=to_replace, value=value)[column][:5]
            .to_list()
        )
        vpy_res = list(chain(*_vpy_res))

        py_res = (
            titanic_pdf[column]
            .replace(to_replace=to_replace, value=value, regex=True)
            .head(5)
            .to_list()
        )

        print(
            f"method name: str_slice \nVerticaPy Result: {vpy_res} \nPython Result :{py_res}\n"
        )

        assert vpy_res == py_res

    @pytest.mark.parametrize("column, start, end", [("name", 0, 3), ("name", 0, 4)])
    # step parameter name does not do its intended work. May needs to change
    def test_str_slice(self, titanic_vd_fun, column, start, end):
        """
        test function - str_slice for vColumns
        """
        titanic_pdf = titanic_vd_fun.to_pandas()

        _vpy_res = (
            titanic_vd_fun[column]
            .str_slice(start=start, step=end + 1)[column][:5]
            .to_list()
        )
        vpy_res = list(chain(*_vpy_res))

        py_res = titanic_pdf[column].str.slice(start=start, stop=end).head(5).to_list()

        print(
            f"method name: str_slice \nVerticaPy Result: {vpy_res} \nPython Result :{py_res}\n"
        )

        assert vpy_res == py_res
