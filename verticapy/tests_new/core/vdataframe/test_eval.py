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


class TestEval:
    """
    test class for eval functions test
    """

    @pytest.mark.parametrize(
        "column, expr",
        [
            ("family_size", "parch + sibsp + 1"),
            ("missing_cabins", "CASE WHEN cabin IS NULL THEN 'missing' ELSE cabin END"),
        ],
    )
    def test_eval(self, titanic_vd, column, expr):
        """
        test function - evaluate expression
        """
        titanic_pdf = titanic_vd.to_pandas()

        vpy_res = list(
            chain(*titanic_vd.eval(name=column, expr=expr)[[column]].to_list())
        )

        if column == "missing_cabins":
            titanic_pdf["cabin"] = titanic_pdf["cabin"].apply(
                lambda row: row if row is not None else "missing"
            )
            py_res = titanic_pdf["cabin"].values.tolist()
        else:
            py_res = titanic_pdf.eval(expr=f"{column}={expr}")[column].values.tolist()

        assert vpy_res == py_res
