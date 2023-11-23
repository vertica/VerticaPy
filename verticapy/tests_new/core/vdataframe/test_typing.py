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
import os
from itertools import chain
import pytest
import verticapy as vp
from verticapy.utilities import read_json


class TestTyping:
    """
    test class for Typing functions test
    """

    def test_astype(self, laliga_vd):
        """
        test function - astype
        """
        laliga_vd["match_id"].astype("varchar")
        assert laliga_vd["match_id"].ctype() == "varchar"

    def test_bool_to_int(self, titanic_vd_fun):
        """
        test function - bool_to_int
        """
        titanic_vd_fun["survived"].astype("bool")
        assert titanic_vd_fun["survived"].dtype() == "bool"

        titanic_vd_fun.bool_to_int()
        assert titanic_vd_fun["survived"].dtype() == "int"

    def test_catcol(self, titanic_vd_fun):
        """
        test function - catcol
        """
        assert titanic_vd_fun.catcol(max_cardinality=6) == [
            '"pclass"',
            '"survived"',
            '"name"',
            '"sex"',
            '"ticket"',
            '"cabin"',
            '"embarked"',
            '"boat"',
            '"home.dest"',
        ]

    def test_datecol(self, amazon_vd):
        """
        test function - datecol
        """
        assert amazon_vd.datecol()[0] == '"date"'

    def test_dtypes(self, amazon_vd):
        """
        test function - dtypes
        """
        assert list(chain(*amazon_vd.dtypes().to_list())) == [
            "date",
            "varchar(32)",
            "int",
        ]

    def test_dtype(self, amazon_vd):
        """
        test function - dtypes
        """
        assert amazon_vd["state"].dtype() == "varchar(32)"

    @pytest.mark.parametrize(
        "exclude_columns, expected",
        [
            (
                [],
                [
                    '"age"',
                    '"body"',
                    '"fare"',
                    '"parch"',
                    '"pclass"',
                    '"sibsp"',
                    '"survived"',
                ],
            ),
            (
                ["survived", "body"],
                ['"fare"', '"pclass"', '"age"', '"parch"', '"sibsp"'],
            ),
        ],
    )
    def test_numcol(self, titanic_vd_fun, exclude_columns, expected):
        """
        test function - numcol
        """
        assert sorted(titanic_vd_fun.numcol(exclude_columns=exclude_columns)) == sorted(
            expected
        )

    def test_astype_vcol(self, laliga_vd):
        """
        test function - astype
        """
        laliga_vd.astype({"match_id": "varchar"})
        assert laliga_vd["match_id"].ctype() == "varchar"

    def test_category(self, laliga_vd):
        """
        test function - category
        """
        assert laliga_vd["away_score"].category() == "int"

    def test_ctype(self, laliga_vd):
        """
        test function - ctype
        """
        assert laliga_vd["away_score"].ctype() == "int"

    def test_isarray(self, laliga_vd):
        """
        test function - isarray
        """
        assert laliga_vd["away_team"]["managers"].isarray()

    def test_isbool(self, titanic_vd_fun):
        """
        test function - isbool
        """
        titanic_vd_fun["survived"].astype("bool")
        assert titanic_vd_fun["survived"].isbool() is True

    def test_isdate(self, laliga_vd):
        """
        test function - isdate
        """
        assert laliga_vd["match_date"].isdate()

    def test_isnum(self, laliga_vd):
        """
        test function - isnum
        """
        assert laliga_vd["away_score"].isnum()

    def test_isvmap(self):
        """
        test function - isvmap
        """
        laliga = read_json(
            os.path.dirname(vp.__file__) + "/datasets/data/laliga/*.json"
        )
        assert laliga["away_team.managers"].isvmap() is True
