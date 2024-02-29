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
import os

import pytest

import verticapy as vp
from verticapy.errors import ConversionError
from verticapy.utilities import read_json, TableSample


class TestVDFTyping:
    """
    test class for Typing functions test for vDataFrame class
    """

    def test_astype(self, titanic_vd_fun):
        """
        test function - astype for vDataframe
        """
        # Testing vDataFrame.astype
        titanic_vd_fun.astype({"fare": "int", "cabin": "varchar(1)"})

        assert titanic_vd_fun["fare"].dtype() == "int"
        assert titanic_vd_fun["cabin"].dtype() == "varchar(1)"

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


class TestVDCTyping:
    """
    test class for Typing functions test for vColumn class
    """

    def test_astype(self, titanic_vd_fun):
        """
        test function - astype for vColumn
        """
        # Testing vDataFrame[].astype
        # expected exception
        with pytest.raises(ConversionError) as exception_info:
            titanic_vd_fun["sex"].astype("int")
        # checking the error message
        assert exception_info.match(
            'Could not convert "female" from column titanic.sex to an int8'
        )

        titanic_vd_fun["sex"].astype("varchar(10)")
        assert titanic_vd_fun["sex"].dtype() == "varchar(10)"

        titanic_vd_fun["age"].astype("float")
        assert titanic_vd_fun["age"].dtype() == "float"

    def test_astype_str_to_vmap(self):
        """
        test function - astype
        string to vmap
        """
        vdf = TableSample(
            {
                "str_test": [
                    '{"name": "Sam", "information": {"age": 29, "numero": [0, 6, 3]}}'
                ]
            }
        ).to_vdf()
        vdf["str_test"].astype("vmap")
        assert int(vdf["str_test"]["information"]["age"][0]) == 29

    def test_astype_csv_strs(self):
        """
        test function - astype
        csv strings
        """
        vdf = TableSample({"str_test": ["a,b,c,d"]}).to_vdf()
        vdf["str_test"].astype("vmap(val1,val2,val3,val4)")
        assert vdf["str_test"]["val2"][0] == "b"

    def test_astype_str_to_array(self):
        """
        test function - astype
        string to array
        """
        vdf = TableSample({"str_test": ["a,b,c,d"]}).to_vdf()
        vdf["str_test"].astype("array")
        assert vdf["str_test"][1][0] == "b"

    def test_astype_vmap_to_str(self):
        """
        test function - astype
        vmap to string
        """
        vdf = TableSample(
            {
                "str_test": [
                    '{"name": "Sam", "information": {"age": 29, "numero": [0, 6, 3]}}'
                ]
            }
        ).to_vdf()
        vdf["str_test"].astype("vmap")
        vdf["str_test"].astype(str)
        assert (
            vdf["str_test"][0]
            == '{\n\t"information": {\n\t\t"age": "29",\n\t\t"numero": {\n\t\t\t"0": "0",\n\t\t\t"1": "6",\n\t\t\t"2": "3"\n\t\t}\n\t},\n\t"name": "Sam"\n}'
        )
        vdf = TableSample(
            {
                "str_test": [
                    '{"name": "Sam", "information": {"age": 29, "numero": [0, 6, 3]}}'
                ]
            }
        ).to_vdf()
        vdf["str_test"].astype("vmap")
        vdf["str_test"].astype("json")
        assert (
            vdf["str_test"][0]
            == '{\n\t"information": {\n\t\t"age": "29",\n\t\t"numero": {\n\t\t\t"0": "0",\n\t\t\t"1": "6",\n\t\t\t"2": "3"\n\t\t}\n\t},\n\t"name": "Sam"\n}'
        )

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

    def test_dtype(self, amazon_vd):
        """
        test function - dtypes
        """
        assert amazon_vd["state"].dtype() == "varchar(32)"
