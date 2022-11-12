# (c) Copyright [2018-2022] Micro Focus or one of its affiliates.
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

# Pytest
import pytest

# Standard Python Modules
import warnings, os

# VerticaPy
import verticapy
from verticapy import drop, set_option, tablesample
from verticapy.datasets import load_titanic
from verticapy.sql import sql

set_option("print_info", False)


@pytest.fixture(scope="module")
def titanic_vd():
    titanic = load_titanic()
    yield titanic
    drop(name="public.titanic")


class TestSQL:
    def test_sql(self, titanic_vd):

        # SQL line Test -c
        result = sql('  -c "SELECT * FROM titanic;"', "")
        assert result.shape() == (1234, 14)
        assert (
            result._VERTICAPY_VARIABLES_["main_relation"]
            == "(SELECT * FROM titanic) VSQL_MAGIC"
        )

        # SQL line Test --command
        result = sql('  --command "SELECT * FROM titanic;"', "")
        assert result.shape() == (1234, 14)

        # SQL line Test - single quote
        result = sql("  -c 'SELECT * FROM titanic;'", "")
        assert result.shape() == (1234, 14)

        # SQL line Test -nrows -ncols
        result = sql('  -c "SELECT * FROM titanic;"   -ncols 4    -nrows 70', "")
        assert result.shape() == (1234, 14)
        assert result._VERTICAPY_VARIABLES_["max_columns"] == 4
        assert result._VERTICAPY_VARIABLES_["max_rows"] == 70

        # SQL Cell Test -nrows -ncols
        result = sql("  -ncols 4    -nrows 70", "SELECT * FROM titanic;")
        assert result.shape() == (1234, 14)
        assert result._VERTICAPY_VARIABLES_["max_columns"] == 4
        assert result._VERTICAPY_VARIABLES_["max_rows"] == 70

        # SQL cell Test
        result = sql(
            "",
            """DROP MODEL IF EXISTS model_test;
               SELECT LINEAR_REG('model_test', 'public.titanic', 'survived', 'age, fare'); 
               SELECT PREDICT_LINEAR_REG(3.0, 4.0 
                      USING PARAMETERS model_name='model_test', 
                                       match_by_pos=True) AS predict;""",
        )
        assert result["predict"][0] == pytest.approx(0.395335892040411)
        result = sql("", "DROP MODEL IF EXISTS model_test; SELECT 1 AS col;;")
        assert result["col"][0] == 1

        # Test: Reading SQL file -f
        result = sql(
            " -f   {}/tests/sql/queries.sql".format(
                os.path.dirname(verticapy.__file__)
            ),
            "",
        )
        assert result["predict"][0] == pytest.approx(0.395335892040411)

        # Test: Reading SQL file --file
        result = sql(
            "   --file   {}/tests/sql/queries.sql".format(
                os.path.dirname(verticapy.__file__)
            ),
            "",
        )
        assert result["predict"][0] == pytest.approx(0.395335892040411)

        # Export to JSON -o
        result = sql(
            "-o verticapy_test_sql.json",
            "SELECT age, fare FROM titanic ORDER BY age DESC, fare DESC LIMIT 2;",
        )
        try:
            file = open("verticapy_test_sql.json", "r")
            result = file.read()
            print(result)
            assert result == (
                '[\n{"age": 80.000, "fare": 30.00000},'
                '\n{"age": 76.000, "fare": 78.85000}\n]'
            )
        except:
            os.remove("verticapy_test_sql.json")
            file.close()
            raise
        os.remove("verticapy_test_sql.json")
        file.close()

        # Export to JSON --output
        result = sql(
            "  --output   verticapy_test_sql.json",
            "SELECT age, fare FROM titanic ORDER BY age DESC, fare DESC LIMIT 2;",
        )
        try:
            file = open("verticapy_test_sql.json", "r")
            result = file.read()
            print(result)
            assert result == (
                '[\n{"age": 80.000, "fare": 30.00000},'
                '\n{"age": 76.000, "fare": 78.85000}\n]'
            )
        except:
            os.remove("verticapy_test_sql.json")
            file.close()
            raise
        os.remove("verticapy_test_sql.json")
        file.close()

        # Export to CSV -o
        result = sql(
            " -o           verticapy_test_sql.csv",
            "SELECT age, fare FROM titanic ORDER BY age DESC, fare DESC LIMIT 2;",
        )
        try:
            file = open("verticapy_test_sql.csv", "r")
            result = file.read()
            assert result == '"age","fare"\n80.000,30.00000\n76.000,78.85000'
        except:
            os.remove("verticapy_test_sql.csv")
            file.close()
            raise
        os.remove("verticapy_test_sql.csv")
        file.close()

        # Export to CSV --output
        result = sql(
            "--output verticapy_test_sql.csv",
            "SELECT age, fare FROM titanic ORDER BY age DESC, fare DESC LIMIT 2;",
        )
        try:
            file = open("verticapy_test_sql.csv", "r")
            result = file.read()
            assert result == '"age","fare"\n80.000,30.00000\n76.000,78.85000'
        except:
            os.remove("verticapy_test_sql.csv")
            file.close()
            raise
        os.remove("verticapy_test_sql.csv")
        file.close()

        # Test on the variables - TEST WORKS ON JUPYTER BUT NOT IN FILES
        # result = sql("", "SELECT * FROM :titanic_vd;")
        # assert result.shape() == (1234, 14)
        # table = "titanic"
        # result = sql("", "SELECT * FROM :titanic;")
        # assert result.shape() == (1234, 14)
        # tb = tablesample({"x": [4, 5, 6], "y": [1, 2, 3]})
        # result = sql("", "SELECT AVG(x) FROM :tb;")
        # assert result == 5

        # TODO: add a test to see if the order by works when
        # dealing with multiple nodes
