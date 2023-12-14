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
from verticapy.core.vdataframe.base import vDataFrame
from verticapy.utilities import drop, TableSample
from verticapy.errors import ConversionError
from verticapy.datasets import load_titanic, load_iris, load_market
from verticapy._config.config import set_option


set_option("print_info", False)


@pytest.fixture(scope="module")
def titanic_vd():
    titanic = load_titanic()
    yield titanic
    drop(
        name="public.titanic",
    )


@pytest.fixture(scope="module")
def iris_vd():
    iris = load_iris()
    yield iris
    drop(
        name="public.iris",
    )


@pytest.fixture(scope="module")
def market_vd():
    market = load_market()
    yield market
    drop(
        name="public.market",
    )


class TestvDFPreprocessing:
    def test_vDF_get_dummies(self, iris_vd):
        ### Testing vDataFrame.get_dummies
        # use_numbers_as_suffix = False
        iris_copy = iris_vd.copy()
        iris_copy.get_dummies()
        assert iris_copy.get_columns() == [
            '"SepalLengthCm"',
            '"SepalWidthCm"',
            '"PetalLengthCm"',
            '"PetalWidthCm"',
            '"Species"',
            '"Species_Iris-setosa"',
            '"Species_Iris-versicolor"',
        ]

        # use_numbers_as_suffix = True
        iris_copy = iris_vd.copy()
        iris_copy.get_dummies(use_numbers_as_suffix=True)
        assert iris_copy.get_columns() == [
            '"SepalLengthCm"',
            '"SepalWidthCm"',
            '"PetalLengthCm"',
            '"PetalWidthCm"',
            '"Species"',
            '"Species_0"',
            '"Species_1"',
        ]

        ### Testing vDataFrame.get_dummies
        # use_numbers_as_suffix = False
        iris_copy = iris_vd.copy()
        iris_copy["Species"].get_dummies(prefix="D", prefix_sep="--")
        assert iris_copy.get_columns() == [
            '"SepalLengthCm"',
            '"SepalWidthCm"',
            '"PetalLengthCm"',
            '"PetalWidthCm"',
            '"Species"',
            '"D--Iris-setosa"',
            '"D--Iris-versicolor"',
        ]

        # use_numbers_as_suffix = True
        iris_copy = iris_vd.copy()
        iris_copy["Species"].get_dummies(use_numbers_as_suffix=True)
        assert iris_copy.get_columns() == [
            '"SepalLengthCm"',
            '"SepalWidthCm"',
            '"PetalLengthCm"',
            '"PetalWidthCm"',
            '"Species"',
            '"Species_0"',
            '"Species_1"',
        ]

    def test_vDF_lable_encode(self, titanic_vd):
        titanic_copy = titanic_vd.copy()
        titanic_copy["embarked"].label_encode()

        assert titanic_copy["embarked"].distinct() == [0, 1, 2, 3]

    def test_vDF_dropna(self, titanic_vd):
        # Testing vDataFrame.dropna
        titanic_copy = titanic_vd.copy()
        titanic_copy.dropna(columns=["fare", "embarked", "age"])
        result = titanic_copy.count_percent(columns=["fare", "embarked", "age"])

        assert result["count"][0] == 994
        assert result["count"][1] == 994
        assert result["count"][2] == 994

        # Testing vDataFrame[].dropna
        titanic_copy = titanic_vd.copy()
        titanic_copy["age"].dropna()
        assert titanic_copy.count_percent(["age"])["count"][0] == 997

    def test_vDF_astype(self, titanic_vd):
        ### Testing vDataFrame.astype
        titanic_copy = titanic_vd.copy()
        titanic_copy.astype({"fare": "int", "cabin": "varchar(1)"})

        assert titanic_copy["fare"].dtype() == "int"
        assert titanic_copy["cabin"].dtype() == "varchar(1)"

        ### Testing vDataFrame[].astype
        # expected exception
        with pytest.raises(ConversionError) as exception_info:
            titanic_copy["sex"].astype("int")
        # checking the error message
        assert exception_info.match(
            'Could not convert "female" from column titanic.sex to an int8'
        )

        titanic_copy["sex"].astype("varchar(10)")
        assert titanic_copy["sex"].dtype() == "varchar(10)"

        titanic_copy["age"].astype("float")
        assert titanic_copy["age"].dtype() == "float"

        # STR to VMAP
        # tests on JSONs vdf
        vdf = TableSample(
            {
                "str_test": [
                    '{"name": "Badr", "information": {"age": 29, "numero": [0, 6, 3]}}'
                ]
            }
        ).to_vdf()
        vdf["str_test"].astype("vmap")
        assert int(vdf["str_test"]["information"]["age"][0]) == 29
        # tests on CSVs strings
        vdf = TableSample({"str_test": ["a,b,c,d"]}).to_vdf()
        vdf["str_test"].astype("vmap(val1,val2,val3,val4)")
        assert vdf["str_test"]["val2"][0] == "b"

        # STR to ARRAY
        vdf = TableSample({"str_test": ["a,b,c,d"]}).to_vdf()
        vdf["str_test"].astype("array")
        assert vdf["str_test"][1][0] == "b"

        # VMAP to STR
        vdf = TableSample(
            {
                "str_test": [
                    '{"name": "Badr", "information": {"age": 29, "numero": [0, 6, 3]}}'
                ]
            }
        ).to_vdf()
        vdf["str_test"].astype("vmap")
        vdf["str_test"].astype(str)
        assert (
            vdf["str_test"][0]
            == '{\n\t"information": {\n\t\t"age": "29",\n\t\t"numero": {\n\t\t\t"0": "0",\n\t\t\t"1": "6",\n\t\t\t"2": "3"\n\t\t}\n\t},\n\t"name": "Badr"\n}'
        )
        vdf = TableSample(
            {
                "str_test": [
                    '{"name": "Badr", "information": {"age": 29, "numero": [0, 6, 3]}}'
                ]
            }
        ).to_vdf()
        vdf["str_test"].astype("vmap")
        vdf["str_test"].astype("json")
        assert (
            vdf["str_test"][0]
            == '{\n\t"information": {\n\t\t"age": "29",\n\t\t"numero": {\n\t\t\t"0": "0",\n\t\t\t"1": "6",\n\t\t\t"2": "3"\n\t\t}\n\t},\n\t"name": "Badr"\n}'
        )

    def test_vDF_rename(self, titanic_vd):
        titanic_copy = titanic_vd.copy()
        titanic_copy["sex"].rename("gender")
        assert '"gender"' in titanic_copy.get_columns()
        assert '"sex"' not in titanic_copy.get_columns()
