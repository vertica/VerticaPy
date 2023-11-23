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

# Standard Python Modules
import pytest

# VerticaPy
from verticapy.core.vdataframe.base import vDataFrame
from verticapy.utilities import drop, TableSample
from verticapy.datasets import load_titanic
from verticapy._config.config import set_option


# Other Modules
import pandas as pd
import numpy as np

set_option("print_info", False)


@pytest.fixture(scope="module")
def titanic_vd():
    titanic = load_titanic()
    yield titanic
    drop(name="public.titanic")


class TestvDFCreate:
    def test_creating_vDF_using_input_relation(self, titanic_vd):
        tvdf = vDataFrame(input_relation="public.titanic")

        assert tvdf["pclass"].count() == 1234

    def test_creating_vDF_using_input_relation_schema(self, titanic_vd):
        tvdf = vDataFrame(input_relation="titanic", schema="public")

        assert tvdf["pclass"].count() == 1234

    def test_creating_vDF_using_input_relation_vDataColumns(self, titanic_vd):
        tvdf = vDataFrame(
            input_relation="public.titanic",
            usecols=["age", "survived"],
        )

        assert tvdf["survived"].count() == 1234

    def test_creating_vDF_using_pandas_dataframe(self, titanic_vd):
        df = vDataFrame(input_relation="public.titanic").to_pandas()
        tvdf = vDataFrame(df)

        assert tvdf["survived"].count() == 1234

    def test_creating_vDF_using_list(self):
        tvdf = vDataFrame(
            input_relation=[[1, "Badr", "Ouali"], [2, "Arash", "Fard"]],
            usecols=["id", "fname", "lname"],
        )

        assert tvdf.shape() == (2, 3)
        assert tvdf["id"].avg() == 1.5

    def test_creating_vDF_using_np_array(self):
        tvdf = vDataFrame(
            input_relation=np.array([[1, "Badr", "Ouali"], [2, "Arash", "Fard"]]),
        )

        assert tvdf.shape() == (2, 3)
        assert tvdf["col0"].avg() == 1.5

    def test_creating_vDF_using_TableSample(self):
        tb = TableSample(
            {"id": [1, 2], "fname": ["Badr", "Arash"], "lname": ["Ouali", "Fard"]}
        )
        tvdf = vDataFrame(
            input_relation=tb,
        )

        assert tvdf.shape() == (2, 3)
        assert tvdf["id"].avg() == 1.5

        tvdf = vDataFrame(input_relation=tb, usecols=["id", "lname"])

        assert tvdf.shape() == (2, 2)
        assert tvdf.get_columns() == ['"id"', '"lname"']

    def test_creating_vDF_using_dict(self):
        tb = {"id": [1, 2], "fname": ["Badr", "Arash"], "lname": ["Ouali", "Fard"]}
        tvdf = vDataFrame(
            input_relation=tb,
        )

        assert tvdf.shape() == (2, 3)
        assert tvdf["id"].avg() == 1.5

        tvdf = vDataFrame(input_relation=tb, usecols=["id", "lname"])

        assert tvdf.shape() == (2, 2)
        assert tvdf.get_columns() == ['"id"', '"lname"']

    def test_creating_vDF_from_sql(self, titanic_vd):
        tvdf = vDataFrame("SELECT * FROM public.titanic")

        assert tvdf["survived"].count() == 1234

        tvdf = vDataFrame("SELECT * FROM public.titanic", usecols=["survived"])

        assert tvdf["survived"].count() == 1234
        assert tvdf.get_columns() == ['"survived"']

    # TODO vDataFrame from a flex table
