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

# Standard Python Modules
import pytest

# VerticaPy
from verticapy import vDataFrame, drop, set_option
from verticapy.datasets import load_titanic

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

    def test_creating_vDF_using_input_relation_vcolumns(self, titanic_vd):
        tvdf = vDataFrame(input_relation="public.titanic", usecols=["age", "survived"],)

        assert tvdf["survived"].count() == 1234
