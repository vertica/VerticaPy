# (c) Copyright [2018-2020] Micro Focus or one of its affiliates.
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

import pytest, warnings
from verticapy import vDataFrame, drop_table

from verticapy import set_option

set_option("print_info", False)


@pytest.fixture(scope="module")
def titanic_vd(base):
    from verticapy.learn.datasets import load_titanic

    titanic = load_titanic(cursor=base.cursor)
    yield titanic
    with warnings.catch_warnings(record=True) as w:
        drop_table(name="public.titanic", cursor=base.cursor)


class TestvDFCreate:
    def test_creating_vDF_using_input_relation(self, base, titanic_vd):
        tvdf = vDataFrame(input_relation="public.titanic", cursor=base.cursor)

        assert tvdf["pclass"].count() == 1234

    def test_creating_vDF_using_input_relation_schema(self, base, titanic_vd):
        tvdf = vDataFrame(input_relation="titanic", schema="public", cursor=base.cursor)

        assert tvdf["pclass"].count() == 1234

    def test_creating_vDF_using_input_relation_vcolumns(self, base, titanic_vd):
        tvdf = vDataFrame(
            input_relation="public.titanic",
            usecols=["age", "survived"],
            cursor=base.cursor,
        )

        assert tvdf["survived"].count() == 1234

    @pytest.mark.skip(reason="test not implemented")
    def test_creating_vDF_using_input_relation_dsn(self):
        pass
