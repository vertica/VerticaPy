# (c) Copyright [2018-2021] Micro Focus or one of its affiliates.
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
from verticapy.learn.preprocessing import Balance
from verticapy import drop_table, set_option
import matplotlib.pyplot as plt

set_option("print_info", False)


@pytest.fixture(scope="module")
def titanic_vd(base):
    from verticapy.datasets import load_titanic

    titanic = load_titanic(cursor=base.cursor)
    yield titanic
    with warnings.catch_warnings(record=True) as w:
        drop_table(name="public.titanic", cursor=base.cursor)


class TestBalance:
    def test_hybrid_method(self, base, titanic_vd):
        base.cursor.execute("DROP VIEW IF EXISTS public.hybrid_balanced")

        bvd = Balance(
            name="public.hybrid_balanced",
            input_relation="public.titanic",
            y="survived",
            cursor=base.cursor,
        )

        assert bvd["survived"].skew() == pytest.approx(0, abs=0.2)

        base.cursor.execute(
            "SELECT table_name FROM views WHERE table_name = 'hybrid_balanced'"
        )
        assert base.cursor.fetchone()[0] == "hybrid_balanced"

        base.cursor.execute("DROP VIEW public.hybrid_balanced")

    def test_under_method(self, base, titanic_vd):
        base.cursor.execute("DROP VIEW IF EXISTS public.under_balanced")

        bvd = Balance(
            name="public.under_balanced",
            input_relation="public.titanic",
            y="survived",
            method="under",
            ratio=1.0,
            cursor=base.cursor,
        )

        assert bvd["survived"].skew() == pytest.approx(0, abs=0.2)

        base.cursor.execute(
            "SELECT table_name FROM views WHERE table_name = 'under_balanced'"
        )
        assert base.cursor.fetchone()[0] == "under_balanced"

        base.cursor.execute("DROP VIEW public.under_balanced")

    def test_over_method(self, base, titanic_vd):
        base.cursor.execute("DROP VIEW IF EXISTS public.over_balanced")

        bvd = Balance(
            name="public.over_balanced",
            input_relation="public.titanic",
            y="survived",
            method="under",
            ratio=0.75,
            cursor=base.cursor,
        )

        assert bvd["survived"].skew() == pytest.approx(0.25, abs=0.2)

        base.cursor.execute(
            "SELECT table_name FROM views WHERE table_name = 'over_balanced'"
        )
        assert base.cursor.fetchone()[0] == "over_balanced"

        base.cursor.execute("DROP VIEW public.over_balanced")
