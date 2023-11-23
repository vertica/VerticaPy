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
from verticapy import drop, set_option
from verticapy.connection import current_cursor
from verticapy.datasets import load_titanic
from verticapy.learn.preprocessing import balance

set_option("print_info", False)


@pytest.fixture(scope="module")
def titanic_vd():
    titanic = load_titanic()
    yield titanic
    drop(
        name="public.titanic",
    )


class TestBalance:
    def test_hybrid_method(self, titanic_vd):
        current_cursor().execute("DROP VIEW IF EXISTS public.hybrid_balanced")

        bvd = balance(
            name="public.hybrid_balanced",
            input_relation="public.titanic",
            y="survived",
        )

        assert bvd["survived"].skew() == pytest.approx(0, abs=0.2)

        current_cursor().execute(
            "SELECT table_name FROM views WHERE table_name = 'hybrid_balanced'"
        )
        assert current_cursor().fetchone()[0] == "hybrid_balanced"

        current_cursor().execute("DROP VIEW public.hybrid_balanced")

    def test_under_method(self, titanic_vd):
        current_cursor().execute("DROP VIEW IF EXISTS public.under_balanced")

        bvd = balance(
            name="public.under_balanced",
            input_relation="public.titanic",
            y="survived",
            method="under",
            ratio=1.0,
        )

        assert bvd["survived"].skew() == pytest.approx(0, abs=0.2)

        current_cursor().execute(
            "SELECT table_name FROM views WHERE table_name = 'under_balanced'"
        )
        assert current_cursor().fetchone()[0] == "under_balanced"

        current_cursor().execute("DROP VIEW public.under_balanced")

    def test_over_method(self, titanic_vd):
        current_cursor().execute("DROP VIEW IF EXISTS public.over_balanced")

        bvd = balance(
            name="public.over_balanced",
            input_relation="public.titanic",
            y="survived",
            method="under",
            ratio=0.75,
        )

        assert bvd["survived"].skew() == pytest.approx(0.25, abs=0.2)

        current_cursor().execute(
            "SELECT table_name FROM views WHERE table_name = 'over_balanced'"
        )
        assert current_cursor().fetchone()[0] == "over_balanced"

        current_cursor().execute("DROP VIEW public.over_balanced")
