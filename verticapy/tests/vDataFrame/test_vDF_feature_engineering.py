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

# Standard Python Modules
import datetime

# VerticaPy
from verticapy import drop, errors, set_option
from verticapy.datasets import (
    load_amazon,
    load_iris,
    load_smart_meters,
    load_titanic,
)

set_option("print_info", False)


@pytest.fixture(scope="module")
def amazon_vd():
    amazon = load_amazon()
    yield amazon
    drop(name="public.amazon")


@pytest.fixture(scope="module")
def iris_vd():
    iris = load_iris()
    yield iris
    drop(name="public.iris")


@pytest.fixture(scope="module")
def smart_meters_vd():
    smart_meters = load_smart_meters()
    yield smart_meters
    drop(name="public.smart_meters")


@pytest.fixture(scope="module")
def titanic_vd():
    titanic = load_titanic()
    yield titanic
    drop(name="public.titanic")


class TestvDFFeatureEngineering:
    def test_vDF_sessionize(self, smart_meters_vd):
        smart_meters_copy = smart_meters_vd.copy()

        # expected exception
        with pytest.raises(errors.QueryError) as exception_info:
            smart_meters_copy.sessionize(
                ts="time", by=["id"], session_threshold="1 time", name="slot"
            )
        # checking the error message
        assert exception_info.match("seems to be incorrect")

        smart_meters_copy.sessionize(
            ts="time", by=["id"], session_threshold="1 hour", name="slot"
        )
        smart_meters_copy.sort({"id": "asc", "time": "asc"})

        assert smart_meters_copy.shape() == (11844, 4)
        assert smart_meters_copy["time"][2] == datetime.datetime(2014, 1, 2, 10, 45)
        assert smart_meters_copy["val"][2] == 0.321
        assert smart_meters_copy["id"][2] == 0
        assert smart_meters_copy["slot"][2] == 2

    def test_vDF_case_when(self, titanic_vd):
        titanic_copy = titanic_vd.copy()
        titanic_copy.case_when(
            "age_category",
            titanic_copy["age"] < 12,
            "children",
            titanic_copy["age"] < 18,
            "teenagers",
            titanic_copy["age"] > 60,
            "seniors",
            titanic_copy["age"] < 25,
            "young adults",
            "adults",
        )

        assert titanic_copy["age_category"].distinct() == [
            "adults",
            "children",
            "seniors",
            "teenagers",
            "young adults",
        ]

    def test_vDF_eval(self, titanic_vd):
        # new feature creation
        titanic_copy = titanic_vd.copy()
        titanic_copy.eval(name="family_size", expr="parch + sibsp + 1")

        assert titanic_copy["family_size"].max() == 11

        # Customized SQL code evaluation
        titanic_copy = titanic_vd.copy()
        titanic_copy.eval(
            name="has_life_boat", expr="CASE WHEN boat IS NULL THEN 0 ELSE 1 END"
        )

        assert titanic_copy["boat"].count() == titanic_copy["has_life_boat"].sum()

    def test_vDF_add_copy(self, titanic_vd):
        titanic_copy = titanic_vd.copy()
        titanic_copy["age"].add_copy(name="copy_age")

        assert titanic_copy["copy_age"].mean() == titanic_copy["age"].mean()

    def test_vDF_copy(self, titanic_vd):
        titanic_copy = titanic_vd.copy()

        assert titanic_copy.get_columns() == titanic_vd.get_columns()
